"""PEtab v2 problem importer: a BNGL-native PEtab v2 problem -> a new-era PyBNF job
(issue #407; the importer read path, ADR-0025 reversed / ADR-0032).

The inverse of :func:`pybnf.petab.export.export_job`. Given a ``problem.yaml`` + its TSV
tables + a BNGL model, :func:`import_job` writes a runnable new-era (edition 2) ``.conf``
plus the ``.exp`` data files and a verbatim copy of the model (new-era binds free
parameters by id, ADR-0034, so the model needs no re-instrumentation) -- the form the
exporter reads. It closes the "two-adapter proof" at the read level for BNGL-native
problems: the reverse asset mappers (parameters/observables/measurements/conditions) run
backwards onto the shared neutral rows, and this module is the *disposable orchestrator*
that ties them together (problem.yaml reader + ``.conf``/``.exp`` writers).

**PEtab is a problem spec; PyBNF is a job spec (the run-recipe gap).** A PEtab problem
fixes the objective landscape (model, data, conditions, parameters + priors, the noise
model) but deliberately says nothing about *how to search it* -- no optimizer/sampler, no
algorithm settings, no simulation method, no seed -- because PEtab is a cross-tool
exchange format and the *method* belongs to the tool. So ``import = PEtab problem +
a supplied run-recipe``. The *problem* half is recovered exactly (and round-trips
byte-for-byte through a re-export); the *recipe* half -- ``job_type`` + that fit's
algorithm settings (SEARCH), the per-experiment ``method:`` (SIMULATION), and
``output_dir`` / ``verbosity`` / required keys (PLUMBING) -- is **supplied, not
recovered**, and is excluded from the round-trip identity. The recipe is not a new
language: each group is an existing PyBNF/ADR-0028 surface, and its defaults come from the
registry/schema, never a parallel table here.

Concretely, the recipe is supplied through :func:`import_job`'s parameters:

* ``job_type`` -- the SEARCH method token (``'de'`` by default). ``'all'`` enumerates the
  fit-type registry (every ``optimizer`` + ``sampler``; the ``check`` checker excluded)
  and writes one runnable ``imported_<jt>.conf`` per method, the existing
  benchmark-harness pattern (ADR-0012): the importer covers the whole toolbox and stays
  correct as it grows. Sampler-vs-optimizer is a genuine scientific choice (a sampler
  treats the priors as Bayesian priors), so the importer must not pick for the user.
* ``method`` / ``method_overrides`` -- the SIMULATION method, emitted on **every**
  ``experiment:`` line (``'ode'`` by default; ``method_overrides={exp: method}`` sets
  per-experiment values). It is per-experiment, never a single global knob: a job can have
  multiple models/experiments each simulated differently, and the method is not derivable
  from data (deterministic and stochastic models yield identically-shaped traces).
  Round-trip is lossy here -- export drops ``method`` (no PEtab home), import defaults it
  to ``ode`` -- so a stochastic model does not survive a PEtab hop.
* ``settings`` -- overrides for the required algorithm/run settings (``population_size`` /
  ``max_iterations`` / ``verbosity``); the per-method schema defaults the rest.

**Dependency-free + simulator-free.** Like the other read-path chunks, the import path
uses only stdlib + ``pybnf.data.Data`` + the asset mappers, so it runs in the
bngsim-less CI tier. ``problem.yaml`` is hand-parsed (the exporter emits a fixed, simple
shape); a non-``bngl`` model language raises (the SBML adapter is separate). The
``petab`` library stays a test-only oracle.

**Scope (read path, BNGL-native).** An expression ``observableFormula`` is now in scope:
it is translated to a synthesized BNGL function (ADR-0035, the optional ``pybnf[petab]``
extra); the bare-name common case still needs no translator and stays dependency-free.
Out of scope, each mirroring an export-side boundary: fitting the imported job (gated on
the ADR-0028 config loader, #423); SBML model import; the ``noiseFormula`` / condition
sympy layer and per-measurement ``observableParameters``/``noiseParameters`` placeholders
(bare names / numbers only); the five PEtab prior families PyBNF lacks; one-sided
truncation; multi-model; parameter-scan / dose-response; replicate reconstruction.
"""

import re
from pathlib import Path

import numpy as np

from ..noise import ConstantSigma, FreeParameterSigma
from ..printing import PybnfError
from .conditions import (
    REF_MARKER,
    condition_name_from_id,
    conditions_from_rows,
    read_condition_table,
    read_experiment_table,
)
from .formula import petab_math_to_bngl_body
from .measurements import data_from_measurement_rows, read_measurement_table
from .observables import noise_model_from_row, read_observable_table
from .parameters import free_parameter_from_row, read_parameter_table
from ._bngl import parse_model
from ._tsv import num

# A bare model-entity name (an observableFormula in the common case). Anything with
# operators / calls / whitespace is an expression for the deferred sympy layer.
_IDENTIFIER = re.compile(r'[A-Za-z_]\w*\Z')

# The required user settings the loader has no schema default for (config.py
# ``_req_user_params`` + the run-level ``verbosity``); supplied with thin defaults so an
# imported conf is runnable. ``settings`` overrides any of them. The per-method schema
# defaults everything else (ADR-0006/0012), so this stays method-agnostic.
_DEFAULT_SETTINGS = {'population_size': 50, 'max_iterations': 100, 'verbosity': 0}

# The fit-type families a job_type='all' emit covers (the checker is not a fit).
_EMIT_ALL_FAMILIES = ('optimizer', 'sampler')


# ---------------------------------------------------------------------------
# The importer driver
# ---------------------------------------------------------------------------

def import_job(problem_yaml_path, out_dir, job_type='de', method='ode',
               method_overrides=None, settings=None):
    """Import the BNGL-native PEtab v2 problem at ``problem_yaml_path`` into ``out_dir``.

    Reads the problem's tables + model, reconstructs the experiments' data, and writes a
    new-era PyBNF job: the ``.exp`` data files, a verbatim copy of the BNGL model (new-era
    binds free parameters by id, so the model needs no re-instrumentation -- ADR-0034), and
    one or more ``.conf`` files. The *problem* (parameters/priors, observables/noise,
    measurements, conditions/experiments) is recovered exactly; the *run-recipe*
    (``job_type``, ``method``, ``settings``) is supplied by the caller (see the module
    docstring). Returns the ``out_dir`` path.

    ``job_type`` is the SEARCH method token, or ``'all'`` to emit one
    ``imported_<jt>.conf`` per registered optimizer + sampler. ``method`` (default
    ``'ode'``) is the per-experiment SIMULATION method; ``method_overrides`` (a
    ``{experiment_name: method}`` map) sets per-experiment values. ``settings`` overrides
    the required algorithm/run settings.

    An **expression** ``observableFormula`` (e.g. a quotient of sums) is no longer refused:
    it is translated to a BNGL function synthesized into the model (ADR-0035, the
    ``pybnf[petab]`` extra). Raises ``NotImplementedError`` at the remaining PEtab/PyBNF
    boundaries (a non-``bngl`` model language; the five unsupported prior families; a
    log-normal/log-laplace or expression noise model; a ``noiseFormula``/condition
    expression; a per-measurement ``observableParameters``/``noiseParameters`` placeholder
    in a formula; replicate rows) and ``PybnfError`` for a malformed problem (including an
    ``observableFormula`` symbol that is not a model entity).
    """
    problem_yaml_path = Path(problem_yaml_path)
    base = problem_yaml_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = read_problem_yaml(problem_yaml_path)
    _require_bngl_model(problem, problem_yaml_path)

    parameter_rows = read_parameter_table(base / problem['parameter_files'][0])
    observable_rows = read_observable_table(base / problem['observable_files'][0])
    measurement_rows = read_measurement_table(base / problem['measurement_files'][0])
    condition_rows = (read_condition_table(base / problem['condition_files'][0])
                      if problem['condition_files'] else [])
    experiment_rows = (read_experiment_table(base / problem['experiment_files'][0])
                       if problem['experiment_files'] else [])

    # Parameters -> conf free-parameter lines (bare ids; new-era binds by id, ADR-0034)
    # + the surrogate set M of fit-and-perturbed model parameters.
    free_param_lines, surrogate_params = _free_parameters(parameter_rows)

    # The model is read now (not just at write time): an expression observableFormula
    # validates its free symbols against the model's entity namespace and synthesizes a
    # function into the model text (ADR-0035), so its entities are needed before the
    # observable mapping. The bare-name path ignores the entities and carries the model
    # byte-verbatim (new-era binds free params by id, ADR-0034 -- no re-instrumentation).
    model_filename = problem['model_file']
    model_text = (base / model_filename).read_text(encoding='utf-8', errors='replace')
    entities = parse_model(model_text)

    # Observables -> the observableId -> model-column map (the data pivot's column order)
    # plus any functions synthesized from expression observableFormulas (ADR-0035).
    observable_id_to_column, synthesized_functions = _observable_id_to_column(
        observable_rows, entities)

    # Measurements -> one wide Data per experiment, then assemble the experiment list.
    datas = data_from_measurement_rows(measurement_rows, observable_id_to_column)
    objective_directive = _objective_directive(
        observable_rows, _column_mean_resolver(datas, observable_id_to_column))
    conditions = conditions_from_rows(condition_rows, surrogate_params)
    experiments = _experiments(datas, experiment_rows, out_dir)

    # The model copy: byte-verbatim on the bare-name path; on the expression path it gains
    # exactly one targeted edit -- a `begin functions` block carrying the synthesized
    # measurement-model functions (the same *kind* of edit ADR-0032 once made for __FREE,
    # the only place the importer no longer carries the model verbatim, ADR-0034/0035).
    if synthesized_functions:
        model_text = _inject_functions(model_text, synthesized_functions)
    (out_dir / model_filename).write_text(model_text)

    merged_settings = {**_DEFAULT_SETTINGS, **(settings or {})}
    job_types = _emit_all_job_types() if job_type == 'all' else [job_type]
    for jt in job_types:
        conf_name = f'imported_{jt}.conf' if len(job_types) > 1 else 'imported.conf'
        _write_conf(
            out_dir / conf_name, model_filename=model_filename, job_type=jt,
            objective_directive=objective_directive, free_param_lines=free_param_lines,
            conditions=conditions, experiments=experiments, method=method,
            method_overrides=method_overrides or {}, settings=merged_settings,
            multi=len(job_types) > 1)
    return out_dir


# ---------------------------------------------------------------------------
# Parameters: rows -> conf free-parameter lines + the surrogate set
# ---------------------------------------------------------------------------

def _free_parameters(parameter_rows):
    """Map estimated parameter rows to conf ``*_var`` lines + the surrogate set.

    Returns ``(free_param_lines, surrogate_params)``: ``free_param_lines`` are the conf
    declarations (**bare ids**, in table order -- new-era binds a free parameter to its
    model parameter by id, ADR-0034, so the declaration *is* ``<id>``, not ``<id>__FREE``);
    ``surrogate_params`` is the set ``M`` of fit-and-perturbed model parameters (a
    ``<p>__REF`` parameterId recovered to ``p`` by :func:`_model_param`).
    ``free_parameter_from_row`` surfaces the prior boundaries (5 families, one-sided
    truncation) as ``NotImplementedError``.
    """
    free_param_lines = []
    surrogate_params = set()
    for row in parameter_rows:
        if not row.estimate:
            continue  # a fixed model constant, not a free parameter (stays in the model)
        model_param, is_surrogate = _model_param(row.parameter_id)
        if is_surrogate:
            surrogate_params.add(model_param)
        fp = free_parameter_from_row(row)
        free_param_lines.append(
            f'{fp.type} = {model_param} {num(fp.p1)} {num(fp.p2)}')
    if not free_param_lines:
        raise PybnfError(
            "The PEtab parameters table declares no estimated (estimate=true) parameters, "
            "so there is nothing to fit.")
    return free_param_lines, surrogate_params


def _model_param(parameter_id):
    """``(model_param, is_surrogate)``: a ``<p>__REF`` surrogate id -> ``(p, True)``;
    a plain parameterId -> ``(parameterId, False)``."""
    if parameter_id.endswith(REF_MARKER):
        return parameter_id[:-len(REF_MARKER)], True
    return parameter_id, False


# ---------------------------------------------------------------------------
# Observables: rows -> column map + objective token
# ---------------------------------------------------------------------------

def _observable_id_to_column(observable_rows, entities):
    """Map each ``observableId`` to the model column it measures, synthesizing a function
    for any expression ``observableFormula`` (ADR-0035). Iteration order = table order,
    which fixes the wide-data column order on the measurement pivot.

    Returns ``(mapping, synthesized_functions)``:

    * A **bare model-entity name** ``observableFormula`` (the common case, ADR-0025) maps
      its ``observableId`` to that name -- PyBNF matches the ``.exp`` column to the model
      observable/function by name and evaluates it *in the model*, so no translator runs
      and the path stays dependency-free.
    * An **expression** ``observableFormula`` is translated to a BNGL function body
      (:func:`~pybnf.petab.formula.petab_math_to_bngl_body`, the optional ``pybnf[petab]``
      extra), recorded as a synthesized function named after the ``observableId``, and the
      column is mapped to that name. The function is injected into the model text by
      :func:`_inject_functions`, and PyBNF matches the ``.exp`` column to it by name.

    The synthesized function name (the ``observableId``) must not shadow an existing model
    entity (``PybnfError``); an unknown free symbol or a per-measurement placeholder in the
    expression raises in the translator (``PybnfError`` / ``NotImplementedError``).
    """
    taken = (set(entities.parameters) | set(entities.observable_names)
             | set(entities.function_names) | set(entities.molecule_type_names)
             | set(entities.compartment_names))
    mapping = {}
    synthesized = []
    for row in observable_rows:
        formula = (row.observable_formula or '').strip()
        if _IDENTIFIER.match(formula):
            mapping[row.observable_id] = formula          # bare-name path (no translator)
            continue
        body = petab_math_to_bngl_body(formula, entities)
        func_name = row.observable_id
        if func_name in taken:
            raise PybnfError(
                f"Cannot synthesize a function '{func_name}()' for the expression "
                f"observableFormula of observable '{row.observable_id}': the name already "
                f"names a model entity (parameter / observable / function / molecule type / "
                f"compartment). Rename the observableId so the synthesized measurement "
                f"function does not shadow a model entity (ADR-0035).")
        taken.add(func_name)
        synthesized.append((func_name, body))
        mapping[row.observable_id] = func_name
    if not mapping:
        raise PybnfError("The PEtab observables table declares no observables.")
    return mapping, synthesized


# The BNGL ``begin/end functions`` anchors + where a fresh block may be inserted (after the
# observables it references; before reactions/actions / the model close). All multiline,
# case-insensitive (BNGL keywords are case-insensitive).
_END_FUNCTIONS = re.compile(r'^[ \t]*end\s+functions\b.*$', re.I | re.M)
_NEW_BLOCK_ANCHOR = re.compile(
    r'^([ \t]*)begin\s+reaction\s+rules\b|^([ \t]*)end\s+model\b|^([ \t]*)begin\s+actions\b',
    re.I | re.M)


def _inject_functions(model_text, functions):
    """Return ``model_text`` with the synthesized ``functions`` (``[(name, body), ...]``)
    added to a ``begin functions`` block (ADR-0035).

    Merges into an existing ``begin functions ... end functions`` block when present
    (inserting before its ``end functions``), else creates a fresh block placed where BNGL
    accepts an output-only global function: before ``begin reaction rules`` if present,
    else before ``end model`` / ``begin actions``, else at end of file. Indentation follows
    the anchor so the edit reads like the surrounding model.
    """
    func_lines = list(functions)
    m_end = _END_FUNCTIONS.search(model_text)
    if m_end:
        lead = re.match(r'[ \t]*', model_text[m_end.start():]).group()
        block = ''.join(f'{lead}  {n}() = {b}\n' for n, b in func_lines)
        return model_text[:m_end.start()] + block + model_text[m_end.start():]
    anchor = _NEW_BLOCK_ANCHOR.search(model_text)
    lead = next((g for g in anchor.groups() if g is not None), '') if anchor else ''
    inner = lead + '  '
    block = (f'{lead}begin functions\n'
             + ''.join(f'{inner}{n}() = {b}\n' for n, b in func_lines)
             + f'{lead}end functions\n')
    if anchor:
        return model_text[:anchor.start()] + block + model_text[anchor.start():]
    return model_text.rstrip('\n') + '\n' + block


def _column_mean_resolver(datas, observable_id_to_column):
    """A ``observableId -> column mean across all experiments`` closure (for distinguishing
    ``sos`` from ``ave_norm_sos``; mirrors the export's column-mean sigma over all data)."""
    def column_mean_of(observable_id):
        col = observable_id_to_column[observable_id]
        values = [data[col] for data in datas.values() if col in data.cols]
        return float(np.average(np.concatenate(values)))
    return column_mean_of


# PEtab noiseDistribution -> (PyBNF noise_model family token, its scale-parameter field).
# The reverse of export.py's _FAMILY_TOKEN_TO_PETAB_DISTRIBUTION for the two families a
# PyBNF objective can carry: the noise_model line names the family and its single noise
# parameter (gaussian/sigma, laplace/scale; ADR-0031, objective._NOISE_PARAM_NAMES). The
# four-token path is preferred where it applies (a tidier 'objective =' line that
# round-trips); this is the fallback for the cases no token names.
_PETAB_DISTRIBUTION_TO_NOISE_MODEL = {
    'normal':  ('gaussian', 'sigma'),
    'laplace': ('laplace',  'scale'),
}


def _objective_directive(observable_rows, column_mean_of):
    """Recover the conf's objective directive (one full line) from the observables' noise.

    The inverse of the objective-family / whole-fit ``noise_model`` export. Returns either:

    * ``objective = <token>`` -- one of the four sugar tokens (``chi_sq`` / ``sos`` /
      ``sod`` / ``ave_norm_sos``), the tidy common case that round-trips byte-for-byte; or
    * ``noise_model = <family>, <param> = <verb> <arg>`` (the ADR-0031 surface) -- the
      broader cases no sugar token names: a uniform **non-unit fixed** sigma
      (``fix_at C``, the symmetric inverse of the exporter's whole-fit ``noise_model``
      line, so it round-trips byte-for-byte) and a single shared **free-parameter** sigma
      (``fit <id>``; import-only, since the exporter raises on a ``fit`` sigma -- it is
      external-problem territory).

    A single PyBNF objective is one family + one sigma source across all observables, so a
    mix of families/sources -- or a *per-observable* free or fixed sigma (e.g. Boehm's
    distinct ``sd_*`` parameters) -- raises ``PybnfError`` (per-observable noise import is
    a later #407 chunk). A ``log-normal`` / ``log-laplace`` distribution, an expression
    ``noiseFormula``, and a per-point laplace placeholder raise ``NotImplementedError`` --
    the boundary is in code, not a silent mis-recovery.
    """
    families, sources = set(), []
    for row in observable_rows:
        dist = (row.noise_distribution or 'normal').lower()
        if dist not in _PETAB_DISTRIBUTION_TO_NOISE_MODEL:
            raise NotImplementedError(
                f"Observable '{row.observable_id}': noiseDistribution {dist!r} maps to a "
                f"PyBNF noise family on the natural-log scale (log-normal / log-laplace; "
                f"neg_bin was removed from PEtab v2), which has neither an objective token "
                f"nor a native noise_model line yet (#407). This chunk recovers the linear "
                f"normal / laplace families.")
        families.add(dist)
        formula = (row.noise_formula or '').strip()
        if not formula:
            raise PybnfError(
                f"Observable '{row.observable_id}' is missing a noiseFormula.")
        if formula.startswith('noiseParameter'):
            sources.append(('placeholder', None))
        else:
            # Reuse the observables asset for the numeric / bare-id / expression split (it
            # raises the deferred-expression boundary); we consume only its SigmaSource.
            _noise_model, source = noise_model_from_row(row)
            if isinstance(source, ConstantSigma):
                sources.append(('constant', source.const))
            elif isinstance(source, FreeParameterSigma):
                sources.append(('free', source.name))
            else:  # defensive: a bare row yields only these two sources
                raise PybnfError(
                    f"Observable '{row.observable_id}': noiseFormula {formula!r} maps to "
                    f"an unexpected sigma source {type(source).__name__}.")

    if len(families) != 1:
        raise PybnfError(
            f"The observables table mixes noise families across observables "
            f"(families={sorted(families)}), so it is not a single PyBNF objective. "
            f"Per-observable noise import is a later #407 chunk.")
    kinds = {kind for kind, _ in sources}
    if len(kinds) != 1:
        raise PybnfError(
            f"The observables table mixes noise sources across observables "
            f"(sources={sorted(kinds)}), so it is not a single PyBNF objective. "
            f"Per-observable noise import is a later #407 chunk.")
    family, kind = families.pop(), kinds.pop()
    petab_family, param = _PETAB_DISTRIBUTION_TO_NOISE_MODEL[family]

    if kind == 'placeholder':
        if family != 'normal':
            raise NotImplementedError(
                f"A per-point ({family}) placeholder noiseFormula has no PyBNF objective "
                f"token (only the Gaussian per-point _SD case, chi_sq, is recovered; #407).")
        return 'objective = chi_sq'
    if kind == 'free':
        return _free_sigma_directive(sources, petab_family, param)
    return _constant_sigma_directive(
        observable_rows, sources, family, petab_family, param, column_mean_of)


def _constant_sigma_directive(observable_rows, sources, family, petab_family, param,
                              column_mean_of):
    """Map an all-constant sigma to its directive: ``sos``/``sod`` (a unit sigma),
    ``ave_norm_sos`` (each observable's own column mean), or a ``fix_at`` ``noise_model``
    line (a uniform non-unit fixed sigma). A *different* fixed sigma per observable that is
    not the column-mean pattern is per-observable noise -> ``PybnfError`` (later #407)."""
    constants = [value for _kind, value in sources]
    if all(c == 1.0 for c in constants):
        return 'objective = sos' if family == 'normal' else 'objective = sod'
    if family == 'normal' and all(
            _approx(c, column_mean_of(row.observable_id))
            for row, c in zip(observable_rows, constants)):
        return 'objective = ave_norm_sos'
    uniq = set(constants)
    if len(uniq) != 1:
        raise PybnfError(
            f"The observables table has a different fixed sigma per observable "
            f"({sorted(uniq)}) that is neither all 1 (sos / sod) nor each observable's "
            f"column mean (ave_norm_sos), so it is per-observable noise, not one PyBNF "
            f"objective. Per-observable noise import is a later #407 chunk.")
    return f'noise_model = {petab_family}, {param} = fix_at {num(uniq.pop())}'


def _free_sigma_directive(sources, petab_family, param):
    """Map a single shared free-parameter sigma across all observables to a ``fit``
    ``noise_model`` line. The bare-id noiseFormula names an estimated parameter (already
    emitted as a bare free parameter by :func:`_free_parameters`), so ``fit <sigma_id>``
    binds to it as a nuisance -- it matches no model parameter id, the new-era typo
    check's allowed nuisance path (ADR-0034). A *distinct* free sigma per observable is
    per-observable noise (later #407). A sigma id that names no estimated parameter yields
    a conf whose ``fit`` reference has no declared free parameter; the config loader's
    bind-by-id typo check rejects it at load (one source of truth, ADR-0034)."""
    ids = {sid for _kind, sid in sources}
    if len(ids) != 1:
        raise PybnfError(
            f"The observables table names a different free-parameter sigma per observable "
            f"({sorted(ids)}); that is per-observable noise, not one PyBNF objective. "
            f"Per-observable noise import is a later #407 chunk.")
    sigma_id = ids.pop()
    return f'noise_model = {petab_family}, {param} = fit {sigma_id}'


def _approx(a, b):
    """Two sigmas are equal up to a relative tolerance (the column-mean comparison)."""
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Experiments: measurement groups + experiment rows -> (name, condition, data files)
# ---------------------------------------------------------------------------

def _experiments(datas, experiment_rows, out_dir):
    """Assemble the conf's experiments and write each one's ``.exp`` file.

    The set of experiments is the measurement groups (one wide ``Data`` per
    ``experimentId``); each ``Data`` is written to ``<name>.exp``. The experiment's
    condition comes from the experiments table (``cond_<c>`` -> ``c``; the synthesized
    ``cond_wildtype`` and an absent row -> no condition). A ``''`` experimentId is the
    "model as is" base time course (PEtab erased its name because the job had no
    fit-and-perturbed parameters); it is synthesized a name, which never reaches the PEtab
    output (it re-exports to ``''`` again). Returns ``[(name, condition_or_None,
    [data_file]), ...]`` in measurement order.
    """
    condition_of = {row.experiment_id: row.condition_id for row in experiment_rows}
    experiments = []
    for eid, data in datas.items():
        name = eid if eid else 'experiment1'
        condition = condition_name_from_id(condition_of.get(eid))
        data_file = f'{name}.exp'
        _write_exp(out_dir / data_file, data)
        experiments.append((name, condition, [data_file]))
    return experiments


def _write_exp(path, data):
    """Write a wide :class:`~pybnf.data.Data` as a PyBNF ``.exp`` file (a ``#``-prefixed
    header line + tab-separated rows, the shape ``Data.load_data`` reads back). ``NaN``
    cells are written ``nan`` (the forward pivot skips them on re-export)."""
    headers = [data.headers[i] for i in range(len(data.headers))]
    lines = ['# ' + '\t'.join(headers)]
    for i in range(data.data.shape[0]):
        lines.append('\t'.join(
            'nan' if np.isnan(data.data[i, j]) else num(data.data[i, j])
            for j in range(len(headers))))
    path.write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# The .conf writer (the disposable output half)
# ---------------------------------------------------------------------------

def _write_conf(path, *, model_filename, job_type, objective_directive, free_param_lines,
                conditions, experiments, method, method_overrides, settings, multi):
    """Write one new-era (edition 2) ``.conf``: the recovered problem + the supplied
    run-recipe (``job_type``, per-experiment ``method:``, required settings).

    ``objective_directive`` is the full recovered objective line -- either ``objective =
    <token>`` or a ``noise_model = <family>, ...`` line (:func:`_objective_directive`)."""
    stem = f'imported_{job_type}' if multi else 'imported'
    lines = [
        '# Imported from a PEtab v2 problem by pybnf.petab.import_job (#407).',
        '# The PEtab *problem* (parameters/priors, observables/noise, measurements,',
        '# conditions) is recovered exactly and round-trips through a re-export. The',
        '# run-recipe below (job_type + algorithm settings, the per-experiment method:,',
        '# and output/verbosity) is SUPPLIED, not recovered: PEtab is a problem spec with',
        '# no home for the method, so it is not part of the round-trip identity.',
        '',
        f'output_dir=output/{stem}',
        'edition = 2',
        '',
        f'model: {model_filename}',
        f'job_type = {job_type}',
        objective_directive,
        '',
    ]
    for name, perts in conditions.items():
        pert_str = ', '.join(f'{var} {op} {num(val)}' for var, op, val in perts)
        lines.append(f'condition: {name}, perturbations: {pert_str}')
    for name, condition, data_files in experiments:
        sim_method = method_overrides.get(name, method)
        cond_field = f', condition: {condition}' if condition else ''
        data_field = ', '.join(f'data: {f}' if i == 0 else f
                               for i, f in enumerate(data_files))
        lines.append(
            f'experiment: {name}{cond_field}, method: {sim_method}, {data_field}')
    lines.append('')
    lines.extend(free_param_lines)
    lines.append('')
    for key in ('population_size', 'max_iterations', 'verbosity'):
        lines.append(f'{key} = {settings[key]}')
    path.write_text('\n'.join(lines) + '\n')


def _emit_all_job_types():
    """The fit-type codes a ``job_type='all'`` import emits, from the registry (every
    ``optimizer`` + ``sampler``; the ``check`` checker excluded). Lazily imports
    ``pybnf.algorithms`` to populate the registry -- only the emit-all path pays for it,
    keeping the common single-job import free of the algorithm modules."""
    import pybnf.algorithms  # noqa: F401 -- side effect: populates FIT_TYPE_REGISTRY
    from ..registry import FIT_TYPE_REGISTRY
    return [code for code, entry in FIT_TYPE_REGISTRY.items()
            if entry.family in _EMIT_ALL_FAMILIES]


# ---------------------------------------------------------------------------
# problem.yaml reader (hand-parsed; the exporter emits a fixed, simple shape)
# ---------------------------------------------------------------------------

def read_problem_yaml(path):
    """Hand-parse the minimal ``problem.yaml`` shape :func:`write_problem_yaml` emits.

    Returns a dict with the table-file lists (``parameter_files`` / ``observable_files`` /
    ``measurement_files`` / ``condition_files`` / ``experiment_files``) and the single
    model (``model_file`` / ``model_id`` / ``model_language``). Dependency-free (no YAML
    library): the writer emits a flat ``key:`` + ``  - item`` list shape and a two-level
    ``model_files`` block, which a small indentation-aware scan reads exactly. The scan is
    **order-independent**, so a real v2 ``problem.yaml`` that lists ``model_files`` first
    (our writer emits it last) reads identically.

    This is a pure *reader*: it records the model ``language`` but does not enforce a
    policy on it, so a real (SBML) v2 problem still parses for inspection. The
    BNGL-native scope is enforced by the importer (:func:`_require_bngl_model`), not here.
    A second model raises ``NotImplementedError`` (multi-model is out of scope).
    """
    file_keys = ('parameter_files', 'observable_files', 'measurement_files',
                 'condition_files', 'experiment_files')
    files = {k: [] for k in file_keys}
    model_id = model_location = model_language = None

    section = None      # the current top-level *_files key (list items follow)
    in_model = False    # inside the model_files: block
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if indent == 0:
            section, in_model = None, False
            if stripped.endswith(':') and stripped[:-1] in files:
                section = stripped[:-1]
            elif stripped == 'model_files:':
                in_model = True
            # format_version and any other scalar top-level key: ignored
            continue
        if section is not None and stripped.startswith('-'):
            files[section].append(stripped[1:].strip())
        elif in_model:
            if stripped.startswith('location:'):
                model_location = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('language:'):
                model_language = stripped.split(':', 1)[1].strip()
            elif stripped.endswith(':'):
                if model_id is not None:
                    raise NotImplementedError(
                        f"problem.yaml declares more than one model ({model_id!r} and "
                        f"{stripped[:-1]!r}); multi-model import is out of scope (#407).")
                model_id = stripped[:-1].strip()

    _require_problem(files, model_location, path)
    return {**files, 'model_file': model_location, 'model_id': model_id,
            'model_language': model_language}


def _require_problem(files, model_location, path):
    for key in ('parameter_files', 'observable_files', 'measurement_files'):
        if not files[key]:
            raise PybnfError(f"problem.yaml at {path} has no {key}.")
    if model_location is None:
        raise PybnfError(f"problem.yaml at {path} declares no model file.")


def _require_bngl_model(problem, path):
    """Enforce the importer's BNGL-native scope on a parsed ``problem.yaml``.

    The reader (:func:`read_problem_yaml`) records the model ``language`` without judging
    it, so a real (SBML) v2 problem parses for inspection; the importer holds the policy:
    a non-``bngl`` model language raises ``NotImplementedError`` early (before any table is
    read), the same boundary the exporter draws on SBML -- it cannot be obtained by
    inversion, the SBML adapter is separate (#407, ADR-0025/0032). A ``None`` language
    (the field was absent) is permitted: the writer omits it only for a BNGL model.
    """
    language = problem.get('model_language')
    if language is not None and language != 'bngl':
        raise NotImplementedError(
            f"problem.yaml model '{problem.get('model_id')}' has language '{language}', "
            f"not 'bngl' (at {path}). Only BNGL-native PEtab problems are importable: an "
            f"SBML model is a separate adapter (the exporter raises on SBML too; it cannot "
            f"be obtained by inversion -- #407, ADR-0025).")
