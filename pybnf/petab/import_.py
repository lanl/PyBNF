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

**Dependency-free + simulator-free on the bare-name path.** Like the other read-path
chunks, the import path uses only stdlib + ``pybnf.data.Data`` + the asset mappers, so the
bare-name common case runs in the bngsim-less CI tier. ``problem.yaml`` is hand-parsed (the
exporter emits a fixed, simple shape). The ``petab`` library is the test-only oracle for the
bare-name path, and the optional ``pybnf[petab]`` extra for an expression ``observableFormula``.

**Scope (read path: BNGL and SBML).** Both model languages import (ADR-0036): the model file
is carried **verbatim** for each, and an expression ``observableFormula`` becomes a
first-class *measurement model* -- a PEtab math expression evaluated as a post-simulation
transform over the output trajectory (the observation layer), emitted as an
``observable: <id>, formula: <expr>`` conf line -- **never** by editing the model file (the
``begin functions`` synthesis of ADR-0035 is superseded). The bare-name common case still
needs no translator and stays dependency-free. SBML observables are 100% expressions, so SBML
import pulls in the ``pybnf[petab]`` extra. Out of scope, each mirroring an export-side
boundary: fitting the imported job (gated on the ADR-0028 config loader, #423); the
``noiseFormula`` / condition sympy layer and per-measurement
``observableParameters``/``noiseParameters`` placeholders (bare names / numbers only); the
five PEtab prior families PyBNF lacks; one-sided truncation; multi-model; parameter-scan /
dose-response; replicate reconstruction.
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
from .measurements import data_from_measurement_rows, read_measurement_table
from .observables import noise_model_from_row, read_observable_table
from .parameters import free_parameter_from_row, read_parameter_table
from ._bngl import parse_model as parse_bngl_model
from ._sbml import parse_model as parse_sbml_model
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

    Both **BNGL and SBML** models import (ADR-0036): the model file is carried verbatim, and
    an **expression** ``observableFormula`` (e.g. a quotient of sums) becomes a conf
    measurement model (``observable: <id>, formula: <expr>``) evaluated post-simulation -- the
    optional ``pybnf[petab]`` extra. Raises ``NotImplementedError`` at the remaining
    PEtab/PyBNF boundaries (a model language other than ``bngl``/``sbml``; the five
    unsupported prior families; a log-normal/log-laplace or expression noise model; a
    ``noiseFormula``/condition expression; a per-measurement
    ``observableParameters``/``noiseParameters`` placeholder in a formula; replicate rows)
    and ``PybnfError`` for a malformed problem (including an ``observableFormula`` symbol that
    is not a model entity).
    """
    problem_yaml_path = Path(problem_yaml_path)
    base = problem_yaml_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = read_problem_yaml(problem_yaml_path)
    _require_supported_model(problem, problem_yaml_path)
    language = (problem.get('model_language') or 'bngl').lower()

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

    # The model is read now (not just at write time) to validate each expression
    # observableFormula's free symbols against the model's entity namespace (the BNGL
    # ParamList, or SBML species u parameters -- ADR-0026/0036). The model file itself is
    # carried **byte-verbatim** for every language -- the measurement model is a post-sim
    # observation layer, never a model-file edit (ADR-0036, superseding the ADR-0035
    # begin-functions synthesis).
    model_filename = problem['model_file']
    model_text = (base / model_filename).read_text(encoding='utf-8', errors='replace')
    namespace, entity_names = _model_namespace(model_text, language)

    # Observables -> the observableId -> model-column map (the data pivot's column order)
    # plus the measurement models (id, formula) synthesized from expression
    # observableFormulas (ADR-0036: emitted as conf `observable: ... formula:` lines).
    observable_id_to_column, measurement_models = _observable_id_to_column(
        observable_rows, namespace, entity_names)

    # Measurements -> one wide Data per experiment, then assemble the experiment list.
    datas = data_from_measurement_rows(measurement_rows, observable_id_to_column)
    objective_directive = _objective_directive(
        observable_rows, _column_mean_resolver(datas, observable_id_to_column))
    conditions = conditions_from_rows(condition_rows, surrogate_params)
    experiments = _experiments(datas, experiment_rows, out_dir)

    # The model file is carried verbatim -- no synthesis, no edit, for BNGL or SBML
    # (ADR-0036). Expression observables live in the conf's measurement-model layer below.
    (out_dir / model_filename).write_text(model_text)

    merged_settings = {**_DEFAULT_SETTINGS, **(settings or {})}
    job_types = _emit_all_job_types() if job_type == 'all' else [job_type]
    for jt in job_types:
        conf_name = f'imported_{jt}.conf' if len(job_types) > 1 else 'imported.conf'
        _write_conf(
            out_dir / conf_name, model_filename=model_filename, job_type=jt,
            objective_directive=objective_directive, free_param_lines=free_param_lines,
            conditions=conditions, experiments=experiments,
            measurement_models=measurement_models, method=method,
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

def _model_namespace(model_text, language):
    """The model's expression namespace + the full entity name set, per language (ADR-0036).

    Returns ``(namespace_symbols, entity_names)``: ``namespace_symbols`` are the names an
    ``observableFormula`` may reference (the BNGL ``ParamList`` -- parameters u observables u
    functions; or SBML species u parameters u compartments -- ADR-0026/0036);
    ``entity_names`` is the broader declared-name set used for the shadow check (a measurement
    model's id must not collide with a model output column). Read from the model text directly
    with the stdlib scanners (``_bngl`` / ``_sbml``), simulator-free.
    """
    if language == 'sbml':
        ent = parse_sbml_model(model_text)
        return ent.namespace_symbols, set(ent.namespace_symbols)
    ent = parse_bngl_model(model_text)
    namespace = (set(ent.parameters) | set(ent.observable_names)
                 | set(ent.function_names))
    entity_names = (namespace | set(ent.molecule_type_names)
                    | set(ent.compartment_names))
    return namespace, entity_names


def _observable_id_to_column(observable_rows, namespace, entity_names):
    """Map each ``observableId`` to the model column it measures, recording a measurement
    model for any expression ``observableFormula`` (ADR-0036). Iteration order = table order,
    which fixes the wide-data column order on the measurement pivot.

    Returns ``(mapping, measurement_models)``:

    * A **bare model-entity name** ``observableFormula`` (the common case, ADR-0025) maps its
      ``observableId`` to that name -- PyBNF matches the ``.exp`` column to the model
      observable/function/species by name and the backend already produces it, so no
      translator runs and the path stays dependency-free.
    * An **expression** ``observableFormula`` becomes a *measurement model* ``(id, formula)``
      -- a PEtab math expression evaluated post-simulation by the observation layer (ADR-0036).
      Its free symbols are validated against the model namespace (the optional ``pybnf[petab]``
      extra), and the ``.exp`` column is named after the ``observableId`` (the column the layer
      materializes). The model file is **not** edited.

    A measurement model's id must not shadow an existing model entity (``PybnfError``, so the
    materialized column does not collide with a model output column); an unknown free symbol or
    a per-measurement placeholder in the expression raises in the validator (``PybnfError`` /
    ``NotImplementedError``).
    """
    taken = set(entity_names)
    mapping = {}
    measurement_models = []
    for row in observable_rows:
        formula = (row.observable_formula or '').strip()
        if _IDENTIFIER.match(formula):
            if formula not in namespace:
                raise PybnfError(
                    f"Observable '{row.observable_id}' has a bare observableFormula "
                    f"'{formula}', which is not a model entity. A bare-name observableFormula "
                    f"must name a model observable/function/species the backend outputs; an "
                    f"unknown name is an error (ADR-0036).",
                    f"Model namespace: {sorted(namespace)}.")
            mapping[row.observable_id] = formula          # bare-name path (no translator)
            continue
        # Expression -> a measurement model. Validate its symbols against the namespace now
        # (fail fast; requires the petab extra). The conf carries it as a formula line and
        # the observation layer evaluates it post-simulation -- no model-file edit (ADR-0036).
        from .formula import compile_petab_formula
        compile_petab_formula(
            formula, namespace,
            detail=f"Model namespace (species/parameters/observables/functions): "
                   f"{sorted(namespace)}.")
        obs_id = row.observable_id
        if obs_id in taken:
            raise PybnfError(
                f"Cannot import the expression observableFormula of observable '{obs_id}': "
                f"its id already names a model entity, so the measurement-model column would "
                f"shadow a model output column. Rename the observableId (ADR-0036).")
        taken.add(obs_id)
        measurement_models.append((obs_id, formula))
        mapping[obs_id] = obs_id    # the materialized measurement-model column is named obs_id
    if not mapping:
        raise PybnfError("The PEtab observables table declares no observables.")
    return mapping, measurement_models


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
                conditions, experiments, measurement_models, method, method_overrides,
                settings, multi):
    """Write one new-era (edition 2) ``.conf``: the recovered problem + the supplied
    run-recipe (``job_type``, per-experiment ``method:``, required settings).

    ``objective_directive`` is the full recovered objective line -- either ``objective =
    <token>`` or a ``noise_model = <family>, ...`` line (:func:`_objective_directive`).
    ``measurement_models`` is the list of ``(observableId, formula)`` expression observables,
    emitted as ``observable: <id>, formula: <expr>`` measurement-model lines (ADR-0036)."""
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
    ]
    # Expression observables: a measurement-model formula evaluated post-simulation (the
    # observation layer, ADR-0036), not a model-file edit. The model is carried verbatim.
    for obs_id, formula in measurement_models:
        lines.append(f'observable: {obs_id}, formula: {formula}')
    lines.append('')
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
    policy on it. The supported-language scope (BNGL or SBML, ADR-0036) is enforced by the
    importer (:func:`_require_supported_model`), not here. A second model raises
    ``NotImplementedError`` (multi-model is out of scope).
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


def _require_supported_model(problem, path):
    """Enforce the importer's supported-language scope on a parsed ``problem.yaml`` (ADR-0036).

    The reader (:func:`read_problem_yaml`) records the model ``language`` without judging it;
    the importer holds the policy. **BNGL and SBML both import**: the model file is carried
    verbatim and an expression ``observableFormula`` becomes a post-simulation measurement
    model (the observation layer), so neither the ``.bngl`` nor the ``.xml`` is ever edited
    (ADR-0036). Any other model language (e.g. ``pysb``) raises ``NotImplementedError`` early,
    before any table is read. A ``None`` language (the field was absent) is permitted -- the
    exporter omits it only for a BNGL model.
    """
    language = problem.get('model_language')
    if language is not None and language.lower() not in ('bngl', 'sbml'):
        raise NotImplementedError(
            f"problem.yaml model '{problem.get('model_id')}' has language '{language}', "
            f"not 'bngl' or 'sbml' (at {path}). Only BNGL and SBML PEtab problems are "
            f"importable (ADR-0036); other model languages are out of scope (#407).")
