"""PEtab v2 problem importer: a BNGL-native PEtab v2 problem -> a new-era PyBNF job
(issue #407; the importer read path, ADR-0025 reversed / ADR-0032).

The inverse of :func:`pybnf.petab.export.export_job`. Given a ``problem.yaml`` + its TSV
tables + a BNGL model, :func:`import_job` writes a runnable new-era (edition 2) ``.conf``
plus the ``.exp`` data files and a fit-instrumented copy of the model -- the form the
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

**Scope (read path, BNGL-native).** Out of scope, each mirroring an export-side boundary:
fitting the imported job (gated on the ADR-0028 config loader, #423); SBML model import;
the ``observableFormula`` / ``noiseFormula`` / condition sympy layer (bare names only);
the five PEtab prior families PyBNF lacks; one-sided truncation; multi-model;
parameter-scan / dose-response; replicate reconstruction. The bare-name common case needs
no formula translator.
"""

import re
from pathlib import Path

import numpy as np

from ..printing import PybnfError
from .conditions import (
    REF_MARKER,
    condition_name_from_id,
    conditions_from_rows,
    read_condition_table,
    read_experiment_table,
)
from .measurements import data_from_measurement_rows, read_measurement_table
from .observables import read_observable_table
from .parameters import free_parameter_from_row, read_parameter_table
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
    new-era PyBNF job: the ``.exp`` data files, a fit-instrumented copy of the BNGL model,
    and one or more ``.conf`` files. The *problem* (parameters/priors, observables/noise,
    measurements, conditions/experiments) is recovered exactly; the *run-recipe*
    (``job_type``, ``method``, ``settings``) is supplied by the caller (see the module
    docstring). Returns the ``out_dir`` path.

    ``job_type`` is the SEARCH method token, or ``'all'`` to emit one
    ``imported_<jt>.conf`` per registered optimizer + sampler. ``method`` (default
    ``'ode'``) is the per-experiment SIMULATION method; ``method_overrides`` (a
    ``{experiment_name: method}`` map) sets per-experiment values. ``settings`` overrides
    the required algorithm/run settings.

    Raises ``NotImplementedError`` at the documented PEtab/PyBNF boundaries (a non-``bngl``
    model language; the five unsupported prior families; a log-normal/log-laplace or
    expression noise model; a ``noiseFormula``/condition/``observableFormula`` expression;
    replicate rows) and ``PybnfError`` for a malformed problem.
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

    # Parameters -> conf free-parameter lines + the surrogate set M + the estimated
    # model parameters (the ones whose __FREE marker the model copy must re-instrument).
    free_param_lines, surrogate_params, estimated_to_free = _free_parameters(parameter_rows)

    # Observables -> the observableId -> model-column map (the data pivot's column order)
    # and the recovered objective token.
    observable_id_to_column = _observable_id_to_column(observable_rows)

    # Measurements -> one wide Data per experiment, then assemble the experiment list.
    datas = data_from_measurement_rows(measurement_rows, observable_id_to_column)
    objective_token = _objective_token(
        observable_rows, _column_mean_resolver(datas, observable_id_to_column))
    conditions = conditions_from_rows(condition_rows, surrogate_params)
    experiments = _experiments(datas, experiment_rows, out_dir)

    # The fit-instrumented model copy (re-add the __FREE markers the export stripped).
    model_filename = problem['model_file']
    model_text = (base / model_filename).read_text(encoding='utf-8', errors='replace')
    (out_dir / model_filename).write_text(
        _reinstrument_free_parameters(model_text, estimated_to_free))

    merged_settings = {**_DEFAULT_SETTINGS, **(settings or {})}
    job_types = _emit_all_job_types() if job_type == 'all' else [job_type]
    for jt in job_types:
        conf_name = f'imported_{jt}.conf' if len(job_types) > 1 else 'imported.conf'
        _write_conf(
            out_dir / conf_name, model_filename=model_filename, job_type=jt,
            objective_token=objective_token, free_param_lines=free_param_lines,
            conditions=conditions, experiments=experiments, method=method,
            method_overrides=method_overrides or {}, settings=merged_settings,
            multi=len(job_types) > 1)
    return out_dir


# ---------------------------------------------------------------------------
# Parameters: rows -> conf free-parameter lines + the surrogate set
# ---------------------------------------------------------------------------

def _free_parameters(parameter_rows):
    """Map estimated parameter rows to conf ``*_var`` lines + bookkeeping.

    Returns ``(free_param_lines, surrogate_params, estimated_to_free)``:
    ``free_param_lines`` are the conf declarations (in table order); ``surrogate_params``
    is the set ``M`` of fit-and-perturbed model parameters (a ``<p>__REF`` parameterId
    recovered to ``p``); ``estimated_to_free`` maps each estimated model parameter to its
    re-added ``<p>__FREE`` name (the model copy uses it to re-instrument the parameters
    block). ``free_parameter_from_row`` surfaces the prior boundaries (5 families,
    one-sided truncation) as ``NotImplementedError``.
    """
    free_param_lines = []
    surrogate_params = set()
    estimated_to_free = {}
    for row in parameter_rows:
        if not row.estimate:
            continue  # a fixed model constant, not a free parameter (stays in the model)
        model_param, is_surrogate = _model_param(row.parameter_id)
        if is_surrogate:
            surrogate_params.add(model_param)
        free_name = f'{model_param}__FREE'
        estimated_to_free[model_param] = free_name
        fp = free_parameter_from_row(row)
        free_param_lines.append(
            f'{fp.type} = {free_name} {num(fp.p1)} {num(fp.p2)}')
    if not free_param_lines:
        raise PybnfError(
            "The PEtab parameters table declares no estimated (estimate=true) parameters, "
            "so there is nothing to fit.")
    return free_param_lines, surrogate_params, estimated_to_free


def _model_param(parameter_id):
    """``(model_param, is_surrogate)``: a ``<p>__REF`` surrogate id -> ``(p, True)``;
    a plain parameterId -> ``(parameterId, False)``."""
    if parameter_id.endswith(REF_MARKER):
        return parameter_id[:-len(REF_MARKER)], True
    return parameter_id, False


# ---------------------------------------------------------------------------
# Observables: rows -> column map + objective token
# ---------------------------------------------------------------------------

def _observable_id_to_column(observable_rows):
    """Map each ``observableId`` to the model column it measures (its ``observableFormula``,
    the bare model-entity name -- ADR-0025). Iteration order = table order, which fixes the
    wide-data column order on the measurement pivot. A non-bare ``observableFormula``
    raises ``NotImplementedError`` (the deferred sympy layer)."""
    mapping = {}
    for row in observable_rows:
        formula = (row.observable_formula or '').strip()
        if not _IDENTIFIER.match(formula):
            raise NotImplementedError(
                f"Observable '{row.observable_id}' has observableFormula {formula!r}, "
                f"which is not a bare model-entity name. Evaluating PEtab observable "
                f"formulae needs the sympy layer (the deferred observableFormula chunk, "
                f"#407), which adopts the petab library.")
        mapping[row.observable_id] = formula
    if not mapping:
        raise PybnfError("The PEtab observables table declares no observables.")
    return mapping


def _column_mean_resolver(datas, observable_id_to_column):
    """A ``observableId -> column mean across all experiments`` closure (for distinguishing
    ``sos`` from ``ave_norm_sos``; mirrors the export's column-mean sigma over all data)."""
    def column_mean_of(observable_id):
        col = observable_id_to_column[observable_id]
        values = [data[col] for data in datas.values() if col in data.cols]
        return float(np.average(np.concatenate(values)))
    return column_mean_of


def _objective_token(observable_rows, column_mean_of):
    """Recover the PyBNF ``objective`` token from the observables' noise columns -- the
    inverse of the objective-family export (``_OBJECTIVE_DESUGAR`` reversed):

    * ``normal`` + a per-point placeholder ``noiseFormula`` -> ``chi_sq``;
    * ``normal`` + a constant ``1`` (every observable) -> ``sos``;
    * ``normal`` + a constant equal to each observable's column mean -> ``ave_norm_sos``;
    * ``laplace`` + a constant ``1`` -> ``sod``.

    A single PyBNF objective is one family + one sigma source across all observables, so a
    mix raises ``PybnfError``. ``log-normal`` / ``log-laplace`` families, an expression
    ``noiseFormula``, a per-point laplace placeholder, and a uniform non-unit fixed sigma
    (a ``noise_model`` line, not a token) raise ``NotImplementedError`` -- the boundary is
    in code, not a silent mis-recovery.
    """
    families, kinds, constants = set(), set(), {}
    for row in observable_rows:
        dist = (row.noise_distribution or 'normal').lower()
        if dist not in ('normal', 'laplace'):
            raise NotImplementedError(
                f"Observable '{row.observable_id}': noiseDistribution {dist!r} maps to a "
                f"PyBNF noise family with no objective token (log-normal / log-laplace are "
                f"natural-log families; neg_bin was removed from PEtab v2). Importing it "
                f"as a noise_model line is a later #407 chunk; this chunk recovers the "
                f"objective tokens chi_sq / sos / sod / ave_norm_sos.")
        families.add(dist)
        formula = (row.noise_formula or '').strip()
        if not formula:
            raise PybnfError(
                f"Observable '{row.observable_id}' is missing a noiseFormula.")
        if formula.startswith('noiseParameter'):
            kinds.add('placeholder')
        else:
            try:
                constants[row.observable_id] = float(formula)
            except ValueError:
                raise NotImplementedError(
                    f"Observable '{row.observable_id}': noiseFormula {formula!r} is an "
                    f"expression, not a number or a per-point placeholder. PEtab noise "
                    f"formulae need the sympy layer (deferred, #407).")
            kinds.add('constant')

    if len(families) != 1 or len(kinds) != 1:
        raise PybnfError(
            "The observables table mixes noise families/sources across observables, so it "
            f"is not a single PyBNF objective (families={sorted(families)}, "
            f"sources={sorted(kinds)}). Per-observable noise import is a later #407 chunk.")
    family, kind = families.pop(), kinds.pop()

    if kind == 'placeholder':
        if family != 'normal':
            raise NotImplementedError(
                f"A per-point ({family}) placeholder noiseFormula has no PyBNF objective "
                f"token (only the Gaussian per-point _SD case, chi_sq, is recovered; #407).")
        return 'chi_sq'
    if family == 'laplace':
        if all(c == 1.0 for c in constants.values()):
            return 'sod'
        raise NotImplementedError(
            "A Laplace likelihood with a non-unit fixed scale maps to a 'noise_model = "
            "laplace, scale = fix_at <v>' line, not an objective token (later #407 chunk).")
    if all(c == 1.0 for c in constants.values()):
        return 'sos'
    # Otherwise it must be ave_norm_sos (each observable's fixed sigma is its own column
    # mean). Verify against the data so a uniform non-unit fixed sigma (a noise_model
    # fix_at line, out of scope for token recovery) raises loudly, not silently.
    for oid, c in constants.items():
        mean = column_mean_of(oid)
        if abs(c - mean) > 1e-9 * max(1.0, abs(mean)):
            raise NotImplementedError(
                f"Observable '{oid}' has a fixed Gaussian sigma {c} that is neither 1 "
                f"(sos) nor its column mean {mean} (ave_norm_sos): a uniform fixed sigma "
                f"maps to a 'noise_model = normal, sigma = fix_at {c}' line, not an "
                f"objective token (later #407 chunk).")
    return 'ave_norm_sos'


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

def _write_conf(path, *, model_filename, job_type, objective_token, free_param_lines,
                conditions, experiments, method, method_overrides, settings, multi):
    """Write one new-era (edition 2) ``.conf``: the recovered problem + the supplied
    run-recipe (``job_type``, per-experiment ``method:``, required settings)."""
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
        f'objective = {objective_token}',
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
# Model re-instrumentation (the inverse of clean_model_for_petab)
# ---------------------------------------------------------------------------

def _reinstrument_free_parameters(text, estimated_to_free):
    """Re-add the ``__FREE`` markers ``clean_model_for_petab`` stripped (its inverse).

    In the ``begin parameters`` block, an estimated parameter's nominal RHS is replaced by
    its ``<name>__FREE`` marker, so the conf's free parameter binds to it again (the
    exporter's ``free_to_param`` lookup). The nominal value the export wrote is lossy (the
    bounds midpoint, not the original marker), so the *estimated set* -- not the value --
    drives this; fixed parameters keep their model value. A re-export then rewrites
    ``<name>__FREE`` back to the same midpoint, so the model round-trips byte-for-byte.
    """
    begin = re.compile(r'^\s*begin\s+parameters\b', re.I)
    end = re.compile(r'^\s*end\s+parameters\b', re.I)
    lines = text.split('\n')
    in_block = False
    for i, raw in enumerate(lines):
        if begin.match(raw):
            in_block = True
        elif end.match(raw):
            in_block = False
        elif in_block:
            code = raw.split('#', 1)[0]
            if code.strip():
                match = re.match(r'^(\s*)(\w+)(?:\s*=\s*|\s+)(.+?)\s*$', code)
                if match and match.group(2) in estimated_to_free:
                    name = match.group(2)
                    lines[i] = f'{match.group(1)}{name} {estimated_to_free[name]}'
    return '\n'.join(lines)


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
