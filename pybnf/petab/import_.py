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

**Scope (read path: BNGL and SBML, one or many models).** Both model languages import
(ADR-0036): the model file is carried **verbatim** for each, and an expression
``observableFormula`` becomes a first-class *measurement model* -- a PEtab math expression
evaluated as a post-simulation transform over the output trajectory (the observation layer),
emitted as an ``observable: <id>, formula: <expr>`` conf line -- **never** by editing the model
file (the ``begin functions`` synthesis of ADR-0035 is superseded). The bare-name common case
still needs no translator and stays dependency-free. SBML observables are 100% expressions, so
SBML import pulls in the ``pybnf[petab]`` extra. A **multi-model** problem imports too (ADR-0041):
each ``model_files`` entry is carried verbatim and declared with its own ``model:`` line, an
expression observableFormula validates against the union of every model's namespace, and each
experiment's model is recovered from the ``modelId`` on its measurement rows (emitted as a
per-experiment ``model:`` field; a BNGL + SBML mix is fine). A **constant-per-observable**
``observableParameters`` scale/offset and an expression ``noiseFormula`` import too (ADR-0044):
the placeholder is substituted into the observable/noise formula (an id resolves from the PSet,
a number inlines), an expression ``noiseFormula`` becoming a ``FormulaSigma`` (``noise_model
<obs> = <family>, <param> = formula <expr>``). A **dose-response** (parameter_scan) problem
imports too (ADR-0046): N Conditions each setting one swept parameter at a constant measurement
time (``inf`` => steady state, or a finite ``t_end``) are reconstructed into a single swept-axis
``.exp`` (column 0 the swept parameter) + a ``parameter_scan`` ``experiment:`` (the inverse of
the exporter's dose-response emission -- ``reconstruct_dose_responses``). Out of scope, each
mirroring an export-side boundary: a condition-table sympy layer; the five PEtab prior families
PyBNF lacks; a dose-response that also carries a named condition or row-varying per-measurement
placeholders. (One-sided truncation now maps to a half-bounded box -- ADR-0047, #432.)
"""

import math
import re
from pathlib import Path

import numpy as np

from ..printing import PybnfError
from ..priors import PRIOR_KEYWORD_MAP
from .conditions import (
    REF_MARKER,
    condition_name_from_id,
    conditions_from_rows,
    read_condition_table,
    read_experiment_table,
)
from .measurements import (
    data_from_measurement_rows,
    measurement_param_bindings,
    noise_parameter_ids_by_observable,
    observable_parameters_by_observable,
    read_measurement_table,
    reconstruct_dose_responses,
    row_varying_noise_ids,
    row_varying_observable_ids,
)
from ._measurement_params import write_measurement_params
from .observables import read_observable_table
from .parameters import free_parameter_from_row, read_parameter_table
from ._bngl import parse_model as parse_bngl_model
from ._sbml import parse_model as parse_sbml_model
from ._tsv import num

# A bare model-entity name (an observableFormula in the common case). Anything with
# operators / calls / whitespace is an expression for the deferred sympy layer.
_IDENTIFIER = re.compile(r'[A-Za-z_]\w*\Z')

# A PEtab per-measurement placeholder anywhere in a formula (``observableParameter1_<id>`` /
# ``noiseParameter2_<id>``). Phase 1 substitutes a constant-per-observable placeholder away
# (ADR-0044); a leftover one (no measurements-table value) is the deferred frontier and raises
# pointing at #428 Phase 2, except a known row-varying one which is kept for per-point binding
# (ADR-0045).
_PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d')

# The FULL placeholder symbol (``observableParameter1_obs_y``), used to admit a kept row-varying
# placeholder into a measurement-model formula's allowed-symbol set at validation (ADR-0045): the
# placeholder is not a model entity nor a free parameter -- its value is bound per data point.
_PLACEHOLDER_SYMBOL = re.compile(r'(?:observable|noise)Parameter\d+_\w+')


def _placeholder_subs(observable_id, obs_params, noise_param_ids=None):
    """The ``{placeholder_name: token}`` substitution for one observable (ADR-0044).

    Binds ``observableParameter${n}_${observable_id}`` to the n-th constant-per-observable
    ``observableParameters`` token (``obs_params[observable_id]``), and -- when
    ``noise_param_ids`` is given (the noiseFormula side) -- ``noiseParameter1_${observable_id}``
    to that observable's constant ``noiseParameters`` id. An empty map means nothing to
    substitute (the bare-name / no-placeholder common case stays dependency-free)."""
    subs = {f'observableParameter{n}_{observable_id}': tok
            for n, tok in enumerate(obs_params.get(observable_id, ()), start=1)}
    if noise_param_ids is not None:
        nid = noise_param_ids.get(observable_id)
        if nid is not None:
            subs[f'noiseParameter1_{observable_id}'] = nid
    return subs


def _require_no_placeholder(formula, observable_id):
    """Raise the deferred-frontier ``NotImplementedError`` if a placeholder survived
    substitution (a placeholder with no measurements-table value, or a row-varying one --
    ADR-0044, #428 Phase 2)."""
    if _PLACEHOLDER.search(formula):
        raise NotImplementedError(
            f"Observable '{observable_id}': the formula still references a PEtab "
            f"per-measurement placeholder after substitution ({formula!r}). Phase 1 "
            f"substitutes a placeholder whose observableParameters/noiseParameters value is "
            f"constant across the observable's rows; an unresolved or row-varying placeholder "
            f"is the deferred per-measurement frontier (#428 Phase 2 / ADR-0044).")


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
    optional ``pybnf[petab]`` extra. A **constant-per-observable** ``observableParameters``
    scale/offset and an expression ``noiseFormula`` are substituted/reduced and import too
    (ADR-0044). A **dose-response** (parameter_scan) problem -- N conditions each setting one
    swept parameter at a constant measurement time (``inf`` => steady state, ADR-0046) -- is
    reconstructed into a single swept-axis ``.exp`` + a ``parameter_scan`` experiment. Raises
    ``NotImplementedError`` at the remaining PEtab/PyBNF boundaries (a model language other than
    ``bngl``/``sbml``; the five unsupported prior families; a log-normal/log-laplace noise
    distribution; a condition expression; a **row-varying** per-measurement
    ``observableParameters``/``noiseParameters`` placeholder; replicate rows) and ``PybnfError``
    for a malformed problem (an ``observableFormula`` symbol that is not a model entity, or an
    ambiguous dose-response group).
    """
    problem_yaml_path = Path(problem_yaml_path)
    base = problem_yaml_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = read_problem_yaml(problem_yaml_path)
    _require_supported_model(problem, problem_yaml_path)
    models = problem['models']

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

    # Fixed PEtab parameters carrying a numeric value: the constants a measurement-model
    # observableFormula may reference that live only in the parameters table, not the
    # model file (Boehm's specC17 = 0.107 -- ADR-0037). Inlined into the formula below.
    fixed_params = {row.parameter_id: float(row.nominal_value)
                    for row in parameter_rows
                    if not row.estimate and row.nominal_value is not None}

    # The conf free-parameter names (a <p>__REF surrogate recovered to p). A measurement-model
    # observableFormula may reference one as an observation-layer nuisance -- an
    # observableParameters scale/offset substituted in (ADR-0044) -- so it must validate
    # against the namespace u these names, not the model namespace alone.
    free_names = {_model_param(row.parameter_id)[0]
                  for row in parameter_rows if row.estimate}

    # The constant-per-observable observableParameters tokens (ADR-0044): the n-th token binds
    # observableParameter${n}_${id}, substituted into the observable/noise formulae below. A
    # row-varying observableParameters is no longer an error -- it routes to the per-measurement
    # binding table (ADR-0045): the observableFormula keeps its placeholder and the per-row
    # scale/offset token is bound per data point from the sidecar (a PerMeasurementModel).
    obs_params = observable_parameters_by_observable(measurement_rows)
    row_varying_obs_params = row_varying_observable_ids(measurement_rows)

    # Each model is read now (not just at write time) to validate each expression
    # observableFormula's free symbols against the models' entity namespace (the BNGL
    # ParamList, or SBML species u parameters -- ADR-0026/0036). A multi-model job (ADR-0041)
    # validates a (model-agnostic) observableFormula against the **union** of every model's
    # namespace. Each model file is carried **byte-verbatim** -- the measurement model is a
    # post-sim observation layer, never a model-file edit (ADR-0036).
    model_texts = {}            # location -> verbatim text
    namespaces, entity_name_sets = [], []
    for m in models:
        loc, lang = m['location'], (m['language'] or 'bngl').lower()
        text = (base / loc).read_text(encoding='utf-8', errors='replace')
        model_texts[loc] = text
        ns, ents = _model_namespace(text, lang)
        namespaces.append(ns)
        entity_name_sets.append(ents)
    namespace = set().union(*namespaces)
    entity_names = set().union(*entity_name_sets)

    # Observables -> the observableId -> model-column map (the data pivot's column order)
    # plus the measurement models (id, formula) synthesized from expression
    # observableFormulas (ADR-0036: emitted as conf `observable: ... formula:` lines).
    observable_id_to_column, measurement_models = _observable_id_to_column(
        observable_rows, namespace, entity_names, fixed_params, obs_params, free_names,
        row_varying_obs_params)

    # Dose-response (parameter_scan) reconstruction (ADR-0046): pull out the experiment groups
    # whose N conditions each set one swept parameter at a constant measurement time (inf =>
    # steady state) and rebuild each as a single swept-axis Data; the remaining rows are time
    # courses. Their conditions/experiments are dropped from the time-course reconstruction (a
    # dose is the scan axis, not a named condition: line).
    dose_responses, tc_rows, dr_condition_ids, dr_experiment_ids = reconstruct_dose_responses(
        measurement_rows, condition_rows, experiment_rows, observable_id_to_column)

    # Time-course measurements -> the wide Data replicates per (experiment, model), then assemble
    # the experiment list (repeated (obs, time) rows are dealt into replicate grids -- ADR-0039;
    # the modelId distinguishes experiments that share an experimentId -- ADR-0041).
    datas = data_from_measurement_rows(tc_rows, observable_id_to_column)
    # The objective/noise are GLOBAL across the fit, so they read ALL measurement rows (a
    # dose-response observable shares the objective). A constant-per-observable parameter-id
    # noiseParameters placeholder is a per-observable estimated sigma (Boehm's sd_*); the map
    # drives the per-observable noise_model lines (ADR-0037). A row-varying id routes to the
    # per-measurement binding table (ADR-0045): the per-data-point sidecar carrying the row's
    # estimated noise id, emitted as a 'sigma = formula <placeholder>' line.
    noise_param_ids = noise_parameter_ids_by_observable(measurement_rows)
    row_varying_obs = row_varying_noise_ids(measurement_rows)
    # One sidecar carries both row-varying frontiers (ADR-0045): the row-varying noise ids and
    # the row-varying observableParameters scale/offset tokens, keyed by data column. Keyed to a
    # time-course experiment's rows (a dose-response carries no per-measurement sidecar).
    param_bindings = measurement_param_bindings(
        tc_rows, observable_id_to_column, row_varying_obs, row_varying_obs_params)
    # The column-mean resolver (sos vs ave_norm_sos) averages over every experiment's data,
    # time courses and dose-response scans alike.
    dr_datas = {(dr['name'], dr['model_id']): [dr['data']] for dr in dose_responses}
    objective_directives = _objective_directives(
        observable_rows, observable_id_to_column, noise_param_ids,
        _column_mean_resolver({**datas, **dr_datas}, observable_id_to_column),
        obs_params, row_varying_obs)
    # Named conditions exclude those absorbed into a dose-response (each dose is the scan axis,
    # not a condition: line).
    tc_condition_rows = [r for r in condition_rows if r.condition_id not in dr_condition_ids]
    conditions = conditions_from_rows(tc_condition_rows, surrogate_params)
    # Each (experiment, model) group recovers its model from the rows' modelId (ADR-0041);
    # a single-model job carries modelId '' and emits no per-experiment model: field. A group
    # with a row-varying noise binding also writes its per-measurement sidecar (ADR-0045). The
    # time-course experiment rows exclude those absorbed into a dose-response scan.
    model_location_of = {m['model_id']: m['location'] for m in models}
    tc_experiment_rows = [r for r in experiment_rows
                          if r.experiment_id not in dr_experiment_ids]
    experiments = _experiments(datas, tc_experiment_rows, out_dir, model_location_of,
                               param_bindings)
    # Dose-response scans become parameter_scan experiments: a steady-state scan (scan_time inf)
    # carries no t_end: (the .exp's swept-axis column 0 infers the type); a finite scan carries
    # t_end: <t> (ADR-0046). Their .exp files are written here.
    experiments += _dose_response_experiments(dose_responses, out_dir, model_location_of)

    # Each model file is carried verbatim -- no synthesis, no edit, for BNGL or SBML
    # (ADR-0036). Expression observables live in the conf's measurement-model layer below.
    for loc, text in model_texts.items():
        (out_dir / loc).write_text(text)

    merged_settings = {**_DEFAULT_SETTINGS, **(settings or {})}
    model_filenames = [m['location'] for m in models]
    job_types = _emit_all_job_types() if job_type == 'all' else [job_type]
    for jt in job_types:
        conf_name = f'imported_{jt}.conf' if len(job_types) > 1 else 'imported.conf'
        _write_conf(
            out_dir / conf_name, model_filenames=model_filenames, job_type=jt,
            objective_directives=objective_directives, free_param_lines=free_param_lines,
            conditions=conditions, experiments=experiments,
            measurement_models=measurement_models, method=method,
            method_overrides=method_overrides or {}, settings=merged_settings,
            multi=len(job_types) > 1)
    return out_dir


# ---------------------------------------------------------------------------
# Parameters: rows -> conf free-parameter lines + the surrogate set
# ---------------------------------------------------------------------------

def _free_parameters(parameter_rows):
    """Map estimated parameter rows to conf free-parameter lines + the surrogate set.

    Returns ``(free_param_lines, surrogate_params)``: ``free_param_lines`` are the conf
    declarations (**bare ids**, in table order -- new-era binds a free parameter to its
    model parameter by id, ADR-0034, so the declaration *is* ``<id>``, not ``<id>__FREE``);
    ``surrogate_params`` is the set ``M`` of fit-and-perturbed model parameters (a
    ``<p>__REF`` parameterId recovered to ``p`` by :func:`_model_param`). A truncated
    prior (two-sided or half-bounded, ADR-0020/0047) is emitted as a new-era
    ``parameter:`` record -- the only grammar carrying ``lower``/``upper`` -- via
    :func:`_free_parameter_conf_line`; ``free_parameter_from_row`` still surfaces the
    remaining boundary (the five unmapped PEtab families) as ``NotImplementedError``.
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
        free_param_lines.append(_free_parameter_conf_line(fp, model_param))
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


_SCALE_PREFIX = {'linear': '', 'log10': 'log', 'ln': 'ln'}


def _free_parameter_conf_line(fp, model_param):
    """One conf line for an imported free parameter.

    An untruncated prior keeps the compact legacy ``<type> = <name> p1 [p2]`` form (a
    one-parameter family carries only ``p1``). A *truncated* prior -- two-sided or
    half-bounded (ADR-0020/0047) -- is emitted as a new-era ``parameter:`` record, the
    only grammar that carries ``lower``/``upper`` bounds (#417/ADR-0043); an open side
    is written as an explicit infinity. The family's stem and scale are recovered from
    the prior registry so the record round-trips to the same ``FreeParameter``."""
    if fp.trunc_lb is None and fp.trunc_ub is None:
        nums = num(fp.p1) if fp.p2 is None else f'{num(fp.p1)} {num(fp.p2)}'
        return f'{fp.type} = {model_param} {nums}'
    fam, scale = PRIOR_KEYWORD_MAP[fp.type]
    stem = fp.type[len(_SCALE_PREFIX[scale.name]):-len('_var')]
    parts = [f'parameter: {model_param}', f'prior: {stem}']
    if scale.name != 'linear':
        parts.append(f'parameter_scale: {scale.name}')
    values = [fp.p1] if fp.p2 is None else [fp.p1, fp.p2]
    parts += [f'{fname}: {num(val)}' for fname, val in zip(fam.field_names, values)]
    parts += [f'lower: {num(fp.trunc_lb)}', f'upper: {num(fp.trunc_ub)}']
    return ', '.join(parts)


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


def _observable_id_to_column(observable_rows, namespace, entity_names, fixed_params,
                             obs_params, free_names, row_varying_obs_params=()):
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
      A fixed PEtab parameter the formula references but the model file lacks (Boehm's
      ``specC17``) is inlined as its numeric value first (``fixed_params``, ADR-0037); the
      remaining free symbols are then validated against the model namespace (the optional
      ``pybnf[petab]`` extra), and the ``.exp`` column is named after the ``observableId`` (the
      column the layer materializes). The model file is **not** edited.

    A measurement model's id must not shadow an existing model entity (``PybnfError``, so the
    materialized column does not collide with a model output column); an unknown free symbol
    raises in the validator (``PybnfError``).

    ``row_varying_obs_params`` (ADR-0045) is the set of observable_ids whose
    ``observableParameters`` scale/offset **differs** across rows: for those the placeholder is
    **kept** in the observableFormula (not substituted, not raised) and emitted as a measurement
    model whose per-row token is bound per data point from the sidecar (a
    :class:`~pybnf.measurement.PerMeasurementModel`, built in ``config.py``). The kept
    placeholder is admitted to the validator's allowed set so the non-placeholder symbols still
    validate against the model namespace. A *constant*-per-observable placeholder is substituted
    away as in Phase 1; an unresolved (neither constant nor row-varying) placeholder still raises.
    """
    # Fixed PEtab parameters that are NOT model entities are inlined as literals; one that
    # IS a model entity stays a symbol (it resolves as a model constant at eval time).
    inline = {n: v for n, v in fixed_params.items() if n not in namespace}
    # A measurement-model formula may reference a model entity OR a declared free parameter
    # (an observableParameters nuisance resolves from the PSet -- ADR-0044); validate against
    # both. An inlined fixed constant is a literal before validation (never a free symbol), so
    # it needs no place here.
    allowed = namespace | free_names
    taken = set(entity_names)
    mapping = {}
    measurement_models = []
    for row in observable_rows:
        raw = (row.observable_formula or '').strip()
        had_placeholder = bool(_PLACEHOLDER.search(raw))
        row_varying = row.observable_id in row_varying_obs_params
        if row_varying:
            # ADR-0045: a row-varying observableParameters scale is bound per data point from
            # the sidecar; KEEP the placeholder in the observableFormula verbatim (a
            # PerMeasurementModel resolves it). Never reduces to a bare name (it has a
            # placeholder), so it falls to the measurement-model branch below.
            formula = raw
        else:
            # Substitute a constant-per-observable observableParameters scale/offset (ADR-0044);
            # an empty substitution returns the formula verbatim (the bare/expression common
            # case stays dependency-free). A placeholder that survives substitution is the
            # deferred frontier.
            subs = _placeholder_subs(row.observable_id, obs_params)
            if subs:
                from .formula import substitute_placeholders
                formula = substitute_placeholders(raw, subs)
            else:
                formula = raw
            if had_placeholder:
                _require_no_placeholder(formula, row.observable_id)
        # A bare model-entity name (and no placeholder was substituted) is the dependency-free
        # common case; a substituted formula is always a measurement model (it references a
        # PSet nuisance) even if it reduced to a bare symbol.
        if not had_placeholder and _IDENTIFIER.match(formula):
            if formula not in namespace:
                raise PybnfError(
                    f"Observable '{row.observable_id}' has a bare observableFormula "
                    f"'{formula}', which is not a model entity. A bare-name observableFormula "
                    f"must name a model observable/function/species the backend outputs; an "
                    f"unknown name is an error (ADR-0036).",
                    f"Model namespace: {sorted(namespace)}.")
            mapping[row.observable_id] = formula          # bare-name path (no translator)
            continue
        # Expression (or substituted) -> a measurement model. Inline any fixed parameter-table
        # constant the model file lacks, then validate the remaining symbols against the
        # namespace u free parameters now (fail fast; requires the petab extra). The conf carries
        # the formula as a line and the observation layer evaluates it post-simulation -- no
        # model-file edit.
        from .formula import compile_petab_formula, inline_constants
        formula = inline_constants(formula, inline)
        # A kept row-varying placeholder (ADR-0045) is neither a model entity nor a free
        # parameter -- its value is bound per data point -- so admit it to the allowed set; the
        # non-placeholder symbols still validate against the namespace u free parameters.
        allowed_here = allowed | set(_PLACEHOLDER_SYMBOL.findall(formula))
        compile_petab_formula(
            formula, allowed_here,
            detail=f"Model namespace (species/parameters/observables/functions) u fit free "
                   f"parameters: {sorted(allowed)}.")
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
    ``sos`` from ``ave_norm_sos``; mirrors the export's column-mean sigma over all data).

    ``datas`` is ``{experiment_id: [Data, ...]}`` (the replicate grids per experiment), so
    the mean is taken over every replicate's column -- the same set of values the forward
    export's column-mean sigma averaged over."""
    def column_mean_of(observable_id):
        col = observable_id_to_column[observable_id]
        values = [data[col] for group in datas.values() for data in group
                  if col in data.cols]
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


def _objective_directives(observable_rows, observable_id_to_column, noise_param_ids,
                          column_mean_of, obs_params, row_varying_obs=()):
    """Recover the conf's objective directive lines from the observables' noise (ADR-0031/0037).

    The inverse of the objective-family / whole-fit / per-observable ``noise_model`` export.
    Returns a **list** of conf lines:

    * **Uniform** (one family + one sigma source across all observables) -- a single line, the
      tidy common case (:func:`_try_uniform_directive`): one of the four sugar tokens
      (``objective = chi_sq`` / ``sos`` / ``sod`` / ``ave_norm_sos``, round-trips
      byte-for-byte), or the ADR-0031 whole-fit ``noise_model = <family>, <param> = <verb>
      <arg>`` line (a uniform non-unit ``fix_at`` constant, or a single shared ``fit`` sigma).
    * **Per-observable** -- a structural base objective plus one ``noise_model <obs> = ...``
      override per observable (:func:`_per_observable_directives`). This is the Boehm shape:
      each observable carries its own estimated Gaussian sigma (its constant-per-observable
      ``noiseParameters`` parameter id), so no single whole-fit line names them (ADR-0021/0037).

    ``noise_param_ids`` is the ``{observable_id: parameter_id}`` map from the measurements'
    constant-per-observable ``noiseParameters`` placeholder; an observable's declared noise
    placeholder (a ``noiseParameter*`` token or a named ``noisePlaceholders`` id) takes its
    sigma source from this map. A ``log-normal`` / ``log-laplace`` distribution, an expression
    ``noiseFormula``, and a per-point laplace placeholder raise ``NotImplementedError`` -- the
    boundary is in code, not a silent mis-recovery.
    """
    per_obs = [(row, *_resolve_noise(
                    row, noise_param_ids.get(row.observable_id),
                    _placeholder_subs(row.observable_id, obs_params, noise_param_ids),
                    row.observable_id in row_varying_obs))
               for row in observable_rows]
    single = _try_uniform_directive(per_obs, column_mean_of)
    if single is not None:
        return [single]
    return _per_observable_directives(per_obs, observable_id_to_column)


def _resolve_noise(row, noise_param_id, obs_subs, row_varying=False):
    """One observables row -> ``(family_token, source)`` where ``source`` is one of
    ``('placeholder', None)`` (per-point ``_SD``), ``('constant', value)`` (a fixed sigma),
    ``('free', parameter_id)`` (an estimated sigma), ``('formula', expr)`` (an expression
    sigma -> ``FormulaSigma``, ADR-0044), or ``('per_measurement', expr)`` (a row-varying
    placeholder bound per data point -> ``PerMeasurementFormulaSigma``, ADR-0045).

    ``row_varying`` (ADR-0045): the observable's ``noiseParameters`` id **differs** across its
    measurement rows, so it cannot reduce to one substituted symbol -- the noiseFormula is
    emitted with its placeholder **kept** (``('per_measurement', formula)``) and the per-row
    token is bound from the experiment's sidecar binding table at eval time. Checked first,
    before the constant-reduction paths below.

    A **declared placeholder** noiseFormula -- a ``noiseParameter*`` token or a bare id listed
    in the row's ``noisePlaceholders`` -- has its value supplied per measurement: a parameter
    id constant across the observable (``noise_param_id``) is an estimated sigma
    (``('free', id)``, Boehm); otherwise it is the per-point ``_SD`` source
    (``('placeholder', None)``, chi_sq). Otherwise (ADR-0044) a constant-per-observable
    ``observableParameter*`` / ``noiseParameter*`` placeholder is substituted in (``obs_subs``)
    and the resulting noiseFormula classified: a number -> ``('constant', v)``, a bare id ->
    ``('free', id)``, an arithmetic expression -> ``('formula', expr)``. A placeholder that
    survives substitution (unresolved, and not row-varying) raises the deferred frontier."""
    dist = (row.noise_distribution or 'normal').lower()
    if dist not in _PETAB_DISTRIBUTION_TO_NOISE_MODEL:
        raise NotImplementedError(
            f"Observable '{row.observable_id}': noiseDistribution {dist!r} maps to a "
            f"PyBNF noise family on the natural-log scale (log-normal / log-laplace; "
            f"neg_bin was removed from PEtab v2), which has neither an objective token "
            f"nor a native noise_model line yet (#407). This chunk recovers the linear "
            f"normal / laplace families.")
    formula = (row.noise_formula or '').strip()
    if not formula:
        raise PybnfError(f"Observable '{row.observable_id}' is missing a noiseFormula.")
    # ADR-0045: a row-varying noiseParameters id is bound per data point from the sidecar; keep
    # the placeholder in the noiseFormula verbatim (PerMeasurementFormulaSigma resolves it).
    if row_varying:
        return dist, ('per_measurement', formula)
    placeholders = {p.strip() for p in (row.noise_placeholders or '').split(';') if p.strip()}
    # ADR-0037 declared-placeholder path FIRST (preserved byte-for-byte: Boehm).
    if formula.startswith('noiseParameter') or formula in placeholders:
        if noise_param_id is not None:
            return dist, ('free', noise_param_id)     # constant-per-observable estimated sigma
        return dist, ('placeholder', None)            # per-point _SD (chi_sq)
    # ADR-0044: substitute a constant-per-observable placeholder, then classify. A bare number
    # / id with no placeholder passes through untouched (substitution stays dependency-free).
    if _PLACEHOLDER.search(formula):
        from .formula import substitute_placeholders
        formula = substitute_placeholders(formula, obs_subs)
        _require_no_placeholder(formula, row.observable_id)
    try:
        return dist, ('constant', float(formula))     # a number -> ConstantSigma
    except ValueError:
        pass
    if _IDENTIFIER.match(formula):
        return dist, ('free', formula)                # a bare id -> FreeParameterSigma
    # An arithmetic expression -> FormulaSigma; validate it parses (+ the petab extra present).
    from .formula import formula_free_symbols
    formula_free_symbols(formula)
    return dist, ('formula', formula)


def _try_uniform_directive(per_obs, column_mean_of):
    """A single whole-fit directive line if the table is one PyBNF objective, else ``None``.

    ``None`` signals a genuinely per-observable table (a mix of families or sigma sources, or
    a distinct ``fit``/``fix_at`` sigma per observable) -> :func:`_per_observable_directives`.
    The uniform cases are exactly the objective-family / whole-fit ``noise_model`` export
    inverse (preserved byte-for-byte)."""
    families = {dist for _row, dist, _src in per_obs}
    kinds = {src[0] for _row, _dist, src in per_obs}
    if len(families) != 1 or len(kinds) != 1:
        return None     # mixed family or source kind -> per-observable
    family = families.pop()
    kind = kinds.pop()
    petab_family, param = _PETAB_DISTRIBUTION_TO_NOISE_MODEL[family]

    if kind == 'per_measurement':
        # A row-varying placeholder sigma (ADR-0045) is inherently per-observable -- its
        # noiseFormula carries the observable-specific placeholder noiseParameter1_<id>, so
        # several observables are never "uniform" -- and it is bound from the experiment's
        # sidecar binding table. Always a per-observable noise_model line.
        return None
    if kind == 'formula':
        # An expression sigma (FormulaSigma, ADR-0044) has no whole-fit objective *token*, but a
        # uniform one (every observable the same expression) is a whole-fit noise_model line --
        # the inverse of the export's whole-fit formula sigma, so it round-trips byte-for-byte.
        # A non-uniform / per-observable formula falls to _per_observable_directives (not yet
        # re-exportable -- the deferred per-observable export boundary, ADR-0045).
        exprs = {src[1] for _row, _dist, src in per_obs}
        if len(exprs) != 1:
            return None
        return f'noise_model = {petab_family}, {param} = formula {exprs.pop()}'
    if kind == 'placeholder':
        if family != 'normal':
            raise NotImplementedError(
                f"A per-point ({family}) placeholder noiseFormula has no PyBNF objective "
                f"token (only the Gaussian per-point _SD case, chi_sq, is recovered; #407).")
        return 'objective = chi_sq'
    if kind == 'free':
        ids = {src[1] for _row, _dist, src in per_obs}
        if len(ids) != 1:
            return None     # distinct free sigma per observable -> per-observable
        return f'noise_model = {petab_family}, {param} = fit {ids.pop()}'
    # All-constant sigma: the sugar tokens (sos/sod unit, ave_norm_sos column-mean) or a
    # uniform fix_at; a different fixed sigma per observable is per-observable.
    constants = [src[1] for _row, _dist, src in per_obs]
    if all(c == 1.0 for c in constants):
        return 'objective = sos' if family == 'normal' else 'objective = sod'
    if family == 'normal' and all(
            _approx(c, column_mean_of(row.observable_id))
            for (row, _dist, _src), c in zip(per_obs, constants)):
        return 'objective = ave_norm_sos'
    uniq = set(constants)
    if len(uniq) != 1:
        return None     # distinct fixed sigma per observable -> per-observable
    return f'noise_model = {petab_family}, {param} = fix_at {num(uniq.pop())}'


def _per_observable_directives(per_obs, observable_id_to_column):
    """A structural base objective + one ``noise_model <obs> = ...`` override per observable.

    The Boehm shape (ADR-0037): each observable has its own sigma source, so PyBNF expresses
    it as per-observable ``noise_model`` overrides (ADR-0021) layered over a whole-fit default.
    Under edition >= 2 a base objective is required (the override surface "accompanies" it,
    config.py), and since every observable is overridden the base is a structural placeholder
    -- ``objective = chi_sq`` (Gaussian, no free parameter, no data column required). Each
    override names the **column** the objective compares (the measurement-model column =
    ``observableId`` for an expression observable, else the model entity); a ``fit`` sigma binds
    its estimated parameter as a nuisance (ADR-0034), a ``fix_at`` a constant, a per-point
    placeholder reads the ``<col>_SD`` companion."""
    lines = ['objective = chi_sq']   # whole-fit default; every observable overridden below
    for row, dist, src in per_obs:
        petab_family, param = _PETAB_DISTRIBUTION_TO_NOISE_MODEL[dist]
        column = observable_id_to_column[row.observable_id]
        kind = src[0]
        if kind == 'free':
            lines.append(f'noise_model {column} = {petab_family}, {param} = fit {src[1]}')
        elif kind in ('formula', 'per_measurement'):
            # Both emit a 'formula' source; for 'per_measurement' the expression keeps its
            # row-varying placeholder, which config.py routes to PerMeasurementFormulaSigma
            # (the placeholder's row token comes from the measurement_params sidecar; ADR-0045).
            lines.append(f'noise_model {column} = {petab_family}, {param} = formula {src[1]}')
        elif kind == 'constant':
            lines.append(
                f'noise_model {column} = {petab_family}, {param} = fix_at {num(src[1])}')
        elif kind == 'placeholder':
            if dist != 'normal':
                raise NotImplementedError(
                    f"Observable '{row.observable_id}': a per-point ({dist}) placeholder "
                    f"noiseFormula has no native noise_model source yet (#407).")
            lines.append(f'noise_model {column} = {petab_family}, {param} = read_exp_file _SD')
        else:  # defensive
            raise PybnfError(
                f"Observable '{row.observable_id}': unexpected sigma source kind {kind!r}.")
    return lines


def _approx(a, b):
    """Two sigmas are equal up to a relative tolerance (the column-mean comparison)."""
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Experiments: measurement groups + experiment rows -> (name, condition, data files)
# ---------------------------------------------------------------------------

def _experiments(datas, experiment_rows, out_dir, model_location_of, param_bindings=None):
    """Assemble the conf's experiments and write each one's ``.exp`` file(s).

    The set of experiments is the measurement groups (the replicate grids per
    ``(experimentId, modelId)``, ADR-0041); each experiment's replicate ``Data`` objects are
    written to ``<name>.exp`` (the first / only replicate) and ``<name>_rep<k>.exp`` (k>=2),
    all bound to the one experiment's ``data:`` list -- the inverse of the forward export,
    which stacks an experiment's replicate ``Data`` objects into repeated measurement rows
    (ADR-0039). The single-replicate case keeps the bare ``<name>.exp`` name, so the common
    round trip is byte-stable. The experiment's condition comes from the experiments table
    (``cond_<c>`` -> ``c``; the synthesized ``cond_wildtype`` and an absent row -> no
    condition). A ``''`` experimentId is the "model as is" base time course (PEtab erased
    its name because the job had no fit-and-perturbed parameters); it is synthesized a name,
    which never reaches the PEtab output (it re-exports to ``''`` again) -- a name keyed on the
    modelId when set, so two wildtype experiments on different models stay distinct. Each
    experiment's model is the ``modelId`` on its rows: ``model_location_of`` maps it to the
    model file, emitted as a per-experiment ``model:`` field (omitted for a single-model job,
    whose modelId is ``''``).

    ``param_bindings`` (ADR-0045) is the ``{(experiment_id, model_id): {column: {placeholder:
    {time: token}}}}`` per-measurement binding table; a group with an entry also writes a
    ``<name>_measparams.tsv`` sidecar carrying its row-varying noise tokens, emitted as the
    experiment's ``measurement_params:`` field. Returns ``[(name, condition_or_None,
    [data_file, ...], model_location_or_None, measparams_file_or_None, t_end_or_None), ...]`` in
    measurement order; ``t_end`` is ``None`` for a time course (the dose-response scans append
    their own entries -- :func:`_dose_response_experiments`).
    """
    param_bindings = param_bindings or {}
    condition_of = {row.experiment_id: row.condition_id for row in experiment_rows}
    experiments = []
    for (eid, mid), group in datas.items():
        if eid:
            name = eid
        elif mid:
            name = f'experiment_{mid}'
        else:
            name = 'experiment1'
        condition = condition_name_from_id(condition_of.get(eid))
        model_location = model_location_of.get(mid)   # None for a single-model job (mid '')
        data_files = []
        for k, data in enumerate(group):
            data_file = f'{name}.exp' if k == 0 else f'{name}_rep{k + 1}.exp'
            _write_exp(out_dir / data_file, data)
            data_files.append(data_file)
        measparams_file = None
        binding = param_bindings.get((eid, mid))
        if binding:
            measparams_file = f'{name}_measparams.tsv'
            write_measurement_params(binding, out_dir / measparams_file)
        experiments.append((name, condition, data_files, model_location, measparams_file, None))
    return experiments


def _dose_response_experiments(dose_responses, out_dir, model_location_of):
    """Build the conf experiment entries for the reconstructed dose-response scans (ADR-0046).

    Each scan's swept-axis :class:`~pybnf.data.Data` is written to ``<name>.exp`` (column 0 the
    swept parameter, so ``config._infer_experiment_type`` reads it as a parameter_scan -- no
    ``type:`` field needed). A steady-state scan (``scan_time`` inf) carries ``t_end = None`` (it
    runs to steady state, PEtab time=inf); a finite scan carries that endpoint. Returns the same
    ``(name, condition, [data_file], model_location, measparams_file, t_end)`` tuple shape as
    :func:`_experiments` (condition / measparams are always ``None`` -- a dose is the scan axis,
    not a named condition, and a dose-response carries no per-measurement sidecar)."""
    experiments = []
    for dr in dose_responses:
        name = dr['name']
        data_file = f'{name}.exp'
        _write_exp(out_dir / data_file, dr['data'])
        model_location = model_location_of.get(dr['model_id'])
        t_end = None if math.isinf(dr['scan_time']) else dr['scan_time']
        experiments.append((name, None, [data_file], model_location, None, t_end))
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

def _write_conf(path, *, model_filenames, job_type, objective_directives, free_param_lines,
                conditions, experiments, measurement_models, method, method_overrides,
                settings, multi):
    """Write one new-era (edition 2) ``.conf``: the recovered problem + the supplied
    run-recipe (``job_type``, per-experiment ``method:``, required settings).

    ``model_filenames`` is the list of the problem's model files (one ``model:`` line each;
    a multi-model job also tags every experiment with its ``model:`` field -- ADR-0041).
    ``objective_directives`` is the list of recovered objective lines -- either a single
    ``objective = <token>`` / whole-fit ``noise_model = ...`` line, or a base objective plus
    per-observable ``noise_model <obs> = ...`` overrides (:func:`_objective_directives`).
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
        *[f'model: {mf}' for mf in model_filenames],
        f'job_type = {job_type}',
        *objective_directives,
    ]
    # Expression observables: a measurement-model formula evaluated post-simulation (the
    # observation layer, ADR-0036), not a model-file edit. The model is carried verbatim.
    for obs_id, formula in measurement_models:
        lines.append(f'observable: {obs_id}, formula: {formula}')
    lines.append('')
    for name, perts in conditions.items():
        pert_str = ', '.join(f'{var} {op} {num(val)}' for var, op, val in perts)
        lines.append(f'condition: {name}, perturbations: {pert_str}')
    for name, condition, data_files, model_location, measparams_file, t_end in experiments:
        sim_method = method_overrides.get(name, method)
        cond_field = f', condition: {condition}' if condition else ''
        model_field = f', model: {model_location}' if model_location else ''
        # A fixed-endpoint dose-response scan's endpoint time (ADR-0046); a steady-state scan
        # and a time course carry none (the scan runs to steady state / the data drives the grid).
        tend_field = f', t_end: {num(t_end)}' if t_end is not None else ''
        # The row-varying per-measurement binding sidecar (ADR-0045), when this experiment
        # carries one; config.py attaches it to the experiment's exp Data.
        mp_field = f', measurement_params: {measparams_file}' if measparams_file else ''
        data_field = ', '.join(f'data: {f}' if i == 0 else f
                               for i, f in enumerate(data_files))
        lines.append(
            f'experiment: {name}{cond_field}{model_field}, method: {sim_method}'
            f'{tend_field}{mp_field}, {data_field}')
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
    ``measurement_files`` / ``condition_files`` / ``experiment_files``) and a ``models`` list
    -- one ``{model_id, location, language}`` entry per ``model_files`` entry, in declaration
    order (one or many, ADR-0041). For single-model convenience the first model is also
    surfaced as ``model_file`` / ``model_id`` / ``model_language``. Dependency-free (no YAML
    library): the writer emits a flat ``key:`` + ``  - item`` list shape and a two-level
    ``model_files`` block, which a small indentation-aware scan reads exactly. The scan is
    **order-independent**, so a real v2 ``problem.yaml`` that lists ``model_files`` first
    (our writer emits it last) reads identically.

    This is a pure *reader*: it records each model's ``language`` but does not enforce a
    policy on it. The supported-language scope (BNGL or SBML, ADR-0036) is enforced by the
    importer (:func:`_require_supported_model`), not here.
    """
    file_keys = ('parameter_files', 'observable_files', 'measurement_files',
                 'condition_files', 'experiment_files')
    files = {k: [] for k in file_keys}
    models = []         # [{model_id, location, language}, ...] in declaration order
    current = None      # the model entry being filled (set by a `<modelId>:` line)

    section = None      # the current top-level *_files key (list items follow)
    in_model = False    # inside the model_files: block
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if indent == 0:
            section, in_model, current = None, False, None
            if stripped.endswith(':') and stripped[:-1] in files:
                section = stripped[:-1]
            elif stripped == 'model_files:':
                in_model = True
            # format_version and any other scalar top-level key: ignored
            continue
        if section is not None and stripped.startswith('-'):
            files[section].append(stripped[1:].strip())
        elif in_model:
            if stripped.startswith('location:') and current is not None:
                current['location'] = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('language:') and current is not None:
                current['language'] = stripped.split(':', 1)[1].strip()
            elif stripped.endswith(':'):
                # A new `<modelId>:` block (the location/language lines follow, indented).
                current = {'model_id': stripped[:-1].strip(), 'location': None,
                           'language': None}
                models.append(current)

    _require_problem(files, models, path)
    first = models[0]
    return {**files, 'models': models, 'model_file': first['location'],
            'model_id': first['model_id'], 'model_language': first['language']}


def _require_problem(files, models, path):
    for key in ('parameter_files', 'observable_files', 'measurement_files'):
        if not files[key]:
            raise PybnfError(f"problem.yaml at {path} has no {key}.")
    if not models or any(m['location'] is None for m in models):
        raise PybnfError(f"problem.yaml at {path} declares no model file.")


def _require_supported_model(problem, path):
    """Enforce the importer's supported-language scope on a parsed ``problem.yaml`` (ADR-0036).

    The reader (:func:`read_problem_yaml`) records each model's ``language`` without judging
    it; the importer holds the policy. **BNGL and SBML both import** (one or many models,
    ADR-0041): each model file is carried verbatim and an expression ``observableFormula``
    becomes a post-simulation measurement model (the observation layer), so neither a
    ``.bngl`` nor an ``.xml`` is ever edited (ADR-0036). Any other model language (e.g.
    ``pysb``) raises ``NotImplementedError`` early, before any table is read. A ``None``
    language (the field was absent) is permitted -- the exporter omits it only for a BNGL
    model.
    """
    for m in problem['models']:
        language = m['language']
        if language is not None and language.lower() not in ('bngl', 'sbml'):
            raise NotImplementedError(
                f"problem.yaml model '{m['model_id']}' has language '{language}', not "
                f"'bngl' or 'sbml' (at {path}). Only BNGL and SBML PEtab problems are "
                f"importable (ADR-0036); other model languages are out of scope (#407).")
