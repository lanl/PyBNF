"""PEtab v2 problem importer: a BNGL-native PEtab v2 problem -> a new-era PyBNF job
(issue #407; the importer read path, ADR-0025 reversed / ADR-0032).

The inverse of :func:`pybnf.petab.export.export_job`. Given a ``problem.yaml`` + its TSV
tables + a BNGL model, :func:`import_job` writes a runnable new-era (edition 2) ``.conf``
plus the ``.exp`` data files and a copy of the model that is verbatim except for marked
``estimate = false`` overrides (new-era binds free parameters by id, ADR-0034, so the model
needs no re-instrumentation; see *Fixed parameters* below) -- the form the exporter reads.
It closes the "two-adapter proof" at the read level for BNGL-native problems: the reverse
asset mappers (parameters/observables/measurements/conditions) run
backwards onto the shared neutral rows, and this module is the *disposable orchestrator*
that ties them together (problem.yaml reader + ``.conf``/``.exp`` writers).

**PEtab is a problem spec; PyBNF is a job spec (the run-recipe gap).** A PEtab problem
fixes the objective landscape (model, data, conditions, parameters + priors, the noise
model) but deliberately says nothing about *how to search it* -- no optimizer/sampler, no
algorithm settings, no simulation method, no seed -- because PEtab is a cross-tool
exchange format and the *method* belongs to the tool. So ``import = PEtab problem +
a supplied run-recipe``. The *problem* half is recovered exactly (and round-trips
byte-for-byte through a re-export; a table the problem splits over several files is read in
full and re-exports as one file, #902); the *recipe* half -- ``job_type`` + that fit's
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
bare-name common case runs in the bngsim-less CI tier. ``problem.yaml`` is hand-parsed: block
and one-line flow lists, every file a key lists, and a refusal for any shape it cannot read
(#902). The ``petab`` library is the test-only oracle for the
bare-name path, and the optional ``pybnf[petab]`` extra for an expression ``observableFormula``.

**Scope (read path: BNGL and SBML, one or many models).** Both model languages import
(ADR-0036): the model file is carried **verbatim** for each (except for marked
``estimate = false`` overrides, below), and an expression
``observableFormula`` becomes a first-class *measurement model* -- a PEtab math expression
evaluated as a post-simulation transform over the output trajectory (the observation layer),
emitted as an ``observable: <id>, formula: <expr>`` conf line -- **never** by editing the model
file (the ``begin functions`` synthesis of ADR-0035 is superseded). The bare-name common case
still needs no translator and stays dependency-free. SBML observables are 100% expressions, so
SBML import pulls in the ``pybnf[petab]`` extra. A **multi-model** problem imports too (ADR-0041):
each ``model_files`` entry is carried verbatim (save the marked overrides below) and declared
with its own ``model:`` line, an expression observableFormula validates against the union of
every model's namespace, and each experiment's model is recovered from the ``modelId`` on its
measurement rows (emitted as a per-experiment ``model:`` field; a BNGL + SBML mix is fine). A
**constant-per-observable** ``observableParameters`` scale/offset and an expression
``noiseFormula`` import too (ADR-0044):
the placeholder is substituted into the observable/noise formula (an id resolves from the PSet,
a number inlines), an expression ``noiseFormula`` becoming a ``FormulaSigma`` (``noise_model
<obs> = <family>, <param> = formula <expr>``). A **dose-response** (parameter_scan) problem
imports too (ADR-0046): N Conditions each setting one swept parameter at a constant measurement
time (``inf`` => steady state, or a finite ``t_end``) are reconstructed into a single swept-axis
``.exp`` (column 0 the swept parameter) + a ``parameter_scan`` ``experiment:`` (the inverse of
the exporter's dose-response emission -- ``reconstruct_dose_responses``). A ``time = inf``
measurement with **no** swept axis (a single condition, e.g. Blasi) needs no reconstruction at
all: the time is written verbatim into the ``.exp``, and the fitter reads an all-``inf`` time
column as a steady-state experiment -- a relaxation to equilibrium (ADR-0086, #521). Out of scope, each
mirroring an export-side boundary: a condition-table sympy layer; the five PEtab prior families
PyBNF lacks; a dose-response that also carries a named condition or row-varying per-measurement
placeholders. (One-sided truncation now maps to a half-bounded box -- ADR-0047, #432.)

**Fixed parameters (#907, ADR-0149).** A parameters-table row with ``estimate = false`` fixes
its parameter at the row's ``nominalValue``, and PEtab gives that value precedence over the
model file; libpetab, AMICI and pyPESTO all simulate with it. When the row names a model
parameter (a BNGL ``begin parameters`` entry, or an SBML global ``<parameter>``), the importer
therefore writes the nominalValue into its copy of every model that declares the parameter --
only where the file disagrees: a numeric value already equal to it is left byte-identical,
while a different number, a BNGL expression, or a missing SBML value is replaced. Each edit is
marked with a comment in the copy, listed in the conf header, and printed; the source files are
never touched, and nothing about it is a warning, because it is what the problem says. A fixed
row that names no model entity keeps the inlining path (a formula constant, a fixed sigma, a
condition targetValue). Refused: a fixed row with no nominalValue, a fixed row naming a model
entity other than a parameter or an SBML rule target (libpetab's lint rejects both), an SBML
parameter an initial assignment, event assignment or algebraic rule also sets, and a BNGL
parameter the model's own actions set (either way the written value would not be its value for
the whole simulation). The exporter never writes an ``estimate =
false`` row: a parameter it does not fit stays in the exported model at the model file's value,
which is what PEtab uses for a parameter absent from the table, so a re-export of an imported
job carries the edited model and no fixed row -- the same problem.
"""

import math
import re
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

from ..data import Data, observed_mean
from ..printing import PybnfError, print0
from ..priors import PRIOR_KEYWORD_MAP
from ..pset import BNGLModel, ModelError
from .conditions import (
    REF_MARKER,
    condition_name_from_id,
    condition_names_and_ids,
    conditions_from_rows,
    drop_synthesized_wildtype,
    equil_t_end_from_period_time,
    free_condition_name,
    is_species_target,
    model_time_reads,
    name_unperturbed_equilibrations,
    read_condition_table,
    read_experiment_table,
    read_mapping_table,
    refuse_inexact_unperturbed_periods,
    refuse_measurements_inside_fixed_equilibration,
)
from .measurements import (
    data_from_measurement_rows,
    measurement_param_bindings,
    noise_parameter_ids_by_observable,
    noise_parameters_by_observable,
    observable_parameters_by_observable,
    read_measurement_table,
    reconstruct_dose_responses,
    reconstruct_preequilibrated_dose_responses,
    row_varying_noise_ids,
    row_varying_noise_param_ids,
    row_varying_observable_ids,
)
from ._measurement_params import write_measurement_params
from .observables import read_observable_table
from .parameters import free_parameter_from_row, read_parameter_table
from ._bngl import parameters_set_by_actions
from ._bngl import parse_model as parse_bngl_model
from ._bngl import set_parameter_values as set_bngl_parameter_values
from ._sbml import parse_model as parse_sbml_model
from ._sbml import set_parameter_values as set_sbml_parameter_values
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

# A noiseFormula that is EXACTLY a single ``noiseParameter${n}_<id>`` placeholder (Boehm /
# Oliveira / per-point _SD -- the ADR-0037 declared-placeholder shape). A multi-parameter
# noiseFormula (``noiseParameter1_X + noiseParameter2_X * y``) is NOT a bare placeholder, so it
# falls through to the substitute-and-classify path (ADR-0075) instead of the _SD/estimated fast
# path -- the tightening of ADR-0037's ``startswith('noiseParameter')`` test.
_BARE_NOISE_PLACEHOLDER = re.compile(r'noiseParameter\d+_\w+\Z')


def _placeholder_subs(observable_id, obs_params, noise_params=None, fixed_params=None):
    """The ``{placeholder_name: token}`` substitution for one observable (ADR-0044/0075).

    Binds ``observableParameter${n}_${observable_id}`` to the n-th constant-per-observable
    ``observableParameters`` token (``obs_params[observable_id]``), and -- when
    ``noise_params`` is given (the noiseFormula side) -- ``noiseParameter${n}_${observable_id}``
    to the n-th constant-per-observable ``noiseParameters`` token
    (``noise_params[observable_id]``, a tuple: one entry for Boehm's single id, several for a
    multi-parameter noiseFormula like Raia's affine ``σ_abs;σ_rel`` -- ADR-0075). A token that
    names a **fixed** PEtab parameter (``fixed_params``, estimate=0) is inlined as its numeric
    value (so Oliveira's ``noiseParameter1_X -> sd_X = 1`` reduces to a constant sigma, not a
    fit); an estimated-parameter or numeric token stays as-is (an id remains a free symbol).
    An empty map means nothing to substitute (the bare-name / no-placeholder common case stays
    dependency-free)."""
    subs = {f'observableParameter{n}_{observable_id}': tok
            for n, tok in enumerate(obs_params.get(observable_id, ()), start=1)}
    if noise_params is not None:
        for n, tok in enumerate(noise_params.get(observable_id, ()), start=1):
            subs[f'noiseParameter{n}_{observable_id}'] = tok
    if fixed_params:
        subs = {ph: (num(fixed_params[tok]) if tok in fixed_params else tok)
                for ph, tok in subs.items()}
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
    new-era PyBNF job: the ``.exp`` data files, a copy of each model (new-era binds free
    parameters by id, so the model needs no re-instrumentation -- ADR-0034), and one or more
    ``.conf`` files. The *problem* (parameters/priors, observables/noise, measurements,
    conditions/experiments) is recovered exactly, from every file each ``problem.yaml`` key
    lists (#902); the *run-recipe* (``job_type``, ``method``, ``settings``) is supplied by the
    caller (see the module docstring). Returns the ``out_dir`` path.

    Each model copy is verbatim except for marked ``estimate = false`` overrides (#907,
    ADR-0149). A parameters-table row with ``estimate = false`` that names a model parameter
    fixes it at the row's ``nominalValue``, which PEtab gives precedence over the model file.
    Where the model file has a different value (or, in BNGL, an expression; in SBML, no
    value), the copy is edited to the table's value and the edit is marked with a comment in
    the copy, listed in the conf header, and printed, one line per override. A model that
    already agrees is copied byte for byte, and the source files are never touched: an edited
    copy whose destination is its own source file raises ``PybnfError``.

    ``job_type`` is the SEARCH method token, or ``'all'`` to emit one
    ``imported_<jt>.conf`` per registered optimizer + sampler. ``method`` (default
    ``'ode'``) is the per-experiment SIMULATION method; ``method_overrides`` (a
    ``{experiment_name: method}`` map) sets per-experiment values. ``settings`` overrides
    the required algorithm/run settings.

    Both **BNGL and SBML** models import (ADR-0036): the model file is carried verbatim
    (except for the overrides above), and an **expression** ``observableFormula`` (e.g. a
    quotient of sums) becomes a conf measurement model
    (``observable: <id>, formula: <expr>``) evaluated post-simulation -- the optional
    ``pybnf[petab]`` extra. A **constant-per-observable** ``observableParameters``
    scale/offset and an expression ``noiseFormula`` are substituted/reduced and import too
    (ADR-0044). A **dose-response** (parameter_scan) problem -- N conditions each setting one
    swept parameter at a constant measurement time (``inf`` => steady state, ADR-0046) -- is
    reconstructed into a single swept-axis ``.exp`` + a ``parameter_scan`` experiment; a
    **steady-state** measurement with no swept axis (``time = inf`` under a single condition)
    imports as an ordinary experiment whose ``.exp`` time is ``inf``, which the fitter reads as a
    relaxation to equilibrium (ADR-0086); a
    **pre-equilibrated dose-response** (ADR-0062, the preincubate -> wash -> dose-scan protocol) --
    N two-period experiments whose species ``setConcentration`` wash targets are aliased through
    the **mapping table** -- is reconstructed into a ``preequilibrate:`` + ``condition:``
    ``parameter_scan`` experiment (the species patterns recovered from the mapping). Raises
    ``NotImplementedError`` at the remaining PEtab/PyBNF boundaries (a model language other than
    ``bngl``/``sbml``; the five unsupported prior families; a log-normal/log-laplace noise
    distribution; a **multi-symbol** condition expression -- a single parameter-valued
    ``targetValue`` is a per-condition estimated initial condition and imports, ADR-0076; a
    **row-varying** per-measurement ``observableParameters``/``noiseParameters`` placeholder;
    a fixed SBML parameter an initial assignment, event assignment or algebraic rule also
    sets, or a fixed BNGL parameter the model's actions set) and ``PybnfError``
    for a malformed problem (an ``observableFormula`` symbol that is not a model entity, an
    ambiguous dose-response group, an id defined twice across a table's files, a
    ``problem.yaml`` shape the reader cannot read -- #902, a fixed row with no nominalValue or
    naming a model entity PEtab does not allow in the parameters table, or a BNGL model whose
    actions are more than its network definition -- #969).
    """
    problem_yaml_path = Path(problem_yaml_path)
    base = problem_yaml_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = read_problem_yaml(problem_yaml_path)
    _require_supported_model(problem, problem_yaml_path)
    models = problem['models']

    # Every file listed under each table key, concatenated in list order (#902). The condition,
    # experiment and mapping tables are optional (an empty list reads as no rows).
    (parameter_rows, observable_rows, measurement_rows, condition_rows, experiment_rows,
     mapping_rows) = _read_problem_tables(problem, base)
    # The species-amount mapping table (ADR-0062): a {petab_id: BNGL pattern} inversion of the
    # exporter's species setConcentration aliasing. Absent for a job with no species conditions.
    species_by_id = {r.petab_id: r.model_id for r in mapping_rows}

    # Parameters -> conf free-parameter lines (bare ids; new-era binds by id, ADR-0034)
    # + the surrogate set M of fit-and-perturbed model parameters.
    free_param_lines, surrogate_params = _free_parameters(parameter_rows)
    # A period read as the model as is (a blank -inf period, a pins-only condition -- #906,
    # ADR-0150) must be the model as is in PEtab too; refuse the ones that are not, on the tables
    # as written, before the base condition below is dropped.
    refuse_inexact_unperturbed_periods(
        condition_rows, experiment_rows, surrogate_params,
        {row.experiment_id for row in measurement_rows})
    # The exporter's synthesized base condition cond_wildtype re-pins M at base (p = p__REF),
    # read as the identity once p__REF is renamed back to p: drop it, and blank its periods, so
    # every reconstruction below reads "no condition" there. A cond_wildtype carrying any real
    # target is a condition in its own right and is kept (#905).
    condition_rows, experiment_rows = drop_synthesized_wildtype(
        condition_rows, experiment_rows, surrogate_params)

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
    # post-sim observation layer, never a model-file edit (ADR-0036) -- save the marked
    # estimate=false overrides applied just below (#907).
    model_texts = {}            # location -> verbatim text
    namespaces, entity_name_sets = [], []
    derived = {}                # SBML entity defined by others -> DerivedSymbol (inlined below, #493/#795)
    for m in models:
        loc, lang = m['location'], (m['language'] or 'bngl').lower()
        text = (base / loc).read_text(encoding='utf-8', errors='replace')
        if lang == 'bngl':
            _require_no_protocol_actions(text, loc)
        model_texts[loc] = text
        ns, ents, rules = _model_namespace(text, lang)
        namespaces.append(ns)
        entity_name_sets.append(ents)
        derived.update(rules)
    namespace = set().union(*namespaces)
    entity_names = set().union(*entity_name_sets)

    # A fixed (estimate=false) row naming a model parameter sets that parameter to its
    # nominalValue: PEtab gives the table precedence over the model file. The value is written
    # into the imported copy of every model that declares the parameter (only where the file
    # disagrees), each edit marked in the copy, listed in the conf header and printed (#907,
    # ADR-0149). The entity sets read above are unchanged by the edit.
    model_texts, fixed_overrides = _apply_fixed_model_parameters(
        parameter_rows, models, model_texts)
    _refuse_overwriting_an_edited_source(fixed_overrides, base, out_dir)

    # Observables -> the observableId -> model-column map (the data pivot's column order)
    # plus the measurement models (id, formula) synthesized from expression
    # observableFormulas (ADR-0036: emitted as conf `observable: ... formula:` lines).
    observable_id_to_column, measurement_models = _observable_id_to_column(
        observable_rows, namespace, entity_names, fixed_params, obs_params, free_names,
        row_varying_obs_params, derived)

    # A time=-inf period that applies no condition equilibrates the model as is, which is a
    # pre-equilibration, not the absence of one (#906, ADR-0150): point each such period at one
    # synthesized condition, declared below as `perturbations: none`, before any reader sees it.
    experiment_rows, unperturbed = name_unperturbed_equilibrations(condition_rows, experiment_rows)

    # Pre-equilibrated dose-response reconstruction (ADR-0062): pull out the two-period scan groups
    # (a -inf pre-equilibration period + a per-dose measurement period) FIRST, so the plain
    # dose-response and time-course reconstructions below never see them. Only the per-dose
    # conditions are consumed; the shared pre-equilibration + wash conditions stay in the condition
    # table (they become preequilibrate:/condition: lines). Only a model parameter can be a scan's
    # swept axis, so a condition setting a species amount is never read as a dose (#904).
    sweepable = set().union(*(_model_parameter_ids(model_texts[m['location']], m['language'])
                              for m in models))
    preequil_scans, meas_rows_1, pdr_condition_ids, pdr_experiment_ids = \
        reconstruct_preequilibrated_dose_responses(
            measurement_rows, condition_rows, experiment_rows, observable_id_to_column,
            sweepable=sweepable)

    # Dose-response (parameter_scan) reconstruction (ADR-0046): pull out the experiment groups
    # whose N conditions each set one swept parameter at a constant measurement time (inf =>
    # steady state) and rebuild each as a single swept-axis Data; the remaining rows are time
    # courses. Their conditions/experiments are dropped from the time-course reconstruction (a
    # dose is the scan axis, not a named condition: line).
    dose_responses, tc_rows, dr_condition_ids, dr_experiment_ids = reconstruct_dose_responses(
        meas_rows_1, condition_rows, experiment_rows, observable_id_to_column)

    # Time-course measurements -> the wide Data replicates per (experiment, model), then assemble
    # the experiment list (repeated (obs, time) rows are dealt into replicate grids -- ADR-0039;
    # the modelId distinguishes experiments that share an experimentId -- ADR-0041).
    datas = data_from_measurement_rows(tc_rows, observable_id_to_column)
    # The objective/noise are GLOBAL across the fit, so they read ALL measurement rows (a
    # dose-response observable shares the objective). A constant-per-observable parameter-id
    # noiseParameters placeholder is a per-observable estimated sigma (Boehm's sd_*); the map
    # drives the per-observable noise_model lines (ADR-0037). A row-varying id routes to the
    # per-measurement binding table (ADR-0045): the per-data-point sidecar carrying the row's
    # estimated noise id, emitted as a 'sigma = formula <placeholder>' line. A MULTI-token
    # noiseParameters cell (a multi-parameter noiseFormula -- Raia's affine, Fiedler's product)
    # is the ADR-0075 generalization: its constant tuple substitutes into the noiseFormula by
    # index (an id -> a free symbol, a fixed parameter -> its value), and a row-varying tuple
    # routes to the same sidecar.
    noise_param_ids = noise_parameter_ids_by_observable(measurement_rows)
    noise_param_tuples = noise_parameters_by_observable(measurement_rows)   # multi-token constant
    # The unified constant-noise substitution map: the single-id Boehm entries as 1-tuples plus
    # the multi-token tuples, so the n-th token binds noiseParameter${n} for any arity (ADR-0075).
    noise_subs = {oid: (nid,) for oid, nid in noise_param_ids.items()}
    noise_subs.update(noise_param_tuples)
    # Row-varying noise = the single-id frontier (ADR-0045) u the multi-token one (ADR-0075).
    row_varying_noise = row_varying_noise_ids(measurement_rows) | row_varying_noise_param_ids(measurement_rows)
    # One sidecar carries both row-varying frontiers (ADR-0045/0075): the row-varying noise
    # tokens (each noiseParameter${n}) and the row-varying observableParameters scale/offset
    # tokens, keyed by data column. Keyed to a time-course experiment's rows (a dose-response
    # carries no per-measurement sidecar).
    param_bindings = measurement_param_bindings(
        tc_rows, observable_id_to_column, row_varying_noise, row_varying_obs_params)
    # The column-mean check (#894) compares a sigma with each IMPORTED experiment's own mean --
    # the one the imported fit will use: every time-course (experimentId, modelId) group with
    # its replicates, and each dose-response scan (plain + pre-equilibrated) as ONE experiment
    # with its replicate grids (#903). A list, not a merged dict: a time course and a scan can
    # share a name key.
    column_means = _ColumnMeans(
        list(datas.values()) + [list(dr['datas']) for dr in dose_responses]
        + [list(s['datas']) for s in preequil_scans],
        observable_id_to_column)
    # A noiseFormula symbol is prediction-dependent (a simulated column) only if it is a model
    # entity that is NOT a declared free parameter: a fit parameter (even one that binds a model
    # parameter by id, ADR-0034) resolves from the PSet, not the trajectory. So the σ scales with
    # the simulation only when it names a model entity outside the free-parameter set (ADR-0075).
    prediction_entities = namespace - free_names
    objective_directives, sd_readers = _objective_directives(
        observable_rows, observable_id_to_column, noise_param_ids, column_means,
        obs_params, noise_subs, row_varying_noise, fixed_params, prediction_entities)
    # The pivot rebuilds an _SD companion for EVERY column of an experiment in which some row
    # carries a numeric noiseParameters. Keep it only for the observables whose recovered sigma
    # reads it (sd_readers): the fitter refuses a data column nothing reads. This matters for a
    # per-row column-mean sigma (#894), which comes back as column_mean and reads no data, and
    # for any observable sharing an experiment with one (its rebuilt companion is all NaN).
    sd_columns = ({col + '_SD' for oid, col in observable_id_to_column.items()
                   if oid not in sd_readers}
                  - set(observable_id_to_column.values()))
    if sd_columns:
        datas = {key: [_without_columns(d, sd_columns) for d in group]
                 for key, group in datas.items()}
        for scan in dose_responses + preequil_scans:
            scan['datas'] = [_without_columns(d, sd_columns) for d in scan['datas']]
    # Named conditions exclude those absorbed into a dose-response (each dose is the scan axis, not
    # a condition: line); a pre-equilibrated scan's per-dose conditions are absorbed too, but its
    # shared pre-equilibration + wash conditions REMAIN (they become preequilibrate:/condition:).
    # A species target's BNGL pattern is recovered from the mapping table (ADR-0062).
    absorbed_condition_ids = dr_condition_ids | pdr_condition_ids
    tc_condition_rows = [r for r in condition_rows
                         if r.condition_id not in absorbed_condition_ids]
    # ``free_names`` / ``fixed_params`` resolve a parameter-valued targetValue -- a per-condition
    # estimated initial condition (ADR-0076): a target set to an estimated parameter id becomes a
    # parameter-reference perturbation (bound from the PSet at apply time), a fixed one inlines.
    # ``applied`` -- the conditions some measured experiment applies, in any period -- decides
    # which of two ids that import under one name (cond_a and a) is kept; two applied ones are
    # refused (#905).
    measured_ids = {row.experiment_id for row in measurement_rows}
    conditions = conditions_from_rows(
        tc_condition_rows, surrogate_params, species_by_id, free_names=free_names,
        fixed_params=fixed_params,
        applied={r.condition_id for r in experiment_rows if r.experiment_id in measured_ids})
    # Each (experiment, model) group recovers its model from the rows' modelId (ADR-0041);
    # a single-model job carries modelId '' and emits no per-experiment model: field. A group
    # with a row-varying noise binding also writes its per-measurement sidecar (ADR-0045). The
    # time-course experiment rows exclude those absorbed into a (plain or pre-equilibrated) scan.
    model_location_of = {m['model_id']: m['location'] for m in models}
    absorbed_experiment_ids = dr_experiment_ids | pdr_experiment_ids
    tc_experiment_rows = [r for r in experiment_rows
                          if r.experiment_id not in absorbed_experiment_ids]
    # Every file the three builders below write is named from ONE registry, so no experiment's
    # data can overwrite another's (#903 review): each experiment's <name>.exp first, then its
    # replicates and sidecar, a taken name moved to a free one (the model files are reserved).
    files = _DataFileNames(_experiment_names(datas, dose_responses, preequil_scans),
                           reserved=model_texts)
    experiments = _experiments(datas, tc_experiment_rows, out_dir, model_location_of,
                               param_bindings, files=files)
    # Dose-response scans become parameter_scan experiments: a steady-state scan (scan_time inf)
    # carries no t_end: (the .exp's swept-axis column 0 infers the type); a finite scan carries
    # t_end: <t> (ADR-0046). Their .exp files are written here.
    experiments += _dose_response_experiments(dose_responses, out_dir, model_location_of,
                                              files=files)
    # Pre-equilibrated scans become preequilibrate:+condition: parameter_scan experiments (ADR-0062).
    experiments += _preequilibrated_dose_response_experiments(
        preequil_scans, out_dir, model_location_of, files=files)
    # The `perturbations: none` conditions the experiments apply (#906, ADR-0150).
    experiments = _declare_unperturbed_conditions(
        conditions, experiments, tc_condition_rows, experiment_rows, unperturbed, measured_ids)
    _refuse_fixed_equilibration_of_time_dependent_models(experiments, models, model_texts)

    # Each model file is carried verbatim -- no synthesis, no edit, for BNGL or SBML
    # (ADR-0036) -- except for the marked estimate=false overrides applied above (#907,
    # ADR-0149). Expression observables live in the conf's measurement-model layer below.
    for loc, text in model_texts.items():
        (out_dir / loc).write_text(text)
    for override in fixed_overrides:
        _report_fixed_override(override)

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
            multi=len(job_types) > 1, fixed_overrides=fixed_overrides)
    return out_dir


# ---------------------------------------------------------------------------
# Tables: every file under each problem.yaml key, concatenated (#902)
# ---------------------------------------------------------------------------

def _read_problem_tables(problem, base):
    """Read every table file the ``problem.yaml`` lists, concatenated in list order (#902).

    PEtab v2 types each ``*_files`` key as a *list*, and libpetab reads a problem by reading
    every listed file and chaining their rows in list order (``Problem.from_yaml``; the
    ``measurements`` / ``parameters`` / ... properties). A problem may therefore split its
    measurements (or any other table) over several files, and ``petab1to2`` keeps a v1
    problem's several measurement/observable/condition files as such a list. The importer used
    to read element ``[0]`` of each list and nothing else, so a split problem was fitted to
    part of its data, and a parameter declared only in a later file stayed fixed at its
    model-file value, with no message.

    Returns ``(parameter_rows, observable_rows, measurement_rows, condition_rows,
    experiment_rows, mapping_rows)``. An id that identifies one table entity may be defined in
    only one place, exactly where libpetab's ``lint_problem`` reports a duplicate
    (``CheckUniquePrimaryKeys`` / ``CheckMappingTable``); :func:`_require_unique_ids` refuses
    it with a ``PybnfError`` naming the id and the file(s). Concatenating instead would let the
    later row silently win, or double-declare a free parameter. Measurement rows carry no id:
    a repeated row is a replicate in PEtab and is kept, as libpetab keeps it.
    """
    def read_all(key, reader):
        return [(name, row) for name in problem[key] for row in reader(base / name)]

    parameters = read_all('parameter_files', read_parameter_table)
    observables = read_all('observable_files', read_observable_table)
    measurements = read_all('measurement_files', read_measurement_table)
    conditions = read_all('condition_files', read_condition_table)
    experiments = read_all('experiment_files', read_experiment_table)
    mappings = read_all('mapping_files', read_mapping_table)

    # One row per parameter / observable / mapping alias, wherever it sits: a repeat inside
    # one file is the same lint error as a repeat across two.
    _require_unique_ids(parameters, lambda r: r.parameter_id, 'parameter', 'parameterId')
    _require_unique_ids(observables, lambda r: r.observable_id, 'observable', 'observableId')
    _require_unique_ids(mappings, lambda r: r.petab_id, 'mapping', 'petabEntityId')
    # A condition (or experiment) is SEVERAL rows of one file -- one per target (or period)
    # -- so only a second file defining the same id is a duplicate. libpetab groups each
    # file's rows by id into one Condition/Experiment, so the id then occurs twice.
    _require_unique_ids(conditions, lambda r: r.condition_id, 'condition', 'conditionId',
                        across_files_only=True)
    _require_unique_ids(experiments, lambda r: r.experiment_id, 'experiment', 'experimentId',
                        across_files_only=True)
    return tuple([row for _name, row in table] for table in
                 (parameters, observables, measurements, conditions, experiments, mappings))


def _require_unique_ids(sourced_rows, id_of, table, column, across_files_only=False):
    """Refuse an id defined more than once in one table kind (#902).

    ``sourced_rows`` is ``[(file_name, row), ...]`` over every file of the table, in list
    order. With ``across_files_only`` a repeat inside one file is allowed (a condition's or an
    experiment's several rows) and only the same id in a second file is refused. The message
    names the id and the file(s) so the user knows which rows to merge or delete."""
    first_file = {}
    for name, row in sourced_rows:
        rid = id_of(row)
        if rid not in first_file:
            first_file[rid] = name
            continue
        prev = first_file[rid]
        if prev == name and across_files_only:
            continue
        where = f'twice in {prev}' if prev == name else f'in both {prev} and {name}'
        raise PybnfError(
            f"The PEtab {table} tables define {column} '{rid}' more than once ({where}). "
            f"PEtab allows each {column} to be defined in one place only (libpetab's "
            f"lint_problem reports it as a duplicate), and the importer cannot tell which "
            f"definition is meant. Keep one definition and delete or merge the other.")


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
            # Not a free parameter. A fixed model parameter's nominalValue is written into the
            # model copy (_apply_fixed_model_parameters, #907); any other fixed row is a
            # constant inlined where the tables reference it.
            continue
        model_param, is_surrogate = _model_param(row.parameter_id)
        if is_surrogate:
            surrogate_params.add(model_param)
        fp = free_parameter_from_row(row)
        free_param_lines.append(_free_parameter_conf_line(fp, model_param))
        if fp.value is not None:
            # The problem's own nominalValue, carried through as the fit's start point
            # (#583). Previously read and discarded, so the seed a PEtab problem ships with
            # had to be transcribed by hand -- which is exactly the "seed a method at a
            # known point" workflow that #559 was filed about.
            free_param_lines.append(f'start_point = {model_param} {num(fp.value)}')
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
    the prior registry so the record round-trips to the same ``FreeParameter``.

    A PEtab ``nominalValue`` becomes the fit's start point (#583). ADR-0043's field table
    has always advertised the ``initial_value`` <-> ``nominalValue`` mapping; the importer
    read the value onto ``FreeParameter.value`` and then dropped it on the way out, so a
    problem's own published point never reached the conf it generated. It is emitted as a
    ``start_point =`` line beside the declaration rather than as a record field, so the
    compact legacy form stays compact and one deletable line drops the seed."""
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


def _require_no_protocol_actions(text, location):
    """Refuse a BNGL model file that carries a protocol of its own (#969, ADR-0152).

    In a PEtab problem the tables define the protocol, and the importer copies the model file
    into an edition-2 job whose ``experiment:`` lines stand for them. An action the model file
    carries would run ahead of each of those experiments, so the fitter refuses such a model at
    config load. The importer applies the same check
    (:meth:`~pybnf.pset.BNGLModel.require_no_protocol_actions`, also run by the exporter) to
    every BNGL model first, so a third-party model with a leftover ``simulate`` or a
    ``setParameter`` is refused before any file is written, naming the model file and the line.
    Its actions may be only ``generate_network`` and ``setOption``.
    """
    try:
        model = BNGLModel(location, suppress_free_param_error=True, text=text)
    except ModelError as exc:
        raise PybnfError(f"Model '{location}' could not be read as BNGL: {exc}.") from exc
    model.require_no_protocol_actions(location, where='petab')


# ---------------------------------------------------------------------------
# Fixed model parameters: estimate=false rows -> the model copies (#907)
# ---------------------------------------------------------------------------

#: One value written into an imported model copy (#907): the parameterId, the model file, the
#: value that file had (a BNGL right-hand side, an SBML ``value`` attribute, or ``None`` for an
#: SBML parameter that declared none), and the parameters table's nominalValue.
_FixedOverride = namedtuple('_FixedOverride',
                            'parameter_id location model_value nominal_value')

#: The SBML constructs that make a parameter something other than a constant, split by
#: whether PEtab itself allows such a parameter in the parameters table. libpetab's lint
#: (``get_valid_parameters_for_parameter_table``) excludes every rule target, so a row naming
#: one is a malformed problem; it accepts the others, which PyBNF declines to rewrite.
_PETAB_FORBIDDEN_CONSTRUCTS = ('assignment rule', 'rate rule')


def _apply_fixed_model_parameters(parameter_rows, models, model_texts):
    """Write each fixed model parameter's nominalValue into the imported model copies (#907).

    PEtab's parameters table has precedence over the model file: a row with ``estimate =
    false`` fixes its parameter at the row's nominalValue, and every PEtab tool (libpetab's
    parameter mapping, AMICI, pyPESTO) simulates with that value. The importer does the same
    by editing its copy of each model that declares the parameter -- the parameters table is
    global, so a multi-model problem is edited in every model -- and only where the file
    disagrees: a BNGL numeric right-hand side or an SBML ``value`` that already equals the
    nominalValue is left byte-identical, while a different number, a BNGL expression, or a
    missing SBML value is replaced (:func:`pybnf.petab._bngl.set_parameter_values`,
    :func:`pybnf.petab._sbml.set_parameter_values`). Each edit carries a comment in the copy,
    and :func:`import_job` lists it in the conf header and prints it once the copies are
    written. The source files are never touched.

    A fixed row that names no model entity is not handled here; it keeps the inlining path
    (a formula constant, a fixed sigma, a condition targetValue -- ADR-0037/0075/0076).

    Returns ``(model_texts, overrides)``: the ``{location: text}`` map with the edits applied,
    and one ``_FixedOverride`` per edited (parameter, model), in table order.

    Raises ``PybnfError`` for a fixed row with no nominalValue (PEtab v2 requires one), for a
    fixed row naming a model entity that is not a parameter or an SBML rule target (both
    rejected by PEtab's own lint), and for a non-finite nominalValue on a model parameter.
    Raises ``NotImplementedError`` for an SBML parameter whose value an initial assignment, an
    event assignment or an algebraic rule also sets, and for a BNGL parameter the model's own
    actions set (``setParameter``, a scan): PEtab allows these, but the written value would not
    be the parameter's value for the whole simulation (ADR-0149).
    """
    fixed = {}
    for row in parameter_rows:
        if row.estimate:
            continue
        if row.nominal_value is None:
            raise PybnfError(
                f"PEtab parameter '{row.parameter_id}' has estimate=false but no "
                f"nominalValue. PEtab v2 requires a nominalValue on every fixed parameter: it "
                f"is the value the parameter takes. Give the row a nominalValue, or set "
                f"estimate=true to fit it.")
        fixed[row.parameter_id] = float(row.nominal_value)
    texts = dict(model_texts)
    overrides = []
    if not fixed:
        return texts, overrides
    for m in models:
        loc, lang = m['location'], (m['language'] or 'bngl').lower()
        if lang == 'sbml':
            targets = _fixed_sbml_targets(parse_sbml_model(texts[loc]), fixed, loc)
            setter = set_sbml_parameter_values
        else:
            targets = _fixed_bngl_targets(parse_bngl_model(texts[loc]), fixed, loc)
            setter = set_bngl_parameter_values
        for pid in targets:
            if not math.isfinite(fixed[pid]):
                raise PybnfError(
                    f"PEtab parameter '{pid}' has estimate=false with nominalValue "
                    f"{fixed[pid]!r}, which is not a finite number, and it names a parameter "
                    f"of the model {loc}. A model parameter cannot be fixed at it.")

        def note(pid, old):
            return (f'PEtab parameters.tsv: estimate=false, nominalValue {num(fixed[pid])} '
                    f'(model file: {_model_value_text(old)})')

        texts[loc], changed = setter(texts[loc], {p: fixed[p] for p in targets}, note)
        overrides += [_FixedOverride(pid, loc, changed[pid], fixed[pid])
                      for pid in targets if pid in changed]
    return texts, overrides


def _refuse_overwriting_an_edited_source(overrides, base, out_dir):
    """Refuse an import whose edited model copy would be written over its source (#907).

    A copy is written at its PEtab ``location`` under ``out_dir``, and a location may leave
    the problem directory (``../models/m.bngl``). With ``out_dir`` beside the problem
    directory, or equal to it, that path is the source file itself, and writing the edited
    copy would change the user's model: a later problem that leaves the parameter to the model
    file would then silently simulate this table's value. A model that needs no edit is still
    written back as it was read, as before #907. Called before anything is written."""
    for loc in dict.fromkeys(o.location for o in overrides):
        dest = (out_dir / loc).resolve()
        if dest == (base / loc).resolve():
            names = ', '.join(o.parameter_id for o in overrides if o.location == loc)
            raise PybnfError(
                f"PEtab import: parameters.tsv fixes {names} (estimate=false) at a value the "
                f"model {loc} does not have, so the imported copy of that model is edited, but "
                f"the copy would be written to {dest}, which is the source model file. The "
                f"import never changes a source model. Import into a directory where {loc} "
                f"does not lead back to the source file.")


def _report_fixed_override(o):
    """Print one applied override (#907) -- on the console whatever the verbosity, since it
    changes the model the job simulates. Called once the copies are written, so a later
    refusal never leaves a message about a copy that does not exist."""
    print0(f"PEtab import: parameters.tsv fixes {o.parameter_id} = {num(o.nominal_value)} "
           f"(estimate=false); {o.location} has {_model_value_text(o.model_value)}, so the "
           f"imported copy uses {num(o.nominal_value)}.")


def _model_value_text(old):
    """How a model file's value for a fixed parameter is shown to the user (#907): the BNGL
    right-hand side or SBML attribute as written (a continued BNGL line's whitespace
    collapsed), or ``no value`` for an SBML parameter that declared none."""
    return 'no value' if old is None else ' '.join(old.split())


def _not_a_parameter(pid, kind, loc):
    return PybnfError(
        f"PEtab parameter '{pid}' has estimate=false, but in the model {loc} '{pid}' is "
        f"{kind}, not a parameter. PEtab allows only model parameters (and parameters the "
        f"tables introduce) in the parameters table, and libpetab's lint rejects this row. "
        f"Remove it, or rename the entity it collides with.")


def _fixed_bngl_targets(ent, fixed, loc):
    """The fixed ids the BNGL model declares in ``begin parameters``, in table order. A fixed
    id naming another kind of BNGL entity is refused (PEtab's BNGL loader admits only
    parameters to the parameters table), and so is a parameter the model's own actions set
    (a job runs them ahead of each experiment's simulation, so they would undo the edit)."""
    others = (('an observable', ent.observable_names),
              ('a global function', ent.function_names),
              ('a compartment', ent.compartment_names),
              ('a molecule type', ent.molecule_type_names))
    set_by_actions = parameters_set_by_actions(ent.text)
    targets = []
    for pid in fixed:
        if pid in ent.parameters:
            if pid in set_by_actions:
                raise NotImplementedError(
                    f"PEtab parameter '{pid}' has estimate=false with nominalValue "
                    f"{num(fixed[pid])}, but an action in the model {loc} (a setParameter or "
                    f"a parameter scan) also sets it, which would override the value written "
                    f"into its begin parameters line. PyBNF does not rewrite a model's actions "
                    f"(#907, ADR-0149). Remove the action, or remove the row if the action's "
                    f"value is the one you want.")
            targets.append(pid)
            continue
        for kind, names in others:
            if pid in names:
                raise _not_a_parameter(pid, kind, loc)
    return targets


def _fixed_sbml_targets(ent, fixed, loc):
    """The fixed ids the SBML model declares as global parameters whose ``value`` attribute
    alone settles them, in table order. Refuses a fixed id naming a species or a compartment,
    or a parameter some other SBML construct assigns (see
    :func:`_apply_fixed_model_parameters`). A ``constant="false"`` parameter that no construct
    assigns is accepted: nothing can change it, so its ``value`` holds throughout."""
    targets = []
    for pid in fixed:
        if pid in ent.species_names:
            raise _not_a_parameter(pid, 'a species', loc)
        if pid in ent.compartment_names:
            raise _not_a_parameter(pid, 'a compartment', loc)
        if pid not in ent.parameter_names:
            continue
        constructs = (ent.assigned_by or {}).get(pid, ())
        forbidden = [c for c in constructs if c in _PETAB_FORBIDDEN_CONSTRUCTS]
        if forbidden:
            raise PybnfError(
                f"PEtab parameter '{pid}' has estimate=false, but in the model {loc} it is "
                f"the target of {_a_list(forbidden)}, which sets its value throughout the "
                f"simulation. PEtab does not allow a rule target in the parameters table "
                f"(libpetab's lint rejects this row). Remove the row, or remove the rule if "
                f"'{pid}' should be a constant.")
        if constructs:
            raise NotImplementedError(
                f"PEtab parameter '{pid}' has estimate=false with nominalValue "
                f"{num(fixed[pid])}, but in the model {loc} its value is also set by "
                f"{_a_list(constructs)}. PyBNF applies a fixed nominalValue by writing it into "
                f"the parameter's value attribute, which would not make it the parameter's "
                f"value for the whole simulation, and it does not rewrite {_a_list(constructs)} "
                f"(#907, ADR-0149). Edit the model so '{pid}' is a plain constant, or remove "
                f"the row if the model's own definition is the one you want.")
        targets.append(pid)
    return targets


def _a_list(constructs):
    """``['initial assignment', 'event assignment']`` -> ``'an initial assignment and an
    event assignment'``."""
    named = [('an ' if c[0] in 'aeiou' else 'a ') + c for c in constructs]
    return named[0] if len(named) == 1 else ', '.join(named[:-1]) + ' and ' + named[-1]


# ---------------------------------------------------------------------------
# Observables: rows -> column map + objective token
# ---------------------------------------------------------------------------

def _model_namespace(model_text, language):
    """The model's expression namespace + entity name set + assignment rules, per language
    (ADR-0036).

    Returns ``(namespace_symbols, entity_names, derived)``: ``namespace_symbols`` are
    the names an ``observableFormula`` may reference (the BNGL ``ParamList`` -- parameters u
    observables u functions; or SBML species u parameters u compartments -- ADR-0026/0036);
    ``entity_names`` is the broader declared-name set used for the shadow check (a measurement
    model's id must not collide with a model output column); ``derived`` maps each SBML entity
    the model file defines in terms of others to a ``DerivedSymbol`` -- an ``assignmentRule``
    target (#465/#493) and a parameter or compartment an ``initialAssignment`` derives (#795) --
    the map the importer **inlines** so a formula naming one resolves down to the
    species/parameters it is computed from, the import peer of the config-load inlining. It is
    ``{}`` for a BNGL model. Read from the model text directly with the stdlib scanners
    (``_bngl`` / ``_sbml``), simulator-free.
    """
    if language == 'sbml':
        ent = parse_sbml_model(model_text)
        return ent.namespace_symbols, set(ent.namespace_symbols), ent.derived_symbols
    ent = parse_bngl_model(model_text)
    namespace = (set(ent.parameters) | set(ent.observable_names)
                 | set(ent.function_names))
    entity_names = (namespace | set(ent.molecule_type_names)
                    | set(ent.compartment_names))
    return namespace, entity_names, {}


def _model_parameter_ids(model_text, language):
    """The model's parameter ids (BNGL ``begin parameters``; SBML ``listOfParameters``) -- the
    entities a ``parameter_scan`` can sweep. A species amount or a compartment size is not one
    (#904)."""
    if (language or 'bngl').lower() == 'sbml':
        return set(parse_sbml_model(model_text).parameter_names)
    return set(parse_bngl_model(model_text).parameters)


def _shared_bare_entities(observable_rows, namespace, derived, row_varying_obs_params):
    """Model entities named as a bare ``observableFormula`` by more than one observable (#503).

    A bare-name observable (no observableParameters placeholder, formula an identifier in the
    model namespace) otherwise maps its ``observableId`` straight to that model column
    (:func:`_observable_id_to_column`); two such observables measuring one entity in different
    experiments would then emit colliding ``noise_model <column>`` overrides that ``parse.ploop``
    rejects. Returns the set of entity names shared by >1 bare-name observable; each such
    observable is materialized to its own ``observableId`` column via an identity measurement
    model instead (ADR-0077), so the data columns and per-observable noise key by the distinct
    ``observableId``. The assignment-rule inlining mirrors the main pass (#493) so a rule-variable
    formula (an expression, never bare) is correctly excluded, and a placeholder-bearing or
    row-varying formula (also never bare) is skipped."""
    counts = Counter()
    for row in observable_rows:
        raw = (row.observable_formula or '').strip()
        if derived:
            from .formula import inline_derived_symbols
            raw = inline_derived_symbols(
                raw, derived, observable_id=row.observable_id)
        if _PLACEHOLDER.search(raw) or row.observable_id in row_varying_obs_params:
            continue
        if _IDENTIFIER.match(raw) and raw in namespace:
            counts[raw] += 1
    return {entity for entity, n in counts.items() if n > 1}


def _observable_id_to_column(observable_rows, namespace, entity_names, fixed_params,
                             obs_params, free_names, row_varying_obs_params=(),
                             derived=None):
    """Map each ``observableId`` to the model column it measures, recording a measurement
    model for any expression ``observableFormula`` (ADR-0036). Iteration order = table order,
    which fixes the wide-data column order on the measurement pivot.

    Returns ``(mapping, measurement_models)``:

    * A **bare model-entity name** ``observableFormula`` (the common case, ADR-0025) maps its
      ``observableId`` to that name -- PyBNF matches the ``.exp`` column to the model
      observable/function/species by name and the backend already produces it, so no
      translator runs and the path stays dependency-free. **Exception (ADR-0077):** when >1
      bare-name observable names the *same* model entity (Bertozzi's ``y_I_NY``/``y_I_CA`` both
      on ``I_``), each is instead **materialized** to its own ``observableId`` column via an
      identity measurement model ``(observableId, entity)`` -- otherwise their per-observable
      ``noise_model`` overrides would both key by the one shared column and collide (#503). The
      uniquely-targeted case is unchanged (byte-identical bare-name path).
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

    ``derived`` (#493/#795) maps an SBML entity the model file defines in terms of others to a
    ``DerivedSymbol``. Such an entity is declared as a parameter/species but has no value of its
    own -- an ``assignmentRule`` target is recomputed every step, and a parameter an
    ``initialAssignment`` derives has only a placeholder attribute -- so it is never a
    simulation-output column and cannot be resolved *as a symbol* (that is exactly why it is
    absent from ``namespace``). Each observableFormula is therefore **inlined** first: every
    referenced entity is replaced by its defining RHS (recursively) so the formula reduces to
    species and parameters the layer can evaluate -- the SBML analogue of a BNGL global function
    in an observableFormula, which PyBNF already accepts. A formula naming no derived entity is
    returned verbatim (the bare-name common case stays dependency-free); this mirrors the
    config-load inlining (#465/#795).
    """
    derived = derived or {}
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
    # A model entity named by >1 bare-name observable is materialized to per-observableId
    # columns below so their per-observable noise_model overrides key by distinct ids, not the
    # one shared column (#503, ADR-0077). A uniquely-targeted entity keeps the bare-name path.
    shared_entities = _shared_bare_entities(
        observable_rows, namespace, derived, row_varying_obs_params)
    for row in observable_rows:
        raw = (row.observable_formula or '').strip()
        # Inline any SBML assignment-rule variable the formula names down to the species /
        # parameters the rule is computed from (#493) BEFORE the bare/expression branch: a bare
        # ``EGFRtot`` becomes its RHS expression (a measurement model), a rule reference inside a
        # larger formula resolves in place. A formula naming no rule variable returns verbatim, so
        # the bare-name common case never reaches the translator (dependency-free, byte-stable);
        # the inlining leaves any observableParameters placeholder untouched (rules are model
        # MathML, never placeholders), so the placeholder handling below is unchanged.
        if derived:
            from .formula import inline_derived_symbols
            raw = inline_derived_symbols(
                raw, derived, observable_id=row.observable_id)
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
            if formula not in shared_entities:
                mapping[row.observable_id] = formula      # unique bare-name path (no translator)
                continue
            # A model entity shared by >1 bare-name observable (#503, ADR-0077): fall through to
            # materialize an identity measurement model (observableId -> the entity) so this
            # observable owns its own obsId column and its per-observable noise keys by obsId,
            # not the shared entity. `formula` is the bare entity name, which validates trivially
            # against the namespace and inlines to itself in the measurement-model branch below.
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


class _ColumnMeans:
    """Decides whether an observable's sigma is a PyBNF ``column_mean`` sigma (#894).

    PyBNF's ``column_mean`` sigma (``objective = ave_norm_sos``, or ``sigma = column_mean``)
    is **per experiment**. The fit scores one experiment at a time and gives each point the
    mean of its own experiment's observed values, replicates pooled. PEtab has no such
    source. The exporter writes that number either as a constant noiseFormula (when every
    experiment has the same mean) or as each measurement row's numeric ``noiseParameters``.
    This class reads either form back to ``column_mean`` **only** when every scored point's
    sigma equals the mean of the experiment it will belong to in the imported job. Then the
    imported fit gives every point exactly the sigma the PEtab problem gives it. Anything
    else stays a fixed sigma (``fix_at``, or the per-point ``_SD`` column). Those are always
    exact, only less tidy.

    ``groups`` lists the imported job's experiments as their replicate ``Data`` lists: each
    time-course ``(experimentId, modelId)`` group and each reconstructed dose-response or
    pre-equilibrated scan (one experiment however many PEtab experiments its doses were).
    The old check compared one mean pooled over every experiment, so a problem whose
    experiments differ in magnitude re-imported as a column mean that no experiment has.
    """

    def __init__(self, groups, observable_id_to_column, sd_suffix='_SD'):
        self._groups = groups
        self._column_of = observable_id_to_column
        self._sd_suffix = sd_suffix

    def _experiment_means(self, observable_id):
        """``[(datas, mean), ...]`` for each experiment with an observed value of the column --
        the mean ``Data.column_mean`` gives that experiment's stacked Data at fit time. It is
        taken over observed values only (#707); an experiment with none scores no point."""
        col = self._column_of[observable_id]
        out = []
        for group in self._groups:
            present = [data for data in group if col in data.cols]
            if not present:
                continue
            mean = float(observed_mean(np.concatenate([data[col] for data in present])))
            if not np.isnan(mean):
                out.append((present, mean))
        return out

    def constant_matches(self, observable_id, sigma):
        """True iff the constant ``sigma`` is the column mean of **every** experiment that
        measures the observable."""
        means = self._experiment_means(observable_id)
        return bool(means) and all(_approx(sigma, mean) for _datas, mean in means)

    def per_row_matches(self, observable_id):
        """True iff every observed point's per-row sigma (its rebuilt ``<col>_SD`` cell) is the
        column mean of its own experiment."""
        col = self._column_of[observable_id]
        sd_col = col + self._sd_suffix
        means = self._experiment_means(observable_id)
        if not means:
            return False
        for datas, mean in means:
            for data in datas:
                if sd_col not in data.cols:
                    return False
                observed = ~np.isnan(data[col])
                if not all(_approx(float(sd), mean) for sd in data[sd_col][observed]):
                    return False
        return True

    def matches(self, observable_id, source):
        """Whether the resolved sigma ``source`` (a :func:`_resolve_noise` source) is this
        observable's per-experiment column mean. Only a fixed number can be: a constant
        noiseFormula, or a per-point numeric placeholder."""
        kind, value = source
        if kind == 'constant':
            return self.constant_matches(observable_id, value)
        if kind == 'placeholder':
            return self.per_row_matches(observable_id)
        return False


def _without_columns(data, names):
    """``data`` without the columns in ``names`` (the same object when it has none of them)."""
    keep = [header for _i, header in sorted(data.headers.items()) if header not in names]
    if len(keep) == len(data.headers):
        return data
    arr = data.data[:, [data.cols[header] for header in keep]]
    return Data.from_columns(arr, keep, indvar=data.indvar)


# PEtab noiseDistribution -> (PyBNF base noise family, its additive scale). The v2
# ``log-`` prefixes are natural log (LN); a re-injected observableTransformation (issue
# #499) can override the scale below. Anything not here (e.g. neg_bin, removed from v2)
# is refused in _resolve_noise.
_PETAB_DISTRIBUTION = {
    'normal':      ('gaussian', 'linear'),
    'log-normal':  ('gaussian', 'ln'),
    'laplace':     ('laplace',  'linear'),
    'log-laplace': ('laplace',  'ln'),
}

# v1 observableTransformation -> the additive scale it names (issue #499). PEtab v2 has no
# log10 noiseDistribution, so a log10 residual arrives only through this re-injected column;
# ``lin`` leaves the noiseDistribution's own scale, ``log`` is natural (LN), ``log10`` is LOG10.
_TRANSFORMATION_SCALE = {'lin': None, 'log': 'ln', 'log10': 'log10'}

# (base family, additive scale) -> the native noise_model family token that names it. The two
# log Gaussian members are explicit: ``lognormal`` = Gaussian(LOG10), following PyBNF's bare-log
# convention, and ``lnnormal`` = Gaussian(LN), the natural-log member PEtab calls ``log-normal``
# (ADR-0022/0084). Log Laplace still has no native token -> NotImplementedError in _resolve_noise.
_NATIVE_FAMILY_TOKEN = {
    ('gaussian', 'linear'): 'gaussian',
    ('gaussian', 'log10'):  'lognormal',
    ('gaussian', 'ln'):     'lnnormal',
    ('laplace',  'linear'): 'laplace',
}

# native noise_model family token -> its scale-parameter field name (for the emitted
# ``noise_model = <family>, <param> = ...`` line). gaussian/lognormal/lnnormal share ``sigma``
# (same Gaussian kernel, different additive scale); laplace uses ``scale`` (ADR-0031).
_NOISE_MODEL_PARAM = {
    'gaussian': 'sigma', 'lognormal': 'sigma', 'lnnormal': 'sigma', 'laplace': 'scale'}


def _objective_directives(observable_rows, observable_id_to_column, noise_param_ids,
                          column_means, obs_params, noise_subs=None, row_varying_obs=(),
                          fixed_params=None, namespace=frozenset()):
    """Recover the conf's objective directive lines from the observables' noise (ADR-0031/0037).

    The inverse of the objective-family / whole-fit / per-observable ``noise_model`` export.
    Returns ``(lines, sd_readers)``: ``sd_readers`` is the set of observableIds whose recovered
    sigma reads the per-point ``<col>_SD`` data column; the caller drops every other rebuilt
    ``_SD`` companion. A fixed sigma that is exactly each experiment's own column mean
    (``column_means``, a :class:`_ColumnMeans` -- #894) is recovered as ``column_mean``, which
    reads no data column. ``lines`` is a **list** of conf lines:

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
    sigma source from this map. A ``log10`` ``observableTransformation`` (issue #499) selects
    the native ``lognormal`` family (``objective = lognormal`` / a ``lognormal`` noise_model
    line); natural-log Gaussian selects ``lnnormal``. A log Laplace, an expression
    ``noiseFormula``, and a per-point laplace placeholder raise ``NotImplementedError`` -- the
    boundary is in code, not a silent mis-recovery.
    """
    per_obs = [(row, *_resolve_noise(
                    row, noise_param_ids.get(row.observable_id),
                    _placeholder_subs(row.observable_id, obs_params, noise_subs, fixed_params),
                    row.observable_id in row_varying_obs, fixed_params, namespace))
               for row in observable_rows]
    # Which observables' fixed sigma is exactly their per-experiment column mean (#894).
    is_column_mean = {row.observable_id: column_means.matches(row.observable_id, src)
                      for row, _family, src in per_obs}
    single = _try_uniform_directive(per_obs, is_column_mean)
    if single is not None:
        lines, via_column_mean = [single[0]], single[1]
    else:
        # The per-observable lines give every column-mean observable a column_mean source.
        lines, via_column_mean = (
            _per_observable_directives(per_obs, observable_id_to_column, is_column_mean), True)
    sd_readers = {row.observable_id for row, _family, src in per_obs
                  if src[0] == 'placeholder'
                  and not (via_column_mean and is_column_mean[row.observable_id])}
    return lines, sd_readers


def _resolve_noise(row, noise_param_id, obs_subs, row_varying=False,
                   fixed_params=None, namespace=frozenset()):
    """One observables row -> ``(family_token, source)`` where ``family_token`` is the native
    noise_model family (``gaussian`` / ``lognormal`` / ``lnnormal`` / ``laplace``) and ``source``
    is one of
    ``('placeholder', None)`` (per-point ``_SD``), ``('constant', value)`` (a fixed sigma),
    ``('free', parameter_id)`` (an estimated sigma), ``('formula', expr)`` (an expression
    sigma over free parameters -> ``FormulaSigma``, ADR-0044), ``('prediction_formula', expr)``
    (an expression whose σ scales with the simulated prediction -> ``PredictionFormulaSigma``,
    ADR-0075), or ``('per_measurement', expr)`` (a row-varying placeholder bound per data point
    -> ``PerMeasurementFormulaSigma``, ADR-0045).

    ``fixed_params`` maps each fixed (estimate=0) PEtab parameter id to its numeric value: a
    ``noiseParameters`` id that resolves to one is a fixed sigma (``('constant', value)``, e.g.
    Oliveira's ``sd_cumulative_deaths = 1``), not an estimated ``fit`` (ADR-0075). ``namespace``
    is the set of **model entities that are not free parameters** (the simulated columns): a
    substituted arithmetic noiseFormula that references any of them is prediction-dependent
    (``prediction_formula``), else a pure free-parameter expression (``formula``).

    The family token comes from ``noiseDistribution`` (the Gaussian/Laplace family) **and** a
    re-injected ``observableTransformation`` (the additive scale; issue #499): ``log10`` +
    ``normal`` selects the native ``lognormal`` token (``Gaussian(LOG10)``), the scale the
    paper (and the v1 problem) score on. Natural-log Gaussian selects ``lnnormal``; a family/scale
    with no native token -- currently log Laplace on either base -- raises ``NotImplementedError``
    (:func:`_native_noise_family`).

    ``row_varying`` (ADR-0045): the observable's ``noiseParameters`` id **differs** across its
    measurement rows, so it cannot reduce to one substituted symbol -- the noiseFormula is
    emitted with its placeholder **kept** (``('per_measurement', formula)``) and the per-row
    token is bound from the experiment's sidecar binding table at eval time. Checked first,
    before the constant-reduction paths below.

    A **declared placeholder** noiseFormula -- a lone ``noiseParameter${n}_<id>`` token or a
    bare id listed in the row's ``noisePlaceholders`` -- has its value supplied per measurement:
    a parameter id constant across the observable (``noise_param_id``) is an estimated sigma
    (``('free', id)``, Boehm), unless that id is **fixed** -> a fixed sigma
    (``('constant', value)``, Oliveira, ADR-0075); otherwise it is the per-point ``_SD`` source
    (``('placeholder', None)``, chi_sq). Otherwise (ADR-0044/0075) a constant-per-observable
    ``observableParameter*`` / ``noiseParameter*`` placeholder is substituted in (``obs_subs``)
    and the resulting noiseFormula classified: a number -> ``('constant', v)``, a bare id ->
    ``('free', id)``, a free-parameter expression -> ``('formula', expr)``, an expression that
    also references a model entity -> ``('prediction_formula', expr)``. A placeholder that
    survives substitution (unresolved, and not row-varying) raises the deferred frontier."""
    fixed_params = fixed_params or {}
    family_token = _native_noise_family(row)
    formula = (row.noise_formula or '').strip()
    if not formula:
        raise PybnfError(f"Observable '{row.observable_id}' is missing a noiseFormula.")
    # ADR-0045/0075: a row-varying noiseParameters cell (a single id, or a multi-parameter tuple
    # whose per-gel scale differs) is bound per data point from the sidecar; keep the
    # placeholder(s) in the noiseFormula verbatim (PerMeasurementFormulaSigma resolves them).
    if row_varying:
        return family_token, ('per_measurement', formula)
    placeholders = {p.strip() for p in (row.noise_placeholders or '').split(';') if p.strip()}
    # ADR-0037 declared-placeholder path FIRST (byte-for-byte: Boehm/chi_sq). A BARE single
    # placeholder only -- a multi-parameter noiseFormula (Raia's affine) is an expression and
    # falls through to substitute-and-classify below (ADR-0075).
    if _BARE_NOISE_PLACEHOLDER.match(formula) or formula in placeholders:
        if noise_param_id is not None:
            if noise_param_id in fixed_params:                 # Oliveira: fixed -> constant sigma
                return family_token, ('constant', fixed_params[noise_param_id])
            return family_token, ('free', noise_param_id)      # const-per-observable estimated sigma
        return family_token, ('placeholder', None)             # per-point _SD (chi_sq/lognormal)
    # ADR-0044/0075: substitute a constant-per-observable placeholder (an id stays a free symbol,
    # a number/fixed-parameter inlines via obs_subs), then classify. A bare number / id with no
    # placeholder passes through untouched (substitution stays dependency-free).
    if _PLACEHOLDER.search(formula):
        from .formula import substitute_placeholders
        formula = substitute_placeholders(formula, obs_subs)
        _require_no_placeholder(formula, row.observable_id)
    try:
        return family_token, ('constant', float(formula))  # a number -> ConstantSigma
    except ValueError:
        pass
    if _IDENTIFIER.match(formula):
        # A bare id: a fixed parameter is a constant sigma, an estimated one a free sigma.
        if formula in fixed_params:
            return family_token, ('constant', fixed_params[formula])
        return family_token, ('free', formula)             # a bare id -> FreeParameterSigma
    # An arithmetic expression: prediction-dependent (σ scales with the simulated output, Raia)
    # if it references a model entity, else a pure free-parameter expression (ADR-0044/0075).
    # Validate it parses (+ the petab extra present) and read its free symbols.
    from .formula import formula_free_symbols
    symbols = formula_free_symbols(formula)
    if any(s in namespace for s in symbols):
        return family_token, ('prediction_formula', formula)   # PredictionFormulaSigma (Raia)
    return family_token, ('formula', formula)                  # FormulaSigma (free params only)


def _native_noise_family(row):
    """The native noise_model family token (``gaussian`` / ``lognormal`` / ``lnnormal`` /
    ``laplace``) for one
    observables row, from ``noiseDistribution`` + a re-injected ``observableTransformation``.

    The distribution names the Gaussian/Laplace family (its ``log-`` prefix a natural-log
    scale); an ``observableTransformation`` of ``log10`` / ``log`` overrides the scale, the
    only channel for a log10 residual (issue #499, ADR-0073). Raises ``NotImplementedError`` for a
    distribution v2 removed (``neg_bin``) or a family/scale with no native token -- currently
    any log Laplace. Linear ``gaussian`` / ``laplace``, log10 Gaussian -> ``lognormal``, and
    natural-log Gaussian -> ``lnnormal`` are recovered (ADR-0084)."""
    dist = (row.noise_distribution or 'normal').lower()
    base = _PETAB_DISTRIBUTION.get(dist)
    if base is None:
        raise NotImplementedError(
            f"Observable '{row.observable_id}': noiseDistribution {dist!r} has no PyBNF "
            f"noise model (neg_bin was removed from PEtab v2; expected normal / laplace / "
            f"log-normal / log-laplace).")
    base_family, scale = base
    transformation = (row.observable_transformation or 'lin').strip().lower()
    if transformation not in _TRANSFORMATION_SCALE:
        raise PybnfError(
            f"Observable '{row.observable_id}': unknown observableTransformation "
            f"{transformation!r} (expected lin / log / log10).")
    trans_scale = _TRANSFORMATION_SCALE[transformation]
    if trans_scale is not None:
        if scale != 'linear' and scale != trans_scale:
            raise PybnfError(
                f"Observable '{row.observable_id}': observableTransformation "
                f"{transformation!r} contradicts the scale of noiseDistribution {dist!r}. "
                f"Give the residual scale in one place -- a log observableTransformation over "
                f"a linear noiseDistribution (normal / laplace).")
        scale = trans_scale
    token = _NATIVE_FAMILY_TOKEN.get((base_family, scale))
    if token is None:
        raise NotImplementedError(
            f"Observable '{row.observable_id}': the {base_family} family on the {scale} scale "
            f"(from noiseDistribution {dist!r}"
            f"{f' / observableTransformation {transformation!r}' if trans_scale else ''}) has "
            f"no native noise_model token yet (#407). Recovered: linear normal / laplace, "
            f"log10 normal (lognormal), and natural-log normal (lnnormal).")
    return token


def _try_uniform_directive(per_obs, is_column_mean):
    """A single whole-fit directive if the table is one PyBNF objective, else ``None``.

    Returns ``(line, via_column_mean)``; ``via_column_mean`` is True when the line's sigma is
    ``column_mean`` (``objective = ave_norm_sos`` or a whole-fit ``... = column_mean`` line).
    ``None`` signals a genuinely per-observable table (a mix of families or sigma sources, or
    a distinct ``fit``/``fix_at`` sigma per observable) -> :func:`_per_observable_directives`.
    The uniform cases are exactly the objective-family / whole-fit ``noise_model`` export
    inverse (preserved byte-for-byte).

    ``is_column_mean`` (``{observable_id: bool}``, from :class:`_ColumnMeans`) marks the
    observables whose fixed sigma is exactly each experiment's own column mean (#894). When
    every observable is marked, the table is one column-mean objective whatever mix of a
    constant noiseFormula and a per-row placeholder carried it. A unit sigma is read as
    ``sos`` / ``sod`` first, as before, even if some column's mean happens to be 1."""
    families = {family for _row, family, _src in per_obs}
    if len(families) != 1:
        return None     # mixed family (incl. a log10 vs linear scale) -> per-observable
    family = next(iter(families))     # native token: gaussian / lognormal / lnnormal / laplace
    param = _NOISE_MODEL_PARAM[family]

    # All-unit constant sigma: the sos / sod sugar tokens.
    unit = all(src == ('constant', 1.0) for _row, _family, src in per_obs)
    if unit and family == 'gaussian':
        return 'objective = sos', False
    if unit and family == 'laplace':
        return 'objective = sod', False
    # Every observable's sigma is its per-experiment column mean (#894).
    if not unit and all(is_column_mean[row.observable_id] for row, _family, _src in per_obs):
        if family == 'gaussian':
            return 'objective = ave_norm_sos', True
        return f'noise_model = {family}, {param} = column_mean', True

    kinds = {src[0] for _row, _family, src in per_obs}
    if len(kinds) != 1:
        return None     # mixed source -> per-observable
    kind = next(iter(kinds))

    if kind == 'per_measurement':
        # A row-varying placeholder sigma (ADR-0045) is inherently per-observable -- its
        # noiseFormula carries the observable-specific placeholder noiseParameter1_<id>, so
        # several observables are never "uniform" -- and it is bound from the experiment's
        # sidecar binding table. Always a per-observable noise_model line.
        return None
    if kind == 'prediction_formula':
        # A prediction-dependent sigma (PredictionFormulaSigma, ADR-0075) is inherently
        # per-observable -- its expression scales with THAT observable's simulated output, so
        # several observables are never "uniform". Always a per-observable noise_model line.
        return None
    if kind == 'formula':
        # An expression sigma (FormulaSigma, ADR-0044) has no whole-fit objective *token*, but a
        # uniform one (every observable the same expression) is a whole-fit noise_model line --
        # the inverse of the export's whole-fit formula sigma, so it round-trips byte-for-byte.
        # A non-uniform / per-observable formula falls to _per_observable_directives (not yet
        # re-exportable -- the deferred per-observable export boundary, ADR-0045).
        exprs = {src[1] for _row, _family, src in per_obs}
        if len(exprs) != 1:
            return None
        return f'noise_model = {family}, {param} = formula {exprs.pop()}', False
    if kind == 'placeholder':
        # Per-point _SD sigma: the Gaussian families have an objective token (chi_sq linear,
        # lognormal log10, lnnormal natural log); Laplace has no per-point token (#407).
        if family == 'gaussian':
            return 'objective = chi_sq', False
        if family == 'lognormal':
            return 'objective = lognormal', False
        if family == 'lnnormal':
            return 'objective = lnnormal', False
        raise NotImplementedError(
            f"A per-point ({family}) placeholder noiseFormula has no PyBNF objective token "
            f"(only the Gaussian per-point _SD cases -- chi_sq, lognormal, lnnormal -- are "
            f"recovered; "
            f"#407).")
    if kind == 'free':
        ids = {src[1] for _row, _family, src in per_obs}
        if len(ids) != 1:
            return None     # distinct free sigma per observable -> per-observable
        return f'noise_model = {family}, {param} = fit {ids.pop()}', False
    # All-constant sigma (the unit and column-mean cases are handled above): a uniform fix_at;
    # a different fixed sigma per observable is per-observable.
    uniq = {src[1] for _row, _family, src in per_obs}
    if len(uniq) != 1:
        return None     # distinct fixed sigma per observable -> per-observable
    return f'noise_model = {family}, {param} = fix_at {num(uniq.pop())}', False


def _per_observable_directives(per_obs, observable_id_to_column, is_column_mean=None):
    """A structural base objective + one ``noise_model <obs> = ...`` override per observable.

    The Boehm shape (ADR-0037): each observable has its own sigma source, so PyBNF expresses
    it as per-observable ``noise_model`` overrides (ADR-0021) layered over a whole-fit default.
    Under edition >= 2 a base objective is required (the override surface "accompanies" it,
    config.py), and since every observable is overridden the base is a structural placeholder
    -- ``objective = chi_sq`` (Gaussian, no free parameter, no data column required). Each
    override names the **column** the objective compares (the measurement-model column =
    ``observableId`` for an expression observable, else the model entity); a ``fit`` sigma binds
    its estimated parameter as a nuisance (ADR-0034), a ``fix_at`` a constant, a per-point
    placeholder reads the ``<col>_SD`` companion. An observable marked in ``is_column_mean``
    (its fixed sigma is exactly each experiment's own column mean, :class:`_ColumnMeans`, #894)
    takes a ``column_mean`` source instead of either fixed form."""
    is_column_mean = is_column_mean or {}
    lines = ['objective = chi_sq']   # whole-fit default; every observable overridden below
    for row, family, src in per_obs:
        param = _NOISE_MODEL_PARAM[family]
        column = observable_id_to_column[row.observable_id]
        kind = src[0]
        if is_column_mean.get(row.observable_id):
            lines.append(f'noise_model {column} = {family}, {param} = column_mean')
        elif kind == 'free':
            lines.append(f'noise_model {column} = {family}, {param} = fit {src[1]}')
        elif kind in ('formula', 'per_measurement'):
            # Both emit a 'formula' source; for 'per_measurement' the expression keeps its
            # row-varying placeholder, which config.py routes to PerMeasurementFormulaSigma
            # (the placeholder's row token comes from the measurement_params sidecar; ADR-0045).
            lines.append(f'noise_model {column} = {family}, {param} = formula {src[1]}')
        elif kind == 'prediction_formula':
            # A σ that scales with the simulated output (Raia's combined additive+proportional
            # error, ADR-0075): config.py routes 'prediction_formula' to a PredictionFormulaSigma
            # whose model-entity symbols read the current simulation, its coefficients the PSet.
            lines.append(f'noise_model {column} = {family}, {param} = prediction_formula {src[1]}')
        elif kind == 'constant':
            lines.append(
                f'noise_model {column} = {family}, {param} = fix_at {num(src[1])}')
        elif kind == 'placeholder':
            # Per-point _SD sigma: the Gaussian families read it (gaussian linear, lognormal
            # log10, lnnormal natural log); Laplace has no per-point _SD source (#407).
            if family == 'laplace':
                raise NotImplementedError(
                    f"Observable '{row.observable_id}': a per-point ({family}) placeholder "
                    f"noiseFormula has no native noise_model source yet (#407).")
            lines.append(f'noise_model {column} = {family}, {param} = read_exp_file _SD')
        else:  # defensive
            raise PybnfError(
                f"Observable '{row.observable_id}': unexpected sigma source kind {kind!r}.")
    return lines


def _approx(a, b):
    """Two sigmas are equal up to a relative tolerance (the column-mean comparison).

    Purely relative (#894): it only has to absorb the round-off of averaging the same numbers
    in another order. The former ``max(1, |b|)`` floor made it an absolute 1e-9 below 1, so
    for small-magnitude data (a mean of 1e-10, say) a sigma several times the mean matched."""
    return abs(a - b) <= 1e-9 * abs(b)


# ---------------------------------------------------------------------------
# Experiments: measurement groups + experiment rows -> conf experiment entries
# ---------------------------------------------------------------------------

# One reconstructed conf experiment. A record shared by :func:`_experiments`,
# :func:`_dose_response_experiments`, and :func:`_write_conf` -- a namedtuple (not a bare
# tuple) so the three sites bind by field name and a new field can't silently mis-align a
# positional unpack. ``preequilibrate`` (ADR-0052) is the unmeasured steady-state condition a
# pre-equilibration experiment equilibrates under before the ``condition:`` measurement period;
# ``None`` for a plain time course or a dose-response scan. ``equil_t_end`` is the fixed duration
# of that equilibration when the PEtab problem gives its leading period a finite start time
# ``-T`` rather than ``-inf`` (#896); ``None`` (the default) for a steady-state equilibration and
# for every experiment without one.
ImportedExperiment = namedtuple(
    'ImportedExperiment',
    ['name', 'condition', 'preequilibrate', 'data_files', 'model_location',
     'measparams_file', 't_end', 'equil_t_end'],
    defaults=(None,))


def _condition_and_preequilibrate(periods, name):
    """Resolve an experiment's measurement ``condition:`` and optional ``preequilibrate:`` from
    its experiments-table period rows (sorted by time) -- the inverse of Phase 2's
    :func:`~pybnf.petab.conditions.build_preequilibration_conditions` (ADR-0052).

    A single-period experiment is a plain time course: its sole ``conditionId`` is the
    measurement condition, no pre-equilibration. A **two-period** experiment whose leading
    period is a ``time = -inf`` steady state is a pre-equilibration: the ``-inf`` period's
    condition equilibrates the system unmeasured (the ``preequilibrate:`` state) before the
    ``time = 0`` measurement period's ``condition:`` (a blank ``conditionId`` there -> ``None``
    = a wash-out measured at the model default). Returns ``(condition, preequilibrate)``, each a
    condition name or ``None``.

    A two-period experiment whose leading period starts at a finite ``time = -T < 0`` and whose
    measurement period starts at exactly ``time = 0`` is a **fixed-duration** pre-equilibration
    (#896): the same ``(condition, preequilibrate)``, with the duration ``T`` read separately by
    :func:`_fixed_equilibration_time` (the conf's ``equil_t_end: T``). Any other finite leading
    period (one not followed by a period at exactly 0), an experiment of more than two periods, or
    a non-leading ``-inf`` raises :class:`NotImplementedError` rather than silently flattening the
    experiment to its last period (the pre-#442 bug).
    """
    if len(periods) <= 1:
        cid = periods[0].condition_id if periods else None
        return condition_name_from_id(cid), None
    if (len(periods) == 2 and math.isinf(periods[0].time) and periods[0].time < 0
            and math.isfinite(periods[1].time)):
        return (condition_name_from_id(periods[1].condition_id),
                condition_name_from_id(periods[0].condition_id))
    if _fixed_equilibration_time(periods) is not None:
        # A FIXED-duration equilibration (#896): a finite leading period at -T followed by the
        # measured period at exactly 0 -- `preequilibrate:` + `equil_t_end: T` (read by
        # _fixed_equilibration_time). The leading period must name its condition: a blank one
        # (equilibrate at the model defaults) has no `preequilibrate:` to carry the duration.
        pre = condition_name_from_id(periods[0].condition_id)
        if pre is None:
            raise NotImplementedError(
                f"Experiment '{name}' has a fixed-duration equilibration period (time "
                f"{periods[0].time}) with no condition. PyBNF carries an equilibration duration "
                f"on a 'preequilibrate:' condition, so an equilibration at the model defaults "
                f"has no PyBNF representation yet.")
        return condition_name_from_id(periods[1].condition_id), pre
    raise NotImplementedError(
        f"Experiment '{name}' has a {len(periods)}-period PEtab experiments-table structure "
        f"(times {[r.time for r in periods]}) the importer does not recover. Only a "
        f"single-period time course or a two-period pre-equilibration (a leading time=-inf "
        f"steady-state period + a finite measurement period, ADR-0052, or a leading finite "
        f"time=-T fixed-duration period + a measurement period at exactly time=0, #896) is "
        f"supported; experiments of more than two periods are deferred.")


def _fixed_equilibration_time(periods):
    """The fixed equilibration duration ``T`` of a two-period experiment whose leading period
    starts at a finite time ``-T < 0`` and whose measured period starts at exactly 0 -- the
    exporter's ``equil_t_end: T`` shape (#896, ``conditions.equilibration_period_time``) -- else
    ``None``. ``periods`` are the experiment's rows, sorted by time.

    PEtab v2 runs the leading period from ``-T`` until the next period starts, so only a next
    period at 0 makes ``T`` the equilibration's duration and the data times relative to the
    intervention, which is what PyBNF's measured phase assumes (its clock restarts at 0)."""
    if (len(periods) == 2 and math.isfinite(periods[0].time) and periods[0].time < 0
            and periods[1].time == 0):
        return equil_t_end_from_period_time(periods[0].time)
    return None


def _experiments(datas, experiment_rows, out_dir, model_location_of, param_bindings=None, *,
                 files):
    """Assemble the conf's experiments and write each one's ``.exp`` file(s).

    The set of experiments is the measurement groups (the replicate grids per
    ``(experimentId, modelId)``, ADR-0041); each experiment's replicate ``Data`` objects are
    written to ``<name>.exp`` (the first / only replicate) and ``<name>_rep<k>.exp`` (k>=2),
    all bound to the one experiment's ``data:`` list -- the inverse of the forward export,
    which stacks an experiment's replicate ``Data`` objects into repeated measurement rows
    (ADR-0039). The single-replicate case keeps the bare ``<name>.exp`` name, so the common
    round trip is byte-stable. Those are the preferred names: ``files`` (the import's one
    :class:`_DataFileNames` registry) moves any name another written file already holds, so no
    experiment's data can overwrite another's. The experiment's condition comes from its
    experiments-table period rows, grouped by experimentId: a single period gives the
    measurement condition (``cond_<c>`` -> ``c``; a blank ``conditionId`` or an absent row -> no
    condition, and so does the exporter's pins-only ``cond_wildtype``, which the importer has
    already blanked -- :func:`~pybnf.petab.conditions.drop_synthesized_wildtype`; a
    ``cond_wildtype`` with real targets keeps its literal id, #905); a two-period
    ``-inf``/finite pair recovers ``preequilibrate:`` + ``condition:`` (ADR-0052,
    :func:`_condition_and_preequilibrate`). A ``''`` experimentId is the "model as is" base time
    course (PEtab erased its name because the job had no fit-and-perturbed parameters); it is
    synthesized a name (:func:`_experiment_name`), which never reaches the PEtab output (it
    re-exports to ``''`` again) -- a name keyed on the modelId when set, so two wildtype
    experiments on different models stay distinct. Each experiment's model is the ``modelId`` on
    its rows: ``model_location_of`` maps it to the model file, emitted as a per-experiment
    ``model:`` field (omitted for a single-model job, whose modelId is ``''``).

    ``param_bindings`` (ADR-0045/0083) is the ``{(experiment_id, model_id): {column:
    {placeholder: {key: token}}}}`` per-measurement binding table, where a multi-replicate
    group's ``key`` is ``(replicate, time)``; a group with an entry also writes a
    ``<name>_measparams.tsv`` sidecar carrying its row-varying tokens, emitted as the experiment's
    ``measurement_params:`` field. Returns a list of :class:`ImportedExperiment`
    in measurement order; ``t_end`` is ``None`` for a time course (the dose-response scans
    append their own entries -- :func:`_dose_response_experiments`).
    """
    param_bindings = param_bindings or {}
    # Group the experiment rows into per-experimentId period lists, sorted by time -- the
    # multi-period structure a pre-equilibration experiment carries (ADR-0052). The pre-#442
    # flat {experiment_id: condition_id} map overwrote here, dropping the -inf period.
    periods_of = {}
    for row in experiment_rows:
        periods_of.setdefault(row.experiment_id, []).append(row)
    for rows in periods_of.values():
        rows.sort(key=lambda r: r.time)
    _refuse_simultaneous_conditions(periods_of, {eid for eid, _mid in datas})
    experiments = []
    for (eid, mid), group in datas.items():
        name = _experiment_name(eid, mid)
        condition, preequilibrate = _condition_and_preequilibrate(periods_of.get(eid, []), name)
        equil_t_end = _fixed_equilibration_time(periods_of.get(eid, []))   # #896
        if equil_t_end is not None:
            # A measurement inside the -T period has no PyBNF home (the equilibration is unmeasured).
            refuse_measurements_inside_fixed_equilibration(
                name, equil_t_end, (t for data in group for t in data[data.indvar]))
        model_location = model_location_of.get(mid)   # None for a single-model job (mid '')
        data_files = _write_replicate_exps(out_dir, files, name, group)
        measparams_file = None
        binding = param_bindings.get((eid, mid))
        if binding:
            measparams_file = files.claim(f'{name}_measparams', '.tsv')
            write_measurement_params(binding, out_dir / measparams_file)
        experiments.append(ImportedExperiment(
            name, condition, preequilibrate, data_files, model_location, measparams_file, None,
            equil_t_end=equil_t_end))
    return experiments


def _refuse_simultaneous_conditions(periods_of, measured):
    """Refuse a measured experiment that applies two or more conditions at the same time (#904).

    PEtab v2 writes such a period as several experiments-table rows sharing one
    ``(experimentId, time)``, and applies their conditions together. A PyBNF ``experiment:``
    applies one ``condition:`` (and one ``preequilibrate:``), so the shape has no PyBNF form
    here; refusing it names the cause instead of letting :func:`_condition_and_preequilibrate`
    read the rows as a sequence of periods. (A pre-equilibrated dose-response's wash + dose
    period is claimed before this, by ``reconstruct_preequilibrated_dose_responses``.)
    ``periods_of`` maps an experimentId to its time-sorted rows; only the ``measured`` ids are
    checked, since an experiment with no measurements is never written to the conf."""
    for eid, rows in periods_of.items():
        if eid not in measured:
            continue
        for time in sorted({r.time for r in rows}):
            cids = [r.condition_id for r in rows if r.time == time]
            if len(cids) > 1:
                raise NotImplementedError(
                    f"Experiment '{eid}' applies {len(cids)} conditions at the same time "
                    f"({num(time)}): {cids}. A PyBNF experiment applies one condition per "
                    f"period, so conditions applied together have no PyBNF representation. "
                    f"Merge their targets into a single PEtab condition.")


def _experiment_name(experiment_id, model_id):
    """The conf name of a time-course experiment: its experimentId, or for the ``''`` "model as
    is" experiment a synthesized name keyed on the modelId when set (ADR-0041)."""
    if experiment_id:
        return experiment_id
    return f'experiment_{model_id}' if model_id else 'experiment1'


def _experiment_names(datas, dose_responses, preequil_scans):
    """The conf name of every imported experiment, in the order the conf lists them: the time
    courses, then the dose-response scans, then the pre-equilibrated scans.

    Refuses two experiments that would take one name -- a scan whose stem equals another
    experiment's id, or one experimentId measured on two models. The conf names each experiment
    once (a repeated ``experiment:`` name fails to load), and its data files are named after it.
    """
    sources = {}
    for eid, mid in datas:
        where = f" on model '{mid}'" if mid else ''
        sources.setdefault(_experiment_name(eid, mid), []).append(
            f"PEtab experiment '{eid}'{where}" if eid else f"the unnamed experiment{where}")
    for kind, scans in (('dose-response scan', dose_responses),
                        ('pre-equilibrated dose-response scan', preequil_scans)):
        for scan in scans:
            where = f" on model '{scan['model_id']}'" if scan['model_id'] else ''
            sources.setdefault(scan['name'], []).append(f"the {kind} '{scan['name']}'{where}")
    clashes = {name: who for name, who in sources.items() if len(who) > 1}
    if clashes:
        name, who = next(iter(clashes.items()))
        raise PybnfError(
            f"{' and '.join(who)} would both import as the PyBNF experiment '{name}', and a "
            f"PyBNF job names each experiment once. Give them distinct experimentIds (a "
            f"dose-response scan takes its name from its experimentIds' '<name>_<i>' stem).")
    return list(sources)


class _DataFileNames:
    """The one registry of the file names an import writes into its output directory.

    Every experiment's first data file is ``<name>.exp``, its k-th replicate ``<name>_rep<k>.exp``
    and its per-measurement sidecar ``<name>_measparams.tsv`` (ADR-0039/0045). Those names are not
    unique on their own: an experiment whose experimentId is literally ``s_rep2`` and a replicated
    experiment ``s`` both want ``s_rep2.exp``, and ``S.exp`` and ``s.exp`` are one file on a
    case-insensitive filesystem. Before this registry the later write silently replaced the
    earlier one, and one experiment was fitted to the other's measurements (#903 review).

    The rule is deterministic and never overwrites. Every experiment's ``<name>.exp`` is claimed
    first, in conf order, so a primary data file keeps its natural name whenever that name is
    free; replicate files and sidecars are claimed after, as each experiment is written. A name
    already taken -- compared case-insensitively, and including the model files -- gets the
    first free ``_<n>`` suffix (``s_rep2_2.exp``, ``s_rep2_3.exp``, ...). The experiment's
    ``data:`` / ``measurement_params:`` fields name the file actually written."""

    def __init__(self, experiment_names, reserved=()):
        self._taken = {name.casefold() for name in reserved}
        self._primary = {name: self.claim(name, '.exp') for name in experiment_names}

    def claim(self, stem, ext):
        """Take ``<stem><ext>``, or the first free ``<stem>_<n><ext>`` (n = 2, 3, ...)."""
        name, n = f'{stem}{ext}', 2
        while name.casefold() in self._taken:
            name, n = f'{stem}_{n}{ext}', n + 1
        self._taken.add(name.casefold())
        return name

    def exp_files(self, name, count):
        """The ``count`` data-file names of experiment ``name``: its claimed ``<name>.exp`` and a
        fresh claim for each further replicate."""
        return [self._primary[name]] + [self.claim(f'{name}_rep{k}', '.exp')
                                        for k in range(2, count + 1)]


def _dose_response_experiments(dose_responses, out_dir, model_location_of, *, files):
    """Build the conf experiment entries for the reconstructed dose-response scans (ADR-0046).

    Each scan's swept-axis :class:`~pybnf.data.Data` replicate grids are written to ``<name>.exp``
    and ``<name>_rep<k>.exp`` (:func:`_write_replicate_exps`; column 0 the swept parameter, so
    ``config._infer_experiment_type`` reads it as a parameter_scan -- no ``type:`` field needed).
    A steady-state scan (``scan_time`` inf) carries ``t_end = None`` (it runs to steady state,
    PEtab time=inf); a finite scan carries that endpoint. Returns the same
    :class:`ImportedExperiment` records as :func:`_experiments` (condition / preequilibrate /
    measparams are always ``None`` -- a dose is the scan axis, not a named condition, a scan is
    never a pre-equilibration, and a dose-response carries no per-measurement sidecar)."""
    experiments = []
    for dr in dose_responses:
        name = dr['name']
        data_files = _write_replicate_exps(out_dir, files, name, dr['datas'])
        model_location = model_location_of.get(dr['model_id'])
        t_end = None if math.isinf(dr['scan_time']) else dr['scan_time']
        experiments.append(ImportedExperiment(name, None, None, data_files, model_location,
                                              None, t_end))
    return experiments


def _preequilibrated_dose_response_experiments(scans, out_dir, model_location_of, *, files):
    """Build the conf experiment entries for the reconstructed pre-equilibrated dose-response scans
    (#477; ADR-0062) -- the two-period sibling of :func:`_dose_response_experiments`.

    Each scan's swept-axis :class:`~pybnf.data.Data` replicate grids are written to ``<name>.exp``
    and ``<name>_rep<k>.exp`` (:func:`_write_replicate_exps`; column 0 the swept parameter, so
    ``config._infer_experiment_type`` reads it as a parameter_scan -- no ``type:`` field needed),
    and the experiment carries its ``preequilibrate:`` (the ``-inf`` pre-equilibration condition)
    and its optional measurement ``condition:`` (the wash). A steady-state scan (``scan_time``
    inf) carries ``t_end = None``; a finite scan carries that endpoint. Returns
    :class:`ImportedExperiment` records (``measparams`` always ``None`` -- a dose-response
    carries no per-measurement sidecar)."""
    experiments = []
    for s in scans:
        name = s['name']
        data_files = _write_replicate_exps(out_dir, files, name, s['datas'])
        model_location = model_location_of.get(s['model_id'])
        t_end = None if math.isinf(s['scan_time']) else s['scan_time']
        experiments.append(ImportedExperiment(
            name, s['wash'], s['preequilibrate'], data_files, model_location, None, t_end,
            equil_t_end=s['equil_t_end']))
    return experiments


def _write_replicate_exps(out_dir, files, name, datas):
    """Write experiment ``name``'s replicate grids -- a time course's or a dose-response scan's
    (#903) -- under the names ``files`` allocates (``<name>.exp``, ``<name>_rep<k>.exp`` unless
    taken, :class:`_DataFileNames`) and return those names, in order."""
    data_files = files.exp_files(name, len(datas))
    for data_file, data in zip(data_files, datas):
        _write_exp(out_dir / data_file, data)
    return data_files


def _refuse_fixed_equilibration_of_time_dependent_models(experiments, models, model_texts):
    """Refuse a fixed-duration equilibration period (``equil_t_end``) on a model that reads time.

    The import peer of ``export._refuse_fixed_equilibration_of_time_dependent_models`` (#896): the
    PEtab leading period runs on ``[-T, 0]``, but PyBNF would run the ``equil_t_end: T`` phase on
    ``[0, T]`` and restart the clock at 0, so for a model whose rates, functions, or events read
    the time the imported job would simulate a different protocol. ``models`` are the problem's
    ``model_files`` entries; ``model_texts`` maps each location to its text. A single-model job's
    experiments carry no ``model_location`` (their model is the sole one)."""
    language_of = {m['location']: (m['language'] or 'bngl').lower() for m in models}
    for exp in experiments:
        if exp.equil_t_end is None:
            continue
        location = exp.model_location or models[0]['location']
        reads = model_time_reads(model_texts[location], language_of[location])
        if reads:
            raise NotImplementedError(
                f"Experiment '{exp.name}' starts with a fixed-duration equilibration period "
                f"(time -{num(exp.equil_t_end)}) on model '{location}', which reads the simulation "
                f"time ({'; '.join(reads)}). PEtab runs that period from t = "
                f"-{num(exp.equil_t_end)} to 0, but PyBNF would run it from t = 0 to "
                f"{num(exp.equil_t_end)} and restart the clock at 0, so the imported job would "
                f"simulate a different protocol (#896).")


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

def _declare_unperturbed_conditions(conditions, experiments, condition_rows, experiment_rows,
                                    synthesized, measured_ids):
    """Add to ``conditions`` (as empty perturbation lists, written ``perturbations: none``) the
    conditions the imported ``experiments`` apply that change nothing (#906, ADR-0150), and
    return the experiments.

    Two kinds:

    * ``synthesized`` -- the name :func:`~pybnf.petab.conditions.name_unperturbed_equilibrations`
      gave every ``time = -inf`` period that applied no condition. A PyBNF condition belongs to
      one model, so when experiments on more than one model apply it, each model after the first
      gets its own copy under the next free name, and its experiments are renamed to match.
    * a named PEtab condition whose every row is a base pin ``p = p__REF``, the identity once
      ``p__REF`` is renamed back to ``p``, so :func:`conditions_from_rows` recovered nothing from
      it. It is the model as is wherever it is applied: as ``preequilibrate:`` an equilibration
      with nothing changed, as ``condition:`` the same as none.

    Only an id a measured experiment (``measured_ids``) applies counts, and only if it has rows.
    A name whose applied id has no rows at all is left undeclared, even when another id of the
    same name has rows but is applied by no measured experiment: the experiments table then
    applies a condition the problem never defines, and loading the conf says so. The periods
    whose reading as the model as is PEtab does not share were refused before this
    (:func:`~pybnf.petab.conditions.refuse_inexact_unperturbed_periods`).
    """
    applied_ids = {r.condition_id for r in experiment_rows if r.experiment_id in measured_ids}
    defined = {condition_name_from_id(r.condition_id) for r in condition_rows
               if r.condition_id in applied_ids} - {None}
    for exp in experiments:
        for name in (exp.condition, exp.preequilibrate):
            if name and name != synthesized and name not in conditions and name in defined:
                conditions[name] = []
    if synthesized is None:
        return experiments
    models = []
    for exp in experiments:
        if exp.preequilibrate == synthesized and exp.model_location not in models:
            models.append(exp.model_location)
    taken = condition_names_and_ids(condition_rows, experiment_rows) | set(conditions)
    name_of_model = {}
    for location in models:
        name = synthesized if not name_of_model else free_condition_name(taken)
        taken.add(name)
        name_of_model[location] = name
        conditions[name] = []
    return [exp._replace(preequilibrate=name_of_model[exp.model_location])
            if exp.preequilibrate == synthesized else exp for exp in experiments]


def _write_conf(path, *, model_filenames, job_type, objective_directives, free_param_lines,
                conditions, experiments, measurement_models, method, method_overrides,
                settings, multi, fixed_overrides=()):
    """Write one new-era (edition 2) ``.conf``: the recovered problem + the supplied
    run-recipe (``job_type``, per-experiment ``method:``, required settings).

    ``model_filenames`` is the list of the problem's model files (one ``model:`` line each;
    a multi-model job also tags every experiment with its ``model:`` field -- ADR-0041).
    ``objective_directives`` is the list of recovered objective lines -- either a single
    ``objective = <token>`` / whole-fit ``noise_model = ...`` line, or a base objective plus
    per-observable ``noise_model <obs> = ...`` overrides (:func:`_objective_directives`).
    ``measurement_models`` is the list of ``(observableId, formula)`` expression observables,
    emitted as ``observable: <id>, formula: <expr>`` measurement-model lines (ADR-0036).
    ``fixed_overrides`` lists the estimate=false values written into the model copies
    (``_FixedOverride``, #907); each is named in the header, which is otherwise unchanged."""
    stem = f'imported_{job_type}' if multi else 'imported'
    lines = [
        '# Imported from a PEtab v2 problem by pybnf.petab.import_job (#407).',
        '# The PEtab *problem* (parameters/priors, observables/noise, measurements,',
        '# conditions) is recovered exactly and round-trips through a re-export. The',
        '# run-recipe below (job_type + algorithm settings, the per-experiment method:,',
        '# and output/verbosity) is SUPPLIED, not recovered: PEtab is a problem spec with',
        '# no home for the method, so it is not part of the round-trip identity.',
    ]
    if fixed_overrides:
        lines += [
            '#',
            '# Fixed model parameters (#907): parameters.tsv fixes each parameter below',
            '# (estimate = false) at its nominalValue, which PEtab gives precedence over the',
            '# model file. The model file disagreed, so the imported copy of the model was',
            "# edited to the table's value; each edited line is marked with a comment there.",
            *[f'#   {o.parameter_id} = {num(o.nominal_value)} in {o.location} '
              f'(the model file had {_model_value_text(o.model_value)})'
              for o in fixed_overrides],
        ]
    lines += [
        '',
        f'output_dir=output/{stem}',
        'edition = 2',
        '',
        *[f'model: {mf}' for mf in model_filenames],
        f'job_type = {job_type}',
        *objective_directives,
    ]
    # Expression observables: a measurement-model formula evaluated post-simulation (the
    # observation layer, ADR-0036), not a model-file edit. The model is carried verbatim, save
    # the marked estimate=false overrides the header lists (#907).
    for obs_id, formula in measurement_models:
        lines.append(f'observable: {obs_id}, formula: {formula}')
    lines.append('')
    # A PyBNF condition belongs to ONE model: the fitter attaches its MutationSet to a
    # specific model and requires a `model:` ref on the condition when the job declares
    # more than one model (config.py::_load_conditions). PEtab conditions are model-
    # agnostic (no modelId column, ADR-0041), so under multiple models recover each
    # condition's owning model from the experiment(s) that apply it -- via `condition:`
    # or, for the equilibration period, `preequilibrate:`. A condition applied by
    # experiments on *different* models has no PyBNF representation (a condition can't
    # span models) -> refuse with a clear boundary error rather than emit a conf the
    # fitter rejects. Single-model jobs carry no `model:` on a condition (byte-identical).
    multi_model = len(model_filenames) > 1
    cond_models = {}
    for exp in experiments:
        for cname in (exp.condition, exp.preequilibrate):
            if cname:
                cond_models.setdefault(cname, set()).add(exp.model_location)
    for name, perts in conditions.items():
        # An empty list is a condition that changes nothing (#906, ADR-0150).
        pert_str = ', '.join(_render_perturbation(var, op, val) for var, op, val in perts) or 'none'
        model_field = ''
        if multi_model:
            locs = {loc for loc in cond_models.get(name, set()) if loc}
            if len(locs) > 1:
                raise NotImplementedError(
                    f"Condition {name!r} is applied by experiments on different models "
                    f"({sorted(locs)}). A PyBNF condition belongs to a single model, so a "
                    f"PEtab condition shared across models has no PyBNF representation.")
            if locs:
                model_field = f', model: {next(iter(locs))}'
        lines.append(f'condition: {name}{model_field}, perturbations: {pert_str}')
    for exp in experiments:
        sim_method = method_overrides.get(exp.name, method)
        # A pre-equilibration experiment (ADR-0052) leads with its unmeasured steady-state
        # `preequilibrate:` condition, then the measured `condition:` -- mirroring the fitter
        # grammar / receptor_v2.conf authoring order (`preequilibrate:` before `condition:`).
        preequil_field = f', preequilibrate: {exp.preequilibrate}' if exp.preequilibrate else ''
        cond_field = f', condition: {exp.condition}' if exp.condition else ''
        model_field = f', model: {exp.model_location}' if exp.model_location else ''
        # A fixed-endpoint dose-response scan's endpoint time (ADR-0046); a steady-state scan
        # and a time course carry none (the scan runs to steady state / the data drives the grid).
        tend_field = f', t_end: {num(exp.t_end)}' if exp.t_end is not None else ''
        # A fixed-duration equilibration (#896): the leading period's finite start time -T.
        equil_field = (f', equil_t_end: {num(exp.equil_t_end)}'
                       if exp.equil_t_end is not None else '')
        # The row-varying per-measurement binding sidecar (ADR-0045), when this experiment
        # carries one; config.py attaches it to the experiment's exp Data.
        mp_field = f', measurement_params: {exp.measparams_file}' if exp.measparams_file else ''
        data_field = ', '.join(f'data: {f}' if i == 0 else f
                               for i, f in enumerate(exp.data_files))
        lines.append(
            f'experiment: {exp.name}{preequil_field}{cond_field}{model_field}, '
            f'method: {sim_method}{tend_field}{equil_field}{mp_field}, {data_field}')
    lines.append('')
    lines.extend(free_param_lines)
    lines.append('')
    for key in ('population_size', 'max_iterations', 'verbosity'):
        lines.append(f'{key} = {settings[key]}')
    path.write_text('\n'.join(lines) + '\n')


def _render_perturbation(var, op, val):
    """Render one recovered condition perturbation as a conf ``perturbations:`` token.

    A **species** ``setConcentration`` target (a BNGL pattern, ADR-0062) is emitted with its
    pattern quoted (it carries commas) and its verbatim value (a number or a param-expression),
    the value itself quoted only when it carries a comma (the grammar's ``cond_species_val``
    convention). A **parameter reference** value (a per-condition estimated initial condition,
    ADR-0076) is a *string* naming a free parameter, emitted verbatim (``I0_ = I0_CA``). A plain
    **parameter** target renders as ``<var> <op> <num(val)>`` (``val`` is a float)."""
    if is_species_target(var):
        value = f'"{val}"' if ',' in str(val) else str(val)
        return f'"{var}" {op} {value}'
    if isinstance(val, str):
        return f'{var} {op} {val}'   # a free-parameter reference (ADR-0076)
    return f'{var} {op} {num(val)}'


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
# problem.yaml reader (hand-parsed, dependency-free; refuses what it cannot read)
# ---------------------------------------------------------------------------

# The six table keys of a PEtab v2 problem.yaml. The schema types each one as a LIST of files,
# and libpetab reads every file in the list (#902).
_TABLE_FILE_KEYS = ('parameter_files', 'observable_files', 'measurement_files',
                    'condition_files', 'experiment_files', 'mapping_files')

# Every top-level key the PEtab v2 schema allows (its additionalProperties is false). ``id`` and
# ``extensions`` carry nothing the importer reads.
_PROBLEM_YAML_KEYS = (*_TABLE_FILE_KEYS, 'format_version', 'id', 'model_files', 'extensions')

# A leading character that makes a YAML scalar something other than a plain name: a nested flow
# collection, an anchor, alias or tag, a block scalar, or a reserved indicator.
_YAML_NON_PLAIN = frozenset('[]{}&*!|>%@`')


def read_problem_yaml(path):
    """Read a PEtab v2 ``problem.yaml`` without a YAML library.

    Returns a dict with the table-file lists (``parameter_files`` / ``observable_files`` /
    ``measurement_files`` / ``condition_files`` / ``experiment_files`` / ``mapping_files``, each
    holding every file listed, in order) and a ``models`` list -- one ``{model_id, location,
    language}`` entry per ``model_files`` entry, in declaration order (one or many, ADR-0041).
    For single-model convenience the first model is also surfaced as ``model_file`` /
    ``model_id`` / ``model_language``.

    A small indentation-aware scan reads the shapes a problem file is written in: our own
    writer's (``key:`` then two-space-indented ``- item`` lines, ``model_files`` last), the
    column-0 ``- item`` lists ``petab.v2.petab1to2`` writes, keys in any order, and the one-line
    flow list ``key: [a.tsv, b.tsv]`` (``[]`` included). Items may be quoted, ``#`` comments
    are dropped, a model entry's fields other than location and language are passed over with
    everything nested beneath them, and a leading directive or ``---`` and a closing ``...``
    are allowed. Whatever else it meets raises ``PybnfError`` naming the line or key rather than
    being skipped: a scalar where a list belongs, a flow list continued over several lines, a
    flow-form ``model_files`` entry, a key the PEtab v2 schema does not allow, a key or model
    given twice, a file listed twice under one key, a ``format_version`` other than 2 (#902).
    The scan used to skip what it did not recognize, so ``condition_files: [conditions.tsv]``
    read as no condition table at all.

    This is a pure *reader*: it records each model's ``language`` but does not enforce a
    policy on it. The supported-language scope (BNGL or SBML, ADR-0036) is enforced by the
    importer (:func:`_require_supported_model`), not here.
    """
    files = {k: [] for k in _TABLE_FILE_KEYS}
    models = []         # [{model_id, location, language}, ...] in declaration order
    current = None      # the model entry being filled (set by a `<modelId>:` line)
    model_indent = None   # the indentation of the `<modelId>:` lines
    field_indent = None   # the indentation of the current model entry's own fields
    field = None          # the model entry's field whose value nested lines belong to

    seen_keys, unknown_keys = set(), []
    format_version = None
    section = None      # the current top-level *_files key (block list items follow)
    in_model = False    # inside the model_files: block
    in_other = False    # inside a key whose nested content the importer does not read
    started = ended = False
    # utf-8-sig drops a byte-order mark, which PyYAML (libpetab's reader) also ignores.
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = _strip_yaml_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if not started and (stripped == '---' or stripped.startswith('%')):
            continue            # a directive (%YAML, %TAG) or the document-start marker
        if stripped == '...' and indent == 0:
            ended = True        # the document-end marker PyYAML writes with explicit_end
            continue
        if ended:
            raise PybnfError(f"problem.yaml at {path} holds more than one YAML document: "
                             f"the line {stripped!r} follows the '...' end marker.")
        started = True
        is_item = stripped == '-' or stripped.startswith(('- ', '-\t'))
        # A column-0 list item (`- item`) is YAML-legal and is exactly what the official
        # petab v1->v2 converter emits (`petab.v2.petab1to2`); it belongs to the *current*
        # section, not a new key, so it must not reset the scan. Only a non-list line at
        # column 0 opens/closes a section -- our own writer indents its list items, so
        # honoring the unindented shape too makes the reader a strict superset of both.
        if indent == 0 and not is_item:
            section, in_model, in_other, current = None, False, False, None
            key, colon, rest = stripped.partition(':')
            if not colon:
                raise PybnfError(f"problem.yaml at {path}: cannot read the line {stripped!r} "
                                 f"(expected '<key>: <value>').")
            key, rest = _yaml_scalar(key, 'a top-level key', path), rest.strip()
            if key in seen_keys:
                raise PybnfError(f"problem.yaml at {path} gives the key '{key}' twice.")
            seen_keys.add(key)
            if key in files:
                if rest:
                    files[key] = _yaml_flow_list(rest, key, path)
                else:
                    section = key         # block `- item` lines follow
            elif key == 'model_files':
                if rest and rest != '{}':
                    raise PybnfError(
                        f"problem.yaml at {path}: model_files is written in YAML flow form "
                        f"({rest!r}), which this reader does not read. Write each model as an "
                        f"indented block: '<modelId>:' with 'location:' and 'language:' lines "
                        f"beneath it.")
                in_model, model_indent = True, None
            elif key == 'format_version':
                format_version = _yaml_scalar(rest, key, path)
            else:
                # `id` / `extensions` carry nothing the importer reads; any other key is not
                # PEtab v2 and is refused after the scan (a v1 problem is named as such first).
                in_other = True
                if key not in _PROBLEM_YAML_KEYS:
                    unknown_keys.append(key)
            continue
        if section is not None:
            if not is_item:
                raise PybnfError(
                    f"problem.yaml at {path}: '{section}' must be a list of files, one "
                    f"'- <file>' line each; cannot read the line {stripped!r}.")
            files[section].append(_yaml_scalar(stripped[1:], section, path))
        elif in_model:
            key, colon, rest = stripped.partition(':')
            key, rest = key.strip(), rest.strip()
            if model_indent is None or indent <= model_indent:
                # A `<modelId>:` line: the entry's fields follow on deeper-indented lines.
                if model_indent is not None and indent < model_indent:
                    raise PybnfError(f"problem.yaml at {path}: the model_files entry "
                                     f"{stripped!r} is not indented like the entries before it.")
                model_indent = indent
                if is_item or not colon or rest:
                    raise PybnfError(
                        f"problem.yaml at {path}: cannot read the model_files entry "
                        f"{stripped!r}. Write each model as '<modelId>:' with indented "
                        f"'location:' and 'language:' lines beneath it (a list, or a "
                        f"flow-form entry such as '{{location: ..., language: ...}}', is not "
                        f"read).")
                model_id = _yaml_scalar(key, 'model_files', path)
                if any(m['model_id'] == model_id for m in models):
                    raise PybnfError(
                        f"problem.yaml at {path} declares the model '{model_id}' twice.")
                current = {'model_id': model_id, 'location': None, 'language': None}
                models.append(current)
                field_indent = field = None
            elif field_indent is None or indent == field_indent:
                # One of the entry's own fields. The schema allows fields beyond location and
                # language on a model entry; they carry nothing the importer reads.
                field_indent, field = indent, (key if colon else None)
                if key in ('location', 'language') and colon:
                    current[key] = _yaml_scalar(
                        rest, f'model_files: {current["model_id"]}: {key}', path)
            elif indent > field_indent:
                # The nested value of the field above. It is never this model's location or
                # language, even when it holds a `location:` key of its own; only the value
                # of location or language itself continuing on this line is unreadable.
                if field in ('location', 'language'):
                    raise PybnfError(
                        f"problem.yaml at {path}: the {field} of model '{current['model_id']}' "
                        f"continues on the line {stripped!r}. Write it on one line.")
            else:
                raise PybnfError(
                    f"problem.yaml at {path}: the line {stripped!r} of model "
                    f"'{current['model_id']}' is not indented like the fields before it.")
        elif not in_other:
            raise PybnfError(
                f"problem.yaml at {path}: cannot read the line {stripped!r}; it is not part of "
                f"a list of files or of model_files.")

    _require_problem(files, models, path, format_version, unknown_keys)
    first = models[0]
    return {**files, 'models': models, 'model_file': first['location'],
            'model_id': first['model_id'], 'model_language': first['language']}


def _strip_yaml_comment(line):
    """``line`` without its YAML comment: a ``#`` at the start of the line or after
    whitespace, outside a quoted scalar (a ``#`` inside a plain word, as in ``a#b.tsv``, is
    part of the word, as it is in YAML)."""
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            if quote == '"' and ch == '\\':
                i += 1                      # skip the escaped character
            elif ch == quote:
                quote = None
        elif ch in '"\'' and (i == 0 or line[i - 1] in ' \t[,'):
            quote = ch                      # a quote opens only at the start of a scalar
        elif ch == '#' and (i == 0 or line[i - 1] in ' \t'):
            return line[:i]
        i += 1
    return line


def _yaml_scalar(text, key, path):
    """One YAML scalar (a file name, a model id or field, a key), with its quotes removed.

    A plain or quoted string is returned as its value. An empty entry, an unterminated quote,
    a backslash escape, and any construct that is not a plain string (a nested list, an
    anchor, a block scalar, ...) raise ``PybnfError`` naming ``key`` -- never a guess."""
    text = text.strip()
    if not text:
        raise PybnfError(f"problem.yaml at {path}: '{key}' has an empty entry.")
    if text[0] in '"\'':
        quote = text[0]
        if len(text) < 2 or text[-1] != quote:
            raise PybnfError(
                f"problem.yaml at {path}: '{key}' has an unterminated quoted entry {text!r}.")
        inner = text[1:-1]
        if quote == "'":
            return inner.replace("''", "'")
        if '\\' in inner:
            raise PybnfError(
                f"problem.yaml at {path}: '{key}' has the entry {text!r}, whose backslash "
                f"escape this reader does not interpret. Write the name without escapes.")
        return inner
    if text[0] in _YAML_NON_PLAIN:
        raise PybnfError(
            f"problem.yaml at {path}: '{key}' has the entry {text!r}, which uses YAML syntax "
            f"this reader does not read. Write a plain file name or id.")
    return text


def _yaml_flow_list(text, key, path):
    """The items of a one-line YAML flow list, ``[a.tsv, 'b.tsv']`` (``[]`` is empty).

    A value that is not a complete one-line ``[...]`` -- a bare scalar (the PEtab v2 schema
    types every ``*_files`` key as a list) or a flow list continued on the next line -- raises
    ``PybnfError`` naming ``key``. A trailing comma is allowed, as YAML allows it."""
    if not (text.startswith('[') and text.endswith(']')):
        raise PybnfError(
            f"problem.yaml at {path}: '{key}' must be a list of files, written either as "
            f"'- <file>' lines beneath the key or as '[<file>, <file>]' on the key's own line; "
            f"got {text!r}.")
    items, start, quote = [], 1, None
    for i in range(1, len(text) - 1):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in '"\'' and not text[start:i].strip():
            quote = ch
        elif ch == ',':
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:-1])
    if not items[-1].strip():
        items.pop()         # `[]`, or a trailing comma
    return [_yaml_scalar(item, key, path) for item in items]


def _require_problem(files, models, path, format_version=None, unknown_keys=()):
    """Refuse a ``problem.yaml`` that is not a readable PEtab v2 problem (#902)."""
    if format_version is not None and format_version.split('.')[0] != '2':
        raise PybnfError(
            f"problem.yaml at {path} declares format_version {format_version}, but the "
            f"importer reads PEtab v2 problems.",
            hint="Convert a PEtab v1 problem first with pybnf.petab.petab1to2_preserve_scale.")
    if unknown_keys:
        raise PybnfError(
            f"problem.yaml at {path} has the key(s) {unknown_keys}, which PEtab v2 does not "
            f"define; the importer would ignore them. The allowed keys are "
            f"{list(_PROBLEM_YAML_KEYS)}.")
    for key, listed in files.items():
        repeated = sorted(name for name, n in Counter(listed).items() if n > 1)
        if repeated:
            raise PybnfError(
                f"problem.yaml at {path} lists {repeated} more than once under {key}, which "
                f"would read the same rows twice. List each file once.")
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
    ``.bngl`` nor an ``.xml`` is edited for an observable (ADR-0036); the one edit is a marked
    ``estimate = false`` override (#907, ADR-0149). Any other model language (e.g.
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
