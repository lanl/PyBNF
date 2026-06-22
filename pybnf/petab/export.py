"""PEtab v2 exporter: a PyBNF/BNGL job -> PEtab v2 artifacts (#407/#423; ADR-0025/27/28).

The **exporter-first** direction of the PEtab interop (ADR-0025): a working PyBNF
BNGL job and a native ``.conf`` are *read* and serialized to a PEtab v2 problem
(``parameters.tsv`` / ``observables.tsv`` / ``measurements.tsv`` / ``conditions.tsv`` /
``experiments.tsv`` / ``problem.yaml`` + a PEtab-clean copy of the model), rather than
*generating* BNGL from a declarative PEtab spec (the harder importer direction, deferred).
The reverse asset mappings live beside their importer twins
(``parameters.petab_parameter_row``, ``observables.petab_observable_row``,
``measurements.measurement_rows_from_data``, ``conditions.build_*``); this module is the
*disposable* glue: it reads the job (the stdlib ``ploop`` config parser, a focused BNGL
block reader, and :class:`pybnf.data.Data` for the ``.exp``) and writes the files.

**Why a function is the measurement model.** A fitted ``.exp`` column matches a BNGL
**observable** *or* a **function** (PyBNF forces ``print_functions=>1``), and the
function is usually the measurement model. So an observable column exports to
``observableFormula = <name>`` and a function column to ``observableFormula =
<name>`` too -- always the *bare model name*, with the function carried verbatim in
the model file (ADR-0025). PEtab ids are prefixed (``obs_``/``func_`` for
observables, the unprefixed model name for parameters) to keep the PEtab-id namespace
disjoint from the model-entity namespace.

**New-era only (ADR-0028/0031): export is transcription.** PEtab v2 interop is a new-era
feature, and the exporter reads *only* the new-era surface -- both the objective
(``objective`` / ``noise_model``, never the retired ``objfunc``, no implicit default;
ADR-0031) **and** the data linkage (``model:`` / ``experiment:`` / ``data:`` /
``condition:`` / ``observable:``; ADR-0028). An ``experiment:`` *is* a PEtab Experiment
(experimentId = the experiment name) carrying its ``data:`` replicates as measurement
rows; a ``condition:`` *is* a PEtab Condition; an ``observable:`` renames a data column
before classification. ``export_job`` **refuses** a legacy (edition 1) job
(:func:`_require_modern_edition`) and a legacy data linkage -- ``model = X : Y.exp`` /
``mutant`` / ``param_scan`` (:func:`_require_new_era_data`) -- rather than reverse-mapping
it. The gate is on the exporter alone; the fitter still runs legacy confs unchanged.

**Conditions/experiments (ADR-0027/0028).** A ``condition:`` referenced by an experiment
becomes a PEtab Condition/Experiment via the surrogate-base ``<p>__REF`` rename of a
fit-and-perturbed parameter (see :mod:`pybnf.petab.conditions`,
:func:`~pybnf.petab.conditions.build_experiment_conditions`); a shared condition emits its
rows once. A dose-response (parameter scan) experiment takes the dual shape (ADR-0046):
each dose becomes a Condition setting the swept parameter + an Experiment, measured at the
scan time (``inf`` for the steady-state default => PEtab time=inf, or a finite ``t_end:``).

The objective and prior surfaces map to PEtab as far as PEtab v2 can express them
(ADR-0023/0031 reversed): the Gaussian/Laplace likelihoods with a ``_SD``-column,
fixed, or column-mean sigma (``chi_sq``/``sos``/``sod``/``ave_norm_sos``), and the
``uniform``/``log-uniform``/``normal``/``laplace`` prior families with their log forms.
A BNGL (``.bngl``) or SBML (``.xml``) model is exported in its own native language
(ADR-0040): a BNGL model PEtab-cleaned, an SBML model carried byte-verbatim with its
observables emitted as ``observableFormula`` expressions from the conf measurement-model
layer (the mirror of the ADR-0036 import). A job may declare **more than one model**
(ADR-0041): each ``experiment:`` names the model it simulates, that model's id is stamped
on the experiment's measurement rows (the ``modelId`` link), free parameters bind across
the union of every model's ids, and ``problem.yaml`` lists every model in its own language
(BNGL + SBML may mix). Everything else raises ``NotImplementedError`` (the boundary is in
code, not silent): an objective PEtab cannot represent (``neg_bin*`` -- removed from v2;
``lognormal`` -- log10 vs PEtab natural log; a free-parameter or relative sigma;
``direct_pass``/``kl``/``wasserstein``); the no-prior ``var``/``logvar``; a ``.con``/``.prop``
Constraint; an Antimony (``.ant``) model. The
oracle is petab's full ``default_validation_tasks`` via ``Problem.from_yaml`` + the native
``BnglModel`` loader (ADR-0026), wired into the tests; see ADR-0025/0027/0028/0036/0040.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import edition
from ..data import Data
from ..objective import _OBJECTIVE_DESUGAR
from ..parse import ploop
from ..printing import PybnfError
from ..pset import FreeParameter
from ._bngl import parse_model as parse_bngl_model
from ._sbml import parse_model as parse_sbml_model
from .conditions import (
    build_dose_response_conditions,
    build_experiment_conditions,
    surrogate_name,
    write_condition_table,
    write_experiment_table,
)
from ._measurement_params import read_measurement_params
from .formula import bngl_body_to_petab_math
from .measurements import (
    dose_response_measurement_rows,
    measurement_rows_from_data,
    write_measurement_table,
)
from .observables import petab_observable_row, write_observable_table
from .parameters import (
    EXPORTABLE_PRIOR_KEYWORDS,
    petab_parameter_row,
    write_parameter_table,
)

logger = logging.getLogger(__name__)

# A noise-model family token (ADR-0031's noise_model grammar / ``_OBJECTIVE_DESUGAR``)
# -> the PEtab v2 noiseDistribution it maps to (ADR-0023 reversed). The LINEAR Gaussian
# / Laplace likelihoods map (``normal`` is the Gaussian alias); the other families are
# explicit boundaries handled in ``_resolve_noise``: ``neg_bin`` was removed from PEtab
# v2, and PyBNF's ``lognormal`` is log10 whereas PEtab's ``log-normal`` is natural log (a
# deferred sigma scale-conversion).
_FAMILY_TOKEN_TO_PETAB_DISTRIBUTION = {
    'gaussian': 'normal', 'normal': 'normal', 'laplace': 'laplace'}

# Free-parameter declaration keywords (the ``(keyword, name)`` tuple keys ``ploop``
# emits). Only ``uniform_var`` exports in chunk 1; the rest raise.
_VAR_DECL = re.compile(r'(_var$|^var$|^logvar$)')

# A legacy ``<name>__FREE`` bind-by-id marker. New-era BNGL binds free parameters by id
# (ADR-0034), so a model carrying this token was not modernized; the exporter refuses it
# rather than ship a ``v1__FREE`` symbol into PEtab (where it would dangle).
_FREE_TOKEN = re.compile(r'\w+__FREE')

# A per-measurement placeholder in a noiseFormula (``noiseParameter1`` / ``observableParameter2``):
# its presence marks a row-varying ``PerMeasurementFormulaSigma`` sigma whose per-row value comes
# from the binding-table sidecar, distinct from a constant ``FormulaSigma`` (ADR-0045). Mirrors
# ``objective._PLACEHOLDER_IN_FORMULA`` / ``import_._PLACEHOLDER``.
_PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d')


# ---------------------------------------------------------------------------
# The exporter driver
# ---------------------------------------------------------------------------

def export_job(conf_path, out_dir, inline_functions=False):
    """Export the PyBNF job at ``conf_path`` to a PEtab v2 problem in ``out_dir``.

    Reads the job's data/conditions/observables from the **new-era surface** (ADR-0028):
    a ``model:`` declaration, named ``experiment:`` lines carrying their ``data:`` files
    (a PEtab Experiment; multiple files = replicates), ``condition:`` perturbations (a
    PEtab Condition), and ``observable:`` column-header overrides. Writes
    ``parameters.tsv``, ``observables.tsv``, ``measurements.tsv``, a PEtab-clean copy of
    the BNGL model, ``problem.yaml``, and -- when some experiment applies a condition --
    ``conditions.tsv`` / ``experiments.tsv``. Returns the ``out_dir`` path.

    ``inline_functions`` (default ``False``) is the opt-in expression mode (ADR-0035):
    when set, a fitted **function** column emits its body as an ``observableFormula``
    expression (translated to PEtab math, requires the ``pybnf[petab]`` extra) instead of
    the bare model name, producing a model-portable problem and the round-trip oracle the
    importer's synthesis is graded against. The default stays bare-name, lossless,
    byte-stable, and ``petab``-free; an observable column is never inlined.

    New-era only (ADR-0028 Chunk 5c, "refuse legacy everything"): a job that binds data
    the legacy way (``model = X : Y.exp`` / ``mutant`` / ``param_scan``) is refused, as is
    a legacy-edition job (:func:`_require_modern_edition`). Raises ``NotImplementedError``
    at every documented boundary and ``PybnfError`` for a malformed/unsupported job (see
    the module docstring).
    """
    conf_path = Path(conf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf = _read_conf_dict(conf_path)
    _require_modern_edition(conf)
    models = _resolve_models(conf)
    languages = {mf: _model_language(mf) for mf in models}
    _require_new_era_data(conf, models)
    noise = _resolve_noise(conf)
    per_obs_noise = _resolve_per_observable_noise(conf)
    free_params = _free_parameters_from_conf(conf)
    # A registry of per-language model views (ADR-0040/0041): a job may mix BNGL + SBML,
    # each read once and threaded through the language-agnostic classification below.
    registry = {mf: _read_model(mf, conf_path.parent / mf, languages[mf]) for mf in models}
    # Observation-layer nuisance free parameters (ADR-0034/0044/0045): a measurement scale, a
    # noise coefficient, or a row-varying per-row sigma -- a free parameter referenced by a
    # measurement-model / noise formula or a binding-table token but NOT a model entity. The
    # bind-by-id check admits them as unbound estimated parameters (else they read as typos);
    # the rest must still bind to a model id.
    model_ids = set().union(*(set(v.parameters) for v in registry.values()))
    nuisances = (_referenced_nuisance_symbols(conf, conf_path, noise, per_obs_noise)
                 & {fp.name for fp in free_params}) - model_ids
    free_to_model = _resolve_free_to_model(free_params, registry, models, nuisances)
    fit_model_params = set(free_to_model.values()) & model_ids

    (observable_rows, measurement_rows, condition_rows, experiment_rows,
     surrogate_params) = _export_new_era(
        conf, conf_path, models, registry, noise, per_obs_noise, fit_model_params,
        inline_functions)

    parameter_rows = _parameter_rows(
        free_params, free_to_model, surrogate_params, registry, models)

    write_parameter_table(parameter_rows, out_dir / 'parameters.tsv')
    write_observable_table(observable_rows, out_dir / 'observables.tsv')
    write_measurement_table(measurement_rows, out_dir / 'measurements.tsv')
    if condition_rows:
        write_condition_table(condition_rows, out_dir / 'conditions.tsv')
    if experiment_rows:
        write_experiment_table(experiment_rows, out_dir / 'experiments.tsv')
    # Each model is emitted in its own native language (ADR-0040): a BNGL model is
    # PEtab-cleaned (drop 'begin actions'); an SBML model is carried byte-verbatim (the
    # measurement model lives in the observables table, never a model-file edit -- ADR-0036).
    for mf in models:
        view, language = registry[mf], languages[mf]
        model_text = clean_model_for_petab(view.text) if language == 'bngl' else view.text
        (out_dir / Path(mf).name).write_text(model_text)
    # One model_files entry per model (ADR-0041), modelId = the file stem, in declaration
    # order (so a re-export reproduces the same problem.yaml model_files ordering).
    model_yaml = [(Path(mf).stem, Path(mf).name, languages[mf]) for mf in models]
    write_problem_yaml(out_dir / 'problem.yaml', model_yaml,
                       has_conditions=bool(condition_rows),
                       has_experiments=bool(experiment_rows))
    return out_dir


# ---------------------------------------------------------------------------
# New-era surface reading (ADR-0028): export becomes transcription
# ---------------------------------------------------------------------------

def _has_new_era_data(conf):
    """True iff the job binds data via the new-era ``experiment:`` surface (ADR-0028).

    A fully new-era conf introduces data through ``('experiment', name)`` entries (a PEtab
    Experiment carrying its ``data:``), never the legacy ``model = X : Y.exp`` linkage.
    """
    return any(isinstance(k, tuple) and len(k) == 2 and k[0] == 'experiment'
               for k in conf)


def _require_new_era_data(conf, models):
    """Refuse a legacy data linkage -- the PEtab v2 exporter reads only the new-era data
    surface (ADR-0028 Chunk 5c, "refuse legacy everything").

    The exporter is already new-era-gated on the *objective* (``_require_modern_edition``);
    now that ADR-0028's data surface exists, the data linkage is held to the same standard.
    A job must introduce data through named ``experiment:`` lines (a PEtab Experiment
    carrying its ``data:``); the legacy filename->suffix binding (``model = X : Y.exp``),
    ``mutant`` lines, and ``param_scan`` actions are refused rather than silently read.
    Mixing the two (a ``mutant``/``param_scan`` line, or data on any ``model =`` line,
    alongside ``experiment:``) is refused too, so a legacy line is never silently dropped.
    The gate is on the *exporter* only; the fitter still runs legacy confs unchanged.
    """
    legacy_data = any(conf.get(mf) for mf in models)  # model = X : Y.exp data list
    legacy_features = [k for k in ('mutant', 'param_scan') if k in conf]
    if not _has_new_era_data(conf):
        raise NotImplementedError(
            "The PEtab v2 exporter reads the new-era data surface; this job binds data "
            "the legacy way (model = X : Y.exp / mutant / param_scan), which is refused "
            "(ADR-0028, 'refuse legacy everything'). Re-author it on the new-era surface: "
            "declare the model with 'model:', bind data through a named 'experiment:' (one "
            "'data:' file per replicate), and write perturbations as 'condition:' lines. "
            "The fitter still runs the legacy form; only export requires the new surface.")
    if legacy_data or legacy_features:
        legacy = (['data on the legacy model = X : Y.exp line'] if legacy_data else []) \
            + [f"a '{k}' line" for k in legacy_features]
        raise NotImplementedError(
            f"This job mixes the new-era 'experiment:' surface with legacy data linkage "
            f"({', '.join(legacy)}). Use the new-era surface exclusively -- move all data "
            f"into 'experiment:'/'data:' and all perturbations into 'condition:' -- so no "
            f"legacy line is silently ignored on export (ADR-0028, #423).")


def _export_new_era(conf, conf_path, models, registry, noise, per_obs_noise,
                    fit_model_params, inline_functions=False):
    """Read a job's data/conditions/observables from the **new-era surface** (ADR-0028).

    Export is *transcription*: an ``experiment:`` is a PEtab Experiment (experimentId =
    the experiment name) carrying its ``data:`` replicates as measurement rows; an
    ``observable:`` line renames a data column to a model entity before classification.
    ``models`` is the ordered list of model files and ``registry`` their per-language views
    (ADR-0041): each experiment names the model it simulates, that model's id is stamped on
    its measurement rows' ``modelId`` (omitted when the job is single-model), and a column
    is classified against its experiment's model. ``noise`` is the whole-fit base and
    ``per_obs_noise`` the ``{column: (dist, verb, arg)}`` per-observable overrides
    (ADR-0021/0045): a column with an override takes its own sigma source, the rest the base.
    ``inline_functions`` is threaded to :func:`_observable_rows` (ADR-0035 inlining). Returns
    ``(observable_rows, measurement_rows, condition_rows, experiment_rows, surrogate_params)``.

    A ``condition:`` referenced by an experiment becomes a PEtab Condition (the
    surrogate-base machinery of ADR-0027, generalized by
    :func:`~pybnf.petab.conditions.build_experiment_conditions`): a fit-and-perturbed
    parameter is renamed to ``<p>__REF`` in the parameter table and pinned in every
    experiment's Condition. A **parameter-scan** (dose-response) experiment takes the
    dual shape (ADR-0046): each dose of its swept-axis ``.exp`` becomes a Condition
    setting the swept parameter + an Experiment, measured at the scan time (``inf`` for the
    steady-state default => PEtab time=inf, or a finite ``t_end:``), via
    :func:`~pybnf.petab.conditions.build_dose_response_conditions` +
    :func:`~pybnf.petab.measurements.dose_response_measurement_rows`. With no referenced
    conditions the surrogate set is empty, so a single wildtype time course is byte-identical
    to the chunk-1 base.
    """
    experiments = _read_experiments(conf, conf_path, models)
    overrides = _read_observable_overrides(conf)
    all_datas = [d for exp in experiments for d in exp['datas']]
    _apply_observable_overrides(all_datas, overrides)

    measurement_models = _read_measurement_models(conf)
    observable_rows, column_to_observable_id = _observable_rows(
        experiments, registry, noise, per_obs_noise, inline_functions, measurement_models)

    # Time-course vs dose-response (parameter_scan) experiments take different PEtab shapes
    # (ADR-0046): a time course is one Experiment over a referenced Condition; a dose-response
    # is N Conditions (each sets the swept parameter to one dose) + N Experiments measured at
    # the scan time (inf for steady state). They build independently and concatenate.
    tc_experiments = [exp for exp in experiments if exp['type'] == 'time_course']
    dr_experiments = [exp for exp in experiments if exp['type'] == 'parameter_scan']

    conditions = _read_conditions(conf, models, registry)
    referenced = {exp['condition'] for exp in tc_experiments if exp['condition'] is not None}
    undefined = referenced - set(conditions)
    if undefined:
        raise PybnfError(
            f"Experiment(s) reference undefined condition(s) {sorted(undefined)}; define "
            f"each with a 'condition:' line.")
    unused = set(conditions) - referenced
    if unused:
        # An unused condition emits no PEtab rows (the fitter would not apply it either);
        # skip it with a debug log rather than warning (ADR-0028 Chunk 5, decision 4).
        logger.debug("Conditions defined but referenced by no experiment (skipped): %s",
                     sorted(unused))

    condition_rows, experiment_rows, surrogate_params, experiment_to_id = \
        build_experiment_conditions(
            [(exp['name'], exp['condition']) for exp in tc_experiments],
            conditions, fit_model_params, lambda v: _nominal_of(registry, v))

    # Per-point numeric noiseParameters are emitted only when a column's sigma comes from a
    # data column (the read_exp_file placeholder source); a fixed / column-mean / formula sigma
    # is carried inline in noiseFormula, so the measurement export must not read _SD then (it
    # would leave a noiseParameters override with no placeholder to bind to). With per-observable
    # overrides the suffix is **per column** (ADR-0045): each column uses its own sigma source
    # (its override, else the whole-fit base) to decide whether it reads a _SD companion.
    def _sd_suffix_for(col):
        _dist, verb, arg = per_obs_noise.get(col, noise)
        return arg if verb == 'read_exp_file' else None
    sd_suffix = {col: _sd_suffix_for(col) for col in column_to_observable_id}

    # The modelId link (ADR-0041): each experiment stamps its model's stem onto its
    # measurement rows. Single-model -> '' (the column is dropped on write, byte-stable).
    multi_model = len(models) > 1
    measurement_rows = []
    for exp in tc_experiments:
        eid = experiment_to_id[exp['name']]
        model_id = Path(exp['model']).stem if multi_model else ''
        # Each replicate Data contributes its own rows under the one experiment (PEtab
        # models replicates as repeated rows -- no need to pre-stack as config.py does). A
        # row-varying placeholder's per-row token comes from the experiment's measurement_params
        # sidecar (ADR-0045), shared across the replicate Datas (each measures the same cells).
        for data in exp['datas']:
            cmap = {c: o for c, o in column_to_observable_id.items() if c in data.cols}
            measurement_rows += measurement_rows_from_data(
                data, cmap, experiment_id=eid, sd_suffix=sd_suffix, model_id=model_id,
                measurement_params=exp['measurement_params'])

    # Dose-response (ADR-0046): each dose row of the swept-axis .exp becomes its own Condition
    # (setting the swept parameter) + Experiment, and the observable columns become measurements
    # at the scan time (inf => steady state). The swept-parameter column (the data's indvar) is
    # the scan axis, not a measurement, so it is dropped from the column map.
    for exp in dr_experiments:
        stem = exp['name']
        model_id = Path(exp['model']).stem if multi_model else ''
        scan_time = exp['scan_time']
        data0 = exp['datas'][0]
        swept_param = data0.indvar if data0.indvar is not None else _independent_variable(data0)
        dose_values = [float(v) for v in data0[swept_param]]
        dr_conditions, dr_experiment_rows, experiment_ids = build_dose_response_conditions(
            stem, swept_param, dose_values, scan_time)
        condition_rows += dr_conditions
        experiment_rows += dr_experiment_rows
        for data in exp['datas']:
            cmap = {c: o for c, o in column_to_observable_id.items()
                    if c in data.cols and c != swept_param}
            measurement_rows += dose_response_measurement_rows(
                data, cmap, experiment_ids, scan_time, sd_suffix=sd_suffix, model_id=model_id)
    return observable_rows, measurement_rows, condition_rows, experiment_rows, \
        surrogate_params


def _read_experiments(conf, conf_path, models):
    """Read + resolve the new-era ``experiment:`` entries from the raw ``ploop`` dict.

    Each ``('experiment', name)`` entry is ``{'data': [files], 'condition': c?, 'model':
    mf?, 'type': t?, 'method': m?, 't_end': t?, 'measurement_params': mp?}``. ``models`` is the
    ordered list of the job's model files. Returns a list (declaration order) of dicts
    ``{'name', 'condition', 'model': model_file, 'datas': [Data, ...], 'type', 'scan_time',
    'measurement_params': table?}`` -- the ``data:`` files read as individual
    :class:`~pybnf.data.Data` replicates (PEtab models replicates as repeated measurement
    rows, so they are not pre-stacked), each experiment's resolved model
    (:func:`_resolve_experiment_model`, ADR-0041), the inferred ``type`` (``'time_course'`` or
    ``'parameter_scan'``), the dose-response ``scan_time`` (``inf`` for the steady-state
    default, a finite ``t_end:`` otherwise; ``None`` for a time course -- ADR-0046), and its
    row-varying per-measurement binding table read from the ``measurement_params:`` sidecar
    (``{column: {placeholder: {time: token}}}``, or ``None`` when none -- ADR-0045). Raises:

    * the ambiguous-model error if an experiment names no model but the job has more than
      one (mirrors ``config.py::_resolve_experiment_model``);
    * a not-yet-supported boundary for a parameter_scan that also names a ``condition:``
      (a dose-response already makes each dose its own condition -- ADR-0046);
    * a constraint refusal for non-``.exp`` data: BPSL ``.con``/``.prop`` constraints are
      PyBNF-native with no core-PEtab representation, so an experiment carrying them cannot
      be exported (the fitter still runs it -- ADR-0028 addendum).
    """
    stem_to_model = {Path(mf).stem: mf for mf in models}
    experiments = []
    for key, fields in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 'experiment'):
            continue
        name = key[1]
        model_file = _resolve_experiment_model(name, fields.get('model'), models,
                                               stem_to_model)
        data_files = fields.get('data', [])
        if not data_files:
            raise PybnfError(f"Experiment '{name}' declares no 'data:' files.")
        non_exp = [f for f in data_files if not f.endswith('.exp')]
        if non_exp:
            raise NotImplementedError(
                f"Experiment '{name}' carries BPSL constraint data ({non_exp}). PyBNF "
                f"constraints (.con/.prop) are a native qualitative-fitting feature with no "
                f"core-PEtab v2 representation, so this experiment cannot be exported "
                f"(ADR-0028). The fitter still runs it; only export is refused. Drop the "
                f"constraint file(s) from 'data:' to export the quantitative measurements "
                f"alone.")
        datas = [Data(file_name=str(conf_path.parent / f)) for f in data_files]
        exp_type = _experiment_type(name, datas[0], fields.get('type'))
        # A parameter_scan (dose-response) experiment's measurement time is its scan endpoint
        # (ADR-0046): inf for the steady-state default (PEtab time=inf), or a finite ``t_end:``.
        # A time course derives its grid from the data, so ``t_end:`` is inert there.
        scan_time = None
        if exp_type == 'parameter_scan':
            t_end = fields.get('t_end')
            scan_time = float(t_end) if t_end is not None else float('inf')
            if fields.get('condition') is not None:
                raise NotImplementedError(
                    f"Experiment '{name}' is a parameter_scan that also names a condition "
                    f"('{fields['condition']}'). A dose-response already makes each dose its "
                    f"own condition (ADR-0046); combining it with a named condition has no "
                    f"export route yet.")
        measurement_params = None
        mp_file = fields.get('measurement_params')
        if mp_file:
            measurement_params = read_measurement_params(conf_path.parent / mp_file)
        experiments.append({'name': name, 'condition': fields.get('condition'),
                            'model': model_file, 'datas': datas, 'type': exp_type,
                            'scan_time': scan_time,
                            'measurement_params': measurement_params})
    return experiments


def _resolve_experiment_model(name, ref, models, stem_to_model):
    """The model file an experiment simulates (ADR-0041), the export peer of
    ``config.py::_resolve_experiment_model``.

    With an explicit ``model:`` ref, resolve it by filename stem (the ``model_files`` key);
    an unknown ref is a typo. With no ref, default to the sole model when the job declares
    exactly one; under more than one model an unnamed experiment is ambiguous -- the
    exporter requires the ``model:`` field rather than guessing which model produced the
    data (matching the fitter's rule).
    """
    if ref is not None:
        stem = Path(ref).stem
        if stem not in stem_to_model:
            raise PybnfError(
                f"Experiment '{name}' names model '{ref}', but the job declares no model "
                f"with id '{stem}' (declared model ids: {sorted(stem_to_model)}).")
        return stem_to_model[stem]
    if len(models) == 1:
        return models[0]
    raise PybnfError(
        f"Experiment '{name}' does not name a model, but the job declares {len(models)} "
        f"models ({models}). Add 'model: <file>' to the experiment to say which model it "
        f"simulates (ADR-0041).")


def _experiment_type(name, data, explicit_type):
    """Infer ``'time_course'`` vs ``'parameter_scan'`` from a ``Data``'s independent
    variable (``time`` => time_course; otherwise the indvar names a swept parameter =>
    parameter_scan), unless ``type:`` states it. Mirrors
    ``config.py::_infer_experiment_type`` (a scan exports each dose as a steady-state
    Condition/Experiment -- ADR-0046)."""
    if explicit_type is not None:
        t = explicit_type.lower()
        if t in ('time_course', 'timecourse'):
            return 'time_course'
        if t in ('parameter_scan', 'param_scan', 'parameterscan'):
            return 'parameter_scan'
        raise PybnfError(
            f"Experiment '{name}' has unrecognized type '{explicit_type}' (use "
            f"'time_course').")
    indvar = data.indvar if data.indvar is not None else _independent_variable(data)
    return 'time_course' if indvar.lower() == 'time' else 'parameter_scan'


def _read_observable_overrides(conf):
    """The new-era ``observable: <entity>, column: <header>`` overrides as
    ``{entity: header}`` (ADR-0028 Chunk 4) -- the renames applied before classification."""
    return {k[1]: v for k, v in conf.items()
            if isinstance(k, tuple) and len(k) == 2 and k[0] == 'observable'}


def _read_measurement_models(conf):
    """The new-era ``observable: <id>, formula: <expr>`` measurement models as
    ``{id: formula}`` (ADR-0036). A measurement model is a PEtab observableFormula evaluated
    post-simulation by the observation layer; on export its ``.exp`` column classifies as the
    measurement model (not a model entity), and its formula is emitted as the
    ``observableFormula`` verbatim -- the inverse of the importer's measurement-model line, so
    an expression observable round-trips export -> import -> re-export."""
    return {k[1]: v for k, v in conf.items()
            if isinstance(k, tuple) and len(k) == 2 and k[0] == 'measurement'}


def _apply_observable_overrides(datas, overrides):
    """Rename each ``<header>`` data column (and its ``<header>_SD`` companion) to the
    model ``<entity>`` across all ``datas`` so the column classifies against a model
    observable/function -- mirroring ``config.py::_load_observables``. Global: a ``Data``
    lacking a header is skipped (it just does not measure that observable); a header
    present in **no** ``Data`` is a typo -> ``PybnfError`` (listing the columns present)."""
    for entity, header in overrides.items():
        found = False
        for data in datas:
            if header in data.cols:
                data.rename_column(header, entity)
                found = True
            sd = f'{header}_SD'
            if sd in data.cols:
                data.rename_column(sd, f'{entity}_SD')
                found = True
        if not found:
            present = sorted({c for data in datas for c in data.cols})
            raise PybnfError(
                f"Observable override 'observable: {entity}, column: {header}' names data "
                f"column '{header}', but no experimental data file contains it (columns "
                f"present: {present}). Check for a typo in the column name.")


def _read_conditions(conf, models, registry):
    """Read + validate the new-era ``condition:`` entries from the raw ``ploop`` dict.

    Each ``('condition', name)`` entry is ``(model_ref_or_None, [(var, op, val_str), ...])``
    (a named set of parameter perturbations -- a PyBNF Mutant = a PEtab Condition). Returns
    ``{condition_name: [(var, op, float(val)), ...]}``. A PEtab condition is model-agnostic
    (no modelId column; ADR-0041), so a perturbation target must be a parameter / compartment
    of **some** model in the job (the union); an explicit ``model:`` ref, when given, must
    name a declared model. The single-model job validates against its one model exactly as
    before."""
    union_params = set().union(*(set(v.parameters) for v in registry.values()))
    union_comparts = set().union(*(set(v.compartment_names) for v in registry.values()))
    stem_to_model = {Path(mf).stem: mf for mf in models}
    conditions = {}
    for key, value in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 'condition'):
            continue
        name = key[1]
        model_ref, perts = value
        if model_ref is not None and Path(model_ref).stem not in stem_to_model:
            raise PybnfError(
                f"Condition '{name}' is declared for model '{model_ref}', but the job "
                f"declares no model with id '{Path(model_ref).stem}' (declared model ids: "
                f"{sorted(stem_to_model)}).")
        muts = []
        for var, op, val in perts:
            if var not in union_params and var not in union_comparts:
                raise PybnfError(
                    f"Condition '{name}' perturbs '{var}', which is not a parameter or "
                    f"compartment of any model in the job (a PEtab condition target must "
                    f"be a model entity).")
            muts.append((var, op, float(val)))
        conditions[name] = muts
    return conditions


# ---------------------------------------------------------------------------
# Scope resolution (the documented boundaries)
# ---------------------------------------------------------------------------

def _resolve_models(conf):
    """Return the job's model files in declaration order (ADR-0041).

    A BNGL (``.bngl``) or SBML (``.xml``) model is exported in its own native language
    (ADR-0040, dispatched by :func:`_model_language`); any other extension raises there. A
    job may declare one or many models; the new-era ``model:`` declarations accumulate into
    ``conf['model']`` in declaration order (a legacy ``model = X : Y`` job has none -- it is
    refused downstream by :func:`_require_new_era_data` -- so fall back to the model set).
    The model id is the file **stem** (the ``model_files`` key); two files sharing a stem
    would collide on that key and the output filename, so a stem collision raises."""
    ordered = list(dict.fromkeys(conf.get('model', [])))
    if not ordered:
        ordered = sorted(conf.get('models', set()))
    if not ordered:
        raise PybnfError(
            "The job declares no model; add a 'model: <file>' declaration (ADR-0028).")
    stems = {}
    for mf in ordered:
        _model_language(mf)   # validate the extension is a supported language
        stem = Path(mf).stem
        if stem in stems:
            raise PybnfError(
                f"Models '{stems[stem]}' and '{mf}' share the file stem '{stem}', which "
                f"would collide on the PEtab modelId and the exported filename. Rename one "
                f"so each model has a distinct stem (ADR-0041).")
        stems[stem] = mf
    return ordered


def _require_modern_edition(conf):
    """Refuse a legacy-edition job: PEtab v2 interop is a new-era (``edition >= 2``)
    feature (Bill's call, ADR-0031). A legacy conf names its objective with the retired
    ``objfunc`` key and binds data through the filename->suffix linkage; the exporter
    reads only the modern surface, so it requires the conf to have opted into the new era
    rather than reverse-mapping legacy syntax. Gates the *exporter* only -- the fitter
    still runs legacy confs unchanged."""
    ed = edition.resolve_edition(conf.get('edition'))
    if not edition.is_modern(ed):
        raise NotImplementedError(
            "The PEtab v2 exporter requires a new-era config (edition >= 2); this job is "
            f"legacy (edition {ed}). PEtab v2 interop is a new-era feature: add "
            f"'edition = {edition.CURRENT_EDITION}' and name the objective on the modern "
            "surface ('objective = <name>' or 'noise_model = <family>, ...') instead of "
            "the legacy 'objfunc' key (ADR-0031, #423).")


def _resolve_noise(conf):
    """The job's whole-fit noise model as ``(noiseDistribution, sigma_verb, sigma_arg)``.

    Modern-only (``export_job`` has already required ``edition >= 2``): the objective is
    named on the ADR-0031 surface -- a whole-fit ``noise_model = <family>, ...`` line or
    the named ``objective`` token -- with **no legacy** ``objfunc`` and **no implicit
    default**, mirroring ``config.py``'s modern ``_load_obj_func`` branch. The resolved
    objective is reduced to one ``(family, {param: (verb, arg)}, location)`` tuple and
    reversed to PEtab.

    The result is the **whole-fit base** applied to every column that has no per-observable
    ``noise_model <obs> = ...`` override; the overrides are resolved separately by
    :func:`_resolve_per_observable_noise` (the additive seam, ADR-0021/0045) and a column
    with one uses its own sigma source instead.

    Raises ``NotImplementedError`` at every PEtab boundary, never a silent default:

    * a column-joint ``profile_objective`` (``kl`` / ``wasserstein``) -- it scores the
      whole column's shape, not a per-observation likelihood, so it has no PEtab
      observable-noise representation;
    * no objective, or more than one global objective key (no implicit default);
    * a ``mean``-centered noise model -- PEtab takes the prediction as the median for
      every family;
    * an objective with no per-point noise model (``score`` / unknown token);
    * a family PEtab v2 cannot express (``neg_bin`` -- removed; ``lognormal`` -- log10 vs
      PEtab natural log, a deferred sigma scale-conversion).
    """
    if conf.get('profile_objective') is not None:
        raise NotImplementedError(
            f"profile_objective = {conf['profile_objective']!r} is a column-joint "
            f"objective (kl / wasserstein): it scores the whole column's shape, not a "
            f"per-observation likelihood, so it has no PEtab observable-noise "
            f"representation (ADR-0031, #423).")

    whole_fit = conf.get(('noise_model', None))
    has_objective = conf.get('objective') is not None
    if whole_fit is not None and has_objective:
        raise PybnfError(
            "Specify exactly one global objective: this job has both a whole-fit "
            "'noise_model = ...' line and an 'objective = ...' key.")
    if whole_fit is not None:
        family_token, fields, location = whole_fit          # modern whole-fit line
    elif has_objective:
        token = conf['objective']
        if token not in _OBJECTIVE_DESUGAR:
            raise NotImplementedError(
                f"objective = {token!r} has no per-point PEtab noise model: 'score' (no "
                f"likelihood) and any unknown token are not PEtab observable noise. "
                f"Per-point objectives: {sorted(_OBJECTIVE_DESUGAR)} (ADR-0031, #423).")
        family_token, fields, location = _OBJECTIVE_DESUGAR[token](conf)
    else:
        raise NotImplementedError(
            "No objective is named. A new-era (edition >= 2) job must name its objective "
            "explicitly -- there is no implicit default. Set 'objective = <name>' or "
            "'noise_model = <family>, ...' (ADR-0031, #423).")

    return _reduce_noise_spec(family_token, fields, location, 'the whole-fit noise model')


def _resolve_per_observable_noise(conf):
    """The per-observable ``noise_model <obs> = ...`` overrides as ``{column:
    (noiseDistribution, sigma_verb, sigma_arg)}`` (ADR-0021/0045) -- the additive companion to
    :func:`_resolve_noise`'s whole-fit base.

    Each ``('noise_model', <col>)`` key (``<col>`` not ``None``) is the parsed
    ``(family_token, {param: (verb, arg)}, location)`` spec for one observable **column** (the
    model entity / measurement-model column the objective scores), reduced through the same
    :func:`_reduce_noise_spec` boundaries as the whole-fit case (a ``mean`` location or a family
    PEtab v2 cannot express raises). Empty when the job declares no override -- then every column
    takes the whole-fit base and the export is byte-identical to the pre-per-observable output.
    The inverse of the importer's :func:`~pybnf.petab.import_._per_observable_directives`, the
    config side that consumes these (``objective._build_noise_overrides``)."""
    overrides = {}
    for key, value in conf.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] == 'noise_model' \
                and key[1] is not None:
            column = key[1]
            family_token, fields, location = value
            overrides[column] = _reduce_noise_spec(
                family_token, fields, location, f"the 'noise_model {column}' override")
    return overrides


def _reduce_noise_spec(family_token, fields, location, where):
    """Reduce one parsed noise spec ``(family_token, {param: (verb, arg)}, location)`` to
    ``(noiseDistribution, sigma_verb, sigma_arg)``, raising the PEtab boundaries shared by the
    whole-fit base (:func:`_resolve_noise`) and the per-observable overrides
    (:func:`_resolve_per_observable_noise`): a ``mean``-centered location (PEtab is median-only)
    and a family PEtab v2 cannot express (``neg_bin`` removed; ``lognormal`` is log10 vs PEtab's
    natural ``log-normal``). ``where`` names the spec in the error message."""
    if location == 'mean':
        raise NotImplementedError(
            f"{where} is mean-centered (location = mean); PEtab v2 takes the prediction as "
            f"the distribution median for every noise family, so mean centering has no PEtab "
            f"representation (ADR-0031, #423). Use median.")
    distribution = _FAMILY_TOKEN_TO_PETAB_DISTRIBUTION.get(family_token.lower())
    if distribution is None:
        raise NotImplementedError(
            f"{where}: the '{family_token}' noise family cannot be expressed in PEtab v2: "
            f"neg_bin was removed from v2, and PyBNF's lognormal is log10 while PEtab's "
            f"log-normal is natural log (the sigma scale-conversion is a later chunk). "
            f"ADR-0023/0031, #423.")
    (_param, (verb, arg)), = fields.items()
    return distribution, verb, arg


def _independent_variable(data):
    """The header of a wide :class:`~pybnf.data.Data`'s column 0 (``time`` or the swept axis)."""
    return min(data.cols, key=data.cols.get)


# ---------------------------------------------------------------------------
# Observable + parameter rows
# ---------------------------------------------------------------------------

def _observable_rows(experiments, registry, noise, per_obs_noise, inline_functions=False,
                     measurement_models=None):
    """Classify each fitted column across all experiments as a model observable, a model
    function, or a conf measurement model, and map it to a PEtab observable row.

    A column is classified against the model of **each** experiment that measures it
    (ADR-0041): the observables table has no per-model namespace, so a column shared across
    models must classify identically in each (the same kind, hence the same observableId and
    formula) -- a column that is an observable in one model and a function in another is a
    real conflict and raises. A column is gathered once, in first-appearance order across the
    experiments' (override-renamed) ``datas``, so the observables table covers the whole job.
    ``noise`` is the whole-fit base ``(noiseDistribution, sigma_verb, sigma_arg)`` from
    :func:`_resolve_noise` and ``per_obs_noise`` the ``{column: (dist, verb, arg)}``
    per-observable overrides (ADR-0021/0045): each column's noise (its family + sigma source)
    is its override if present, else the base, resolved across every experiment's data (it can
    depend on the column's data, e.g. a ``column_mean`` sigma).

    ``inline_functions`` (ADR-0035) emits a **function** column's body as an
    ``observableFormula`` expression instead of the bare name -- the opt-in path that
    generates the importer's round-trip oracle; the default keeps every column bare.
    ``measurement_models`` (``{id: formula}``, ADR-0036) are conf-declared measurement models
    (model-agnostic): a column matching one is emitted with that formula as its
    ``observableFormula`` and its id verbatim (the inverse of the importer's ``observable: ...
    formula:`` line)."""
    measurement_models = measurement_models or {}
    all_datas = [d for exp in experiments for d in exp['datas']]

    # Gather the fitted columns in first-appearance order, each tagged with the model
    # file(s) that measure it (distinct, declaration order). A measurement-model column is
    # model-agnostic (it never touches a model's entities), so its models list is inert.
    columns = []
    column_models = {}
    for exp in experiments:
        mf = exp['model']
        for data in exp['datas']:
            indvar = data.indvar if data.indvar is not None else _independent_variable(data)
            for col in sorted(data.cols, key=data.cols.get):
                if col == indvar or col.endswith('_SD'):
                    continue
                if col not in column_models:
                    columns.append(col)
                    column_models[col] = []
                if mf not in column_models[col]:
                    column_models[col].append(mf)

    observable_rows = []
    column_to_observable_id = {}
    for col in columns:
        classes = [(mf, _classify_column(col, registry[mf], mf, measurement_models,
                                         inline_functions))
                   for mf in column_models[col]]
        distinct = {kind_formula for _mf, kind_formula in classes}
        if len(distinct) != 1:
            detail = '; '.join(f"{mf} -> {kind}" for mf, (kind, _f) in classes)
            raise PybnfError(
                f"Exp column '{col}' classifies inconsistently across the models that "
                f"measure it ({detail}). PEtab's observables table has no per-model "
                f"namespace, so a column shared across models must mean the same observable "
                f"in each (ADR-0041).")
        kind, formula = classes[0][1]
        # A column's noise is its per-observable override if one is declared, else the
        # whole-fit base (ADR-0021/0045); the override carries its own family + sigma source.
        distribution, verb, arg = per_obs_noise.get(col, noise)
        noise_source = _noise_source_for_column(verb, arg, col, all_datas)
        row = petab_observable_row(col, kind, distribution, noise_source,
                                   observable_formula=formula)
        observable_rows.append(row)
        column_to_observable_id[col] = row.observable_id
    if not observable_rows:
        raise PybnfError(
            "The job's experiment data has no fittable observable/function columns "
            "(only an independent variable and/or _SD columns).")
    return observable_rows, column_to_observable_id


def _classify_column(col, model, model_file, measurement_models, inline_functions):
    """Classify one fitted column against one model view, returning ``(kind, formula)``
    (the formula is ``None`` unless it is a measurement model or an inlined function body).

    A conf-declared measurement model wins first (it is model-agnostic, ADR-0036); else the
    column must be a model observable or function (its bare name being the
    ``observableFormula``). A column matching nothing in this model raises -- naming the
    model so a multi-model job points at the right namespace (ADR-0041)."""
    if col in measurement_models:
        return 'measurement', measurement_models[col]   # the conf observableFormula, verbatim
    if col in model.observable_names:
        return 'observable', None
    if col in model.function_names:
        formula = _inlined_formula(col, 'function', model, model_file) \
            if inline_functions else None
        return 'function', formula
    raise PybnfError(
        f"Exp column '{col}' matches no observable, function, or measurement model in model "
        f"'{model_file}' (its observables: {sorted(model.observable_names)}; functions: "
        f"{sorted(model.function_names)}; measurement models: {sorted(measurement_models)}).")


def _inlined_formula(col, kind, model, model_file):
    """The ``observableFormula`` for a column under inlining mode (ADR-0035), or ``None``.

    Only a **function** column is inlined (an observable is a model species/group, not an
    algebraic expression, so it stays bare); its captured body is translated to PEtab math
    by :func:`~pybnf.petab.formula.bngl_body_to_petab_math`. A function with an empty body
    -- a forward declaration, or a function *of arguments* (only zero-arg global functions
    are the BNGL measurement-model convention) -- cannot be inlined and raises rather than
    emitting a bare-name formula that silently contradicts the requested mode. Reached only
    for a BNGL model (SBML has no functions -- :class:`_SbmlModelView.function_names` is
    empty), so ``model`` here is always a :class:`~pybnf.petab._bngl.BnglEntities`.
    """
    if kind != 'function':
        return None
    body = model.function_bodies.get(col, '')
    if not body:
        raise NotImplementedError(
            f"Function '{col}' in model '{model_file}' has no inlinable body (a forward "
            f"declaration or a function with arguments); only a zero-arg global function "
            f"'{col}() = <body>' can be inlined as an observableFormula (ADR-0035). Export "
            f"without inline_functions to reference it by bare name.")
    return bngl_body_to_petab_math(body, model)


def _noise_source_for_column(verb, arg, col, datas):
    """The PEtab noise representation for one fitted column, from the desugared sigma
    source verb (ADR-0021 reversed) -- evaluated across every experiment's ``datas``:

    * ``read_exp_file`` (the ``_SD`` data column) -> a per-point placeholder, fed by the
      measurements' ``noiseParameters``. Every ``Data`` carrying the column must also
      carry its ``<col><suffix>`` companion (else a measurement row would lack the noise
      value its declared placeholder binds to).
    * ``fix_at`` -> a constant noiseFormula (the fixed sigma).
    * ``column_mean`` -> a constant noiseFormula = the column's mean across all data.
    * ``formula`` -> the expression noiseFormula verbatim (a ``FormulaSigma``, ADR-0044/0045):
      a PEtab-math expression over free-parameter ids + constants. The expression's symbols are
      PEtab parameter ids (exported as estimated parameters); a noise nuisance that is not a
      model parameter is still a deferred boundary (it would fail the model-id binding check,
      shared with the ``fit`` sigma), so the whole-fit ``formula`` export covers an expression
      over model parameters. When the ``formula`` expression still carries a per-measurement
      **placeholder** (``noiseParameter*``), the sigma is row-varying
      (``PerMeasurementFormulaSigma``, ADR-0045): it returns ``('per_measurement', expr)``, the
      noiseFormula emitted verbatim with its placeholder and the per-row token supplied by the
      measurements' ``noiseParameters`` column (the binding-table sidecar).

    A free-parameter sigma (``fit``) and a relative sigma (``relative``) are deferred
    boundaries: the former needs the noise parameter wired into the PEtab parameter
    table (it is not a model parameter), and the latter is a ``noiseFormula``
    expression (the sympy layer, mirroring the importer's expression boundary).
    """
    holders = [data for data in datas if col in data.cols]
    if verb == 'formula':
        return ('per_measurement', arg) if _PLACEHOLDER.search(arg) else ('formula', arg)
    if verb == 'read_exp_file':
        sd_col = col + arg
        if any(sd_col not in data.cols for data in holders):
            raise NotImplementedError(
                f"Observable column '{col}': the objective reads its noise from the "
                f"'{sd_col}' data column, but a data file carrying '{col}' has no such "
                f"column. A constant or free-parameter sigma without per-point data is a "
                f"separate path (ADR-0023, #423).")
        return ('placeholder', None)
    if verb == 'fix_at':
        return ('constant', float(arg))
    if verb == 'column_mean':
        return ('constant', float(np.average(np.concatenate([d[col] for d in holders]))))
    raise NotImplementedError(
        f"Observable column '{col}': the '{verb}' sigma source is a later export chunk "
        f"-- a free-parameter sigma (fit) needs the noise parameter wired into the "
        f"PEtab parameter table, and a relative sigma is a noiseFormula expression (the "
        f"sympy layer, mirroring the importer boundary). ADR-0021/0023, #423.")


def _resolve_free_to_model(free_params, registry, models, nuisances=()):
    """Validate each free parameter binds to a model parameter id (or is an admitted
    observation-layer nuisance); return the identity map.

    New-era binds free parameters **by id** (ADR-0034): a free parameter's name *is* the
    model parameter it drives -- no ``__FREE`` marker. The id space is the **union** of every
    model's parameter ids (a BNGL ``begin parameters`` id, or an SBML global parameter id --
    ADR-0040/0041); a free parameter present in at least one model binds (the same id means
    the same knob, so a multi-model job binds it across all models that carry it). This is the
    exporter's analogue of the new-era config typo check
    (:meth:`config._check_variable_correspondence_modern`, which unions the same way): a free
    parameter matching no model parameter id is a typo (or a ``fit`` sigma, which the exporter
    rejects separately at column classification).

    ``nuisances`` are free parameters that bind to no model id but are legitimate
    **observation-layer** nuisances (a measurement scale, a noise coefficient, a row-varying
    per-row sigma -- ADR-0044/0045), gathered by :func:`_referenced_nuisance_symbols` from the
    measurement-model / noise formulae and the binding-table tokens. They pass through as
    estimated parameters with no model binding (the export peer of config widening the
    measurement-model namespace + the ``_per_measurement_free_params`` orphan union); only a
    free parameter that is neither a model id nor a referenced nuisance is a typo.

    The exporter never builds a ``Configuration``, so it validates against the views'
    ``parameters`` directly. Returns ``{name: name}`` -- the identity map the rest of the
    exporter threads as ``free_to_model``.
    """
    model_ids = set().union(*(set(v.parameters) for v in registry.values()))
    free_to_model = {}
    for fp in free_params:
        if fp.name not in model_ids and fp.name not in nuisances:
            legacy_hint = ''
            if fp.name.endswith('__FREE'):
                legacy_hint = (
                    f" The '__FREE' marker is legacy-edition only (ADR-0034); declare "
                    f"the bare parameter id '{fp.name[:-len('__FREE')]}' instead.")
            where = (f"model '{models[0]}'" if len(models) == 1
                     else f"any of the job's {len(models)} models ({models})")
            raise PybnfError(
                f"Free parameter '{fp.name}' matches no parameter id in {where}.",
                f"Under edition >= 2 a BNGL free parameter binds to a model parameter by "
                f"id (the SBML/PEtab convention; ADR-0034/0041), so '{fp.name}' must be one "
                f"of the models' parameter ids: {sorted(model_ids)}.{legacy_hint}")
        free_to_model[fp.name] = fp.name
    return free_to_model


# A bare identifier in a measurement-model / noise formula: scanned to find which free
# parameters a formula references (over-matches model entities + placeholders, harmlessly --
# only the intersection with declared free parameters is used). Dependency-free (no petab).
_FORMULA_SYMBOL = re.compile(r'[A-Za-z_]\w*')


def _referenced_nuisance_symbols(conf, conf_path, noise, per_obs_noise):
    """The names referenced as observation-layer nuisances by the job's measurement-model and
    noise surfaces (ADR-0034/0044/0045) -- the candidates :func:`_resolve_free_to_model` admits
    as model-unbound estimated parameters. The union of:

    * the symbols of every ``observable: <id>, formula: <expr>`` measurement-model formula
      (an ``observableParameters`` scale/offset substituted in -- ADR-0044 -- reads as a free
      symbol, e.g. ``scaling`` in ``scaling*x``);
    * the symbols of every ``formula``-verb ``noiseFormula`` (whole-fit or per-observable),
      e.g. ``slope`` in ``0.05*slope + 0.1`` (a ``FormulaSigma`` noise coefficient);
    * the non-numeric (parameter-id) tokens of every experiment's ``measurement_params:``
      binding-table sidecar, e.g. the per-row ``sd_lo`` / ``s_lo`` (a row-varying estimated
      sigma / scale -- ADR-0045).

    Over-matches model entities and placeholders, harmlessly: the caller intersects with the
    declared free parameters, so only a genuine free-parameter nuisance is admitted (an
    unreferenced free parameter that is not a model id stays a typo)."""
    referenced = set()
    for formula in _read_measurement_models(conf).values():
        referenced |= set(_FORMULA_SYMBOL.findall(formula))
    for _dist, verb, arg in [noise, *per_obs_noise.values()]:
        if verb == 'formula':
            referenced |= set(_FORMULA_SYMBOL.findall(arg))
    for key, fields in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 'experiment'):
            continue
        mp_file = fields.get('measurement_params')
        if not mp_file:
            continue
        table = read_measurement_params(conf_path.parent / mp_file)
        for by_placeholder in table.values():
            for by_time in by_placeholder.values():
                referenced |= {tok for tok in by_time.values() if not _is_numeric_token(tok)}
    return referenced


def _is_numeric_token(token):
    """Whether a binding-table token is a numeric literal (inlined) vs a parameter id (an
    estimated nuisance -- the export peer of ``config._is_numeric_token``)."""
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def _parameter_rows(free_params, free_to_model, surrogate_params, registry, models):
    """Map each free parameter to a row; a fit-and-mutated one renamed to ``<p>__REF``."""
    union_ids = set().union(*(set(v.parameters) for v in registry.values()))
    parameter_rows = []
    for fp in free_params:
        model_param = free_to_model[fp.name]
        if model_param in surrogate_params:
            ref = surrogate_name(model_param)
            if ref in union_ids:
                raise PybnfError(
                    f"The surrogate-base name '{ref}' for fit-and-mutated parameter "
                    f"'{model_param}' clashes with an existing parameter in the job's "
                    f"models. Rename that model parameter.")
            parameter_id = ref
        else:
            parameter_id = model_param
        parameter_rows.append(petab_parameter_row(fp, parameter_id=parameter_id))
    return parameter_rows


def _nominal_of(registry, var):
    """A fixed parameter's numeric nominal value across the job's models (ADR-0041), or
    ``None``: the value from the first model view that declares ``var`` (a fixed target's
    nominal is read from whichever model defines it, since a PEtab condition is
    model-agnostic). A free target never reaches here (the surrogate path handles it)."""
    for view in registry.values():
        if var in view.parameters:
            return _numeric_nominal(view, var)
    return None


def _numeric_nominal(model, var):
    """A fixed parameter's numeric nominal value, or ``None`` (expression/unknown RHS).

    Works for both model views: a BNGL ``parameters`` value is the raw RHS string (a number
    floats, an expression raises ``ValueError`` -> ``None``); an SBML view's value is already
    a float or ``None`` (a value-less parameter -> ``None`` via the ``TypeError`` guard)."""
    rhs = model.parameters.get(var)
    if rhs is None:
        return None
    try:
        return float(rhs)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Reading the job (the disposable input half of the seam)
# ---------------------------------------------------------------------------

def _read_conf_dict(conf_path):
    """Parse a ``.conf`` to the raw ``ploop`` dict (no model loading, no BNG)."""
    with open(conf_path) as fh:
        return ploop(fh.readlines())


def _free_parameters_from_conf(conf):
    """Build ``FreeParameter`` objects from the config's ``(keyword, name)`` entries."""
    free_params = []
    for key, value in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], str) and isinstance(key[1], str)):
            continue
        keyword, name = key
        if not _VAR_DECL.search(keyword):
            continue
        if keyword not in EXPORTABLE_PRIOR_KEYWORDS:
            raise NotImplementedError(
                f"Free parameter '{name}' is a '{keyword}'; the exporter writes the "
                f"PEtab prior families {sorted(EXPORTABLE_PRIOR_KEYWORDS)}. The "
                f"no-prior 'var'/'logvar' point-start keywords have no PEtab prior "
                f"representation (a flat improper prior is not a PEtab probability "
                f"family; ADR-0025, #423).")
        # p1/p2 are the family's governing values (bounds for the Uniform families,
        # loc/scale or shape/scale for the two-parameter location families); a 3rd token
        # is the native ``bounded`` flag, inert for the location families. A one-parameter
        # unbounded family (exponential/chisquare/rayleigh, #417) carries only p1.
        p2 = float(value[1]) if len(value) >= 2 else None
        free_params.append(FreeParameter(name, keyword, float(value[0]), p2))
    if not free_params:
        raise PybnfError(
            "No exportable free parameters found in the config (expected one of "
            f"{sorted(EXPORTABLE_PRIOR_KEYWORDS)}).")
    return free_params


@dataclass(frozen=True)
class _SbmlModelView:
    """An SBML model presented through the attribute surface the exporter's shared
    classification reads (the :class:`~pybnf.petab._bngl.BnglEntities` subset), with SBML
    semantics (ADR-0040 -- the export mirror of the ADR-0036 import dispatch):

    * ``observable_names`` are the SBML **species** (the trajectory's bare-name output
      columns). SBML has no BNGL-style observables, so a bare-name ``observableFormula``
      names a species directly -- the inverse of the importer mapping a bare-name formula
      back to a species column.
    * ``function_names`` / ``function_bodies`` are **empty**: SBML has no global BNGL
      functions, so an SBML observable is never inlined; an expression observable is carried
      in the conf measurement-model layer (``observable: <id>, formula: <expr>``) and
      classified via ``measurement_models``.
    * ``parameters`` maps each global parameter id to its numeric nominal (or ``None``) --
      the free-parameter binding set, the condition-target set, and the nominal source.
    * ``compartment_names`` are the SBML compartments (also a valid condition target).
    """

    text: str
    parameters: dict
    observable_names: frozenset
    compartment_names: frozenset
    function_names: frozenset = frozenset()
    function_bodies: dict = field(default_factory=dict)


def _model_language(model_file):
    """The PEtab model language for a model file, by extension: ``.bngl`` -> ``'bngl'``,
    ``.xml`` -> ``'sbml'``. Mirrors ``config.py``'s model-suffix dispatch (which also maps
    ``.xml`` to its SBML backends)."""
    if model_file.endswith('.bngl'):
        return 'bngl'
    if model_file.endswith('.xml'):
        return 'sbml'
    raise NotImplementedError(
        f"Model '{model_file}' has an unrecognized extension; the exporter emits BNGL "
        f"('.bngl') and SBML ('.xml') models (ADR-0025/0040). An Antimony ('.ant') model "
        f"would need an Antimony->SBML conversion first -- a later chunk.")


def _read_model(model_file, path, language):
    """Read the job's model into the entity view the exporter consumes, per ``language``.

    BNGL -> :class:`~pybnf.petab._bngl.BnglEntities`; SBML -> :class:`_SbmlModelView`
    (the same attribute surface, SBML semantics). Both expose ``text`` (verbatim source),
    ``parameters`` (the bindable ids + nominals), ``observable_names`` (bare-name observable
    columns), ``function_names``/``function_bodies`` (BNGL only), and ``compartment_names``
    -- everything the language-agnostic classification reads. The dispatch is the export
    peer of the importer's :func:`~pybnf.petab.import_._model_namespace`.
    """
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    if language == 'sbml':
        ent = parse_sbml_model(text)
        return _SbmlModelView(
            text=text,
            # Every global parameter id -> its nominal (float or None); the binding set is
            # the key set, the nominal source is the value (a value-less parameter -> None).
            parameters={p: ent.parameter_values.get(p) for p in ent.parameter_names},
            observable_names=ent.species_names,
            compartment_names=ent.compartment_names)
    return parse_bngl_model(text)


# ---------------------------------------------------------------------------
# Emitting the PEtab-clean model and problem.yaml
# ---------------------------------------------------------------------------

def clean_model_for_petab(text):
    """Return a PEtab-clean copy of a BNGL model: the ``begin actions`` block stripped.

    New-era BNGL binds free parameters **by id** (ADR-0034), so the source model already
    carries bare parameter ids with real nominal values -- exactly what PEtab estimates.
    "PEtab-clean" therefore collapses to dropping the ``begin actions`` block (PEtab
    drives simulation via the measurement times / experiments, not the model's own
    actions); the reaction network and the ``begin functions`` block -- which carry the
    measurement model -- are carried verbatim. A fit-and-mutated parameter keeps its model
    name (``v1``) here as a plain nominal-valued parameter (always overridden by its
    Condition); only the parameter *table* carries the surrogate ``v1__REF`` (ADR-0027).

    A legacy ``<name>__FREE`` marker in the model text is **rejected**: new-era binds by
    id, so a model still carrying one was not modernized, and shipping it would dangle an
    undefined ``v1__FREE`` symbol in PEtab. The error names the bind-by-id contract rather
    than letting the PEtab oracle reject it opaquely.
    """
    if _FREE_TOKEN.search(text):
        raise PybnfError(
            "This BNGL model carries a legacy '__FREE' marker, but PEtab export is a "
            "new-era feature where free parameters bind by id (ADR-0034). Declare the "
            "model's fit parameters as bare ids with nominal values (e.g. 'v1 0.5', not "
            "'v1 v1__FREE') and list them as free parameters in the .conf.")
    return re.sub(r'^[ \t]*begin\s+actions\b.*?^[ \t]*end\s+actions\b[^\n]*\n?',
                  '', text, flags=re.S | re.I | re.M)


def write_problem_yaml(path, models, has_conditions=False, has_experiments=False):
    """Write a PEtab v2 ``problem.yaml`` referencing the tables and the model(s).

    ``models`` is a list of ``(model_id, location, language)`` tuples (ADR-0041): one entry
    for a single-model job (byte-identical to the pre-multi-model output), or N entries in
    declaration order for a multi-model job. ``language`` is each model's PEtab language
    (``bngl`` or ``sbml``, ADR-0040) -- emitted verbatim so every model declares its own
    native language (a BNGL + SBML mix is just two entries)."""
    parts = [
        'format_version: 2.0.0\n',
        'parameter_files:\n  - parameters.tsv\n',
        'observable_files:\n  - observables.tsv\n',
        'measurement_files:\n  - measurements.tsv\n',
    ]
    if has_conditions:
        parts.append('condition_files:\n  - conditions.tsv\n')
    if has_experiments:
        parts.append('experiment_files:\n  - experiments.tsv\n')
    parts.append('model_files:\n')
    for model_id, location, language in models:
        parts += [
            f'  {model_id}:\n',
            f'    location: {location}\n',
            f'    language: {language}\n',
        ]
    Path(path).write_text(''.join(parts))
