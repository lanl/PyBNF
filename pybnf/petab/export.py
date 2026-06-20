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
rows once. Dose-response (a parameter scan) is deferred: a fully new-era conf cannot
author the scan's simulation endpoint time (#426), so the exporter never sees one (a
parameter-scan experiment raises that deferral).

The objective and prior surfaces map to PEtab as far as PEtab v2 can express them
(ADR-0023/0031 reversed): the Gaussian/Laplace likelihoods with a ``_SD``-column,
fixed, or column-mean sigma (``chi_sq``/``sos``/``sod``/``ave_norm_sos``), and the
``uniform``/``log-uniform``/``normal``/``laplace`` prior families with their log forms.
Everything else raises ``NotImplementedError`` (the boundary is in code, not silent): a
second model; an objective PEtab cannot represent (``neg_bin*`` -- removed from v2;
``lognormal`` -- log10 vs PEtab natural log; a free-parameter or relative sigma;
``direct_pass``/``kl``/``wasserstein``); the no-prior ``var``/``logvar``; a
``.con``/``.prop`` Constraint; an SBML model. The oracle is petab's full
``default_validation_tasks`` via ``Problem.from_yaml`` + the native ``BnglModel`` loader
(ADR-0026), wired into the tests; see ADR-0025/0027/0028.
"""

import logging
import re
from pathlib import Path

import numpy as np

from .. import edition
from ..data import Data
from ..objective import _OBJECTIVE_DESUGAR
from ..parse import ploop
from ..printing import PybnfError
from ..pset import FreeParameter
from ._bngl import parse_model
from .conditions import (
    build_experiment_conditions,
    surrogate_name,
    write_condition_table,
    write_experiment_table,
)
from .formula import bngl_body_to_petab_math
from .measurements import (
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
    model_file = _resolve_model(conf)
    _require_new_era_data(conf, model_file)
    noise = _resolve_noise(conf)
    free_params = _free_parameters_from_conf(conf)
    bngl = _read_bngl(conf_path.parent / model_file)
    free_to_model = _resolve_free_to_model(free_params, bngl, model_file)
    fit_model_params = set(free_to_model.values())

    (observable_rows, measurement_rows, condition_rows, experiment_rows,
     surrogate_params) = _export_new_era(
        conf, conf_path, model_file, bngl, noise, fit_model_params, inline_functions)

    parameter_rows = _parameter_rows(
        free_params, free_to_model, surrogate_params, bngl, model_file)

    model_filename = Path(model_file).name
    model_id = re.sub(r'\.bngl$', '', model_filename)
    write_parameter_table(parameter_rows, out_dir / 'parameters.tsv')
    write_observable_table(observable_rows, out_dir / 'observables.tsv')
    write_measurement_table(measurement_rows, out_dir / 'measurements.tsv')
    if condition_rows:
        write_condition_table(condition_rows, out_dir / 'conditions.tsv')
    if experiment_rows:
        write_experiment_table(experiment_rows, out_dir / 'experiments.tsv')
    (out_dir / model_filename).write_text(clean_model_for_petab(bngl.text))
    write_problem_yaml(out_dir / 'problem.yaml', model_filename, model_id,
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


def _require_new_era_data(conf, model_file):
    """Refuse a legacy data linkage -- the PEtab v2 exporter reads only the new-era data
    surface (ADR-0028 Chunk 5c, "refuse legacy everything").

    The exporter is already new-era-gated on the *objective* (``_require_modern_edition``);
    now that ADR-0028's data surface exists, the data linkage is held to the same standard.
    A job must introduce data through named ``experiment:`` lines (a PEtab Experiment
    carrying its ``data:``); the legacy filename->suffix binding (``model = X : Y.exp``),
    ``mutant`` lines, and ``param_scan`` actions are refused rather than silently read.
    Mixing the two (a ``mutant``/``param_scan`` line, or data on the ``model =`` line,
    alongside ``experiment:``) is refused too, so a legacy line is never silently dropped.
    The gate is on the *exporter* only; the fitter still runs legacy confs unchanged.
    """
    legacy_data = bool(conf.get(model_file))          # model = X : Y.exp data list
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


def _export_new_era(conf, conf_path, model_file, bngl, noise, fit_model_params,
                    inline_functions=False):
    """Read a job's data/conditions/observables from the **new-era surface** (ADR-0028).

    Export is *transcription*: an ``experiment:`` is a PEtab Experiment (experimentId =
    the experiment name) carrying its ``data:`` replicates as measurement rows; an
    ``observable:`` line renames a data column to a model entity before classification.
    ``inline_functions`` is threaded to :func:`_observable_rows` (ADR-0035 inlining).
    Returns ``(observable_rows, measurement_rows, condition_rows, experiment_rows,
    surrogate_params)``.

    A ``condition:`` referenced by an experiment becomes a PEtab Condition (the
    surrogate-base machinery of ADR-0027, generalized by
    :func:`~pybnf.petab.conditions.build_experiment_conditions`): a fit-and-perturbed
    parameter is renamed to ``<p>__REF`` in the parameter table and pinned in every
    experiment's Condition. A parameter-scan experiment is deferred (#426; raised in
    :func:`_read_experiments`). With no referenced conditions the surrogate set is empty,
    so a single wildtype experiment is byte-identical to the chunk-1 base time-course.
    """
    experiments = _read_experiments(conf, conf_path, model_file, bngl)
    overrides = _read_observable_overrides(conf)
    all_datas = [d for exp in experiments for d in exp['datas']]
    _apply_observable_overrides(all_datas, overrides)

    measurement_models = _read_measurement_models(conf)
    observable_rows, column_to_observable_id = _observable_rows(
        all_datas, bngl, noise, model_file, inline_functions, measurement_models)

    conditions = _read_conditions(conf, model_file, bngl)
    referenced = {exp['condition'] for exp in experiments if exp['condition'] is not None}
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
            [(exp['name'], exp['condition']) for exp in experiments],
            conditions, fit_model_params, lambda v: _numeric_nominal(bngl, v))

    # Per-point noiseParameters are emitted only when the objective's sigma comes from a
    # data column (the placeholder source); a fixed / column-mean sigma is carried inline
    # in noiseFormula, so the measurement export must not read _SD then (it would leave a
    # noiseParameters override with no placeholder to bind to).
    _dist, sigma_verb, sigma_arg = noise
    sd_suffix = sigma_arg if sigma_verb == 'read_exp_file' else None

    measurement_rows = []
    for exp in experiments:
        eid = experiment_to_id[exp['name']]
        # Each replicate Data contributes its own rows under the one experiment (PEtab
        # models replicates as repeated rows -- no need to pre-stack as config.py does).
        for data in exp['datas']:
            cmap = {c: o for c, o in column_to_observable_id.items() if c in data.cols}
            measurement_rows += measurement_rows_from_data(
                data, cmap, experiment_id=eid, sd_suffix=sd_suffix)
    return observable_rows, measurement_rows, condition_rows, experiment_rows, \
        surrogate_params


def _read_experiments(conf, conf_path, model_file, bngl):
    """Read + resolve the new-era ``experiment:`` entries from the raw ``ploop`` dict.

    Each ``('experiment', name)`` entry is ``{'data': [files], 'condition': c?, 'model':
    mf?, 'type': t?, 'method': m?}``. Returns a list (declaration order) of dicts
    ``{'name', 'condition', 'datas': [Data, ...]}`` -- the ``data:`` files read as
    individual :class:`~pybnf.data.Data` replicates (PEtab models replicates as repeated
    measurement rows, so they are not pre-stacked). Raises:

    * the parameter-scan deferral for a non-time-course experiment -- the scan's
      simulation endpoint time has no home in the ``experiment:`` grammar yet, so a fully
      new-era conf cannot author one (deferred, #426; mirrors
      ``config.py::_load_experiments``);
    * a multi-model boundary if an experiment names a different model (a later chunk);
    * a constraint boundary for non-``.exp`` data (``.con``/``.prop`` is deferred,
      ADR-0028 Open/deferred).
    """
    model_stem = Path(model_file).stem
    experiments = []
    for key, fields in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 'experiment'):
            continue
        name = key[1]
        ref = fields.get('model')
        if ref is not None and Path(ref).stem != model_stem:
            raise PybnfError(
                f"Experiment '{name}' names model '{ref}', but this job's model is "
                f"'{model_file}'. Multi-model export is a later chunk.")
        data_files = fields.get('data', [])
        if not data_files:
            raise PybnfError(f"Experiment '{name}' declares no 'data:' files.")
        non_exp = [f for f in data_files if not f.endswith('.exp')]
        if non_exp:
            raise NotImplementedError(
                f"Experiment '{name}' has non-.exp data ({non_exp}); constraint "
                f"(.con/.prop) export has no core-PEtab representation -- a later chunk "
                f"(ADR-0028 Open/deferred).")
        datas = [Data(file_name=str(conf_path.parent / f)) for f in data_files]
        if _experiment_type(name, datas[0], fields.get('type')) != 'time_course':
            raise NotImplementedError(
                f"Experiment '{name}' is a parameter_scan (independent variable "
                f"'{_independent_variable(datas[0])}'), not a time course. Parameter-scan "
                f"/ dose-response experiments are not yet exportable via the new-era "
                f"'experiment:' surface: the scan's simulation endpoint time has no home "
                f"in the experiment grammar yet, so a fully new-era conf cannot author "
                f"one (deferred, #426). Export covers time-course experiments.")
        experiments.append(
            {'name': name, 'condition': fields.get('condition'), 'datas': datas})
    return experiments


def _experiment_type(name, data, explicit_type):
    """Infer ``'time_course'`` vs ``'parameter_scan'`` from a ``Data``'s independent
    variable (``time`` => time_course; otherwise the indvar names a swept parameter =>
    parameter_scan), unless ``type:`` states it. Mirrors
    ``config.py::_infer_experiment_type`` (the caller defers parameter_scan -- #426)."""
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


def _read_conditions(conf, model_file, bngl):
    """Read + validate the new-era ``condition:`` entries from the raw ``ploop`` dict.

    Each ``('condition', name)`` entry is ``(model_ref_or_None, [(var, op, val_str), ...])``
    (a named set of parameter perturbations -- a PyBNF Mutant = a PEtab Condition). Returns
    ``{condition_name: [(var, op, float(val)), ...]}``. Validates the model ref
    (single-model boundary) and that every perturbation target is a model parameter /
    compartment -- a PEtab condition target must be a model entity (mirrors the legacy
    :func:`_read_mutants` checks)."""
    model_stem = Path(model_file).stem
    conditions = {}
    for key, value in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 'condition'):
            continue
        name = key[1]
        model_ref, perts = value
        if model_ref is not None and Path(model_ref).stem != model_stem:
            raise PybnfError(
                f"Condition '{name}' is declared for model '{model_ref}', but this job's "
                f"model is '{model_file}'. Multi-model export is a later chunk.")
        muts = []
        for var, op, val in perts:
            if var not in bngl.parameters and var not in bngl.compartment_names:
                raise PybnfError(
                    f"Condition '{name}' perturbs '{var}', which is not a parameter or "
                    f"compartment of model '{model_file}' (a PEtab condition target must "
                    f"be a model entity).")
            muts.append((var, op, float(val)))
        conditions[name] = muts
    return conditions


# ---------------------------------------------------------------------------
# Scope resolution (the documented boundaries)
# ---------------------------------------------------------------------------

def _resolve_model(conf):
    """Return the job's single BNGL model file, raising at the chunk boundaries."""
    models = sorted(conf.get('models', set()))
    if len(models) != 1:
        raise NotImplementedError(
            f"This chunk exports a single-model job; this one has {len(models)} models "
            f"({models}). Multi-model export (PEtab modelId) is a later chunk.")
    model_file = models[0]
    if not model_file.endswith('.bngl'):
        raise NotImplementedError(
            f"This chunk exports BNGL models; '{model_file}' is not '.bngl'. SBML/Antimony "
            f"export is a later chunk (ADR-0025 settles the BNGL side first).")
    return model_file


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

    Raises ``NotImplementedError`` at every PEtab boundary, never a silent default:

    * a column-joint ``profile_objective`` (``kl`` / ``wasserstein``) -- it scores the
      whole column's shape, not a per-observation likelihood, so it has no PEtab
      observable-noise representation;
    * per-observable ``noise_model <obs> = ...`` overrides -- a later chunk (they map to
      per-observable PEtab noise);
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
    if any(isinstance(k, tuple) and k[0] == 'noise_model' and k[1] is not None
           for k in conf):
        raise NotImplementedError(
            "Per-observable 'noise_model <obs> = ...' overrides are a later export chunk "
            "-- they map to per-observable PEtab observable noise; this chunk exports one "
            "whole-fit noise model (ADR-0021/0023, #423).")

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

    if location == 'mean':
        raise NotImplementedError(
            "This noise model is mean-centered (location = mean); PEtab v2 takes the "
            "prediction as the distribution median for every noise family, so mean "
            "centering has no PEtab representation (ADR-0031, #423). Use median.")

    distribution = _FAMILY_TOKEN_TO_PETAB_DISTRIBUTION.get(family_token.lower())
    if distribution is None:
        raise NotImplementedError(
            f"The '{family_token}' noise family cannot be expressed in PEtab v2: neg_bin "
            f"was removed from v2, and PyBNF's lognormal is log10 while PEtab's "
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

def _observable_rows(datas, bngl, noise, model_file, inline_functions=False,
                     measurement_models=None):
    """Classify each fitted column across all experiments' ``datas`` as a model observable,
    a model function, or a conf measurement model, and map it to a PEtab observable row.

    ``datas`` is the list of every experiment's (override-renamed) :class:`~pybnf.data.Data`
    (one element for the legacy single base time-course); a column is gathered once, in
    first-appearance order, so the observables table covers the whole job. ``noise`` is the
    ``(noiseDistribution, sigma_verb, sigma_arg)`` from :func:`_resolve_noise`; the sigma
    source is resolved per column (it can depend on the column's data, e.g. a
    ``column_mean`` sigma).

    ``inline_functions`` (ADR-0035) emits a **function** column's body as an
    ``observableFormula`` expression instead of the bare name -- the opt-in path that
    generates the importer's round-trip oracle; the default keeps every column bare.
    ``measurement_models`` (``{id: formula}``, ADR-0036) are conf-declared measurement models:
    a column matching one is emitted with that formula as its ``observableFormula`` and its id
    verbatim (the inverse of the importer's ``observable: ... formula:`` line)."""
    distribution, verb, arg = noise
    measurement_models = measurement_models or {}
    columns = []
    for data in datas:
        indvar = data.indvar if data.indvar is not None else _independent_variable(data)
        for col in sorted(data.cols, key=data.cols.get):
            if col == indvar or col.endswith('_SD') or col in columns:
                continue
            columns.append(col)
    observable_rows = []
    column_to_observable_id = {}
    for col in columns:
        formula = None
        if col in measurement_models:
            kind = 'measurement'
            formula = measurement_models[col]   # the conf observableFormula, verbatim
        elif col in bngl.observable_names:
            kind = 'observable'
        elif col in bngl.function_names:
            kind = 'function'
            formula = _inlined_formula(col, kind, bngl, model_file) if inline_functions \
                else None
        else:
            raise PybnfError(
                f"Exp column '{col}' matches no observable, function, or measurement model "
                f"in model '{model_file}' (its observables: {sorted(bngl.observable_names)}; "
                f"functions: {sorted(bngl.function_names)}; measurement models: "
                f"{sorted(measurement_models)}).")
        noise_source = _noise_source_for_column(verb, arg, col, datas)
        row = petab_observable_row(col, kind, distribution, noise_source,
                                   observable_formula=formula)
        observable_rows.append(row)
        column_to_observable_id[col] = row.observable_id
    if not observable_rows:
        raise PybnfError(
            f"Exp data for model '{model_file}' has no fittable observable/function "
            f"columns (only an independent variable and/or _SD columns).")
    return observable_rows, column_to_observable_id


def _inlined_formula(col, kind, bngl, model_file):
    """The ``observableFormula`` for a column under inlining mode (ADR-0035), or ``None``.

    Only a **function** column is inlined (an observable is a model species/group, not an
    algebraic expression, so it stays bare); its captured body is translated to PEtab math
    by :func:`~pybnf.petab.formula.bngl_body_to_petab_math`. A function with an empty body
    -- a forward declaration, or a function *of arguments* (only zero-arg global functions
    are the BNGL measurement-model convention) -- cannot be inlined and raises rather than
    emitting a bare-name formula that silently contradicts the requested mode.
    """
    if kind != 'function':
        return None
    body = bngl.function_bodies.get(col, '')
    if not body:
        raise NotImplementedError(
            f"Function '{col}' in model '{model_file}' has no inlinable body (a forward "
            f"declaration or a function with arguments); only a zero-arg global function "
            f"'{col}() = <body>' can be inlined as an observableFormula (ADR-0035). Export "
            f"without inline_functions to reference it by bare name.")
    return bngl_body_to_petab_math(body, bngl)


def _noise_source_for_column(verb, arg, col, datas):
    """The PEtab noise representation for one fitted column, from the desugared sigma
    source verb (ADR-0021 reversed) -- evaluated across every experiment's ``datas``:

    * ``read_exp_file`` (the ``_SD`` data column) -> a per-point placeholder, fed by the
      measurements' ``noiseParameters``. Every ``Data`` carrying the column must also
      carry its ``<col><suffix>`` companion (else a measurement row would lack the noise
      value its declared placeholder binds to).
    * ``fix_at`` -> a constant noiseFormula (the fixed sigma).
    * ``column_mean`` -> a constant noiseFormula = the column's mean across all data.

    A free-parameter sigma (``fit``) and a relative sigma (``relative``) are deferred
    boundaries: the former needs the noise parameter wired into the PEtab parameter
    table (it is not a model parameter), and the latter is a ``noiseFormula``
    expression (the sympy layer, mirroring the importer's expression boundary).
    """
    holders = [data for data in datas if col in data.cols]
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


def _resolve_free_to_model(free_params, bngl, model_file):
    """Validate each free parameter binds to a model parameter id; return the identity map.

    New-era BNGL binds free parameters **by id** (ADR-0034): a free parameter's name *is*
    the model parameter it drives -- no ``__FREE`` marker. This is the exporter's analogue
    of the new-era config typo check (:meth:`config._check_variable_correspondence_modern`):
    a free parameter matching a model parameter id binds; one matching none is a typo (or
    a ``fit`` sigma, which the exporter rejects separately at column classification). The
    exporter never builds a ``Configuration``, so it validates against ``bngl.parameters``
    directly rather than calling into ``config``. Returns ``{name: name}`` -- the identity
    map the rest of the exporter threads as ``free_to_model``.
    """
    model_ids = set(bngl.parameters)
    free_to_model = {}
    for fp in free_params:
        if fp.name not in model_ids:
            legacy_hint = ''
            if fp.name.endswith('__FREE'):
                legacy_hint = (
                    f" The '__FREE' marker is legacy-edition only (ADR-0034); declare "
                    f"the bare parameter id '{fp.name[:-len('__FREE')]}' instead.")
            raise PybnfError(
                f"Free parameter '{fp.name}' matches no parameter id in model "
                f"'{model_file}'.",
                f"Under edition >= 2 a BNGL free parameter binds to a model parameter by "
                f"id (the SBML/PEtab convention; ADR-0034), so '{fp.name}' must be one of "
                f"the model's parameter ids: {sorted(model_ids)}.{legacy_hint}")
        free_to_model[fp.name] = fp.name
    return free_to_model


def _parameter_rows(free_params, free_to_model, surrogate_params, bngl, model_file):
    """Map each free parameter to a row; a fit-and-mutated one renamed to ``<p>__REF``."""
    parameter_rows = []
    for fp in free_params:
        model_param = free_to_model[fp.name]
        if model_param in surrogate_params:
            ref = surrogate_name(model_param)
            if ref in bngl.parameters:
                raise PybnfError(
                    f"The surrogate-base name '{ref}' for fit-and-mutated parameter "
                    f"'{model_param}' clashes with an existing parameter in model "
                    f"'{model_file}'. Rename that model parameter.")
            parameter_id = ref
        else:
            parameter_id = model_param
        parameter_rows.append(petab_parameter_row(fp, parameter_id=parameter_id))
    return parameter_rows


def _numeric_nominal(bngl, var):
    """A fixed parameter's numeric nominal value, or ``None`` (expression/unknown RHS)."""
    rhs = bngl.parameters.get(var)
    if rhs is None:
        return None
    try:
        return float(rhs)
    except ValueError:
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


def _read_bngl(path):
    """Read a BNGL model's named entities (delegates to the shared stdlib parser)."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return parse_model(text)


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


def write_problem_yaml(path, model_filename, model_id, has_conditions=False,
                       has_experiments=False):
    """Write a PEtab v2 ``problem.yaml`` referencing the tables and the model."""
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
    parts += [
        'model_files:\n',
        f'  {model_id}:\n',
        f'    location: {model_filename}\n',
        '    language: bngl\n',
    ]
    Path(path).write_text(''.join(parts))
