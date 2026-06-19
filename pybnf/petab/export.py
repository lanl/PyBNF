"""PEtab v2 exporter: a PyBNF/BNGL job -> PEtab v2 artifacts (issue #407; ADR-0025/0027).

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

**New-era only (ADR-0031).** PEtab v2 interop is a new-era feature: ``export_job``
**refuses a legacy (edition 1) job** and requires ``edition >= 2``, so the exporter reads
only the modern config surface -- the objective is named with ``objective`` /
``noise_model`` (never the retired ``objfunc``), with no implicit default -- rather than
reverse-mapping legacy syntax. The gate is on the exporter alone; the fitter still runs
legacy confs unchanged.

**Scope (chunk 2, ADR-0027).** Adds the two tables that vary the simulation per dataset:
a PyBNF **Mutant** -> a PEtab **Condition**/**Experiment** (a fit-and-mutated parameter
via a surrogate-base ``<p>__REF`` rename; see :mod:`pybnf.petab.conditions`), and a
dose-response **Parameter Scan** -> one Condition+Experiment per measured dose. A single
job is Mutants *xor* dose-response (the surrogate set is problem-wide).

The objective and prior surfaces map to PEtab as far as PEtab v2 can express them
(ADR-0023/0031 reversed): the Gaussian/Laplace likelihoods with a ``_SD``-column,
fixed, or column-mean sigma (``chi_sq``/``sos``/``sod``/``ave_norm_sos``), and the
``uniform``/``log-uniform``/``normal``/``laplace`` prior families with their log forms.
Everything else raises ``NotImplementedError`` (the boundary is in code, not silent): a
second model; an objective PEtab cannot represent (``neg_bin*`` -- removed from v2;
``lognormal`` -- log10 vs PEtab natural log; a free-parameter or relative sigma;
``direct_pass``/``kl``/``wasserstein``); the no-prior ``var``/``logvar``; a
``.con``/``.prop`` Constraint; an SBML model; both feature families in one job. The
oracle is petab's full ``default_validation_tasks`` via ``Problem.from_yaml`` + the
native ``BnglModel`` loader (ADR-0026), wired into the tests; see ADR-0025/0027.
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
from ._tsv import num
from .conditions import (
    build_dose_response_conditions,
    build_experiment_conditions,
    build_mutant_conditions,
    surrogate_name,
    write_condition_table,
    write_experiment_table,
)
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


# ---------------------------------------------------------------------------
# The exporter driver
# ---------------------------------------------------------------------------

def export_job(conf_path, out_dir):
    """Export the PyBNF job at ``conf_path`` to a PEtab v2 problem in ``out_dir``.

    Writes ``parameters.tsv``, ``observables.tsv``, ``measurements.tsv``, a PEtab-clean
    copy of the BNGL model, ``problem.yaml``, and -- when the job has Mutants or a
    dose-response Parameter Scan -- ``conditions.tsv`` / ``experiments.tsv``. Returns the
    ``out_dir`` path. Raises ``NotImplementedError`` at every documented boundary and
    ``PybnfError`` for a malformed/unsupported job (see the module docstring).
    """
    conf_path = Path(conf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf = _read_conf_dict(conf_path)
    _require_modern_edition(conf)
    model_file = _resolve_model(conf)
    noise = _resolve_noise(conf)
    free_params = _free_parameters_from_conf(conf)
    bngl = _read_bngl(conf_path.parent / model_file)
    free_to_model = _resolve_free_to_model(free_params, bngl, model_file)
    fit_model_params = set(free_to_model.values())

    if _has_new_era_data(conf):
        (observable_rows, measurement_rows, condition_rows, experiment_rows,
         surrogate_params) = _export_new_era(
            conf, conf_path, model_file, bngl, noise, fit_model_params)
    else:
        (observable_rows, measurement_rows, condition_rows, experiment_rows,
         surrogate_params) = _export_legacy(
            conf, conf_path, model_file, bngl, noise, fit_model_params)

    parameter_rows, free_to_nominal = _parameter_rows(
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
    (out_dir / model_filename).write_text(
        clean_model_for_petab(bngl.text, free_to_nominal))
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


def _export_new_era(conf, conf_path, model_file, bngl, noise, fit_model_params):
    """Read a job's data/conditions/observables from the **new-era surface** (ADR-0028).

    Export is *transcription*: an ``experiment:`` is a PEtab Experiment (experimentId =
    the experiment name) carrying its ``data:`` replicates as measurement rows; an
    ``observable:`` line renames a data column to a model entity before classification.
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

    observable_rows, column_to_observable_id = _observable_rows(
        all_datas, bngl, noise, model_file)

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


def _export_legacy(conf, conf_path, model_file, bngl, noise, fit_model_params):
    """Read a job's data/conditions from the **legacy** linkage (``model = X : Y.exp`` /
    ``mutant`` / ``param_scan``).

    Retired in Chunk 5c (refused under the edition-2 gate once the new-era surface fully
    replaces it; ADR-0028); kept here only so Chunks 5a/5b stay green while the existing
    legacy-binding fixtures migrate. Returns the same 5-tuple as :func:`_export_new_era`.
    """
    has_mutants = 'mutant' in conf
    has_scan = 'param_scan' in conf
    if has_mutants and has_scan:
        raise NotImplementedError(
            "This job has BOTH 'mutant' conditions and a 'param_scan' (dose-response). A "
            "surrogate parameter is removed from the parameter table problem-wide, so "
            "mixing the two would force the dose-response experiments to re-supply it -- "
            "export one feature family per job for now (ADR-0027, #422).")

    base_exp = _resolve_base_exp(conf, model_file)
    base_data = Data(file_name=str(conf_path.parent / base_exp))
    base_stem = Path(base_exp).stem
    indvar = _independent_variable(base_data)

    observable_rows, column_to_observable_id = _observable_rows(
        [base_data], bngl, noise, model_file)

    # Per-point noiseParameters are emitted only when the objective's sigma comes from
    # a data column (the placeholder source); a fixed / column-mean sigma is carried
    # inline in noiseFormula, so the measurement export must not read _SD (which would
    # leave a noiseParameters override with no placeholder to bind to).
    _dist, sigma_verb, sigma_arg = noise
    sd_suffix = sigma_arg if sigma_verb == 'read_exp_file' else None

    if indvar.lower() == 'time':
        measurement_rows, condition_rows, experiment_rows, surrogate_params = \
            _export_time_course(conf, conf_path, model_file, bngl, base_stem, base_data,
                                column_to_observable_id, fit_model_params, has_mutants,
                                sd_suffix)
    else:
        measurement_rows, condition_rows, experiment_rows, surrogate_params = \
            _export_dose_response(conf, conf_path, model_file, bngl, base_stem, base_data,
                                  indvar, column_to_observable_id, fit_model_params,
                                  has_scan, sd_suffix)
    return observable_rows, measurement_rows, condition_rows, experiment_rows, \
        surrogate_params


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


def _resolve_base_exp(conf, model_file):
    """Return the single base ``.exp`` the model line binds, raising at the boundaries."""
    files = conf.get(model_file, [])
    exp_files = [e for e in files if e.endswith('.exp')]
    others = [e for e in files if not e.endswith('.exp')]
    if others:
        raise NotImplementedError(
            f"Model '{model_file}' has non-.exp data ({others}); constraint (.con/.prop) "
            f"export has no core-PEtab representation -- a later (extension) chunk.")
    if len(exp_files) != 1:
        raise NotImplementedError(
            f"This chunk binds one base time-course / dose-response exp per model; model "
            f"'{model_file}' is bound to {len(exp_files)} .exp files ({exp_files}). "
            f"Multiple base time-courses are a later chunk.")
    return exp_files[0]


def _independent_variable(data):
    """The header of a wide :class:`~pybnf.data.Data`'s column 0 (``time`` or the swept axis)."""
    return min(data.cols, key=data.cols.get)


# ---------------------------------------------------------------------------
# Time-course export (plain base, or base + Mutant conditions/experiments)
# ---------------------------------------------------------------------------

def _export_time_course(conf, conf_path, model_file, bngl, base_stem, base_data,
                        column_to_observable_id, fit_model_params, has_mutants, sd_suffix):
    """Build measurements (+ conditions/experiments for Mutants) for a time-course job."""
    mutant_specs = _read_mutants(conf, model_file, bngl) if has_mutants else []
    if not mutant_specs:
        # Plain chunk-1 path: a single base time-course, "model as is" (empty experimentId).
        rows = measurement_rows_from_data(
            base_data, column_to_observable_id, experiment_id='', sd_suffix=sd_suffix)
        return rows, [], [], set()

    mutants_for_build = [(name, muts, Path(exp).stem) for name, muts, exp in mutant_specs]
    condition_rows, experiment_rows, surrogate_params, base_experiment_id = \
        build_mutant_conditions(base_stem, mutants_for_build, fit_model_params,
                                lambda v: _numeric_nominal(bngl, v))

    measurement_rows = list(measurement_rows_from_data(
        base_data, column_to_observable_id, experiment_id=base_experiment_id,
        sd_suffix=sd_suffix))
    for (_name, _muts, exp), (_n, _m, stem) in zip(mutant_specs, mutants_for_build):
        mdata = Data(file_name=str(conf_path.parent / exp))
        cmap = {c: o for c, o in column_to_observable_id.items() if c in mdata.cols}
        measurement_rows += measurement_rows_from_data(
            mdata, cmap, experiment_id=stem, sd_suffix=sd_suffix)
    return measurement_rows, condition_rows, experiment_rows, surrogate_params


def _read_mutants(conf, model_file, bngl):
    """Read + validate the job's Mutants from the raw ``ploop`` dict.

    Returns a list of ``(name, mutations, exp_filename)``, where ``mutations`` is a list of
    ``(var, op, val)`` with ``val`` a float. Raises at the chunk's Mutant boundaries (a
    Mutant of a different model, a ``.con``/``.prop`` constraint, >1 data file, or a
    mutation target that is not a model parameter/compartment).
    """
    model_stem = Path(model_file).stem
    mutants = []
    for base, name, mutations, exps in conf.get('mutant', []):
        if base not in (model_file, model_stem):
            raise PybnfError(
                f"Mutant '{name}' is declared for model '{base}', but this job's model is "
                f"'{model_file}'. Multi-model export is a later chunk.")
        exp_files = [e for e in exps if e.endswith('.exp')]
        others = [e for e in exps if not e.endswith('.exp')]
        if others:
            raise NotImplementedError(
                f"Mutant '{name}' carries non-.exp data ({others}); a Constraint "
                f"(.con/.prop) has no core-PEtab representation -- a later chunk (ADR-0027).")
        if len(exp_files) != 1:
            raise NotImplementedError(
                f"This chunk binds one .exp per Mutant; Mutant '{name}' is bound to "
                f"{len(exp_files)} ({exp_files}).")
        muts = [(var, op, float(val)) for var, op, val in mutations]
        for var, _op, _val in muts:
            if var not in bngl.parameters and var not in bngl.compartment_names:
                raise PybnfError(
                    f"Mutant '{name}' mutates '{var}', which is not a parameter or "
                    f"compartment of model '{model_file}' (a PEtab condition target must "
                    f"be a model entity).")
        mutants.append((name, muts, exp_files[0]))
    return mutants


# ---------------------------------------------------------------------------
# Dose-response export (a Parameter Scan: one Condition+Experiment per dose)
# ---------------------------------------------------------------------------

def _export_dose_response(conf, conf_path, model_file, bngl, base_stem, base_data,
                          swept_param, column_to_observable_id, fit_model_params, has_scan,
                          sd_suffix):
    """Build conditions/experiments/measurements for a dose-response Parameter Scan job."""
    if not has_scan:
        raise NotImplementedError(
            f"The exp '{base_stem}.exp' has independent axis '{swept_param}' (not 'time'), "
            f"so it is a dose-response, but the config declares no 'param_scan' action. A "
            f"param_scan in the BNGL begin actions block is not parsed; declare it in the "
            f".conf so the exporter can read its fixed measurement time (ADR-0027).")
    if swept_param not in bngl.parameters and swept_param not in bngl.compartment_names:
        raise PybnfError(
            f"The dose-response swept axis '{swept_param}' is not a parameter or "
            f"compartment of model '{model_file}' (a PEtab condition target must be a "
            f"model entity).")
    if swept_param in fit_model_params:
        raise NotImplementedError(
            f"The dose-response sweeps the fit parameter '{swept_param}'. Scanning an "
            f"estimated parameter is out of scope for this chunk (it would re-trigger the "
            f"parameter-table/condition-target overlap; ADR-0027).")

    scan_time = _scan_time_for(conf, base_stem)
    iv = base_data.cols[swept_param]
    dose_values = [float(base_data.data[i, iv]) for i in range(base_data.data.shape[0])]

    condition_rows, experiment_rows, experiment_ids = build_dose_response_conditions(
        base_stem, swept_param, dose_values, scan_time)
    measurement_rows = dose_response_measurement_rows(
        base_data, column_to_observable_id, experiment_ids, scan_time, sd_suffix=sd_suffix)
    return measurement_rows, condition_rows, experiment_rows, set()


def _scan_time_for(conf, stem):
    """The fixed measurement time of the ``param_scan`` whose suffix is the exp ``stem``."""
    for entry in conf.get('param_scan', []):
        if entry.get('suffix', 'param_scan') == stem:
            if 'time' not in entry:
                raise PybnfError(
                    f"The param_scan with suffix '{stem}' has no 'time' attribute; the "
                    f"dose-response measurement time is required for export.")
            return float(entry['time'])
    raise NotImplementedError(
        f"The dose-response exp '{stem}.exp' has no matching 'param_scan' action "
        f"(suffix '{stem}') in the config (ADR-0027).")


# ---------------------------------------------------------------------------
# Observable + parameter rows
# ---------------------------------------------------------------------------

def _observable_rows(datas, bngl, noise, model_file):
    """Classify each fitted column across all experiments' ``datas`` as a model observable
    or function and map it to a PEtab observable row.

    ``datas`` is the list of every experiment's (override-renamed) :class:`~pybnf.data.Data`
    (one element for the legacy single base time-course); a column is gathered once, in
    first-appearance order, so the observables table covers the whole job. ``noise`` is the
    ``(noiseDistribution, sigma_verb, sigma_arg)`` from :func:`_resolve_noise`; the sigma
    source is resolved per column (it can depend on the column's data, e.g. a
    ``column_mean`` sigma).
    """
    distribution, verb, arg = noise
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
        if col in bngl.observable_names:
            kind = 'observable'
        elif col in bngl.function_names:
            kind = 'function'
        else:
            raise PybnfError(
                f"Exp column '{col}' matches no observable or function in model "
                f"'{model_file}' (its observables: {sorted(bngl.observable_names)}; "
                f"functions: {sorted(bngl.function_names)}).")
        noise_source = _noise_source_for_column(verb, arg, col, datas)
        row = petab_observable_row(col, kind, distribution, noise_source)
        observable_rows.append(row)
        column_to_observable_id[col] = row.observable_id
    if not observable_rows:
        raise PybnfError(
            f"Exp data for model '{model_file}' has no fittable observable/function "
            f"columns (only an independent variable and/or _SD columns).")
    return observable_rows, column_to_observable_id


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
    """Map each free-parameter name (``v1__FREE``) to its model parameter (``v1``)."""
    free_to_model = {}
    for fp in free_params:
        model_param = bngl.free_to_param.get(fp.name)
        if model_param is None:
            raise PybnfError(
                f"Free parameter '{fp.name}' is not assigned to any parameter in model "
                f"'{model_file}' (expected a line '<param> {fp.name}' in begin "
                f"parameters).")
        free_to_model[fp.name] = model_param
    return free_to_model


def _parameter_rows(free_params, free_to_model, surrogate_params, bngl, model_file):
    """Map each free parameter to a row; a fit-and-mutated one renamed to ``<p>__REF``."""
    parameter_rows = []
    free_to_nominal = {}
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
        # A syntactically valid default for the PEtab-clean model; PEtab overrides it
        # during estimation. The bounds' midpoint is a reasonable in-range nominal.
        free_to_nominal[fp.name] = (float(fp.p1) + float(fp.p2)) / 2.0
    return parameter_rows, free_to_nominal


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
        # p1/p2 are the family's two governing values (bounds for the Uniform
        # families, loc/scale for the location-scale ones); a 3rd token is the native
        # ``bounded`` flag, inert for the location-scale families and left at the
        # FreeParameter default for the Uniform ones (matching chunk 1).
        free_params.append(FreeParameter(name, keyword, float(value[0]),
                                         float(value[1])))
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

def clean_model_for_petab(text, free_to_nominal):
    """Return a PEtab-clean copy of a BNGL model.

    PEtab estimates the model parameters directly, so the ``__FREE`` markers are
    replaced by a plain in-range nominal value (``v1 v1__FREE`` -> ``v1 5``), and the
    ``begin actions`` block is stripped (PEtab drives simulation via the measurement
    times / experiments, not the model's own actions). The reaction network and the
    ``begin functions`` block -- which carry the measurement model -- are left intact.
    A fit-and-mutated parameter keeps its model name (``v1``) here as a plain
    nominal-valued parameter (always overridden by its Condition); only the parameter
    *table* carries the surrogate ``v1__REF`` (ADR-0027).
    """
    for free_name, nominal in free_to_nominal.items():
        text = re.sub(rf'\b{re.escape(free_name)}\b', num(nominal), text)
    text = re.sub(r'^[ \t]*begin\s+actions\b.*?^[ \t]*end\s+actions\b[^\n]*\n?',
                  '', text, flags=re.S | re.I | re.M)
    return text


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
