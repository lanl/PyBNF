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

import re
from pathlib import Path

import numpy as np

from ..data import Data
from ..objective import _OBJECTIVE_DESUGAR
from ..parse import ploop
from ..printing import PybnfError
from ..pset import FreeParameter
from ._bngl import parse_model
from ._tsv import num
from .conditions import (
    build_dose_response_conditions,
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

# A desugared noise-model family token (ADR-0031's ``_OBJECTIVE_DESUGAR``) -> the
# PEtab v2 noiseDistribution it maps to (ADR-0023 reversed). The LINEAR Gaussian /
# Laplace likelihoods map; the other two families are explicit boundaries handled in
# ``_resolve_noise``: ``neg_bin`` was removed from PEtab v2, and PyBNF's ``lognormal``
# is log10 whereas PEtab's ``log-normal`` is natural log (a deferred sigma
# scale-conversion).
_FAMILY_TOKEN_TO_PETAB_DISTRIBUTION = {'gaussian': 'normal', 'laplace': 'laplace'}

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
    model_file = _resolve_model(conf)
    noise = _resolve_noise(conf)
    free_params = _free_parameters_from_conf(conf)
    bngl = _read_bngl(conf_path.parent / model_file)
    free_to_model = _resolve_free_to_model(free_params, bngl, model_file)
    fit_model_params = set(free_to_model.values())

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
        base_data, bngl, noise, model_file)

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


def _resolve_noise(conf):
    """The job's per-point noise model as ``(noiseDistribution, sigma_verb, sigma_arg)``.

    Reverses ADR-0031's ``_OBJECTIVE_DESUGAR`` -- the single source of truth that maps
    every legacy objfunc / modern ``objective`` token to a ``(family, {param: (verb,
    arg)}, location)`` noise-model tuple. The exporter reads the family token to a PEtab
    ``noiseDistribution`` and returns the sigma source's ``(verb, arg)`` for the
    per-column resolution in :func:`_observable_rows`.

    Raises ``NotImplementedError`` at the PEtab boundaries: an objective with no
    per-point noise model (``direct_pass``, the column-joint ``kl`` / ``wasserstein``
    profile objectives, an unknown token), and the two families PEtab v2 cannot express
    (``neg_bin`` -- removed from v2; ``lognormal`` -- PyBNF's is log10, PEtab's
    ``log-normal`` is natural log, a deferred sigma scale-conversion).
    """
    objfunc = conf.get('objfunc', conf.get('objective', 'chi_sq'))
    if objfunc not in _OBJECTIVE_DESUGAR:
        raise NotImplementedError(
            f"Objective '{objfunc}' has no per-point PEtab noise model: direct_pass "
            f"(no likelihood), kl / wasserstein (column-joint profile objectives), and "
            f"any unknown token are not PEtab observable noise. Exportable objectives: "
            f"{sorted(_OBJECTIVE_DESUGAR)} (ADR-0031, #423). Per-observable noise_model "
            f"lines are a later export chunk.")
    family_token, fields, _location = _OBJECTIVE_DESUGAR[objfunc](conf)
    distribution = _FAMILY_TOKEN_TO_PETAB_DISTRIBUTION.get(family_token)
    if distribution is None:
        raise NotImplementedError(
            f"Objective '{objfunc}' is the '{family_token}' noise family, which PEtab v2 "
            f"cannot express: neg_bin was removed from PEtab v2, and PyBNF's lognormal "
            f"is log10 while PEtab's log-normal is natural log (the sigma "
            f"scale-conversion is a later chunk). ADR-0023/0031, #423.")
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

def _observable_rows(data, bngl, noise, model_file):
    """Classify each fitted ``.exp`` column as a model observable or function and map it.

    ``noise`` is the ``(noiseDistribution, sigma_verb, sigma_arg)`` from
    :func:`_resolve_noise`; the sigma source is resolved per column (it can depend on
    the column's data, e.g. a ``column_mean`` sigma).
    """
    distribution, verb, arg = noise
    indvar = _independent_variable(data)
    observable_rows = []
    column_to_observable_id = {}
    for col in sorted(data.cols, key=data.cols.get):
        if col == indvar or col.endswith('_SD'):
            continue
        if col in bngl.observable_names:
            kind = 'observable'
        elif col in bngl.function_names:
            kind = 'function'
        else:
            raise PybnfError(
                f"Exp column '{col}' matches no observable or function in model "
                f"'{model_file}' (its observables: {sorted(bngl.observable_names)}; "
                f"functions: {sorted(bngl.function_names)}).")
        noise_source = _noise_source_for_column(verb, arg, col, data)
        row = petab_observable_row(col, kind, distribution, noise_source)
        observable_rows.append(row)
        column_to_observable_id[col] = row.observable_id
    if not observable_rows:
        raise PybnfError(
            f"Exp file for model '{model_file}' has no fittable observable/function "
            f"columns (only an independent variable and/or _SD columns).")
    return observable_rows, column_to_observable_id


def _noise_source_for_column(verb, arg, col, data):
    """The PEtab noise representation for one fitted column, from the desugared sigma
    source verb (ADR-0021 reversed):

    * ``read_exp_file`` (the ``_SD`` data column) -> a per-point placeholder, fed by the
      measurements' ``noiseParameters``. Requires the ``<col><suffix>`` column.
    * ``fix_at`` -> a constant noiseFormula (the fixed sigma).
    * ``column_mean`` -> a constant noiseFormula = the observable's own column mean.

    A free-parameter sigma (``fit``) and a relative sigma (``relative``) are deferred
    boundaries: the former needs the noise parameter wired into the PEtab parameter
    table (it is not a model parameter), and the latter is a ``noiseFormula``
    expression (the sympy layer, mirroring the importer's expression boundary).
    """
    if verb == 'read_exp_file':
        sd_col = col + arg
        if sd_col not in data.cols:
            raise NotImplementedError(
                f"Observable column '{col}': the objective reads its noise from the "
                f"'{sd_col}' data column, but the .exp has no such column. A constant "
                f"or free-parameter sigma without per-point data is a separate path "
                f"(ADR-0023, #423).")
        return ('placeholder', None)
    if verb == 'fix_at':
        return ('constant', float(arg))
    if verb == 'column_mean':
        return ('constant', float(np.average(data[col])))
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
