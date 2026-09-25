"""Scale-preserving PEtab v1 -> v2 conversion (wraps ``petab.v2.petab1to2``).

``petab.v2.petab1to2`` drops **two** v1 scale columns PyBNF needs -- the parameter
``parameterScale`` and the observable ``observableTransformation`` -- because PEtab v2
removed both. This module wraps the standard converter and re-injects each so the converted
v2 problem still runs the search *and* scores the objective the v1 problem specified.

**parameterScale (the estimation scale).** PEtab v2 removed it on the view that estimation
scale is an *optimiser* concern, not part of the problem spec, and ``petab1to2`` only
*warns* ("Parameter scales are not supported in PEtab v2"). For a parameter that was
``parameterScale = log10`` **without** an objective prior (the common case: a
maximum-likelihood fit over a multi-decade kinetic parameter), the converted v2 problem is
read as *linear* ``uniform_var`` over the raw bounds. That is the same argmin, but a
vastly harder, worse-conditioned optimisation than the log10 search the modeller specified.
We re-inject that **bare** estimation scale in the **v2-native** form: each log/log10
estimated parameter with no declared prior gets ``priorDistribution = log-uniform`` over its
bounds. PyBNF -- and any PEtab v2 tool -- then reads a log search (a ``log-uniform`` prior
maps to a ``loguniform_var`` on the Log10 scale; :mod:`pybnf.petab.parameters`). PyBNF's
optimiser objective *excludes* the prior, so the log-uniform prior sets only the search
scale and initial sampling, not the objective -- the fit stays the pure-MLE problem v1
specified.

**Declared objective priors (issue #893).** A v1 prior is stated either on the parameter
itself (``uniform`` / ``normal`` / ``laplace``), on its natural log (``logNormal`` /
``logLaplace``), or on its ``parameterScale`` (``parameterScale{Uniform,Normal,Laplace}``,
whose numbers are in ``ln`` or ``log10`` units on a log scale). PEtab v2 has only the linear
and the natural-log forms, with ``log-uniform`` taking bounds on the parameter itself.
petab1to2 renames the ``parameterScale*`` types but leaves their numbers alone. That is
right on a ``lin`` scale, and on a natural-log scale for normal and laplace, but not
elsewhere: a ``log10`` ``parameterScaleNormal`` becomes a natural-log ``log-normal`` whose
mean and sd v2 then reads ``ln 10`` times too small (petab1to2 warns), and a natural-log
``parameterScaleUniform(a;b)`` becomes ``log-uniform(a;b)`` instead of
``log-uniform(e^a;e^b)`` (no warning). So petab1to2's prior is not kept for any row whose v1
author declared one: :func:`v2_prior_from_v1` rewrites each from the v1 row, mapping every
v1 prior type on every scale to the v2 prior with the same distribution over the parameter.
The cases petab1to2 refuses outright (``logNormal``, ``logLaplace``, a ``log10``
``parameterScaleUniform`` / ``parameterScaleLaplace``) still stop the conversion there;
ADR-0073's 2026-09-25 addendum tabulates every case.

**observableTransformation (the residual scale; issue #499, ADR-0073).** A v1 observable with
``observableTransformation = log10`` fits the residual on the log10 scale (with the
change-of-variables Jacobian) -- a *different objective* from the linear residual, not just
a different search. PEtab v2 removed the column and folded transformation into
``noiseDistribution`` as **natural-log** ``log-normal`` / ``log-laplace`` prefixes, with **no
log10 form**; ``petab1to2`` therefore cannot carry a v1 ``log10`` transformation faithfully.
petab < 0.9.0 dropped it entirely (a missing ``return`` left ``noiseDistribution`` blank, so
the observable imported as a linear Gaussian and the fit optimised the wrong objective);
petab >= 0.9.0 substitutes the natural-log family (``log10-normal`` -> ``log-normal``, with a
warning), which silently rescales sigma by ``ln 10``. Since v2 has no faithful
representation, we re-inject ``observableTransformation`` as a **preserved extra column** on
the v2 observables table and reset ``noiseDistribution`` to the v1 **linear base family**
(``normal`` / ``laplace``), so the residual scale is stated in exactly one place: PyBNF's
importer reads the column to select the noise family's additive scale (``lin`` / ``log10`` /
``log``; :mod:`pybnf.petab.observables`, :mod:`pybnf.petab.import_`) and refuses a log
transformation over an already-log distribution as a contradiction (issue #679), while other
PEtab v2 tools ignore the unknown column (it passes v2 lint). This is directly parallel to
the parameterScale re-injection above -- the same "re-add the scale petab1to2 dropped"
migration, on the observable axis.

**petab1to2's warnings.** petab1to2 warns when it drops or substitutes something. The
converter silences exactly the warnings whose subject it repairs (listed in
``_HANDLED_PETAB1TO2_WARNINGS``) and lets every other warning reach the caller. A v1
``initializationPrior*`` has no v2 home and no PyBNF channel; petab1to2's blanket warning
about it is replaced by one that names each dropped initialization prior, and a stated v1
default (``parameterScaleUniform`` over the bounds) is dropped without comment because it
is what PyBNF does anyway.

This is the migration ``petab1to2`` should offer as an opt-in; it lives here as an
explicit, named converter so :func:`pybnf.petab.import_job` stays a pure v2 importer with
no reach-back to v1 in the read path.
"""

import math
import warnings
from pathlib import Path

from ..printing import PybnfError
from ._tsv import num

#: v1 ``parameterScale`` values that mean "estimate in log space" (base-independent for a
#: *uniform* prior: uniform-in-ln and uniform-in-log10 are the same distribution over the
#: same bounds, and PyBNF searches log-uniform on its Log10 scale either way).
_LOG_SCALES = frozenset({'log', 'log10'})

#: v1 ``observableTransformation`` values that name a non-linear residual scale (issue #499).
#: Unlike ``parameterScale`` the base matters here -- the residual and its Jacobian live on
#: that exact scale -- so ``log`` (natural) and ``log10`` are re-injected verbatim, not folded.
_LOG_TRANSFORMATIONS = frozenset({'log', 'log10'})

#: The preserved extra column re-injected onto the v2 observables table (PEtab v2 removed the
#: v1 spelling; PyBNF's importer reads it, other v2 tools ignore it).
_OBSERVABLE_TRANSFORMATION_COLUMN = 'observableTransformation'

#: The v2 observables column petab1to2 folds the v1 transformation into (as the natural-log
#: family, petab >= 0.9.0; blank before). Reset to the v1 linear base family alongside the
#: re-injected transformation so the scale is stated once (issue #679).
_NOISE_DISTRIBUTION_COLUMN = 'noiseDistribution'

#: The petab1to2 warnings (petab 0.9.0) this converter silences, each because the converter
#: repairs what the warning is about (issue #893). ``warnings.filterwarnings`` matches each
#: pattern against the start of the message; every other warning petab1to2 raises reaches
#: the caller. The two "Using `log-normal` instead" substitutions are the dangerous ones:
#: left alone, each silently rescales a spread by ``ln 10``.
_HANDLED_PETAB1TO2_WARNINGS = (
    # parameterScale dropped. A bare log/log10 scale is re-injected as a log-uniform prior; a
    # row with a declared prior is searched on that prior's own scale (a linear prior on a
    # log-scale parameter therefore linearly -- ADR-0073, 2026-09-25 addendum).
    r'Parameter scales are not supported in PEtab v2\.',
    # log10 + normal observable -> log-normal. Every log observable's noiseDistribution is
    # reset to its v1 base and its observableTransformation re-injected (#499, #679).
    r"Noise distribution `log10-normal' for observable `[^']*' is not supported in PEtab v2\.",
    # log10 parameterScaleNormal -> log-normal with the numbers unchanged. Every declared
    # prior is rewritten from its v1 row by v2_prior_from_v1 (#893).
    r"Prior distribution `log10-normal' for parameter `[^']*' is not supported in PEtab v2\.",
    # initializationPrior* dropped. Replaced by a warning that names each non-default one.
    r'Initialisation priors in parameter table are not supported in PEtab v2\.',
)

#: v1 prior types stated on the parameter itself, on any ``parameterScale`` -> the v2 prior
#: that reads the same numbers the same way.
_V1_LINEAR_PRIORS = {'uniform': 'uniform', 'normal': 'normal', 'laplace': 'laplace'}

#: v1 prior types stated on the parameter's natural log, on any ``parameterScale`` -> the v2
#: natural-log family (location and scale of ``ln theta`` in both versions).
_V1_LN_PRIORS = {'logNormal': 'log-normal', 'logLaplace': 'log-laplace'}

#: v1 ``parameterScale*`` prior types -> the family they apply on the parameter's own scale.
_V1_SCALE_PRIORS = {
    'parameterScaleUniform': 'uniform',
    'parameterScaleNormal': 'normal',
    'parameterScaleLaplace': 'laplace',
}

#: The v1 default prior type (PEtab v1: a blank ``*PriorType`` cell means this one).
_V1_DEFAULT_PRIOR = 'parameterScaleUniform'


def petab1to2_preserve_scale(v1_yaml_path, out_dir):
    """Convert a PEtab **v1** problem to **v2**, preserving log estimation scales.

    Runs :func:`petab.v2.petab1to2`, then rewrites the converted v2 parameter table: every
    row whose v1 author declared an objective prior gets the v2 prior with the same
    distribution over the parameter (:func:`v2_prior_from_v1`, issue #893), and each other
    estimated parameter with a v1 ``parameterScale`` in {``log``, ``log10``} gains a
    v2-native ``priorDistribution = log-uniform`` over its bounds. ``lin``-scale parameters
    without a declared prior stay plain linear ``uniform_var``. Returns the ``Path`` to the
    converted v2 ``problem.yaml``.
    """
    try:
        import pandas as pd
        import petab.v1 as petab_v1
        import petab.v1.C as C1
        from petab.v2.petab1to2 import petab1to2
    except ImportError as e:
        raise PybnfError(
            'Scale-preserving PEtab v1->v2 conversion needs the petab library.',
            "Install the PEtab extra: pip install 'pybnf[petab]'.") from e

    v1_yaml_path = Path(v1_yaml_path)
    out_dir = Path(out_dir)

    # 1. Standard conversion. Silence only the petab1to2 warnings whose subject this function
    #    repairs below; any other drop or substitution petab1to2 reports reaches the caller
    #    (issue #893 -- a blanket filter here hid the log10 prior substitution).
    with warnings.catch_warnings():
        for message in _HANDLED_PETAB1TO2_WARNINGS:
            warnings.filterwarnings('ignore', message=message, category=UserWarning)
        petab1to2(str(v1_yaml_path), str(out_dir))
    v2_yaml = _sole_yaml(out_dir)

    # 2. The v1 estimation scales, per parameterId (estimated log/log10 only).
    v1_spec = petab_v1.yaml.load_yaml(str(v1_yaml_path))
    v1_pdf = petab_v1.get_parameter_df(str(v1_yaml_path.parent / v1_spec['parameter_file']))
    log_estimated = {
        str(pid)
        for pid, row in v1_pdf.iterrows()
        if str(row.get(C1.PARAMETER_SCALE, C1.LIN)) in _LOG_SCALES
        and _is_estimated(row.get(C1.ESTIMATE, 1))
    }
    # The objective prior each row's *v1 author wrote*, as its exact v2 equivalent. Reading
    # v1 is the only reliable way to tell a declared prior from the implicit `uniform` default
    # petab1to2 materializes into the v2 column (#548): after conversion the two are the same
    # cell, while in v1 the blank is still a blank. It is also the only way to get the prior
    # right, since petab1to2 carries some v1 priors over with the wrong numbers (#893).
    v2_priors = {
        pid: v2_prior_from_v1(pid, prior_type, prior_parameters,
                              row.get(C1.PARAMETER_SCALE, C1.LIN),
                              row[C1.LOWER_BOUND], row[C1.UPPER_BOUND])
        for pid, prior_type, prior_parameters, row in _v1_declared_objective_priors(v1_pdf)
    }
    _warn_dropped_initialization_priors(v1_pdf)
    v2_spec = petab_v1.yaml.load_yaml(str(v2_yaml))

    # 3. Priors. Replace petab1to2's translation of every declared prior with the exact one,
    #    then re-inject each bare log scale as a v2-native log-uniform prior over the bounds.
    #    The table is rewritten only when a cell changes, so a problem petab1to2 already
    #    converted exactly keeps petab1to2's bytes.
    if v2_priors or log_estimated:
        v2_param_path = out_dir / v2_spec['parameter_files'][0]
        v2_pdf = pd.read_csv(v2_param_path, sep='\t')
        changed = write_v2_priors(v2_pdf, v2_priors)
        if log_estimated:
            inject_log_uniform_priors(v2_pdf, log_estimated, set(v2_priors))
            changed = True
        if changed:
            v2_pdf.to_csv(v2_param_path, sep='\t', index=False)

    # 4. Re-inject the dropped observableTransformation as a preserved column (issue #499).
    #    v2 has no log10 noiseDistribution, so a log/log10 observable has no v2-native home;
    #    the importer reads this extra column to pick the noise family's additive scale.
    #    petab1to2 folded the same transformation into noiseDistribution as the natural-log
    #    family (petab >= 0.9.0; blank before), so that column is reset to the v1 linear base
    #    (normal / laplace) -- the importer refuses a log transformation over an already-log
    #    distribution as a contradiction (issue #679).
    scales = _v1_observable_scales(v1_yaml_path, v1_spec)
    if scales:
        transformations = {oid: transformation for oid, (transformation, _) in scales.items()}
        distributions = {oid: distribution for oid, (_, distribution) in scales.items()}
        for obs_file in v2_spec.get('observable_files', []):
            v2_obs_path = out_dir / obs_file
            v2_odf = pd.read_csv(v2_obs_path, sep='\t')
            inject_observable_transformations(v2_odf, transformations, distributions)
            v2_odf.to_csv(v2_obs_path, sep='\t', index=False)

    return v2_yaml


def v2_prior_from_v1(parameter_id, prior_type, prior_parameters, parameter_scale,
                     lower_bound, upper_bound):
    """The PEtab v2 prior with the same distribution over the parameter as a v1 objective prior.

    Returns ``(priorDistribution, priorParameters)`` as v2 table cells. ``prior_type`` is
    the v1 ``objectivePriorType`` (``None`` for a blank cell, which v1 reads as
    ``parameterScaleUniform``), ``prior_parameters`` the v1 ``objectivePriorParameters``
    text (``None`` for a blank cell), and ``parameter_scale`` the v1 ``parameterScale``.
    ADR-0073's 2026-09-25 addendum tabulates the mapping and what petab1to2 does instead.

    Numbers pass through verbatim where the two versions read them the same way. A ``log10``
    ``parameterScaleNormal`` / ``parameterScaleLaplace`` becomes the natural-log family with
    both numbers times ``ln 10``, since ``ln theta = ln 10 * log10 theta``. A log-scale
    ``parameterScaleUniform(a;b)`` becomes ``log-uniform`` over ``e^a;e^b`` or
    ``10^a;10^b``, because v2 states log-uniform bounds on the parameter itself. A blank
    ``parameterScaleUniform`` is v1's default, a uniform prior over the bounds on the
    parameter's scale, and is written over the bounds like petab1to2's own default. Raises
    ``PybnfError`` for a prior type, scale, or parameter count PEtab v1 does not define.
    """
    scale = _cell(parameter_scale) or 'lin'
    if scale not in ('lin', 'log', 'log10'):
        raise PybnfError(
            f"PEtab v1 parameter '{parameter_id}' has parameterScale {scale!r}; PEtab v1 "
            "defines only lin, log and log10.")
    prior_type = prior_type or _V1_DEFAULT_PRIOR
    family = _V1_SCALE_PRIORS.get(prior_type)
    if family is None and prior_type not in _V1_LINEAR_PRIORS | _V1_LN_PRIORS:
        raise PybnfError(
            f"PEtab v1 parameter '{parameter_id}' has objectivePriorType {prior_type!r}, "
            "which is not a PEtab v1 prior type.",
            hint='The v1 prior types are uniform, normal, laplace, logNormal, logLaplace, '
                 'parameterScaleUniform, parameterScaleNormal and parameterScaleLaplace.')
    if prior_parameters is None and family == 'uniform':
        # v1's default parameters: the bounds on the parameter's scale, which on any scale
        # is a (log-)uniform prior over the bounds themselves.
        return ('uniform' if scale == 'lin' else 'log-uniform',
                f'{lower_bound};{upper_bound}')
    a, b = _parse_v1_prior_parameters(parameter_id, prior_type, prior_parameters)
    verbatim = prior_parameters.strip()
    if prior_type in _V1_LINEAR_PRIORS:
        return _V1_LINEAR_PRIORS[prior_type], verbatim
    if prior_type in _V1_LN_PRIORS:
        return _V1_LN_PRIORS[prior_type], verbatim
    if scale == 'lin':
        return family, verbatim
    if family == 'uniform':
        to_theta = math.exp if scale == 'log' else (lambda x: 10.0 ** x)
        return 'log-uniform', f'{num(to_theta(a))};{num(to_theta(b))}'
    if scale == 'log':
        return f'log-{family}', verbatim
    ln10 = math.log(10.0)
    return f'log-{family}', f'{num(a * ln10)};{num(b * ln10)}'


def write_v2_priors(v2_pdf, v2_priors):
    """Write each ``{parameterId: (priorDistribution, priorParameters)}`` into a v2 table.

    Mutates ``v2_pdf`` (a v2 parameter :class:`pandas.DataFrame`) in place, adding either
    column if petab1to2 did not write it, and returns whether any cell changed. Every id must
    be a row of the table: the map is built from the v1 table petab1to2 converted, so a
    missing row means the two tables disagree, which is an error, not a row to skip.
    """
    import petab.v2.C as C2

    if not v2_priors:
        return False
    for col in (C2.PRIOR_DISTRIBUTION, C2.PRIOR_PARAMETERS):
        if col not in v2_pdf.columns:
            v2_pdf[col] = ''
        v2_pdf[col] = v2_pdf[col].astype('object')
    rows = {str(pid): i for i, pid in v2_pdf[C2.PARAMETER_ID].items()}
    missing = sorted(set(v2_priors) - set(rows))
    if missing:
        raise PybnfError(
            f"The converted PEtab v2 parameter table has no row for {', '.join(missing)}, "
            "whose PEtab v1 row declares an objective prior.")
    changed = False
    for pid, (distribution, parameters) in v2_priors.items():
        i = rows[pid]
        for col, value in ((C2.PRIOR_DISTRIBUTION, distribution),
                           (C2.PRIOR_PARAMETERS, parameters)):
            if str(v2_pdf.at[i, col]) != value:
                v2_pdf.at[i, col] = value
                changed = True
    return changed


def _v1_declared_objective_priors(v1_pdf):
    """Yield ``(parameterId, priorType, priorParameters, row)`` for each v1 row that
    declares an objective prior.

    A row declares one when either its ``objectivePriorType`` or its
    ``objectivePriorParameters`` cell is filled. A filled parameters cell under a blank type
    is v1's default type, ``parameterScaleUniform``, over those parameters: PEtab v1 reads it
    so, and petab1to2 does not (it writes a linear ``uniform`` over them). Blank cells are
    yielded as ``None``.
    """
    import petab.v1.C as C1

    for pid, row in v1_pdf.iterrows():
        prior_type = _cell(row.get(C1.OBJECTIVE_PRIOR_TYPE))
        prior_parameters = _cell(row.get(C1.OBJECTIVE_PRIOR_PARAMETERS))
        if prior_type is None and prior_parameters is None:
            continue
        yield str(pid), prior_type, prior_parameters, row


def _warn_dropped_initialization_priors(v1_pdf):
    """Warn, naming each one, about the v1 initialization priors the conversion drops.

    PEtab v2 has no initialization prior, and PyBNF has no channel for one: its importer
    starts a fit from ``nominalValue`` and draws initial points from the bounds. v1's default
    initialization prior, ``parameterScaleUniform`` over the bounds, is that same box on the
    parameter's scale, so a blank cell or a stated default loses nothing and is skipped, as
    is a parameter that is not estimated. Anything else is reported. Dropping it leaves the
    objective, the prior, and the posterior unchanged, so this is a warning, not a refusal.
    """
    import petab.v1.C as C1

    dropped = []
    for pid, row in v1_pdf.iterrows():
        if not _is_estimated(row.get(C1.ESTIMATE, 1)):
            continue
        prior_type = _cell(row.get(C1.INITIALIZATION_PRIOR_TYPE)) or _V1_DEFAULT_PRIOR
        prior_parameters = _cell(row.get(C1.INITIALIZATION_PRIOR_PARAMETERS))
        scale = _cell(row.get(C1.PARAMETER_SCALE)) or 'lin'
        if _is_v1_default_initialization(prior_type, prior_parameters, scale,
                                         row[C1.LOWER_BOUND], row[C1.UPPER_BOUND]):
            continue
        dropped.append(f"{pid} ({prior_type} {prior_parameters or ''} on {scale} scale)")
    if dropped:
        warnings.warn(
            'PEtab v2 has no initialization prior, so the conversion drops the v1 '
            f"initializationPriorType of {', '.join(dropped)}. The objective and its priors "
            'are unchanged; a PyBNF fit of the converted problem starts from nominalValue and '
            'draws initial points from each parameter\'s bounds instead.',
            UserWarning, stacklevel=3)


def _is_v1_default_initialization(prior_type, prior_parameters, scale, lower_bound,
                                  upper_bound):
    """Whether a v1 initialization prior is v1's default: uniform over the bounds on the
    parameter's scale (a blank or bounds-valued ``parameterScaleUniform``, or on a ``lin``
    scale a bounds-valued ``uniform``)."""
    if prior_type == _V1_DEFAULT_PRIOR and prior_parameters is None:
        return True
    if prior_parameters is None:
        return False
    if prior_type == _V1_DEFAULT_PRIOR:
        to_scale = {'lin': float, 'log': math.log, 'log10': math.log10}.get(scale)
    elif prior_type == 'uniform' and scale == 'lin':
        to_scale = float
    else:
        return False
    try:
        bounds = (to_scale(float(lower_bound)), to_scale(float(upper_bound)))
        stated = tuple(float(x) for x in prior_parameters.split(';'))
    except (TypeError, ValueError):
        return False
    return len(stated) == 2 and all(
        math.isclose(s, b, rel_tol=1e-12, abs_tol=1e-12) for s, b in zip(stated, bounds))


def _parse_v1_prior_parameters(parameter_id, prior_type, prior_parameters):
    """The two numbers of a v1 ``objectivePriorParameters`` cell (every v1 prior takes two)."""
    try:
        values = tuple(float(x) for x in (prior_parameters or '').split(';'))
    except ValueError:
        values = ()
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise PybnfError(
            f"PEtab v1 parameter '{parameter_id}': objective prior {prior_type} needs two "
            f"finite objectivePriorParameters separated by ';', got {prior_parameters!r}.")
    return values


def inject_log_uniform_priors(v2_pdf, log_estimated_ids, declared_prior_ids=None):
    """Give each v2 parameter row in ``log_estimated_ids`` a ``log-uniform`` prior in place.

    For every row whose ``parameterId`` is in ``log_estimated_ids`` **and** that carries no
    prior the v1 author declared, sets ``priorDistribution = log-uniform`` and
    ``priorParameters`` to its ``[lowerBound, upperBound]``. Rows with a declared prior
    (written from v1 by :func:`v2_prior_from_v1`) and rows not in the set are left untouched.
    Mutates and returns ``v2_pdf`` (a v2 parameter :class:`pandas.DataFrame`).

    ``declared_prior_ids`` is the set of parameter ids whose **v1** row declares an objective
    prior. It is required to get this right, because petab1to2 *materializes*
    PEtab v2's implicit default -- ``priorDistribution = uniform`` over the bounds -- into the
    converted table whenever the v1 table merely *has* a prior column, even an entirely empty
    one. After conversion a materialized default and a declared ``uniform`` are the same cell,
    so a v2-only check cannot separate them; asking v1 can.

    Without this argument the function falls back to "any prior blocks injection", which is
    safe but silently loses the log scale on exactly those problems. That regression cost
    `Zhao_QuantBiol2020` all 28 of its log10 parameters (its four v1 prior columns are present
    and 100% empty) and `Schwen_PONE2014` 24 of 25 (six real ``parameterScaleNormal`` priors,
    the rest blank), while `Giordano_Nature2020` -- whose v1 table has no prior column at all --
    converted correctly. The failure is silent: the objective stays right, the finite-difference
    gradient check still passes, and the fit merely searches a multi-decade parameter on a
    linear box, which presents as needing more starts.
    """
    import petab.v2.C as C2

    for col in (C2.PRIOR_DISTRIBUTION, C2.PRIOR_PARAMETERS):
        if col not in v2_pdf.columns:
            v2_pdf[col] = ''
        # petab1to2 emits an all-empty priorParameters as float64 (NaN); coerce to object
        # so the string cells below don't raise a dtype error. NaNs still write as blank.
        v2_pdf[col] = v2_pdf[col].astype('object')
    for i, row in v2_pdf.iterrows():
        pid = str(row[C2.PARAMETER_ID])
        if pid not in log_estimated_ids:
            continue
        if declared_prior_ids is None:
            # Legacy, v2-only reading: cannot tell a declared prior from a materialized
            # default, so anything present blocks. Kept only for callers that have no v1
            # table to consult.
            if _has_prior(row.get(C2.PRIOR_DISTRIBUTION)):
                continue
        elif pid in declared_prior_ids:
            continue  # the v1 author wrote this prior -- don't clobber it.
        lb, ub = row[C2.LOWER_BOUND], row[C2.UPPER_BOUND]
        v2_pdf.at[i, C2.PRIOR_DISTRIBUTION] = C2.LOG_UNIFORM
        v2_pdf.at[i, C2.PRIOR_PARAMETERS] = f'{lb}{C2.PARAMETER_SEPARATOR}{ub}'
    return v2_pdf


def inject_observable_transformations(v2_odf, transformations, distributions=None):
    """Add an ``observableTransformation`` column to a v2 observables DataFrame in place.

    ``transformations`` is a ``{observableId: 'log' | 'log10'}`` map (linear observables are
    absent from it -- they need no column, ``lin`` being the default). Every row whose
    ``observableId`` is in the map gets its transformation written; the rest get a blank cell.
    PEtab v2 dropped the column, so this is a **preserved extra column** the importer reads to
    select the noise family's additive scale (:mod:`pybnf.petab.observables`); other v2 tools
    ignore it. Mutates and returns ``v2_odf`` (a v2 observables :class:`pandas.DataFrame`).

    ``distributions``, when given, is the matching ``{observableId: 'normal' | 'laplace'}``
    map of each log observable's **v1** ``noiseDistribution`` -- its linear base family. Every
    row that receives a transformation also has its ``noiseDistribution`` reset to that base.
    petab1to2 folds the v1 transformation into this column as the natural-log family
    (``log10-normal`` -> ``log-normal``, petab >= 0.9.0; a blank cell before that), and the
    importer refuses a log transformation stacked over an already-log distribution, so the
    scale must be stated in the transformation column alone (issue #679). Rows not in the
    map keep whatever distribution petab1to2 wrote.
    """
    col = _OBSERVABLE_TRANSFORMATION_COLUMN
    if col not in v2_odf.columns:
        v2_odf[col] = ''
    # petab1to2 may emit an all-empty column as float64 (NaN); coerce to object so the string
    # cells below don't raise a dtype error. Untouched rows still write as blank.
    v2_odf[col] = v2_odf[col].astype('object')
    dcol = _NOISE_DISTRIBUTION_COLUMN
    if distributions:
        if dcol not in v2_odf.columns:
            v2_odf[dcol] = ''
        v2_odf[dcol] = v2_odf[dcol].astype('object')
    for i, row in v2_odf.iterrows():
        oid = str(row['observableId'])
        transformation = transformations.get(oid)
        if transformation is not None:
            v2_odf.at[i, col] = transformation
            if distributions and oid in distributions:
                v2_odf.at[i, dcol] = distributions[oid]
    return v2_odf


def _v1_observable_scales(v1_yaml_path, v1_spec):
    """``{observableId: ('log' | 'log10', 'normal' | 'laplace')}`` for every v1 observable
    with a log residual scale: its transformation and the linear base family of its v1
    ``noiseDistribution`` (blank -> ``normal``, the v1 default).

    Reads each v1 problem's observable file(s) and keeps only the ``log`` / ``log10``
    transformations (linear -- or an absent column -- is the v2 default and needs no
    re-injection). ``petab.v2.petab1to2`` drops the transformation column and folds it into
    ``noiseDistribution``, so the scale-preserving converter re-injects the transformation and
    restores the base distribution from what this returns (issues #499, #679), mirroring the
    parameterScale re-injection.
    """
    import pandas as pd
    import petab.v1 as petab_v1
    import petab.v1.C as C1

    scales = {}
    for problem in v1_spec.get('problems', []):
        for obs_file in problem.get('observable_files', []):
            odf = petab_v1.get_observable_df(str(Path(v1_yaml_path).parent / obs_file))
            if C1.OBSERVABLE_TRANSFORMATION not in odf.columns:
                continue
            for oid, row in odf.iterrows():
                value = row.get(C1.OBSERVABLE_TRANSFORMATION)
                transformation = str(value if value is not None else C1.LIN).strip().lower()
                if transformation not in _LOG_TRANSFORMATIONS:
                    continue
                dist = row.get(C1.NOISE_DISTRIBUTION)
                if dist is None or pd.isna(dist) or not str(dist).strip():
                    dist = C1.NORMAL
                scales[str(oid)] = (transformation, str(dist).strip().lower())
    return scales


def _sole_yaml(out_dir):
    """The single ``problem.yaml`` petab1to2 wrote into ``out_dir``."""
    yamls = sorted(Path(out_dir).glob('*.yaml'))
    if not yamls:
        raise PybnfError(f'petab1to2 wrote no problem.yaml into {out_dir}.')
    return yamls[0]


def _is_estimated(value):
    """v1 ``estimate`` cell (``1``/``0`` or truthy) -> whether the parameter is fit."""
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return str(value).strip().lower() in ('1', 'true')


def _has_prior(value):
    """Whether a v2 ``priorDistribution`` cell already names a prior (non-empty, non-NaN)."""
    return value is not None and str(value).strip().lower() not in ('', 'nan')


def _cell(value):
    """A table cell as stripped text, or ``None`` when it is blank or NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value).strip() or None
