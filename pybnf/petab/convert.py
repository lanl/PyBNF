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
petab1to2 already preserves scale where it is attached to an *objective prior*
(``parameterScale{Normal,Uniform,Laplace}`` -> ``log-normal`` / ``log-laplace`` / ...); it
is only the **bare** estimation scale it drops. We re-inject that scale in the **v2-native**
form: each bare log/log10 estimated parameter gets ``priorDistribution = log-uniform`` over
its bounds. PyBNF -- and any PEtab v2 tool -- then reads a log search (a ``log-uniform``
prior maps to a ``loguniform_var`` on the Log10 scale; :mod:`pybnf.petab.parameters`).
PyBNF's optimiser objective *excludes* the prior, so the log-uniform prior sets only the
search scale and initial sampling, not the objective -- the fit stays the pure-MLE problem
v1 specified.

**observableTransformation (the residual scale; issue #499, ADR-0073).** A v1 observable with
``observableTransformation = log10`` fits the residual on the log10 scale (with the
change-of-variables Jacobian) -- a *different objective* from the linear residual, not just
a different search. PEtab v2 removed the column and folded transformation into
``noiseDistribution`` as **natural-log** ``log-normal`` / ``log-laplace`` prefixes, with **no
log10 form**; ``petab1to2`` therefore drops a v1 ``log10`` transformation entirely (it
downgrades ``log10-normal`` to a blank ``noiseDistribution``), so the observable imports as a
linear Gaussian and the fit optimises the wrong objective. Since v2 has no faithful
representation, we re-inject ``observableTransformation`` as a **preserved extra column** on
the v2 observables table: PyBNF's importer reads it to select the noise family's additive
scale (``lin`` / ``log10`` / ``log``; :mod:`pybnf.petab.observables`,
:mod:`pybnf.petab.import_`), and other PEtab v2 tools ignore the unknown column (it passes v2
lint). This is directly parallel to the parameterScale re-injection above -- the same
"re-add the scale petab1to2 dropped" migration, on the observable axis.

This is the migration ``petab1to2`` should offer as an opt-in; it lives here as an
explicit, named converter so :func:`pybnf.petab.import_job` stays a pure v2 importer with
no reach-back to v1 in the read path.
"""

import warnings
from pathlib import Path

from ..printing import PybnfError

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


def petab1to2_preserve_scale(v1_yaml_path, out_dir):
    """Convert a PEtab **v1** problem to **v2**, preserving log estimation scales.

    Runs :func:`petab.v2.petab1to2`, then rewrites the converted v2 parameter table so each
    estimated v1 ``parameterScale`` in {``log``, ``log10``} that carries no prior gains a
    v2-native ``priorDistribution = log-uniform`` over its bounds. Returns the ``Path`` to
    the converted v2 ``problem.yaml``.

    Parameters petab1to2 already gave a prior (a v1 ``parameterScale*`` objective prior) are
    left untouched; ``lin``-scale parameters stay plain linear ``uniform_var``.
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

    # 1. Standard conversion. Silence the "Parameter scales are not supported" warning --
    #    re-adding that scale is exactly this function's job.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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
    # Which rows carried a prior the *v1 author wrote*. This is the only reliable way to tell a
    # declared prior from the implicit `uniform` default petab1to2 materializes into the v2
    # column: after conversion the two are the same cell. Read it from v1, where the blank is
    # still a blank.
    declared_priors = {
        str(pid)
        for pid, row in v1_pdf.iterrows()
        if _has_prior(row.get(C1.OBJECTIVE_PRIOR_TYPE))
    }
    v2_spec = petab_v1.yaml.load_yaml(str(v2_yaml))

    # 3. Re-inject the dropped parameter scale as a v2-native log-uniform prior over the bounds.
    if log_estimated:
        v2_param_path = out_dir / v2_spec['parameter_files'][0]
        v2_pdf = pd.read_csv(v2_param_path, sep='\t')
        inject_log_uniform_priors(v2_pdf, log_estimated, declared_priors)
        v2_pdf.to_csv(v2_param_path, sep='\t', index=False)

    # 4. Re-inject the dropped observableTransformation as a preserved column (issue #499).
    #    v2 has no log10 noiseDistribution, so a log/log10 observable has no v2-native home;
    #    the importer reads this extra column to pick the noise family's additive scale.
    transformations = _v1_observable_transformations(v1_yaml_path, v1_spec)
    if transformations:
        for obs_file in v2_spec.get('observable_files', []):
            v2_obs_path = out_dir / obs_file
            v2_odf = pd.read_csv(v2_obs_path, sep='\t')
            inject_observable_transformations(v2_odf, transformations)
            v2_odf.to_csv(v2_obs_path, sep='\t', index=False)

    return v2_yaml


def inject_log_uniform_priors(v2_pdf, log_estimated_ids, declared_prior_ids=None):
    """Give each v2 parameter row in ``log_estimated_ids`` a ``log-uniform`` prior in place.

    For every row whose ``parameterId`` is in ``log_estimated_ids`` **and** that carries no
    prior the v1 author declared, sets ``priorDistribution = log-uniform`` and
    ``priorParameters`` to its ``[lowerBound, upperBound]``. Rows with a declared prior (a
    scale petab1to2 already folded into one) and rows not in the set are left untouched.
    Mutates and returns ``v2_pdf`` (a v2 parameter :class:`pandas.DataFrame`).

    ``declared_prior_ids`` is the set of parameter ids that carried a **v1**
    ``objectivePriorType``. It is required to get this right, because petab1to2 *materializes*
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


def inject_observable_transformations(v2_odf, transformations):
    """Add an ``observableTransformation`` column to a v2 observables DataFrame in place.

    ``transformations`` is a ``{observableId: 'log' | 'log10'}`` map (linear observables are
    absent from it -- they need no column, ``lin`` being the default). Every row whose
    ``observableId`` is in the map gets its transformation written; the rest get a blank cell.
    PEtab v2 dropped the column, so this is a **preserved extra column** the importer reads to
    select the noise family's additive scale (:mod:`pybnf.petab.observables`); other v2 tools
    ignore it. Mutates and returns ``v2_odf`` (a v2 observables :class:`pandas.DataFrame`).
    """
    col = _OBSERVABLE_TRANSFORMATION_COLUMN
    if col not in v2_odf.columns:
        v2_odf[col] = ''
    # petab1to2 may emit an all-empty column as float64 (NaN); coerce to object so the string
    # cells below don't raise a dtype error. Untouched rows still write as blank.
    v2_odf[col] = v2_odf[col].astype('object')
    for i, row in v2_odf.iterrows():
        transformation = transformations.get(str(row['observableId']))
        if transformation is not None:
            v2_odf.at[i, col] = transformation
    return v2_odf


def _v1_observable_transformations(v1_yaml_path, v1_spec):
    """``{observableId: 'log' | 'log10'}`` for every v1 observable with a log residual scale.

    Reads each v1 problem's observable file(s) and keeps only the ``log`` / ``log10``
    transformations (linear -- or an absent column -- is the v2 default and needs no
    re-injection). ``petab.v2.petab1to2`` drops this column, so the scale-preserving converter
    re-injects what this returns (issue #499), mirroring the parameterScale re-injection.
    """
    import petab.v1 as petab_v1
    import petab.v1.C as C1

    transformations = {}
    for problem in v1_spec.get('problems', []):
        for obs_file in problem.get('observable_files', []):
            odf = petab_v1.get_observable_df(str(Path(v1_yaml_path).parent / obs_file))
            if C1.OBSERVABLE_TRANSFORMATION not in odf.columns:
                continue
            for oid, row in odf.iterrows():
                value = row.get(C1.OBSERVABLE_TRANSFORMATION)
                transformation = str(value if value is not None else C1.LIN).strip().lower()
                if transformation in _LOG_TRANSFORMATIONS:
                    transformations[str(oid)] = transformation
    return transformations


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
