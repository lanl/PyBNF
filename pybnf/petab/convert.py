"""Scale-preserving PEtab v1 -> v2 conversion (wraps ``petab.v2.petab1to2``).

``petab.v2.petab1to2`` drops the v1 ``parameterScale`` column -- PEtab v2 removed it, on
the view that estimation scale is an *optimiser* concern, not part of the problem spec --
and only *warns* ("Parameter scales are not supported in PEtab v2"). For a parameter that
was ``parameterScale = log10`` **without** an objective prior (the common case: a
maximum-likelihood fit over a multi-decade kinetic parameter), the converted v2 problem is
read as *linear* ``uniform_var`` over the raw bounds. That is the same argmin, but a
vastly harder, worse-conditioned optimisation than the log10 search the modeller specified
-- and running the problem *as specified* is the whole point of a benchmark.

petab1to2 already preserves scale where it is attached to an *objective prior*
(``parameterScale{Normal,Uniform,Laplace}`` -> ``log-normal`` / ``log-laplace`` / ...); it
is only the **bare** estimation scale it drops. This module wraps the standard converter
and re-injects that scale in the **v2-native** form: each bare log/log10 estimated
parameter gets ``priorDistribution = log-uniform`` over its bounds. PyBNF -- and any PEtab
v2 tool -- then reads a log search (a ``log-uniform`` prior maps to a ``loguniform_var`` on
the Log10 scale; :mod:`pybnf.petab.parameters`). PyBNF's optimiser objective *excludes* the
prior, so the log-uniform prior sets only the search scale and initial sampling, not the
objective -- the fit stays the pure-MLE problem v1 specified.

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
    if not log_estimated:
        return v2_yaml

    # 3. Re-inject the dropped scale as a v2-native log-uniform prior over the bounds.
    v2_spec = petab_v1.yaml.load_yaml(str(v2_yaml))
    v2_param_path = out_dir / v2_spec['parameter_files'][0]
    v2_pdf = pd.read_csv(v2_param_path, sep='\t')
    inject_log_uniform_priors(v2_pdf, log_estimated)
    v2_pdf.to_csv(v2_param_path, sep='\t', index=False)
    return v2_yaml


def inject_log_uniform_priors(v2_pdf, log_estimated_ids):
    """Give each v2 parameter row in ``log_estimated_ids`` a ``log-uniform`` prior in place.

    For every row whose ``parameterId`` is in ``log_estimated_ids`` **and** that carries no
    prior yet, sets ``priorDistribution = log-uniform`` and ``priorParameters`` to its
    ``[lowerBound, upperBound]``. Rows with an existing prior (a scale petab1to2 already
    folded into one) and rows not in the set are left untouched. Mutates and returns
    ``v2_pdf`` (a v2 parameter :class:`pandas.DataFrame`).
    """
    import petab.v2.C as C2

    for col in (C2.PRIOR_DISTRIBUTION, C2.PRIOR_PARAMETERS):
        if col not in v2_pdf.columns:
            v2_pdf[col] = ''
        # petab1to2 emits an all-empty priorParameters as float64 (NaN); coerce to object
        # so the string cells below don't raise a dtype error. NaNs still write as blank.
        v2_pdf[col] = v2_pdf[col].astype('object')
    for i, row in v2_pdf.iterrows():
        if str(row[C2.PARAMETER_ID]) not in log_estimated_ids:
            continue
        if _has_prior(row.get(C2.PRIOR_DISTRIBUTION)):
            continue  # petab1to2 already carried this scale into a prior -- don't clobber.
        lb, ub = row[C2.LOWER_BOUND], row[C2.UPPER_BOUND]
        v2_pdf.at[i, C2.PRIOR_DISTRIBUTION] = C2.LOG_UNIFORM
        v2_pdf.at[i, C2.PRIOR_PARAMETERS] = f'{lb}{C2.PARAMETER_SEPARATOR}{ub}'
    return v2_pdf


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
