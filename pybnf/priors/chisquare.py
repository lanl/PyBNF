"""The Chi-square prior family (ADR-0010; PEtab v2 catalog parity, #417).

A **one-parameter** family with support ``(0, inf)``. The PEtab ``chisquare`` priorParameters
is a single ``(dof)`` (degrees of freedom) -> ``scipy.stats.chi2(df=dof)`` (verified against
petab's own ``v1.distributions.ChiSquare``). Being one-parameter, ``build`` reads only ``p1``
and ignores ``p2`` (the grammar admits a single number for an unbounded-support family).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('chisquare')
class ChiSquare(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    n_params = 1
    field_names = ('dof',)

    def __init__(self, dof):
        self.frozen = stats.chi2(df=dof)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(dof,)`` -- one parameter; ``p2``/``p3`` are unused."""
        return cls(dof=p1)
