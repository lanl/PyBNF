"""The Exponential prior family (ADR-0010; PEtab v2 catalog parity, #417).

A **one-parameter** family with support ``(0, inf)``. The PEtab ``exponential``
priorParameters is a single ``(scale)`` -> ``scipy.stats.expon(scale=scale)`` (verified
against petab's own ``v1.distributions.Exponential`` -- the parameter is the **scale**
``1/rate``, not the rate). Being one-parameter, ``build`` reads only ``p1`` and ignores
``p2`` (the grammar admits a single number for an unbounded-support family, ADR-0010/#417).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('exponential')
class Exponential(FrozenPrior):
    has_bounded_support = False
    n_params = 1
    field_names = ('scale',)

    def __init__(self, exp_scale):
        self.frozen = stats.expon(scale=exp_scale)

    @classmethod
    def build(cls, p1, p2, scale):
        """Build from config ``(scale,)`` -- one parameter; ``p2`` is unused."""
        return cls(exp_scale=p1)
