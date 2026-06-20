"""The Gamma prior family (ADR-0010; PEtab v2 catalog parity, #417).

A two-parameter family with support ``(0, inf)``. The PEtab ``gamma`` priorParameters are
``(shape, scale)`` -> ``scipy.stats.gamma(a=shape, scale=scale)`` (verified against petab's
own ``v1.distributions.Gamma`` -- it is **shape + scale**, not shape + rate). Its config
values ``p1``/``p2`` are the shape and scale already in the parameter's scale, like the other
location/scale families; the parameter's PyBNF :class:`~pybnf.priors.Scale` shows up only in
the ``theta <-> u`` mapping the owning ``FreeParameter`` applies.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('gamma')
class Gamma(FrozenPrior):
    has_bounded_support = False

    def __init__(self, shape, gamma_scale):
        # a = shape, scale = scale (scipy's gamma is shape-scale; PEtab matches).
        self.frozen = stats.gamma(a=shape, scale=gamma_scale)

    @classmethod
    def build(cls, p1, p2, scale):
        """Build from config ``(shape, scale)`` -- given in-scale, untransformed."""
        return cls(shape=p1, gamma_scale=p2)
