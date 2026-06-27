"""The Rayleigh prior family (ADR-0010; PEtab v2 catalog parity, #417).

A **one-parameter** family with support ``(0, inf)``. The PEtab ``rayleigh`` priorParameters
is a single ``(scale)`` -> ``scipy.stats.rayleigh(scale=scale)`` (verified against petab's own
``v1.distributions.Rayleigh``). Being one-parameter, ``build`` reads only ``p1`` and ignores
``p2`` (the grammar admits a single number for an unbounded-support family).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('rayleigh')
class Rayleigh(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    n_params = 1
    field_names = ('scale',)

    def __init__(self, ray_scale):
        self.frozen = stats.rayleigh(scale=ray_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(scale,)`` -- one parameter; ``p2``/``p3`` are unused."""
        return cls(ray_scale=p1)
