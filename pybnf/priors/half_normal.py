"""The half-normal prior family (ADR-0010; ADR-0057, #438 item 1).

A **one-parameter** positive family: the right half of a zero-centered normal, with support
``(0, inf)``. The standard weakly-informative **scale** prior for hierarchical models (Gelman) --
reach for it on a standard deviation or any non-negative magnitude. Its single config value is
the ``scale`` (the underlying normal's sigma) already in the parameter's scale ->
``scipy.stats.halfnorm(scale=scale)``.

Equivalent to a zero-mean normal one-sidedly truncated at 0 (ADR-0047): truncating ``N(0, s)``
to ``[0, inf)`` renormalizes the density by ``1/2``, which is exactly ``halfnorm``'s doubled
density (``halfnorm.logpdf(x) == log 2 + norm(0, s).logpdf(x)``). This file exposes it as a
named, self-normalizing keyword so a modeler writes ``half_normal`` directly instead of spelling
a truncated normal.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('half_normal')
class HalfNormal(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    n_params = 1
    field_names = ('scale',)

    def __init__(self, hn_scale):
        self.frozen = stats.halfnorm(scale=hn_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(scale,)`` -- one parameter; ``p2``/``p3`` are unused."""
        return cls(hn_scale=p1)
