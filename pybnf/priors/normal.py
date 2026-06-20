"""The Normal prior family (ADR-0010).

A location-scale family with infinite support. Its config values ``p1``/``p2``
are the mean and standard deviation **already in the parameter's scale** -- so
``build`` does not transform them (this is why ``normal_var`` and
``lognormal_var`` historically constructed the identical ``stats.norm(p1, p2)``;
the scale shows up only in the ``theta <-> u`` mapping the ``FreeParameter``
applies around this family).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('normal')
class Normal(FrozenPrior):
    has_bounded_support = False

    def __init__(self, loc, sigma):
        self.frozen = stats.norm(loc=loc, scale=sigma)

    @classmethod
    def build(cls, p1, p2, scale):
        """Build from config ``(mean, sd)`` -- given in-scale, untransformed."""
        return cls(loc=p1, sigma=p2)
