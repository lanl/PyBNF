"""The Gumbel prior family (ADR-0010; ADR-0057, #438 item 1).

A two-parameter location-scale family with infinite support on both sides -- the extreme-value
(maximum) distribution, a right-skewed alternative to the Normal for a quantity governed by a
maximum. Its config values are the ``(location, scale)`` already in the parameter's scale ->
``scipy.stats.gumbel_r(loc=location, scale=scale)`` (the right-skewed ``gumbel_r`` /
maximum convention). Symmetric handling to Normal/Cauchy: ``build`` does not transform the
values; the parameter's :class:`~pybnf.priors.Scale` shows up only in the ``theta <-> u`` mapping.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('gumbel')
class Gumbel(FrozenPrior):
    has_bounded_support = False
    field_names = ('location', 'scale')

    def __init__(self, loc, scale):
        self.frozen = stats.gumbel_r(loc=loc, scale=scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(location, scale)`` -- given in-scale, untransformed."""
        return cls(loc=p1, scale=p2)
