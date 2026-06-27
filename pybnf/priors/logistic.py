"""The logistic prior family (ADR-0010; ADR-0057, #438 item 1).

A two-parameter location-scale family with infinite support -- a symmetric, slightly
heavier-tailed sibling of the Normal (the distribution behind logistic regression). Its config
values are the ``(location, scale)`` already in the parameter's scale ->
``scipy.stats.logistic(loc=location, scale=scale)``; like Normal/Cauchy, ``build`` does not
transform them and the parameter's :class:`~pybnf.priors.Scale` enters only through the
``theta <-> u`` mapping.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('logistic')
class Logistic(FrozenPrior):
    has_bounded_support = False
    field_names = ('location', 'scale')

    def __init__(self, loc, scale):
        self.frozen = stats.logistic(loc=loc, scale=scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(location, scale)`` -- given in-scale, untransformed."""
        return cls(loc=p1, scale=p2)
