"""The inverse-gamma prior family (ADR-0010; ADR-0057, #438 item 1).

A two-parameter positive family with support ``(0, inf)`` -- the classic **conjugate prior for a
variance** in a Gaussian model. Its config values are ``(shape, scale)`` ->
``scipy.stats.invgamma(a=shape, scale=scale)`` (scipy's inverse-gamma is shape-scale, matching
the conventional ``InvGamma(alpha, beta)`` with ``alpha = shape``, ``beta = scale``). Like the
other positive families it floors at 0 (``support_lo_u``), so a one-sided truncation lands there
(ADR-0047).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('inv_gamma')
class InvGamma(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    field_names = ('shape', 'scale')

    def __init__(self, shape, ig_scale):
        self.frozen = stats.invgamma(a=shape, scale=ig_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(shape, scale)`` -- given in-scale, untransformed."""
        return cls(shape=p1, ig_scale=p2)
