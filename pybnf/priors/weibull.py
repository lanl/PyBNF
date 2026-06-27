"""The Weibull prior family (ADR-0010; ADR-0057, #438 item 1).

A two-parameter positive family with support ``(0, inf)`` -- a flexible lifetime/time-to-event
prior whose shape interpolates between an exponential (``shape == 1``) and increasingly
bell-shaped, light-tailed forms (``shape > 1``). Its config values are ``(shape, scale)`` ->
``scipy.stats.weibull_min(c=shape, scale=scale)`` (the ``weibull_min`` / Frechet-min convention,
``loc`` fixed at 0). Floors at 0 (``support_lo_u``), so a one-sided truncation lands there
(ADR-0047).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('weibull')
class Weibull(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    field_names = ('shape', 'scale')

    def __init__(self, shape, wb_scale):
        self.frozen = stats.weibull_min(c=shape, scale=wb_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(shape, scale)`` -- given in-scale, untransformed."""
        return cls(shape=p1, wb_scale=p2)
