"""The Beta prior family (ADR-0010; ADR-0057, #438 item 1).

A two-parameter family with finite support ``[0, 1]`` -- the canonical prior for a **fraction**
or **probability** (a binding saturation, a branching ratio, a Hill cooperativity normalized to
[0,1]). Its config values ``alpha``/``beta`` are the two shape parameters ->
``scipy.stats.beta(a=alpha, b=beta)``: ``alpha == beta == 1`` is uniform on [0,1], larger values
concentrate the mass.

Unlike :class:`Uniform`, Beta is **not** a box whose bounds are its config values, so
``has_bounded_support`` stays ``False`` -- its ``p1``/``p2`` are shapes, not the support, and it
samples from its own density rather than a latin-hypercube box. Its support floor is 0
(``support_lo_u``), so a one-sided truncation floors there (ADR-0047); truncating to a
sub-interval of [0,1] via finite ``lower``/``upper`` works through the family-agnostic
:class:`TruncatedPrior`. (The upper support 1 is the frozen distribution's own -- a value above
it already has zero density -- so it needs no separate floor attribute.)
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('beta')
class Beta(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support [0, 1]: a one-sided truncation floors at 0 (ADR-0047)
    field_names = ('alpha', 'beta')

    def __init__(self, alpha, beta):
        self.frozen = stats.beta(a=alpha, b=beta)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(alpha, beta)`` -- given in-scale, untransformed."""
        return cls(alpha=p1, beta=p2)
