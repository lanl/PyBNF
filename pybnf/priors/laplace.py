"""The Laplace prior family (ADR-0010).

A heavier-tailed location-scale family (PEtab parity), unbounded support. Like
Normal, its config values ``p1``/``p2`` are the location and scale ``b`` already
in the parameter's scale, so ``build`` does not transform them; the scale shows
up only in the ``theta <-> u`` mapping the ``FreeParameter`` applies around the
family.

This file is the M2.3 seam proof: registering one family yields ``laplace_var``
(linear) and ``loglaplace_var`` (log10) end-to-end -- grammar, keyword map,
prior density and sampling -- with no other change anywhere in the codebase.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('laplace')
class Laplace(FrozenPrior):
    has_bounded_support = False
    field_names = ('location', 'scale')

    def __init__(self, loc, b):
        # b is the Laplace scale (diversity) parameter, i.e. scipy's `scale`.
        self.frozen = stats.laplace(loc=loc, scale=b)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(location, b)`` -- given in-scale, untransformed."""
        return cls(loc=p1, b=p2)
