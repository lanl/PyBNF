"""The Uniform prior family (ADR-0010).

A bounded family: its config values ``p1``/``p2`` are the raw ``theta`` bounds,
which ``build`` maps **into** the sampling space ``u`` via the scale (so
``loguniform_var`` becomes a uniform box over ``[log10(p1), log10(p2)]``). The
finite support is the family's defining property -- it makes a Uniform parameter
eligible for reflecting bounds and for latin-hypercube stratification.

``ppf`` is computed as ``lo + q*(hi - lo)`` (not via scipy) to reproduce the
existing latin-hypercube rescale arithmetic bit-for-bit.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import Prior


@register_prior_family('uniform')
class Uniform(Prior):
    has_bounded_support = True

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.frozen = stats.uniform(loc=lo, scale=hi - lo)

    @classmethod
    def build(cls, p1, p2, scale):
        """Build from raw ``theta`` bounds, mapped into sampling space ``u``."""
        return cls(lo=scale.forward(p1), hi=scale.forward(p2))

    def logpdf(self, u):
        return float(self.frozen.logpdf(u))

    def rvs(self):
        return self.frozen.rvs()

    def ppf(self, q):
        return self.lo + q * (self.hi - self.lo)

    def support(self):
        return (self.lo, self.hi)
