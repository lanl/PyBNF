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
    field_names = ('lower', 'upper')

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.frozen = stats.uniform(loc=lo, scale=hi - lo)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from raw ``theta`` bounds, mapped into sampling space ``u``."""
        return cls(lo=scale.forward(p1), hi=scale.forward(p2))

    def logpdf(self, u):
        return float(self.frozen.logpdf(u))

    def rvs(self, rng):
        return self.frozen.rvs(random_state=rng)

    def ppf(self, q):
        return self.lo + q * (self.hi - self.lo)

    def support(self):
        return (self.lo, self.hi)

    def logpdf_jax(self, u):
        """The uniform-box log-density in JAX (ADR-0059): the constant
        ``-log(hi - lo)`` inside ``[lo, hi]`` (in sampling space ``u``), ``-inf``
        outside. Both ``jnp.where`` branches are constant in ``u``, so the
        derivative is exactly ``0`` inside the box and the ``-inf`` wall injects
        no NaN gradient (the standard ``where`` autodiff pitfall is avoided because
        neither branch depends on ``u``).

        This first HMC slice supports the box prior as a *flat* prior over a wide
        support -- the closed-form benchmark posteriors sit far inside the box, so
        NUTS never reaches the walls. Divergence-free sampling *at* a constrained
        boundary (the unconstraining bijection) is the deferred follow-on (ADR-0059
        item 5)."""
        import jax.numpy as jnp
        inside = (u >= self.lo) & (u <= self.hi)
        return jnp.where(inside, -jnp.log(self.hi - self.lo), -jnp.inf)
