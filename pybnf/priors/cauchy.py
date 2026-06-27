"""The Cauchy prior family (ADR-0010; PEtab v2 catalog parity, #417).

A heavy-tailed location-scale family with infinite support on both sides. Its config values
``p1``/``p2`` are the location and scale already in the parameter's scale (so ``build`` does
not transform them, like Normal/Laplace); the scale shows up only in the ``theta <-> u``
mapping the owning ``FreeParameter`` applies. The PEtab ``cauchy`` priorParameters are
``(loc, scale)`` -> ``scipy.stats.cauchy(loc, scale)`` (verified against petab's own
``v1.distributions.Cauchy``).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('cauchy')
class Cauchy(FrozenPrior):
    has_bounded_support = False
    field_names = ('location', 'scale')

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.frozen = stats.cauchy(loc=loc, scale=scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(location, scale)`` -- given in-scale, untransformed."""
        return cls(loc=p1, scale=p2)

    def logpdf_jax(self, u):
        """The Cauchy log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf``: ``-log(pi) - log(scale) - log(1 + z^2)`` with
        ``z = (u - loc)/scale`` (``log1p`` for accuracy). Support is all of R (no
        wall) and the density is smooth, so ``jax.grad`` of the composed target is
        well-defined."""
        import jax.numpy as jnp
        z = (u - self.loc) / self.scale
        return -jnp.log(jnp.pi) - jnp.log(self.scale) - jnp.log1p(z * z)
