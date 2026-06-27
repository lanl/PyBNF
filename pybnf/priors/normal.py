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
    field_names = ('mean', 'sd')

    def __init__(self, loc, sigma):
        self.loc = loc
        self.sigma = sigma
        self.frozen = stats.norm(loc=loc, scale=sigma)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(mean, sd)`` -- given in-scale, untransformed."""
        return cls(loc=p1, sigma=p2)

    def logpdf_jax(self, u):
        """The Gaussian log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf`` this family uses on the gradient-free path. Written by
        hand (rather than ``jax.scipy.stats.norm``) so the HMC slice needs no JAX
        statistics import beyond ``jax.numpy``; it is smooth everywhere, so
        ``jax.grad`` of the composed target is well-defined."""
        import jax.numpy as jnp
        z = (u - self.loc) / self.sigma
        return -0.5 * z * z - jnp.log(self.sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
