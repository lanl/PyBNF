"""The Gamma prior family (ADR-0010; PEtab v2 catalog parity, #417).

A two-parameter family with support ``(0, inf)``. The PEtab ``gamma`` priorParameters are
``(shape, scale)`` -> ``scipy.stats.gamma(a=shape, scale=scale)`` (verified against petab's
own ``v1.distributions.Gamma`` -- it is **shape + scale**, not shape + rate). Its config
values ``p1``/``p2`` are the shape and scale already in the parameter's scale, like the other
location/scale families; the parameter's PyBNF :class:`~pybnf.priors.Scale` shows up only in
the ``theta <-> u`` mapping the owning ``FreeParameter`` applies.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('gamma')
class Gamma(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    field_names = ('shape', 'scale')

    def __init__(self, shape, gamma_scale):
        # a = shape, scale = scale (scipy's gamma is shape-scale; PEtab matches).
        self.shape = shape
        self.gamma_scale = gamma_scale
        self.frozen = stats.gamma(a=shape, scale=gamma_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(shape, scale)`` -- given in-scale, untransformed."""
        return cls(shape=p1, gamma_scale=p2)

    def logpdf_jax(self, u):
        """The Gamma log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf`` on ``(0, inf)``:
        ``(a-1) log u - u/scale - gammaln(a) - a log(scale)`` with ``a = shape``.
        The half-line support uses the safe-``u`` double-``where`` -- evaluate the
        body at an in-support stand-in (``1.0``) outside the support, then mask to
        ``-inf`` -- so ``log u`` never sees ``u <= 0`` and ``jax.grad`` stays finite
        (``0``) outside, which NUTS leapfrog steps require (ADR-0059 item 4). HMC
        quality *at* the hard ``u=0`` boundary awaits the unconstraining bijection
        (item 5); until then the divergence/R-hat gate flags it honestly."""
        import jax.numpy as jnp
        from jax.scipy.special import gammaln
        a, scale = self.shape, self.gamma_scale
        inside = u > 0.0
        su = jnp.where(inside, u, 1.0)
        val = (a - 1.0) * jnp.log(su) - su / scale - gammaln(a) - a * jnp.log(scale)
        return jnp.where(inside, val, -jnp.inf)
