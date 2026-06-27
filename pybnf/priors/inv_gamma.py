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
        self.shape = shape
        self.ig_scale = ig_scale
        self.frozen = stats.invgamma(a=shape, scale=ig_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(shape, scale)`` -- given in-scale, untransformed."""
        return cls(shape=p1, ig_scale=p2)

    def logpdf_jax(self, u):
        """The inverse-gamma log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf`` on ``(0, inf)``:
        ``a log(scale) - gammaln(a) - (a+1) log u - scale/u`` with ``a = shape``.
        Both ``log u`` and ``scale/u`` blow up at ``u <= 0``, so the safe-``u``
        double-``where`` evaluates the body at ``1.0`` outside the support and masks
        it to ``-inf``, keeping ``jax.grad`` finite (``0``) there (ADR-0059 item 4).
        HMC at the ``u=0`` boundary awaits item 5's bijection."""
        import jax.numpy as jnp
        from jax.scipy.special import gammaln
        a, scale = self.shape, self.ig_scale
        inside = u > 0.0
        su = jnp.where(inside, u, 1.0)
        val = a * jnp.log(scale) - gammaln(a) - (a + 1.0) * jnp.log(su) - scale / su
        return jnp.where(inside, val, -jnp.inf)
