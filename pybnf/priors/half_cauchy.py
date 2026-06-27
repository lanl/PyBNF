"""The half-Cauchy prior family (ADR-0010; ADR-0057, #438 item 1).

A **one-parameter** positive family: the right half of a zero-centered Cauchy, with support
``(0, inf)``. The standard *heavy-tailed* weakly-informative **scale** prior (Gelman 2006) --
the half-Cauchy's fat tail places more prior mass on large scales than the half-normal, the
usual choice for a hierarchical standard deviation when you want to be permissive. Its single
config value is the ``scale`` (the Cauchy's half-width at half-maximum) already in the
parameter's scale -> ``scipy.stats.halfcauchy(scale=scale)``.

Like the half-normal, this is a zero-centered Cauchy one-sidedly truncated at 0 (ADR-0047),
exposed as a named, self-normalizing keyword.
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('half_cauchy')
class HalfCauchy(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    n_params = 1
    field_names = ('scale',)

    def __init__(self, hc_scale):
        self.hc_scale = hc_scale
        self.frozen = stats.halfcauchy(scale=hc_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(scale,)`` -- one parameter; ``p2``/``p3`` are unused."""
        return cls(hc_scale=p1)

    def logpdf_jax(self, u):
        """The half-Cauchy log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf`` on ``[0, inf)``:
        ``log 2 - log(pi) - log(scale) - log(1 + (u/scale)^2)``. The body (a
        ``log1p`` of ``u^2``) has a finite gradient everywhere, so the ``where`` only
        adds the ``-inf`` wall below ``0`` (masked-out gradient stays ``0``). HMC at
        the ``u=0`` boundary awaits item 5's bijection."""
        import jax.numpy as jnp
        sig = self.hc_scale
        z = u / sig
        val = jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log(sig) - jnp.log1p(z * z)
        return jnp.where(u > 0.0, val, -jnp.inf)
