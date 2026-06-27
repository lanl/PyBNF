"""The Exponential prior family (ADR-0010; PEtab v2 catalog parity, #417).

A **one-parameter** family with support ``(0, inf)``. The PEtab ``exponential``
priorParameters is a single ``(scale)`` -> ``scipy.stats.expon(scale=scale)`` (verified
against petab's own ``v1.distributions.Exponential`` -- the parameter is the **scale**
``1/rate``, not the rate). Being one-parameter, ``build`` reads only ``p1`` and ignores
``p2`` (the grammar admits a single number for an unbounded-support family, ADR-0010/#417).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('exponential')
class Exponential(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    n_params = 1
    field_names = ('scale',)

    def __init__(self, exp_scale):
        self.exp_scale = exp_scale
        self.frozen = stats.expon(scale=exp_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(scale,)`` -- one parameter; ``p2``/``p3`` are unused."""
        return cls(exp_scale=p1)

    def logpdf_jax(self, u):
        """The Exponential log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf`` on ``(0, inf)``: ``-u/scale - log(scale)``. The body is
        linear in ``u`` (no ``log u``), so its gradient is finite everywhere; the
        ``where`` only injects the ``-inf`` wall outside the support, and the
        ``where`` autodiff rule keeps the masked-out gradient at ``0`` (no NaN). HMC
        behaviour at the ``u=0`` boundary awaits item 5's bijection."""
        import jax.numpy as jnp
        return jnp.where(u > 0.0, -u / self.exp_scale - jnp.log(self.exp_scale), -jnp.inf)
