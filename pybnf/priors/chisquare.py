"""The Chi-square prior family (ADR-0010; PEtab v2 catalog parity, #417).

A **one-parameter** family with support ``(0, inf)``. The PEtab ``chisquare`` priorParameters
is a single ``(dof)`` (degrees of freedom) -> ``scipy.stats.chi2(df=dof)`` (verified against
petab's own ``v1.distributions.ChiSquare``). Being one-parameter, ``build`` reads only ``p1``
and ignores ``p2`` (the grammar admits a single number for an unbounded-support family).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('chisquare')
class ChiSquare(FrozenPrior):
    has_bounded_support = False
    support_lo_u = 0.0   # support (0, inf): a one-sided truncation floors here (ADR-0047)
    n_params = 1
    field_names = ('dof',)

    def __init__(self, dof):
        self.dof = dof
        self.frozen = stats.chi2(df=dof)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(dof,)`` -- one parameter; ``p2``/``p3`` are unused."""
        return cls(dof=p1)

    def logpdf_jax(self, u):
        """The chi-square log-density in JAX (ADR-0059), oracle-equal to the scipy
        ``frozen.logpdf`` on ``(0, inf)``. Chi-square is ``Gamma(df/2, scale=2)``, so
        with ``k = df/2`` the density is ``(k-1) log u - u/2 - gammaln(k) - k log 2``.
        The half-line support uses the safe-``u`` double-``where`` so ``log u`` never
        sees ``u <= 0`` and ``jax.grad`` stays finite (``0``) outside the support
        (ADR-0059 item 4); the ``hmc`` sampler reparameterizes ``u = exp(z)`` so the
        ``u=0`` boundary is sampled divergence-free (ADR-0059 item 5)."""
        import jax.numpy as jnp
        from jax.scipy.special import gammaln
        k = self.dof / 2.0
        inside = u > 0.0
        su = jnp.where(inside, u, 1.0)
        val = (k - 1.0) * jnp.log(su) - su / 2.0 - gammaln(k) - k * jnp.log(2.0)
        return jnp.where(inside, val, -jnp.inf)
