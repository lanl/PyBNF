"""Support-aware unconstraining bijections for gradient-based sampling (ADR-0059 item 5).

The gradient-free samplers (``am`` / ``dream`` / ``p_dream``) keep a draw inside a
constrained prior support with **reflecting bounds** -- a Metropolis device with no HMC
analogue. NUTS, by contrast, samples an *unbounded* momentum-driven trajectory; at a
``-inf`` support wall (a positive-support prior's ``u <= 0``, ``beta``'s ``[0, 1]``, a
:class:`~pybnf.priors.truncated.TruncatedPrior` box) the leapfrog integrator diverges.

The fix is a **change of variables**: HMC samples an unconstrained ``z in R``, mapped to the
prior's support by a monotone bijection ``u = b(z)``, and the target gains the Jacobian term
``log|b'(z)|``. Because ``b`` lands strictly inside the open support for every finite ``z``,
the ``-inf`` wall (and the divergence it caused) is never reached.

The transform is a property of the support *shape*, not the family, so it lives here once and
keys purely on ``support()`` (the ``(lo, hi)`` finiteness pattern) -- the same family-agnostic
ethos as :mod:`~pybnf.priors.truncated`. Four cases cover the whole catalog:

==================  ==========================  ===============  =====================
support ``(lo,hi)``  ``u = b(z)``                ``z = b^{-1}(u)``  ``log|b'(z)|``
==================  ==========================  ===============  =====================
``(-inf, inf)``      ``z``                       ``u``            ``0``
``(lo, inf)``        ``lo + exp(z)``             ``log(u - lo)``  ``z``
``(-inf, hi)``       ``hi - exp(z)``             ``log(hi - u)``  ``z``
``(lo, hi)``         ``lo + (hi-lo)*sigmoid(z)``  ``logit((u-lo)/(hi-lo))``  ``log(hi-lo)+logsig(z)+logsig(-z)``
==================  ==========================  ===============  =====================

Each bijector exposes a numpy ``to_unconstrained`` / ``to_constrained`` / ``logdet`` (host-side
init seeding and draw-writing) and a JAX-traceable ``to_constrained_jax`` / ``logdet_jax`` (the
differentiable target the ``hmc`` sampler composes), mirroring :class:`~pybnf.priors.scale.Scale`'s
``inverse`` / ``inverse_jax`` split. ``jax`` is imported lazily so importing this module (and the
whole ``priors`` package) never requires the optional ``pybnf[jax]`` extra.
"""

import numpy as np
from scipy import special as _sp

#: How far inside an open boundary an init point is nudged before ``to_unconstrained``
#: (``log`` / ``logit`` of an on-the-boundary ``u`` is ``-inf``). The latin-hypercube
#: seed sits strictly inside support in the common case; this only guards the corner where
#: a quantile lands exactly on an edge, so the NUTS start position stays finite.
_EDGE_EPS = 1e-9


def bijector_for_support(lo, hi):
    """The unconstraining bijector for a prior whose sampling-space support is ``(lo, hi)``.

    Keys on the finiteness of each endpoint (the support *shape*), so one factory serves
    every family: an unbounded support is the identity, one finite endpoint is a log/exp
    half-line map, two finite endpoints are the logit/sigmoid box map."""
    lo_finite, hi_finite = np.isfinite(lo), np.isfinite(hi)
    if not lo_finite and not hi_finite:
        return IdentityBijector()
    if lo_finite and not hi_finite:
        return LowerBoundedBijector(float(lo))
    if hi_finite and not lo_finite:
        return UpperBoundedBijector(float(hi))
    return BoxBijector(float(lo), float(hi))


class Bijector:
    """A monotone unconstraining map ``u = b(z)`` from ``R`` onto a prior's support.

    ``to_constrained`` is ``b`` (``z -> u``); ``to_unconstrained`` is ``b^{-1}`` (``u -> z``,
    used once to seed NUTS from the latin-hypercube start point); ``logdet`` is the
    change-of-variables Jacobian ``log|b'(z)|`` added to the HMC target. The ``*_jax`` peers
    are the JAX-traceable forms the differentiable log-density uses."""

    def to_unconstrained(self, u):
        """Map a support-space ``u`` to the unconstrained ``z`` (numpy; init seeding)."""
        raise NotImplementedError

    def to_constrained(self, z):
        """Map an unconstrained ``z`` back to support-space ``u`` (numpy; draw-writing)."""
        raise NotImplementedError

    def logdet(self, z):
        """``log|du/dz|`` at ``z`` (numpy; un-Jacobians the recorded posterior density)."""
        raise NotImplementedError

    def to_constrained_jax(self, z):
        """JAX-traceable :meth:`to_constrained` (the differentiable target)."""
        raise NotImplementedError

    def logdet_jax(self, z):
        """JAX-traceable :meth:`logdet` (the differentiable Jacobian term)."""
        raise NotImplementedError


class IdentityBijector(Bijector):
    """Unbounded support ``(-inf, inf)``: ``u = z``, no Jacobian. The real-support families
    (normal/laplace/cauchy/student_t/gumbel/logistic, and any ``lognormal_var`` whose prior is
    a normal in ``u``) need no reparameterization, so HMC samples them in ``u`` directly."""

    def to_unconstrained(self, u):
        return u

    def to_constrained(self, z):
        return z

    def logdet(self, z):
        return 0.0

    def to_constrained_jax(self, z):
        return z

    def logdet_jax(self, z):
        import jax.numpy as jnp
        return jnp.zeros_like(z)


class LowerBoundedBijector(Bijector):
    """Half-line support ``(lo, inf)``: ``u = lo + exp(z)``, ``log|b'(z)| = z``. Covers the
    positive families (gamma/exponential/chisquare/rayleigh/weibull/inv_gamma/the half-* scale
    priors), whose ``lo`` is ``0``. ``exp(z) > 0`` for every finite ``z``, so ``u > lo``
    strictly -- the ``u <= lo`` wall is unreachable."""

    def __init__(self, lo):
        self.lo = lo

    def to_unconstrained(self, u):
        return np.log(max(u - self.lo, _EDGE_EPS))

    def to_constrained(self, z):
        return self.lo + np.exp(z)

    def logdet(self, z):
        return float(z)

    def to_constrained_jax(self, z):
        import jax.numpy as jnp
        return self.lo + jnp.exp(z)

    def logdet_jax(self, z):
        return z


class UpperBoundedBijector(Bijector):
    """Half-line support ``(-inf, hi)``: ``u = hi - exp(z)``, ``log|b'(z)| = z``. The mirror of
    :class:`LowerBoundedBijector` (no catalog family is upper-only by default, but a one-sided
    upper :class:`~pybnf.priors.truncated.TruncatedPrior` lands here); ``u < hi`` strictly."""

    def __init__(self, hi):
        self.hi = hi

    def to_unconstrained(self, u):
        return np.log(max(self.hi - u, _EDGE_EPS))

    def to_constrained(self, z):
        return self.hi - np.exp(z)

    def logdet(self, z):
        return float(z)

    def to_constrained_jax(self, z):
        import jax.numpy as jnp
        return self.hi - jnp.exp(z)

    def logdet_jax(self, z):
        return z


class BoxBijector(Bijector):
    """Finite box support ``(lo, hi)``: ``u = lo + (hi-lo)*sigmoid(z)``, with
    ``log|b'(z)| = log(hi-lo) + logsigmoid(z) + logsigmoid(-z)``. Covers ``uniform`` /
    ``loguniform`` boxes, ``beta``'s ``[0, 1]``, and a two-sided
    :class:`~pybnf.priors.truncated.TruncatedPrior`. ``sigmoid(z) in (0, 1)`` for every finite
    ``z``, so ``u`` stays strictly inside the box -- both walls are unreachable."""

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.width = hi - lo

    def to_unconstrained(self, u):
        # logit of the in-box fraction; clip a hair off the edges so an on-the-boundary
        # init quantile maps to a finite z rather than +-inf.
        frac = (u - self.lo) / self.width
        frac = min(max(frac, _EDGE_EPS), 1.0 - _EDGE_EPS)
        return float(_sp.logit(frac))

    def to_constrained(self, z):
        return self.lo + self.width * _sp.expit(z)

    def logdet(self, z):
        # log(width) + log sigmoid(z) + log sigmoid(-z); log_expit is the overflow-safe
        # log-sigmoid (scipy >= 1.8).
        return float(np.log(self.width) + _sp.log_expit(z) + _sp.log_expit(-z))

    def to_constrained_jax(self, z):
        import jax.nn as jnn
        return self.lo + self.width * jnn.sigmoid(z)

    def logdet_jax(self, z):
        import jax.nn as jnn
        import jax.numpy as jnp
        return jnp.log(self.width) + jnn.log_sigmoid(z) + jnn.log_sigmoid(-z)
