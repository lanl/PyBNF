"""The ``Prior`` abstraction: a distribution family evaluated entirely in the
sampling space ``u`` (ADR-0010).

A ``Prior`` is **scale-agnostic** -- it knows nothing about ``theta`` or
``log10``. The owning ``FreeParameter`` holds the ``Scale`` and applies the
``theta <-> u`` transform, calling ``prior.logpdf(scale.forward(theta))`` and
``scale.inverse(prior.rvs(rng))``. This keeps each family's math pure and the
scale in one place (ADR-0003).

Concrete families (``Normal``, ``Uniform``, ...) live one-per-file and
self-register with ``@register_prior_family``. ``NoPrior`` is the first-class
null-object for ``var``/``logvar`` Simplex start points: a free parameter with a
scale but no distribution.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..printing import PybnfError


class Prior(ABC):
    """A distribution family operating in the sampling space ``u``.

    Subclasses expose ``logpdf``/``rvs``/``ppf`` in ``u`` and report their
    ``support`` (in ``u``) and ``has_bounded_support``. ``frozen`` is the
    underlying ``scipy.stats`` frozen distribution (or ``None`` for ``NoPrior``),
    surfaced for ``FreeParameter._distribution`` back-compat.
    """

    #: Whether the family has a proper distribution (``False`` only for ``NoPrior``).
    has_prior = True
    #: Whether the family's support is finite -- drives reflecting-bounds
    #: eligibility and latin-hypercube participation. ``Uniform`` overrides.
    has_bounded_support = False
    #: How many config numbers the family's ``*_var`` keyword takes (``p1 p2`` for the
    #: location/scale/bounds families; the one-parameter exponential/chisquare/rayleigh
    #: override to 1, so the grammar admits a single number -- ADR-0010/#417).
    n_params = 2
    #: The underlying scipy frozen distribution, or ``None``.
    frozen = None

    @abstractmethod
    def logpdf(self, u):
        """Log prior density at sampling-space value ``u``."""

    @abstractmethod
    def rvs(self, rng):
        """Draw one sample in sampling space ``u`` using ``rng``.

        ``rng`` is the caller's :class:`numpy.random.Generator`; it is passed to
        scipy as ``random_state`` so prior sampling draws from the algorithm's
        seeded Generator rather than NumPy's legacy global RNG.
        """

    @abstractmethod
    def ppf(self, q):
        """Inverse CDF at quantile ``q`` (in ``[0, 1]``), in sampling space ``u``."""

    @abstractmethod
    def support(self):
        """The ``(lo, hi)`` support in sampling space ``u`` (may be infinite)."""


class FrozenPrior(Prior):
    """A :class:`Prior` backed entirely by a ``scipy.stats`` frozen distribution in the
    sampling space ``u`` (ADR-0010).

    Every location/scale/shape family (Normal, Laplace, Cauchy, Gamma, Exponential,
    ChiSquare, Rayleigh) has the *same* density/sampling/inverse-CDF/support shape -- a thin
    delegation to its frozen distribution -- so it lives here once. A subclass sets
    ``self.frozen`` in ``__init__`` (and ``has_bounded_support`` as a class attribute) and
    provides the family-specific ``build`` classmethod. ``NoPrior`` (no distribution) and
    ``Uniform`` (a custom latin-hypercube ``ppf``) do not use this base.
    """

    def logpdf(self, u):
        return float(self.frozen.logpdf(u))

    def rvs(self, rng):
        return self.frozen.rvs(random_state=rng)

    def ppf(self, q):
        return float(self.frozen.ppf(q))

    def support(self):
        return tuple(self.frozen.support())


class NoPrior(Prior):
    """The absence of a prior: a ``var``/``logvar`` Simplex start point.

    Contributes nothing to the log prior, cannot be sampled, and has no support.
    It still pairs with a ``Scale`` (``logvar`` is ``Log10``); the scale lives on
    the ``FreeParameter``, not here.
    """

    has_prior = False
    has_bounded_support = False
    frozen = None

    @classmethod
    def build(cls, p1, p2, scale):
        """Factory matching the family ``build`` signature; ``p1``/``p2``/``scale``
        are ignored -- a no-prior parameter carries only a start value."""
        return cls()

    def logpdf(self, u):
        return 0.0

    def rvs(self, rng):
        raise PybnfError("Parameter does not have a sampling distribution")

    def ppf(self, q):
        raise PybnfError("Parameter does not have a sampling distribution")

    def support(self):
        return (-np.inf, np.inf)
