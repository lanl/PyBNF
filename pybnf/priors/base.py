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
    #: The family's natural lower support endpoint in sampling space ``u`` (the lower
    #: edge of ``support()``, a family constant independent of the distribution's
    #: parameters). ``-inf`` for the doubly-unbounded families; the positive-support
    #: families (gamma/exponential/chisquare/rayleigh, the half-* scale priors,
    #: inv_gamma/weibull, and beta's ``[0,1]``) override to ``0.0``. The owning
    #: ``FreeParameter``'s ``Scale.inverse`` maps it to the theta-space floor a one-sided
    #: truncation measures bounds against (ADR-0047).
    support_lo_u = -np.inf
    #: How many config numbers the family's parameterization takes -- ``2`` for the
    #: location/scale/bounds families; the one-parameter families (exponential/chisquare/
    #: rayleigh, the half-* scale priors) override to ``1`` so the positional grammar admits a
    #: single number (ADR-0010/#417); the three-parameter families (student_t) override to
    #: ``3``. A ``n_params >= 3`` family is authored only through the new-era ``parameter:``
    #: record -- the legacy positional ``*_var`` grammar carries at most two numbers, so
    #: ``var_keyword_grammar`` omits it (ADR-0057).
    n_params = 2
    #: The config field names for the family's distribution parameters, in ``build()`` order
    #: -- the new-era ``parameter:`` record names each one (ADR-0043), so a positional
    #: ``p1 p2`` becomes ``mean: .. , sd: ..``. Concrete families override; the length must
    #: match ``n_params`` (the record builds ``p1``/``p2``/``p3`` from the first three).
    field_names = ('p1', 'p2')
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
    field_names = ()
    frozen = None

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Factory matching the family ``build`` signature; ``p1``/``p2``/``p3``/``scale``
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
