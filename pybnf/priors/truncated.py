"""The ``TruncatedPrior`` decorator: a family confined to a finite box in the
sampling space ``u`` (ADR-0020, issue #411).

PEtab v2 truncates a prior by the parameter's ``lowerBound``/``upperBound``, so a
``normal`` prior with finite bounds is a *truncated* normal. PyBNF's
unbounded-support families (Normal, Laplace, and their log forms) could not carry
bounds; this wraps any such family in a decorator that renormalizes the density
over ``[lo_u, hi_u]`` and samples *inside* the box by inverse-CDF.

Truncation is a hybrid concern (ADR-0010): it changes both the family's *support*
and its *normalization*. This decorator owns both halves of the distribution math
in one family-agnostic place, while the owning :class:`~pybnf.pset.FreeParameter`
owns the matching reflecting box (the bounds in ``theta``). The decorator works
for any inner family backed by a ``scipy.stats`` frozen distribution -- it reads
only ``cdf``/``ppf``/``logpdf`` -- so no per-family code is needed (the M2.3
ethos).

The bounds ``lo_u``/``hi_u`` are in sampling space ``u``: the owning
``FreeParameter`` maps the ``theta`` box through its ``Scale`` before wrapping, so
a log-scale parameter truncates in ``log10`` space (the same space its family and
proposal arithmetic already live in).
"""

import numpy as np

from .base import Prior


class TruncatedPrior(Prior):
    """An inner :class:`Prior` family confined to ``[lo_u, hi_u]`` in ``u``.

    Density is the inner family's, renormalized over the box
    (``logpdf = inner.logpdf - log Z`` inside, ``-inf`` outside). Sampling and the
    inverse CDF go through the truncated inverse-CDF, so a draw lands inside the
    box by construction (not an unbounded draw folded back). ``Z`` is
    parameter-independent, so it cancels in the MCMC acceptance ratio and in MAP
    optimization (ADR-0003); it is kept only so reported densities are genuinely
    those of the truncated distribution.
    """

    has_prior = True
    has_bounded_support = True

    def __init__(self, inner, lo_u, hi_u):
        if inner.frozen is None:
            raise ValueError(
                "TruncatedPrior requires an inner family with a scipy frozen "
                "distribution (cannot truncate a NoPrior).")
        if not (lo_u < hi_u):
            raise ValueError(
                f"TruncatedPrior needs lo_u < hi_u, got [{lo_u}, {hi_u}].")
        self.inner = inner
        self.lo_u = lo_u
        self.hi_u = hi_u
        # The truncated mass, in u-space. cdf(hi) - cdf(lo) > 0 because the inner
        # families have support on all of R (or, for the log forms, all of R in u).
        self._cdf_lo = float(inner.frozen.cdf(lo_u))
        self._z = float(inner.frozen.cdf(hi_u)) - self._cdf_lo
        if not self._z > 0.0:
            raise ValueError(
                f"TruncatedPrior box [{lo_u}, {hi_u}] carries no prior mass "
                f"(Z={self._z}); the bounds exclude the family's bulk.")
        self._log_z = float(np.log(self._z))
        # Back-compat passthrough for FreeParameter._distribution: the underlying
        # (untruncated) scipy frozen, which exposes logpdf/rvs as callers expect.
        self.frozen = inner.frozen

    def logpdf(self, u):
        """Renormalized log density: ``inner.logpdf(u) - log Z`` in the box, else
        ``-inf``. The ``-log Z`` constant cancels for inference but makes the
        reported value the true truncated-density log-pdf."""
        if u < self.lo_u or u > self.hi_u:
            return -np.inf
        return self.inner.logpdf(u) - self._log_z

    def rvs(self, rng):
        """Draw one sample in the box by inverse-CDF: ``ppf(U)`` with ``U ~ U(0,1)``
        drawn from the caller's Generator (so the draw lands inside the box by
        construction, not by folding an unbounded draw back in)."""
        return self.ppf(float(rng.random()))

    def ppf(self, q):
        """Truncated inverse CDF over ``[lo_u, hi_u]``: map ``q in [0, 1]`` onto the
        retained CDF interval, then invert with the inner family's ``ppf``."""
        return float(self.inner.frozen.ppf(self._cdf_lo + q * self._z))

    def support(self):
        return (self.lo_u, self.hi_u)
