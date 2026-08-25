"""Dependency-free quantiles for confidence thresholds.

PyBNF's production loop never imports scipy (ADR-0007), so the two quantiles its confidence
statements need are computed here. Both a profile-likelihood run (which asks where a profile
crosses its ``Delta chi2`` threshold) and an experimental design (which reports the confidence
interval an information matrix implies) read the same threshold from the same confidence level, so
they share this rather than each carrying a copy.
"""

import math

from .printing import PybnfError


def normal_quantile(p):
    """Standard-normal inverse cumulative distribution (the probit) via Acklam's rational
    approximation, refined by one Halley step against :func:`math.erf`.

    Accurate to full double precision after the refinement, far more than a chi-square threshold
    needs. ``0 < p < 1``."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # One Halley refinement using the exact erf-based CDF.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


def chi2_quantile_1dof(confidence, key='confidence'):
    """The chi-square (1 degree of freedom) quantile at probability ``confidence`` -- the profile
    ``Delta chi2`` threshold (Raue et al. 2009), and the same threshold a design's predicted
    intervals are quoted at.

    A single parameter has one degree of freedom, and ``chi2_1 = Z**2`` with ``Z ~ N(0, 1)``, so
    ``P(chi2_1 <= x) = 2*Phi(sqrt(x)) - 1`` and the quantile is
    ``Phi^-1((1 + confidence) / 2)**2`` (0.95 gives 3.8415). ``key`` names the configuration key
    the confidence level came from, so a rejected value points at the right setting."""
    if not (0.0 < confidence < 1.0):
        raise PybnfError("%s must be strictly between 0 and 1, got %r." % (key, confidence))
    z = normal_quantile(0.5 * (1.0 + confidence))
    return z * z
