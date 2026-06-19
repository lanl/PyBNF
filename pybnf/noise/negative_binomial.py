"""Negative-binomial observation noise (ADR-0011, ADR-0031)."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc, loggamma

from .base import NoiseModel
from .location import MEDIAN


def _mean_for_median(prediction, r):
    """Solve for the mean ``mu`` of ``NB(mean=mu, dispersion=r)`` whose **continuous**
    0.5-quantile equals ``prediction`` -- the negative-binomial median realization
    (issue #419, ADR-0031's "every means every").

    The continuous CDF is ``F(x; mu, r) = I_p(r, x + 1)`` with ``p = r / (r + mu)``
    (``scipy.special.betainc``, the regularized incomplete beta), which is exactly
    ``scipy.stats.nbinom.cdf(k, r, p)`` at integer ``k`` but smooth in its second
    argument -- so we use it instead of the discrete ``nbinom.ppf`` step to keep the
    objective continuous in the prediction (the optimizers need it). ``F`` is strictly
    decreasing in ``mu`` (larger mean shifts the distribution right, lowering the mass
    at or below the prediction), so there is a unique ``mu`` placing the median at the
    prediction, found by a bounded root-find.

    A prediction is a count median, so it is clamped to ``>= 0``; ``mu = 0`` gives
    ``p = 1`` and ``F = 1 > 0.5``, while ``mu -> inf`` gives ``F -> 0``, bracketing the
    root in ``[0, hi]``.
    """
    target = max(prediction, 0.0)

    def gap(mu):
        p = r / (r + mu)
        return betainc(r, target + 1.0, p) - 0.5

    # gap(0) == 0.5 > 0; grow the upper bound until the median exceeds the target.
    hi = max(target, 1.0)
    while gap(hi) > 0.0:
        hi *= 2.0
        if hi > 1e15:
            return hi  # pathological prediction; give up at a huge mean
    return brentq(gap, 0.0, hi)


class NegBinomial(NoiseModel):
    """Negative-binomial observation noise for count data. The dispersion ``r`` is the
    noise parameter (``neg_bin`` reads it from the config constant ``neg_bin_r``;
    ``neg_bin_dynamic`` from the ``r__FREE`` free parameter).

    The **location** axis (ADR-0011/0031) sets which distributional summary the
    prediction is taken to be. The default is ``MEDIAN`` -- median is the universal
    prediction-centering default for *every* noise family (ADR-0031, "every means
    every"), true in code at the constructor like Gaussian/Laplace. ``MEDIAN``
    interprets the prediction as the 0.5-quantile and solves for the mean placing the
    continuous median there (issue #419). ``MEAN`` is the native parameterization (the
    prediction *is* the mean) -- the legacy ``neg_bin`` objfuncs pin it explicitly to
    stay frozen-mean. Unlike Gaussian/Laplace, the count family is **not additive on a
    scale**, so it owns this realization directly rather than going through
    ``location.py``'s additive-offset abstraction -- it reuses the ``MEAN``/``MEDIAN``
    markers, not the ``offset`` math.

    A negative observed count contributes nothing (the count-domain guard). A PMF is
    self-normalizing, so there is no separable normalizer (``log_normalizer`` stays 0)
    and the full ``-logpmf`` lives in ``data_fit``.
    """

    def __init__(self, location=MEDIAN):
        self.location = location

    def with_location(self, location):
        return NegBinomial(location=location)

    def _mean(self, prediction, noise):
        """The distribution mean for ``prediction`` under the location interpretation:
        the prediction itself for ``MEAN``, the median inversion for ``MEDIAN``."""
        if self.location is MEDIAN:
            return _mean_for_median(prediction, noise)
        return prediction

    def data_fit(self, prediction, observation, noise):
        if observation < 0:
            return 0
        mean = self._mean(prediction, noise)
        prob = np.clip(noise / (noise + mean), 1e-10, 1 - 1e-10)
        assert isinstance(noise, float)
        # log of the negative-binomial PMF P(observation | r=noise, prob)
        # == scipy.stats.nbinom.logpmf(observation, noise, prob).
        log_pmf = loggamma(observation + noise) - loggamma(observation + 1) - loggamma(noise) \
            + noise * np.log(prob) + observation * np.log(1 - prob)
        # A PMF is <= 1, so log_pmf <= 0; PyBNF minimizes the negative
        # log-likelihood -log_pmf >= 0.
        return -log_pmf
