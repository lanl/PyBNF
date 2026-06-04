"""Negative-binomial observation noise (ADR-0011)."""

import numpy as np
from scipy.special import loggamma

from .base import NoiseModel


class NegBinomial(NoiseModel):
    """Negative-binomial observation noise for count data, with the prediction
    interpreted as the distribution's mean. The dispersion ``r`` is the noise
    parameter (``neg_bin`` reads it from the config constant ``neg_bin_r``;
    ``neg_bin_dynamic`` from the ``r__FREE`` free parameter).

    A negative observed count contributes nothing (the count-domain guard). A PMF
    is self-normalizing, so there is no separable normalizer (``log_normalizer``
    stays 0) and the full ``-logpmf`` lives in ``data_fit``.
    """

    def data_fit(self, prediction, observation, noise):
        if observation < 0:
            return 0
        prob = np.clip(noise / (noise + prediction), 1e-10, 1 - 1e-10)
        assert isinstance(noise, float)
        # log of the negative-binomial PMF P(observation | r=noise, prob)
        # == scipy.stats.nbinom.logpmf(observation, noise, prob).
        log_pmf = loggamma(observation + noise) - loggamma(observation + 1) - loggamma(noise) \
            + noise * np.log(prob) + observation * np.log(1 - prob)
        # A PMF is <= 1, so log_pmf <= 0; PyBNF minimizes the negative
        # log-likelihood -log_pmf >= 0.
        return -log_pmf
