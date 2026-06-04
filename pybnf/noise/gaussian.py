"""Gaussian observation noise (ADR-0011)."""

import numpy as np

from .base import NoiseModel


class Gaussian(NoiseModel):
    """Gaussian (normal) observation noise, additive on the linear scale, with the
    prediction interpreted as the mean. Symmetric, so mean = median = mode and the
    location axis is trivial.

    ``data_fit`` is the weighted squared residual ``(pred - obs)^2 / (2 sigma^2)``;
    ``log_normalizer`` is ``log sigma`` (PyBNF drops the further parameter-
    independent ``0.5 log 2*pi`` constant). ``chi_sq`` (sigma fixed, from the data's
    ``_SD`` column) sums only ``data_fit``; ``chi_sq_dynamic`` (sigma a free
    parameter) sums the full ``nll``, retaining ``log sigma``.
    """

    def data_fit(self, prediction, observation, noise):
        return 1. / (2. * noise ** 2.) * (prediction - observation) ** 2.

    def log_normalizer(self, noise):
        return np.log(noise)
