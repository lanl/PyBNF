"""Laplace observation noise (ADR-0011, ADR-0021)."""

import numpy as np

from .base import NoiseModel
from .location import MEDIAN
from .scale import LINEAR


class Laplace(NoiseModel):
    """Laplace (double-exponential) observation noise -- the heavy-tailed sibling of
    Gaussian. Its negative log-likelihood penalizes the residual *linearly*
    (``|prediction - observation| / b``) rather than quadratically, so it is the
    maximum-likelihood model behind least-absolute-deviation fitting and is robust
    to outliers; PEtab v2's ``noiseDistribution = laplace``.

    Configured by the same two axes as ``Gaussian`` -- the scale its noise is
    additive on and the location interpretation of the prediction. Laplace is
    symmetric, so mean and median coincide and the location axis is trivial at the
    default (``LINEAR``, ``MEDIAN``) -- the only combination exercised today, as
    Gaussian started. ``data_fit`` is then ``|prediction - observation| / b`` with
    the scale ``b`` as the noise parameter; ``log_normalizer`` is ``log(2 b)``.
    With a fixed ``b`` the caller drops the normalizer; with a free ``b`` (the
    ``laplace`` objfunc's ``b__FREE``) it keeps the full ``nll`` -- the ``log(2 b)``
    term is exactly what prevents the fit from driving ``b -> inf``. Value oracle:
    ``scipy.stats.laplace.logpdf``.
    """

    def __init__(self, additive_on=LINEAR, location=MEDIAN):
        self.additive_on = additive_on
        self.location = location

    def with_location(self, location):
        return type(self)(additive_on=self.additive_on, location=location)

    def _mu(self, prediction, noise):
        """The additive-space location parameter for ``prediction``."""
        return self.additive_on.forward(prediction) - self.location.offset(self.additive_on, noise)

    def data_fit(self, prediction, observation, noise):
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return abs(residual) / noise

    def log_normalizer(self, noise):
        return np.log(2. * noise)
