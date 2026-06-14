"""Gaussian observation noise (ADR-0011)."""

import numpy as np

from .base import NoiseModel
from .location import MEAN
from .scale import LINEAR


class Gaussian(NoiseModel):
    """Gaussian (normal) observation noise, configured by two of the three axes:
    the scale its noise is **additive on** and the **location interpretation** of
    the prediction. The defaults (``LINEAR``, ``MEAN``) are ordinary additive
    error where the prediction is the mean -- symmetric, so all locations coincide
    and the axes are trivial. Reconfigured as (``LOG10``, ``MEDIAN``) it is
    (log10) lognormal error with the prediction as the median (the ``lognormal``
    objfunc); that one reconfiguration -- adding **no** new distribution family --
    proves the axes are orthogonal and live (ADR-0011, the analogue of Laplace
    proving the prior seam). (``LN`` gives the natural-log lognormal density;
    ADR-0022.)

    ``data_fit`` is the squared residual in the additive space,
    ``(mu - forward(obs))^2 / (2 sigma^2)``, where ``mu`` is the additive-space
    location parameter; ``log_normalizer`` is ``log sigma``. With a fixed sigma the
    caller drops the normalizer (and, on the log scale, the parameter-independent
    Jacobian) -- so ``chi_sq``/``lognormal`` sum only ``data_fit`` -- while a free
    sigma keeps the full ``nll`` (``chi_sq_dynamic``).
    """

    def __init__(self, additive_on=LINEAR, location=MEAN):
        self.additive_on = additive_on
        self.location = location

    def _mu(self, prediction, noise):
        """The additive-space location parameter for ``prediction``."""
        return self.additive_on.forward(prediction) - self.location.offset(self.additive_on, noise)

    def data_fit(self, prediction, observation, noise):
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return 1. / (2. * noise ** 2.) * residual ** 2.

    def log_normalizer(self, noise):
        return np.log(noise)
