"""Gaussian observation noise (ADR-0011)."""

import numpy as np

from .base import NoiseModel
from .location import MEDIAN
from .scale import LINEAR

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


class Gaussian(NoiseModel):
    """Gaussian (normal) observation noise, configured by two of the three axes:
    the scale its noise is **additive on** and the **location interpretation** of
    the prediction. The defaults (``LINEAR``, ``MEDIAN``) are ordinary additive
    error where the prediction is the median -- symmetric, so all locations
    coincide and the axes are trivial. Reconfigured as (``LOG10``, ``MEDIAN``) it is
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

    noise_params = ('sigma',)

    def __init__(self, additive_on=LINEAR, location=MEDIAN):
        self.additive_on = additive_on
        self.location = location

    def with_location(self, location):
        return type(self)(additive_on=self.additive_on, location=location)

    def mean_offset(self, noise):
        """The Gaussian moment correction ``ln(base)*sigma**2/2`` (0 on the linear
        scale): the mean of ``base**N(mu, sigma)`` is ``base**(mu + ln(base)*sigma**2/2)``,
        so recovering ``mu`` from a prediction taken to be that mean subtracts this in
        additive space (ADR-0022)."""
        return self.additive_on.ln_base * noise ** 2. / 2.

    def _mu(self, prediction, noise):
        """The additive-space location parameter for ``prediction``."""
        return self.additive_on.forward(prediction) - self.location.offset(self, noise)

    def data_fit(self, prediction, observation, noise, extra=None):
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return 1. / (2. * noise ** 2.) * residual ** 2.

    def log_normalizer(self, noise):
        return np.log(noise)

    def _density_constant(self):
        # The Gaussian's ½ log(2π): the part of -log N that is constant in the
        # parameters, which the sampler never needed (it cancels in accept ratios)
        # but a normalized density (log_density, for LOO/WAIC) keeps. Restoring it
        # makes log_density match scipy.stats.norm.logpdf (and, on a log scale plus
        # the Jacobian, scipy.stats.lognorm.logpdf).
        return _HALF_LOG_2PI
