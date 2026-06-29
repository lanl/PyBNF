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

    def d_mean_offset_d_noise(self, noise):
        """``d(mean_offset)/d sigma = ln(base)*sigma`` (0 on the linear scale, where ``ln(base)``
        is 0). The term that couples the offset to the noise scale when a free sigma centers a MEAN
        on a log scale -- folded into the estimated-sigma gradient column (#385)."""
        return self.additive_on.ln_base * noise

    def _mu(self, prediction, noise):
        """The additive-space location parameter for ``prediction``."""
        return self.additive_on.forward(prediction) - self.location.offset(self, noise)

    def data_fit(self, prediction, observation, noise, extra=None):
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return 1. / (2. * noise ** 2.) * residual ** 2.

    def d_data_fit_d_prediction(self, prediction, observation, noise, extra=None):
        """``d(data_fit)/d(prediction) = (mu - forward(obs))/sigma**2 * forward'(pred)``
        (#454). The location offset is prediction-independent, so this holds for both MEAN
        and MEDIAN; on the linear scale ``forward'(pred) = 1``. Equal to ``rho * d rho/d pred``
        (the ``residual_point`` pair) -- the Gaussian is the one family whose data fit is
        exactly ``1/2 rho**2``, so the assembly routes it through the residual-Jacobian and
        this seam exists for symmetry / unit checks."""
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return residual / noise ** 2. * self.additive_on.dforward(prediction)

    def residual(self, prediction, observation, noise, extra=None):
        """The Gaussian standardized residual ``rho = (mu - forward(obs))/sigma`` -- the
        least-squares residual the assembly stacks (#449/#459). The Gaussian is the one family
        whose ``data_fit`` is exactly ``1/2 rho**2``, so ``rho`` is already the signed
        square-root-loss residual (``sign(z) sqrt(2 data_fit) == rho`` identically); it is
        returned directly (not through the sqrt round-trip) so the historical path is
        byte-identical. The offset is prediction-independent: 0 for the MEDIAN (any scale) and a
        MEAN on the linear scale, the family's moment correction for a MEAN on a log scale."""
        return (self._mu(prediction, noise) - self.additive_on.forward(observation)) / noise

    def d_residual_d_prediction(self, prediction, observation, noise, extra=None):
        """``d(rho)/d(prediction) = forward'(pred)/sigma`` (#449/#459): ``1`` on the linear scale,
        ``1/(pred*ln10*sigma)`` for log10, ``1/(pred*sigma)`` for ln -- the scale's chain factor.
        The offset is prediction-independent, so MEAN and MEDIAN agree."""
        return self.additive_on.dforward(prediction) / noise

    def d_nll_d_noise_params(self, prediction, observation, noise, extra=None):
        """``{'sigma': (1 - rho**2)/sigma - rho * d(offset)/d sigma / sigma}`` -- the estimated-sigma
        gradient column (#451/#454/#385). ``d(data_fit)/d sigma`` has the symmetric ``-rho**2/sigma``
        part plus ``d(log sigma)/d sigma = 1/sigma`` (together ``(1 - rho**2)/sigma``), and -- when a
        MEAN is centered on a log scale -- a coupling term: the offset ``mu = forward(pred) - offset``
        itself depends on sigma there (``offset = ln(base)*sigma**2/2``), so ``rho`` carries an extra
        ``-rho * (d offset/d sigma)/sigma`` with ``d offset/d sigma = ln(base)*sigma``, giving
        ``-ln(base)*rho`` (#385). ``d(offset)/d sigma`` is 0 for the MEDIAN and on the linear scale,
        so this reduces to ``(1 - rho**2)/sigma`` byte-for-byte off the log-mean corner."""
        rho = (self._mu(prediction, noise) - self.additive_on.forward(observation)) / noise
        return {'sigma': (1. - rho ** 2) / noise
                - rho * self.location.d_offset_d_noise(self, noise) / noise}

    def log_normalizer(self, noise):
        return np.log(noise)

    def _density_constant(self):
        # The Gaussian's ½ log(2π): the part of -log N that is constant in the
        # parameters, which the sampler never needed (it cancels in accept ratios)
        # but a normalized density (log_density, for LOO/WAIC) keeps. Restoring it
        # makes log_density match scipy.stats.norm.logpdf (and, on a log scale plus
        # the Jacobian, scipy.stats.lognorm.logpdf).
        return _HALF_LOG_2PI
