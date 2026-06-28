"""Laplace observation noise (ADR-0011, ADR-0021)."""

import numpy as np

from ..printing import PybnfError
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
    additive on and the location interpretation of the prediction. On the linear
    scale Laplace is symmetric, so mean and median coincide and the location axis is
    trivial (the default ``LINEAR``, ``MEDIAN``); on a log scale (``log-laplace``)
    the distribution is asymmetric in the original space, so ``location = mean`` picks
    up the **Laplace** moment correction (``mean_offset``), which is *not* Gaussian's
    (#419). ``data_fit`` is ``|mu - forward(obs)| / b`` with the scale ``b`` as the
    noise parameter; ``log_normalizer`` is ``log(2 b)``. With a fixed ``b`` the
    caller drops the normalizer; with a free ``b`` (the ``laplace`` objfunc's
    ``b__FREE``) it keeps the full ``nll`` -- the ``log(2 b)`` term is exactly what
    prevents the fit from driving ``b -> inf``. Value oracle:
    ``scipy.stats.laplace.logpdf``.
    """

    noise_params = ('scale',)

    def __init__(self, additive_on=LINEAR, location=MEDIAN):
        self.additive_on = additive_on
        self.location = location

    def with_location(self, location):
        return type(self)(additive_on=self.additive_on, location=location)

    def mean_offset(self, noise):
        """The **Laplace** moment correction (distinct from Gaussian's, #419). For
        ``base**Laplace(mu, b)`` the mean is ``base**mu / (1 - b**2 t**2)`` with
        ``t = ln(base)`` (the Laplace MGF ``E[e**(tL)] = e**(mu t)/(1 - b**2 t**2)``,
        which exists only for ``|b t| < 1``), so recovering ``mu`` from a prediction
        taken to be that mean subtracts ``-ln(1 - b**2 t**2)/t`` in additive space.
        0 on the linear scale (symmetric: mean = median)."""
        t = self.additive_on.ln_base
        if t == 0.0:
            return 0.0
        bt = noise * t
        if bt >= 1.0:
            raise PybnfError(
                f"log-Laplace mean-centering needs b*ln(base) < 1 for the mean to "
                f"exist (the tail is too heavy otherwise): got scale b={noise}, "
                f"ln(base)={t} (b*ln(base)={bt} >= 1). Use location = median, or a "
                f"smaller Laplace scale b.")
        return -np.log(1.0 - bt ** 2.) / t

    def _mu(self, prediction, noise):
        """The additive-space location parameter for ``prediction``."""
        return self.additive_on.forward(prediction) - self.location.offset(self, noise)

    def data_fit(self, prediction, observation, noise, extra=None):
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return abs(residual) / noise

    def d_data_fit_d_prediction(self, prediction, observation, noise, extra=None):
        """``d(data_fit)/d(prediction) = sign(mu - forward(obs))/b * forward'(pred)`` (#454).

        The Laplace data fit ``|mu - forward(obs)|/b`` is non-smooth where ``mu == forward(obs)``;
        PyBNF takes the **subgradient 0** there -- ``np.sign(0) == 0``, the symmetric least-
        absolute-deviation choice -- documented like layer E's positivity decision. The
        derivative is undefined at the kink, so the optimizer must not rely on it being exactly
        the slope of a finite difference straddling the kink; an FD acceptance oracle therefore
        evaluates away from it (data offset from the prediction). ``forward'(pred) = 1`` on the
        linear scale, and the offset is prediction-independent, so MEAN and MEDIAN agree."""
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return np.sign(residual) / noise * self.additive_on.dforward(prediction)

    def d_nll_d_noise_params(self, prediction, observation, noise, extra=None):
        """``{'scale': -|mu - forward(obs)|/b**2 + 1/b}`` -- the estimated-scale gradient
        column (#451/#454). ``d(data_fit)/d b = -|mu - forward(obs)|/b**2`` (the offset is
        b-independent off the gated mean-on-log corner) plus ``d(log(2 b))/d b = 1/b`` -- the
        ``log(2 b)`` term that keeps a free Laplace scale from running to infinity."""
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return {'scale': -abs(residual) / noise ** 2. + 1. / noise}

    def log_normalizer(self, noise):
        return np.log(2. * noise)
