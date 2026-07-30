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

    def d_mean_offset_d_noise(self, noise):
        """``d(mean_offset)/d b = 2 b t / (1 - b**2 t**2)`` (0 on the linear scale), ``t =
        ln(base)``. The offset's own dependence on the scale ``b``, folded into the estimated-scale
        gradient column when a free Laplace scale centers a MEAN on a log scale (#385). Shares
        ``mean_offset``'s ``b*ln(base) < 1`` domain (the mean exists only there); off the log-mean
        corner (linear scale or median centering) it is 0, so the column reduces byte-for-byte."""
        t = self.additive_on.ln_base
        if t == 0.0:
            return 0.0
        bt = noise * t
        if bt >= 1.0:
            raise PybnfError(
                f"log-Laplace mean-centering needs b*ln(base) < 1 for the mean to exist: got "
                f"scale b={noise}, ln(base)={t} (b*ln(base)={bt} >= 1). Use location = median, "
                f"or a smaller Laplace scale b.")
        return 2. * noise * t / (1. - bt ** 2.)

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

    def residual(self, prediction, observation, noise, extra=None):
        """**Laplace has no exact least-squares residual -- it stays scalar-only** (#459).

        The sqrt-loss residual ``sign(z) * sqrt(2 * data_fit) = sign(z) * sqrt(2|z|/b)`` (``z = mu -
        forward(obs)``) is ``~ sqrt|z|`` near the residual zero: a **cusp with infinite slope at
        z=0** (and the IRLS weight ``1/|z| -> inf`` there too). L1 / least-absolute-deviation is
        *inherently* not cleanly least-squares, so -- unlike Student-t, whose sqrt-loss residual is
        smooth through 0 (#459) -- Laplace carries no residual a trust-region solver could minimize.
        Its gradient rides the scalar :meth:`d_data_fit_d_prediction` path instead
        (:meth:`~pybnf.objective.LikelihoodObjective.has_least_squares_residual` returns False for
        Laplace, so this is never reached on the normal path); a smoothed pseudo-Huber surrogate
        would be a separate, explicitly opt-in approximation of the loss, not exposed here."""
        raise NotImplementedError(
            'Laplace has no exact least-squares residual: sqrt(2*data_fit) ~ sqrt|z| is a cusp '
            'with infinite slope at z=0 (L1 / least-absolute-deviation is not cleanly '
            'least-squares), so a Laplace observable rides the scalar data-fit gradient, not the '
            'residual-Jacobian (#459). Use the L-BFGS-B path for a Laplace objective.')

    def d_residual_d_prediction(self, prediction, observation, noise, extra=None):
        """Laplace has no least-squares residual Jacobian (the residual itself has an infinite-slope
        cusp at ``z=0``); it stays scalar-only -- see :meth:`residual` (#459)."""
        raise NotImplementedError(
            'Laplace has no least-squares residual Jacobian; it rides the scalar data-fit '
            'gradient (#459). See Laplace.residual.')

    def d_nll_d_noise_params(self, prediction, observation, noise, extra=None):
        """``{'scale': -sign(R) * d(offset)/d b / b - |R|/b**2 + 1/b}`` with ``R = mu -
        forward(obs)`` -- the estimated-scale gradient column (#451/#454/#385). The
        ``-|R|/b**2`` is ``d(data_fit)/d b`` holding the offset fixed and ``1/b`` is
        ``d(log(2 b))/d b`` (the normalizer that keeps a free Laplace scale from running to
        infinity); the ``-sign(R) * (d offset/d b)/b`` is the coupling that appears when a MEAN is
        centered on a log scale, where ``mu = forward(pred) - offset`` and the offset depends on
        ``b`` (``d offset/d b = 2 b t/(1 - b**2 t**2)``). ``d(offset)/d b`` is 0 for the MEDIAN and
        on the linear scale, so this reduces to ``-|R|/b**2 + 1/b`` byte-for-byte off that corner."""
        residual = self._mu(prediction, noise) - self.additive_on.forward(observation)
        return {'scale': -np.sign(residual) * self.location.d_offset_d_noise(self, noise) / noise
                - abs(residual) / noise ** 2. + 1. / noise}

    def location_fisher(self, prediction, observation, noise, extra=None):
        """``kappa = (forward'(pred))**2 / b**2`` -- the Laplace location Fisher, the
        Gauss-Newton curvature the EFIM trust-region path (``job_type = gntr``, #481) weights
        ``d(prediction)/d(theta)`` by. The unit Fisher of a Laplace location is ``1/b**2``
        (``E[(sign(R)/b)**2] = 1/b**2``); the scale's chain factor ``forward'(pred)`` carries
        it to the prediction (``1`` on the linear scale). Laplace is **not** residual-bearing
        (its L1 data fit is the cusp ``sqrt|z|``, #459), so the assembly reads its location
        curvature here rather than off a residual Jacobian. Holds for MEDIAN and MEAN alike --
        the offset is prediction-independent, so ``d mu/d pred = forward'(pred)`` either way.
        Note the *observed* second derivative of ``|R|/b`` is 0 almost everywhere; the
        *expected* Fisher ``1/b**2`` is the well-posed curvature (why an EFIM step, not a plain
        Newton step, is what makes L1 tractable)."""
        return (self.additive_on.dforward(prediction) / noise) ** 2.

    def noise_param_fisher(self, prediction, observation, noise, extra=None):
        """``{'scale': 1/b**2}`` -- the expected Fisher of an estimated Laplace scale, the noise
        block of the EFIM Hessian (``job_type = gntr``, #481). With the per-point loss
        ``|R|/b + log(2 b)``, ``d^2/d b^2 = 2|R|/b^3 - 1/b^2`` and ``E[|R|] = b``, so
        ``E[d^2] = 2/b^2 - 1/b^2 = 1/b^2``. Diagonal (cross-Fisher 0 by symmetry), except a MEAN
        on a log scale, refused as for Gaussian (its scalar gradient still fits under
        ``job_type = lbfgs``)."""
        if self.location.d_offset_d_noise(self, noise) != 0.0:
            from ..gradient.errors import GradientNotSupported
            raise GradientNotSupported(
                "An estimated Laplace scale centering a MEAN on a log scale couples the "
                "location and scale on the EFIM trust-region path (job_type = gntr, #481); "
                "use job_type = lbfgs.")
        return {'scale': 1.0 / noise ** 2.}

    def log_normalizer(self, noise):
        return np.log(2. * noise)
