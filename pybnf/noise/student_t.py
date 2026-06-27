"""Student-t observation noise (ADR-0058) -- the robust-regression likelihood."""

import numpy as np
from scipy.special import loggamma

from ..printing import PybnfError
from .base import NoiseModel
from .location import MEDIAN
from .scale import LINEAR


class StudentT(NoiseModel):
    """Student-t (heavy-tailed) observation noise -- the outlier-robust likelihood, a
    Gaussian with a tail-heaviness knob. It is the **first two-parameter** noise family
    (ADR-0058): a location-scale family (like Gaussian/Laplace) carrying both a scale
    ``sigma`` and a shape ``df`` (degrees of freedom, nu). Small ``df`` gives fat tails
    that downweight outliers (robust regression); ``df -> inf`` recovers the Gaussian.
    This is Stan's/PyMC's ``student_t(nu, mu, sigma)``, with ``mu`` pinned to the
    prediction as for every family.

    Both noise parameters are **independently sourced** by the objective -- each may be
    ``fix_at`` a constant or ``fit`` a free parameter -- so a fit estimates 0, 1, or 2
    noise parameters. ``df`` is the one parameter with a default (``DEFAULT_DF``, a
    fixed 4): omit the ``df`` field and ``noise_param_defaults`` fills it, giving the
    common fixed-nu robust recipe. Estimating ``df`` is statistically weakly identified
    (the likelihood is nearly flat in large nu), so pair ``df = fit nu__FREE`` with a
    positive prior (gamma / half_*, ADR-0057) -- not enforced here.

    With ``z = (mu - forward(obs)) / sigma`` the per-point NLL splits (oracle:
    ``scipy.stats.t(df=nu, loc=mu, scale=sigma).logpdf``):

    - ``data_fit`` (always summed): ``(nu+1)/2 * log(1 + z**2/nu)`` -- the
      parameter-dependent core (depends on sigma via z, and on nu).
    - the ``sigma`` normalizer ``log sigma`` -- summed iff sigma is estimated.
    - the ``df`` normalizer ``-logGamma((nu+1)/2) + logGamma(nu/2) + 0.5*log(nu*pi)`` --
      summed iff df is estimated. When df is fixed this whole block is a constant the
      sampler drops; when df is free it is the term that keeps the fit honest. Either
      way ``log_density`` (LOO/WAIC) includes it, so student_t needs no
      ``_density_constant`` -- the "constant when fixed" the Gaussian carries as
      ``0.5*log(2*pi)`` is, for student_t, this estimated-gated normalizer (ADR-0058).

    Configured by the same two axes as Gaussian -- the scale its noise is additive on
    and the location interpretation -- but exposed on the **linear** scale only (no
    ``log_student_t`` token). On the linear scale t is symmetric, so mean = median = mu
    trivially. On a log scale ``base**StudentT`` has **no finite mean** (its tails are
    too heavy for the MGF to exist, for any nu), so ``location = mean`` on a log scale
    raises (only median centering is safe there) -- the Laplace log-scale-mean guard
    (#419) taken to its limit.
    """

    DEFAULT_DF = 4.0
    noise_params = ('sigma', 'df')
    noise_param_defaults = {'df': DEFAULT_DF}

    def __init__(self, additive_on=LINEAR, location=MEDIAN):
        self.additive_on = additive_on
        self.location = location

    def with_location(self, location):
        return type(self)(additive_on=self.additive_on, location=location)

    def mean_offset(self, noise):
        """0 on the linear scale (t is symmetric: mean = median). On **any** log scale
        the original-space mean does not exist (the t-distribution's tails are heavier
        than any exponential, so ``E[base**T]`` diverges for every nu), so mean-centering
        is undefined -- raise, directing the user to ``location = median`` (ADR-0058)."""
        t = self.additive_on.ln_base
        if t == 0.0:
            return 0.0
        raise PybnfError(
            "log-Student-t has no finite mean (its tails are too heavy for the mean to "
            "exist on a log scale, for any df), so mean-centering is undefined. Use "
            "location = median (the only safe centering for a Student-t on a log scale).")

    def _mu(self, prediction, sigma):
        """The additive-space location parameter for ``prediction``."""
        return self.additive_on.forward(prediction) - self.location.offset(self, sigma)

    def data_fit(self, prediction, observation, noise, extra=None):
        sigma, nu = noise, extra['df']
        z = (self._mu(prediction, sigma) - self.additive_on.forward(observation)) / sigma
        return (nu + 1.) / 2. * np.log1p(z * z / nu)

    def param_normalizers(self, noise, extra=None):
        sigma, nu = noise, extra['df']
        df_block = -loggamma((nu + 1.) / 2.) + loggamma(nu / 2.) + 0.5 * np.log(nu * np.pi)
        return {'sigma': np.log(sigma), 'df': df_block}
