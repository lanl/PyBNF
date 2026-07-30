"""Negative-binomial observation noise (ADR-0011, ADR-0031)."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc, betaln, digamma, loggamma

from .base import NoiseModel
from .location import MEDIAN

#: The clip :meth:`NegBinomial.data_fit` puts on ``prob = r/(r+mean)`` to keep a
#: degenerate point's ``-logpmf`` finite.
_PROB_CLIP = 1e-10

#: The mean-side image of that clip: ``prob <= 1 - _PROB_CLIP`` iff
#: ``mean >= r * _PROB_CLIP/(1 - _PROB_CLIP)``. Every derivative that divides by the mean
#: floors it here (:meth:`NegBinomial._mean_for_slope`) so a zero prediction -- which the
#: value path scores finitely -- yields a finite slope rather than ``nan``/``-inf``.
_MEAN_FLOOR_REL = _PROB_CLIP / (1 - _PROB_CLIP)


def _d_betainc_d_b(a, b, x, rel=1e-6):
    """``d/db`` of the regularized incomplete beta ``betainc(a, b, x) = I_x(a, b)`` w.r.t. its
    **second** parameter, by a central finite difference (issue #458).

    The median centering puts the prediction in this second parameter (``b = target + 1``, the
    continuous CDF ``I_p(r, pred + 1)``), so the implicit derivative ``d mean/d pred`` needs
    ``dI_x/db`` -- which is **not elementary**: it brings in the digamma function and a
    non-elementary ``int_0^x t^(a-1)(1-t)^(b-1) ln(1-t) dt``. #458 takes the issue-sanctioned
    *numerically-evaluated* parameter derivative -- a central difference in ``b`` with ``x`` held
    fixed (``x = r/(r+mean)`` does not depend on ``pred``). ``b = target + 1 >= 1`` here, so the
    relative step keeps ``b - db > 0``; at the median ``x`` is bounded away from 0 and 1, so
    ``betainc`` is smooth and the difference is accurate to ~1e-10 (validated against the integral
    representation ``int .../B(a,b) - I_x(a,b)(psi(b) - psi(a+b))``)."""
    db = rel * max(abs(b), 1.0)
    return (betainc(a, b + db, x) - betainc(a, b - db, x)) / (2.0 * db)


def _d_betainc_d_a(a, b, x, rel=1e-6):
    """``d/da`` of the regularized incomplete beta ``betainc(a, b, x) = I_x(a, b)`` w.r.t. its
    **first** parameter, by a central finite difference -- the sibling of :func:`_d_betainc_d_b`
    (issue #458). A free dispersion under MEDIAN centering makes the median's mean depend on ``r``
    through the first beta parameter ``a = r`` (the CDF is ``I_p(r, target+1)``), so the implicit
    ``d mean/d r`` needs ``dI_x/da`` -- as non-elementary as the second-parameter derivative.
    ``a = r > 0``, so the relative step keeps ``a - da > 0``; at the median ``x`` is bounded away
    from 0 and 1, so the difference is accurate to ~1e-10 (validated against the integral
    representation, like :func:`_d_betainc_d_b`)."""
    da = rel * max(abs(a), 1.0)
    return (betainc(a + da, b, x) - betainc(a - da, b, x)) / (2.0 * da)


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

    noise_params = ('dispersion',)

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

    def _mean_for_slope(self, prediction, noise):
        """:meth:`_mean`, floored away from zero for the expressions that divide by it.

        A MEAN-centered prediction of **exactly** zero is not pathological -- it is what
        any model whose output is gated off over part of the fit window predicts there (an
        epidemic model before its start time ``t0``, a stimulus-driven readout before the
        stimulus). :meth:`data_fit` scores such a point finitely because it clips
        ``prob = r/(r+mean)`` at ``1 - _PROB_CLIP``; every *derivative* below instead
        divides by the mean (``r (mean - obs) / (mean (r + mean))``, ``r / (mean (r+mean))``),
        which at ``mean == 0`` is ``0/0 -> nan`` for an observed zero and ``-inf`` otherwise.
        That ``nan`` propagates into the gradient and out to the optimizer, which then tries
        to assign a free parameter ``nan``.

        So mirror the value path's clip on the mean side: ``prob <= 1 - _PROB_CLIP`` is
        exactly ``mean >= r _PROB_CLIP / (1 - _PROB_CLIP)``, i.e. this floor. The slope
        is then the finite value belonging to the objective actually being scored -- large
        (order ``obs / (r _PROB_CLIP)``) where the model predicts zero against a positive
        count, which is correct: the objective there really is nearly flat-then-cliff, and a
        large finite push toward a positive prediction is the right search direction.
        MEDIAN centering never reaches the floor (``_mean_for_median`` returns a strictly
        positive mean for any prediction), so it is a no-op there."""
        return max(self._mean(prediction, noise), noise * _MEAN_FLOOR_REL)

    def data_fit(self, prediction, observation, noise, extra=None):
        if observation < 0:
            return 0
        mean = self._mean(prediction, noise)
        prob = np.clip(noise / (noise + mean), _PROB_CLIP, 1 - _PROB_CLIP)
        assert isinstance(noise, float)
        # log of the negative-binomial PMF P(observation | r=noise, prob)
        # == scipy.stats.nbinom.logpmf(observation, noise, prob).
        log_pmf = loggamma(observation + noise) - loggamma(observation + 1) - loggamma(noise) \
            + noise * np.log(prob) + observation * np.log(1 - prob)
        # A PMF is <= 1, so log_pmf <= 0; PyBNF minimizes the negative
        # log-likelihood -log_pmf >= 0.
        return -log_pmf

    def d_data_fit_d_prediction(self, prediction, observation, noise, extra=None):
        """``d(data_fit)/d(prediction)`` for one count point (#458, the deferred layer-G follow-up
        of #385). The count family carries no least-squares residual (its data fit is a ``-logpmf``,
        not a sum of squares), so PyBNF emits **only** this scalar data-fit gradient.

        The chain ``d(data_fit)/d(mean) * d(mean)/d(prediction)``. The mean-slope is the closed-form
        negative-binomial score ``r (mean - obs) / (mean (r + mean))`` (``r = noise`` the
        dispersion); the mean-factor depends on the location interpretation:

        * **MEAN**: the prediction *is* the mean, so ``d mean/d pred = 1`` and the slope is that
          closed form directly -- the clean case (``neg_bin`` / ``neg_bin_dynamic``).
        * **MEDIAN**: the mean is the CDF inversion ``_mean_for_median`` placing the continuous
          median at the prediction, so ``d mean/d pred`` is the **implicit derivative** of that
          root-find, ``-(dG/d pred) / (dG/d mean)`` with ``G(mean, pred) = betainc(r, target+1, p)
          - 0.5`` and ``p = r/(r+mean)``. ``dG/d mean = -beta_pdf(p; r, target+1) * r/(r+mean)**2``
          is the smooth beta-density chain rule (``F`` strictly decreasing in the mean, so this is
          negative); ``dG/d pred`` puts the prediction in the beta's *second* parameter, so it is
          the non-elementary ``d betainc/d b`` (:func:`_d_betainc_d_b`), times ``d target/d pred =
          1``. The result is positive (a larger predicted median needs a larger mean).

        A negative observation contributes nothing (the count-domain guard, mirroring
        :meth:`data_fit`). A prediction clamped to the count floor (``pred <= 0``, where ``target``
        floors at 0 and the median stops moving) has slope 0 -- a kink at ``pred == 0`` where PyBNF
        takes the floor-side subgradient, like the Laplace kink (#454). The mean-slope divides by
        the mean, so it reads it through :meth:`_mean_for_slope` -- the value path's ``prob`` clip
        expressed on the mean -- which keeps a MEAN-centered zero prediction finite instead of
        ``nan``; away from that floor the form is the un-clipped analytic one."""
        if observation < 0:
            return 0.0
        r = noise
        mean = self._mean_for_slope(prediction, r)
        d_fit_d_mean = r * (mean - observation) / (mean * (r + mean))
        if self.location is not MEDIAN:
            return d_fit_d_mean   # MEAN: d mean/d pred = 1
        if prediction <= 0.0:
            return 0.0            # clamped to the count floor: the median stops moving
        target = prediction
        p = r / (r + mean)
        # beta_pdf(p; r, target+1) = p**(r-1) (1-p)**target / B(r, target+1) -- via betaln for range.
        beta_pdf = np.exp((r - 1.0) * np.log(p) + target * np.log1p(-p) - betaln(r, target + 1.0))
        dG_d_mean = -beta_pdf * r / (r + mean) ** 2.
        dG_d_pred = _d_betainc_d_b(r, target + 1.0, p)
        d_mean_d_pred = -dG_d_pred / dG_d_mean
        return d_fit_d_mean * d_mean_d_pred

    def d_nll_d_noise_params(self, prediction, observation, noise, extra=None):
        """``{'dispersion': d(data_fit)/d r}`` -- the estimated-dispersion gradient column for a free
        ``r`` (``neg_bin_dynamic``'s ``r__FREE``; #458, generalizing layer D/G of #385).

        The negative-binomial PMF is **self-normalizing** (``log_normalizer == 0``), so -- unlike a
        Gaussian's ``+log sigma`` or a Laplace's ``log(2 b)`` -- there is no separable normalizer:
        the whole dispersion gradient lives in the data fit. With ``mean`` the distribution mean and
        ``prob = r/(r+mean)`` the partial holding the mean fixed is the negative-binomial dispersion
        score::

            d(data_fit)/d r |_mean = psi(r) - psi(obs + r) - log(prob) - 1 + (r + obs)/(r + mean)

        (``psi`` the digamma; the ``-logpmf``'s ``r``-dependence through its gamma terms and ``prob``).

        * **MEAN**: the prediction *is* the mean, r-independent, so this partial is the whole
          derivative.
        * **MEDIAN**: the mean is solved from ``r`` (the CDF inversion ``_mean_for_median``), so add
          the coupling ``d(data_fit)/d mean * d mean/d r``. ``d mean/d r`` is the implicit derivative
          of ``G(mean, r) = betainc(r, target+1, p) - 0.5 = 0`` (``p = r/(r+mean)``) holding the
          prediction fixed, ``-(dG/d r)/(dG/d mean)``; ``dG/d r`` brings in the betainc **first**-
          parameter derivative (``a = r``, :func:`_d_betainc_d_a`) plus ``p``'s own ``r``-dependence,
          and ``dG/d mean`` is the same beta-density chain rule as the prediction gradient. The count
          floor (``target = max(pred, 0)``) does not zero this -- the mean still moves with ``r`` even
          at a clamped prediction. (Validated FD-first: the partial-only form is off 5-15%, sometimes
          far more, on the median.)

        A negative observation contributes nothing (the count-domain guard). The mean is read
        through :meth:`_mean_for_slope` for consistency with :meth:`d_data_fit_d_prediction` --
        it matters only for the MEDIAN coupling term, which divides by the mean; the MEAN
        partial above is finite at a zero prediction either way, and the floor shifts it by
        ``O(_PROB_CLIP)``. MEDIAN centering never reaches the floor."""
        if observation < 0:
            return {'dispersion': 0.0}
        r = noise
        mean = self._mean_for_slope(prediction, r)
        prob = r / (r + mean)
        d_fit_d_r = (digamma(r) - digamma(observation + r) - np.log(prob) - 1.0
                     + (r + observation) / (r + mean))
        if self.location is MEDIAN:
            target = max(prediction, 0.0)
            d_fit_d_mean = r * (mean - observation) / (mean * (r + mean))
            beta_pdf = np.exp((r - 1.0) * np.log(prob) + target * np.log1p(-prob)
                              - betaln(r, target + 1.0))
            dG_d_mean = -beta_pdf * r / (r + mean) ** 2.
            dG_d_r = _d_betainc_d_a(r, target + 1.0, prob) + beta_pdf * mean / (r + mean) ** 2.
            d_fit_d_r += d_fit_d_mean * (-dG_d_r / dG_d_mean)
        return {'dispersion': d_fit_d_r}

    def location_fisher(self, prediction, observation, noise, extra=None):
        """``kappa = I_mean = r / (mean (r + mean))`` -- the negative-binomial **mean's** Fisher
        information, the Gauss-Newton curvature the EFIM trust-region path (``job_type = gntr``,
        #481) weights ``d(prediction)/d(theta)`` by. It is the variance of the mean score
        ``d(-logpmf)/d mean = r (mean - obs) / (mean (r + mean))`` (``Var(obs) = mean (r + mean)/r``,
        so ``I_mean = [r/(mean(r+mean))]**2 * Var(obs) = r/(mean(r+mean))``).

        * **MEAN**: the prediction *is* the mean (``d mean/d pred = 1``), so ``kappa`` is this
          directly -- the case this cut supports (``neg_bin`` / ``neg_bin_dynamic`` pinned to MEAN).
        * **MEDIAN**: the mean sits behind the betainc CDF inversion (``_mean_for_median``), whose
          ``d mean/d pred`` is the non-elementary implicit derivative; its location Fisher is out of
          scope for this cut -- refused, pointing at ``job_type = lbfgs`` (which fits it via the
          scalar data-fit gradient). A negative observation contributes no curvature (the
          count-domain guard, mirroring :meth:`data_fit`).

        A **zero mean** deliberately reports **no curvature** rather than the
        :meth:`_mean_for_slope` floor the *gradient* uses. The two want opposite things: the
        true ``I_mean`` diverges as ``1/mean``, so flooring it would put a ``~1/(r
        _PROB_CLIP)`` weight on every gated-off row (an epidemic model's whole pre-``t0``
        stretch) and let those rows dominate the Fisher matrix, crushing the trust-region
        step. Zero says "this row carries no information about the mean", which is the stable
        reading and leaves the gradient -- which does floor -- to supply the descent
        direction."""
        if self.location is MEDIAN:
            from ..gradient.errors import GradientNotSupported
            raise GradientNotSupported(
                "A MEDIAN-centered negative-binomial has its mean behind a betainc CDF inversion, "
                "whose location Fisher the EFIM trust-region path (job_type = gntr, #481) does not "
                "assemble this cut; use job_type = lbfgs.")
        mean = self._mean(prediction, noise)
        if observation < 0 or mean <= 0.0:
            return 0.0
        return noise / (mean * (noise + mean))
