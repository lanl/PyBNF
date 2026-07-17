"""The ``NoiseModel`` abstraction: a per-point negative-log-likelihood kernel
for one observation given the model's prediction and a noise parameter (ADR-0011).

A ``NoiseModel`` is **pure and scale-agnostic** -- it knows nothing about the
iteration over data points, the weighting, or where its noise parameter comes
from. The owning ``SummationObjective`` (in ``objective.py``) drives the per-row
loop and sources the noise parameter, calling the kernel one point at a time.
This mirrors M2.3's ``Prior``/``FreeParameter`` split (ADR-0010): the harness that
owns the iteration delegates the pure family math to a small object.

A per-point noise model is defined by three orthogonal axes (ADR-0004):
distribution **family** x the **scale the noise is additive on** x the
**location interpretation**. All three are live: ``lognormal`` is Gaussian
additive-on-log (the scale axis); ``mean`` vs ``median`` centering picks up each
family's own moment correction on a log scale (the location axis, ADR-0031/#419) --
Gaussian's, Laplace's, and the count family's median CDF inversion all differ.

The full NLL splits as ``nll = data_fit(prediction, observation, noise, extra)
+ sum(param_normalizers(noise, extra).values())``. The data-fit term is the
parameter-dependent part; each parameter's normalizer is constant whenever *that*
noise parameter is held fixed. So a caller sums ``data_fit`` always, plus each
parameter's normalizer iff that parameter is *estimated* (a free parameter) rather
than fixed (a data column or a config constant) -- see the **Noise Parameter**
glossary entry. This is why ``chi_sq`` (fixed sigma) drops Gaussian's ``log sigma``
while ``chi_sq_dynamic`` (free sigma) keeps it: one family, the normalizer governed
by estimated-ness, now keyed per parameter so a two-parameter family (student_t's
sigma + df, ADR-0058) gates each independently.

Most families carry one noise parameter (the scalar ``noise``); a multi-parameter
family receives its primary scalar as ``noise`` and the rest in a trailing ``extra``
mapping, named by the family's ``noise_params`` (ADR-0058). The single-parameter
defaults below mean the existing families need no body change for the generalization.
"""

from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """A per-point noise distribution family: the NLL kernel for one observation.

    Subclasses implement ``data_fit`` (the parameter-dependent term) and, if the
    family has a separable normalizer, override ``log_normalizer`` (one parameter) or
    ``param_normalizers`` (several, keyed by name; ADR-0058).
    """

    #: This family's noise-parameter names in declaration order (ADR-0058). The
    #: **first is the primary** scalar passed as ``noise``; any others arrive in the
    #: trailing ``extra`` mapping. Single-parameter families inherit ``('sigma',)`` or
    #: override it (Laplace's ``('scale',)``, NegBinomial's ``('dispersion',)``); the
    #: one multi-parameter family is StudentT (``('sigma', 'df')``). This is the single
    #: source of truth for a family's parameter names -- the objective reads it to
    #: validate a ``noise_model`` line's fields.
    noise_params = ('sigma',)

    #: Default fixed values for parameters that may be omitted from a ``noise_model``
    #: line (ADR-0058): ``{param_name: value}``. The objective fills an omitted such
    #: parameter with a ``ConstantSigma`` of this value (student_t's ``{'df': 4.0}``).
    #: A parameter named here is optional; one absent here (e.g. ``sigma``) is required.
    #: The *value* lives with the family (a distributional fact); the engine builds the
    #: source, keeping source construction out of the pure kernel (ADR-0011).
    noise_param_defaults = {}

    @abstractmethod
    def data_fit(self, prediction, observation, noise, extra=None):
        """The parameter-dependent negative-log-likelihood term for one point.
        ``noise`` is the family's primary scalar parameter; ``extra`` is a mapping of
        any secondary parameters (``{'df': nu}`` for student_t), ``None`` / empty for
        the single-parameter families, which ignore it (ADR-0058)."""

    def log_normalizer(self, noise):
        """The single-parameter likelihood normalizer -- constant when ``noise`` is
        fixed. Zero unless the family has a separable normalizer (e.g. Gaussian's
        ``log sigma``); a self-normalizing PMF (NegBinomial) leaves it at 0. A
        multi-parameter family overrides :meth:`param_normalizers` instead."""
        return 0.0

    def param_normalizers(self, noise, extra=None):
        """``{param_name: separable normalizer}`` -- the normalizer each noise
        parameter contributes, which the objective adds iff *that* parameter's source
        is estimated (ADR-0058). The default attributes the family's whole
        :meth:`log_normalizer` to its single primary parameter, so a one-parameter
        family needs no override; StudentT splits ``log sigma`` (``'sigma'``) from the
        df-block (``'df'``)."""
        return {self.noise_params[0]: self.log_normalizer(noise)}

    def with_location(self, location):
        """Return a copy of this family reinterpreting the prediction as a different
        distributional summary -- the location axis (ADR-0011/0024/0031). Every noise
        family implements it ("every means every"): the location-scale families via
        the additive offset, the count family via a per-point CDF inversion. The base
        raises for a (hypothetical) family with no location axis."""
        raise NotImplementedError(
            f'{type(self).__name__} has no location interpretation axis')

    def mean_offset(self, noise):
        """The additive-space offset for **mean**-centering -- the family's moment
        correction, subtracted from ``scale.forward(prediction)`` to recover the
        location parameter when the prediction is taken to be the distribution mean
        (``MEAN``, via ``location.py``). It is family-specific (Gaussian's differs
        from Laplace's, #419), so each location-scale family overrides it; the base
        raises for a family that has no additive moment correction (e.g. the count
        family, which realizes its mean centering directly, not through this seam)."""
        raise NotImplementedError(
            f'{type(self).__name__} has no additive mean offset')

    def d_mean_offset_d_noise(self, noise):
        """``d(mean_offset)/d(noise parameter)`` -- how the mean's moment correction moves with
        the noise scale, the term the estimated-scale gradient column needs when the prediction is
        a MEAN on a log scale (#385). It is family-specific (Gaussian's ``ln(base)*sigma``,
        Laplace's ``2 b t/(1 - b**2 t**2)``) and **0 on the linear scale** (the offset is 0 there),
        so the estimated-scale column reduces byte-for-byte off the log-mean corner. Each
        location-scale family overrides it; the base raises like :meth:`mean_offset`."""
        raise NotImplementedError(
            f'{type(self).__name__} has no additive mean offset')

    def d_data_fit_d_prediction(self, prediction, observation, noise, extra=None):
        """``d(data_fit)/d(prediction)`` for one point -- the per-point loss slope the
        gradient path chains through ``d(prediction)/d(theta)`` (layer G, #454/#385).

        This is the **universal** scalar-gradient seam: an asymmetric family (Laplace,
        Student-t) whose ``data_fit`` is not a sum of squares carries no least-squares
        residual, so its objective gradient is assembled as
        ``sum_i w_i * d(data_fit_i)/d(prediction_i) * d(prediction_i)/d(theta)`` rather than
        from a residual-Jacobian. Overridden by the location-scale families (Gaussian / Laplace /
        Student-t) and -- via its median CDF-inversion implicit derivative (#458) -- the count
        family (NegBinomial); the base raises for a (hypothetical) family with no prediction
        gradient."""
        raise NotImplementedError(
            f'{type(self).__name__} has no prediction gradient on the noise-model gradient '
            f'path (#454/#458)')

    def residual(self, prediction, observation, noise, extra=None):
        """The signed **square-root-loss residual** ``r`` for one point -- the least-squares
        residual form a trust-region solver (LM / TRF, #386) minimizes (layer G follow-up, #459).

        Defined so that ``1/2 r**2 == data_fit`` *and* ``r * d_residual_d_prediction ==
        d_data_fit_d_prediction`` -- i.e. ``r`` reproduces both this point's loss and its
        prediction gradient, so ``scipy.least_squares`` minimizes the **true** objective (not a
        frozen-weight IRLS surrogate). The canonical form is ``r = sign(z) * sqrt(2 * data_fit)``
        with ``z = (mu - forward(obs))/sigma`` the additive-space residual: for a **Gaussian**
        this is exactly the standardized residual ``rho`` (its data fit is ``1/2 rho**2``); for
        **Student-t** the sqrt-loss residual is smooth through ``z=0`` (``r ~ sqrt((nu+1)/nu) z``)
        and downweights the tails, a clean exact least-squares reformulation (#459).

        Only a family whose data fit reformulates as a *smooth* half-square overrides this
        (Gaussian, Student-t); the base raises. **Laplace** stays scalar-only -- its data fit
        ``|z|/b`` has a cusp at ``z=0`` (``sqrt(2*data_fit) ~ sqrt|z|``, infinite slope), so L1 /
        least-absolute-deviation is inherently not cleanly least-squares (it overrides this with a
        pointed raise). The count family (NegBinomial) likewise has no least-squares residual. A
        family the base refuses is routed through the scalar :meth:`d_data_fit_d_prediction`
        instead; :meth:`~pybnf.objective.LikelihoodObjective.has_least_squares_residual` decides."""
        raise NotImplementedError(
            f'{type(self).__name__} has no least-squares residual on the gradient path; its '
            f'data fit is not a smooth sum of squares, so it rides the scalar data-fit gradient '
            f'(#459)')

    def d_residual_d_prediction(self, prediction, observation, noise, extra=None):
        """``d(residual)/d(prediction)`` for one point -- the residual-Jacobian seam, paired with
        :meth:`residual` (#459). The assembly multiplies it by the forward sensitivity
        ``d pred/d theta`` to build the residual-Jacobian column, exactly as for a Gaussian.

        Equal to ``d_data_fit_d_prediction / residual`` away from ``z=0``; at the residual zero
        both numerator and denominator vanish, so the override supplies the **smooth limit** (for
        Student-t, ``sqrt((nu+1)/nu) * forward'(pred)/sigma``). Overridden by exactly the families
        that override :meth:`residual` (Gaussian, Student-t); the base raises."""
        raise NotImplementedError(
            f'{type(self).__name__} has no least-squares residual Jacobian on the gradient path '
            f'(#459)')

    def d_nll_d_noise_params(self, prediction, observation, noise, extra=None):
        """``{param_name: d(data_fit + that parameter's normalizer)/d param}`` for each noise
        parameter -- the estimated-scale gradient columns (layer D/G, #451/#454/#385).

        The objective sums each entry into the scalar gradient **iff that parameter is
        estimated** (a free parameter), exactly as ``eval_point`` adds each parameter's
        normalizer iff estimated -- so the returned derivative folds in the parameter's own
        normalizer (Gaussian's ``log sigma``, Student-t's df-block), which is the term that
        keeps a free scale from running away. Each family's normalizers depend only on their
        own parameter, so the cross terms vanish and a per-parameter entry suffices.
        Overridden by the location-scale families and the count family (NegBinomial's
        self-normalizing dispersion score, #458); the base raises for a family with none."""
        raise NotImplementedError(
            f'{type(self).__name__} has no noise-parameter gradient on the gradient path '
            f'(#454/#458)')

    def location_fisher(self, prediction, observation, noise, extra=None):
        """The expected per-point **Fisher information of the location** ``kappa`` -- the
        Gauss-Newton curvature the EFIM trust-region path (``fit_type = gntr``, #481)
        weights ``d(prediction)/d(theta)`` by to build its Hessian ``H = sum_i kappa_i
        s_i s_i^T``. ``kappa`` is the expected ``E[-d^2 log p / d mu^2]`` carried through
        the prediction: for a location-scale family additive on a scale it is
        ``(forward'(pred))**2 * I_additive`` (the unit location Fisher in the additive
        space times the scale's chain factor squared).

        Only a **non-residual** family in scope overrides it -- Laplace
        (``(forward'(pred))**2 / b**2``) and the count family (its mean's Fisher, through
        the location realization). A **residual-bearing** family (Gaussian, Student-t) is
        never asked: the assembly reads its location curvature straight off the residual
        Jacobian (``d_residual_d_prediction**2``, which *is* the Fisher there), so it takes
        that path. The base raises, so an unsupported family/config refuses the EFIM step
        with a pointer back to the scalar-gradient (L-BFGS-B) path."""
        from ..gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "%s has no expected location Fisher on the EFIM trust-region path "
            "(fit_type = gntr, #481)." % type(self).__name__)

    def noise_param_fisher(self, prediction, observation, noise, extra=None):
        """``{param_name: I_scale}`` -- the expected **Fisher information of each estimated
        noise parameter**, the diagonal noise block of the EFIM Hessian (``fit_type =
        gntr``, #481). The assembly reads it only for a parameter whose source is
        *estimated* (a free parameter), exactly as :meth:`noise_grad_point` emits a column
        only for an estimated parameter, and accumulates ``w_i * I_scale *
        outer(e_param, e_param)`` (the free parameter *is* the noise parameter, so the
        curvature direction is that parameter's own coordinate). The location-scale cross
        Fisher is 0 for the symmetric families on linear/MEDIAN, so the block is diagonal
        there -- a family whose estimated scale couples to the location (a MEAN on a log
        scale) refuses instead.

        Overridden by the families whose estimated-scale block this cut assembles --
        Gaussian (``{'sigma': 2/sigma**2}``) and Laplace (``{'scale': 1/b**2}``). The base
        raises, which is exactly the refusal for an out-of-scope estimated scale (the count
        family's free dispersion, the Student-t 2-parameter block): the fit falls back to
        the scalar-gradient (L-BFGS-B) path."""
        from ..gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "%s has no estimated-noise Fisher block on the EFIM trust-region path "
            "(fit_type = gntr, #481)." % type(self).__name__)

    def nll(self, prediction, observation, noise, extra=None):
        """The full per-point negative log-likelihood (data fit + every parameter's
        normalizer)."""
        return (self.data_fit(prediction, observation, noise, extra)
                + sum(self.param_normalizers(noise, extra).values()))

    def _density_constant(self):
        """The parameter-independent additive constant a *normalized* density keeps
        but ``nll`` drops -- 0 by default (the count family, whose ``-data_fit`` is
        already a complete log-pmf), Gaussian's ``½ log(2π)`` for the normal family.
        Distinct from ``log_normalizer``, which is the noise-parameter-*dependent*
        part PyBNF sums only when that parameter is estimated; this is the pure
        constant the sampler never needed (it cancels in every accept ratio)."""
        return 0.0

    def log_density(self, prediction, observation, noise, extra=None):
        """The genuine per-point log-density ``log p(observation | prediction,
        noise)`` in **data space** -- the complete, normalized value model-comparison
        (LOO/WAIC, ADR-0056) consumes, as opposed to ``-nll``.

        ``nll`` is built for the sampler, which only needs likelihood *ratios*, so it
        drops every term constant in the parameters: the family constant
        (``_density_constant``) and, for a family additive on a log scale, the
        change-of-variables Jacobian (``scale.log_abs_dforward``). A predictive
        density needs them, so this restores both -- giving a value that matches
        ``scipy.stats.<dist>.logpdf`` / ``.logpmf`` (the oracle each family documents).
        The count family carries no ``additive_on`` (its PMF is self-normalizing and
        needs no Jacobian), so the scale term is skipped there. **Every** parameter's
        normalizer is included regardless of estimated-ness (unlike the sampler's
        ``eval_point``), so for student_t the df-block is always present even when df
        is fixed -- which is why student_t needs no ``_density_constant`` (ADR-0058)."""
        log_dens = -(self.data_fit(prediction, observation, noise, extra)
                     + sum(self.param_normalizers(noise, extra).values())
                     + self._density_constant())
        scale = getattr(self, 'additive_on', None)
        if scale is not None:
            log_dens += scale.log_abs_dforward(observation)
        return log_dens
