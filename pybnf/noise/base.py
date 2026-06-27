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
