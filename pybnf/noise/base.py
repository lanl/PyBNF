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

The full NLL splits as ``nll = data_fit(prediction, observation, noise)
+ log_normalizer(noise)``. The data-fit term is the parameter-dependent part; the
normalizer is constant whenever the noise parameter is held fixed. So a caller
whose noise parameter is fixed (a data column or a config constant) sums only
``data_fit``, while a caller estimating the noise parameter (a free parameter)
sums the full ``nll`` -- see the **Noise Parameter** glossary entry. This is why
``chi_sq`` (fixed sigma) drops Gaussian's ``log sigma`` while ``chi_sq_dynamic``
(free sigma) keeps it: one family, the normalizer governed by estimated-ness.
"""

from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """A per-point noise distribution family: the NLL kernel for one observation.

    Subclasses implement ``data_fit`` (the parameter-dependent term) and, if the
    family has a separable normalizer, override ``log_normalizer``.
    """

    @abstractmethod
    def data_fit(self, prediction, observation, noise):
        """The parameter-dependent negative-log-likelihood term for one point."""

    def log_normalizer(self, noise):
        """The likelihood normalizer -- constant when ``noise`` is fixed. Zero
        unless the family has a separable normalizer (e.g. Gaussian's
        ``log sigma``); a self-normalizing PMF (NegBinomial) leaves it at 0."""
        return 0.0

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

    def nll(self, prediction, observation, noise):
        """The full per-point negative log-likelihood (data fit + normalizer)."""
        return self.data_fit(prediction, observation, noise) + self.log_normalizer(noise)

    def _density_constant(self):
        """The parameter-independent additive constant a *normalized* density keeps
        but ``nll`` drops -- 0 by default (the count family, whose ``-data_fit`` is
        already a complete log-pmf), Gaussian's ``½ log(2π)`` for the normal family.
        Distinct from ``log_normalizer``, which is the noise-parameter-*dependent*
        part PyBNF sums only when that parameter is estimated; this is the pure
        constant the sampler never needed (it cancels in every accept ratio)."""
        return 0.0

    def log_density(self, prediction, observation, noise):
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
        needs no Jacobian), so the scale term is skipped there."""
        log_dens = -(self.data_fit(prediction, observation, noise)
                     + self.log_normalizer(noise)
                     + self._density_constant())
        scale = getattr(self, 'additive_on', None)
        if scale is not None:
            log_dens += scale.log_abs_dforward(observation)
        return log_dens
