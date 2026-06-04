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
**location interpretation**. Today every family is additive-on-linear with the
prediction as the mean -- both trivial because the current families are
symmetric -- so only the family axis is exercised; the ``lognormal`` seam
(ADR-0011) gives the scale and location axes real behavior.

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

    def nll(self, prediction, observation, noise):
        """The full per-point negative log-likelihood (data fit + normalizer)."""
        return self.data_fit(prediction, observation, noise) + self.log_normalizer(noise)
