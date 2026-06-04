"""The additive-noise-scale axis (ADR-0011): the scale a noise model's noise is
additive on.

This is **distinct from** a free parameter's ``priors.Scale`` (the *Parameter
Scale* -- the space a parameter is sampled in). Both happen to be Linear/Log, but
they are different domain concepts (see the CONTEXT.md glossary), so they are
deliberately separate code. An ``AdditiveNoiseScale`` maps a value into the space
the noise lives on: ``Gaussian`` noise additive on ``LINEAR`` is ordinary
additive error; additive on ``LOG`` is lognormal error.

``mean_offset`` is the correction between the additive-space location parameter
and the (log-)scale of the observation's *mean*, for a Gaussian additive
distribution: the mean of ``exp(N(mu, sigma))`` is ``exp(mu + sigma^2/2)``, so
recovering ``mu`` from a prediction interpreted as the mean subtracts
``sigma^2/2`` on the log scale (and nothing on the linear scale). It is the only
location-scale family with these axes today; generalize when a second arrives
(ADR-0009).
"""

import numpy as np


class AdditiveNoiseScale:
    """Maps a value into the space a noise model's noise is additive on."""

    def forward(self, x):
        """Transform an original-space value into the additive space."""
        raise NotImplementedError

    def mean_offset(self, noise):
        """Additive-space offset when the prediction is the distribution's mean
        (Gaussian moment correction); 0 unless the scale is logarithmic."""
        raise NotImplementedError


class _Linear(AdditiveNoiseScale):
    def forward(self, x):
        return x

    def mean_offset(self, noise):
        return 0.0


class _Log(AdditiveNoiseScale):
    def forward(self, x):
        return np.log(x)

    def mean_offset(self, noise):
        return noise ** 2. / 2.


LINEAR = _Linear()
LOG = _Log()
