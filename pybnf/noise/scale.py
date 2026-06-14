"""The additive-noise-scale axis (ADR-0011, ADR-0022): the scale a noise model's
noise is additive on.

This is **distinct from** a free parameter's ``priors.Scale`` (the *Parameter
Scale* -- the space a parameter is sampled in). They are different domain concepts
(see the CONTEXT.md glossary), so they are deliberately separate code -- but they
share one log-base convention (ADR-0022): **a bare "log" means log10 everywhere in
PyBNF**, matching ``logvar`` / ``loguniform_var`` / ``lognormal_var`` and the
proposal arithmetic. Natural log is never implied; it exists only as the explicit
``LN``. So this axis has three named members: ``LINEAR``, ``LOG10``, and ``LN`` --
there is no ambiguous bare ``LOG``. ``Gaussian`` noise additive on ``LINEAR`` is
ordinary additive error; additive on ``LOG10`` is (log10) lognormal error.

``mean_offset`` is the correction between the additive-space location parameter
and the (log-)scale of the observation's *mean*, for a Gaussian additive
distribution. The mean of ``b**N(mu, sigma)`` (base ``b`` = 10 or e) is
``b**(mu + ln(b)*sigma**2/2)``, so recovering ``mu`` from a prediction interpreted
as the mean subtracts ``ln(b)*sigma**2/2`` in the additive (log-base-b) space --
``sigma**2/2`` for natural log, ``ln10*sigma**2/2`` for log10, and nothing on the
linear scale. It is the only location-scale family with these axes today;
generalize when a second arrives (ADR-0009).
"""

import numpy as np

_LN10 = np.log(10.0)


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


class _Log10(AdditiveNoiseScale):
    def forward(self, x):
        return np.log10(x)

    def mean_offset(self, noise):
        return noise ** 2. * _LN10 / 2.


class _Ln(AdditiveNoiseScale):
    def forward(self, x):
        return np.log(x)

    def mean_offset(self, noise):
        return noise ** 2. / 2.


LINEAR = _Linear()
LOG10 = _Log10()
LN = _Ln()
