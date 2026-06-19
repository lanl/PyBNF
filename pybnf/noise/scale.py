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

A scale exposes ``ln_base`` -- the natural log of its base (the ``t`` in
``X = base**L = e**(t*L)``): 0 on the linear scale, ``ln 10`` on log10, 1 on
natural log. That is all a family's moment-generating function needs to convert an
additive-space location into the original-space mean. The moment correction itself
is **family-specific** -- Gaussian's ``t*sigma**2/2`` differs from Laplace's
``-ln(1 - b**2 t**2)/t`` (#419) -- so it lives on each ``NoiseModel`` family (their
``mean_offset``), not here. The scale owns only the transform (``forward``) and its
base (``ln_base``).
"""

import numpy as np

_LN10 = np.log(10.0)


class AdditiveNoiseScale:
    """Maps a value into the space a noise model's noise is additive on.

    ``ln_base`` is the natural log of the scale's base -- the only thing a family's
    moment correction needs from the scale (the family owns the correction itself).
    """

    ln_base = 0.0

    def forward(self, x):
        """Transform an original-space value into the additive space."""
        raise NotImplementedError


class _Linear(AdditiveNoiseScale):
    ln_base = 0.0

    def forward(self, x):
        return x


class _Log10(AdditiveNoiseScale):
    ln_base = _LN10

    def forward(self, x):
        return np.log10(x)


class _Ln(AdditiveNoiseScale):
    ln_base = 1.0

    def forward(self, x):
        return np.log(x)


LINEAR = _Linear()
LOG10 = _Log10()
LN = _Ln()
