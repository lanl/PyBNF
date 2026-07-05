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

    def dforward(self, x):
        """``d forward(x)/dx`` -- the plain first derivative of the scale transform,
        the per-point seam a gradient needs (#452): the standardized residual on a log
        scale is ``rho = (forward(pred) - forward(obs))/sigma``, so ``d rho/d pred =
        forward'(pred)/sigma``. ``1`` on the linear scale (identity), ``1/(x ln 10)`` on
        log10, ``1/x`` on natural log -- so the linear short-circuits to the historical
        ``1/sigma`` byte-for-byte. ``forward``'s sibling for the optimizer, as
        :meth:`log_abs_dforward` is for a normalized density: this is the *signed*
        derivative the chain rule multiplies, that one the *log-absolute* change-of-
        variables term a density carries; deliberately the Additive Noise Scale axis,
        not a parameter's ``priors.Scale`` (the CONTEXT.md glossary keeps them apart)."""
        raise NotImplementedError

    def log_abs_dforward(self, x):
        """``log|d forward(x)/dx|`` -- the change-of-variables Jacobian term that
        turns an additive-space log-density into the original-space (data-space)
        log-density, ``log p_X(x) = log p_L(forward(x)) + log|d forward/dx|``
        (ADR-0056). It is 0 on the linear scale (identity transform) and the only
        thing besides a family's own normalizer that a *normalized* per-point
        likelihood (LOO/WAIC) needs that the optimizer/sampler never did -- the
        Jacobian is constant in the parameters, so it cancels in every accept ratio
        and PyBNF's ``nll`` omits it. ``forward``'s sibling: ``forward`` moves the
        value, this accounts for the density's stretch under that move."""
        raise NotImplementedError


class _Linear(AdditiveNoiseScale):
    ln_base = 0.0

    def forward(self, x):
        return x

    def dforward(self, x):
        # Identity transform: d x/d x = 1.
        return 1.0

    def log_abs_dforward(self, x):
        # Identity transform: |d x/d x| = 1, log 1 = 0.
        return 0.0


class _Log10(AdditiveNoiseScale):
    ln_base = _LN10

    def forward(self, x):
        # A non-positive value has no place on the log scale: log10(x<=0) is
        # mathematically -inf/undefined. A model *prediction* can land there from an
        # ODE solver's tiny sub-tolerance undershoot in a deep decay tail (a value
        # like -1e-10 that is physically zero). Map any x<=0 to -inf rather than let
        # numpy return NaN: the residual then blows up and the lognormal NLL is +inf
        # -- a hard penalty that steers the optimizer away -- instead of a NaN score
        # that silently corrupts the fit (observations are validated positive
        # upstream, so this guards the prediction). `[()]` keeps a scalar in, scalar
        # out (0-d array -> numpy scalar), preserving the plain-log10 return type.
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x > 0, np.log10(x), -np.inf)[()]

    def dforward(self, x):
        # d log10(x)/dx = 1/(x ln 10).
        return 1.0 / (x * _LN10)

    def log_abs_dforward(self, x):
        # d log10(x)/dx = 1/(x ln 10); log|.| = -log(x) - log(ln 10).
        return -np.log(x) - np.log(_LN10)


class _Ln(AdditiveNoiseScale):
    ln_base = 1.0

    def forward(self, x):
        # Same non-positive-prediction guard as _Log10.forward (natural-log twin):
        # x<=0 -> -inf, so the NLL is +inf rather than NaN.
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x > 0, np.log(x), -np.inf)[()]

    def dforward(self, x):
        # d ln(x)/dx = 1/x.
        return 1.0 / x

    def log_abs_dforward(self, x):
        # d ln(x)/dx = 1/x; log|.| = -log(x).
        return -np.log(x)


LINEAR = _Linear()
LOG10 = _Log10()
LN = _Ln()
