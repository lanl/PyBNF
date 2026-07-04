"""A bring-your-own objective for PyBNF's ``objective = callable`` surface (ADR-0050).

The config line ::

    objective = callable
    callable  = robust_mixture.py:robust_nll

points the fit at the function ``robust_nll`` below. PyBNF calls it once per
candidate parameter set with the contract ::

    f(params: dict[str, float], data=None) -> float        # the score to MINIMIZE

where ``params`` is the current ``{name: value}`` map (bind-by-name -- index it by
the names you declared with ``uniform_var``) and the return value is the negative
log-likelihood (or any score) to minimize. ``data`` is ``None`` here because this
target is self-contained: the calibration measurements are embedded right in this
module (no model file, no ``.exp``). A ``data = curve.exp`` config line would
instead pass the loaded experiment(s) as the second argument.

WHY A CALLABLE?  The inline ``objective = expression`` grammar (lesson 38) handles a
closed-form scalar/per-point NLL, but it cannot express a ``logsumexp`` over mixture
components, a loop over replicate groups, or a ``scipy.stats`` density. A Python
callable is the escape hatch for exactly those. Here we score a straight-line fit
with a ROBUST two-component (inlier + wide-outlier) Gaussian MIXTURE likelihood --
a logsumexp density -- so a few gross outliers are down-weighted instead of dragging
the fit. (Being a general callable, it is gradient-free: use ``de`` / ``mh`` / etc.,
NOT ``hmc``, which needs a JAX-traceable target.)
"""
import numpy as np

# --- Embedded calibration data ------------------------------------------------
# Ten measurements of y at x = 0..9. Seven lie exactly on the line y = 2*x + 1;
# three (at x = 2, 5, 8) are gross OUTLIERS a plain least-squares fit would chase.
_X = np.arange(10, dtype=float)
_Y = 2.0 * _X + 1.0
_Y[2], _Y[5], _Y[8] = 15.0, 2.0, 30.0     # true (m, b) = (2, 1); outliers spliced in

# --- Mixture hyper-parameters (fixed; you could fit these too) -----------------
_SIGMA_IN = 0.7     # inlier noise scale (tight)
_SIGMA_OUT = 10.0   # outlier component scale (wide -- "catches" gross errors)
_W_OUT = 0.3        # prior probability a point is an outlier


def _log_normal(resid, sigma):
    """log N(resid; 0, sigma) -- a Gaussian log-density, elementwise over an array."""
    return -0.5 * np.log(2.0 * np.pi * sigma ** 2) - resid ** 2 / (2.0 * sigma ** 2)


def robust_nll(params, data=None):
    """Negative log-likelihood of the line y = m*x + b under a two-component Gaussian
    mixture: each point is an inlier (tight ``_SIGMA_IN``) with probability ``1-_W_OUT``
    or an outlier (wide ``_SIGMA_OUT``) with probability ``_W_OUT``. The per-point
    likelihood is the mixture ``(1-w)*N(r;0,sig_in) + w*N(r;0,sig_out)``, summed in log
    space with ``logaddexp`` (a numerically stable logsumexp) -- the density the inline
    expression grammar cannot write, which is why this is a callable.

    The wide component absorbs the outliers, so the minimum sits at the true
    ``(m, b) = (2, 1)`` instead of being pulled toward the outliers.
    """
    resid = _Y - (params['m'] * _X + params['b'])
    log_inlier = np.log(1.0 - _W_OUT) + _log_normal(resid, _SIGMA_IN)
    log_outlier = np.log(_W_OUT) + _log_normal(resid, _SIGMA_OUT)
    return float(-np.sum(np.logaddexp(log_inlier, log_outlier)))


def sse(params, data=None):
    """A NAIVE sum-of-squares score on the same data, for contrast (naive_sse.conf).
    With no outlier component every point is trusted equally, so the three outliers
    drag the fitted line off the truth -- the cautionary counterpart to ``robust_nll``.
    """
    resid = _Y - (params['m'] * _X + params['b'])
    return float(np.sum(resid ** 2))
