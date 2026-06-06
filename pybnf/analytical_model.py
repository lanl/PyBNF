"""Analytical test models for sampler comparison benchmarks.

These models compute a negative log-likelihood directly from the free parameters,
bypassing any external simulator. Used with objfunc = direct_pass.

Supported target types:
  gaussian         - Axis-aligned Gaussian (diagonal variance; a *separable* objective)
  rotated_gaussian - Correlated Gaussian with a full covariance Sigma (non-separable)
  rotated_quartic  - Smooth, non-separable, NON-quadratic, trap-free valley (2D)
  banana           - Rosenbrock/banana-shaped distribution (2D)
  multimodal       - Mixture of Gaussians with configurable modes
"""

import copy
import json
import logging
import time
import numpy as np
from os.path import splitext, basename

from .data import Data
from .pset import Model

logger = logging.getLogger(__name__)


class AnalyticalModel(Model):
    """
    A model that computes a target score directly from free parameters.

    Reads a .target JSON file specifying the target type and parameters.
    Returns a Data object with a single 'score' column containing the NLL.
    """

    def __init__(self, target_file, pset=None):
        self.file_path = target_file
        self.name = splitext(basename(target_file))[0]
        self.suffixes = ['target']
        self.stochastic = False
        self.has_observables = True
        self.param_names = set()  # All params come from the config, not the model file

        with open(target_file, encoding='utf-8') as f:
            self.target_def = json.load(f)

        self.target_type = self.target_def['type']
        self._pset = pset

        # Pre-compute target-specific constants
        if self.target_type == 'gaussian':
            self._mean = np.array(self.target_def['mean'])
            self._var = np.array(self.target_def['variance'])
            self._inv_var = 1.0 / self._var
        elif self.target_type == 'rotated_gaussian':
            self._mean = np.array(self.target_def['mean'], dtype=float)
            self._cov = np.array(self.target_def['covariance'], dtype=float)
            # Precision Sigma^{-1}; symmetrize to clear any inversion round-off so
            # the quadratic form stays exactly symmetric.
            prec = np.linalg.inv(self._cov)
            self._prec = 0.5 * (prec + prec.T)
        elif self.target_type == 'rotated_quartic':
            self._mean = np.array(self.target_def['mean'], dtype=float)
            angle = float(self.target_def['angle'])
            c, s = np.cos(angle), np.sin(angle)
            self._rot = np.array([[c, -s], [s, c]])   # R(angle); r = R (x - mu)
            self._coeff = np.array(self.target_def['coeff'], dtype=float)  # (k1, k2)
        elif self.target_type == 'banana':
            self._a = self.target_def.get('a', 1.0)
            self._b = self.target_def.get('b', 100.0)
        elif self.target_type == 'multimodal':
            self._modes = []
            for mode in self.target_def['modes']:
                w = mode['weight']
                mu = np.array(mode['mean'])
                var = np.array(mode['variance'])
                self._modes.append((np.log(w), mu, 1.0 / var))
        else:
            raise ValueError('Unknown analytical target type: %s' % self.target_type)

    def copy_with_param_set(self, pset):
        m = copy.copy(self)
        m._pset = pset
        return m

    def save(self, file_prefix, **kwargs):
        pass

    def get_suffixes(self):
        return self.suffixes

    def execute(self, folder, filename, timeout):
        """Compute the NLL score from the current parameter set."""
        # Small delay to prevent dask race condition with instant-completion tasks
        time.sleep(0.01)
        params = self._get_param_values()
        score = self._compute_nll(params)

        # Return Data with 'index' and 'score' columns (index is the independent variable)
        data = Data(arr=np.array([[0.0, score]]))
        data.cols = {'index': 0, 'score': 1}
        data.headers = {0: 'index', 1: 'score'}
        return {'target': data}

    def _get_param_values(self):
        """Extract parameter values as a numpy array, sorted by name."""
        if self._pset is None:
            raise ValueError('AnalyticalModel has no parameter set')
        names = sorted(self._pset.keys())
        return np.array([self._pset[n] for n in names])

    def _compute_nll(self, params):
        """Compute negative log-likelihood for the target distribution."""
        if self.target_type == 'gaussian':
            return self._nll_gaussian(params)
        elif self.target_type == 'rotated_gaussian':
            return self._nll_rotated_gaussian(params)
        elif self.target_type == 'rotated_quartic':
            return self._nll_rotated_quartic(params)
        elif self.target_type == 'banana':
            return self._nll_banana(params)
        elif self.target_type == 'multimodal':
            return self._nll_multimodal(params)
        else:
            # Unreachable in practice (__init__ already rejects unknown types),
            # but fail loud rather than return an implicit None if a new target
            # type is ever added to __init__ but not here.
            raise ValueError('Unknown analytical target type: %s' % self.target_type)

    def _nll_gaussian(self, params):
        """NLL of multivariate Gaussian: 0.5 * sum((x - mu)^2 / sigma^2)"""
        diff = params - self._mean
        return 0.5 * np.sum(diff ** 2 * self._inv_var)

    def _nll_rotated_gaussian(self, params):
        """NLL of a correlated (rotated) multivariate Gaussian with full
        covariance Sigma: ``0.5 * (x - mu)^T Sigma^{-1} (x - mu)``.

        Unlike the axis-aligned ``gaussian`` (diagonal variance, a *separable*
        objective whose principal axes are the coordinate axes), the off-diagonal
        precision couples the coordinates, so the quadratic bowl's principal axes
        are rotated off the coordinate axes. That is the textbook validator for
        conjugate-direction (Powell) and covariance-adapting (CMA-ES) methods:
        coordinate-only descent zig-zags, while those methods discover the
        rotation (#405). The mode is still ``mu`` (NLL 0 there).
        """
        diff = params - self._mean
        return 0.5 * float(diff @ self._prec @ diff)

    def _nll_rotated_quartic(self, params):
        """NLL of a smooth, non-separable, NON-quadratic, trap-free valley:
        ``k1 * r1**4 + k2 * r2**2`` where ``r = R(angle) (x - mu)``.

        Quartic along the first rotated axis, quadratic along the second; with
        ``k1 << k2`` this is a long, flat, curved valley. Its *only* stationary
        point is ``mu`` (the mode, NLL 0), so it is trap-free for a local
        optimizer — unlike the banana. Because it is non-quadratic, a single
        fixed-step parabolic line search is a poor 1-D model and converges slowly
        / stalls, whereas a bracketing + Brent line search follows the valley.
        This is the discriminating target for Powell's robustified line search
        (#406); the rotated *Gaussian* (quadratic) cannot discriminate, since a
        parabola fits a quadratic exactly.
        """
        r = self._rot @ (params - self._mean)
        return self._coeff[0] * r[0] ** 4 + self._coeff[1] * r[1] ** 2

    def _nll_banana(self, params):
        """
        NLL of Rosenbrock/banana distribution:
        -log p(x1, x2) = 0.5 * [(a - x1)^2 + b * (x2 - x1^2)^2]

        Generalizes to d dimensions as:
        -log p(x) = 0.5 * sum_{i=1}^{d-1} [(a - x_i)^2 + b * (x_{i+1} - x_i^2)^2]
        """
        a, b = self._a, self._b
        nll = 0.0
        for i in range(len(params) - 1):
            nll += 0.5 * ((a - params[i]) ** 2 + b * (params[i + 1] - params[i] ** 2) ** 2)
        return nll

    def _nll_multimodal(self, params):
        """
        NLL of a mixture of Gaussians:
        -log p(x) = -log sum_k w_k * N(x; mu_k, Sigma_k)
                   = -logsumexp(log(w_k) - 0.5 * (x - mu_k)^T Sigma_k^{-1} (x - mu_k))
        """
        log_components = []
        for log_w, mu, inv_var in self._modes:
            diff = params - mu
            log_density = log_w - 0.5 * np.sum(diff ** 2 * inv_var)
            log_components.append(log_density)
        # logsumexp for numerical stability
        max_log = max(log_components)
        log_sum = max_log + np.log(sum(np.exp(lc - max_log) for lc in log_components))
        return -log_sum
