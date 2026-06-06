"""Tests for pybnf.analytical_model.AnalyticalModel."""

import json

import numpy as np
import pytest

from pybnf import analytical_model


def _load_model(tmp_path, spec):
    """Write ``spec`` to a .target file and load it as an AnalyticalModel (so the
    __init__ path — including the rotated-Gaussian precision inversion — runs)."""
    path = tmp_path / 'm.target'
    path.write_text(json.dumps(spec))
    return analytical_model.AnalyticalModel(str(path))


def test_compute_nll_unknown_target_type_raises():
    """_compute_nll must fail loud on an unrecognized target_type rather than
    fall off the if/elif chain and return an implicit None. __init__ already
    rejects unknown types, so this guards the case where a new target type is
    added to __init__ but not wired into _compute_nll."""
    m = object.__new__(analytical_model.AnalyticalModel)
    m.target_type = 'not_a_real_target_type'
    with pytest.raises(ValueError):
        m._compute_nll(np.array([0.0, 0.0]))


# --------------------------------------------------------------------------- #
# rotated_gaussian: NLL = 0.5 (x-mu)^T Sigma^{-1} (x-mu)  (#405)
# --------------------------------------------------------------------------- #
def test_rotated_gaussian_nll_is_zero_at_the_mode(tmp_path):
    """The mode is ``mu`` (NLL 0 there), so optimizer mode-recovery still has
    ``mu`` as its oracle."""
    mu = [2.0, -1.0]
    cov = [[4.0, 1.5], [1.5, 2.0]]
    m = _load_model(tmp_path, {'type': 'rotated_gaussian', 'mean': mu, 'covariance': cov})
    assert m._nll_rotated_gaussian(np.array(mu)) == pytest.approx(0.0, abs=1e-12)


def test_rotated_gaussian_nll_matches_quadratic_form(tmp_path):
    """NLL equals the closed-form ``0.5 (x-mu)^T Sigma^{-1} (x-mu)`` for an
    off-diagonal (correlated) covariance, computed independently here."""
    mu = np.array([1.0, -2.0])
    cov = np.array([[4.0, 1.5], [1.5, 3.0]])
    m = _load_model(tmp_path, {'type': 'rotated_gaussian',
                               'mean': mu.tolist(), 'covariance': cov.tolist()})
    x = np.array([2.5, 0.5])
    diff = x - mu
    expected = 0.5 * diff @ np.linalg.inv(cov) @ diff
    assert m._nll_rotated_gaussian(x) == pytest.approx(expected, rel=1e-12)


def test_rotated_gaussian_with_diagonal_cov_reduces_to_axis_aligned(tmp_path):
    """A *diagonal* covariance is the separable case: the rotated-Gaussian NLL
    must then equal the axis-aligned ``gaussian`` NLL (variance = the diagonal).
    Anchors the full-covariance form to the existing diagonal one."""
    mu = [0.5, -1.0, 2.0]
    var = [1.0, 4.0, 0.25]
    rot = _load_model(tmp_path, {'type': 'rotated_gaussian', 'mean': mu,
                                 'covariance': np.diag(var).tolist()})
    diag = _load_model(tmp_path, {'type': 'gaussian', 'mean': mu, 'variance': var})
    x = np.array([1.5, 0.0, 1.0])
    assert rot._nll_rotated_gaussian(x) == pytest.approx(diag._nll_gaussian(x), rel=1e-12)
