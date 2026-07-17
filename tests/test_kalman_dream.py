"""Kalman-inspired DREAM (DREAM(KZS); ADR-0067 Stage 3, issue #358) tests.

Stage 3 adds the ``proposal = kalman`` operator, whose gain is built from the
archive's parameter<->output cross-covariance. That needs the *model output
vector* ``f(theta)`` (not just the scalar score), the observed data ``d``, and the
Gaussian measurement variance ``R`` -- surfaced by the objective seam
``LikelihoodObjective.aligned_prediction_data`` (Stage 3a, the output-augmented
archive plumbing).

This file starts with the Stage 3a extractor unit tests (BNG-free: they exercise
``aligned_prediction_data`` directly on hand-built Data). The Kalman proposal's
own gain-math unit tests and the closed-form linear-Gaussian posterior-recovery
oracle land with Stage 3b.
"""
import numpy as np

from pybnf import objective, data
from pybnf.pset import PSet, FreeParameter


def _data(arr, cols):
    d = data.Data()
    d.data = np.array(arr, dtype=float)
    d.cols = dict(cols)
    d.headers = {v: k for k, v in cols.items()}
    return d


def _pset():
    return PSet([FreeParameter('p1', 'uniform_var', 0.0, 1.0, value=0.5)])


# --------------------------------------------------------------------------- #
# Stage 3a: LikelihoodObjective.aligned_prediction_data
# --------------------------------------------------------------------------- #
def test_gaussian_extractor_aligns_prediction_observation_variance():
    """chi_sq (a linear-scale Gaussian likelihood) returns the raw model output
    f(theta), the observed data d, and sigma**2 -- index-aligned over the walk
    row -> sorted(observable columns)."""
    obj = objective.ChiSquareObjective()
    sim = _data([[0.0, 2.0], [1.0, 4.0]], {'time': 0, 'y': 1})
    exp = _data([[0.0, 2.5, 0.5], [1.0, 3.5, 0.25]], {'time': 0, 'y': 1, 'y_SD': 2})
    out = obj.aligned_prediction_data({'m': {'s': sim}}, {'m': {'s': exp}}, _pset())
    assert out is not None
    prediction, observation, variance = out
    assert np.allclose(prediction, [2.0, 4.0])
    assert np.allclose(observation, [2.5, 3.5])
    assert np.allclose(variance, [0.25, 0.0625])   # sigma**2 from the _SD column


def test_extractor_alignment_matches_pointwise_order():
    """The extractor walks the same points in the same order as evaluate_pointwise,
    so its observation vector equals the data the pointwise log-likelihoods score --
    the guarantee the Kalman innovation d - f(x) relies on for correspondence."""
    obj = objective.ChiSquareObjective()
    # Two observables (x, y) over two rows -> sorted columns give x before y per row.
    sim = _data([[0.0, 1.0, 10.0], [1.0, 2.0, 20.0]], {'time': 0, 'x': 1, 'y': 2})
    exp = _data([[0.0, 1.1, 10.1, 1.0, 1.0], [1.0, 2.1, 20.1, 1.0, 1.0]],
                {'time': 0, 'x': 1, 'y': 2, 'x_SD': 3, 'y_SD': 4})
    sd, ed = {'m': {'s': sim}}, {'m': {'s': exp}}
    prediction, observation, _var = obj.aligned_prediction_data(sd, ed, _pset())
    ids, _vals = obj.evaluate_pointwise(sd, ed, _pset())
    assert len(prediction) == len(ids) == 4
    # row0: x,y  then row1: x,y  -> predictions in that exact order
    assert np.allclose(prediction, [1.0, 10.0, 2.0, 20.0])
    assert np.allclose(observation, [1.1, 10.1, 2.1, 20.1])


def test_direct_pass_returns_none():
    """A non-likelihood objective (analytical direct_pass) has no output/residual
    vector, so the extractor is a no-op -- the gate that makes proposal = kalman
    error clearly on an analytical target."""
    sim = _data([[0.0, 1.23]], {'index': 0, 'score': 1})
    exp = _data([[0.0, 0.0]], {'index': 0, 'score': 1})
    out = objective.DirectPassObjective().aligned_prediction_data(
        {'m': {'s': sim}}, {'m': {'s': exp}}, _pset())
    assert out is None


def test_lognormal_returns_none():
    """A log-scale Gaussian (lognormal) has no linear-space measurement variance R,
    so the Kalman extractor declines it (returns None) rather than mis-forming R."""
    obj = objective.LogNormalObjective()
    sim = _data([[0.0, 2.0], [1.0, 4.0]], {'time': 0, 'y': 1})
    exp = _data([[0.0, 2.5, 0.5], [1.0, 3.5, 0.5]], {'time': 0, 'y': 1, 'y_SD': 2})
    out = obj.aligned_prediction_data({'m': {'s': sim}}, {'m': {'s': exp}}, _pset())
    assert out is None
