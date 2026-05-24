"""
Regression tests for lanl/PyBNF#388.

A failed fit evaluation must be turned into a FailedSimulation (penalizing
objective) so the run continues, rather than crashing the whole fit with
``AttributeError: 'tuple' object has no attribute 'score'``.

The crash came from ``custom_as_completed`` silently no longer wrapping errored
futures (dask renamed the coroutine it used to override), so an errored future
leaked into the main loop as a raw ``(type, exc, traceback)`` tuple. These tests
pin down both the wrapping behavior and the eval-failure -> FailedSimulation path.
"""

import os
import queue as queuemod
import shutil
import tempfile
import types

import numpy as np

from .context import algorithms, data, pset, printing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pset():
    return pset.PSet([pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 5.0)])


def _make_data():
    d = data.Data()
    d.cols = {'time': 0, 'v1_result': 1}
    d.data = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    return d


class _FakeFuture:
    def __init__(self, status, key='sim_1'):
        self.status = status
        self.key = key


def _bare_completed(with_results=True):
    """A custom_as_completed instance with just the attributes _get_and_raise needs."""
    ac = object.__new__(algorithms.custom_as_completed)
    ac.with_results = with_results
    ac.queue = queuemod.Queue()
    return ac


# ---------------------------------------------------------------------------
# custom_as_completed._get_and_raise: the root-cause fix
# ---------------------------------------------------------------------------

def test_errored_future_wrapped_in_daskerror():
    """An errored future must surface as a DaskError, not a raw (typ, exc, tb) tuple."""
    ac = _bare_completed()
    fut = _FakeFuture('error')
    exc = ValueError('boom')
    ac.queue.put((fut, (ValueError, exc, exc.__traceback__)))

    out_fut, result = ac._get_and_raise()

    assert out_fut is fut
    assert isinstance(result, algorithms.DaskError)
    assert result.error is exc
    assert 'boom' in result.traceback


def test_cancelled_future_wrapped_in_cancellederror():
    ac = _bare_completed()
    fut = _FakeFuture('cancelled', key='sim_42')
    ac.queue.put((fut, None))

    _out_fut, result = ac._get_and_raise()

    assert isinstance(result, algorithms.CancelledError)


def test_successful_future_passes_through_unchanged():
    ac = _bare_completed()
    fut = _FakeFuture('finished')
    res = algorithms.Result(_make_pset(), {}, 'sim_1')
    res.score = 1.23
    ac.queue.put((fut, res))

    out_fut, result = ac._get_and_raise()

    assert out_fut is fut
    assert result is res


# ---------------------------------------------------------------------------
# Job.run_simulation: a scoring failure becomes a FailedSimulation
# ---------------------------------------------------------------------------

class _FakeModel:
    name = 'fake'

    def copy_with_param_set(self, params):
        return self

    def execute(self, folder, filename, timeout):
        return {'time_course': _make_data()}


class _RaisingCalc:
    """Stand-in for a scattered ObjectiveCalculator whose scoring blows up."""

    def __init__(self, exc):
        self._exc = exc

    def result(self):
        return self

    def evaluate_objective(self, simdata, ps, show_warnings=False):
        raise self._exc


def _run_job_with_calc(calc_exc):
    tmp = tempfile.mkdtemp(prefix='pybnf388_')
    try:
        job = algorithms.Job(
            [_FakeModel()], _make_pset(), 'sim_1', tmp, None,
            calc_future=_RaisingCalc(calc_exc), norm_settings=None,
            postproc_settings=dict(),
        )
        return job.run_simulation()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_objective_eval_pybnferror_becomes_failed_simulation():
    """The exact issue scenario: scoring raises PybnfError -> FailedSimulation, no crash."""
    res = _run_job_with_calc(printing.PybnfError('simulation output missing exp column'))
    assert isinstance(res, algorithms.FailedSimulation)
    assert res.fail_type == 1


def test_objective_eval_generic_error_becomes_failed_simulation():
    res = _run_job_with_calc(RuntimeError('numerical blowup'))
    assert isinstance(res, algorithms.FailedSimulation)
    assert res.fail_type == 1


# ---------------------------------------------------------------------------
# add_to_trajectory: local-eval failure penalizes instead of crashing
# ---------------------------------------------------------------------------

def test_add_to_trajectory_eval_failure_penalizes():
    """When the objective is scored locally and raises, the result is penalized, not fatal."""
    recorded = []

    def raising_eval(simdata, exp_data, ps, constraints):
        raise printing.PybnfError('missing column')

    algo = object.__new__(algorithms.Algorithm)
    algo.config = types.SimpleNamespace(
        config={'normalization': None}, postprocessing={}, constraints=[])
    algo.objective = types.SimpleNamespace(evaluate_multiple=raising_eval)
    algo.exp_data = {}
    algo.trajectory = types.SimpleNamespace(
        add=lambda ps, score, name: recorded.append((score, name)))

    res = algorithms.Result(_make_pset(), {'fake': {'time_course': _make_data()}}, 'sim_1')
    assert res.score is None

    algo.add_to_trajectory(res)  # must not raise

    assert res.score == np.inf
    assert recorded == [(np.inf, 'sim_1')]
