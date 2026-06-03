"""
Regression tests for lanl/PyBNF#388.

A failed fit evaluation must be turned into a FailedSimulation (penalizing
objective) so the run continues, rather than crashing the whole fit with
``AttributeError: 'tuple' object has no attribute 'score'``.

The crash came from ``custom_as_completed`` silently no longer wrapping errored
futures (dask renamed the private coroutine it used to override), so an errored
future leaked into the main loop as a raw ``(type, exc, traceback)`` tuple. That
subclass has been replaced by ``result_from_completed``, which translates the
output of stock ``as_completed(with_results=True, raise_errors=False)`` using only
dask's public Future API. These tests pin down that translation and the
eval-failure -> FailedSimulation path.
"""

import shutil
import tempfile
import types

import numpy as np
import pytest

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
    """Minimal stand-in for a dask Future: result_from_completed only reads .status."""
    def __init__(self, status):
        self.status = status


# ---------------------------------------------------------------------------
# result_from_completed: the root-cause fix
# ---------------------------------------------------------------------------

def test_errored_future_becomes_failed_simulation():
    """An errored future (raw (typ, exc, tb) tuple) must become a FailedSimulation."""
    exc = RuntimeError('boom')
    res = algorithms.result_from_completed(
        _FakeFuture('error'), (RuntimeError, exc, exc.__traceback__), _make_pset(), 'sim_1')

    assert isinstance(res, algorithms.FailedSimulation)
    assert res.fail_type == 3


def test_errored_future_pybnferror_is_reraised():
    """A user-targeted PybnfError should abort the run, not be silently penalized."""
    exc = printing.PybnfError('bad config that would fail every job')
    with pytest.raises(printing.PybnfError):
        algorithms.result_from_completed(
            _FakeFuture('error'), (printing.PybnfError, exc, exc.__traceback__),
            _make_pset(), 'sim_1')


def test_cancelled_future_returned_unchanged():
    fut = _FakeFuture('cancelled')
    ce = algorithms.CancelledError('sim_1')
    res = algorithms.result_from_completed(fut, ce, _make_pset(), 'sim_1')

    assert res is ce


def test_successful_result_passes_through_unchanged():
    fut = _FakeFuture('finished')
    result = algorithms.Result(_make_pset(), {}, 'sim_1')
    result.score = 1.23
    out = algorithms.result_from_completed(fut, result, _make_pset(), 'sim_1')

    assert out is result


def test_unexpected_result_type_becomes_failed_simulation():
    """A bare tuple leaking through (the original #388 crash) is handled, not fatal."""
    out = algorithms.result_from_completed(
        _FakeFuture('finished'), ('a', 'bare', 'tuple'), _make_pset(), 'sim_1')

    assert isinstance(out, algorithms.FailedSimulation)
    assert out.fail_type == 3


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

class _ConcreteAlgorithm(algorithms.Algorithm):
    """Concrete Algorithm: the base's start_run/got_result are @abstractmethod
    (ADR-0007), so the bare base can't be instantiated. add_to_trajectory (the
    method under test) is inherited unchanged."""

    def start_run(self):
        return []

    def got_result(self, res):
        return []


def test_add_to_trajectory_eval_failure_penalizes():
    """When the objective is scored locally and raises, the result is penalized, not fatal."""
    recorded = []

    def raising_eval(simdata, exp_data, ps, constraints):
        raise printing.PybnfError('missing column')

    algo = object.__new__(_ConcreteAlgorithm)
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
