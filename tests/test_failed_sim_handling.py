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

import errno
import logging
import os
import shutil
import tempfile
import types

import numpy as np
import pytest

from .context import algorithms, data, pset, printing
from pybnf.algorithms import core as algorithms_core


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


class _ScoringCalc:
    """Stand-in for a scattered ObjectiveCalculator whose scoring succeeds, so a job
    reaches its result and the only thing under test is the folder creation."""

    def result(self):
        return self

    def evaluate_objective(self, simdata, ps, show_warnings=False):
        return 1.0


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


class _RefusingModel(_FakeModel):
    """A model whose *simulation* refuses -- a construct this job_type cannot handle."""

    def execute(self, folder, filename, timeout):
        raise printing.PybnfError('this model cannot be run on the gradient path')


def _run_job_with_model(model):
    tmp = tempfile.mkdtemp(prefix='pybnf532_')
    try:
        job = algorithms.Job(
            [model], _make_pset(), 'sim_1', tmp, None,
            calc_future=None, norm_settings=None, postproc_settings=dict())
        return job.run_simulation()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_simulation_pybnferror_is_raised_not_swallowed():
    """A refusal raised while SIMULATING is a property of the setup, not of this parameter
    set, so it must reach the user rather than become one "unknown error" per evaluation and
    a fit that "finished" with ``inf`` at every start (#532). Contrast the scoring arm above,
    which keeps penalizing the point (#388)."""
    with pytest.raises(printing.PybnfError, match='gradient path'):
        _run_job_with_model(_RefusingModel())


def test_simulation_generic_error_still_becomes_failed_simulation():
    """Only the user-targeted refusal fails fast; an ordinary blowup still penalizes the
    point so the run continues."""

    class _BlowingUpModel(_FakeModel):
        def execute(self, folder, filename, timeout):
            raise RuntimeError('integrator exploded')

    res = _run_job_with_model(_BlowingUpModel())
    assert isinstance(res, algorithms.FailedSimulation)
    assert res.fail_type == 2


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


# ---------------------------------------------------------------------------
# Job.run_simulation: the simulation folder cannot be created (#791)
# ---------------------------------------------------------------------------
class TestSimulationFolderCreationFailure:
    """Creating the job's folder is retried by taking a new name. That recovers the
    one case it was written for -- dask running the same job twice, so the folder is
    already there -- and cannot recover any other ``OSError``. Retrying those anyway
    spent 1000 attempts and 1001 warnings per job to reach a message that named
    neither the errno nor the strerror the exception was already carrying (#791).
    """

    def _job(self, out_dir):
        return algorithms.Job(
            [_FakeModel()], _make_pset(), 'sim_1', out_dir, None,
            calc_future=_ScoringCalc(), norm_settings=None, postproc_settings=dict(),
        )

    def _attempts(self, job, monkeypatch, err):
        """Count os.mkdir calls, raising `err` on each."""
        calls = []

        def fake_mkdir(path, *a, **kw):
            calls.append(path)
            raise err

        monkeypatch.setattr(algorithms_core.os, 'mkdir', fake_mkdir)
        res = job.run_simulation()
        return res, calls

    def test_an_existing_folder_is_retried_under_a_new_name(self, tmp_path, monkeypatch):
        """The dask-double-run case: the first name is taken, the next is free, and the
        job runs. One retry, not a failure."""
        job = self._job(str(tmp_path))
        taken = job.folder
        os.mkdir(taken)

        res = job.run_simulation()

        assert not isinstance(res, algorithms.FailedSimulation)
        assert job.folder != taken and os.path.isdir(job.folder)

    def test_a_permission_error_fails_immediately(self, tmp_path, monkeypatch):
        """Renaming cannot fix EACCES, so it must not be tried 1000 times."""
        err = PermissionError(errno.EACCES, 'Permission denied')
        res, calls = self._attempts(self._job(str(tmp_path)), monkeypatch, err)
        assert isinstance(res, algorithms.FailedSimulation)
        assert len(calls) == 1

    def test_a_missing_parent_fails_immediately(self, tmp_path, monkeypatch):
        """Same for ENOENT: a new name is still under the parent that is not there."""
        err = FileNotFoundError(errno.ENOENT, 'No such file or directory')
        res, calls = self._attempts(self._job(str(tmp_path)), monkeypatch, err)
        assert isinstance(res, algorithms.FailedSimulation)
        assert len(calls) == 1

    def test_the_reason_reaches_the_log(self, tmp_path, monkeypatch, caplog):
        """The whole point: 'no space left on device' is the answer, it is already in
        the exception, and it used to be discarded."""
        err = OSError(errno.ENOSPC, 'No space left on device')
        with caplog.at_level(logging.ERROR):
            res, _ = self._attempts(self._job(str(tmp_path)), monkeypatch, err)
        assert isinstance(res, algorithms.FailedSimulation)
        text = caplog.text
        assert 'No space left on device' in text
        assert str(errno.ENOSPC) in text
        assert 'unable to write to the Simulations folder' in text

    def test_an_unbroken_run_of_taken_names_still_gives_up(self, tmp_path, monkeypatch):
        """The 1000-attempt cap stays for the case it guards: every candidate name
        taken. Without it a pathological directory would spin forever."""
        err = FileExistsError(errno.EEXIST, 'File exists')
        res, calls = self._attempts(self._job(str(tmp_path)), monkeypatch, err)
        assert isinstance(res, algorithms.FailedSimulation)
        assert len(calls) == 1001
