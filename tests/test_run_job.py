"""
Tests for ``run_job`` — the thin wrapper PyBNF hands to ``client.submit`` instead
of ``Job.run_simulation`` (passing the bound method leaks the Job into dask's
memory). Its whole job is to delegate and to translate the one error dask can't
see — the OS thread-limit ``RuntimeError`` — into a ``FailedSimulation`` so the
run survives. Everything else must propagate unchanged.

These use a fake Job (run_job only calls ``.run_simulation`` and, on the
thread-limit path, reads ``.params`` / ``.job_id``), so no simulation backend is
required.
"""
import pytest

from .context import algorithms, pset


def _make_pset():
    return pset.PSet([pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 5.0)])


class _FakeJob:
    """Records the (debug, failed_logs_dir) it was called with and returns a
    canned value or raises a canned exception."""

    def __init__(self, *, returns=None, raises=None):
        self.params = _make_pset()
        self.job_id = 'sim_1'
        self._returns = returns
        self._raises = raises
        self.calls = []

    def run_simulation(self, debug=False, failed_logs_dir=''):
        self.calls.append((debug, failed_logs_dir))
        if self._raises is not None:
            raise self._raises
        return self._returns


def test_delegates_and_returns_result():
    """Happy path: run_job returns whatever run_simulation returns, and forwards
    the debug flag and failed-logs directory verbatim."""
    sentinel = algorithms.Result(_make_pset(), {}, 'sim_1')
    job = _FakeJob(returns=sentinel)

    out = algorithms.run_job(job, debug=True, failed_logs_dir='/tmp/fl')

    assert out is sentinel
    assert job.calls == [(True, '/tmp/fl')]


def test_thread_limit_runtimeerror_becomes_failed_simulation():
    """The one error dask can't catch: 'can't start new thread'. It must be
    converted to a FailedSimulation (fail_type 1) carrying the job's identity,
    not propagated (which would crash the worker / the run)."""
    job = _FakeJob(raises=RuntimeError("can't start new thread"))

    out = algorithms.run_job(job)

    assert isinstance(out, algorithms.FailedSimulation)
    assert out.fail_type == 1
    assert out.pset is job.params
    assert out.name == job.job_id


def test_other_runtimeerror_propagates():
    """A RuntimeError that is *not* the thread-limit message must propagate
    unchanged — run_job only special-cases the thread-limit string."""
    boom = RuntimeError('some other runtime problem')
    job = _FakeJob(raises=boom)

    with pytest.raises(RuntimeError, match='some other runtime problem'):
        algorithms.run_job(job)


def test_non_runtime_exception_propagates():
    """Non-RuntimeError exceptions are outside run_job's contract and propagate
    (Job.run_simulation is responsible for turning sim failures into
    FailedSimulation; run_job does not double-handle them)."""
    job = _FakeJob(raises=ValueError('not a runtime error'))

    with pytest.raises(ValueError):
        algorithms.run_job(job)
