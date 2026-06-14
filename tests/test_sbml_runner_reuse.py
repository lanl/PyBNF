"""Tests for the process-level RoadRunner reuse cache (perf follow-up to #415).

SbmlModelNoTimeout reuses one compiled RoadRunner per process across
evaluations instead of rebuilding it every execute(). Reuse must be
bit-for-bit identical to the prior fresh-load path; these tests pin that
parity across the action types that touch runner state (TimeCourse,
ParamScan, IC-fit, mutant) and confirm the runner is built exactly once
across many evaluations.
"""
import os
import pickle

import numpy as np
import pytest

from .context import pset


# The cached runner is compiled from XML once per process and restored with
# resetToOrigin() before each reuse; that must reproduce a fresh per-evaluation
# load exactly.
FILE = 'bngl_files/raf.xml'


def _abs(file):
    return os.path.join(os.getcwd(), file)


def _pset(values):
    params = [pset.FreeParameter(name, 'uniform_var', 0.0, 1e9, val)
              for name, val in values.items()]
    return pset.PSet(params)


def _model(values, actions, mutant=None):
    m = pset.SbmlModelNoTimeout(FILE, _abs(FILE), pset=_pset(values), actions=actions)
    if mutant is not None:
        m.add_mutant(mutant)
    return m


def _execute(values, actions, mutant=None):
    """Run one fresh model instance, mirroring a per-job unpickled model."""
    return _model(values, actions, mutant).execute(os.getcwd(), 'reuse_test', 1000)


def _columns(result):
    """suffix -> {column_name: column_values}.

    Keyed by name rather than position because a model's species column order
    derives from a Python set and so can differ between two model instances
    (e.g. a fresh build vs. an unpickled one); the Data object is
    self-describing, so only the per-column values matter for correctness.
    """
    return {suffix: {name: data[name] for name in data.cols}
            for suffix, data in result.items()}


def _assert_bit_equal(fresh, reused):
    assert set(fresh) == set(reused)
    for suffix in fresh:
        a, b = fresh[suffix], reused[suffix]
        assert set(a) == set(b)
        for name in a:
            assert np.array_equal(a[name], b[name]), (
                f'suffix {suffix!r} column {name!r}: reused runner diverged '
                f'from fresh load (max abs diff {np.max(np.abs(a[name] - b[name])):.3e})'
            )


@pytest.mark.roadrunner
class TestRoadRunnerReuse:

    def setup_method(self):
        # Start each test from an empty cache so the first execute is a fresh
        # build and "constructed once" counts are unambiguous.
        pset._RUNNER_CACHE.clear()

    def teardown_method(self):
        pset._RUNNER_CACHE.clear()

    def _fresh_then_reused(self, values, actions, mutant=None):
        """Result with the cache forced cold (fresh build) vs warm (reuse)."""
        pset._RUNNER_CACHE.clear()
        fresh = _columns(_execute(values, actions, mutant))
        assert pset._RUNNER_CACHE, 'expected a runner to be cached after execute'
        reused = _columns(_execute(values, actions, mutant))  # warm cache -> reuse
        return fresh, reused

    @pytest.mark.parametrize('values', [
        {'K3': 8000.0, 'K5': 0.3},
        {'K3': 2000.0, 'K5': 0.3},
        {'K3': 5000.0, 'K5': 0.7},
    ])
    def test_timecourse_parity(self, values):
        action = pset.TimeCourse({'time': '1000', 'step': '10'})
        fresh, reused = self._fresh_then_reused(values, (action,))
        _assert_bit_equal(fresh, reused)

    def test_param_scan_parity(self):
        action = pset.ParamScan(
            {'param': 'K3', 'min': '500', 'max': '10000', 'step': '500', 'time': '1000'})
        fresh, reused = self._fresh_then_reused({'K5': 0.3}, (action,))
        _assert_bit_equal(fresh, reused)

    def test_ic_fit_parity(self):
        # A species initial condition in the param_set exercises the
        # init([species]) restore path of resetToOrigin().
        action = pset.TimeCourse({'time': '1000', 'step': '10'})
        fresh, reused = self._fresh_then_reused({'K3': 8000.0, 'RR': 50.0}, (action,))
        _assert_bit_equal(fresh, reused)

    def test_mutant_parity(self):
        mutant = pset.MutationSet((pset.Mutation('K3', '*', 4),), suffix='k3x4')
        action = pset.TimeCourse({'time': '1000', 'step': '10'})
        fresh, reused = self._fresh_then_reused(
            {'K3': 2000.0, 'K5': 0.3}, (action,), mutant=mutant)
        _assert_bit_equal(fresh, reused)

    def test_runner_built_exactly_once_across_evaluations(self, monkeypatch):
        # Each evaluation in a fit is a freshly unpickled model instance with
        # the same abs_file_path; the runner must be built once and reused.
        calls = {'n': 0}
        original = pset.SbmlModelNoTimeout._build_runner

        def counting_build(self):
            calls['n'] += 1
            return original(self)

        monkeypatch.setattr(pset.SbmlModelNoTimeout, '_build_runner', counting_build)

        action = pset.TimeCourse({'time': '1000', 'step': '10'})
        results = []
        for i in range(5):
            values = {'K3': 2000.0 + 1000.0 * i, 'K5': 0.3}
            results.append(_columns(_execute(values, (action,))))

        assert calls['n'] == 1, f'expected one runner build, got {calls["n"]}'
        # Distinct parameter sets must still yield distinct simulation output
        # (i.e. reuse did not freeze the first evaluation's parameters).
        assert not np.array_equal(
            results[0]['time_course']['R'], results[-1]['time_course']['R'])

    def test_dask_lifecycle_pickle_roundtrip(self, monkeypatch):
        # Faithful to the real fit: each evaluation is a model unpickled in the
        # worker (no __init__), sharing one process-cached runner. Results must
        # vary with the parameters and the runner must be built exactly once.
        calls = {'n': 0}
        original = pset.SbmlModelNoTimeout._build_runner

        def counting_build(self):
            calls['n'] += 1
            return original(self)

        monkeypatch.setattr(pset.SbmlModelNoTimeout, '_build_runner', counting_build)

        action = pset.TimeCourse({'time': '1000', 'step': '10'})
        base = _model({'K3': 8000.0, 'K5': 0.3}, (action,))
        blob = pickle.dumps(base)

        reused_results = []
        for k3 in (8000.0, 2000.0, 5000.0):
            job_model = pickle.loads(blob)            # mirrors a dask worker
            job_model.param_set = pset.PSet([
                pset.FreeParameter('K3', 'uniform_var', 0.0, 1e9, k3),
                pset.FreeParameter('K5', 'uniform_var', 0.0, 1e9, 0.3)])
            reused_results.append(
                _columns(job_model.execute(os.getcwd(), 'reuse_test', 1000)))

        assert calls['n'] == 1, f'expected one runner build, got {calls["n"]}'

        # Independently recompute each result with a cold cache (fresh build).
        for k3, reused in zip((8000.0, 2000.0, 5000.0), reused_results):
            pset._RUNNER_CACHE.clear()
            fresh = _columns(_execute({'K3': k3, 'K5': 0.3}, (action,)))
            _assert_bit_equal(fresh, reused)

    def test_interleaved_actions_no_drift(self):
        # A scan followed by a time course on the shared runner must match a
        # time course from a fresh load (resetToOrigin clears scan residue).
        scan = pset.ParamScan(
            {'param': 'K3', 'min': '500', 'max': '10000', 'step': '500', 'time': '1000'})
        tc = pset.TimeCourse({'time': '1000', 'step': '10'})

        pset._RUNNER_CACHE.clear()
        fresh_tc = _columns(_execute({'K3': 8000.0, 'K5': 0.3}, (tc,)))

        pset._RUNNER_CACHE.clear()
        _execute({'K5': 0.3}, (scan,))                     # warms + dirties cache
        reused_tc = _columns(_execute({'K3': 8000.0, 'K5': 0.3}, (tc,)))

        _assert_bit_equal(fresh_tc, reused_tc)
