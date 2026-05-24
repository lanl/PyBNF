"""
Orchestration tests for ``Algorithm.run`` — the dask-driven main loop that
submits jobs, drains ``as_completed``, translates each completion via
``result_from_completed``, folds smoothing/model groups, scores results, asks the
algorithm what to run next, and resubmits.

These are *orchestration* tests, not numerical ones: they verify the wiring
(what gets submitted, how completions are routed, when the loop stops, what gets
cancelled), not simulation values. The substitution strategy:

  * **dask client**: a synchronous **fake** (`_FakeClient`) that runs each
    submitted ``run_job`` inline and returns a finished future. ``as_completed``
    is monkeypatched to a synchronous **fake** queue that supports ``.update()``
    for resubmissions — the same public surface (``with_results=True,
    raise_errors=False``, iteration, ``update``) the real loop relies on, which
    is exactly the #388/#393 contract.
  * **model**: a backend-free **fake** returning canned data.
  * **objective**: a **fake** returning a deterministic score from the pset.
  * **algorithm**: a tiny `Algorithm` subclass driving a controlled sequence of
    generations, built via ``object.__new__`` so we set only the attributes
    ``run`` reads (the real constructor parses models and builds directories).

NOTE: ``Algorithm.run`` is a ~180-line monolith that mixes the core loop with
end-of-run file shuffling (best-fit copying, sim-dir teardown). We keep that tail
cheap (delete_old_files=0, empty model suffixes) rather than exercise it. Pulling
the per-result body out of ``run`` into its own method would let these decisions
be unit-tested without the fake-client harness — recommended before the
algorithms.py split.
"""
import os

import numpy as np
import pytest

from .context import algorithms, config, pset, printing
from pybnf.pset import Trajectory, FailedSimulationError


# --------------------------------------------------------------------------- #
# Fakes for the dask layer
# --------------------------------------------------------------------------- #
class _FakeFuture:
    def __init__(self, result, status='finished'):
        self._result = result
        self.status = status

    def result(self):
        return self._result


class _FakeClient:
    """Synchronous stand-in for a distributed.Client. Runs submitted callables
    inline and records cancellations."""

    def __init__(self):
        self.submitted = []      # list of (fn, args)
        self.cancelled = []

    def scatter(self, objs, broadcast=False):
        return [_FakeFuture(o) for o in objs]

    def submit(self, fn, *args, **kwargs):
        self.submitted.append((fn, args))
        return _FakeFuture(fn(*args))

    def cancel(self, futures):
        self.cancelled.extend(futures)


class _FakeAsCompleted:
    """Synchronous as_completed: yields (future, future.result()) and supports
    update() so the loop can enqueue resubmitted futures."""

    def __init__(self, futures, with_results=False, raise_errors=True):
        assert with_results and not raise_errors  # the contract run() depends on
        self._queue = list(futures)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            raise StopIteration
        f = self._queue.pop(0)
        return f, f.result()

    def update(self, new_futures):
        self._queue.extend(new_futures)


# --------------------------------------------------------------------------- #
# Fakes for model / objective
# --------------------------------------------------------------------------- #
def _data():
    from pybnf import data as data_mod
    d = data_mod.Data()
    d.cols = {'time': 0, 'v1_result': 1}
    d.data = np.array([[0.0, 1.0]], dtype=float)
    return d


class _ModelCopy:
    def __init__(self, name, fail):
        self.name = name
        self._fail = fail

    def execute(self, folder, file_prefix, timeout):
        if self._fail:
            raise FailedSimulationError('forced failure')
        return {'time_course': _data()}

    def save_all(self, prefix):  # called on the best-fit copy in run()'s tail
        pass


class _FakeModel:
    """Backend-free model. With fail=True every execute raises, so every Job
    becomes a FailedSimulation (drives the all-failing abort path)."""

    suffixes = []  # keeps the end-of-run best-fit copy loop a no-op

    def __init__(self, name='m', fail=False):
        self.name = name
        self._fail = fail

    def copy_with_param_set(self, params):
        return _ModelCopy(self.name, self._fail)

    def save_all(self, prefix):
        pass


class _FakeObjective:
    """Scores a pset deterministically: score = value of its single parameter."""

    def evaluate_multiple(self, simdata, exp_data, ps, constraints, show_warnings=True):
        return float(ps['v1__FREE'])


def _pset(name, value):
    p = pset.PSet([pset.FreeParameter('v1__FREE', 'uniform_var', 0, 100, value)])
    p.name = name
    return p


# --------------------------------------------------------------------------- #
# A controllable Algorithm
# --------------------------------------------------------------------------- #
class _ScriptedAlgorithm(algorithms.Algorithm):
    """Drives a fixed sequence of generations. start_run submits the initial
    psets; got_result collects them and, once a generation completes, either
    submits the next generation or returns 'STOP'."""

    def start_run(self):
        return list(self._generations[0])

    def got_result(self, res):
        self.seen.append(res)
        self._gen_count += 1
        if self._gen_count == self._gen_size:
            self._gen_count = 0
            self._gen_index += 1
            if self._gen_index < len(self._generations):
                return list(self._generations[self._gen_index])
            return 'STOP'
        return []


def _make_algorithm(tmp_path, generations, *, fail=False, max_failed=1000, cls=_ScriptedAlgorithm):
    out = str(tmp_path)
    sim_dir = out + '/Simulations'
    res_dir = out + '/Results'
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    algo = object.__new__(cls)
    algo.config = type('Cfg', (), {})()
    algo.config.config = {
        'backup_every': 10 ** 9, 'population_size': len(generations[0]), 'smoothing': 1,
        'parallelize_models': 1, 'local_objective_eval': 1, 'min_objective': -np.inf,
        'max_failed_simulations': max_failed, 'wall_time_sim': None, 'normalization': None,
        'stochastic_seed': 'auto', 'delete_old_files': 0, 'save_best_data': 0,
        'refine': 0, 'bootstrap': None, 'num_to_output': 100, 'output_dir': out,
    }
    algo.config.postprocessing = {}
    algo.config.constraints = []
    fake_model = _FakeModel(fail=fail)
    algo.config.models = {'m': fake_model}

    algo.objective = _FakeObjective()
    algo.exp_data = {}
    algo.trajectory = Trajectory(100)
    algo.job_id_counter = 0
    algo.output_counter = 0
    algo.job_group_dir = {}
    algo.fail_count = 0
    algo.success_count = 0
    algo.max_iterations = 100
    algo.model_list = [fake_model]
    algo.sim_dir = sim_dir
    algo.res_dir = res_dir
    algo.failed_logs_dir = out + '/FailedSimLogs'
    algo.calc_future = None
    algo.refine = False
    algo.bootstrap_number = None

    # scripted-generation state
    algo._generations = generations
    algo._gen_index = 0
    algo._gen_count = 0
    algo._gen_size = len(generations[0])
    algo.seen = []
    return algo


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_submits_initial_generation_and_resubmits_then_stops(tmp_path, monkeypatch):
    """The loop submits a job per initial pset, scores each returned result into
    the trajectory, resubmits the next generation when the algorithm asks, and
    stops on 'STOP'. Two generations of two psets => four run_job submissions and
    four trajectory entries."""
    monkeypatch.setattr(algorithms, 'as_completed', _FakeAsCompleted)
    gens = [
        [_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)],
        [_pset('iter1run0', 3.0), _pset('iter1run1', 4.0)],
    ]
    algo = _make_algorithm(tmp_path, gens)
    client = _FakeClient()

    algo.run(client)

    # All four jobs went through run_job.
    assert len(client.submitted) == 4
    assert all(fn is algorithms.run_job for fn, _ in client.submitted)
    # All four results scored into the trajectory (score == pset value).
    assert len(algo.seen) == 4
    assert sorted(r.score for r in algo.seen) == [3.0, 4.0, 10.0, 20.0]
    assert algo.success_count == 4 and algo.fail_count == 0
    # Best fit is the lowest score.
    np.testing.assert_allclose(algo.trajectory.best_score(), 3.0)


def test_first_job_shows_warnings_rest_do_not(tmp_path, monkeypatch):
    """Only the first submitted job carries show_warnings=True (so unused-exp-data
    warnings fire once, not once per parameter set)."""
    monkeypatch.setattr(algorithms, 'as_completed', _FakeAsCompleted)
    captured = []

    real_run_job = algorithms.run_job

    def recording_run_job(job, debug=False, failed_logs_dir=''):
        captured.append(job.show_warnings)
        return real_run_job(job, debug, failed_logs_dir)

    monkeypatch.setattr(algorithms, 'run_job', recording_run_job)
    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)], [_pset('iter1run0', 1.0), _pset('iter1run1', 2.0)]]
    algo = _make_algorithm(tmp_path, gens)

    algo.run(_FakeClient())

    assert captured[0] is True
    assert captured[1:] == [False, False, False]


def test_min_objective_breaks_loop_early(tmp_path, monkeypatch):
    """A result scoring below min_objective ends the run immediately. The break
    happens after the result is recorded in the trajectory but *before* the
    algorithm's got_result is consulted — so got_result is never called and only
    the first drained result is scored."""
    monkeypatch.setattr(algorithms, 'as_completed', _FakeAsCompleted)
    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]]
    algo = _make_algorithm(tmp_path, gens)
    algo.config.config['min_objective'] = 15.0  # the value-10 pset beats this

    algo.run(_FakeClient())

    assert algo.seen == []  # got_result never reached
    np.testing.assert_allclose(algo.trajectory.best_score(), 10.0)


def test_aborts_when_all_simulations_fail(tmp_path, monkeypatch):
    """When every job fails and none has succeeded, the run raises PybnfError
    once fail_count reaches max_failed_simulations (rather than looping forever
    on doomed work)."""
    monkeypatch.setattr(algorithms, 'as_completed', _FakeAsCompleted)
    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]]
    algo = _make_algorithm(tmp_path, gens, fail=True, max_failed=1)

    with pytest.raises(printing.PybnfError, match='all jobs are failing'):
        algo.run(_FakeClient())

    assert algo.success_count == 0 and algo.fail_count >= 1


# --------------------------------------------------------------------------- #
# Direct unit tests of the extracted per-result decisions.
#
# These need NO dask client, NO as_completed, NO filesystem — the whole point of
# pulling _fold_group_result and _record_result_and_decide out of run(). They
# exercise the same decisions as the fake-client tests above, but in isolation.
# --------------------------------------------------------------------------- #
def _bare_algo(got_result=lambda res: [], *, max_failed=3, min_objective=-np.inf):
    algo = object.__new__(algorithms.Algorithm)
    algo.fail_count = 0
    algo.success_count = 0
    algo.config = type('Cfg', (), {})()
    algo.config.config = {'max_failed_simulations': max_failed,
                          'min_objective': min_objective, 'normalization': None}
    algo.config.postprocessing = {}
    algo.config.constraints = []
    algo.objective = _FakeObjective()
    algo.exp_data = {}
    algo.trajectory = Trajectory(100)
    algo.best_fit_obj = None
    algo.job_group_dir = {}
    algo.got_result = got_result
    return algo


def _scored(name, score):
    res = algorithms.Result(_pset(name, score), {}, name)
    res.score = score
    return res


class TestRecordResultAndDecide:

    def test_success_records_and_returns_next_psets(self):
        algo = _bare_algo(got_result=lambda res: ['next_gen'])
        out = algo._record_result_and_decide(_scored('s1', 5.0))
        assert out == ['next_gen']
        assert algo.success_count == 1 and algo.fail_count == 0
        np.testing.assert_allclose(algo.trajectory.best_score(), 5.0)

    def test_min_objective_stops_before_consulting_algorithm(self):
        consulted = []
        algo = _bare_algo(got_result=lambda res: consulted.append(res) or [],
                          min_objective=10.0)
        out = algo._record_result_and_decide(_scored('s1', 5.0))  # 5 < 10
        assert out == 'STOP'
        assert consulted == []  # got_result not called on the min-objective path

    def test_got_result_stop_sets_best_fit_obj(self):
        algo = _bare_algo(got_result=lambda res: 'STOP')
        out = algo._record_result_and_decide(_scored('s1', 5.0))
        assert out == 'STOP'
        np.testing.assert_allclose(algo.best_fit_obj, 5.0)

    def test_failed_simulation_aborts_when_none_succeeded(self):
        algo = _bare_algo(max_failed=1)
        fs = algorithms.FailedSimulation(_pset('s1', 1.0), 's1', 1)
        with pytest.raises(printing.PybnfError, match='all jobs are failing'):
            algo._record_result_and_decide(fs)
        assert algo.fail_count == 1

    def test_failed_simulation_does_not_abort_after_a_success(self):
        algo = _bare_algo(max_failed=1)
        algo.success_count = 1  # something has already worked
        fs = algorithms.FailedSimulation(_pset('s1', 1.0), 's1', 1)
        out = algo._record_result_and_decide(fs)  # must not raise
        assert algo.fail_count == 1
        assert out == []

    def test_cancelled_error_is_fatal(self):
        algo = _bare_algo()
        with pytest.raises(printing.PybnfError):
            algo._record_result_and_decide(algorithms.CancelledError('sim_1'))


class TestFoldGroupResult:

    def test_returns_none_until_every_subjob_finishes(self):
        algo = object.__new__(algorithms.Algorithm)
        group = algorithms.JobGroup('g', ['g_rep0', 'g_rep1'])
        algo.job_group_dir = {'g_rep0': group, 'g_rep1': group}

        first = algorithms.Result(_pset('g_rep0', 1.0), {'m': {'s': _data()}}, 'g_rep0')
        assert algo._fold_group_result(first) is None  # 1 of 2 in

        second = algorithms.Result(_pset('g_rep1', 1.0), {'m': {'s': _data()}}, 'g_rep1')
        combined = algo._fold_group_result(second)
        assert isinstance(combined, algorithms.Result)
        assert combined.name == 'g'  # carries the group's job_id


def test_pending_jobs_cancelled_on_stop(tmp_path, monkeypatch):
    """When the loop stops, any still-pending futures are cancelled (clean dask
    teardown)."""
    monkeypatch.setattr(algorithms, 'as_completed', _FakeAsCompleted)
    # One generation, then STOP. The second result's got_result returns STOP, but
    # both initial jobs were already drained, so pending is empty at stop — assert
    # cancel was called (with whatever remained) rather than skipped.
    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]]
    algo = _make_algorithm(tmp_path, gens)
    client = _FakeClient()

    algo.run(client)

    # cancel() is always called at teardown; with a synchronous fake everything
    # has completed, so the cancelled list is empty but the call happened.
    assert client.cancelled == []
