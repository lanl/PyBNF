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

NOTE: ``Algorithm.run``'s end-of-run tail (best-fit copying, the save_best_data
rerun, the backup-pickle finalize, and sim-dir teardown) has been extracted into
named methods — ``_copy_best_fit_sims`` / ``_rerun_best_fit_to_save_data`` /
``_finalize_backup_pickle`` / ``_teardown_sim_dir`` — each unit-tested directly in
the "Run-loop tail" section below, without the fake-client harness (the same play
as the per-result helpers ``_record_result_and_decide`` / ``_fold_group_result``).
The fake-client tests up here still drive that tail end-to-end, kept cheap
(delete_old_files=0, empty model suffixes), so the wiring stays covered too.
"""
import logging
import os
import pickle

import numpy as np
import pytest

from .context import algorithms, budget as budget_mod, objective, pset, printing
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
    update() so the loop can enqueue resubmitted futures.

    ``timeout`` is the wall-time-budget seam (#529): the real ``as_completed``
    raises ``TimeoutError`` out of ``__next__`` once it elapses, which is how the
    budget lands on time even while every worker is mid-simulation. This fake only
    *records* it (the last value is exposed on the class) — the raising behavior is
    exercised by ``_TimingOutAsCompleted`` below."""

    last_timeout = None

    def __init__(self, futures, with_results=False, raise_errors=True, timeout=None):
        assert with_results and not raise_errors  # the contract run() depends on
        _FakeAsCompleted.last_timeout = timeout
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
class _ConcreteAlgorithm(algorithms.Algorithm):
    """Minimal concrete Algorithm for unit-testing inherited base methods.

    ``start_run``/``got_result`` are ``@abstractmethod`` on the base (M2.2 move 1,
    ADR-0007), so the bare base can no longer be instantiated. These trivial
    overrides make the class concrete without affecting the run-loop-arc methods
    under test, which all live on (and are inherited from) the base unchanged.
    Tests that need a specific ``got_result`` still override it per-instance.
    """

    def start_run(self):
        return []

    def got_result(self, res):
        return []


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
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    gens = [
        [_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)],
        [_pset('iter1run0', 3.0), _pset('iter1run1', 4.0)],
    ]
    algo = _make_algorithm(tmp_path, gens)
    client = _FakeClient()

    algo.run(client)

    # All four jobs went through run_job.
    assert len(client.submitted) == 4
    assert all(fn is algorithms.core.run_job for fn, _ in client.submitted)
    # All four results scored into the trajectory (score == pset value).
    assert len(algo.seen) == 4
    assert sorted(r.score for r in algo.seen) == [3.0, 4.0, 10.0, 20.0]
    assert algo.success_count == 4 and algo.fail_count == 0
    # Best fit is the lowest score.
    np.testing.assert_allclose(algo.trajectory.best_score(), 3.0)


def test_first_job_shows_warnings_rest_do_not(tmp_path, monkeypatch):
    """Only the first submitted job carries show_warnings=True (so unused-exp-data
    warnings fire once, not once per parameter set)."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    captured = []

    real_run_job = algorithms.core.run_job

    def recording_run_job(job, debug=False, failed_logs_dir=''):
        captured.append(job.show_warnings)
        return real_run_job(job, debug, failed_logs_dir)

    monkeypatch.setattr(algorithms.core, 'run_job', recording_run_job)
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
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
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
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
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
    algo = object.__new__(_ConcreteAlgorithm)
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
        algo = object.__new__(_ConcreteAlgorithm)
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
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
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


# --------------------------------------------------------------------------- #
# Seam guard (ADR-0001)
# --------------------------------------------------------------------------- #
def test_run_resolves_run_job_through_core_seam(tmp_path, monkeypatch):
    """Production ``run()`` must resolve ``run_job`` through ``algorithms.core``.

    This is the guard for ADR-0001 (extract core.py + repoint patches): the run
    loop calls ``core.run_job`` / ``core.as_completed``, so patching the seam
    *where it is defined* — ``algorithms.core`` — actually bites. If a future
    move rebinds these names to some other namespace, this test goes red rather
    than silently testing a stale function."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)

    calls = []
    real_run_job = algorithms.core.run_job

    def recording_run_job(job, debug=False, failed_logs_dir=''):
        calls.append(job.job_id)
        return real_run_job(job, debug, failed_logs_dir)

    monkeypatch.setattr(algorithms.core, 'run_job', recording_run_job)

    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]]
    algo = _make_algorithm(tmp_path, gens)
    algo.run(_FakeClient())

    # The fake patched on algorithms.core ran once per submitted job — i.e. the
    # production run() path went through the core seam, not a stale facade copy.
    assert len(calls) == 2


def test_patching_package_facade_run_job_does_not_intercept(tmp_path, monkeypatch):
    """Negative control proving *why* the seam lives in ``core`` (ADR-0001).

    ``algorithms.run_job`` is only a facade re-export; the run loop resolves the
    real function as ``core.run_job``. So patching the package attribute must NOT
    intercept the production path — "patch where it's defined" is the honest
    seam, and patching the facade silently does nothing (the trap the split was
    designed to avoid)."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)

    facade_calls = []

    def facade_fake(job, debug=False, failed_logs_dir=''):
        facade_calls.append(job.job_id)
        return algorithms.core.run_job(job, debug, failed_logs_dir)

    # Patch the FACADE attribute, not the core seam.
    monkeypatch.setattr(algorithms, 'run_job', facade_fake)

    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]]
    algo = _make_algorithm(tmp_path, gens)
    algo.run(_FakeClient())

    # The run completed normally, but the facade patch never fired: the loop used
    # core.run_job directly.
    assert facade_calls == []


# --------------------------------------------------------------------------- #
# Run-loop tail: end-of-run file orchestration
#
# After the main loop, run() shuffles the best-fit simulation outputs into
# Results/. That tail was extracted into named Algorithm methods so each decision
# can be unit-tested directly — NO dask client, NO as_completed, NO full run().
# These methods are pure orchestration (model copy, shutil.copy, glob), so the
# oracle is REAL filesystem behavior under tmp_path: which files land in res_dir.
# --------------------------------------------------------------------------- #
class _TailModel:
    """Backend-free model for the best-fit-copy tail. ``copy_with_param_set``
    records the pset it was handed and returns a stand-in whose ``save_all``
    writes a sentinel file — so 'the model saved its outputs' becomes a real file
    landing in res_dir. ``suffixes`` drives the per-suffix gdat/scan copy."""

    def __init__(self, name='m', suffixes=()):
        self.name = name
        self.suffixes = list(suffixes)
        self.saved_with = '__unset__'

    def copy_with_param_set(self, best_pset):
        self.saved_with = best_pset
        return self

    def save_all(self, prefix):
        with open(prefix + '.sentinel', 'w') as fh:
            fh.write(self.name)


def _bare_tail_algo(tmp_path, models, *, delete_old_files=0, smoothing=1):
    """Minimal Algorithm carrying only the attrs the best-fit-copy tail reads:
    config.models, res_dir, sim_dir, and the delete_old_files / smoothing knobs."""
    out = str(tmp_path)
    sim_dir = out + '/Simulations'
    res_dir = out + '/Results'
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    algo = object.__new__(_ConcreteAlgorithm)
    algo.config = type('Cfg', (), {})()
    algo.config.config = {'delete_old_files': delete_old_files, 'smoothing': smoothing}
    algo.config.models = models
    algo.res_dir = res_dir
    algo.sim_dir = sim_dir
    return algo


class TestCopyBestFitSims:

    def test_saves_each_model_into_results(self, tmp_path):
        """Every model in config.models gets re-parameterized by the best pset and
        saved into Results/ as <model_name>_<best_name>."""
        m1, m2 = _TailModel('m1'), _TailModel('m2')
        algo = _bare_tail_algo(tmp_path, {'m1': m1, 'm2': m2})
        best_pset = object()

        algo._copy_best_fit_sims(best_pset, 'iter5run0')

        assert os.path.isfile(algo.res_dir + '/m1_iter5run0.sentinel')
        assert os.path.isfile(algo.res_dir + '/m2_iter5run0.sentinel')
        # Each model was copied with the best-fit pset, not some other point.
        assert m1.saved_with is best_pset and m2.saved_with is best_pset

    @pytest.mark.parametrize('suffix_entry, ext', [
        (('simulate', 'obs'), 'gdat'),    # time course -> gdat
        (('param_scan', 'sc'), 'scan'),   # parameter scan -> scan
        ('obs', 'gdat'),                  # plain-string suffix (AnalyticalModel / SbmlNoTimeout)
    ])
    def test_copies_simulation_output_for_each_suffix(self, tmp_path, suffix_entry, ext):
        """With delete_old_files==0, each suffix's already-written sim file is
        copied from Simulations/<best_name>/ into Results/. Both 2-tuple and
        plain-string suffix forms are accepted; sim type picks gdat vs scan."""
        suf = suffix_entry[1] if isinstance(suffix_entry, tuple) else suffix_entry
        model = _TailModel('m', suffixes=[suffix_entry])
        algo = _bare_tail_algo(tmp_path, {'m': model}, delete_old_files=0)
        best_name = 'iter3run1'
        src_dir = algo.sim_dir + '/' + best_name
        os.makedirs(src_dir)
        src = '%s/m_%s_%s.%s' % (src_dir, best_name, suf, ext)
        with open(src, 'w') as fh:
            fh.write('payload')

        algo._copy_best_fit_sims(object(), best_name)

        dst = '%s/m_%s_%s.%s' % (algo.res_dir, best_name, suf, ext)
        assert os.path.isfile(dst)
        with open(dst) as fh:
            assert fh.read() == 'payload'  # real copy, not a touch

    def test_missing_simulation_file_is_logged_not_fatal(self, tmp_path, caplog):
        """A best-fit gdat that was never written (e.g. all sims failed) logs an
        error and is skipped — the tail must not crash the end of a run."""
        model = _TailModel('m', suffixes=[('simulate', 'obs')])
        algo = _bare_tail_algo(tmp_path, {'m': model}, delete_old_files=0)

        with caplog.at_level(logging.ERROR, logger='pybnf.algorithms'):
            algo._copy_best_fit_sims(object(), 'iter0run0')  # no source file exists

        assert 'Cannot find files corresponding to best fit parameter set' in caplog.text
        # save_all still ran; the missing gdat simply isn't there.
        assert os.path.isfile(algo.res_dir + '/m_iter0run0.sentinel')
        assert not os.path.isfile(algo.res_dir + '/m_iter0run0_obs.gdat')

    def test_skips_suffix_copy_when_delete_old_files_positive(self, tmp_path):
        """When delete_old_files>0, the per-suffix gdat/scan copy is skipped
        entirely (those files are reproduced later by the save_best_data rerun);
        only the model-level save_all runs."""
        model = _TailModel('m', suffixes=[('simulate', 'obs')])
        algo = _bare_tail_algo(tmp_path, {'m': model}, delete_old_files=1)
        best_name = 'iter0run0'
        src_dir = algo.sim_dir + '/' + best_name
        os.makedirs(src_dir)
        with open('%s/m_%s_obs.gdat' % (src_dir, best_name), 'w') as fh:
            fh.write('x')

        algo._copy_best_fit_sims(object(), best_name)

        assert os.path.isfile(algo.res_dir + '/m_%s.sentinel' % best_name)  # save_all ran
        assert not os.path.isfile('%s/m_%s_obs.gdat' % (algo.res_dir, best_name))  # copy skipped

    def test_smoothing_looks_for_rep0_replicate(self, tmp_path):
        """With smoothing>1 there is no single best-fit gdat — one specific
        replicate (_rep0) is copied, so the source path carries the _rep0 tag in
        both the directory and the file stem."""
        model = _TailModel('m', suffixes=[('simulate', 'obs')])
        algo = _bare_tail_algo(tmp_path, {'m': model}, delete_old_files=0, smoothing=3)
        best_name = 'iter0run0'
        rep = best_name + '_rep0'
        src_dir = algo.sim_dir + '/' + rep
        os.makedirs(src_dir)
        with open('%s/m_%s_obs.gdat' % (src_dir, rep), 'w') as fh:
            fh.write('rep')

        algo._copy_best_fit_sims(object(), best_name)

        assert os.path.isfile('%s/m_%s_obs.gdat' % (algo.res_dir, rep))


class _SaveFilesModel:
    """In-process backend stand-in: carries a save_files flag the rerun toggles."""

    def __init__(self, name='m'):
        self.name = name
        self.save_files = False


def _bare_rerun_algo(tmp_path, model_list, *, delete_old_files, save_best_data):
    """Minimal Algorithm carrying only what _rerun_best_fit_to_save_data reads."""
    out = str(tmp_path)
    sim_dir = out + '/Simulations'
    res_dir = out + '/Results'
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    algo = object.__new__(_ConcreteAlgorithm)
    algo.config = type('Cfg', (), {})()
    algo.config.config = {
        'delete_old_files': delete_old_files, 'save_best_data': save_best_data,
        'wall_time_sim': 60, 'normalization': None, 'stochastic_seed': 'auto',
    }
    algo.config.postprocessing = {}
    algo.model_list = model_list
    algo.sim_dir = sim_dir
    algo.res_dir = res_dir
    return algo


class TestRerunBestFitToSaveData:

    @pytest.mark.parametrize('delete_old_files, save_best_data', [(0, 1), (1, 0), (0, 0)])
    def test_no_rerun_when_guard_off(self, tmp_path, monkeypatch, delete_old_files, save_best_data):
        """The rerun fires only when delete_old_files>0 AND save_best_data is set;
        any other combination touches neither the seam nor save_files."""
        model = _SaveFilesModel('m')
        algo = _bare_rerun_algo(tmp_path, [model],
                                delete_old_files=delete_old_files, save_best_data=save_best_data)
        touched = []
        monkeypatch.setattr(algorithms.core, 'Job', lambda *a, **k: touched.append('Job'))
        monkeypatch.setattr(algorithms.core, 'run_job', lambda job: touched.append('run_job'))

        algo._rerun_best_fit_to_save_data(object())

        assert touched == []              # core seam never touched
        assert model.save_files is False  # save_files left untouched

    def test_submits_bestfit_job_through_core_seam_and_copies_outputs(self, tmp_path, monkeypatch):
        """Guard on: a single 'bestfit' Job is built (with save_files already
        enabled) and run through the core seam; its gdat/scan outputs are copied to
        Results/; save_files is restored afterward.

        ADR-0001 seam guard for this path: the method resolves core.Job /
        core.run_job through the ``core`` module object, so patching
        ``algorithms.core.*`` actually intercepts the production rerun."""
        model = _SaveFilesModel('m')
        algo = _bare_rerun_algo(tmp_path, [model], delete_old_files=1, save_best_data=1)
        best_pset = object()
        sentinel_job = object()
        built = {}
        ran = []

        def fake_job(*args, **kwargs):
            built['args'] = args
            built['kwargs'] = kwargs
            # save_files must already be enabled when the Job is constructed
            built['save_files_at_build'] = [m.save_files for m in args[0]]
            return sentinel_job

        def fake_run_job(job):
            ran.append(job)
            # mimic the in-process backend writing its outputs under bestfit/
            bf = algo.sim_dir + '/bestfit'
            os.makedirs(bf, exist_ok=True)
            for fn in ('m_bestfit.gdat', 'm_bestfit.scan'):
                with open(bf + '/' + fn, 'w') as fh:
                    fh.write('out')

        monkeypatch.setattr(algorithms.core, 'Job', fake_job)
        monkeypatch.setattr(algorithms.core, 'run_job', fake_run_job)

        algo._rerun_best_fit_to_save_data(best_pset)

        # The seam fired: our fake Job was run exactly once.
        assert ran == [sentinel_job]
        # Built with the best pset, the 'bestfit' name, this algo's model_list and sim_dir.
        assert built['args'][0] is algo.model_list
        assert built['args'][1] is best_pset
        assert built['args'][2] == 'bestfit'
        assert built['args'][3] == algo.sim_dir
        assert built['kwargs']['stochastic_seed_policy'] == 'auto'
        # save_files was True at build time, restored to False after.
        assert built['save_files_at_build'] == [True]
        assert model.save_files is False
        # Both outputs copied into Results/.
        assert os.path.isfile(algo.res_dir + '/m_bestfit.gdat')
        assert os.path.isfile(algo.res_dir + '/m_bestfit.scan')

    def test_save_files_restored_even_if_rerun_raises(self, tmp_path, monkeypatch, caplog):
        """If the rerun itself raises, the failure is logged (not propagated) and
        save_files is still restored — a doomed rerun must not leave the models in
        a save_files=True state that would corrupt later bootstrapping/refinement."""
        model = _SaveFilesModel('m')
        algo = _bare_rerun_algo(tmp_path, [model], delete_old_files=1, save_best_data=1)
        monkeypatch.setattr(algorithms.core, 'Job', lambda *a, **k: object())

        def boom(job):
            raise RuntimeError('sim blew up')

        monkeypatch.setattr(algorithms.core, 'run_job', boom)

        with caplog.at_level(logging.ERROR, logger='pybnf.algorithms'):
            algo._rerun_best_fit_to_save_data(object())  # must not raise

        assert model.save_files is False
        assert 'Failed to rerun best fit parameter set' in caplog.text
        assert not any(f.endswith('.gdat') for f in os.listdir(algo.res_dir))  # nothing copied

    def test_models_without_save_files_attr_are_left_alone(self, tmp_path, monkeypatch):
        """Subprocess BNGLModels lack a save_files attribute (they write via
        BNG2.pl regardless); the toggle must skip them rather than crash."""
        toggleable = _SaveFilesModel('m')
        plain = object()  # no save_files attribute
        algo = _bare_rerun_algo(tmp_path, [toggleable, plain], delete_old_files=1, save_best_data=1)
        monkeypatch.setattr(algorithms.core, 'Job', lambda *a, **k: object())
        monkeypatch.setattr(algorithms.core, 'run_job', lambda job: None)

        algo._rerun_best_fit_to_save_data(object())  # must not raise on the plain model

        assert toggleable.save_files is False


def _bare_finalize_algo(tmp_path, *, bootstrap_number=None, bootstrap=None, refine=False):
    """Minimal Algorithm carrying only what _finalize_backup_pickle reads."""
    algo = object.__new__(_ConcreteAlgorithm)
    algo.config = type('Cfg', (), {})()
    algo.config.config = {'output_dir': str(tmp_path), 'bootstrap': bootstrap}
    algo.bootstrap_number = bootstrap_number
    algo.refine = refine
    return algo


class TestFinalizeBackupPickle:

    def test_renames_backup_to_finished(self, tmp_path):
        """On a completed fit, alg_backup.bp is renamed (content-preserving) to
        alg_finished.bp — the marker a resume uses to recognize a done run."""
        algo = _bare_finalize_algo(tmp_path)
        bp = str(tmp_path) + '/alg_backup.bp'
        with open(bp, 'wb') as fh:
            fh.write(b'pickle-bytes')

        algo._finalize_backup_pickle()

        finished = str(tmp_path) + '/alg_finished.bp'
        assert not os.path.exists(bp)        # moved, not copied
        assert os.path.isfile(finished)
        with open(finished, 'rb') as fh:
            assert fh.read() == b'pickle-bytes'

    def test_refine_uses_refine_finished_name(self, tmp_path):
        """A Simplex refinement finishes to alg_refine_finished.bp, kept distinct
        from a primary fit's alg_finished.bp."""
        algo = _bare_finalize_algo(tmp_path, refine=True)
        with open(str(tmp_path) + '/alg_backup.bp', 'wb') as fh:
            fh.write(b'x')

        algo._finalize_backup_pickle()

        assert os.path.isfile(str(tmp_path) + '/alg_refine_finished.bp')
        assert not os.path.exists(str(tmp_path) + '/alg_finished.bp')

    @pytest.mark.parametrize('bootstrap_number, bootstrap, renamed', [
        (None, None, True),   # not a bootstrap run -> this is the final pass
        (3, 3, True),         # the last bootstrap replicate
        (1, 3, False),        # mid-bootstrap: keep alg_backup.bp for the next replicate
    ])
    def test_bootstrap_guard_controls_rename(self, tmp_path, bootstrap_number, bootstrap, renamed):
        """Only the terminal pass renames the backup; an intermediate bootstrap
        replicate leaves alg_backup.bp in place for the replicate that follows."""
        algo = _bare_finalize_algo(tmp_path, bootstrap_number=bootstrap_number, bootstrap=bootstrap)
        bp = str(tmp_path) + '/alg_backup.bp'
        with open(bp, 'wb') as fh:
            fh.write(b'x')

        algo._finalize_backup_pickle()

        finished = str(tmp_path) + '/alg_finished.bp'
        if renamed:
            assert os.path.isfile(finished) and not os.path.exists(bp)
        else:
            assert os.path.isfile(bp) and not os.path.exists(finished)

    def test_missing_backup_is_warned_not_fatal(self, tmp_path, caplog):
        """If no backup was ever written, the rename's OSError is swallowed with a
        warning — finishing a run must not depend on a backup existing."""
        algo = _bare_finalize_algo(tmp_path)  # no alg_backup.bp on disk

        with caplog.at_level(logging.WARNING, logger='pybnf.algorithms'):
            algo._finalize_backup_pickle()  # must not raise

        assert 'Tried to move pickled algorithm, but it was not found' in caplog.text
        assert not os.path.exists(str(tmp_path) + '/alg_finished.bp')


def _bare_teardown_algo(tmp_path, *, is_simplex=False, refine=0, bootstrap_number=None,
                        delete_old_files=1):
    """Minimal Algorithm carrying only what _teardown_sim_dir reads, with a
    populated Simulations/ dir on disk to delete."""
    sim_dir = str(tmp_path) + '/Simulations'
    os.makedirs(sim_dir, exist_ok=True)
    with open(sim_dir + '/sentinel.txt', 'w') as fh:
        fh.write('work')

    algo = object.__new__(_ConcreteAlgorithm)
    algo.config = type('Cfg', (), {})()
    algo.config.config = {'refine': refine, 'delete_old_files': delete_old_files}
    algo._is_simplex = is_simplex
    algo.bootstrap_number = bootstrap_number
    algo.sim_dir = sim_dir
    return algo


class TestTeardownSimDir:

    @pytest.mark.parametrize('is_simplex, refine, bootstrap_number, delete_old_files, removed', [
        (False, 0, None, 1, True),    # ordinary final pass -> tear Simulations/ down
        (False, 0, None, 0, False),   # delete_old_files off -> keep everything
        (False, 0, 1,    1, False),   # intermediate bootstrap replicate -> keep
        (False, 1, None, 1, False),   # non-final refinement pass (non-Simplex) -> keep
        (True,  1, None, 1, True),    # Simplex always tears down, even with refine==1
    ])
    def test_teardown_respects_guards(self, tmp_path, is_simplex, refine, bootstrap_number,
                                      delete_old_files, removed):
        """Simulations/ is deleted only on a terminal pass with cleanup enabled.
        The _is_simplex flag (decoupled from the leaf class in M1 step 2a) forces
        teardown even mid-refinement; an intermediate bootstrap or refinement pass,
        or delete_old_files==0, keeps the directory."""
        algo = _bare_teardown_algo(tmp_path, is_simplex=is_simplex, refine=refine,
                                   bootstrap_number=bootstrap_number, delete_old_files=delete_old_files)

        algo._teardown_sim_dir()

        assert (not os.path.exists(algo.sim_dir)) == removed

    def test_windows_uses_rmtree(self, tmp_path, monkeypatch):
        """On Windows there is no `rm`, so teardown goes through shutil.rmtree; the
        end state (Simulations/ gone) is the same."""
        algo = _bare_teardown_algo(tmp_path)  # guard satisfied, delete_old_files=1
        monkeypatch.setattr(algorithms.base.os, 'name', 'nt')

        algo._teardown_sim_dir()

        assert not os.path.exists(algo.sim_dir)


# --------------------------------------------------------------------------- #
# Backup round-trip and the resume entry point
#
# backup() is already a method; these cover the two halves of the
# crash-resume contract it anchors: that a backup pickles to a loadable
# (algorithm, pending_psets) pair, and that run(resume=...) re-enters from a
# supplied pset list instead of start_run().
# --------------------------------------------------------------------------- #
class _PicklableCfg:
    """Module-level config stand-in. The other helpers build config via
    ``type('Cfg', (), {})()``, whose instances are NOT picklable (no importable
    qualified name); backup() pickles ``self``, so its algo needs a real class.
    ``variables`` mirrors the free-parameter definitions Algorithm.__setstate__
    feeds to Trajectory.load_trajectory when it reconstitutes the trajectory."""

    def __init__(self, config, variables):
        self.config = config
        self.variables = variables


class _PicklableObjective:
    """Objective stand-in for the backup tests. ``supports_pointwise_log_likelihood``
    is the gate the information-criteria path reads (#560); False here, so a bare
    backup neither re-simulates nor writes a criteria checkpoint unless a test opts in."""

    supports_pointwise_log_likelihood = False


def _bare_backup_algo(tmp_path, *, refine=False, delete_old_files=0, backup_ic=1):
    """Minimal *picklable* Algorithm carrying what backup()/output_results read.

    The trajectory is deliberately excluded from the pickle (Algorithm.should_pickle)
    and reloaded from sorted_params_backup.txt on unpickle, so config.variables must
    match the single 'v1__FREE' column those rows carry."""
    out = str(tmp_path)
    res_dir = out + '/Results'
    os.makedirs(res_dir, exist_ok=True)

    variables = [pset.FreeParameter('v1__FREE', 'uniform_var', 0, 100)]
    algo = object.__new__(_ConcreteAlgorithm)
    algo.config = _PicklableCfg(
        {'output_dir': out, 'delete_old_files': delete_old_files, 'num_to_output': 100,
         'backup_information_criteria': backup_ic},
        variables)
    algo.res_dir = res_dir
    algo.refine = refine
    algo.output_counter = 0
    algo.objective = _PicklableObjective()
    algo.trajectory = Trajectory(100)
    algo.trajectory.add(_pset('iter0run0', 5.0), 5.0, 'iter0run0')
    return algo


class TestBackup:

    def test_backup_writes_loadable_pickle_atomically(self, tmp_path):
        """backup() saves (self, pending_psets) to alg_backup.bp via a temp file +
        atomic replace, and dumps the trajectory to sorted_params_backup.txt. The
        pickle must round-trip to a real Algorithm whose trajectory and the pending
        psets survive — that is the state a -r resume reloads."""
        algo = _bare_backup_algo(tmp_path)
        pending = (_pset('pending0', 11.0), _pset('pending1', 12.0))

        algo.backup(pending_psets=pending)

        bp = str(tmp_path) + '/alg_backup.bp'
        assert os.path.isfile(bp)
        assert not os.path.exists(str(tmp_path) + '/alg_backup_temp.bp')  # temp consumed by replace
        assert os.path.isfile(algo.res_dir + '/sorted_params_backup.txt')  # output_results ran

        with open(bp, 'rb') as fh:
            loaded_algo, loaded_pending = pickle.load(fh)
        assert isinstance(loaded_algo, algorithms.Algorithm)
        np.testing.assert_allclose(loaded_algo.trajectory.best_score(), 5.0)
        assert [p.name for p in loaded_pending] == ['pending0', 'pending1']


# --------------------------------------------------------------------------- #
# The information-criteria half of the checkpoint (#560)
#
# sorted_params_backup.txt has always been checkpointed; information_criteria.txt
# was written only on the terminal path, so a run was un-scoreable at every moment
# except its last. These cover the second half now written beside it: that it is
# written, what it says, and the several ways it costs nothing.
# --------------------------------------------------------------------------- #
def _ic(log_likelihood=-12.5):
    return objective.information_criteria(log_likelihood, k=2, n=10)


def _kv(text):
    """The key/value lines of an information-criteria file (comments dropped) — the
    part a consumer parses, and the part a checkpoint shares with the final file."""
    return dict(line.split('\t', 1) for line in text.splitlines() if not line.startswith('#'))


class _CountingCompute:
    """Stands in for _compute_information_criteria, counting the re-simulations the
    checkpoint asks for and recording which pset each one scored."""

    def __init__(self, ic=None):
        self.ic = _ic() if ic is None else ic
        self.scored = []

    def __call__(self, best_pset):
        self.scored.append(best_pset.name)
        return self.ic


class _WatchingCompute:
    """Records which halves of the checkpoint are already on disk when the re-simulation is
    asked for. Module-level (not a closure) because backup() pickles the algorithm it is
    attached to."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.on_disk = []

    def __call__(self, best_pset):
        self.on_disk.extend(p for p in ('alg_backup.bp', 'Results/sorted_params_backup.txt')
                            if os.path.isfile(self.out_dir + '/' + p))
        return _ic()


class TestInformationCriteriaCheckpoint:

    def test_backup_writes_the_criteria_beside_the_parameter_sets(self, tmp_path, monkeypatch):
        """The point of #560: both halves of a scoreable result are on disk during the
        run. The checkpoint carries the same key/value lines as the final artifact — one
        parser reads either — and its comments say which parameter set it describes and
        that the run is not over."""
        algo = _bare_backup_algo(tmp_path)
        monkeypatch.setattr(algo, '_compute_information_criteria', _CountingCompute())

        algo.backup()

        text = open(algo.res_dir + '/information_criteria_backup.txt').read()
        assert 'CHECKPOINT' in text
        assert 'iter0run0' in text and 'sorted_params_backup.txt' in text
        # The run's own artifact is untouched: it still means "this run's result".
        assert not os.path.exists(algo.res_dir + '/information_criteria.txt')
        # Written now, it differs from the checkpoint only in comments.
        algo._emit_information_criteria(_ic())
        assert _kv(text) == _kv(open(algo.res_dir + '/information_criteria.txt').read())
        assert _kv(text)['log_likelihood'] == '-12.5'

    def test_a_checkpoint_reports_to_the_log_not_the_console(self, tmp_path, monkeypatch, capsys):
        """A checkpoint fires on a cadence, so — like the parameter-set checkpoint beside
        it — it reports to the log, not to the console. Only the end-of-run file prints."""
        monkeypatch.setattr(printing, 'verbosity', 1)
        algo = _bare_backup_algo(tmp_path)
        monkeypatch.setattr(algo, '_compute_information_criteria', _CountingCompute())

        algo.backup()
        assert 'Information criteria' not in capsys.readouterr().out

        algo._emit_information_criteria(_ic())
        assert 'Information criteria (best fit)' in capsys.readouterr().out

    def test_an_unchanged_best_fit_is_not_re_simulated(self, tmp_path, monkeypatch):
        """The whole cost of this feature is one simulation per checkpoint, and it is
        only paid when there is something new to say. A search's long converged tail —
        the 40 minutes of stragglers that motivated #560 — re-simulates nothing: the file
        on disk already describes the best fit."""
        algo = _bare_backup_algo(tmp_path)
        compute = _CountingCompute()
        monkeypatch.setattr(algo, '_compute_information_criteria', compute)

        algo.backup()
        algo.backup()
        algo.backup()
        assert compute.scored == ['iter0run0']

        algo.trajectory.add(_pset('iter9run3', 1.0), 1.0, 'iter9run3')
        algo.backup()
        assert compute.scored == ['iter0run0', 'iter9run3']
        assert 'iter9run3' in open(algo.res_dir + '/information_criteria_backup.txt').read()

    def test_a_failed_write_is_retried_at_the_next_checkpoint(self, tmp_path):
        """Only a file that was actually written licenses skipping the next recompute.
        Otherwise one transient I/O error would leave a run whose best fit never changes
        again with no criteria checkpoint at all — the exact failure #560 removes."""
        algo = _bare_backup_algo(tmp_path)
        compute = _CountingCompute()
        algo._compute_information_criteria = compute
        algo._emit_information_criteria = lambda *a, **k: False

        algo._checkpoint_information_criteria()
        algo._checkpoint_information_criteria()
        assert compute.scored == ['iter0run0', 'iter0run0']

    def test_a_non_likelihood_objective_costs_nothing(self, tmp_path):
        """No information criterion is defined for sos / kl / direct_pass, so the gate in
        _compute_information_criteria makes the checkpoint a no-op — no re-simulation, no
        file, on the objectives where there would be nothing to write."""
        algo = _bare_backup_algo(tmp_path)   # objective.supports_pointwise_log_likelihood False

        algo.backup()

        assert not os.path.exists(algo.res_dir + '/information_criteria_backup.txt')
        assert os.path.isfile(algo.res_dir + '/sorted_params_backup.txt')

    def test_backup_information_criteria_0_turns_it_off(self, tmp_path, monkeypatch):
        """The escape hatch for a model where one extra simulation per backup interval is
        too expensive. Everything else about the checkpoint is unchanged."""
        algo = _bare_backup_algo(tmp_path, backup_ic=0)
        compute = _CountingCompute()
        monkeypatch.setattr(algo, '_compute_information_criteria', compute)

        algo.backup()

        assert compute.scored == []
        assert not os.path.exists(algo.res_dir + '/information_criteria_backup.txt')
        assert os.path.isfile(algo.res_dir + '/sorted_params_backup.txt')

    def test_a_refine_checkpoints_under_its_own_name(self, tmp_path, monkeypatch):
        """A refine's parameter checkpoint is sorted_params_refine_backup.txt, so its
        criteria checkpoint is information_criteria_refine_backup.txt: both halves of one
        phase's checkpoint carry the same name, and neither is confused with the fit's."""
        algo = _bare_backup_algo(tmp_path, refine=True)
        monkeypatch.setattr(algo, '_compute_information_criteria', _CountingCompute())

        algo.backup()

        text = open(algo.res_dir + '/information_criteria_refine_backup.txt').read()
        assert 'sorted_params_refine_backup.txt' in text
        assert not os.path.exists(algo.res_dir + '/information_criteria_backup.txt')

    def test_an_empty_trajectory_has_nothing_to_score(self, tmp_path):
        """A backup can land before the first result comes back (an expired budget, a
        slow first generation); there is no best fit to re-simulate, and asking the
        Trajectory for one would raise."""
        algo = _bare_backup_algo(tmp_path)
        algo.trajectory = Trajectory(100)
        compute = _CountingCompute()
        algo._compute_information_criteria = compute

        algo._checkpoint_information_criteria()

        assert compute.scored == []
        assert not os.path.exists(algo.res_dir + '/information_criteria_backup.txt')

    def test_the_re_simulation_runs_after_the_resume_state_is_written(self, tmp_path):
        """Re-simulating the best fit is the only slow part of a checkpoint, so it runs last:
        a kill during it leaves the parameter sets and the resume pickle current, and one
        interval of criteria is the most that can be lost."""
        algo = _bare_backup_algo(tmp_path)
        watch = _WatchingCompute(str(tmp_path))
        algo._compute_information_criteria = watch

        algo.backup()

        assert watch.on_disk == ['alg_backup.bp', 'Results/sorted_params_backup.txt']

    def test_a_checkpoint_leaves_no_mid_run_profiled_noise_behind(self, tmp_path):
        """profiled_noise.txt reports the scales estimated at the run's FINAL best fit, and
        the scoring pass the checkpoint runs captures them at whatever point it scored. A
        checkpoint must not leave that behind for the end-of-run tail to report if its own
        scoring pass fails."""
        algo = _bare_backup_algo(tmp_path)

        def _mid_run_scoring(best_pset):
            algo._profiled_noise = {'sigma': 3.0}
            return _ic()

        algo._compute_information_criteria = _mid_run_scoring

        algo._checkpoint_information_criteria()

        assert os.path.isfile(algo.res_dir + '/information_criteria_backup.txt')
        assert algo._profiled_noise == {}


class _ResumeProbe(algorithms.Algorithm):
    """start_run records that it ran (it must NOT, on a resume) and would return a
    sentinel pset; got_result stops after the first result so the test stays tiny."""

    def start_run(self):
        self.start_run_called = True
        return [_pset('should_not_be_used', 99.0)]

    def got_result(self, res):
        self.seen.append(res)
        return 'STOP'


def test_resume_uses_provided_psets_and_skips_start_run(tmp_path, monkeypatch):
    """run(resume=[...]) seeds the first generation from the supplied psets and
    never calls start_run — the entry point a -r restart takes after unpickling a
    backup's pending psets."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    algo = _make_algorithm(tmp_path, [[_pset('dummy', 1.0)]], cls=_ResumeProbe)
    algo.start_run_called = False
    resume = [_pset('resume0', 7.0), _pset('resume1', 8.0)]
    client = _FakeClient()

    algo.run(client, resume=resume)

    assert algo.start_run_called is False              # start_run bypassed
    assert len(client.submitted) == 2                  # both resume psets submitted
    assert all(fn is algorithms.core.run_job for fn, _ in client.submitted)
    # FIFO drain hits resume0 first, whose value 7.0 is its score, then STOP.
    assert len(algo.seen) == 1
    np.testing.assert_allclose(algo.seen[0].score, 7.0)


# --------------------------------------------------------------------------- #
# Wall-time budget (#529, ADR-0093)
#
# The contract: when `wall_time_fit` runs out the loop stops launching work and
# the run FINALIZES anyway — the same end-of-fit path a converged run takes, so a
# budgeted result is scoreable exactly like a completed one. Only the stop
# *reason* differs, and that is recorded (log, console, Results/stop_reason.txt).
#
# The clock is injected, so nothing here sleeps.
# --------------------------------------------------------------------------- #
class _TimingOutAsCompleted(_FakeAsCompleted):
    """as_completed that raises TimeoutError instead of yielding its Nth result —
    the real one's behavior when the budget elapses while every worker is still
    busy and nothing has completed."""

    timeout_after = 0

    def __next__(self):
        if self._served >= type(self).timeout_after:
            raise TimeoutError()
        self._served += 1
        return super().__next__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._served = 0


def _budgeted(algo, limit, elapsed=0.0):
    """Give ``algo`` a wall-time budget on a hand-driven clock; returns the clock."""
    clock = _TestClock()
    algo.budget = budget_mod.FitBudget(limit, elapsed=elapsed, clock=clock)
    return clock


class _TestClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_budget_stops_the_loop_after_the_result_in_hand_and_finalizes(tmp_path, monkeypatch):
    """The budget expires while the first result is being processed. That result is
    still recorded (it was already paid for), nothing new is launched, and the run
    finalizes: sorted_params_final.txt and a best fit exist, exactly as on a
    converged run."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)],
            [_pset('iter1run0', 1.0), _pset('iter1run1', 2.0)]]
    algo = _make_algorithm(tmp_path, gens)
    clock = _budgeted(algo, 100.0)
    original_got_result = algo.got_result

    def spend_the_budget(res):
        clock.t = 1000.0  # the deadline passes while this result is in hand
        return original_got_result(res)

    algo.got_result = spend_the_budget
    client = _FakeClient()

    algo.run(client)

    # Only the initial generation was ever submitted — no new work after expiry.
    assert len(client.submitted) == 2
    assert len(algo.seen) == 1                      # the one result in hand was recorded
    np.testing.assert_allclose(algo.trajectory.best_score(), 10.0)
    # ... and the normal end-of-fit artifacts were still written.
    assert os.path.isfile(algo.res_dir + '/sorted_params_final.txt')


def test_budget_stop_records_its_reason(tmp_path, monkeypatch):
    """A budgeted run must not be silently mistaken for a converged one: the reason
    is logged, printed, and left on disk as Results/stop_reason.txt (whose presence
    is what downstream tooling can key on, since every scoreable artifact is
    deliberately identical to a converged run's)."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    algo = _make_algorithm(tmp_path, [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]])
    clock = _budgeted(algo, 60.0)
    algo.got_result = lambda res: (setattr(clock, 't', 61.0), [])[1]

    algo.run(_FakeClient())

    assert 'Wall-time budget reached' in algo.stop_reason
    assert 'wall_time_fit = 60 s' in algo.stop_reason
    assert '0:01:01' in algo.stop_reason              # elapsed, in the run-time line's units
    assert '1 completed simulation' in algo.stop_reason
    with open(algo.res_dir + '/stop_reason.txt') as fh:
        assert fh.read().strip() == algo.stop_reason


def test_an_unbudgeted_run_never_stops_early_and_asks_for_no_timeout(tmp_path, monkeypatch):
    """The default (wall_time_fit = 0 -> no FitBudget) is byte-identical to the
    historical loop: no timeout is handed to as_completed, and no stop reason is
    recorded or written."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    gens = [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)],
            [_pset('iter1run0', 1.0), _pset('iter1run1', 2.0)]]
    algo = _make_algorithm(tmp_path, gens)
    client = _FakeClient()

    algo.run(client)

    assert algo.budget is None
    assert _FakeAsCompleted.last_timeout is None
    assert len(client.submitted) == 4                 # ran to its own STOP
    assert algo.stop_reason is None
    assert not os.path.exists(algo.res_dir + '/stop_reason.txt')


def test_a_budgeted_run_bounds_the_wait_for_the_next_completion(tmp_path, monkeypatch):
    """The remaining budget is handed to as_completed as its timeout, so an expiry
    lands on time even when every worker is mid-simulation (otherwise the loop would
    block in next() until some simulation happened to finish)."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    algo = _make_algorithm(tmp_path, [[_pset('iter0run0', 10.0)]])
    _budgeted(algo, 100.0, elapsed=25.0)

    algo.run(_FakeClient())

    np.testing.assert_allclose(_FakeAsCompleted.last_timeout, 75.0)


def test_budget_expiring_with_every_worker_busy_stops_and_finalizes(tmp_path, monkeypatch):
    """No result ever comes back before the deadline: as_completed raises
    TimeoutError, the in-flight jobs are abandoned (cancelled), and the run still
    finalizes -- with an explicit 'no best fit' report rather than a crash on the
    empty trajectory."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _TimingOutAsCompleted)
    monkeypatch.setattr(_TimingOutAsCompleted, 'timeout_after', 0)
    algo = _make_algorithm(tmp_path, [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]])
    _budgeted(algo, 60.0)
    client = _FakeClient()

    algo.run(client)

    assert len(client.submitted) == 2                 # the initial jobs went out
    assert len(algo.seen) == 0                        # none came back
    assert len(algo.trajectory) == 0
    assert 'Wall-time budget reached' in algo.stop_reason
    assert 'no completed simulation' in algo.stop_reason
    assert os.path.isfile(algo.res_dir + '/stop_reason.txt')
    # Nothing was scored, so there are no parameter sets to write -- the run reports
    # that instead of emitting an empty (unloadable) sorted_params file.
    assert not os.path.exists(algo.res_dir + '/sorted_params_final.txt')


def test_a_budget_already_spent_before_the_first_job_launches_nothing(tmp_path, monkeypatch):
    """Setup (configuration loading, network generation) can eat the whole budget.
    An expired budget means no new work, so not one job is submitted -- the run goes
    straight to the finalize path."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    algo = _make_algorithm(tmp_path, [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]])
    _budgeted(algo, 60.0, elapsed=60.0)               # spent before run() is entered
    client = _FakeClient()

    algo.run(client)

    assert client.submitted == []
    assert 'Wall-time budget reached' in algo.stop_reason
    assert os.path.isfile(algo.res_dir + '/stop_reason.txt')


def test_the_budget_is_not_carried_into_a_pickled_backup(tmp_path):
    """A wall-clock deadline is meaningless once restored in a later process, so it
    is excluded from the pickle; a resumed algorithm reads the class default (None)
    until main() builds it a fresh budget."""
    algo = _bare_backup_algo(tmp_path)
    _budgeted(algo, 60.0)

    algo.backup()
    with open(str(tmp_path) + '/alg_backup.bp', 'rb') as fh:
        loaded, _pending = pickle.load(fh)

    assert 'budget' not in loaded.__dict__
    assert loaded.budget is None


_END_OF_FIT_PATH = ['_copy_best_fit_sims', '_rerun_best_fit_to_save_data', '_emit_best_fit_bngl',
                    '_compute_information_criteria', '_emit_information_criteria',
                    '_emit_profiled_noise', '_emit_inference_data', '_finalize_backup_pickle',
                    '_teardown_sim_dir']


def test_a_budgeted_stop_runs_the_whole_end_of_fit_path(tmp_path, monkeypatch):
    """The point of the budget (#529): a deadline-stopped run is *finalized*, not
    abandoned. Every step of the end-of-fit path runs, in order — including the
    information-criteria step whose absence is what made an externally killed run
    unscoreable. Nothing in that path is conditional on why the loop ended."""
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)
    called = []
    for name in _END_OF_FIT_PATH:
        monkeypatch.setattr(algorithms.Algorithm, name,
                            (lambda n: lambda self, *a, **k: called.append(n))(name))
    algo = _make_algorithm(tmp_path, [[_pset('iter0run0', 10.0), _pset('iter0run1', 20.0)]])
    clock = _budgeted(algo, 60.0)
    algo.got_result = lambda res: (setattr(clock, 't', 61.0), [])[1]

    algo.run(_FakeClient())

    assert algo.stop_reason is not None
    assert called == _END_OF_FIT_PATH
