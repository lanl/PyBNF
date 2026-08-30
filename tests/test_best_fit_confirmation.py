"""The end-of-fit stage that confirms the best fit of a stochastic fit (#659).

A stochastic model gives a different objective value every time it is run, so a fit that
picks its answer by taking the best value it ever saw usually picks the parameter set that
got a lucky simulation. The fix runs the top parameter sets again several times each and
ranks them by their average.

Three layers are covered here, in this order.

  * ``pybnf.algorithms.best_fit_confirmation``: the arithmetic and the report text. It
    holds no state and touches no files, so it is tested on its own.
  * ``pybnf.pset.Trajectory``: ``top_fits`` picks the candidates, and ``pin_best`` is how
    the winner becomes the run's answer everywhere at once.
  * ``Algorithm._confirm_best_fit``: the stage itself, driven by the same synchronous fake
    dask client ``test_run_loop`` uses. The fake model here is deliberately noisy in a
    controlled way -- its objective value depends on the replicate index -- so a test can
    set up the exact situation the issue describes: a mediocre parameter set that got one
    lucky simulation sitting on top of a genuinely better one.
"""
import math
import os

import numpy as np
import pytest

from .context import algorithms, config, printing, pset
import pybnf.algorithms.base as base
from pybnf.algorithms import best_fit_confirmation as bfc
from pybnf.pset import Trajectory, PSet, FreeParameter


# --------------------------------------------------------------------------- #
# The report module: arithmetic and text
# --------------------------------------------------------------------------- #
def _candidate(name, search_objective, scores, failures=0):
    return bfc.Candidate(name=name, pset=None, search_objective=search_objective,
                         scores=tuple(scores), failures=failures)


def test_mean_is_the_average_of_the_replicate_values():
    c = _candidate('a', 1.0, [2.0, 4.0, 6.0])
    assert bfc.mean_objective(c) == 4.0
    np.testing.assert_allclose(bfc.standard_deviation(c), 2.0)
    np.testing.assert_allclose(bfc.standard_error(c), 2.0 / math.sqrt(3))


def test_a_candidate_with_no_usable_value_sorts_last_and_never_wins():
    good = _candidate('good', 1.0, [3.0])
    dead = _candidate('dead', 0.1, [], failures=10)
    assert bfc.mean_objective(dead) == math.inf
    assert bfc.standard_deviation(dead) is None
    assert bfc.standard_error(dead) is None
    assert bfc.ranked([dead, good]) == [1, 0]
    assert bfc.winner([dead, good]).name == 'good'


def test_no_winner_when_nothing_could_be_run_again():
    assert bfc.winner([_candidate('a', 1.0, []), _candidate('b', 2.0, [])]) is None
    assert bfc.winner([]) is None


def test_ties_keep_the_order_the_search_ranked_them_in():
    a = _candidate('a', 1.0, [5.0])
    b = _candidate('b', 2.0, [5.0])
    assert bfc.ranked([a, b]) == [0, 1]
    assert bfc.ranked([b, a]) == [0, 1]


def test_summary_reports_the_optimism_and_the_search_rank_of_the_winner():
    """The two facts the issue is about: the reported objective was too good, and the
    search would have reported a different parameter set."""
    lucky = _candidate('lucky', 1.0, [10.0, 10.0])     # looked best, is not
    real = _candidate('real', 3.0, [4.0, 4.0])         # looked worse, is best
    text = '\n'.join(bfc.summary_lines([lucky, real], replicates=2))

    assert 'winner\treal' in text
    assert 'winner_mean_objective\t4' in text
    # The search said 3.0 for the winner but running it again says 4.0.
    assert 'optimism\t1' in text
    # The winner sat second in the search's own ranking.
    assert 'search_rank\t2' in text
    rows = [l for l in text.splitlines() if l and not l.startswith('#') and l[0].isdigit()]
    assert rows[0].startswith('1\treal\t4')
    assert rows[1].startswith('2\tlucky\t10')


def test_summary_has_no_search_rank_when_the_search_already_had_it_right():
    text = '\n'.join(bfc.summary_lines(
        [_candidate('a', 1.0, [2.0]), _candidate('b', 3.0, [5.0])], replicates=1))
    assert 'search_rank' not in text
    assert 'winner\ta' in text


def test_summary_says_so_when_no_candidate_could_be_run_again():
    text = '\n'.join(bfc.summary_lines([_candidate('a', 1.0, [], failures=3)], replicates=3))
    assert 'winner\tnone' in text
    assert 'the best fit is the one the search picked' in text


def test_console_lines_name_the_file_and_flag_a_changed_answer():
    lines = bfc.console_lines([_candidate('lucky', 1.0, [10.0, 10.0]),
                               _candidate('real', 3.0, [4.0, 4.0])],
                              replicates=2, path='/tmp/x.txt')
    text = '\n'.join(lines)
    assert '/tmp/x.txt' in text
    assert 'not the parameter set the search would have reported' in text


# --------------------------------------------------------------------------- #
# Trajectory: choosing the candidates and recording the winner
# --------------------------------------------------------------------------- #
def _ps(value, name=None):
    p = PSet([FreeParameter('v1__FREE', 'uniform_var', 0, 100, value)])
    p.name = name
    return p


def test_top_fits_returns_the_best_entries_best_first():
    t = Trajectory(100)
    for name, value, obj in [('a', 1.0, 9.0), ('b', 2.0, 3.0), ('c', 3.0, 5.0)]:
        t.add(_ps(value), obj, name)
    assert [(o, n) for o, n, _ in t.top_fits(2)] == [(3.0, 'b'), (5.0, 'c')]
    assert len(t.top_fits(10)) == 3
    assert t.top_fits(0) == []


def test_top_fits_counts_each_set_of_parameter_values_once():
    """A search that re-evaluates its population records the same parameter values many
    times. Asking for three candidates should get three different ones, not three records
    of the same one."""
    t = Trajectory(100)
    t.add(_ps(1.0), 3.0, 'a')
    t.add(_ps(1.0), 3.5, 'a2')     # same values, scored again
    t.add(_ps(2.0), 4.0, 'b')
    assert [n for _, n, _ in t.top_fits(3)] == ['a', 'b']
    assert [n for _, n, _ in t.top_fits(3, distinct=False)] == ['a', 'a2', 'b']


def test_pinning_a_best_fit_overrides_the_recorded_values():
    t = Trajectory(100)
    t.add(_ps(1.0), 1.0, 'lucky')
    t.add(_ps(2.0), 2.0, 'real')
    assert t.best_fit_name() == 'lucky'

    t.pin_best(_ps(2.0), 4.0, 'real')
    assert t.best_fit_name() == 'real'
    assert t.best_score() == 4.0
    assert t.best_fit()['v1__FREE'] == 2.0


def test_a_new_evaluation_drops_the_pin():
    """A refine shares the fit's trajectory, so a pin left over from the fit would make the
    run report the point the refine started from instead of the one it reached."""
    t = Trajectory(100)
    t.add(_ps(1.0), 1.0, 'lucky')
    t.pin_best(_ps(1.0), 4.0, 'lucky')
    t.add(_ps(3.0), 0.5, 'refined')
    assert t.pinned_best() is None
    assert t.best_fit_name() == 'refined'


def test_clearing_the_pin_restores_the_recorded_best():
    t = Trajectory(100)
    t.add(_ps(1.0), 1.0, 'a')
    t.pin_best(_ps(1.0), 9.0, 'a')
    t.clear_pinned_best()
    assert t.best_score() == 1.0


# --------------------------------------------------------------------------- #
# Fakes: a noisy model, a noise-reading objective, a synchronous dask client
# --------------------------------------------------------------------------- #
#: Objective value of one run, as (parameter value, replicate index) -> value. Anything
#: not listed scores as its parameter value, so a test only spells out the runs it cares
#: about. Replicate index 0 is what the search itself used.
NOISE = {}


class _NoisyModelCopy:
    """Reports back the replicate index it was run with, which is the only thing that
    changes from one run of the same parameter set to the next."""

    def __init__(self, name):
        self.name = name
        self._pybnf_replicate_index = 0

    def execute(self, folder, file_prefix, timeout):
        from pybnf import data as data_mod
        d = data_mod.Data()
        d.cols = {'time': 0, 'v1_result': 1}
        d.data = np.array([[0.0, float(self._pybnf_replicate_index)]], dtype=float)
        return {'time_course': d}

    def save_all(self, prefix):
        pass


class _NoisyModel:
    suffixes = []

    def __init__(self, name='m', stochastic=True, seeded=False):
        self.name = name
        self.stochastic = stochastic
        self.seeded = seeded

    def copy_with_param_set(self, params):
        return _NoisyModelCopy(self.name)

    def save_all(self, prefix):
        pass


class _NoisyObjective:
    """Scores a run from its parameter value and the replicate index the model reported,
    through the NOISE table."""

    def evaluate_multiple(self, simdata, exp_data, ps, constraints, show_warnings=True):
        value = float(ps['v1__FREE'])
        replicate = int(simdata['m']['time_course'].data[0][1])
        return float(NOISE.get((value, replicate), value))


class _FakeFuture:
    def __init__(self, result):
        self._result = result
        self.status = 'finished'

    def result(self):
        return self._result


class _FakeClient:
    """Runs each submitted callable inline and returns a finished future."""

    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append(args[0])   # the Job
        return _FakeFuture(fn(*args))


class _FakeAsCompleted:
    def __init__(self, futures, with_results=False, raise_errors=True, timeout=None):
        assert with_results and not raise_errors
        self._queue = list(futures)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            raise StopIteration
        f = self._queue.pop(0)
        return f, f.result()


class _ConcreteAlgorithm(algorithms.Algorithm):
    def start_run(self):
        return []

    def got_result(self, res):
        return []


def _algo(tmp_path, *, models=None, edition=2, candidates=None, replicates=None,
          smoothing=1, stochastic_seed='auto', parallelize_models=1):
    out = str(tmp_path)
    sim_dir = out + '/Simulations'
    res_dir = out + '/Results'
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    if models is None:
        models = {'m': _NoisyModel()}

    algo = object.__new__(_ConcreteAlgorithm)
    algo.config = type('Cfg', (), {})()
    algo.config.config = {
        'smoothing': smoothing, 'parallelize_models': parallelize_models,
        'local_objective_eval': 1, 'wall_time_sim': None, 'normalization': None,
        'stochastic_seed': stochastic_seed, 'delete_old_files': 0,
        'edition': edition, 'best_fit_candidates': candidates,
        'best_fit_replicates': replicates, 'num_to_output': 100,
    }
    algo.config.postprocessing = {}
    algo.config.constraints = []
    algo.config.models = models

    algo.objective = _NoisyObjective()
    algo.exp_data = {}
    algo.trajectory = Trajectory(100)
    algo.job_id_counter = 0
    algo.job_group_dir = {}
    algo.model_list = list(models.values())
    algo.sim_dir = sim_dir
    algo.res_dir = res_dir
    algo.failed_logs_dir = out + '/FailedSimLogs'
    os.makedirs(algo.failed_logs_dir, exist_ok=True)
    algo.calc_future = None
    algo.models_future = None
    algo.refine = False
    return algo


@pytest.fixture(autouse=True)
def _clean_noise():
    NOISE.clear()
    yield
    NOISE.clear()


@pytest.fixture
def _sync_dask(monkeypatch):
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)


# --------------------------------------------------------------------------- #
# Settings and gating
# --------------------------------------------------------------------------- #
def test_the_stage_is_on_by_default_under_a_modern_edition(tmp_path):
    assert _algo(tmp_path, edition=2)._best_fit_confirmation_settings() == (10, 10)


def test_the_stage_is_off_by_default_under_the_legacy_edition(tmp_path):
    """A conf that names no edition keeps behaving exactly as it always has."""
    assert _algo(tmp_path, edition=None)._best_fit_confirmation_settings() == (0, 0)


def test_an_explicit_setting_wins_under_either_edition(tmp_path):
    assert _algo(tmp_path, edition=None, candidates=4,
                 replicates=6)._best_fit_confirmation_settings() == (4, 6)
    assert _algo(tmp_path, edition=2, candidates=0,
                 replicates=0)._best_fit_confirmation_settings() == (0, 0)


def test_replicates_differ_only_when_a_model_is_stochastic(tmp_path):
    assert _algo(tmp_path)._replicates_would_differ() is True
    deterministic = {'m': _NoisyModel(stochastic=False)}
    assert _algo(tmp_path, models=deterministic)._replicates_would_differ() is False


def test_replicates_do_not_differ_when_every_stochastic_model_pins_its_seed(tmp_path):
    """Under an _honorbngl policy an explicit seed in the model is honored, so every run
    reproduces one trajectory and there is nothing to average."""
    pinned = {'m': _NoisyModel(seeded=True)}
    algo = _algo(tmp_path, models=pinned, stochastic_seed='auto_honorbngl')
    assert algo._replicates_would_differ() is False
    # One unpinned stochastic model is enough for replicates to mean something.
    mixed = {'m': _NoisyModel(seeded=True), 'n': _NoisyModel(name='n', seeded=False)}
    assert _algo(tmp_path, models=mixed,
                 stochastic_seed='auto_honorbngl')._replicates_would_differ() is True


@pytest.mark.parametrize('kwargs', [
    dict(models={'m': _NoisyModel(stochastic=False)}),  # nothing stochastic
    dict(edition=None),                                 # legacy: off by default
    dict(replicates=1),                                 # too few replicates to average
    dict(candidates=0),                                 # no candidates asked for
])
def test_the_stage_does_nothing_when_it_does_not_apply(tmp_path, _sync_dask, kwargs):
    algo = _algo(tmp_path, **kwargs)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')
    client = _FakeClient()
    algo._confirm_best_fit(client)
    assert client.submitted == []
    assert algo.trajectory.pinned_best() is None
    assert not os.path.exists(algo.res_dir + '/best_fit_confirmation.txt')


def test_the_stage_does_nothing_once_the_wall_time_budget_is_spent(
        tmp_path, _sync_dask, monkeypatch):
    """A budget is a promise about the whole run, and this stage costs a hundred more
    simulations. A run that is already out of time says what it is skipping instead."""
    printed = []
    monkeypatch.setattr(base, 'print1', lambda msg, *a, **k: printed.append(msg))
    algo = _algo(tmp_path, candidates=2, replicates=2)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')
    algo.budget = type('B', (), {'expired': lambda self: True})()
    client = _FakeClient()

    algo._confirm_best_fit(client)

    assert client.submitted == []
    assert algo.trajectory.pinned_best() is None
    assert any('wall-time budget is spent' in m for m in printed)


def test_the_stage_does_nothing_without_a_client(tmp_path):
    algo = _algo(tmp_path, candidates=2, replicates=2)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')
    algo._confirm_best_fit(None)
    assert algo.trajectory.pinned_best() is None


# --------------------------------------------------------------------------- #
# The stage itself
# --------------------------------------------------------------------------- #
def test_a_lucky_parameter_set_loses_to_a_better_one_when_both_are_run_again(
        tmp_path, _sync_dask, monkeypatch):
    """The situation the issue describes. Parameter set 1.0 is genuinely worse than 2.0,
    but the one simulation the search gave it came out at 0.1, so the search ranked it
    first. Running both again three times each puts 2.0 back on top, and the run reports
    2.0 with its honest average rather than 1.0 with 0.1."""
    printed = []
    monkeypatch.setattr(base, 'print1', lambda msg, *a, **k: printed.append(msg))

    for replicate in (1, 2, 3):
        NOISE[(1.0, replicate)] = 5.0     # the lucky set is really a 5
        NOISE[(2.0, replicate)] = 2.0     # the better set is really a 2

    algo = _algo(tmp_path, candidates=2, replicates=3)
    algo.trajectory.add(_ps(1.0), 0.1, 'lucky')
    algo.trajectory.add(_ps(2.0), 2.0, 'real')
    client = _FakeClient()

    algo._confirm_best_fit(client)

    # Two candidates, three replicates each.
    assert len(client.submitted) == 6
    # The run's answer is now the genuinely better parameter set, at its honest value.
    assert algo.trajectory.best_fit()['v1__FREE'] == 2.0
    assert algo.trajectory.best_score() == 2.0
    assert algo.trajectory.best_fit_name() == 'real'

    text = open(algo.res_dir + '/best_fit_confirmation.txt').read()
    assert 'winner\treal' in text
    assert 'search_rank\t2' in text
    assert 'not the parameter set the search would have reported' in '\n'.join(printed)


def test_the_reported_objective_stops_being_the_luckiest_draw(tmp_path, _sync_dask):
    """Even when the search picked the right parameter set, the value it reported was the
    best of many noisy draws. After the stage the reported value is an average."""
    for replicate in (1, 2, 3, 4):
        NOISE[(1.0, replicate)] = 5.0

    algo = _algo(tmp_path, candidates=1, replicates=4)
    algo.trajectory.add(_ps(1.0), 0.5, 'best')
    algo._confirm_best_fit(_FakeClient())

    assert algo.trajectory.best_fit_name() == 'best'
    assert algo.trajectory.best_score() == 5.0
    text = open(algo.res_dir + '/best_fit_confirmation.txt').read()
    assert 'optimism\t4.5' in text


def test_every_replicate_gets_a_fresh_index_past_the_ones_the_fit_used(tmp_path, _sync_dask):
    """Under the default seed policy the seed comes from the parameter values and the
    replicate index, so reusing index 0 would reproduce the simulation the search already
    ran instead of drawing a new one."""
    algo = _algo(tmp_path, candidates=2, replicates=3)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')
    algo.trajectory.add(_ps(2.0), 2.0, 'b')
    client = _FakeClient()

    algo._confirm_best_fit(client)

    indices = sorted(job.replicate_index for job in client.submitted)
    assert indices == [1, 1, 2, 2, 3, 3]


def test_the_indices_clear_the_smoothing_replicates_too(tmp_path, _sync_dask):
    """With smoothing on, the fit itself already used indices 0 .. smoothing-1 for every
    parameter set, so the confirmation runs have to start above them."""
    algo = _algo(tmp_path, candidates=1, replicates=2, smoothing=3)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')
    client = _FakeClient()

    algo._confirm_best_fit(client)

    # 1 candidate x 2 replicates x 3 smoothing sub-runs, none of them reusing 0, 1 or 2.
    assert len(client.submitted) == 6
    assert sorted(job.replicate_index for job in client.submitted) == [3, 4, 5, 6, 7, 8]
    # Each replicate is still one whole evaluation, so it is one score, not three.
    text = open(algo.res_dir + '/best_fit_confirmation.txt').read()
    row = [l for l in text.splitlines() if l.startswith('1\t')][0]
    assert row.split('\t')[5] == '2'   # the "runs" column


def test_a_failed_replicate_is_counted_and_the_rest_still_decide(tmp_path, _sync_dask):
    class _SometimesFails(_NoisyModel):
        def copy_with_param_set(self, params):
            copy = _NoisyModelCopy(self.name)
            original = copy.execute

            def execute(folder, prefix, timeout):
                from pybnf.pset import FailedSimulationError
                if copy._pybnf_replicate_index == 2:
                    raise FailedSimulationError('forced failure')
                return original(folder, prefix, timeout)

            copy.execute = execute
            return copy

    NOISE[(1.0, 1)] = 7.0
    NOISE[(1.0, 3)] = 9.0
    algo = _algo(tmp_path, models={'m': _SometimesFails()}, candidates=1, replicates=3)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')

    algo._confirm_best_fit(_FakeClient())

    assert algo.trajectory.best_score() == 8.0    # mean of 7 and 9; the failure is left out
    row = [l for l in open(algo.res_dir + '/best_fit_confirmation.txt').read().splitlines()
           if l.startswith('1\t')][0]
    fields = row.split('\t')
    assert fields[5] == '2' and fields[6] == '1'  # 2 runs, 1 failed


def test_the_search_answer_stands_when_no_replicate_can_be_run(tmp_path, _sync_dask):
    class _AlwaysFails(_NoisyModel):
        def copy_with_param_set(self, params):
            copy = _NoisyModelCopy(self.name)

            def execute(folder, prefix, timeout):
                from pybnf.pset import FailedSimulationError
                raise FailedSimulationError('forced failure')

            copy.execute = execute
            return copy

    algo = _algo(tmp_path, models={'m': _AlwaysFails()}, candidates=1, replicates=2)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')

    algo._confirm_best_fit(_FakeClient())

    assert algo.trajectory.pinned_best() is None
    assert algo.trajectory.best_score() == 1.0
    assert 'winner\tnone' in open(algo.res_dir + '/best_fit_confirmation.txt').read()


def test_a_broken_stage_never_kills_a_finished_fit(tmp_path, _sync_dask, monkeypatch):
    algo = _algo(tmp_path, candidates=1, replicates=2)
    algo.trajectory.add(_ps(1.0), 1.0, 'a')
    monkeypatch.setattr(_ConcreteAlgorithm, 'make_job',
                        lambda self, params, replicate_offset=0: 1 / 0)

    algo._confirm_best_fit(_FakeClient())    # does not raise

    assert algo.trajectory.best_score() == 1.0


def test_a_refine_writes_its_own_table(tmp_path, _sync_dask):
    algo = _algo(tmp_path, candidates=1, replicates=2)
    algo.refine = True
    algo.trajectory.add(_ps(1.0), 1.0, 'a')

    algo._confirm_best_fit(_FakeClient())

    assert os.path.exists(algo.res_dir + '/best_fit_confirmation_refine.txt')
    assert not os.path.exists(algo.res_dir + '/best_fit_confirmation.txt')


# --------------------------------------------------------------------------- #
# Wiring into the end-of-fit path
# --------------------------------------------------------------------------- #
def test_the_rest_of_the_end_of_fit_path_uses_the_confirmed_parameter_set(
        tmp_path, _sync_dask, monkeypatch):
    """The winner is recorded on the trajectory rather than passed around, so everything
    the run writes after this stage -- the saved simulations, the best-fit model file, the
    information criteria -- describes the same parameter set."""
    for replicate in (1, 2):
        NOISE[(1.0, replicate)] = 5.0
        NOISE[(2.0, replicate)] = 2.0

    algo = _algo(tmp_path, candidates=2, replicates=2)
    algo.trajectory.add(_ps(1.0), 0.1, 'lucky')
    algo.trajectory.add(_ps(2.0), 2.0, 'real')
    algo.stop_reason = None
    algo.output_counter = 0

    handed = {}
    monkeypatch.setattr(_ConcreteAlgorithm, 'output_results', lambda self, name='', **k: None)
    monkeypatch.setattr(_ConcreteAlgorithm, '_copy_best_fit_sims',
                        lambda self, p, n: handed.update(pset=p, name=n))
    for skipped in ('_rerun_best_fit_to_save_data', '_emit_best_fit_bngl',
                    '_emit_profiled_noise', '_emit_inference_data',
                    '_finalize_backup_pickle', '_teardown_sim_dir'):
        monkeypatch.setattr(_ConcreteAlgorithm, skipped, lambda self, *a, **k: None)
    monkeypatch.setattr(_ConcreteAlgorithm, '_compute_information_criteria',
                        lambda self, p: None)
    monkeypatch.setattr(_ConcreteAlgorithm, '_emit_information_criteria', lambda self, ic: None)

    algo._finalize_run(_FakeClient())

    assert handed['name'] == 'real'
    assert handed['pset']['v1__FREE'] == 2.0


def test_run_hands_its_client_to_the_end_of_fit_path(tmp_path, monkeypatch):
    """The stage submits simulations, so it needs the client the fit was driven with."""
    seen = []
    monkeypatch.setattr(_ConcreteAlgorithm, '_finalize_run',
                        lambda self, client=None: seen.append(client))

    algo = _algo(tmp_path)
    algo.config.config.update({'backup_every': 10 ** 9, 'population_size': 1})
    algo.stop_reason = None
    algo.completed_simulations = 0
    algo.budget = None
    algo.variables = []
    monkeypatch.setattr(_ConcreteAlgorithm, '_budget_spent', lambda self: True)
    monkeypatch.setattr(_ConcreteAlgorithm, '_wall_time_stop_reason', lambda self, n: 'stopped')
    monkeypatch.setattr(_ConcreteAlgorithm, '_emit_start_point', lambda self, p, resumed: None)

    client = _FakeClient()
    client.scatter = lambda objs, broadcast=False: [_FakeFuture(o) for o in objs]
    client.cancel = lambda futures: None
    algo.run(client)

    assert seen == [client]


# --------------------------------------------------------------------------- #
# make_job's replicate offset
# --------------------------------------------------------------------------- #
def test_replicate_offset_shifts_a_plain_job(tmp_path):
    algo = _algo(tmp_path)
    assert algo.make_job(_ps(1.0, 'j'))[0].replicate_index == 0
    assert algo.make_job(_ps(1.0, 'j'), replicate_offset=5)[0].replicate_index == 5


def test_replicate_offset_shifts_a_whole_smoothing_group(tmp_path):
    algo = _algo(tmp_path, smoothing=3)
    assert [j.replicate_index for j in algo.make_job(_ps(1.0, 'j'))] == [0, 1, 2]
    assert [j.replicate_index for j in
            algo.make_job(_ps(1.0, 'k'), replicate_offset=3)] == [3, 4, 5]


# --------------------------------------------------------------------------- #
# What the configuration says at startup
# --------------------------------------------------------------------------- #
class _FakeBNGL(pset.BNGLModel):
    def __init__(self, name, *, seeded=False, stochastic=False):
        self.name = name
        self.seeded = seeded
        self.stochastic = stochastic
        self.has_observables = True
        self.file_path = '/tmp/%s.bngl' % name


def _cfg(models, *, candidates=None, replicates=None, edition=2, stochastic_seed='auto'):
    cfg = object.__new__(config.Configuration)
    cfg.models = models
    cfg.config = {'best_fit_candidates': candidates, 'best_fit_replicates': replicates,
                  'edition': edition, 'stochastic_seed': stochastic_seed}
    return cfg


def test_a_legacy_stochastic_fit_is_told_its_answer_will_be_optimistic(capsys):
    _cfg({'m': _FakeBNGL('m', stochastic=True)}, edition=None)._check_best_fit_confirmation()
    assert 'best_fit_replicates = 10' in capsys.readouterr().out


def test_nothing_is_said_when_no_model_is_stochastic(capsys):
    _cfg({'m': _FakeBNGL('m')}, edition=None)._check_best_fit_confirmation()
    assert capsys.readouterr().out == ''


def test_nothing_is_said_when_a_legacy_fit_has_already_turned_the_stage_on(capsys):
    _cfg({'m': _FakeBNGL('m', stochastic=True)}, edition=None,
         candidates=10, replicates=10)._check_best_fit_confirmation()
    assert capsys.readouterr().out == ''


def test_nothing_is_said_when_a_legacy_fit_has_turned_the_stage_off_on_purpose(capsys):
    _cfg({'m': _FakeBNGL('m', stochastic=True)}, edition=None,
         replicates=0)._check_best_fit_confirmation()
    assert capsys.readouterr().out == ''


def test_a_modern_fit_that_pins_every_seed_is_told_the_stage_cannot_help(capsys):
    _cfg({'m': _FakeBNGL('m', seeded=True, stochastic=True)},
         stochastic_seed='auto_honorbngl')._check_best_fit_confirmation()
    assert 'skip confirming the best fit' in capsys.readouterr().out


def test_a_negative_setting_is_refused():
    for bad in ({'best_fit_candidates': -1}, {'best_fit_replicates': -3}):
        key, value = next(iter(bad.items()))
        cfg = _cfg({'m': _FakeBNGL('m')})
        cfg.config[key] = value
        with pytest.raises(printing.PybnfError, match=key):
            cfg._check_best_fit_confirmation()
