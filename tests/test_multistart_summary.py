"""The per-start summary a multi-start fit writes at the end of a run (#658).

Several fit types run more than one search from different starting points and report the
best of them. That one number cannot be checked: a run whose twenty starts all reached
the same objective value and a run whose twenty starts all landed somewhere different
used to print exactly the same thing, so nobody could tell whether the answer was worth
believing. The fix is ``Results/multistart_summary.txt``, one row per start sorted by
final objective value, plus a short version on the screen.

Three families of fit type produce those rows, and each keeps the numbers somewhere
different, so each is checked here:

* the concurrent local optimizers (``powell`` / ``sim``, and the gradient methods
  ``trf`` / ``lbfgs`` / ``gntr``, which share the same base) read them off their per-start
  runner objects;
* the metaheuristics (``de`` / ``ade`` / ``ss`` / ``pso``) tally them in the mixin that
  drives their restarts, because those methods keep no per-start final value of their own;
* multiple shooting (``ms``) reads them off its per-start homotopy results. Its rows are
  checked white-box here rather than end to end, since that fit type needs a
  sensitivity-capable simulation backend (``tests/test_shooting_sbml.py`` owns those).

The formatting itself is pure and file-free (``pybnf.algorithms.multistart_report``), so
the first group of tests exercises it directly.
"""
import numpy as np
import pytest

from pybnf.algorithms import multistart_report as R
from pybnf.algorithms.multistart_report import StartRecord

from . import integration_harness as H
from .context import algorithms


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #
def _rows(lines):
    """The data rows of a written summary, split into columns."""
    return [line.split('\t') for line in lines if line and not line.startswith('#')
            and line.split('\t')[0].isdigit()]


def test_rows_are_sorted_best_objective_first():
    """The whole point of the table is the shape of the sorted objective column, so the
    rows come out best first however the starts happened to be numbered."""
    records = [StartRecord(1, 9.0), StartRecord(2, 1.0), StartRecord(3, 5.0)]
    assert [r.start for r in R.sorted_records(records)] == [2, 3, 1]
    rows = _rows(R.summary_lines(records))
    assert [row[0] for row in rows] == ['1', '2', '3']            # rank column
    assert [row[1] for row in rows] == ['2', '3', '1']            # start numbers
    assert [row[2] for row in rows] == ['1', '5', '9']            # objective values


def test_a_start_with_no_objective_sorts_last():
    """A start that never ran has nothing to compare, so it goes to the bottom rather than
    sorting as if it had scored zero."""
    records = [StartRecord(1, None, stop_reason=R.NOT_STARTED), StartRecord(2, 4.0)]
    rows = _rows(R.summary_lines(records))
    assert [row[1] for row in rows] == ['2', '1']
    assert rows[1][2] == 'none'
    assert rows[1][5] == R.NOT_STARTED


def test_a_start_that_never_stopped_says_so():
    """A start still running when the fit ended (a wall-time budget, usually) is listed
    with what it had reached. Dropping it would make a cut-short fit look like a complete
    set of starts that agreed with each other."""
    rows = _rows(R.summary_lines([StartRecord(1, 2.0, stop_reason='converged'),
                                  StartRecord(2, 3.0, stop_reason=None)]))
    assert rows[0][5] == 'converged'
    assert rows[1][5] == R.UNFINISHED


def test_missing_counts_are_reported_as_such():
    """A method that keeps no iteration count says so rather than printing a zero, which
    would read as a start that took no steps."""
    rows = _rows(R.summary_lines([StartRecord(1, 2.0, iterations=None, evaluations=40),
                                  StartRecord(2, 3.0, iterations=7, evaluations=12)]))
    assert rows[0][3] == 'n/a' and rows[0][4] == '40'
    assert rows[1][3] == '7' and rows[1][4] == '12'


def test_the_plateau_count_separates_agreement_from_disagreement():
    """The number a reader acts on. Starts bunched at the same low value count together
    (run more starts, probably pointless); starts spread out do not (run more starts)."""
    agreed = [StartRecord(i + 1, 10.0 + i * 1e-6) for i in range(5)]
    spread = [StartRecord(i + 1, 10.0 * (i + 1)) for i in range(5)]
    assert R.plateau_count(agreed) == 5
    assert R.plateau_count(spread) == 1


def test_the_plateau_count_works_on_a_negative_objective():
    """Log-likelihood objectives are negative, so "within a fraction of the best" has to be
    measured against the size of the best value, not against its sign."""
    records = [StartRecord(1, -200.0), StartRecord(2, -199.99), StartRecord(3, -20.0)]
    assert R.plateau_count(records) == 2


def test_starts_that_all_failed_are_reported_rather_than_hidden():
    """Every start failing to simulate is the single most useful thing this table can say,
    so it survives to both the file and the screen."""
    records = [StartRecord(i + 1, float('inf'), stop_reason='start point failed to simulate')
               for i in range(3)]
    rows = _rows(R.summary_lines(records))
    assert [row[2] for row in rows] == ['inf', 'inf', 'inf']
    assert R.plateau_count(records) == 0
    console = ' '.join(R.console_lines(records, '/tmp/x.txt'))
    assert 'none of the 3 starts produced a usable fit' in console


def test_a_reason_carrying_a_tab_cannot_break_its_row():
    """The file is tab separated and the reasons come from the methods themselves, so the
    text is flattened before it goes in a cell."""
    rows = _rows(R.summary_lines([StartRecord(1, 1.0, stop_reason='a\treason\nwith breaks'),
                                  StartRecord(2, 2.0)]))
    assert len(rows[0]) == 6
    assert rows[0][5] == 'a reason with breaks'


# --------------------------------------------------------------------------- #
# The concurrent local optimizers (powell / sim, and the gradient methods)
# --------------------------------------------------------------------------- #
_LOCAL = {'powell': algorithms.PowellAlgorithm, 'sim': algorithms.SimplexAlgorithm}

# A shallow local mode at the box center and a deeper one off center, so the starts
# genuinely disagree and the table has something to report.
_MODES = [(0.5, [0.0, 0.0], [1.0, 1.0]), (0.5, [6.0, 6.0], [4.0, 4.0])]


def _local_config(tmp_path, fit_type, n_starts, **overrides):
    tgt, exp = H.write_target(tmp_path, H.multimodal_spec(_MODES))
    base = dict(n_params=2, var_type='uniform_var', bounds=(-10.0, 10.0),
                population_size=8, max_iterations=40, random_seed=1234, n_starts=n_starts)
    base.update(overrides)
    return H.make_config(tmp_path, fit_type, tgt, exp, **base)


def _read_summary(alg, name='multistart_summary.txt'):
    with open('%s/%s' % (alg.res_dir, name)) as f:
        return f.read()


@pytest.mark.parametrize('fit_type', list(_LOCAL))
def test_a_local_multistart_fit_writes_one_row_per_start(tmp_path, fit_type):
    """The deliverable, end to end through the real run loop: every start of a concurrent
    multi-start fit gets a row, with the objective it reached, the steps it took, the
    simulations it cost, and why it stopped."""
    alg = _LOCAL[fit_type](_local_config(tmp_path, fit_type, n_starts=4))
    H.drive(alg)

    text = _read_summary(alg)
    rows = _rows(text.splitlines())
    assert len(rows) == 4
    assert sorted(int(row[1]) for row in rows) == [1, 2, 3, 4]
    assert 'starts\t4' in text
    objectives = [float(row[2]) for row in rows]
    assert objectives == sorted(objectives)                  # best first
    # Every start reported a real objective, ran at least one step, cost at least one
    # simulation, and said why it stopped.
    assert all(np.isfinite(v) for v in objectives)
    assert all(int(row[3]) >= 1 and int(row[4]) >= 1 for row in rows)
    assert all(row[5] and row[5] != R.UNFINISHED for row in rows)
    # The run's reported best fit is the best over all the starts, so no start can beat it.
    assert alg.trajectory.best_score() <= min(objectives) + 1e-9


@pytest.mark.parametrize('fit_type', list(_LOCAL))
def test_a_single_start_fit_writes_no_summary(tmp_path, fit_type):
    """One start has nothing to be compared against, so the table would only restate the
    reported best fit. Not written."""
    alg = _LOCAL[fit_type](_local_config(tmp_path, fit_type, n_starts=1))
    H.drive(alg)
    with pytest.raises(FileNotFoundError):
        _read_summary(alg)


def test_the_evaluation_count_adds_up_to_the_simulations_the_fit_ran(tmp_path):
    """The per-start evaluation counts are a breakdown of the run's own work, so they
    account for every completed simulation and none twice."""
    alg = _LOCAL['powell'](_local_config(tmp_path, 'powell', n_starts=3))
    H.drive(alg)
    records = alg.multistart_records()
    assert sum(r.evaluations for r in records) == alg.completed_simulations


def test_an_unfinished_start_still_appears(tmp_path):
    """White-box, because a wall-time budget expiring mid-start is awkward to arrange:
    a runner that never set a stop reason is still a row, so a fit cut short does not
    silently report a smaller, better-agreeing set of starts than it ran."""
    alg = _LOCAL['powell'](_local_config(tmp_path, 'powell', n_starts=3))
    alg.start_run()
    alg.runners[0].fval, alg.runners[0].iteration = 1.0, 5
    alg.runners[0].stop_reason = 'converged'
    alg.runners[1].fval, alg.runners[1].iteration = 2.0, 3   # still running: no stop reason
    alg.runners[2].fval, alg.runners[2].iteration = 3.0, 4
    alg.runners[2].stop_reason = 'converged'

    records = alg.multistart_records()
    assert [r.start for r in records] == [1, 2, 3]
    assert records[1].stop_reason is None
    assert R.UNFINISHED in '\n'.join(R.summary_lines(records))


# --------------------------------------------------------------------------- #
# The metaheuristics, whose starts run one after another
# --------------------------------------------------------------------------- #
_META = {'de': algorithms.DifferentialEvolution,
         'ss': algorithms.ScatterSearch,
         'pso': algorithms.ParticleSwarm,
         'ade': algorithms.AsynchronousDifferentialEvolution}

_META_BUDGET = {'de': dict(population_size=8, max_iterations=15),
                'ss': dict(population_size=5, max_iterations=6),
                'pso': dict(population_size=8, max_iterations=15),
                'ade': dict(population_size=8, max_iterations=15)}


def _meta_config(tmp_path, fit_type, n_starts):
    tgt, exp = H.write_target(tmp_path, H.multimodal_spec(_MODES))
    base = dict(n_params=2, var_type='uniform_var', bounds=(-10.0, 10.0),
                random_seed=1234, n_starts=n_starts)
    base.update(_META_BUDGET[fit_type])
    return H.make_config(tmp_path, fit_type, tgt, exp, **base)


@pytest.mark.parametrize('fit_type', list(_META))
def test_a_metaheuristic_multistart_fit_writes_one_row_per_start(tmp_path, fit_type):
    """These methods run their starts one after another and keep no final value of their
    own, so the mixin takes the best objective each start reached. Every start still gets
    a row, and the iteration column says n/a rather than inventing a shared unit."""
    alg = _META[fit_type](_meta_config(tmp_path, fit_type, n_starts=3))
    H.drive(alg)

    rows = _rows(_read_summary(alg).splitlines())
    assert len(rows) == 3
    assert sorted(int(row[1]) for row in rows) == [1, 2, 3]
    objectives = [float(row[2]) for row in rows]
    assert objectives == sorted(objectives) and all(np.isfinite(v) for v in objectives)
    assert all(row[3] == 'n/a' for row in rows)              # no shared iteration count
    assert all(int(row[4]) >= 1 for row in rows)             # but every start cost work
    assert alg.trajectory.best_score() <= min(objectives) + 1e-9


def test_a_single_start_metaheuristic_writes_no_summary(tmp_path):
    alg = _META['de'](_meta_config(tmp_path, 'de', n_starts=1))
    H.drive(alg)
    with pytest.raises(FileNotFoundError):
        _read_summary(alg)


def test_a_start_the_fit_never_reached_is_listed_as_such(tmp_path):
    """These starts run in sequence, so a fit that stops early never reaches the later
    ones. They are listed as starts that never ran, so a six-row table for a twenty-start
    fit cannot be misread as twenty starts that agreed."""
    alg = _META['de'](_meta_config(tmp_path, 'de', n_starts=5))
    alg._start_stats = [{'objective': 3.0, 'evaluations': 20, 'stop_reason': 'ended'},
                        {'objective': 2.0, 'evaluations': 20, 'stop_reason': 'ended'}]
    records = alg.multistart_records()
    assert len(records) == 5
    assert [r.stop_reason for r in records[2:]] == [R.NOT_STARTED] * 3
    assert all(r.objective is None for r in records[2:])


# --------------------------------------------------------------------------- #
# Multiple shooting
# --------------------------------------------------------------------------- #
class _FakeOuter:
    def __init__(self, n_iterates, n_evaluations):
        self.iterates = [None] * n_iterates
        self.n_evaluations = n_evaluations


class _FakeStage:
    def __init__(self, n_iterates, n_evaluations):
        self.outer = _FakeOuter(n_iterates, n_evaluations)


class _FakeHomotopy:
    def __init__(self, score, stages, stop_reason):
        self.best_score = score
        self.stages = stages
        self.stop_reason = stop_reason

    @property
    def n_evaluations(self):
        return sum(s.outer.n_evaluations for s in self.stages)


def test_multiple_shooting_reads_its_rows_off_its_ladder_results():
    """``ms`` drives its own search rather than the per-start step machines the shared base
    reads, so its rows come from the homotopy result each start produced. Checked against
    the shape of those results, so it needs no simulation backend."""
    from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm

    alg = MultipleShootingAlgorithm.__new__(MultipleShootingAlgorithm)
    alg.start_psets = [object(), object(), object()]
    alg.homotopies = [
        _FakeHomotopy(5.0, [_FakeStage(3, 30), _FakeStage(2, 20)], 'converged'),
        _FakeHomotopy(1.0, [_FakeStage(4, 40)], 'max_outer'),
    ]

    records = alg.multistart_records()
    assert [r.start for r in records] == [1, 2, 3]
    assert records[0].objective == 5.0
    assert records[0].iterations == 5 and records[0].evaluations == 50
    assert records[1].stop_reason == 'max_outer'
    # The third start was never reached: the run stopped before the loop got to it.
    assert records[2].objective is None and records[2].stop_reason == R.NOT_STARTED
