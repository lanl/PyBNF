"""Several independent runs of the Liu–West filter (``lwf_independent_runs``).

Oracles: run 0 of several is the single run, byte for byte; the combined files are each run's
own files stacked in run order; several runs on a real dask cluster, each on a worker's copy of
the filter, write byte for byte what the same runs write in sequence in this process.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.budget import FitBudget
from pybnf.cluster import Cluster
from pybnf.printing import PybnfError

from . import lwf_harness as C
from . import recovery_harness as H
from .integration_harness import FakeClient

pytestmark = pytest.mark.bngsim

TABLES = ('parameters.txt', 'parameters_by_row.txt', 'predicted_counts.txt', 'weight_ess.txt',
          'forecasts.txt')


@pytest.fixture(autouse=True)
def _bng2pl():
    H.require_bng2pl()


def _build(tmp_path, runs=2, particles=30, replace=None, **kw):
    replace = dict(replace or {})
    replace.setdefault('lwf_particles', 'lwf_particles = %d' % particles)
    extra = list(kw.pop('extra', ())) + (['lwf_independent_runs = %d' % runs] if runs > 1 else [])
    return C.build(C.write_job(tmp_path, replace=replace, extra=extra, **kw))


def _without(text, line):
    return ''.join(x for x in text.splitlines(keepends=True) if x != line)


# --------------------------------------------------------------------------------------------
# What each run is, and how runs are combined
# --------------------------------------------------------------------------------------------

def test_run_0_of_several_is_the_single_run_and_run_1_draws_its_own_streams(tmp_path):
    single = C.run(_build(tmp_path / 'one', runs=1))
    several = C.run(_build(tmp_path / 'two', runs=2))
    for name in TABLES:
        own = (several / 'run_0' / name).read_text()
        assert '# Independent run 0 of 2.\n' in own
        assert _without(own, '# Independent run 0 of 2.\n') == (single / name).read_text(), name
    _, run0 = C.read_table(several / 'run_0' / 'parameters.txt')
    _, run1 = C.read_table(several / 'run_1' / 'parameters.txt')
    assert not np.array_equal(run0, run1)
    assert 'independent run 1 of 2' in (several / 'run_1' / 'summary.txt').read_text()


def test_the_combined_files_stack_each_runs_own_resample_in_run_order(tmp_path):
    out = C.run(_build(tmp_path, runs=3, particles=20))
    for name in ('parameters.txt', 'predicted_counts.txt', 'forecasts.txt'):
        header, combined = C.read_table(out / name)
        parts = [C.read_table(out / ('run_%d' % r) / name) for r in range(3)]
        assert all(h == header for h, _ in parts)
        np.testing.assert_array_equal(combined, np.vstack([rows for _, rows in parts]))
        text = (out / name).read_text()
        assert lw.SAMPLE_STATEMENT in text
        assert ('The 3 independent runs combined with equal weight: rows 20 r + 1 to 20 (r + 1) '
                'are independent run r\'s own equal-weight resample (run_<r>/%s)' % name) in text
    # The per-row histories describe one population each, and are not combined.
    assert not (out / 'weight_ess.txt').exists() and not (out / 'parameters_by_row.txt').exists()
    summary = (out / 'summary.txt').read_text()
    assert '3 independent runs of 20 particles each' in summary
    assert 'combined with equal weight' in summary and 'whatever its log evidence' in summary
    for r in range(3):
        assert 'Run %d: 12 of 12 rows assimilated (t = 1 to 12); weight ESS' % r in summary


def test_several_runs_on_a_cluster_equal_the_same_runs_in_sequence_byte_for_byte(
        tmp_path, monkeypatch):
    """Fresh runs, and runs continued from their state files after row 5; the main process's
    own run method fails, so no run was filtered here."""
    from distributed import Client, LocalCluster
    sequential = C.run(_build(tmp_path / 'seq'))
    fresh = _build(tmp_path / 'cluster')
    C.run(_build(tmp_path / 'parts', counts=C.COUNTS[:5]))
    continued = _build(tmp_path / 'parts', extra=['lwf_continue = 1'])
    shutil.rmtree(continued.res_dir)

    def not_here(self, run, saved=None):
        raise AssertionError('run %d was filtered in the main process' % run)

    real = lw.LiuWestFilter._filter_run
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                      dashboard_address=':0') as cluster, Client(cluster) as client:
        # The workers import the module afresh, so only this process's runs would fail.
        monkeypatch.setattr(lw.LiuWestFilter, '_filter_run', not_here)
        try:
            C.run(fresh, client)
            C.run(continued, client)
        finally:
            monkeypatch.setattr(lw.LiuWestFilter, '_filter_run', real)
    assert C.files(sequential) == C.files(Path(fresh.res_dir) / 'LWF')
    assert C.files(sequential) == C.files(Path(continued.res_dir) / 'LWF')
    # Each worker filtered only rows 6 to 12 and the forecast, not every row again.
    assert continued.completed_simulations == 2 * 30 * (12 - 5 + 2)


def test_a_worker_is_handed_the_budget_left_when_its_task_was_sent(tmp_path, monkeypatch):
    """A monotonic clock does not cross processes: the task carries the limit, what was spent
    and the wall-clock time it was sent."""
    alg = _build(tmp_path)
    seen = []
    real = lw.LiuWestFilter._filter_run

    def recording(self, run, saved=None):
        seen.append((self.budget.limit, self.budget.elapsed()))
        return real(self, run, saved)

    monkeypatch.setattr(lw.LiuWestFilter, '_filter_run', recording)
    alg.budget = FitBudget(3600.0, elapsed=100.0)
    C.run(alg, FakeClient())
    assert [limit for limit, _ in seen] == [3600.0, 3600.0]
    assert all(100.0 <= spent < 160.0 for _, spent in seen)


# --------------------------------------------------------------------------------------------
# The cluster main() starts
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('runs, given, workers', [(2, 8, 2), (3, 1, 1), (16, None, 4), (3, None, 3)],
                         ids=['capped', 'read', 'cpus', 'runs'])
def test_several_runs_ask_main_for_a_cluster_of_at_most_one_worker_each(tmp_path, monkeypatch,
                                                                        runs, given, workers):
    monkeypatch.setattr(Cluster, 'cpus_per_node', staticmethod(lambda: (4, 'test')))
    extra = [] if given is None else ['parallel_count = %d' % given]
    alg = _build(tmp_path, runs=runs, extra=extra)
    assert alg.needs_cluster is True
    assert alg.config.config['parallel_count'] == workers


# --------------------------------------------------------------------------------------------
# Runs that end early
# --------------------------------------------------------------------------------------------

def test_runs_a_budget_stops_at_different_rows_write_their_own_files_and_no_combined_ones(
        tmp_path):
    """In sequence the runs share the clock: run 0 assimilates three rows, run 1 none. Each
    run's files go through its own last row and its forecast starts there; the combined files
    would stack populations at different rows, so they are not written."""
    alg = _build(tmp_path)
    C.one_second_per_row(alg, 2.5)
    out = C.run(alg)
    assert C.read_table(out / 'run_0' / 'weight_ess.txt')[1].shape[0] == 3
    assert C.read_table(out / 'run_1' / 'weight_ess.txt')[1].shape[0] == 0
    assert 'before any row was assimilated' in (out / 'run_1' / 'parameters.txt').read_text()
    for name in ('parameters.txt', 'predicted_counts.txt', 'forecasts.txt'):
        assert not (out / name).exists(), name
    assert C.read_table(out / 'run_0' / 'forecasts.txt')[0][0] == '4'
    assert C.read_table(out / 'run_1' / 'forecasts.txt')[0][0] == '1'
    reason = (Path(alg.res_dir) / 'stop_reason.txt').read_text()
    assert ('stopped its 2 independent runs at different rows: run 0 after assimilating row 3 '
            'of 12 (t = 3); run 1 before assimilating any row') in reason
    assert 'the combined parameters and predicted counts are not written' in reason.lower()
    summary = (out / 'summary.txt').read_text()
    assert 'The combined parameters and predicted counts were not written: the runs assimilated ' \
           'different numbers of rows (run 0: 3, run 1: 0).' in summary
    assert 'Stopped early: ' + reason.strip() in summary
    assert 'Wall-time budget reached: this run stopped after assimilating row 3' in (
        out / 'run_0' / 'summary.txt').read_text()


def test_runs_a_budget_stops_at_the_same_row_are_combined_through_it_and_forecast_from_it(
        tmp_path, monkeypatch):
    alg = _build(tmp_path)
    alg.budget = FitBudget(1.0)
    monkeypatch.setattr(alg, '_budget_spent', lambda: len(alg._history) >= 3)
    out = C.run(alg)
    header, combined = C.read_table(out / 'predicted_counts.txt')
    assert header == ['1', '2', '3'] and combined.shape == (60, 3)
    assert 'through row 3 of 12 (t = 3)' in (out / 'parameters.txt').read_text()
    header, forecasts = C.read_table(out / 'forecasts.txt')
    assert header[0] == '4' and forecasts.shape[0] == 60
    assert 'stopped after assimilating row 3 of 12 (t = 3) in each of its 2 independent runs' \
        in (Path(alg.res_dir) / 'stop_reason.txt').read_text()


def test_a_degenerate_update_in_every_run_names_each_and_combines_the_rows_before_it(tmp_path):
    model = C.MODEL.replace('rho    0.5', 'rho    0')
    alg = _build(tmp_path, particles=20, model=model, counts=(0, 0, 3, 0))
    with pytest.raises(PybnfError) as info:
        C.run(alg)
    message = info.value.message
    assert message.startswith('Degenerate update in 2 of 2 independent runs. Run 0: Degenerate '
                              'update at row 3 (t = 3, count 3)')
    assert 'Run 1: Degenerate update at row 3' in message
    out = Path(alg.res_dir) / 'LWF'
    assert C.read_table(out / 'predicted_counts.txt')[1].shape == (40, 2)
    summary = (out / 'summary.txt').read_text()
    assert 'Stopped early, run 0: Degenerate update at row 3' in summary
    assert 'Stopped early, run 1: Degenerate update at row 3' in summary
