"""The record of which methods a run actually executed (#564, ADR-0107).

A conf requests a method *chain* -- ``job_type = cmaes`` with ``refine = 1,
refine_method = gntr`` requests "search, then polish". What ran can be shorter, and
until ``Results/method_chain.json`` the only trace of that was a line on stdout: a
harness that scores a directory could not tell whether the method it believed it
measured had run at all.

Three layers are covered here: the record object itself, the artifacts a refined run
leaves in Results/, and the end-to-end chain a real fit-then-refine writes.
"""
import json
import os

import pytest

from . import integration_harness as H
from .context import algorithms, budget as budget_mod, method_chain as mc


class _Clock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def _read(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# The record object
# --------------------------------------------------------------------------- #
def test_a_phase_is_on_disk_as_soon_as_it_is_recorded(tmp_path):
    """Written after every phase, not at the end: a run whose refine raises must still
    leave behind the record of the fit that did happen."""
    path = tmp_path / 'method_chain.json'
    chain = mc.MethodChain(path, requested=['de', 'sim'], clock=_Clock())

    chain.record('fit', 'de', mc.COMPLETED, simulations=120, best_objective=1.5)

    doc = _read(path)
    assert doc['requested_methods'] == ['de', 'sim']
    assert doc['executed_methods'] == ['de']
    assert doc['phases'][0]['simulations'] == 120
    assert doc['phases'][0]['best_objective'] == 1.5


def test_a_skipped_phase_is_recorded_but_is_not_an_executed_method(tmp_path):
    """The whole point: requested != executed is visible without reading stdout."""
    clock = _Clock()
    chain = mc.MethodChain(tmp_path / 'm.json', requested=['cmaes', 'gntr'], clock=clock)

    clock.advance(1500.0)
    chain.record('fit', 'cmaes', mc.WALL_TIME_EXPIRED, reason='Wall-time budget reached: ...')
    chain.record('refine', 'gntr', mc.SKIPPED, reason='the search overran the reserve')

    doc = _read(tmp_path / 'm.json')
    assert doc['requested_methods'] == ['cmaes', 'gntr']
    assert doc['executed_methods'] == ['cmaes']
    assert doc['phases'][0]['elapsed_seconds'] == 1500.0
    assert doc['phases'][1]['status'] == mc.SKIPPED
    assert 'overran' in doc['phases'][1]['reason']


def test_bootstrap_replicates_are_recorded_but_stay_out_of_the_runs_chain(tmp_path):
    """A replicate re-runs the whole chain; "which methods did this run execute" is a
    question about the run, not about replicate 3."""
    chain = mc.MethodChain(tmp_path / 'm.json', requested=['de', 'sim'], clock=_Clock())
    chain.record('fit', 'de', mc.COMPLETED)
    chain.record('refine', 'sim', mc.COMPLETED)
    chain.record('fit', 'de', mc.COMPLETED, bootstrap_replicate=0)
    chain.record('bootstrap', 'de', mc.WALL_TIME_EXPIRED,
                 extra={'replicates_requested': 5, 'replicates_completed': 1})

    doc = _read(tmp_path / 'm.json')
    assert doc['executed_methods'] == ['de', 'sim']
    assert doc['phases'][2]['bootstrap_replicate'] == 0
    assert doc['phases'][3]['replicates_completed'] == 1


def test_a_non_finite_objective_is_recorded_as_null_rather_than_invalid_json(tmp_path):
    """``inf`` (every simulation failed) would serialize as ``Infinity``, which a strict
    parser rejects -- losing the whole record over one field."""
    chain = mc.MethodChain(tmp_path / 'm.json', clock=_Clock())
    chain.record('fit', 'de', mc.COMPLETED, best_objective=float('inf'))

    assert json.loads((tmp_path / 'm.json').read_text())['phases'][0]['best_objective'] is None


@pytest.mark.parametrize('conf,expected', [
    ({'fit_type': 'de'}, ['de']),
    ({'fit_type': 'de', 'refine': 0, 'refine_method': 'sim'}, ['de']),
    ({'fit_type': 'cmaes', 'refine': 1, 'refine_method': 'gntr'}, ['cmaes', 'gntr']),
    # refine_method == fit_type has always been skipped, so it was never a request the
    # run could satisfy; promising it in requested_methods would fail every comparison.
    ({'fit_type': 'sim', 'refine': 1, 'refine_method': 'sim'}, ['sim']),
])
def test_the_requested_chain_is_read_off_the_conf(conf, expected):
    assert mc.requested_methods(conf) == expected


def test_a_write_failure_never_takes_the_run_down_with_it(tmp_path):
    """A provenance file is a diagnostic; the same rule information_criteria.txt follows."""
    chain = mc.MethodChain(tmp_path / 'no_such_dir' / 'm.json', clock=_Clock())
    assert chain.record('fit', 'de', mc.COMPLETED) is not None   # did not raise
    assert chain.write() is False


# --------------------------------------------------------------------------- #
# The artifacts a refined run leaves behind
# --------------------------------------------------------------------------- #
def _best_objective_in(path):
    """The best (lowest) objective in a sorted_params file: column 1 of each data row."""
    with open(path) as f:
        rows = [ln.split() for ln in f if ln.strip() and not ln.startswith('#')]
    return min(float(r[1]) for r in rows)


def _fit_then_refine(tmp_path, monkeypatch, **overrides):
    """A real DE fit of a 2-D Gaussian, followed by the real Simplex refine
    ``main()`` would run -- the whole post-fit path, with only dask faked."""
    import types
    from pybnf import pybnf as pybnf_main

    H.install(monkeypatch)
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2, population_size=12,
                         max_iterations=10, refine=1, refine_method='sim',
                         simplex_max_iterations=10, **overrides)
    alg = algorithms.DifferentialEvolution(conf)
    alg.method_chain = mc.chain_for_run(alg.res_dir, conf.config, budget=alg.budget,
                                        version='test')
    H.drive(alg)
    pybnf_main._record_phase(alg, 'fit', 'de', mc.COMPLETED,
                             simulations=alg.completed_simulations,
                             best_objective=alg.trajectory.best_score())
    before = _best_objective_in(os.path.join(alg.res_dir, 'sorted_params_final.txt'))
    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)
    return conf, alg, alg.res_dir, before


def test_the_conventional_final_file_describes_the_refined_point(tmp_path, monkeypatch):
    """#564: ``sorted_params_final.txt`` kept the PRE-refine point while
    ``information_criteria.txt`` -- rewritten by the same end-of-run tail -- described
    the refined one, so the two files in Results/ disagreed about which parameter set
    they were about. The conventional name now means what it says."""
    conf, alg, res_dir, before = _fit_then_refine(tmp_path, monkeypatch)
    # The polish found a better point, so "which point is in the file" is a question
    # with two different answers -- which is what makes this assertion bite.
    assert alg.trajectory.best_score() < before

    final = os.path.join(res_dir, 'sorted_params_final.txt')
    refined = os.path.join(res_dir, 'sorted_params_refine_final.txt')
    assert os.path.isfile(final) and os.path.isfile(refined)
    # Both name the run's final point, and it is the point the trajectory ended on --
    # the one information_criteria.txt was computed from.
    assert _best_objective_in(final) == pytest.approx(alg.trajectory.best_score())
    with open(final) as a, open(refined) as b:
        assert a.read() == b.read()


def test_the_refine_is_still_reported_under_its_own_name(tmp_path, monkeypatch):
    """The refine_-prefixed file is unchanged: anything already reading it keeps
    working, and a reader that wants the pre-refine trajectory can still take the
    fit's periodic dumps."""
    conf, alg, res_dir, _ = _fit_then_refine(tmp_path, monkeypatch)
    assert os.path.isfile(os.path.join(res_dir, 'sorted_params_refine_final.txt'))


def test_the_refines_stop_reason_is_added_to_the_fits_rather_than_replacing_it(tmp_path):
    """Both phases write into one Results directory. A run where the search hit the
    deadline AND the polish did has two facts to report."""
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0], [1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=1, population_size=4,
                         max_iterations=1)
    alg = algorithms.DifferentialEvolution(conf)
    os.makedirs(alg.res_dir, exist_ok=True)

    alg.stop_reason = 'Wall-time budget reached: the search stopped'
    alg._announce_stop_reason()
    alg.refine = True
    alg.stop_reason = 'Wall-time budget reached: the refine stopped'
    alg._announce_stop_reason()

    with open(os.path.join(alg.res_dir, 'stop_reason.txt')) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines == ['Wall-time budget reached: the search stopped',
                     'Wall-time budget reached: the refine stopped']


# --------------------------------------------------------------------------- #
# End to end: the chain a real run writes
# --------------------------------------------------------------------------- #
def test_a_completed_fit_and_refine_record_the_chain_the_conf_asked_for(tmp_path, monkeypatch):
    conf, alg, res_dir, _ = _fit_then_refine(tmp_path, monkeypatch)

    doc = _read(os.path.join(res_dir, 'method_chain.json'))
    assert doc['requested_methods'] == ['de', 'sim']
    assert doc['executed_methods'] == ['de', 'sim']
    assert [p['phase'] for p in doc['phases']] == ['fit', 'refine']
    assert doc['phases'][1]['status'] == mc.COMPLETED
    assert doc['job_type'] == 'de' and doc['pybnf_version'] == 'test'


def test_a_refine_the_clock_cut_short_is_not_recorded_as_a_converged_one(tmp_path, monkeypatch):
    """The issue's quieter half: a refine that starts with four seconds left runs for
    four seconds, and nothing in the artifacts distinguished that from a polish that
    converged. Its phase now carries the budget's stop reason."""
    import types
    from pybnf import pybnf as pybnf_main

    H.install(monkeypatch)
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2, population_size=12,
                         max_iterations=10, refine=1, refine_method='sim')
    alg = algorithms.DifferentialEvolution(conf)
    alg.method_chain = mc.chain_for_run(alg.res_dir, conf.config, version='test')
    H.drive(alg)
    pybnf_main._record_phase(alg, 'fit', 'de', mc.COMPLETED)

    def _cut_short(self, client, resume=None, debug=False):
        self.stop_reason = 'Wall-time budget reached: the refine stopped after 0:00:04'

    monkeypatch.setattr(algorithms.SimplexAlgorithm, 'run', _cut_short)
    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)

    refine = _read(os.path.join(alg.res_dir, 'method_chain.json'))['phases'][1]
    assert refine['status'] == mc.WALL_TIME_EXPIRED
    assert '0:00:04' in refine['reason']


def test_a_downgraded_run_says_so_in_the_record_not_only_on_stdout(tmp_path, monkeypatch):
    """The provenance defect the issue reports: 15 benchmark runs configured as
    cmaes+gntr actually ran cmaes alone, and nothing on disk said so."""
    import types
    from pybnf import pybnf as pybnf_main

    H.install(monkeypatch)
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2, population_size=12,
                         max_iterations=10, refine=1, refine_method='sim')
    alg = algorithms.DifferentialEvolution(conf)
    alg.method_chain = mc.chain_for_run(alg.res_dir, conf.config, version='test')
    H.drive(alg)
    pybnf_main._record_phase(alg, 'fit', 'de', mc.WALL_TIME_EXPIRED, reason='out of time',
                             simulations=alg.completed_simulations)
    # The budget is gone by the time the refine would start.
    alg.budget = budget_mod.FitBudget(60.0, elapsed=60.0, clock=_Clock())
    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)

    doc = _read(os.path.join(alg.res_dir, 'method_chain.json'))
    assert doc['requested_methods'] == ['de', 'sim']
    assert doc['executed_methods'] == ['de']        # <- the downgrade, machine-readable
    assert doc['phases'][1] == dict(doc['phases'][1],
                                    phase='refine', method='sim', status=mc.SKIPPED)
