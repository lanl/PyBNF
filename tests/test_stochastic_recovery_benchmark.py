"""Tests for the stochastic parameter recovery benchmark (lanl/PyBNF#663, small version).

The benchmark lives in ``benchmarks/stochastic_recovery/`` (it ships nothing and is not
an importable package), so this module adds that directory to ``sys.path`` and imports
it, as ``test_benchmark_harness`` does for the sampler benchmarks.

Three layers, so a failure points at the right one:

* **Frozen definitions** (default tier, no backend): every problem directory is
  self-consistent -- the JSON names a model that declares each free parameter's
  ``__FREE`` alias and one simulate action matching the frozen settings, the true
  values sit inside the bounds and away from their edges, and the committed data
  file has exactly the columns and sampling times the definition promises.
* **Scoring rules** (default tier, pure Python): the log-error, success, cost-to-success
  and aggregation functions on hand-made records.
* **The backend path** (``recovery`` tier, needs bngsim and BNG2.pl; the network-free
  problem also needs NFsim): the committed data can be regenerated, every identifiable
  parameter has leverage on the frozen objective, and a budgeted fit produces a
  complete record whose simulation counts add up.
"""
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / 'benchmarks'))
from stochastic_recovery import protocol  # noqa: E402  (path-dependent import of the benchmark)

PROBLEMS = protocol.load_problems()


def _param(problem):
    marks = [pytest.mark.bngsim_nfsim] if problem.method == 'nf' else []
    return pytest.param(problem, id=problem.id, marks=marks)


# --------------------------------------------------------------------------- #
# Frozen definitions
# --------------------------------------------------------------------------- #
def test_six_problems_are_defined():
    assert [p.id for p in PROBLEMS] == ['Hlavacek_PNAS2001', 'Lin_PhysRevE2016', 'McKane_PhysRevLett2005',
                                       'Munsky_Science2012', 'Shahrezaei_PNAS2008', 'Yang_PhysRevE2008']
    assert {p.method for p in PROBLEMS} == {'ssa', 'nf'}


@pytest.mark.parametrize('problem', [pytest.param(p, id=p.id) for p in PROBLEMS])
def test_problem_definition_is_self_consistent(problem):
    assert problem.directory.name == problem.id
    assert problem.version == protocol.FORMAT_VERSION
    assert problem.reference and 'doi:' in problem.reference
    assert problem.source.startswith('BNGL-Models/')
    assert problem.identifiable, 'a problem with nothing identifiable cannot be scored'
    assert problem.budget_simulations > 0 and problem.smoothing >= 1
    assert problem.data_replicates >= 10
    assert problem.data_seed_offset >= 10 ** 5, 'data replicates must not collide with a fit\'s'

    model = problem.model_path.read_text()
    assert '#@reference:' in model and '#@source:' in model
    assert re.search(r'#@model_id:\s*%s\b' % re.escape(problem.id), model)
    for p in problem.parameters:
        assert p.name.endswith('__FREE')
        assert re.search(r'^\s*%s\s+%s\b' % (re.escape(p.name[:-6]), re.escape(p.name)), model, re.M), \
            '%s must be declared as the alias of %s' % (p.name, p.name[:-6])
        assert p.low < p.true < p.high
        assert math.log10(p.true / p.low) >= 0.3 and math.log10(p.high / p.true) >= 0.3, \
            '%s: the true value should sit at least 0.3 decades from each bound' % p.name
    actions = re.findall(r'^\s*simulate\(\{(.*)\}\)', model, re.M)
    assert len(actions) == 1, 'exactly one simulate action'
    settings = dict(re.findall(r'(\w+)=>"?([\w.]+)"?', actions[0]))
    assert settings['method'] == problem.method
    assert settings['suffix'] == problem.suffix
    assert float(settings['t_start']) == problem.t_start
    assert float(settings['t_end']) == problem.t_end
    assert int(settings['n_steps']) == problem.n_steps
    assert 'seed' not in settings, 'seeds come from PyBNF\'s policy, never from the action'
    for obs in problem.observables:
        assert re.search(r'^\s*(Molecules|Species)\s+%s\s' % re.escape(obs), model, re.M), obs


@pytest.mark.parametrize('problem', [pytest.param(p, id=p.id) for p in PROBLEMS])
def test_committed_data_matches_definition(problem):
    lines = [ln for ln in problem.data_path.read_text().splitlines() if ln.strip()]
    header = lines[0].lstrip('#').split()
    expected = ['time'] + list(problem.observables) + [o + '_SD' for o in problem.observables]
    assert header == expected
    arr = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    assert arr.shape == (problem.n_steps + 1, len(expected))
    assert np.allclose(arr[:, 0], problem.sample_times)
    assert np.isfinite(arr).all()
    n = len(problem.observables)
    sd = arr[:, 1 + n:]
    assert (sd > 0).all(), 'every sigma is floored above zero'
    means = arr[:, 1:1 + n]
    assert (means.max(axis=0) > 0).all(), 'every observable is seen'


# --------------------------------------------------------------------------- #
# Scoring rules
# --------------------------------------------------------------------------- #
def test_log10_errors_are_in_decades():
    errs = protocol.log10_errors({'a': 20.0, 'b': 0.5, 'c': None, 'd': -1.0}, {'a': 2.0, 'b': 1.0, 'c': 1.0, 'd': 1.0})
    assert errs['a'] == pytest.approx(1.0)
    assert errs['b'] == pytest.approx(math.log10(2))
    assert math.isinf(errs['c']) and math.isinf(errs['d'])


def test_max_error_scores_only_identifiable_parameters():
    errs = {'a': 0.05, 'b': 2.0}
    assert protocol.max_error(errs, ['a']) == 0.05
    assert protocol.max_error(errs, ['a', 'b']) == 2.0
    assert protocol.rms_error({'a': 0.3, 'b': 0.4}, ['a', 'b']) == pytest.approx(math.sqrt(0.125))


def test_success_thresholds():
    assert protocol.TOL_LOOSE == pytest.approx(math.log10(2))
    assert protocol.TOL_TIGHT == 0.1


def _trace(*points):
    """A trace from ``(sims, error_a, error_b)`` triples."""
    return [(s, {'a': ea, 'b': eb}) for s, ea, eb in points]


def test_simulations_to_success_reads_the_first_crossing_on_identifiable_parameters():
    trace = _trace((100, 1.2, 0.0), (300, 0.5, 0.0), (500, 0.2, 0.9), (700, 0.25, 0.0), (900, 0.05, 0.0))
    assert protocol.simulations_to_success(trace, ['a']) == 500
    assert protocol.simulations_to_success(trace, ['a', 'b']) == 700
    assert protocol.simulations_to_success(trace, ['a'], tol=0.1) == 900
    assert protocol.simulations_to_success(_trace((100, 0.9, 0.0)), ['a']) is None


def _problem(pid='Shahrezaei_PNAS2008'):
    return next(p for p in PROBLEMS if p.id == pid)


def _errors(problem, value):
    return {name: value for name in problem.names}


def test_score_fit_reports_cost_to_success_only_for_a_success():
    problem = _problem()
    truth = problem.truth
    near = {k: v * 1.1 for k, v in truth.items()}
    far = dict(truth)
    far[problem.identifiable[0]] *= 5
    trace = [(200, _errors(problem, 1.0)), (600, _errors(problem, 0.2)), (1000, _errors(problem, 0.7))]
    ok = protocol.score_fit(problem, near, 1500, trace + [(1500, _errors(problem, 0.04))], method='m', seed=1)
    assert ok['success_loose'] and ok['success_tight']
    assert ok['simulations_to_success'] == 600 and ok['first_within_loose'] == 600
    assert ok['simulations'] == 1500 and ok['method'] == 'm'
    bad = protocol.score_fit(problem, far, 1500, trace + [(1500, _errors(problem, 0.7))], method='m', seed=2)
    assert not bad['success_loose']
    assert bad['simulations_to_success'] is None and bad['first_within_loose'] == 600
    json.dumps([ok, bad])   # records must be serializable as written


def test_rescore_applies_revised_identifiability_flags():
    from dataclasses import replace
    problem = _problem()
    weak = problem.identifiable[-1]
    estimate = dict(problem.truth)
    estimate[weak] *= 10          # only the last parameter is off
    trace = [(300, protocol.log10_errors(estimate, problem.truth))]
    rec = protocol.score_fit(problem, estimate, 300, trace, method='m', seed=1)
    assert not rec['success_loose'] and rec['max_error'] == pytest.approx(1.0)
    revised = replace(problem, parameters=tuple(
        replace(p, identifiable=(p.name != weak)) for p in problem.parameters))
    protocol.rescore([rec], [revised])
    assert rec['success_loose'] and rec['max_error'] == pytest.approx(0.0)
    assert rec['simulations_to_success'] == 300


def test_aggregate_rates_and_medians():
    problem = _problem()
    truth = problem.truth
    recs = [
        protocol.score_fit(problem, {k: v * 1.05 for k, v in truth.items()}, 1000,
                           [(400, _errors(problem, 0.02)), (1000, _errors(problem, 0.02))],
                           method='m', seed=1, wall_time=10.0),
        protocol.score_fit(problem, {k: v * 1.5 for k, v in truth.items()}, 1000,
                           [(800, _errors(problem, 0.18)), (1000, _errors(problem, 0.18))],
                           method='m', seed=2, wall_time=20.0),
        protocol.score_fit(problem, {k: v * 4 for k, v in truth.items()}, 1200,
                           [(1200, _errors(problem, 0.6))],
                           method='m', seed=3, wall_time=30.0),
    ]
    rows = protocol.aggregate(recs)
    assert len(rows) == 1
    row = rows[0]
    assert row['n_seeds'] == 3
    assert row['success_loose'] == pytest.approx(2 / 3)
    assert row['success_tight'] == pytest.approx(1 / 3)
    assert row['median_max_error'] == pytest.approx(math.log10(1.5))
    assert row['median_simulations_to_success'] == 600     # median of 400 and 800; the failure is excluded
    assert row['mean_simulations'] == pytest.approx(3200 / 3)
    assert row['mean_wall_time'] == pytest.approx(20.0)
    table = protocol.format_table(rows)
    assert table.count('\n') == 2 and problem.id in table and '67%' in table


# --------------------------------------------------------------------------- #
# The backend path (recovery tier)
# --------------------------------------------------------------------------- #
@pytest.fixture
def harness():
    from .recovery_harness import require_bng2pl
    require_bng2pl()
    from stochastic_recovery import harness as h
    return h


@pytest.mark.recovery
@pytest.mark.bngsim
@pytest.mark.parametrize('problem', [_param(p) for p in PROBLEMS])
def test_identifiable_parameters_have_leverage_on_the_objective(problem, harness, tmp_path):
    """The frozen objective at the truth is small, and moving any identifiable
    parameter by a factor of four in one direction or the other makes it worse. This
    is what makes recovering the parameter possible at all; a parameter with no
    leverage would be a parameter the definition should not mark identifiable."""
    conf = harness.make_config(problem, tmp_path, fit_type='de', seed=3)
    alg = harness.build(conf)
    truth = problem.truth
    at_truth = harness.objective_at(alg, truth)
    n_points = (problem.n_steps + 1) * len(problem.observables)
    assert math.isfinite(at_truth) and at_truth < n_points
    for name in problem.identifiable:
        worse = []
        for factor in (0.25, 4.0):
            values = dict(truth)
            values[name] = truth[name] * factor
            worse.append(harness.objective_at(alg, values))
        assert max(worse) > at_truth, '%s: %s vs %s at the truth' % (name, worse, at_truth)


@pytest.mark.recovery
@pytest.mark.bngsim
def test_data_generation_is_deterministic(harness, tmp_path):
    problem = _problem('Shahrezaei_PNAS2008')
    a = harness.generate_data(problem, workdir=tmp_path / 'a', out_path=tmp_path / 'a.exp')
    b = harness.generate_data(problem, workdir=tmp_path / 'b', out_path=tmp_path / 'b.exp')
    assert Path(a).read_text() == Path(b).read_text()
    if Path(a).read_text() != problem.data_path.read_text():
        # The committed file is the benchmark; the simulator's random stream is not
        # part of the contract, so drift here is worth knowing about, not a failure.
        warnings.warn('%s: regenerated data differ from the committed file; the simulator\'s '
                      'random stream has changed since the data were frozen' % problem.id)


@pytest.mark.recovery
@pytest.mark.bngsim
def test_budgeted_fit_produces_a_complete_record(harness, tmp_path):
    problem = _problem('Shahrezaei_PNAS2008')
    budget = 200
    rec = harness.run_fit(problem, 'de', seed=11, workdir=tmp_path, budget=budget, keep_workdir=True)
    assert rec['problem'] == problem.id and rec['method'] == 'de' and rec['seed'] == 11
    assert set(rec['estimate']) == set(problem.names)
    assert set(rec['errors']) == set(problem.names)
    # The search stops within one evaluation of the budget ...
    assert budget <= rec['simulations_search'] <= budget + problem.smoothing
    # ... and the confirmation stage (candidates x replicates x smoothing) plus the
    # information-criteria replicates account for everything after it.
    confirmation = harness.BEST_FIT_CANDIDATES * harness.BEST_FIT_REPLICATES * problem.smoothing
    assert rec['simulations_search'] < rec['simulations'] <= (rec['simulations_search'] + confirmation
                                                              + harness.BEST_FIT_REPLICATES)
    assert rec['trace'] and rec['trace'][-1][0] == rec['simulations']
    assert all(a <= b for (a, _), (b, _) in zip(rec['trace'], rec['trace'][1:]))
    assert all(set(errs) == set(problem.names) for _, errs in rec['trace'])
    assert rec['trace'][-1][1] == rec['errors']
    assert 'budget reached' in rec['stop_reason']
    assert isinstance(rec['success_loose'], bool) and rec['wall_time'] > 0
    json.dumps(rec)


@pytest.mark.recovery
@pytest.mark.bngsim
def test_method_variant_overrides_are_recorded(harness, tmp_path):
    """A variant of a baseline method (a conf key laid over it) runs under its own name and
    the record says what ran, so a results file never holds two different fits under one
    method name."""
    problem = _problem('Shahrezaei_PNAS2008')
    rec = harness.run_fit(problem, 'ss_noise', seed=11, workdir=tmp_path, budget=200,
                          overrides={'ss_noise_max_draws': 2}, label='ss_noise_d2')
    assert rec['method'] == 'ss_noise_d2'
    assert rec['base_method'] == 'ss_noise'
    assert rec['overrides'] == {'ss_noise_max_draws': 2}
    plain = harness.run_fit(problem, 'ss_noise', seed=11, workdir=tmp_path / 'plain', budget=200)
    assert plain['method'] == 'ss_noise' and plain['overrides'] == {}
