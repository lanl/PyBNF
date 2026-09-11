"""Scatter search's improvement method (#660 step 1, ADR-0138).

Glover's template refines the candidates combination produces with a local search; PyBNF's
scatter search relied on recombination alone. Now the best child a round accepts is refined
by a Nelder-Mead simplex (the ``SimplexRunner`` the concurrent multi-start already drives),
run asynchronously alongside the rounds and folded into the reference set when it finishes,
under Egea's filters: a cadence, a concurrency cap, and a distance filter against
refinements already made. Off under noise handling, since a simplex over single draws of a
stochastic model converges on the noise.
"""
import os
import pickle

import numpy as np
import numpy.testing as npt
import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.algorithms.optimizers.scatter_search import _LOCAL_MIN_DISTANCE
from pybnf.pset import PSet, FreeParameter

MU = np.array([2.0, -1.0])


def _conf(tmp_path, **overrides):
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(list(MU), [1.0, 1.0]))
    base = dict(n_params=2, population_size=4, max_iterations=60, init_size=8,
                reserve_size=10, random_seed=5)
    base.update(overrides)
    conf = H.make_config(tmp_path, 'ss', tgt, exp, **base)
    os.makedirs(os.path.join(conf.config['output_dir'], 'Results'), exist_ok=True)
    os.makedirs(os.path.join(conf.config['output_dir'], 'Simulations'), exist_ok=True)
    return conf


def _on(tmp_path, **overrides):
    overrides.setdefault('ss_local_search', 1)
    overrides.setdefault('ss_local_every', 1)
    return algorithms.ScatterSearch(_conf(tmp_path, **overrides))


def _pset(alg, values, name=None):
    p = PSet([FreeParameter(v.name, 'uniform_var', -10.0, 10.0, float(x))
              for v, x in zip(alg.variables, values)])
    p.name = name
    return p


class _Res:
    def __init__(self, pset, score):
        self.pset, self.score, self.name = pset, score, pset.name


def _quadratic(pset):
    return float(np.sum((np.array([pset['p1'], pset['p2']]) - MU) ** 2))


def _feed(alg, psets, scorer=_quadratic):
    response = []
    for p in psets:
        r = alg.got_result(_Res(p, scorer(p)))
        if r == 'STOP':
            return 'STOP'
        response.extend(r)
    return response


def _no_output(*args, **kwargs):
    """Stands in for the periodic sorted_params write, which the hand-driven tests never
    fill a trajectory for; a module-level function so the algorithm still pickles."""
    return None


def _primed(alg):
    """A scatter search past its initialization round, with a real reference set and the
    initial spread recorded, and no refinement started yet."""
    alg.output_results = _no_output
    first_round = _feed(alg, alg.start_run())
    assert alg.local_runners == {}
    return first_round


class TestTheGate:

    def test_off_under_the_legacy_edition_and_on_under_a_modern_one(self, tmp_path):
        alg = algorithms.ScatterSearch(_conf(tmp_path))
        assert alg.local_search is False
        alg.config.config['edition'] = 2
        assert alg._resolve_local_search() is True

    def test_an_explicit_setting_wins(self, tmp_path):
        assert _on(tmp_path).local_search is True
        alg = algorithms.ScatterSearch(_conf(tmp_path, ss_local_search=0))
        alg.config.config['edition'] = 2
        assert alg._resolve_local_search() is False

    def test_off_under_noise_handling_even_when_asked_for(self, tmp_path):
        conf = _conf(tmp_path, ss_local_search=1)
        for model in conf.models.values():
            model.stochastic = True
        alg = algorithms.ScatterSearch(conf)
        assert alg.noise_handling is True and alg.local_search is False

    def test_the_knobs_are_read(self, tmp_path):
        alg = _on(tmp_path, ss_local_every=3, ss_local_max_iterations=7, ss_local_max_running=2)
        assert (alg.local_every, alg.local_max_iterations, alg.local_max_running) == (3, 7, 2)


class TestStarting:

    def test_nothing_starts_before_a_round_accepts_a_child(self, tmp_path):
        alg = _on(tmp_path)
        first_round = _primed(alg)
        assert not any(p.name.startswith('ls') for p in first_round)
        assert alg.last_local_iteration is None

    def test_a_refinement_starts_from_the_best_accepted_child(self, tmp_path):
        alg = _on(tmp_path)
        _primed(alg)
        worse = _pset(alg, [3.0, 3.0], 'childA')
        best = _pset(alg, [1.0, 1.0], 'childB')
        alg._round_accepted = [(worse, 5.0), (best, 3.0)]
        psets = alg._maybe_start_local_search()
        # The n + 1 vertices of the initial simplex, tagged for the refinement.
        assert len(psets) == len(alg.variables) + 1
        assert all(p.name.startswith('ls0_simplex_init') for p in psets)
        assert alg.local_origins['ls0'] is best
        assert set(alg.pending_local) == {p.name for p in psets}
        assert psets[0]['p1'] == 1.0 and psets[0]['p2'] == 1.0
        assert alg.last_local_iteration == alg.iteration
        assert len(alg.local_starts) == 1

    def test_the_initial_simplex_is_a_tenth_of_the_reference_sets_spread(self, tmp_path):
        alg = _on(tmp_path)
        _primed(alg)
        members = [m for m, _ in alg.refs]
        u = np.array([alg._param_vec(m) for m in members])
        spread = u.max(axis=0) - u.min(axis=0)
        steps = alg._local_steps()
        for v, s in zip(alg.variables, spread):
            npt.assert_allclose(steps[v.name], 0.1 * (s if s > 0 else 1.0))

    def test_the_cadence_filter(self, tmp_path):
        alg = _on(tmp_path, ss_local_every=3, ss_local_max_running=5)
        _primed(alg)
        alg.iteration = 4
        alg._round_accepted = [(_pset(alg, [1.0, 1.0], 'c1'), 3.0)]
        assert alg._maybe_start_local_search()
        alg.iteration = 6
        alg._round_accepted = [(_pset(alg, [-5.0, -5.0], 'c2'), 3.0)]
        assert alg._maybe_start_local_search() == []          # two rounds later: too soon
        alg.iteration = 7
        assert alg._maybe_start_local_search()                # three rounds later: allowed

    def test_the_concurrency_filter(self, tmp_path):
        alg = _on(tmp_path, ss_local_max_running=1)
        _primed(alg)
        alg.iteration = 4
        alg._round_accepted = [(_pset(alg, [1.0, 1.0], 'c1'), 3.0)]
        assert alg._maybe_start_local_search()
        alg.iteration = 9
        alg._round_accepted = [(_pset(alg, [-5.0, -5.0], 'c2'), 3.0)]
        assert alg._maybe_start_local_search() == []          # one still running

    def test_the_distance_filter(self, tmp_path):
        alg = _on(tmp_path, ss_local_max_running=5)
        _primed(alg)
        alg._init_spread = np.array([20.0, 20.0])
        alg.iteration = 4
        alg._round_accepted = [(_pset(alg, [1.0, 1.0], 'c1'), 3.0)]
        assert alg._maybe_start_local_search()
        alg.iteration = 9
        near = _pset(alg, [1.0 + 0.5 * _LOCAL_MIN_DISTANCE * 20.0, 1.0], 'near')
        alg._round_accepted = [(near, 2.0)]
        assert alg._maybe_start_local_search() == []          # inside a refined basin
        alg._round_accepted = [(_pset(alg, [-8.0, 8.0], 'far'), 2.0)]
        assert alg._maybe_start_local_search()

    def test_a_failed_child_is_not_refined(self, tmp_path):
        alg = _on(tmp_path)
        _primed(alg)
        alg.iteration = 4
        alg._round_accepted = [(_pset(alg, [1.0, 1.0], 'c1'), np.inf)]
        assert alg._maybe_start_local_search() == []


class TestRunningAndFolding:

    def _started(self, tmp_path, **overrides):
        alg = _on(tmp_path, **overrides)
        _primed(alg)
        alg.iteration = 4
        origin = _pset(alg, [1.0, 1.0], 'origin')
        # Put the origin into the reference set so the fold can find it.
        worst = max(range(len(alg.refs)), key=lambda i: alg.refs[i][1])
        alg._replace_member(worst, origin, 3.0)
        alg._round_accepted = [(origin, 3.0)]
        vertices = alg._maybe_start_local_search()
        return alg, origin, vertices

    def test_results_route_to_the_refinement_and_never_end_a_round(self, tmp_path):
        alg, origin, vertices = self._started(tmp_path)
        pending_before = dict(alg.pending)
        out = []
        for p in vertices:
            out.extend(alg.got_result(_Res(p, _quadratic(p))))
        # The refinement advanced (its reflections came back, tagged) and the round's own
        # pending children are untouched.
        assert out and all(p.name.startswith('ls0_simplex_iter') for p in out)
        assert set(alg.pending_local) == {p.name for p in out}
        assert alg.pending == pending_before

    # The best vertex is the second one (the start moved along p1), so the refined point
    # differs from its start by value and the fold's placement is visible.
    @pytest.mark.parametrize('vertex_scores, expect', [
        ([9.0, 2.0, 9.0], 'origin'),      # better than its start: replaces it
        ([9.0, 4.0, 9.0], 'worst'),       # worse than its start, better than the worst member
        ([900.0, 500.0, 900.0], 'archive'),
    ])
    def test_a_finished_refinement_is_folded_in(self, tmp_path, vertex_scores, expect):
        alg, origin, vertices = self._started(tmp_path, ss_local_max_iterations=1)
        worst_before = max(alg.refs, key=lambda x: x[1])
        assert worst_before[1] < 500.0
        for p, s in zip(vertices, vertex_scores):
            assert alg.got_result(_Res(p, s)) == []
        assert 'ls0' in alg.local_finished and 'ls0' not in alg.local_runners
        alg._fold_finished_local_searches()
        assert alg.local_finished == {} and len(alg.local_optima) == 1
        refined = alg.local_optima[0]
        assert refined[1] == min(vertex_scores) and refined[0].name == 'ls0_best'
        members = [m for m, _ in alg.refs]
        if expect == 'origin':
            assert origin not in members and refined[0] in members
            assert alg.draws[refined[0]] == [2.0] and alg.stuckcounter[refined[0]] == 0
        elif expect == 'worst':
            assert origin in members and refined[0] in members
            assert worst_before[0] not in members
        else:
            assert refined[0] not in members
            assert (refined[0], 500.0) in alg.local_mins

    def test_a_straggler_of_a_folded_refinement_is_dropped(self, tmp_path):
        alg, origin, vertices = self._started(tmp_path, ss_local_max_iterations=1)
        for p in vertices:
            alg.got_result(_Res(p, 5.0))
        alg._fold_finished_local_searches()
        late = _pset(alg, [1.0, 1.0], 'ls0_simplex_iter0_pt0')
        alg.pending_local[late.name] = 'ls0'
        assert alg.got_result(_Res(late, 1.0)) == []
        assert late.name not in alg.pending_local

    def test_the_state_pickles_and_resets(self, tmp_path):
        alg, origin, vertices = self._started(tmp_path)
        clone = pickle.loads(pickle.dumps(alg))
        assert set(clone.local_runners) == {'ls0'} and set(clone.pending_local) == set(alg.pending_local)
        alg._reset_search_state()
        assert alg.local_runners == {} and alg.pending_local == {} and alg.local_optima == []
        assert alg.last_local_iteration is None and alg._init_spread is None


def test_a_scatter_search_with_the_improvement_method_finds_the_mode(tmp_path, monkeypatch):
    H.install(monkeypatch)
    conf = _conf(tmp_path, ss_local_search=1, ss_local_every=1, ss_local_max_iterations=5,
                 population_size=6, init_size=20, max_iterations=15)
    seen = []
    real = H.slim_run_job

    def run_job(j, debug=False, failed_logs_dir=''):
        seen.append(j.job_id)
        return real(j, debug, failed_logs_dir)

    monkeypatch.setattr(algorithms.core, 'run_job', run_job)
    alg = algorithms.ScatterSearch(conf)
    assert alg.local_search
    H.drive(alg)
    assert any(name.startswith('ls') for name in seen)
    assert alg.local_optima or alg.local_runners or alg.local_finished
    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, MU, atol=0.25), recovered
