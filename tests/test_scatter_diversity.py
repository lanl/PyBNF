"""The diverse half of scatter search's first reference set is chosen by distance
(#660 step 2, ADR-0137).

Glover's template fills the second half of the first reference set with the members most
distant from what is already in it. The pre-#660 code filled it at random, which is
diverse only on average. The rule is on under edition 2 and off under the legacy edition,
whose contract is that an unchanged conf keeps behaving as it always has. The integration
harness writes legacy-syntax confs, so the tests here turn the rule on with the explicit
key and check the edition resolution on its own.
"""
import numpy as np

from . import integration_harness as H
from .context import algorithms
from pybnf.pset import PSet, FreeParameter

MU = np.array([2.0, -1.0])


def _conf(tmp_path, **overrides):
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(list(MU), [1.0, 1.0]))
    base = dict(n_params=2, population_size=4, max_iterations=30, init_size=8,
                reserve_size=10, random_seed=3)
    base.update(overrides)
    return H.make_config(tmp_path, 'ss', tgt, exp, **base)


def _pset(alg, values):
    return PSet([FreeParameter(v.name, 'uniform_var', -1000.0, 1000.0, float(x))
                 for v, x in zip(alg.variables, values)])


class _Rng:
    """Delegates to a real generator, except that ``choice`` is recorded or refused: a
    numpy generator's methods are read-only, so they cannot be patched in place."""

    def __init__(self, real, allow_choice):
        self._real, self._allow, self.choices = real, allow_choice, 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def choice(self, *args, **kwargs):
        if not self._allow:
            raise AssertionError('rng.choice must not be called under the distance rule')
        self.choices += 1
        return self._real.choice(*args, **kwargs)


def _on(tmp_path, **overrides):
    return algorithms.ScatterSearch(_conf(tmp_path, ss_diverse_by_distance=1, **overrides))


class TestTheRule:

    def test_each_pick_is_the_candidate_farthest_from_the_set_so_far(self, tmp_path):
        alg = _on(tmp_path)
        alg.refs = [(_pset(alg, [0.0, 0.0]), 1.0)]
        near = (_pset(alg, [0.1, 0.1]), 2.0)          # best score, but next to the set
        far_a = (_pset(alg, [5.0, 5.0]), 3.0)
        far_b = (_pset(alg, [-5.0, 5.0]), 4.0)
        close = (_pset(alg, [0.2, 0.0]), 5.0)
        chosen = alg._most_diverse([near, far_a, far_b, close], 2)
        assert chosen == [far_a, far_b]

    def test_ties_go_to_the_better_score(self, tmp_path):
        alg = _on(tmp_path)
        alg.refs = [(_pset(alg, [0.0, 0.0]), 1.0)]
        east = (_pset(alg, [4.0, 0.0]), 2.0)
        west = (_pset(alg, [-4.0, 0.0]), 3.0)
        assert alg._most_diverse([east, west], 1) == [east]

    def test_coordinates_are_measured_against_their_spread(self, tmp_path):
        """A parameter whose values span two hundred units must not drown one that spans
        one. The set holds (-100, -0.1) and (0, 0), so the spreads are 200 and 1; a
        candidate 100 units out along the wide parameter is half a spread from the set,
        one 0.9 out along the narrow parameter is nine tenths of a spread, and the raw
        distance (100 against 0.9) would have picked the other one."""
        alg = _on(tmp_path)
        alg.refs = [(_pset(alg, [-100.0, -0.1]), 1.0), (_pset(alg, [0.0, 0.0]), 1.5)]
        wide = (_pset(alg, [100.0, 0.0]), 2.0)
        narrow = (_pset(alg, [0.0, 0.9]), 3.0)
        assert alg._most_diverse([wide, narrow], 1) == [narrow]

    def test_a_failed_point_is_taken_only_when_nothing_finite_is_left(self, tmp_path):
        alg = _on(tmp_path)
        alg.refs = [(_pset(alg, [0.0, 0.0]), 1.0)]
        failed_far = (_pset(alg, [9.0, 9.0]), np.inf)
        finite_near = (_pset(alg, [1.0, 1.0]), 2.0)
        assert alg._most_diverse([failed_far, finite_near], 1) == [finite_near]
        assert alg._most_diverse([failed_far, finite_near], 2) == [finite_near, failed_far]

    def test_nothing_to_choose(self, tmp_path):
        alg = _on(tmp_path)
        alg.refs = [(_pset(alg, [0.0, 0.0]), 1.0)]
        assert alg._most_diverse([], 2) == []
        assert alg._most_diverse([(_pset(alg, [1.0, 1.0]), 2.0)], 0) == []


class TestTheGate:

    def test_on_under_a_modern_edition_and_off_under_the_legacy_one(self, tmp_path):
        alg = algorithms.ScatterSearch(_conf(tmp_path))
        assert alg.diverse_by_distance is False                  # legacy: as it always was
        alg.config.config['edition'] = 2
        assert alg._resolve_diverse_by_distance() is True
        alg.config.config['edition'] = 1
        assert alg._resolve_diverse_by_distance() is False

    def test_an_explicit_setting_wins_under_either_edition(self, tmp_path):
        alg = _on(tmp_path)
        assert alg.diverse_by_distance is True
        alg.config.config['edition'] = 2
        alg.config.config['ss_diverse_by_distance'] = 0
        assert alg._resolve_diverse_by_distance() is False

    def _first_round(self, alg):
        psets = alg.start_run()
        for i, p in enumerate(psets):
            res = type('R', (), {})()
            res.pset, res.score, res.name = p, float(i), p.name
            alg.got_result(res)
        return psets

    def test_the_distance_rule_never_draws_from_the_rng(self, tmp_path, monkeypatch):
        alg = _on(tmp_path)
        monkeypatch.setattr(alg, 'output_results', lambda *a, **k: None)
        alg.rng = _Rng(alg.rng, allow_choice=False)
        psets = self._first_round(alg)
        top = int(np.ceil(alg.popsize / 2.0))
        assert [r[1] for r in alg.refs[:top]] == [0.0, 1.0]           # best half by score
        assert len(alg.refs) == alg.popsize
        # The diverse half is the greedy max-min choice from the rest.
        chosen = list(alg.refs[top:])
        rest = [(p, float(i)) for i, p in enumerate(psets)][top:]
        alg.refs = list(alg.refs[:top])
        assert alg._most_diverse(rest, alg.popsize - top) == chosen

    def test_the_legacy_first_round_still_draws_at_random(self, tmp_path, monkeypatch):
        alg = algorithms.ScatterSearch(_conf(tmp_path))
        monkeypatch.setattr(alg, 'output_results', lambda *a, **k: None)
        alg.rng = _Rng(alg.rng, allow_choice=True)
        self._first_round(alg)
        assert alg.rng.choices == 1


def test_a_scatter_search_under_the_rule_still_finds_the_gaussian_mode(tmp_path, monkeypatch):
    H.install(monkeypatch)
    conf = _conf(tmp_path, ss_diverse_by_distance=1, population_size=12, max_iterations=20,
                 init_size=40)
    alg = algorithms.ScatterSearch(conf)
    assert alg.diverse_by_distance
    H.drive(alg)
    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, MU, atol=0.25), recovered
