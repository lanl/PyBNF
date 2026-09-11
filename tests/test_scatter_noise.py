"""Scatter search's noise-aware reference set (#660 step 3, ADR-0136).

Scatter search decides everything by ranking, and for a stochastic model each objective
value is one draw. The reference set stored that draw as fact, so a member whose value was
a lucky draw could never be beaten by an honest child and was retired into the archive of
local minima as one it never was. Now every member is ranked on the mean of its draws, the
draw-to-draw spread is pooled across the fit, and a decision the spread leaves in doubt is
not made until both sides have been drawn again.

Three layers: the pooled spread and the separation test on their own; the reference set
driven by hand with a scorer whose noise is keyed by the replicate offset, so a test states
exactly what every draw of every parameter set is worth; and the whole run loop through the
integration harness with a noisy fake runner.
"""
import os
import pickle

import numpy as np
import numpy.testing as npt
import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.algorithms.noise_handling import pooled_sd, separated
from pybnf.pset import PSet

MU = np.array([2.0, -1.0])


def _conf(tmp_path, **overrides):
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(list(MU), [1.0, 1.0]))
    base = dict(n_params=2, population_size=4, max_iterations=60, init_size=8,
                reserve_size=20, random_seed=11)
    base.update(overrides)
    conf = H.make_config(tmp_path, 'ss', tgt, exp, **base)
    # The hand-driven tests never run main(), which makes these; output_results writes here.
    os.makedirs(os.path.join(conf.config['output_dir'], 'Results'), exist_ok=True)
    os.makedirs(os.path.join(conf.config['output_dir'], 'Simulations'), exist_ok=True)
    return conf


def _stochastic(conf):
    for model in conf.models.values():
        model.stochastic = True
    return conf


class _Res:
    def __init__(self, pset, score):
        self.pset, self.score, self.name = pset, score, pset.name


def _key(pset):
    return (round(float(pset['p1']), 9), round(float(pset['p2']), 9))


class _Scorer:
    """``truth(x)`` plus Gaussian noise keyed by the parameter values and the replicate
    offset, so the same draw comes back for the same offset and a different offset is a
    fresh one. ``lucky`` overrides particular (values, offset) draws outright."""

    def __init__(self, sd, truth=None, lucky=None):
        self.sd = sd
        self.truth = truth or (lambda x: float(np.sum((x - MU) ** 2)))
        self.lucky = dict(lucky or {})
        self.calls = []

    def __call__(self, pset):
        offset = int(getattr(pset, 'replicate_offset', 0))
        self.calls.append((pset.name, offset))
        key = (_key(pset), offset)
        if key in self.lucky:
            return self.lucky[key]
        x = np.array([pset['p1'], pset['p2']])
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        return self.truth(x) + self.sd * rng.standard_normal()


def _feed(alg, psets, scorer):
    response = []
    for p in psets:
        r = alg.got_result(_Res(p, scorer(p)))
        if r == 'STOP':
            return 'STOP'
        response.extend(r)
    return response


def _drive(alg, scorer, iterations):
    batch = alg.start_run()
    while alg.iteration < iterations and batch != 'STOP':
        batch = _feed(alg, batch, scorer)
    return batch


# --------------------------------------------------------------------------- #
# The pooled spread and the separation test
# --------------------------------------------------------------------------- #
class TestNoiseArithmetic:

    def test_pooled_sd_pools_within_set_variance_over_every_set_drawn_twice(self):
        sd = pooled_sd([[1.0, 3.0], [10.0, 10.0, 13.0], [5.0], []])
        # Sums of squares 2 and 6 over 1 + 2 degrees of freedom.
        npt.assert_allclose(sd, np.sqrt(8.0 / 3.0))

    def test_pooled_sd_ignores_failed_draws_and_is_none_without_a_repeat(self):
        assert pooled_sd([[1.0, np.inf, 3.0]]) == pytest.approx(np.sqrt(2.0))
        assert pooled_sd([[1.0], [2.0]]) is None

    def test_separated_is_one_standard_error_of_the_difference(self):
        assert separated(0.0, 1, 3.0, 1, sd=2.0)          # 3 > 2*sqrt(2)
        assert not separated(0.0, 1, 2.0, 1, sd=2.0)      # 2 < 2*sqrt(2)
        assert separated(0.0, 4, 2.0, 4, sd=2.0)          # 2 > 2*sqrt(1/2)

    def test_separated_edge_cases(self):
        assert separated(np.inf, 1, 1.0, 1, sd=2.0)       # a failed set is worse than anything
        assert not separated(0.0, 1, 1.0, 1, sd=None)     # no spread known yet: in doubt
        assert separated(0.0, 1, 1e-9, 1, sd=0.0)         # deterministic: any difference orders
        assert not separated(0.0, 0, 1.0, 1, sd=1.0)      # nothing drawn: in doubt


# --------------------------------------------------------------------------- #
# The reference set, driven by hand
# --------------------------------------------------------------------------- #
class TestReferenceSet:

    @pytest.fixture(autouse=True)
    def _no_periodic_output(self, monkeypatch):
        """The tests here feed results to the search directly, never through the run loop
        that fills the trajectory, so the periodic sorted_params write has nothing to write."""
        monkeypatch.setattr(algorithms.ScatterSearch, 'output_results', lambda self, *a, **k: None)

    def test_a_deterministic_fit_queues_no_re_draw_and_keeps_single_draws(self, tmp_path):
        alg = algorithms.ScatterSearch(_conf(tmp_path))
        assert alg.noise_handling is False
        scorer = _Scorer(0.0)
        batch = _drive(alg, scorer, iterations=3)
        assert all(offset == 0 for _, offset in scorer.calls)
        assert not any('_d' in name for name, _ in scorer.calls)
        assert all(len(d) == 1 for d in alg.draws.values())
        assert alg.pending_draws == {} and alg.contenders == {}
        assert len(batch) == alg.popsize * (alg.popsize - 1)

    def test_the_switch_turns_it_off_for_a_stochastic_fit(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, ss_noise_handling=0)))
        assert alg.noise_handling is False

    def test_a_stochastic_fit_bootstraps_the_spread_with_one_re_draw_of_every_member(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        assert alg.noise_handling is True and alg.max_draws == 3
        scorer = _Scorer(0.5)
        init = alg.start_run()
        first = _feed(alg, init, scorer)
        redraws = [p for p in first if '_d' in p.name]
        children = [p for p in first if '_d' not in p.name]
        assert len(children) == alg.popsize * (alg.popsize - 1)
        assert len(redraws) == alg.popsize
        assert all(p.name.endswith('_d1') and p.replicate_offset == 1 for p in redraws)
        assert alg._noise_sd() is None
        _feed(alg, first, scorer)
        assert alg._noise_sd() is not None and alg._noise_sd() > 0.0
        # Every member of the first reference set was drawn twice, and the spread keeps
        # those draws even where a child has since replaced the member.
        assert sum(len(d) >= 2 for d in alg.repeat_draws.values()) == alg.popsize

    def _lucky_setup(self, tmp_path, handling):
        """A reference set whose best member got a lucky first draw of 10 while its true
        value is 30; every other parameter set is worth 20 plus noise of spread 3."""
        conf = _stochastic(_conf(tmp_path, ss_noise_handling=handling, local_min_limit=3,
                                 max_iterations=100))
        alg = algorithms.ScatterSearch(conf)
        init = alg.start_run()
        lucky = init[0]

        def truth(x):
            return 30.0 if tuple(np.round(x, 9)) == _key(lucky) else 20.0

        scorer = _Scorer(3.0, truth=truth, lucky={(_key(lucky), 0): 10.0})
        return alg, init, lucky, scorer

    def test_with_the_handling_off_the_lucky_member_is_archived_at_its_lucky_value(self, tmp_path):
        alg, init, lucky, scorer = self._lucky_setup(tmp_path, handling=0)
        batch = _feed(alg, init, scorer)
        assert alg.refs[0][0] == lucky and alg.refs[0][1] == 10.0
        while alg.iteration < 12 and batch != 'STOP':
            batch = _feed(alg, batch, scorer)
        assert any(m == lucky and s == 10.0 for m, s in alg.local_mins)

    def test_with_the_handling_on_the_lucky_member_is_drawn_again_and_never_archived_lucky(self, tmp_path):
        alg, init, lucky, scorer = self._lucky_setup(tmp_path, handling=1)
        batch = _feed(alg, init, scorer)
        assert alg.refs[0][0] == lucky
        while alg.iteration < 12 and batch != 'STOP':
            batch = _feed(alg, batch, scorer)
        # It was drawn again at fresh offsets, its estimate regressed towards 30, and it
        # is no longer the best member; wherever it ended up, it was not archived at 10.
        lucky_draws = alg.draws.get(lucky)
        drawn = [offset for name, offset in scorer.calls if name.startswith(lucky.name)]
        assert max(drawn) >= 1
        if lucky_draws:
            assert np.mean(lucky_draws) > 15.0
        assert not any(m == lucky and s <= 12.0 for m, s in alg.local_mins)
        assert not any(m == lucky and s <= 12.0 for m, s in alg.refs)

    def test_a_contest_in_doubt_is_kept_open_and_settled_by_re_draws(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        scorer = _Scorer(1.0)
        _feed(alg, alg.start_run(), scorer)
        # Force a known spread and a parent with a contender-worthy child: make the round
        # by hand from the algorithm's own state.
        parent = alg.refs[0][0]
        alg.draws[parent] = [5.0, 5.4]
        # The other members sit at the cap, so a stuck count cannot re-draw them and the
        # re-draws below belong to the contest alone.
        for m, _ in alg.refs[1:]:
            alg.draws[m] = [50.0, 50.2, 50.1, 50.3, 50.2]
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        alg.repeat_draws = dict(alg.draws)
        alg.stuckcounter = {m: 0 for m, _ in alg.refs}
        alg.contenders = {}
        alg.pending_draws = {}
        alg.pending = {}
        # A child of the parent at 5.1: better than 5.2 but inside the noise.
        child = PSet([parent.get_param(v.name).add(0.01, reflect=False)
                                 for v in alg.variables])
        child.name = 'child'
        alg.received = {m: [] for m, _ in alg.refs}
        alg.received[parent] = [(child, 5.1)]
        redraws = alg._update_reference_set()
        assert alg.contenders[parent] == child
        assert alg.refs[0][0] == parent                    # not replaced yet
        assert alg.stuckcounter[parent] == 0               # not counted stuck either
        names = sorted(p.name for p in redraws)
        assert names == ['child_d1', parent.name + '_d2']
        assert {p.replicate_offset for p in redraws} == {1, 2}

    def test_a_contest_at_the_cap_is_decided_on_the_means(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, ss_noise_max_draws=2)))
        scorer = _Scorer(1.0)
        _feed(alg, alg.start_run(), scorer)
        parent = alg.refs[0][0]
        alg.draws[parent] = [5.0, 5.4]
        for m, _ in alg.refs[1:]:
            alg.draws[m] = [50.0, 50.2]
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        alg.repeat_draws = dict(alg.draws)
        alg.stuckcounter = {m: 0 for m, _ in alg.refs}
        alg.pending, alg.pending_draws = {}, {}
        child = PSet([parent.get_param(v.name).add(0.01, reflect=False)
                                 for v in alg.variables])
        child.name = 'child'
        alg.draws[child] = [5.1, 5.15]                     # the contender, at the cap too
        alg.contenders = {parent: child}
        alg.received = {m: [] for m, _ in alg.refs}
        alg._update_reference_set()
        assert alg.refs[0][0] == child                     # 5.125 < 5.2 decides
        assert alg.draws[child] == [5.1, 5.15]
        assert parent not in alg.draws

    def test_a_stuck_member_is_drawn_again_up_to_the_cap(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, ss_noise_max_draws=3)))
        scorer = _Scorer(1.0)
        _feed(alg, alg.start_run(), scorer)
        parent = alg.refs[0][0]
        alg.draws[parent] = [5.0, 5.4]
        for m, _ in alg.refs[1:]:
            alg.draws[m] = [50.0, 50.2]
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        alg.repeat_draws = dict(alg.draws)
        alg.stuckcounter = {m: 0 for m, _ in alg.refs}
        alg.pending, alg.pending_draws, alg.contenders = {}, {}, {}
        worse = PSet([parent.get_param(v.name).add(0.01, reflect=False)
                                 for v in alg.variables])
        worse.name = 'worse'
        alg.received = {m: [] for m, _ in alg.refs}
        alg.received[parent] = [(worse, 40.0)]              # clearly worse
        redraws = alg._update_reference_set()
        assert alg.stuckcounter[parent] == 1
        assert [p.name for p in redraws if p.name.startswith(parent.name)] == [parent.name + '_d2']
        alg.draws[parent].append(5.2)                       # now at the cap of 3
        alg.pending_draws = {}
        alg.received = {m: [] for m, _ in alg.refs}
        alg.received[parent] = [(worse, 40.0)]
        redraws = alg._update_reference_set()
        assert alg.stuckcounter[parent] == 2
        assert not any(p.name.startswith(parent.name) for p in redraws)

    def test_unordered_neighbours_are_drawn_again_and_ordered_ones_are_not(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        scorer = _Scorer(1.0)
        _feed(alg, alg.start_run(), scorer)
        members = [m for m, _ in alg.refs]
        alg.draws = {members[0]: [1.0, 1.2], members[1]: [1.1, 1.3],
                     members[2]: [20.0, 20.1], members[3]: [40.0, 40.1]}
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m in members]
        alg.repeat_draws = dict(alg.draws)
        wanted = alg._unseparated_neighbours()
        assert wanted == [members[0], members[1]]
        alg.pending_draws = {}
        queued = alg._redraws(wanted)
        assert sorted(p.name for p in queued) == sorted(m.name + '_d2' for m in members[:2])
        assert alg._redraws(wanted) == []                  # not queued twice in a round

    def test_a_failed_draw_makes_a_member_infinite_and_beatable(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        _feed(alg, alg.start_run(), _Scorer(1.0))
        assert alg._estimate_draws([1.0, np.inf]) == (np.inf, 2)
        assert alg._estimate_draws([]) == (np.inf, 0)

    def test_the_state_pickles_and_resets_between_starts(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        scorer = _Scorer(1.0)
        first = _feed(alg, alg.start_run(), scorer)
        for p in first[:5]:
            alg.got_result(_Res(p, scorer(p)))
        clone = pickle.loads(pickle.dumps(alg))
        assert clone.pending_draws.keys() == alg.pending_draws.keys()
        alg._reset_search_state()
        assert alg.draws == {} and alg.contenders == {} and alg.pending_draws == {}


# --------------------------------------------------------------------------- #
# The whole run loop, with a noisy fake runner
# --------------------------------------------------------------------------- #
def test_a_stochastic_fit_runs_end_to_end_and_re_draws_at_fresh_indices(tmp_path, monkeypatch):
    H.install(monkeypatch)
    conf = _stochastic(_conf(tmp_path, population_size=6, max_iterations=25, init_size=20))
    seen = []

    def noisy_run_job(j, debug=False, failed_logs_dir=''):
        res = H.slim_run_job(j, debug, failed_logs_dir)
        key = (_key(j.params), int(j.replicate_index))
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        res.score = float(res.score) + 0.5 * rng.standard_normal()
        seen.append(int(j.replicate_index))
        return res

    monkeypatch.setattr(algorithms.core, 'run_job', noisy_run_job)
    alg = algorithms.ScatterSearch(conf)
    assert alg.noise_handling
    H.drive(alg)
    assert any(index > 0 for index in seen)
    assert alg._noise_sd() is not None
    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, MU, atol=0.75), recovered
