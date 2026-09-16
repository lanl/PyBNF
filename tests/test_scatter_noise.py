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


# --------------------------------------------------------------------------- #
# Paying for the deferral (#696, ADR-0145)
# --------------------------------------------------------------------------- #
def _in_doubt(alg, gaps):
    """A reference set of ``len(gaps)`` members ten apart, each drawn twice, each with one
    child whose single draw is ``gaps[k]`` below its mean. A gap inside the pooled spread
    (about 0.28 here) leaves that slot's contest in doubt, and the gaps rank the contests
    against each other. Returns the children in slot order."""
    members = [m for m, _ in alg.refs]
    for k, m in enumerate(members):
        alg.draws[m] = [10.0 * (k + 1), 10.0 * (k + 1) + 0.4]
    alg.refs = [(m, float(np.mean(alg.draws[m]))) for m in members]
    alg.repeat_draws = dict(alg.draws)
    alg.stuckcounter = {m: 0 for m in members}
    alg.contenders, alg.pending, alg.pending_draws = {}, {}, {}
    alg.received = {m: [] for m in members}
    children = []
    for k, (m, mean) in enumerate(alg.refs):
        child = PSet([m.get_param(v.name).add(0.01 * (k + 1), reflect=False)
                      for v in alg.variables])
        child.name = 'child%d' % k
        alg.received[m] = [(child, mean - gaps[k])]
        children.append(child)
    return children


class TestOptimisticAcceptance:
    """``ss_noise_optimistic``: a candidate that leads on the mean takes the reference
    slot while the draws that would settle its contest continue (#696 lever 1)."""

    @pytest.fixture(autouse=True)
    def _no_periodic_output(self, monkeypatch):
        monkeypatch.setattr(algorithms.ScatterSearch, 'output_results', lambda self, *a, **k: None)

    def _alg(self, tmp_path, **overrides):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, **overrides)))
        _feed(alg, alg.start_run(), _Scorer(1.0))
        return alg

    def test_it_is_off_by_default_and_does_nothing_to_a_deterministic_fit(self, tmp_path):
        assert algorithms.ScatterSearch(_stochastic(_conf(tmp_path))).optimistic is False
        alg = algorithms.ScatterSearch(_conf(tmp_path, ss_noise_optimistic=1))
        assert alg.optimistic is True and alg.noise_handling is False
        batch = _drive(alg, _Scorer(0.0), iterations=3)
        assert alg.contenders == {} and alg.pending_draws == {}
        assert len(batch) == alg.popsize * (alg.popsize - 1)

    def test_a_leading_candidate_takes_the_slot_and_the_parent_becomes_its_contender(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_optimistic=1)
        children = _in_doubt(alg, [0.3, 0.2, 0.1, 0.05])
        parent = alg.refs[0][0]
        redraws = alg._update_reference_set()
        # The child leads on the mean, so it is in the reference set now, and the parent
        # it displaced is its contender with its draws intact.
        assert alg.refs[0][0] == children[0]
        assert alg.refs[0][1] == pytest.approx(10.2 - 0.3)
        assert alg.contenders[children[0]] == parent
        assert alg.draws[parent] == [10.0, 10.4]
        assert alg.stuckcounter[children[0]] == 0 and parent not in alg.stuckcounter
        # Both sides are still being drawn: the contest is running, not decided.
        assert 'child0_d1' in [p.name for p in redraws]
        assert parent.name + '_d2' in [p.name for p in redraws]

    def test_the_slots_stuck_count_starts_again_because_its_point_changed(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_optimistic=1)
        children = _in_doubt(alg, [0.3, 0.2, 0.1, 0.05])
        parent = alg.refs[0][0]
        alg.stuckcounter[parent] = 2
        alg._update_reference_set()
        assert alg.stuckcounter[children[0]] == 0
        assert parent not in alg.stuckcounter

    def test_winning_an_open_contest_does_not_archive_the_winner(self, tmp_path):
        """A slot the candidate has taken is counted stuck when the contest settles its
        way, since the winner already holds the slot and nothing replaced it that round.
        With the count carried across the takeover rather than restarted, a slot would
        reach ``local_min_limit`` after a few contests won and be thrown away."""
        alg = self._alg(tmp_path, ss_noise_optimistic=1, local_min_limit=2)
        children = _in_doubt(alg, [0.3, 0.2, 0.1, 0.05])
        parent, child = alg.refs[0][0], children[0]
        alg.stuckcounter[parent] = 1                       # one round from the archive
        alg._update_reference_set()
        assert alg.refs[0][0] == child and alg.stuckcounter[child] == 0
        # The draws settle for the new occupant: the member it displaced is clearly worse.
        alg.draws[child] += [9.9, 9.95]
        alg.draws[parent] += [30.0]
        alg.repeat_draws.update({child: alg.draws[child], parent: alg.draws[parent]})
        alg.pending_draws = {}
        alg.received = {m: [] for m, _ in alg.refs}
        alg._update_reference_set()
        assert alg.refs[0][0] == child and alg.stuckcounter[child] == 1
        assert not any(m == child for m, _ in alg.local_mins)

    def test_the_draws_can_settle_the_other_way_and_the_parent_takes_the_slot_back(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_optimistic=1)
        children = _in_doubt(alg, [0.3, 0.2, 0.1, 0.05])
        parent, child = alg.refs[0][0], children[0]
        alg._update_reference_set()
        assert alg.refs[0][0] == child
        # The draws come back: the child's first draw was lucky and the parent's was not.
        # Both reach the cap, so the means decide, and they decide for the parent.
        alg.draws[child] += [11.0, 11.2]
        alg.draws[parent] += [10.2]
        alg.repeat_draws.update({child: alg.draws[child], parent: alg.draws[parent]})
        alg.pending_draws = {}
        alg.received = {m: [] for m, _ in alg.refs}
        alg._update_reference_set()
        assert alg.refs[0][0] == parent
        assert alg.refs[0][1] == pytest.approx(np.mean([10.0, 10.4, 10.2]))
        assert child not in alg.draws and child not in alg.contenders
        assert alg.stuckcounter[parent] == 0

    def test_a_candidate_that_does_not_lead_leaves_the_parent_in_the_slot(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_optimistic=1)
        children = _in_doubt(alg, [-0.1, -0.1, -0.1, -0.1])   # every child is worse
        parent = alg.refs[0][0]
        alg._update_reference_set()
        assert alg.refs[0][0] == parent
        assert alg.contenders[parent] == children[0]
        assert alg.stuckcounter[parent] == 0                  # in doubt, so not stuck either

    def test_a_settled_contest_is_decided_the_same_way_with_it_on(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_optimistic=1)
        children = _in_doubt(alg, [5.0, 0.0, 0.0, 0.0])       # slot 0 far outside the spread
        parent = alg.refs[0][0]
        alg._update_reference_set()
        assert alg.refs[0][0] == children[0]
        assert children[0] not in alg.contenders and parent not in alg.contenders
        assert alg.draws[children[0]] == [pytest.approx(5.2)]
        assert parent not in alg.draws                        # a settled winner takes the slot


class TestRedrawBudget:
    """``ss_noise_redraw_budget``: a cap on the re-draws one round queues for the
    orderings it cannot settle, spent where a wrong decision would cost most (#696
    lever 2)."""

    @pytest.fixture(autouse=True)
    def _no_periodic_output(self, monkeypatch):
        monkeypatch.setattr(algorithms.ScatterSearch, 'output_results', lambda self, *a, **k: None)

    def _alg(self, tmp_path, **overrides):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, **overrides)))
        _feed(alg, alg.start_run(), _Scorer(1.0))
        return alg

    def test_it_is_off_by_default(self, tmp_path):
        assert algorithms.ScatterSearch(_stochastic(_conf(tmp_path))).redraw_budget == 0

    def test_the_priority_is_the_objective_gap_weighted_by_the_slots_rank(self, tmp_path):
        alg = self._alg(tmp_path)                             # popsize 4
        assert alg._redraw_priority(0, -0.4) == pytest.approx(0.4)
        assert alg._redraw_priority(1, 0.4) == pytest.approx(0.3)
        assert alg._redraw_priority(3, 0.4) == pytest.approx(0.1)
        assert alg._redraw_priority(99, 0.4) == pytest.approx(0.1)   # clamped to the last slot
        assert alg._redraw_priority(0, np.inf) == np.inf

    def test_without_a_budget_every_request_is_queued_in_the_order_it_was_made(self, tmp_path):
        alg = self._alg(tmp_path)
        members = [m for m, _ in alg.refs]
        for m in members:
            alg.draws[m] = [1.0, 1.2]
        alg.pending_draws = {}
        queued = alg._select_redraws([(0.1, [members[0]]), (9.0, [members[1]]),
                                      (None, [members[2]])])
        assert [p.name for p in queued] == [m.name + '_d2' for m in members[:3]]

    def test_a_budget_spends_the_round_on_the_highest_priority_decisions(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_redraw_budget=2)
        members = [m for m, _ in alg.refs]
        for m in members:
            alg.draws[m] = [1.0, 1.2]
        alg.pending_draws = {}
        alg._redraw_budget_left = alg.redraw_budget
        queued = alg._select_redraws([(0.1, [members[0]]), (9.0, [members[1]]),
                                      (1.0, [members[2]]), (5.0, [members[3]])])
        assert [p.name for p in queued] == [members[1].name + '_d2', members[3].name + '_d2']
        assert alg._redraw_budget_left == 0

    def test_a_request_the_cap_has_emptied_costs_nothing(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_redraw_budget=1)
        members = [m for m, _ in alg.refs]
        alg.draws[members[0]] = [1.0, 1.2, 1.1]               # at ss_noise_max_draws
        alg.draws[members[1]] = [2.0, 2.2]
        alg.pending_draws = {}
        alg._redraw_budget_left = alg.redraw_budget
        queued = alg._select_redraws([(9.0, [members[0]]), (1.0, [members[1]])])
        assert [p.name for p in queued] == [members[1].name + '_d2']

    def test_the_stuck_re_draw_is_never_rationed(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_redraw_budget=1)
        members = [m for m, _ in alg.refs]
        for m in members:
            alg.draws[m] = [1.0, 1.2]
        alg.pending_draws = {}
        alg._redraw_budget_left = alg.redraw_budget
        queued = alg._select_redraws([(9.0, [members[0]]), (None, [members[1]]),
                                      (None, [members[2]]), (0.5, [members[3]])])
        names = [p.name for p in queued]
        assert names[:2] == [members[1].name + '_d2', members[2].name + '_d2']
        assert names[2:] == [members[0].name + '_d2']         # the budget bought one contest
        assert members[3].name + '_d2' not in names

    def test_a_round_keeps_open_the_contests_that_matter_and_leaves_the_rest_to_the_means(self, tmp_path):
        alg = self._alg(tmp_path, ss_noise_redraw_budget=2)
        children = _in_doubt(alg, [0.3, 0.2, 0.1, 0.05])
        members = [m for m, _ in alg.refs]
        queued = alg._update_reference_set()
        # Slot 0's contest costs most to get wrong, so it is the one kept open and drawn
        # for; the other three are decided now on their means, which every child leads.
        assert alg.contenders == {members[0]: children[0]}
        assert sorted(p.name for p in queued) == sorted(['child0_d1', members[0].name + '_d2'])
        assert alg._redraw_budget_left == 0
        assert [r[0] for r in alg.refs] == [members[0]] + children[1:]
        assert [c for c, _ in alg._round_accepted] == children[1:]
        # The neighbour pairs the round would draw for next find the budget already spent,
        # as _search_got_result runs them (after the reference set is sorted).
        alg.refs = sorted(alg.refs, key=lambda x: x[1])
        assert alg._select_redraws(alg._unseparated_neighbour_pairs()) == []

    def test_without_a_budget_every_contest_is_kept_open(self, tmp_path):
        alg = self._alg(tmp_path)
        children = _in_doubt(alg, [0.3, 0.2, 0.1, 0.05])
        members = [m for m, _ in alg.refs]
        queued = alg._update_reference_set()
        assert alg.contenders == dict(zip(members, children))
        assert len(queued) == 8                            # both sides of all four
        assert alg._round_accepted == []

    def test_a_budget_leaves_fewer_re_draws_over_a_whole_fit(self, tmp_path):
        def redraws_of(**overrides):
            alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, **overrides)))
            scorer = _Scorer(1.0)
            _drive(alg, scorer, iterations=12)
            return sum(1 for name, _ in scorer.calls if '_d' in name)
        assert redraws_of(ss_noise_redraw_budget=1) < redraws_of()
