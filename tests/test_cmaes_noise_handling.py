"""CMA-ES uncertainty handling for a stochastic model (#661, ADR-0135).

CMA-ES sorts each generation on one objective value per candidate. For a stochastic model
that value is a draw, so when the noise is comparable to the real differences the sort is
partly random, the distribution update is pulled in arbitrary directions, and the step
size shrinks because noise reads as stagnation. The fix is Hansen's uncertainty handling:
re-simulate a few candidates at fresh seeds, measure how far they move in the ranking
against what pure noise would do, and while the ranking is unreliable simulate every
candidate more times and hold the step size up.

Four layers, in order:

  * ``pybnf.algorithms.noise_handling.RankChangeNoise``: the measurement and the
    adaptation, on hand-built values;
  * the seam: a PSet an algorithm returns can carry a replicate offset, and ``make_job``
    honours it, which is what makes a re-simulation a fresh draw under the default seed
    policy instead of the same trajectory again;
  * ``CMAESAlgorithm`` driven by hand through ``start_run`` / ``got_result`` with a scorer
    whose noise is keyed by the replicate offset, so a test states exactly what each draw
    of each candidate is worth; and
  * the whole run loop, through the integration harness with a noisy fake runner.
"""
import pickle

import numpy as np
import numpy.testing as npt
import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.algorithms.noise_handling import RankChangeNoise
from .test_best_fit_confirmation import _algo, _ps

MU = np.array([2.0, -1.0])


def _draw_seed(values, offset):
    """A seed from the parameter VALUES and the replicate offset. Floats and ints hash the
    same in every process, where a PSet's hash (through its parameter names) does not, so
    the noise a test sees is the same on every run."""
    return abs(hash((tuple(round(float(v), 9) for v in values), int(offset)))) % (2 ** 32)


def _conf(tmp_path, **overrides):
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(list(MU), [1.0, 1.0]))
    base = dict(n_params=2, population_size=8, max_iterations=40, cmaes_sigma0=0.3,
                random_seed=7)
    base.update(overrides)
    return H.make_config(tmp_path, 'cmaes', tgt, exp, **base)


def _stochastic(conf):
    """Mark the analytical model stochastic, which is what turns the handling on."""
    for model in conf.models.values():
        model.stochastic = True
    return conf


class _Res:
    def __init__(self, pset, score):
        self.pset, self.score, self.name = pset, score, pset.name


class _Scorer:
    """``sum((x - mu)**2)`` plus noise keyed by the parameter values AND the replicate offset:
    the same draw comes back for the same offset, and a different offset is a fresh one,
    which is what the default seed policy does."""

    def __init__(self, sd):
        self.sd = sd
        self.calls = []

    def __call__(self, pset):
        offset = int(getattr(pset, 'replicate_offset', 0))
        self.calls.append((pset.name, offset))
        x = np.array([pset['p1'], pset['p2']])
        rng = np.random.default_rng(_draw_seed(x, offset))
        return float(np.sum((x - MU) ** 2)) + self.sd * rng.standard_normal()


def _feed(alg, psets, scorer):
    """Score every queued pset; the last one's response is the next batch (or 'STOP')."""
    response = []
    for p in psets:
        r = alg.got_result(_Res(p, scorer(p)))
        if r == 'STOP':
            return 'STOP'
        response.extend(r)
    return response


def _drive(alg, scorer, generations):
    """Run ``generations`` distribution updates by hand."""
    batch = alg.start_run()
    while alg.generation < generations and batch != 'STOP':
        batch = _feed(alg, batch, scorer)
    return batch


# --------------------------------------------------------------------------- #
# The measurement and the adaptation
# --------------------------------------------------------------------------- #
class TestRankChangeNoise:

    def test_the_re_evaluated_subset_is_a_tenth_floored_at_three(self):
        """Three, not two: with two the pure-noise limit at every rank is zero, so the
        statistic could never come out negative and the draw count could only grow."""
        noise = RankChangeNoise(n_dim=3)
        assert noise.count_to_reevaluate(8) == 3
        assert noise.count_to_reevaluate(12) == 3
        assert noise.count_to_reevaluate(40) == 4
        assert noise.count_to_reevaluate(1) == 1
        assert noise.rank_change_limit(2, 3) == 0.0          # the two-candidate limit
        assert all(noise.rank_change_limit(r, 5) == 1.0 for r in range(1, 6))

    def test_the_pure_noise_limit_is_a_quantile_of_the_possible_rank_changes(self):
        noise = RankChangeNoise(n_dim=3, theta=0.5)
        # A value at rank 4 among 7 can move 0, 1, 1, 2, 2, 3 or 3 places: the 25th
        # percentile (lower) is 1. At rank 1 the changes are 0..6, again 1.
        assert noise.rank_change_limit(4, 7) == 1.0
        assert noise.rank_change_limit(1, 7) == 1.0
        assert noise.rank_change_limit(1, 1) == 0.0

    def test_a_deterministic_function_measures_negative(self):
        noise = RankChangeNoise(n_dim=3)
        values = [3.0, 1.0, 2.0]
        assert noise.measure(values, values) < 0.0

    def test_pure_noise_measures_positive_on_average(self):
        noise = RankChangeNoise(n_dim=3)
        rng = np.random.default_rng(1)
        measurements = [noise.measure(rng.standard_normal(5), rng.standard_normal(5))
                        for _ in range(300)]
        assert np.mean(measurements) > 0.5

    def test_a_small_shift_under_large_gaps_measures_negative(self):
        noise = RankChangeNoise(n_dim=3)
        f_old = np.array([0.0, 10.0, 20.0, 30.0])
        assert noise.measure(f_old, f_old + 0.1) < 0.0

    def test_the_update_grows_the_evaluations_and_holds_the_step_size(self):
        noise = RankChangeNoise(n_dim=10, max_evals=4)
        assert noise.evaluations() == 1
        factors = [noise.update(5.0) for _ in range(8)]
        assert all(f == pytest.approx(1.0 + 2.0 / 20.0) for f in factors)
        assert noise.level > 0.0
        assert noise.evaluations() == 4 and noise.evals == 4.0     # capped

    def test_the_update_shrinks_the_evaluations_when_the_ranking_is_reliable(self):
        noise = RankChangeNoise(n_dim=10, max_evals=4)
        for _ in range(8):
            noise.update(5.0)
        factors = [noise.update(-5.0) for _ in range(30)]
        assert factors[-1] == 1.0 and noise.level < 0.0
        assert noise.evaluations() == 1 and noise.evals == 1.0

    def test_a_zero_level_counts_as_reliable(self):
        noise = RankChangeNoise(n_dim=10, max_evals=4)
        noise.evals = 3.0
        assert noise.update(0.0) == 1.0
        assert noise.level == 0.0 and noise.evals < 3.0

    def test_the_level_is_filtered(self):
        noise = RankChangeNoise(n_dim=3, cum=0.3)
        noise.update(1.0)
        assert noise.level == pytest.approx(0.3)
        noise.update(1.0)
        assert noise.level == pytest.approx(0.3 * 0.7 + 0.3)

    def test_the_ranking_value_of_a_re_evaluated_candidate_is_the_mean(self):
        npt.assert_allclose(RankChangeNoise.combined([1.0, 3.0], [3.0, 1.0]), [2.0, 2.0])


# --------------------------------------------------------------------------- #
# The seam: a returned PSet can carry a replicate offset
# --------------------------------------------------------------------------- #
class TestReplicateOffsetSeam:

    def test_a_pset_carries_no_offset_by_default(self):
        assert _ps(1.0).replicate_offset == 0

    def test_make_job_honours_the_offset_a_pset_carries(self, tmp_path):
        algo = _algo(tmp_path)
        pset = _ps(1.0, name='again')
        pset.replicate_offset = 3
        (job,) = algo.make_job(pset)
        assert job.replicate_index == 3

    def test_an_explicit_offset_wins_over_the_pset_attribute(self, tmp_path):
        algo = _algo(tmp_path)
        pset = _ps(1.0, name='again')
        pset.replicate_offset = 3
        (job,) = algo.make_job(pset, replicate_offset=7)
        assert job.replicate_index == 7

    def test_the_offset_shifts_every_smoothing_replicate(self, tmp_path):
        algo = _algo(tmp_path, smoothing=2)
        pset = _ps(1.0, name='again')
        pset.replicate_offset = 4
        jobs = algo.make_job(pset)
        assert sorted(job.replicate_index for job in jobs) == [4, 5]

    def test_the_offset_is_not_part_of_the_identity(self):
        a, b = _ps(1.0), _ps(1.0)
        b.replicate_offset = 5
        assert a == b and hash(a) == hash(b)


# --------------------------------------------------------------------------- #
# CMA-ES, driven by hand
# --------------------------------------------------------------------------- #
class TestCMAES:

    def test_a_deterministic_fit_has_no_handling_and_queues_one_draw_per_candidate(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_conf(tmp_path))
        assert alg.noise is None
        psets = alg.start_run()
        assert len(psets) == alg.lam
        assert all(p.replicate_offset == 0 and '_e' not in p.name for p in psets)
        # Every generation is a single phase: the last result of one samples the next.
        response = _feed(alg, psets, _Scorer(0.0))
        assert len(response) == alg.lam and alg.generation == 1

    def test_the_switch_turns_it_off_for_a_stochastic_fit(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path, cmaes_noise_handling=0)))
        assert alg.noise is None

    def test_a_stochastic_fit_re_evaluates_a_subset_at_fresh_offsets(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path)))
        assert alg.noise is not None and alg.noise.max_evals == 10
        scorer = _Scorer(1.0)
        psets = alg.start_run()
        assert len(psets) == alg.lam and alg.gen_evals == 1

        reev = _feed(alg, psets, scorer)
        # Three candidates come back for a fresh draw, at an offset past the one they used.
        assert len(reev) == 3 and alg.phase == 'reev'
        assert all(p.replicate_offset == 1 and p.name.endswith('_r0') for p in reev)
        assert sorted(alg.reev_indices) == sorted(alg.pending[p.name][0] for p in reev)
        assert alg.generation == 0

        old = {i: alg.gen_score[i] for i in alg.reev_indices}
        nxt = _feed(alg, reev, scorer)
        # The measurement happened, the re-evaluated candidates rank on the mean of their
        # two draws, and the next generation was sampled.
        assert alg.noise.last_measurement is not None
        for i in alg.reev_indices:
            new = alg.reev_scores[i][0]
            npt.assert_allclose(alg.gen_score[i], 0.5 * (old[i] + new))
        assert alg.generation == 1 and len(nxt) == alg.lam * alg.gen_evals

    def test_heavy_noise_raises_the_evaluations_and_holds_the_step_size(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path, cmaes_noise_max_evals=4)))
        scorer = _Scorer(1e4)               # noise dwarfs every real difference
        factors = []
        update = alg.noise.update
        alg.noise.update = lambda m: factors.append(update(m)) or factors[-1]
        batch = _drive(alg, scorer, generations=15)
        assert alg.noise.level > 0.0
        assert alg.noise.evaluations() > 1
        # The generation just queued carries every candidate several times, at fresh offsets.
        evals = alg.gen_evals
        assert evals == alg.noise.evaluations() and len(batch) == alg.lam * evals
        offsets = sorted(p.replicate_offset for p in batch if '_ind0_e' in p.name)
        assert offsets == list(range(evals))
        # A held-up step size: the factor was applied to at least one update.
        assert max(factors) == pytest.approx(1.0 + 2.0 / (alg.n + 10.0))

    def test_a_quiet_stochastic_model_keeps_one_evaluation(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path)))
        _drive(alg, _Scorer(1e-9), generations=8)
        assert alg.noise.level < 0.0
        assert alg.noise.evaluations() == 1
        assert alg._noise_sigma_factor == 1.0

    def test_candidates_are_ranked_on_the_mean_of_their_draws(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path)))
        alg.noise.evals = 3.0
        psets = alg.start_run()
        assert alg.gen_evals == 3 and len(psets) == 3 * alg.lam
        by_index = {}
        for p in psets:
            index, phase = alg.pending[p.name]
            assert phase == 'eval'
            by_index.setdefault(index, []).append(p)
        assert all(sorted(q.replicate_offset for q in group) == [0, 1, 2]
                   for group in by_index.values())
        scores = {}
        for p in psets[:-1]:
            index = alg.pending[p.name][0]
            scores.setdefault(index, []).append(float(index + p.replicate_offset))
            assert alg.got_result(_Res(p, float(index + p.replicate_offset))) == []
        last = psets[-1]
        index = alg.pending[last.name][0]
        scores.setdefault(index, []).append(float(index + last.replicate_offset))
        alg.got_result(_Res(last, float(index + last.replicate_offset)))
        for i, values in scores.items():
            npt.assert_allclose(alg.gen_score[i], np.mean(values))

    def test_a_failed_draw_makes_the_candidate_infinite(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path)))
        alg.noise.evals = 2.0
        psets = alg.start_run()
        target = alg.pending[psets[0].name][0]
        for p in psets[:-1]:
            index = alg.pending[p.name][0]
            alg.got_result(_Res(p, np.inf if (index == target and p.replicate_offset == 1) else 1.0))
        alg.got_result(_Res(psets[-1], 1.0))
        assert alg.gen_score[target] == np.inf
        assert all(np.isfinite(s) for i, s in enumerate(alg.gen_score) if i != target)

    def test_smoothing_spaces_the_offsets_by_the_smoothing_count(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path, smoothing=2)))
        alg.noise.evals = 2.0
        psets = alg.start_run()
        assert sorted({p.replicate_offset for p in psets}) == [0, 2]
        reev = _feed(alg, psets, _Scorer(1.0))
        assert sorted({p.replicate_offset for p in reev}) == [4, 6]

    def test_the_state_pickles_mid_generation(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path)))
        psets = alg.start_run()
        for p in psets[:3]:
            alg.got_result(_Res(p, 1.0))
        clone = pickle.loads(pickle.dumps(alg))
        assert clone.pending == alg.pending and clone.noise.level == alg.noise.level

    def test_a_restart_clears_the_re_evaluation_phase(self, tmp_path):
        alg = algorithms.CMAESAlgorithm(_stochastic(_conf(tmp_path)))
        psets = alg.start_run()
        _feed(alg, psets, _Scorer(1.0))
        assert alg.phase == 'reev'
        alg._seed_distribution(alg.mean, 0.3)
        assert alg.phase == 'eval' and alg.pending == {} and alg.reev_indices == []


# --------------------------------------------------------------------------- #
# The whole run loop, with a noisy fake runner
# --------------------------------------------------------------------------- #
def test_a_stochastic_fit_runs_end_to_end_and_re_simulates_at_fresh_indices(tmp_path, monkeypatch):
    H.install(monkeypatch)
    conf = _stochastic(_conf(tmp_path, max_iterations=25, population_size=8))
    seen = []

    def noisy_run_job(j, debug=False, failed_logs_dir=''):
        res = H.slim_run_job(j, debug, failed_logs_dir)
        rng = np.random.default_rng(_draw_seed([j.params['p1'], j.params['p2']], j.replicate_index))
        res.score = float(res.score) + 0.5 * rng.standard_normal()
        seen.append(int(j.replicate_index))
        return res

    monkeypatch.setattr(algorithms.core, 'run_job', noisy_run_job)
    alg = algorithms.CMAESAlgorithm(conf)
    assert alg.noise is not None
    H.drive(alg)
    # Re-simulations went through the run loop with a fresh replicate index.
    assert any(index > 0 for index in seen)
    assert alg.noise.last_measurement is not None
    # And the fit still finds the mode of the Gaussian under the noise.
    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, MU, atol=0.75), recovered
