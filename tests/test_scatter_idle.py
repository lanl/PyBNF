"""Scatter search gives idle processors another draw of a member whose rank is in doubt
(#660 step 4, ADR-0139).

Scatter search waits for every simulation of a round before building the next, so toward
the end of a round processors sit idle waiting for the slowest simulation, and for a
stochastic model the spread in running times is wide. When the fit knows how many
simulations it can run at once and fewer are in flight, the difference is filled with
fresh draws of the members the noise cannot order, within the same cap as every other
re-draw. Nothing happens for a deterministic fit, when the count is unknown, or when
nothing is idle.
"""
import os

import numpy as np

from . import integration_harness as H
from .context import algorithms

MU = np.array([2.0, -1.0])


def _conf(tmp_path, **overrides):
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(list(MU), [1.0, 1.0]))
    base = dict(n_params=2, population_size=4, max_iterations=60, init_size=8,
                reserve_size=20, random_seed=11)
    base.update(overrides)
    conf = H.make_config(tmp_path, 'ss', tgt, exp, **base)
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


def _no_output(*args, **kwargs):
    return None


def _key(pset):
    return (round(float(pset['p1']), 9), round(float(pset['p2']), 9))


def _scorer(sd):
    def score(pset):
        offset = int(getattr(pset, 'replicate_offset', 0))
        rng = np.random.default_rng(abs(hash((_key(pset), offset))) % (2 ** 32))
        x = np.array([pset['p1'], pset['p2']])
        return float(np.sum((x - MU) ** 2)) + sd * rng.standard_normal()
    return score


def _into_a_round(alg, scorer):
    """Drive a stochastic scatter search into its second combination round, so a pooled
    spread exists and the reference set has settled estimates; return that round's batch."""
    alg.output_results = _no_output
    batch = alg.start_run()
    for _ in range(2):
        response = []
        for p in batch:
            response.extend(alg.got_result(_Res(p, scorer(p))))
        batch = response
    assert alg.pending and alg._noise_sd() is not None
    return batch


class TestCountingWorkers:

    def test_sums_the_threads_of_every_connected_worker(self):
        class Client:
            def scheduler_info(self):
                return {'workers': {'a': {'nthreads': 2}, 'b': {'nthreads': 3}, 'c': {}}}
        assert algorithms.Algorithm._count_workers(Client()) == 6

    def test_none_when_it_cannot_be_read_or_nothing_is_connected(self):
        class Broken:
            def scheduler_info(self):
                raise RuntimeError('no scheduler')

        class Empty:
            def scheduler_info(self):
                return {'workers': {}}
        assert algorithms.Algorithm._count_workers(Broken()) is None
        assert algorithms.Algorithm._count_workers(Empty()) is None
        assert algorithms.Algorithm._count_workers(object()) is None

    def test_the_default_is_unknown(self, tmp_path):
        assert algorithms.ScatterSearch(_conf(tmp_path)).worker_count is None


class TestFillingIdleProcessors:

    def test_a_deterministic_fit_never_fills(self, tmp_path):
        alg = algorithms.ScatterSearch(_conf(tmp_path))
        alg.output_results = _no_output
        batch = alg.start_run()
        alg.worker_count = 1000
        first = alg.got_result(_Res(batch[0], 1.0))
        assert first == [] and alg.pending_draws == {}

    def test_the_switch_turns_it_off_and_leaves_the_rest_of_the_handling_on(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, ss_fill_idle=0)))
        assert alg.noise_handling is True and alg.fill_idle is False
        batch = _into_a_round(alg, _scorer(1.0))
        for i, (m, _) in enumerate(alg.refs):
            alg.draws[m] = [9.5 + 0.001 * i, 10.5 + 0.001 * i]
        alg.repeat_draws = dict(alg.draws)
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        alg.worker_count = 1000
        assert alg.got_result(_Res(batch[0], 5.0)) == []

    def test_nothing_without_a_processor_count(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        batch = _into_a_round(alg, _scorer(1.0))
        assert alg.worker_count is None
        assert alg.got_result(_Res(batch[0], 5.0)) == []

    def test_nothing_while_every_processor_is_busy(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        batch = _into_a_round(alg, _scorer(1.0))
        alg.worker_count = len(alg.pending) + len(alg.pending_draws) - 1
        assert alg.got_result(_Res(batch[0], 5.0)) == []

    def test_idle_processors_get_draws_of_members_the_noise_cannot_order(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        batch = _into_a_round(alg, _scorer(1.0))
        # Make every neighbouring pair of the reference set indistinguishable: means a
        # thousandth apart under a draw-to-draw spread of about seven tenths.
        for i, (m, _) in enumerate(alg.refs):
            alg.draws[m] = [9.5 + 0.001 * i, 10.5 + 0.001 * i]
        alg.repeat_draws = dict(alg.draws)
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        in_flight = len(alg.pending) + len(alg.pending_draws)
        alg.worker_count = in_flight + 2       # two would idle once this result frees one
        extra = alg.got_result(_Res(batch[0], 5.0))
        assert len(extra) == 3                 # the freed processor plus the two idle ones
        members = [m for m, _ in alg.refs]
        assert all(any(p.name.startswith(m.name + '_d') for m in members) for p in extra)
        assert all(p.replicate_offset == 2 for p in extra)
        assert all(p.name in alg.pending_draws for p in extra)

    def test_the_cap_and_the_round_still_hold(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path, ss_noise_max_draws=2)))
        scorer = _scorer(1.0)
        batch = _into_a_round(alg, scorer)
        for m, _ in alg.refs:
            alg.draws[m] = [10.0, 10.001]      # every member already at the cap of two
        alg.repeat_draws = dict(alg.draws)
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        alg.worker_count = 1000
        assert alg.got_result(_Res(batch[0], 5.0)) == []
        # The rest of the round completes normally and the next round is queued.
        response = []
        for p in batch[1:]:
            response.extend(alg.got_result(_Res(p, scorer(p))))
        assert alg.pending and not alg.pending_draws or response

    def test_both_sides_of_an_open_contest_are_drawn(self, tmp_path):
        alg = algorithms.ScatterSearch(_stochastic(_conf(tmp_path)))
        batch = _into_a_round(alg, _scorer(1.0))
        for i, (m, _) in enumerate(alg.refs):
            alg.draws[m] = [float(10 * (i + 1)), float(10 * (i + 1))]   # well separated
        alg.repeat_draws = dict(alg.draws)
        alg.refs = [(m, float(np.mean(alg.draws[m]))) for m, _ in alg.refs]
        parent = alg.refs[0][0]
        child = batch[0]
        child.name = 'contender'
        alg.draws[child] = [9.99]
        alg.contenders = {parent: child}
        alg.pending_draws = {}
        alg.worker_count = len(alg.pending) + 5
        extra = alg.got_result(_Res(batch[1], 5.0))
        names = sorted(p.name for p in extra)
        assert names == sorted([parent.name + '_d2', 'contender_d1'])
