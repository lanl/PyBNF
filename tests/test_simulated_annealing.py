"""Unit tests for ``SimulatedAnnealing`` (pybnf/algorithms/optimizers/
simulated_annealing.py), the ``sa`` fit type.

M2.2 (ADR-0008) rewrote ``sa`` from a mode of ``BasicBayesMCMCAlgorithm`` into a
standalone optimizer that minimizes the *raw* objective. These are the cooling-
schedule tests moved over from ``test_basic_mcmc_class.py``'s ``TestSimulatedAnnealing``
and rewritten against the new class, plus a check of its own random-walk proposal.

The end-to-end functional behavior (sa converges to the known optimum on an
all-uniform-prior analytical fit, where the prior-drop is a numerical no-op) is
covered by ``test_optimizer_integration.test_sa_finds_gaussian_mode``.

Construction follows the sibling sampler/optimizer suites: a real
``config.Configuration`` over the three v*__FREE params in parabola.bngl, with the
per-chain state set directly so each decision in ``got_result`` can be exercised
without a dask client.
"""
import os

import numpy as np

from .context import algorithms, config, pset

PARABOLA = 'bngl_files/parabola.bngl'


def _make_config(tmp_path, var_spec, **overrides):
    out = str(tmp_path) + '/'
    os.makedirs(out + 'Results', exist_ok=True)
    base = {
        'population_size': 4, 'max_iterations': 10000, 'step_size': 0.2,
        'output_every': 1000, 'output_dir': out,
        'models': {PARABOLA}, 'exp_data': {'bngl_files/par1.exp'},
        'initialization': 'lh', PARABOLA: ['bngl_files/par1.exp'],
        'fit_type': 'sa',
    }
    base.update(var_spec)
    base.update(overrides)
    return config.Configuration(base)


NORMAL_VARS = {
    ('normal_var', 'v1__FREE'): [0.0, 1.0],
    ('normal_var', 'v2__FREE'): [0.0, 1.0],
    ('normal_var', 'v3__FREE'): [0.0, 1.0],
}
UNIFORM_VARS = {
    ('uniform_var', 'v1__FREE'): [0.0, 10.0],
    ('uniform_var', 'v2__FREE'): [0.0, 10.0],
    ('uniform_var', 'v3__FREE'): [0.0, 10.0],
}

_KEYS = ('v1__FREE', 'v2__FREE', 'v3__FREE')


def _normal_pset(values):
    return pset.PSet([
        pset.FreeParameter('v1__FREE', 'normal_var', 0.0, 1.0, values[0]),
        pset.FreeParameter('v2__FREE', 'normal_var', 0.0, 1.0, values[1]),
        pset.FreeParameter('v3__FREE', 'normal_var', 0.0, 1.0, values[2]),
    ])


def _uniform_pset(values):
    return pset.PSet([
        pset.FreeParameter('v1__FREE', 'uniform_var', 0.0, 10.0, values[0]),
        pset.FreeParameter('v2__FREE', 'uniform_var', 0.0, 10.0, values[1]),
        pset.FreeParameter('v3__FREE', 'uniform_var', 0.0, 10.0, values[2]),
    ])


def _vec(p):
    return np.array([p[k] for k in _KEYS])


# --------------------------------------------------------------------------- #
# Cooling schedule (in got_result)
# --------------------------------------------------------------------------- #
class TestCoolingSchedule:

    def _sa_algo(self, tmp_path, beta0=0.5, cooling=0.5, beta_max=2.0, pop=1):
        cfg = _make_config(tmp_path, NORMAL_VARS, population_size=pop, beta=[beta0],
                           cooling=cooling, beta_max=beta_max)
        algo = algorithms.SimulatedAnnealing(cfg)
        # A good (low-objective) current point for each chain.
        algo.current_pset = [_normal_pset((5.0, 5.0, 5.0)) for _ in range(pop)]
        algo.current_score = [10.0] * pop
        algo.iteration = [0] * pop
        return algo

    def _worse_result(self, index=0):
        """An accepted but unfavorable proposal (higher objective than current)."""
        cand = _normal_pset((5.0, 5.0, 5.0))
        cand.name = 'iter1run%i' % index
        res = algorithms.Result(cand, {}, cand.name)
        res.score = 50.0          # high score (worse) than the current 10.0
        return res

    def test_unfavorable_accepted_move_increases_beta(self, tmp_path, monkeypatch):
        """An accepted *uphill* move (ln_p_accept < 0) cools the chain:
        beta += cooling. Forcing acceptance (rand=0) of a worse proposal must
        raise beta from 0.5 to exactly 1.0 (cooling=0.5)."""
        algo = self._sa_algo(tmp_path, beta0=0.5, cooling=0.5)
        monkeypatch.setattr(np.random, 'rand', lambda *a: 0.0)   # force accept
        algo.got_result(self._worse_result())
        np.testing.assert_allclose(algo.betas[0], 1.0, rtol=1e-12)

    def test_reaching_beta_max_stops(self, tmp_path, monkeypatch):
        """Once cooling drives the (only) chain's beta to beta_max, the run is
        finished: got_result returns 'STOP'. Starting at 1.8 with cooling 0.5 the
        next unfavorable accept reaches 2.3 >= beta_max=2.0."""
        algo = self._sa_algo(tmp_path, beta0=1.8, cooling=0.5, beta_max=2.0)
        monkeypatch.setattr(np.random, 'rand', lambda *a: 0.0)
        assert algo.got_result(self._worse_result()) == 'STOP'

    def test_one_replica_finishing_returns_empty_not_stop(self, tmp_path, monkeypatch):
        """With several chains, the run only stops once *all* have finished. When
        one chain reaches beta_max but another is still cooling, got_result
        returns [] (that chain idles) rather than 'STOP'. Chain 0 is set just
        below beta_max; cooling it once finishes it while chain 1 (beta 0.5) is
        far from done."""
        algo = self._sa_algo(tmp_path, beta0=1.0, cooling=0.5, beta_max=2.0, pop=2)
        algo.betas = [1.8, 0.5]                  # chain 0 about to finish
        monkeypatch.setattr(np.random, 'rand', lambda *a: 0.0)
        assert algo.got_result(self._worse_result(index=0)) == []
        assert algo.betas[0] >= 2.0 and algo.betas[1] == 0.5

    def test_first_result_per_chain_is_always_accepted(self, tmp_path):
        """A chain's first result (current_pset still None) is accepted
        unconditionally — it seeds the chain's current point — regardless of the
        RNG. After it, current_pset/current_score reflect that point."""
        algo = self._sa_algo(tmp_path, pop=1)
        algo.current_pset = [None]               # un-seed: simulate the very first result
        algo.current_score = [np.inf]
        seed = _normal_pset((1.0, 2.0, 3.0))
        seed.name = 'iter0run0'
        res = algorithms.Result(seed, {}, seed.name)
        res.score = 99.0
        algo.got_result(res)
        assert algo.current_pset[0] is seed
        assert algo.current_score[0] == 99.0


# --------------------------------------------------------------------------- #
# choose_new_pset: fixed-magnitude random-walk proposal (sa's own copy)
# --------------------------------------------------------------------------- #
class TestProposal:

    def test_step_magnitude_is_exactly_step_size(self, tmp_path):
        """The proposal rescales a random Gaussian direction so the per-parameter
        deltas have L2 norm exactly step_size, so ||proposed - current|| ==
        step_size on every draw (unbounded normal_var params -> no reflection)."""
        cfg = _make_config(tmp_path, NORMAL_VARS, step_size=0.37)
        algo = algorithms.SimulatedAnnealing(cfg)
        old = _normal_pset((5.0, -2.0, 1.0))
        old_vec = _vec(old)
        np.random.seed(7)
        for _ in range(100):
            new = algo.choose_new_pset(old)
            np.testing.assert_allclose(np.linalg.norm(_vec(new) - old_vec), 0.37, rtol=1e-12)

    def test_oversized_step_reflects_into_box(self, tmp_path):
        """A step far larger than the box is reflected back inside rather than
        rejected: every coordinate of the proposal stays within [0, 10]."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, step_size=100.0)
        algo = algorithms.SimulatedAnnealing(cfg)
        old = _uniform_pset((5.0, 5.0, 5.0))     # step 100 >> box width 10
        np.random.seed(3)
        for _ in range(50):
            new = algo.choose_new_pset(old)
            for k in _KEYS:
                assert 0.0 <= new[k] <= 10.0
