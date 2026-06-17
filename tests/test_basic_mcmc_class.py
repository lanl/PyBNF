"""
Oracle-anchored tests for ``BasicBayesMCMCAlgorithm`` (pybnf/algorithms.py),
which implements plain Metropolis MCMC (``mh``) and parallel tempering (``pt``).
(Simulated annealing, formerly a third mode of this class, is now a standalone
optimizer — see ``test_simulated_annealing.py``.)

The Metropolis acceptance rule and the convergence diagnostics are inherited /
already covered (test_adaptive_mcmc, test_bayesian_diagnostics).  What is
specific to this class and tested here:

  * ``choose_new_pset`` — a *fixed-magnitude* random-walk proposal: a random
    direction scaled so the step vector has L2 norm exactly ``step_size``.  This
    differs from Adaptive_MCMC's per-coordinate N(0, step_size) walk, so it is
    not a duplicate.  Oracles: ‖proposed − current‖ = step_size exactly, the
    direction is isotropic, and an out-of-box move returns None.
  * ``replica_exchange`` — the parallel-tempering swap acceptance
    ln_p = min(0, −(β_lo − β_hi)(P_lo − P_hi)) and the state swap it performs.
    Oracle: moving the better posterior to the colder chain is always accepted
    (and swaps current_pset / ln_current_P), the reverse is (essentially) never.
  * ``should_sample`` — only max-β replicas are sampled under pt; all under mcmc.
  * ``start_run`` — refuses a run with max_iterations ≤ burn_in (no samples).

Construction follows test_dream_class / test_adaptive_mcmc: a real
config.Configuration over the three v*__FREE params in parabola.bngl.
"""
import os

import numpy as np
import pytest

from .context import algorithms, config, pset, printing

PARABOLA = 'bngl_files/parabola.bngl'


def _make_config(tmp_path, var_spec, fit_type='mh', **overrides):
    out = str(tmp_path) + '/'
    os.makedirs(out + 'Results/Histograms', exist_ok=True)
    base = {
        'population_size': 4, 'max_iterations': 1000, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 1000, 'burn_in': 0,
        'output_every': 1000, 'credible_intervals': [68, 95], 'num_bins': 10,
        'output_dir': out, 'models': {PARABOLA}, 'exp_data': {'bngl_files/par1.exp'},
        'initialization': 'lh', PARABOLA: ['bngl_files/par1.exp'],
        'fit_type': fit_type,
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


# --------------------------------------------------------------------------- #
# choose_new_pset: fixed-magnitude random-walk proposal
# --------------------------------------------------------------------------- #
class TestChooseNewPset:

    def test_step_magnitude_is_exactly_step_size(self, tmp_path):
        """The proposal draws a random Gaussian direction and rescales it so the
        per-parameter deltas have L2 norm exactly step_size: delta_norm =
        step_size * g / ‖g‖. Hence ‖proposed − current‖ = step_size on every
        draw, independent of the random direction. (A missing normalization
        would give a Chi-distributed, not constant, magnitude.)"""
        step = 0.37
        cfg = _make_config(tmp_path, NORMAL_VARS, step_size=step)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        old = _normal_pset((5.0, -2.0, 1.0))     # unbounded -> never rejected
        old_vec = algo._param_vec(old)

        algo.chain_rngs[0] = np.random.default_rng(7)
        for _ in range(200):
            new = algo.choose_new_pset(old, 0)
            jump = algo._param_vec(new) - old_vec
            np.testing.assert_allclose(np.linalg.norm(jump), step, rtol=1e-12)

    def test_direction_is_isotropic(self, tmp_path):
        """The normalized direction is uniform on the sphere, so averaged over
        many draws the mean step is ~0 in every coordinate (no preferred
        direction). SEM per axis ~ step/sqrt(3*N); 4000 draws -> ~0.0017, test
        at a generous 0.02."""
        step = 0.2
        cfg = _make_config(tmp_path, NORMAL_VARS, step_size=step)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        old = _normal_pset((5.0, 5.0, 5.0))
        old_vec = algo._param_vec(old)
        algo.chain_rngs[0] = np.random.default_rng(11)
        jumps = np.array([algo._param_vec(algo.choose_new_pset(old, 0)) - old_vec
                          for _ in range(4000)])
        np.testing.assert_allclose(jumps.mean(axis=0), np.zeros(3), atol=0.02)

    def test_bounded_move_reflects_into_box(self, tmp_path):
        """choose_new_pset calls FreeParameter.add with the default reflect=True,
        so a step that overshoots a box bound is *reflected* back inside rather
        than rejected: the proposal is non-None and every coordinate stays
        within [0, 10] even for a step_size far larger than the box."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, step_size=100.0)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        old = _uniform_pset((5.0, 5.0, 5.0))     # step_size 100 >> box width 10
        algo.chain_rngs[0] = np.random.default_rng(3)
        for _ in range(50):
            new = algo.choose_new_pset(old, 0)
            assert new is not None
            for name in ('v1__FREE', 'v2__FREE', 'v3__FREE'):
                assert 0.0 <= new[name] <= 10.0


# --------------------------------------------------------------------------- #
# start_run: guard against a run that would collect no samples
# --------------------------------------------------------------------------- #
class TestStartRunValidation:

    @pytest.mark.parametrize("max_iter,burn_in", [
        (5, 10),     # strictly fewer iterations than burn-in
        (10, 10),    # equal — the boundary the <= guards (no post-burn samples)
    ])
    def test_rejects_max_iterations_not_above_burn_in(self, tmp_path, max_iter, burn_in):
        """If max_iterations <= burn_in the chain would never produce a
        post-burn-in sample, so start_run must raise rather than run a pointless
        fit. The equal case pins the <= (a strict < would wrongly allow it)."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, max_iterations=max_iter, burn_in=burn_in)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        with pytest.raises(printing.PybnfError):
            algo.start_run()


# --------------------------------------------------------------------------- #
# should_sample: which replicas contribute to the posterior
# --------------------------------------------------------------------------- #
class TestShouldSample:

    def test_mcmc_samples_every_replica(self, tmp_path):
        """Plain MCMC (no tempering) samples all replicas."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, population_size=4)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        assert all(algo.should_sample(i) for i in range(4))

    def test_pt_samples_only_max_beta_replicas(self, tmp_path):
        """Under parallel tempering only the cold (max-beta) replica of each
        group is sampled: with betas_per_group=3, indices 2 and 5 (the last of
        each group of 3) sample, the others do not."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, fit_type='pt',
                           population_size=6, beta=[0.5, 0.8, 1.0],
                           reps_per_beta=2, exchange_every=10)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        assert algo.betas_per_group == 3
        sampled = [i for i in range(6) if algo.should_sample(i)]
        assert sampled == [2, 5]


# --------------------------------------------------------------------------- #
# replica_exchange: the parallel-tempering swap
# --------------------------------------------------------------------------- #
class TestReplicaExchange:

    def _pt_algo(self, tmp_path):
        cfg = _make_config(tmp_path, NORMAL_VARS, fit_type='pt',
                           population_size=2, beta=[0.5, 1.0], reps_per_beta=1,
                           exchange_every=10000, max_iterations=10000)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        algo.start_run()
        algo.iteration = [5, 5]
        return algo

    def test_better_hot_replica_swaps_to_cold(self, tmp_path):
        """Index 0 is the hotter chain (beta=0.5), index 1 the colder (beta=1.0).
        When the hot chain holds the *better* (higher) posterior, the exchange
        criterion ln_p = min(0, −(β1−β0)(P1−P0)) evaluates to 0 (always accept),
        so the algorithm moves the good state to the cold chain. Oracle: after
        the exchange, current_pset and ln_current_P at 0 and 1 are swapped."""
        algo = self._pt_algo(tmp_path)
        hot, cold = _normal_pset((1., 2., 3.)), _normal_pset((4., 5., 6.))
        algo.current_pset = [hot, cold]
        algo.ln_current_P = [10.0, 0.0]          # hot (idx 0) is better
        algo.replica_exchange()
        assert algo.current_pset[0] is cold and algo.current_pset[1] is hot
        assert algo.ln_current_P[0] == 0.0 and algo.ln_current_P[1] == 10.0

    def test_better_cold_replica_does_not_swap(self, tmp_path):
        """When the cold chain is already far better, the swap probability
        exp(−(β1−β0)(P1−P0)) underflows to ~0, so no exchange occurs and the
        state is unchanged. (P1−P0 = 1000, β1−β0 = 0.5 -> exp(−500) ≈ 0.)"""
        algo = self._pt_algo(tmp_path)
        hot, cold = _normal_pset((1., 2., 3.)), _normal_pset((4., 5., 6.))
        algo.current_pset = [hot, cold]
        algo.ln_current_P = [0.0, 1000.0]        # cold (idx 1) is better
        algo.replica_exchange()
        assert algo.current_pset[0] is hot and algo.current_pset[1] is cold
        assert algo.ln_current_P[0] == 0.0 and algo.ln_current_P[1] == 1000.0


# --------------------------------------------------------------------------- #
# Driven MCMC run: the got_result / try_to_choose_new_pset loop to completion
# --------------------------------------------------------------------------- #
class TestDrivenMcmcRun:

    def test_run_terminates_at_max_iterations(self, tmp_path):
        """Drive a full plain-MCMC run end to end. Feeding scored results until
        the scheduler says 'STOP' must (a) actually stop, (b) leave every chain
        at >= max_iterations, and (c) keep the bookkeeping consistent
        (0 <= accepted <= attempts). This exercises the accept/reject body, the
        iteration-advance loop, periodic output, and the end-of-run STOP path."""
        # output_every beyond max_iterations skips output_results(), which is
        # base-class trajectory plumbing that needs a real backend (out of scope).
        cfg = _make_config(tmp_path, NORMAL_VARS, population_size=2,
                           max_iterations=12, burn_in=2, sample_every=2,
                           output_hist_every=2, output_every=10000, step_size=0.2)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        queue = list(algo.start_run())

        rng = np.random.default_rng(0)
        stopped = False
        guard = 0
        while queue and guard < 10000:
            guard += 1
            ps = queue.pop(0)
            res = algorithms.Result(ps, {}, ps.name)
            res.score = float(rng.uniform(5, 15))
            out = algo.got_result(res)
            if out == 'STOP':
                stopped = True
                break
            queue.extend(out)

        assert stopped
        assert min(algo.iteration) >= algo.max_iterations
        assert 0 <= algo.accepted <= algo.attempts


# --------------------------------------------------------------------------- #
# _pset_from_u: the inverse PSet bridge, hoisted onto Algorithm (#412)
# --------------------------------------------------------------------------- #
class TestPsetFromUBridge:
    """``_pset_from_u`` is the inverse peer of ``_param_vec``, hoisted onto
    ``Algorithm`` so the u-vector↔PSet conversion lives in one place (#412).
    Round-trips a PSet through u and back (linear and log10 parameters), and
    rejects an out-of-box coordinate when ``reflect=False`` (the DREAM path)."""

    _NAMES = ('v1__FREE', 'v2__FREE', 'v3__FREE')

    def test_round_trips_linear_params(self, tmp_path):
        cfg = _make_config(tmp_path, NORMAL_VARS)            # unbounded -> no reflect
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        ps = _normal_pset((5.0, -2.0, 1.3))
        back = algo._pset_from_u(algo._param_vec(ps))
        for name in self._NAMES:
            np.testing.assert_allclose(back[name], ps[name], rtol=1e-12)

    def test_round_trips_log10_params(self, tmp_path):
        """loguniform parameters live in log10 space, so the bridge must invert
        the forward log10 with 10**u, bit-for-bit."""
        spec = {('loguniform_var', n): [0.01, 100.0] for n in self._NAMES}
        cfg = _make_config(tmp_path, spec)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        ps = pset.PSet([pset.FreeParameter(n, 'loguniform_var', 0.01, 100.0, val)
                        for n, val in zip(self._NAMES, (0.05, 3.0, 42.0))])
        back = algo._pset_from_u(algo._param_vec(ps))
        for name in self._NAMES:
            np.testing.assert_allclose(back[name], ps[name], rtol=1e-12)

    def test_reflect_false_rejects_out_of_box(self, tmp_path):
        cfg = _make_config(tmp_path, UNIFORM_VARS)           # box [0, 10]
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        with pytest.raises(pset.OutOfBoundsException):
            algo._pset_from_u(np.array([20.0, 5.0, 5.0]), reflect=False)

    def test_name_is_applied(self, tmp_path):
        cfg = _make_config(tmp_path, NORMAL_VARS)
        algo = algorithms.BasicBayesMCMCAlgorithm(cfg)
        ps = algo._pset_from_u(np.zeros(3), name='probe')
        assert ps.name == 'probe'
