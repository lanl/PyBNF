"""
Oracle-anchored tests for the Metropolis acceptance rule and the random-walk
proposal in ``Adaptive_MCMC`` (the ``am`` sampler, pybnf/algorithms.py).

The two sampler contracts under test:

  * **Acceptance** (in ``got_result``): a candidate with higher log-posterior is
    always accepted; otherwise it is accepted with probability
    alpha = exp(ln_posterior_new - ln_posterior_current). This is the
    Metropolis ratio — the heart of MCMC correctness. We check the acceptance
    *probability* alpha both as an exact formula and as an empirical acceptance
    frequency over many trials.
  * **Proposal** (in ``pick_new_pset``, pre-adaptation branch): a Gaussian
    random walk centred on the current point with per-parameter standard
    deviation equal to ``step_size``. We verify the proposal distribution
    (mean, std, independence) over many draws.

Uniform priors are used so that ``ln_prior`` is constant inside the box and the
acceptance ratio reduces to a function of the scores alone; unbounded
``normal_var`` parameters are used for the proposal so no box cuts the Gaussian
off. What happens at the box -- a proposal that leaves it is a rejection, which
is the rule that makes the two contracts fit together -- has its own section
below (#709).
"""
from pathlib import Path

import numpy as np
import pytest

from .context import algorithms, config, pset, constraint, data
from pybnf.printing import PybnfError


class _FixedMvnRng:
    """Chain Generator stand-in returning a fixed multivariate_normal draw,
    delegating all other draws to a real Generator. Used because Generator methods
    are read-only and cannot be monkeypatched in place."""

    def __init__(self, mvn, seed=0):
        self._mvn = mvn
        self._g = np.random.default_rng(seed)

    def multivariate_normal(self, mean, cov, *a, **k):
        return self._mvn(mean, cov)

    def __getattr__(self, name):
        if name in ('_mvn', '_g'):
            raise AttributeError(name)
        return getattr(self._g, name)


# All three FREE parameters declared by tests/bngl_files/parabola.bngl must be
# present or model setup fails.
PARABOLA = 'bngl_files/parabola.bngl'


def _make_config(tmp_path, num_parallel, var_spec, **overrides):
    out = str(tmp_path) + '/'
    base = {
        'population_size': num_parallel, 'max_iterations': 100000, 'step_size': 0.2,
        'output_hist_every': 5, 'sample_every': 2, 'burn_in': 1000, 'adaptive': 1000,
        'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': out,
        'models': {PARABOLA}, 'exp_data': {'bngl_files/par1.exp'},
        'initialization': 'lh', PARABOLA: ['bngl_files/par1.exp'], 'fit_type': 'am',
    }
    base.update(var_spec)
    base.update(overrides)
    return config.Configuration(base)


# Three unbounded normal_var parameters (no reflection in proposals).
NORMAL_VARS = {
    ('normal_var', 'v1__FREE'): [0.0, 1.0],
    ('normal_var', 'v2__FREE'): [0.0, 1.0],
    ('normal_var', 'v3__FREE'): [0.0, 1.0],
}
# Three box-uniform parameters (constant prior inside the box).
UNIFORM_VARS = {
    ('uniform_var', 'v1__FREE'): [0.0, 100.0],
    ('uniform_var', 'v2__FREE'): [0.0, 100.0],
    ('uniform_var', 'v3__FREE'): [0.0, 100.0],
}
# Three log-space (base-10) parameters, bounds [1, 1e4] -> log10 in [0, 4].
LOGUNIFORM_VARS = {
    ('loguniform_var', 'v1__FREE'): [1.0, 1.0e4],
    ('loguniform_var', 'v2__FREE'): [1.0, 1.0e4],
    ('loguniform_var', 'v3__FREE'): [1.0, 1.0e4],
}


def _fake_result(ps, score):
    """A scored Result with .out set so got_result's accepted-branch runs
    without a real simulation backend (parallelize_models defaults to 1)."""
    res = algorithms.Result(ps, {'parabola': {}}, ps.name)
    res.score = score
    res.out = res.simdata
    return res


def _uniform_pset(name, values=(50.0, 50.0, 50.0)):
    ps = pset.PSet([
        pset.FreeParameter('v1__FREE', 'uniform_var', 0.0, 100.0, values[0]),
        pset.FreeParameter('v2__FREE', 'uniform_var', 0.0, 100.0, values[1]),
        pset.FreeParameter('v3__FREE', 'uniform_var', 0.0, 100.0, values[2]),
    ])
    ps.name = name
    return ps


# --------------------------------------------------------------------------- #
# Metropolis acceptance rule
# --------------------------------------------------------------------------- #
class TestAcceptance:

    def test_first_sample_always_accepted(self, tmp_path):
        """The initial log-posterior is NaN, which forces acceptance of the
        first result on every chain regardless of its score."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        start = am.start_run()
        res = _fake_result(start[0], score=1e6)  # arbitrarily bad
        am.got_result(res)
        assert am.accept is True
        assert am.alpha[0] == 1
        assert am.current_pset[0] is start[0]

    def test_better_proposal_always_accepted(self, tmp_path):
        """A candidate with strictly higher log-posterior (lower score, equal
        prior) is accepted deterministically with alpha set to 1."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        cand = _uniform_pset('iter5run0')
        lp = am.ln_prior(cand)
        am.current_pset[0] = _uniform_pset('iter4run0')
        am.iteration[0] = 5
        # current posterior worse than candidate's (-10 + lp)
        am.ln_current_P[0] = (-10.0 + lp) - 5.0
        am.got_result(_fake_result(cand, score=10.0))
        assert am.accept is True
        assert am.alpha[0] == 1
        assert am.current_pset[0] is cand

    def test_worse_proposal_alpha_is_metropolis_ratio(self, tmp_path):
        """For a worse candidate, alpha must equal exp(ln_post_new - ln_post_cur).
        We pin the current posterior one unit above the candidate's, so the
        Metropolis ratio is exactly exp(-1)."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        cand = _uniform_pset('iter5run0')
        ln_post_new = -10.0 + am.ln_prior(cand)
        am.current_pset[0] = _uniform_pset('iter4run0')
        am.iteration[0] = 5
        am.ln_current_P[0] = ln_post_new + 1.0  # candidate is worse by exactly 1
        am.got_result(_fake_result(cand, score=10.0))
        np.testing.assert_allclose(am.alpha[0], np.exp(-1.0), rtol=1e-12)

    def test_acceptance_frequency_matches_alpha(self, tmp_path):
        """Empirical Metropolis oracle: with alpha fixed at 0.4, the fraction of
        worse proposals accepted over many independent draws must match 0.4.
        Reporting only chain 0 (of 2) keeps the generation un-synced, so the
        only random draw per trial is the accept/reject uniform."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        base = _uniform_pset('iter4run0')
        cand = _uniform_pset('iter5run0')
        ln_post_new = -10.0 + am.ln_prior(cand)
        target_alpha = 0.4
        ln_cur = ln_post_new - np.log(target_alpha)  # exp(new - cur) = target_alpha

        N = 2000
        am.chain_rngs[0] = np.random.default_rng(20240517)
        accepts = 0
        for _ in range(N):
            am.iteration[0] = 5
            am.wait_for_sync = [False, False]
            am.current_pset[0] = base
            am.ln_current_P[0] = ln_cur
            am.accept = False
            am.got_result(_fake_result(cand, score=10.0))
            accepts += int(am.accept)

        freq = accepts / N
        # Binomial std ~ sqrt(0.4*0.6/2000) ~ 0.011; 5-sigma window.
        assert abs(freq - target_alpha) < 0.055


# --------------------------------------------------------------------------- #
# Random-walk proposal (pre-adaptation branch)
# --------------------------------------------------------------------------- #
class TestProposal:

    def test_random_walk_is_gaussian_step_size(self, tmp_path):
        """Before adaptation kicks in, pick_new_pset proposes
        current + step_size * N(0, I). Over many draws the per-parameter step
        (proposed - current) is zero-mean Gaussian with std == step_size, and
        the parameters are mutually independent."""
        cfg = _make_config(tmp_path, 2, NORMAL_VARS, step_size=0.2)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()

        current_vals = (5.0, 5.0, 5.0)  # far from any bound (normal_var unbounded)
        base = pset.PSet([
            pset.FreeParameter('v1__FREE', 'normal_var', 0.0, 1.0, current_vals[0]),
            pset.FreeParameter('v2__FREE', 'normal_var', 0.0, 1.0, current_vals[1]),
            pset.FreeParameter('v3__FREE', 'normal_var', 0.0, 1.0, current_vals[2]),
        ])
        am.current_pset = [base, base]
        am.iteration = [0, 0]  # well before burn_in + adaptive -> random-walk branch

        names = [v.name for v in am.variables]
        am.chain_rngs[0] = np.random.default_rng(7)
        N = 4000
        deltas = np.empty((N, 3))
        for i in range(N):
            prop = am.pick_new_pset(0)
            deltas[i] = [prop[n] - dict(zip(names, current_vals))[n] for n in names]

        # Mean ~ 0: SEM = 0.2/sqrt(4000) ~ 0.0032, allow ~6 SEM.
        np.testing.assert_allclose(deltas.mean(axis=0), np.zeros(3), atol=0.02)
        # Std ~ step_size = 0.2 within 8%.
        np.testing.assert_allclose(deltas.std(axis=0, ddof=1), [0.2, 0.2, 0.2], rtol=0.08)
        # Independent across parameters: off-diagonal correlation ~ 0.
        corr = np.corrcoef(deltas, rowvar=False)
        off_diag = corr[~np.eye(3, dtype=bool)]
        assert np.max(np.abs(off_diag)) < 0.1

    def test_proposal_preserves_parameter_names(self, tmp_path):
        """A proposed PSet carries exactly the model's free-parameter names."""
        cfg = _make_config(tmp_path, 2, NORMAL_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        base = pset.PSet([
            pset.FreeParameter('v1__FREE', 'normal_var', 0.0, 1.0, 5.0),
            pset.FreeParameter('v2__FREE', 'normal_var', 0.0, 1.0, 5.0),
            pset.FreeParameter('v3__FREE', 'normal_var', 0.0, 1.0, 5.0),
        ])
        am.current_pset = [base, base]
        am.iteration = [0, 0]
        prop = am.pick_new_pset(0)
        assert set(prop.keys()) == {'v1__FREE', 'v2__FREE', 'v3__FREE'}


# --------------------------------------------------------------------------- #
# Adaptive proposal — the post-burn-in branch of pick_new_pset.
#
# Once iteration >= burn_in + adaptive, pick_new_pset switches from the fixed
# random walk to the Haario (2001) adaptive-Metropolis kernel: it seeds a
# proposal covariance from the recorded chain, then on every step does a
# Robbins-Monro update of the running mean, the covariance, and a global scale
# `diff`, and proposes current + diff * N(0, Sigma). The math under test:
#
#   * seeding scale  = 2.38^2 / d           (Gelman-Roberts optimal scaling)
#   * seeding cov    = X^T X / n - mu mu^T + eps*I   (sample cov + stabilizer)
#   * scale update   driven by (alpha - 0.234)       (0.234 optimal acceptance)
#   * proposal       = current + diff * delta, delta ~ N(0, Sigma_adapted)
#
# The accept/reject rule and the pre-adaptation random walk are covered above.
# These tests freeze the Gaussian draw (swapping in a chain Generator stand-in
# whose multivariate_normal is fixed) so each contract is checked against its
# analytical oracle, not RNG.
# --------------------------------------------------------------------------- #
OPTIMAL_ACCEPT = 0.234  # Gelman/Roberts optimal Metropolis acceptance rate

# A 5-sample, 3-parameter chain with strong anti-correlation between the first
# two coordinates (so a wrong covariance computation is visible off-diagonal).
CHAIN_X = np.array([
    [40., 60., 50.],
    [45., 55., 52.],
    [50., 50., 48.],
    [55., 45., 51.],
    [60., 40., 49.],
])


def _adaptive_am(tmp_path, *, burn_in, adaptive, stablizingCov=0.01, num_parallel=2):
    cfg = _make_config(tmp_path, num_parallel, UNIFORM_VARS,
                       burn_in=burn_in, adaptive=adaptive, stablizingCov=stablizingCov)
    am = algorithms.Adaptive_MCMC(cfg)
    am.start_run()
    return am


def _write_chain_file(am, idx, X):
    """Write X as the per-chain history file pick_new_pset's seeding step reads,
    with a plain (un-commented) header of the parameter names in `variables`
    order so np.genfromtxt(names=True) recovers the columns."""
    names = [v.name for v in am.variables]
    path = am.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt'
    np.savetxt(path, X, header=' '.join(names), comments='')


def _pset_from_values(am, values, vartype='uniform_var', lo=0.0, hi=100.0):
    """Build a PSet with one parameter per model variable, in `variables` order."""
    ps = pset.PSet([pset.FreeParameter(v.name, vartype, lo, hi, float(values[i]))
                    for i, v in enumerate(am.variables)])
    ps.name = 'iter0run0'
    return ps


def _freeze_draw_to_zero(monkeypatch, am):
    """Replace the Gaussian proposal draw with the zero vector so pick_new_pset's
    state updates (mean/cov/scale) are isolated and the trivial step lands in
    bounds on the first try. Swaps in a stand-in for each chain's Generator (whose
    methods are read-only)."""
    monkeypatch.setattr(am, 'chain_rngs',
                        [_FixedMvnRng(lambda mean, cov: np.zeros(len(mean)))] * am.num_parallel)


class TestAdaptiveScaling:

    def test_seeded_to_optimal_2_38_sq_over_d(self, tmp_path, monkeypatch):
        """At the first adaptive iteration (it == burn_in + adaptive) the global
        proposal scale is seeded to the Gelman-Roberts optimum 2.38^2/d. With the
        acceptance rate at the 0.234 target the same-step Robbins-Monro nudge
        vanishes (and its 1/(1+it-adaptive-burn_in) factor is 1 here), so the
        post-call scale is exactly 2.38^2/d."""
        burn_in, adaptive = 2, 5
        am = _adaptive_am(tmp_path, burn_in=burn_in, adaptive=adaptive)
        d = len(am.variables)
        _write_chain_file(am, 0, CHAIN_X)
        am.current_pset[0] = _pset_from_values(am, CHAIN_X.mean(0))
        am.iteration[0] = burn_in + adaptive
        am.alpha[0] = OPTIMAL_ACCEPT
        _freeze_draw_to_zero(monkeypatch, am)

        am.pick_new_pset(0)

        np.testing.assert_allclose(am.diff[0], 2.38 ** 2 / d, rtol=1e-12)

    @pytest.mark.parametrize("alpha, expect", [(1.0, 'up'), (0.0, 'down'),
                                               (OPTIMAL_ACCEPT, 'same')])
    def test_robbins_monro_adapts_toward_optimal_acceptance(self, tmp_path, monkeypatch,
                                                            alpha, expect):
        """Past seeding, the log-scale is nudged by (alpha - 0.234): an acceptance
        rate above the 0.234 optimum grows the step, below shrinks it, exactly at
        it leaves the step unchanged. This pins both the sign of the update and
        the 0.234 fixed point."""
        burn_in, adaptive = 2, 5
        am = _adaptive_am(tmp_path, burn_in=burn_in, adaptive=adaptive)
        d = len(am.variables)
        base_vals = np.full(d, 50.0)
        am.current_pset[0] = _pset_from_values(am, base_vals)
        am.iteration[0] = burn_in + adaptive + 1     # past the seeding step
        am.mu[0] = base_vals.reshape(1, d)           # zero innovation -> isolate the scale
        am.diffMatrix[0] = 0.01 * np.eye(d)
        D = 0.5
        am.diff[0] = D
        am.alpha[0] = alpha
        _freeze_draw_to_zero(monkeypatch, am)

        am.pick_new_pset(0)

        if expect == 'up':
            assert am.diff[0] > D
        elif expect == 'down':
            assert am.diff[0] < D
        else:
            np.testing.assert_allclose(am.diff[0], D, rtol=1e-12)


class TestAdaptiveCovariance:

    def test_seeded_from_chain_sample_covariance(self, tmp_path, monkeypatch):
        """At the seeding step the proposal covariance is the chain's sample
        covariance (X^T X / n - mu mu^T, the 1/n estimator) plus the stabilizing
        eps*I. We pin the post-call matrix including the single Robbins-Monro EMA
        step: with the current point set to the chain mean the innovation is zero,
        so that step shrinks only the sample-cov part by the EMA weight (1-w),
        leaving cov_pop * (1-w) + eps*I.

        The EMA weight is w = 1/(1 + iteration - burn_in) (AM-2): the running
        sample count is samples-since-burn_in, matching the seed (built from the
        `adaptive` post-burn-in history rows). At the seeding step iteration ==
        burn_in + adaptive, so w = 1/(1 + adaptive) and 1-w = adaptive/(1+adaptive).
        The pre-AM-2 code used the global counter w = 1/(1+iteration), i.e.
        1-w = iteration/(1+iteration) -- a different (stickier) factor."""
        burn_in, adaptive, eps = 2, 5, 0.01
        am = _adaptive_am(tmp_path, burn_in=burn_in, adaptive=adaptive, stablizingCov=eps)
        d = len(am.variables)
        _write_chain_file(am, 0, CHAIN_X)
        am.current_pset[0] = _pset_from_values(am, CHAIN_X.mean(0))
        it = burn_in + adaptive
        am.iteration[0] = it
        am.alpha[0] = OPTIMAL_ACCEPT
        _freeze_draw_to_zero(monkeypatch, am)

        am.pick_new_pset(0)

        mu = CHAIN_X.mean(0)
        cov_pop = CHAIN_X.T @ CHAIN_X / adaptive - np.outer(mu, mu)  # n == adaptive rows
        one_minus_w = adaptive / (1 + adaptive)                      # w = 1/(1+it-burn_in)
        oracle = cov_pop * one_minus_w + eps * np.eye(d)
        np.testing.assert_allclose(am.diffMatrix[0], oracle, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# AM-2: the online mean/covariance recurrence must weight each new sample by
# 1/(samples folded so far + 1), where the count is samples-since-burn_in
# (iteration - burn_in), NOT the global iteration. The seed holds `adaptive`
# samples (its divisor is iteration - burn_in == adaptive), so a consistent
# running estimator continues from that count. Using the global counter
# under-weights new samples by ~(1+iteration)/(1+adaptive) at the seeding step,
# freezing the proposal near the seed. The contract is invariant to burn_in for
# a fixed (iteration - burn_in); the pre-AM-2 global-counter weight was not.
# --------------------------------------------------------------------------- #
class TestAdaptiveCovarianceSampleWeight:

    def _run_one_ema_step(self, tmp_path, monkeypatch, burn_in, k, diff_pre, eps):
        """Drive a single post-seeding EMA update (iteration = burn_in+adaptive+k,
        k>=1, so the re-seed branch is skipped). Current point is pinned to the
        running mean (zero innovation), so the covariance update reduces to
        diff_pre*(1-w) + w*eps*I with w the per-sample weight under test.
        Returns the post-update diffMatrix[0]."""
        adaptive = 5
        am = _adaptive_am(tmp_path, burn_in=burn_in, adaptive=adaptive, stablizingCov=eps)
        d = len(am.variables)
        base_vals = np.full(d, 50.0)
        am.current_pset[0] = _pset_from_values(am, base_vals)
        am.mu[0] = base_vals.reshape(1, d)            # zero innovation -> isolate weight
        am.diffMatrix[0] = diff_pre.copy()
        am.iteration[0] = burn_in + adaptive + k      # k>=1 -> no re-seed
        am.alpha[0] = OPTIMAL_ACCEPT                  # freeze the scalar `diff`
        _freeze_draw_to_zero(monkeypatch, am)
        am.pick_new_pset(0)
        return np.array(am.diffMatrix[0]), d

    def test_ema_weight_is_invariant_to_burn_in(self, tmp_path, monkeypatch):
        """For a fixed number of samples-since-burn_in (here k+adaptive), the
        covariance EMA weight must be identical regardless of burn_in. Two runs
        with burn_in 3 vs 500 but the same post-seeding offset k must produce the
        same updated covariance. The global-counter bug makes the weights differ
        (1/(1+503+...) vs 1/(1+6+...)), so this assertion is the AM-2
        discriminator. eps=0 isolates the shrink factor exactly."""
        d = 3
        diff_pre = np.full((d, d), 0.4) + 2.0 * np.eye(d)
        eps = 0.0
        k = 1
        m_small, _ = self._run_one_ema_step(tmp_path / 'a', monkeypatch, 3, k, diff_pre, eps)
        m_large, _ = self._run_one_ema_step(tmp_path / 'b', monkeypatch, 500, k, diff_pre, eps)
        np.testing.assert_allclose(m_small, m_large, rtol=1e-12, atol=1e-14)

    def test_ema_weight_matches_samples_since_burn_in(self, tmp_path, monkeypatch):
        """Pin the exact post-update covariance to the analytical EMA with weight
        w = 1/(1 + iteration - burn_in). With zero innovation and eps=0 the update
        is diff_pre*(1-w); here iteration-burn_in = adaptive + k so w = 1/(1+5+1).
        The old global-counter weight 1/(1+iteration) = 1/(1+burn_in+5+1) would
        give a materially larger 1-w, so this both fixes the value and guards the
        discriminator."""
        burn_in, adaptive, k, eps = 50, 5, 1, 0.0
        d = 3
        diff_pre = np.full((d, d), 0.4) + 2.0 * np.eye(d)
        m_post, _ = self._run_one_ema_step(tmp_path, monkeypatch, burn_in, k, diff_pre, eps)
        w = 1.0 / (1 + adaptive + k)               # 1/(1 + iteration - burn_in)
        oracle = diff_pre * (1 - w)
        np.testing.assert_allclose(m_post, oracle, rtol=1e-12, atol=1e-14)
        # Guard: the old global-counter weight is materially different, so this
        # test actually distinguishes the two implementations.
        w_bug = 1.0 / (1 + burn_in + adaptive + k)
        assert not np.allclose(m_post, diff_pre * (1 - w_bug), rtol=1e-3)


# --------------------------------------------------------------------------- #
# AM-1: the adaptive proposal must work in the SAME log base it is applied in.
#
# For log-space parameters, pick_new_pset builds its working vector and seeds
# the covariance from the chain history in base-10 log (np.log10), matching how
# the proposal is applied (FreeParameter.add -> 10**(log10(value)+summand)) and
# the rest of the codebase (loguniform_var dist, prior_logpdf, _param_vec R-hat
# history, FreeParameter.diff). A natural-log implementation (the AM-1 bug)
# learns a covariance off by (ln 10)^2 ~ 5.3 per log axis and mis-shaped on
# mixed linear+log targets — hurting mixing while leaving the posterior intact.
# --------------------------------------------------------------------------- #
# Regular-space chain values for three log-space params: well-separated decades
# so the log10-vs-ln distinction is unmistakable, with anti-correlation between
# the first two coords so a wrong base is visible off-diagonal too.
CHAIN_LOG = np.array([
    [10.,    1000.,  100.],
    [31.62,  316.2,  120.],
    [100.,   100.,   90.],
    [316.2,  31.62,  110.],
    [1000.,  10.,    95.],
])


class TestAdaptiveCovarianceLogSpace:

    def test_seeded_covariance_is_in_base10_log_space(self, tmp_path, monkeypatch):
        """For loguniform params the seeded proposal covariance is the sample
        covariance of log10(value) (+ stabilizer), NOT log_e(value).

        Same seeding contract as test_seeded_from_chain_sample_covariance, but
        the oracle is built from log10 of the chain. The natural-log bug would
        inflate every entry of cov_pop by (ln 10)^2 ~ 5.3, so this assertion is
        the direct AM-1 discriminator. The covariance lives in the parameter
        sampling space, which for log params is base-10 (see _param_vec)."""
        burn_in, adaptive, eps = 2, 5, 0.01
        cfg = _make_config(tmp_path, 2, LOGUNIFORM_VARS,
                           burn_in=burn_in, adaptive=adaptive, stablizingCov=eps)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        d = len(am.variables)

        # History file stores regular-space values (write_out_params writes the
        # raw parameter values); pick_new_pset is responsible for the log10.
        _write_chain_file(am, 0, CHAIN_LOG)
        # Current point at the chain's log10-space mean -> zero innovation, so the
        # single Robbins-Monro EMA step only rescales the sample-cov part.
        log_chain = np.log10(CHAIN_LOG)
        regular_at_log_mean = 10.0 ** log_chain.mean(0)
        am.current_pset[0] = _pset_from_values(
            am, regular_at_log_mean, vartype='loguniform_var', lo=1.0, hi=1.0e4)
        it = burn_in + adaptive
        am.iteration[0] = it
        am.alpha[0] = OPTIMAL_ACCEPT
        _freeze_draw_to_zero(monkeypatch, am)

        am.pick_new_pset(0)

        # EMA shrink factor at the seeding step: 1-w with w = 1/(1+it-burn_in)
        # = 1/(1+adaptive) per AM-2, so 1-w = adaptive/(1+adaptive).
        one_minus_w = adaptive / (1 + adaptive)
        mu = log_chain.mean(0)
        cov_pop = log_chain.T @ log_chain / adaptive - np.outer(mu, mu)
        oracle = cov_pop * one_minus_w + eps * np.eye(d)
        np.testing.assert_allclose(am.diffMatrix[0], oracle, rtol=1e-10, atol=1e-12)
        # Guard the discriminator itself: the natural-log oracle is materially
        # different (factor ~ (ln 10)^2 on cov_pop), so this test actually
        # distinguishes the two implementations rather than passing vacuously.
        ln_chain = np.log(CHAIN_LOG)
        ln_mu = ln_chain.mean(0)
        ln_cov = ln_chain.T @ ln_chain / adaptive - np.outer(ln_mu, ln_mu)
        ln_oracle = ln_cov * one_minus_w + eps * np.eye(d)
        assert not np.allclose(am.diffMatrix[0], ln_oracle, rtol=1e-3)


class TestAdaptiveProposalDraw:

    def test_proposal_is_current_plus_scaled_draw_from_adapted_covariance(self, tmp_path,
                                                                         monkeypatch):
        """The adaptive proposal is current + diff * delta with delta ~ N(0, Sigma)
        and Sigma the just-updated covariance. Fixing delta to a known vector and
        intercepting the sampling covariance: each parameter moves by exactly
        diff*delta_i, and the matrix handed to the Gaussian draw is the current
        diffMatrix — so proposals inherit the estimated posterior correlations
        rather than stepping isotropically."""
        burn_in, adaptive = 2, 5
        am = _adaptive_am(tmp_path, burn_in=burn_in, adaptive=adaptive)
        d = len(am.variables)
        base_vals = np.full(d, 5.0)
        # normal_var is unbounded, so the (possibly large) step never reflects.
        base = _pset_from_values(am, base_vals, vartype='normal_var', lo=0.0, hi=1.0)
        am.current_pset[0] = base
        am.iteration[0] = burn_in + adaptive + 1
        am.mu[0] = base_vals.reshape(1, d)                       # zero innovation
        am.diffMatrix[0] = np.full((d, d), 0.5) + 1.5 * np.eye(d)
        D = 0.3
        am.diff[0] = D
        am.alpha[0] = OPTIMAL_ACCEPT                             # keep diff == D

        delta = np.arange(1.0, d + 1.0) * np.array([1.0, -1.0, 1.0][:d] + [1.0] * max(0, d - 3))
        rec = {}

        def fake_mvn(mean, cov):
            rec['cov'] = np.array(cov)
            rec['calls'] = rec.get('calls', 0) + 1
            return delta
        monkeypatch.setattr(am, 'chain_rngs', [_FixedMvnRng(fake_mvn)] * am.num_parallel)

        prop = am.pick_new_pset(0)

        names = list(base.keys())
        for i, n in enumerate(names):
            np.testing.assert_allclose(prop[n] - base[n], D * delta[i], rtol=1e-12)
        # The draw sampled from the adapted (post-update) covariance, once.
        np.testing.assert_allclose(rec['cov'], am.diffMatrix[0], rtol=1e-12)
        assert rec['calls'] == 1


# --------------------------------------------------------------------------- #
# A proposal that leaves the box is rejected, not redrawn (#709)
#
# pick_new_pset used to redraw the Gaussian step until it landed inside the box, and
# got_result then accepted it with the plain Metropolis ratio. That ratio is right only
# for a symmetric proposal, and a redrawn one is not: its density is the Gaussian
# renormalized over the box, q(x -> y) = phi(y - x) / Z(x), where Z(x) is the share of
# the Gaussian centred on x that lies inside, and Z is smallest at a wall. The chain
# then samples pi(x) Z(x) rather than pi(x), thin along every wall -- on a flat
# posterior over [0, 1] with the default step of 0.2, 0.71 of the right density in the
# outermost tenth and 1.17 in the middle.
#
# The acceptance tests above and the proposal tests above each pinned their half in
# isolation and both passed throughout; nothing held the two together. What is pinned
# here is the rule that makes them fit: one draw, and a draw that leaves the box is a
# rejection that spends the iteration without a simulation. The stationary distribution
# itself is checked end to end in test_sampler_integration.py.
# --------------------------------------------------------------------------- #
class _ScriptedMvnRng(_FixedMvnRng):
    """Hands out the scripted steps in order, then keeps repeating the last, and counts
    the draws it was asked for."""

    def __init__(self, steps, seed=0):
        self.calls = 0
        steps = [np.asarray(s, dtype=float) for s in steps]

        def mvn(mean, cov):
            self.calls += 1
            return steps[min(self.calls, len(steps)) - 1]
        super().__init__(mvn, seed)


# In units of the draw, before the sampler's scale multiplies it. OUT carries v1 far past
# its upper bound from anywhere in [0, 100] at any scale these tests use; IN is a short
# step from the middle of the box.
OUT = [1e9, 0.0, 0.0]
IN = [1.0, -2.0, 3.0]


def _write_continue_run_files(tmp_path, d=3):
    """The files continue_run = 1 reads from a completed prior run."""
    adaptive_dir = tmp_path / 'adaptive_files'
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(adaptive_dir / 'diff.txt', [0.5])
    np.savetxt(adaptive_dir / 'MLE_params.txt', np.full(d, 50.0))
    np.savetxt(adaptive_dir / 'diffMatrix.txt', np.eye(d))


def _am_in_branch(tmp_path, branch):
    """An am whose next pick_new_pset(0) takes the named branch, from the middle of the
    box. Returns (am, the scale that branch multiplies its draw by)."""
    burn_in, adaptive = 2, 5
    if branch == 'continue_run':
        _write_continue_run_files(tmp_path)
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS, burn_in=burn_in, adaptive=adaptive,
                           continue_run=1)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
    else:
        am = _adaptive_am(tmp_path, burn_in=burn_in, adaptive=adaptive)
    d = len(am.variables)
    base_vals = np.full(d, 50.0)
    am.current_pset[0] = _pset_from_values(am, base_vals)
    if branch == 'adaptive':
        am.iteration[0] = burn_in + adaptive + 1     # past the seeding step
        am.mu[0] = base_vals.reshape(1, d)
        am.diffMatrix[0] = np.eye(d)
        am.diff[0] = 0.5
        am.alpha[0] = OPTIMAL_ACCEPT                 # leaves the scale where it is
        return am, 0.5
    am.iteration[0] = 1                              # before burn_in + adaptive
    return am, (float(am.diff[0]) if branch == 'continue_run' else am.step_size)


BRANCHES = ['fixed_step', 'adaptive', 'continue_run']


class TestAProposalThatLeavesTheBoxIsRejected:

    @pytest.mark.parametrize('branch', BRANCHES)
    def test_it_is_not_redrawn(self, tmp_path, branch):
        """Every branch of pick_new_pset draws once and answers None for a step that
        leaves the box. The redraw loop was written out three times, once per branch, so
        each is checked: a second draw here is the bug coming back."""
        am, _ = _am_in_branch(tmp_path, branch)
        rng = _ScriptedMvnRng([OUT])
        am.chain_rngs = [rng, rng]

        assert am.pick_new_pset(0) is None
        assert rng.calls == 1

    @pytest.mark.parametrize('branch', BRANCHES)
    def test_a_step_that_stays_inside_is_the_gaussian_step_unchanged(self, tmp_path, branch):
        """Inside the box nothing changed: the proposal is current + scale * draw, to the
        bit, so a run that never meets a wall reproduces what it did before."""
        am, scale = _am_in_branch(tmp_path, branch)
        am.chain_rngs = [_ScriptedMvnRng([IN])] * 2

        prop = am.pick_new_pset(0)

        for i, v in enumerate(am.variables):
            assert prop[v.name] == 50.0 + scale * IN[i]

    def test_one_coordinate_outside_rejects_the_whole_vector(self, tmp_path):
        """The proposal is one point. If any coordinate of it is outside the box the
        posterior there is zero, however the others landed."""
        am, _ = _am_in_branch(tmp_path, 'fixed_step')
        am.chain_rngs = [_ScriptedMvnRng([[1.0, -2.0, 1e9]])] * 2   # only the last leaves
        assert am.pick_new_pset(0) is None

    def test_the_box_of_a_log_parameter_is_met_in_log_space(self, tmp_path):
        """A loguniform_var steps in log10, so the wall at 1e4 is 0.1 above a chain
        sitting at 10**3.9: a step of +0.2 leaves the box and one of -0.2 does not."""
        cfg = _make_config(tmp_path, 2, LOGUNIFORM_VARS, step_size=1.0)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        am.current_pset[0] = _pset_from_values(am, np.full(3, 10 ** 3.9),
                                               vartype='loguniform_var', lo=1.0, hi=1.0e4)
        am.iteration[0] = 1

        am.chain_rngs = [_ScriptedMvnRng([[0.2, 0.0, 0.0]])] * 2
        assert am.pick_new_pset(0) is None

        am.chain_rngs = [_ScriptedMvnRng([[-0.2, 0.0, 0.0]])] * 2
        prop = am.pick_new_pset(0)
        np.testing.assert_allclose(prop['v1__FREE'], 10 ** 3.7, rtol=1e-12)


def _two_chains(tmp_path, scripts, **overrides):
    """A two-chain am on the box-uniform parameters, started, with each chain's Gaussian
    draws scripted. Returns (am, the two starting psets).

    The starting psets are the middle of the box, not the ones start_run drew: the config
    pins no seed, so those land anywhere, and from beside a wall a scripted IN step is an
    OUT step."""
    kw = dict(burn_in=1000, adaptive=1000)
    kw.update(overrides)
    cfg = _make_config(tmp_path, 2, UNIFORM_VARS, **kw)
    am = algorithms.Adaptive_MCMC(cfg)
    am.start_run()
    am.chain_rngs = [_ScriptedMvnRng(s) for s in scripts]
    return am, [_uniform_pset('iter0run0'), _uniform_pset('iter0run1')]


def _vec(am, ps):
    return [ps[v.name] for v in am.variables]


class TestARejectionAtTheBoundarySpendsTheIteration:
    """What the barrier does with the None. The score is held constant throughout, so
    every simulated proposal is accepted (ratio 1) and the only rejections are the ones
    at the boundary."""

    def test_the_chain_stays_put_and_nothing_is_submitted_for_it(self, tmp_path):
        am, start = _two_chains(tmp_path, [[OUT], [IN]])
        assert am.got_result(_fake_result(start[0], 5.0)) == []

        generation = am.got_result(_fake_result(start[1], 5.0))

        # Only chain 1 has something to simulate.
        assert [ps.name for ps in generation] == ['iter1run1']
        # Chain 0 is where it was, one iteration further on, and already waiting.
        assert am.current_pset[0] is start[0]
        assert am.iteration == [2, 1]
        assert am.wait_for_sync == [True, False]

    def test_the_rejected_iteration_is_recorded_as_the_current_point_again(self, tmp_path):
        """As for any rejected move: the chain's history gains the point it is standing
        on, with the posterior it already had."""
        am, start = _two_chains(tmp_path, [[OUT], [IN]])
        am.got_result(_fake_result(start[0], 5.0))
        am.got_result(_fake_result(start[1], 5.0))

        here = _vec(am, start[0])
        assert [list(row) for row in am.chain_history[0]] == [here, here]
        assert am.ln_posterior_history[0] == [am.ln_current_P[0]] * 2
        assert len(am.chain_history[1]) == 1

    def test_it_counts_as_an_attempt_with_alpha_zero_and_as_no_evaluation(self, tmp_path):
        """alpha is what the scale adaptation steers to 0.234
        (test_robbins_monro_adapts_toward_optimal_acceptance), so a zero here is what
        shortens the step of a chain that keeps walking into a wall. total_evaluations
        counts simulations, and none was run."""
        am, start = _two_chains(tmp_path, [[OUT], [IN]])
        am.got_result(_fake_result(start[0], 5.0))
        am.got_result(_fake_result(start[1], 5.0))

        assert am.alpha[0] == 0
        assert am.attempts == 3                  # two results and the rejection
        assert am.boundary_rejections == 1
        assert am.acceptances == 2
        assert am.total_evaluations == 2

    def test_the_generation_completes_when_the_other_chain_reports(self, tmp_path):
        """The rejected chain is not left behind or waited on: the next result closes the
        generation and both chains propose again, from the same iteration."""
        am, start = _two_chains(tmp_path, [[OUT, IN], [IN]])
        am.got_result(_fake_result(start[0], 5.0))
        generation = am.got_result(_fake_result(start[1], 5.0))

        generation = am.got_result(_fake_result(generation[0], 5.0))

        assert sorted(ps.name for ps in generation) == ['iter2run0', 'iter2run1']
        assert am.iteration == [2, 2]
        assert am.wait_for_sync == [False, False]

    def test_the_history_file_carries_a_row_for_the_rejected_iteration(self, tmp_path):
        """params_<chain>.txt seeds the adaptive covariance and is what the posterior is
        read from, one row per iteration from burn_in on. A rejected iteration is one of
        them, as the point the chain stayed on; dropping it would thin the record exactly
        where the chain presses against a wall, which is the bias over again."""
        burn_in = 2
        am, generation = _two_chains(tmp_path, [[IN, IN, OUT, IN, IN], [IN]],
                                     burn_in=burn_in, adaptive=50)
        for _ in range(6):
            nxt = []
            for ps in generation:
                nxt += am.got_result(_fake_result(ps, 5.0))
            generation = nxt

        rows = np.loadtxt(str(tmp_path / 'Results' / 'A_MCMC' / 'Runs' / 'params_0.txt'),
                          skiprows=1)
        history = np.array(am.chain_history[0])
        # Every iteration from burn_in on is in the file, in order.
        np.testing.assert_array_equal(rows, history[burn_in:])
        # And the rejection is among them: the chain stood still for exactly one row.
        stood_still = [i for i in range(1, len(rows)) if np.array_equal(rows[i], rows[i - 1])]
        assert len(stood_still) == 1

    def test_the_covariance_seed_reads_a_row_for_every_iteration(self, tmp_path):
        """The seed covariance divides by iteration - burn_in, taking it for granted that
        the history file holds that many rows when it is read. Rejections on the iterations
        just before the read are what could break that: a rejected iteration's row is
        written at the next barrier, so the barrier has to write before it proposes."""
        burn_in, adaptive = 2, 5
        # Chain 1 is the one rejected, from its fifth proposal on, because it is the last
        # to seed and so the one whose history is left on the sampler to look at.
        am, generation = _two_chains(tmp_path, [[IN], [IN, IN, IN, IN, OUT]],
                                     burn_in=burn_in, adaptive=adaptive)
        while min(am.iteration) <= burn_in + adaptive:
            generation = [ps for done in generation
                          for ps in am.got_result(_fake_result(done, 5.0))]

        assert am.boundary_rejections >= 2
        assert am.parameter_index_file.shape == (adaptive, len(am.variables))

    def test_a_run_resumed_from_an_older_backup_counts_from_zero(self, tmp_path):
        """A backup pickled before the counter existed restores a sampler without it. The
        counter is a class attribute as well, so that run meets its first wall and carries
        on rather than ending in AttributeError."""
        am, start = _two_chains(tmp_path, [[OUT], [IN]])
        del am.__dict__['boundary_rejections']
        am.got_result(_fake_result(start[0], 5.0))

        am.got_result(_fake_result(start[1], 5.0))

        assert am.boundary_rejections == 1


class TestAGenerationInWhichEveryProposalLeavesTheBox:
    """With nothing to submit, no result would arrive to reach the barrier again: the
    scheduler finds its job pool empty and ends the run, silently short. One chain
    makes this every boundary rejection, not a coincidence of several."""

    def test_the_barrier_goes_around_until_there_is_something_to_run(self, tmp_path):
        am, start = _two_chains(tmp_path, [[OUT, OUT, IN], [OUT, OUT, IN]])
        am.got_result(_fake_result(start[0], 5.0))

        generation = am.got_result(_fake_result(start[1], 5.0))

        # Two generations went by without a simulation, and the third is returned.
        assert sorted(ps.name for ps in generation) == ['iter3run0', 'iter3run1']
        assert am.boundary_rejections == 4
        assert [len(h) for h in am.chain_history] == [3, 3]

    def test_a_chain_that_never_gets_back_inside_ends_the_run_properly(self, tmp_path):
        """Each pass advances every chain by one iteration, so max_iterations bounds the
        loop, and the run ends through the same door as any other: 'STOP', with the final
        files written -- not an empty list, which the scheduler reads as a job pool that
        ran dry."""
        am, start = _two_chains(tmp_path, [[OUT], [OUT]],
                                burn_in=1, adaptive=4, max_iterations=12)
        am.got_result(_fake_result(start[0], 5.0))

        assert am.got_result(_fake_result(start[1], 5.0)) == 'STOP'

        assert am.iteration == [12, 12]
        assert am.total_evaluations == 2
        assert (tmp_path / 'Results' / 'A_MCMC' / 'Runs' / 'combined_params.txt').is_file()


# --------------------------------------------------------------------------- #
# Constraint satisfaction on the default worker-scoring path (lanl/PyBNF#480)
# --------------------------------------------------------------------------- #
def _data(col, rows):
    """A one-column Data object: header '# time <col>' plus (time, value) rows."""
    d = data.Data()
    lines = [f'# time {col}\n'] + [f'{t} {v}\n' for t, v in rows]
    d.data = d._read_file_lines(lines, r'\s+')
    return d


def _prime(am, suffixes, columns, noise=(), model='m'):
    """Point ``am``'s trajectory recording at ``columns`` (and ``noise`` columns) of the
    experiments ``suffixes`` of one model named ``model``, as ``__init__`` sets it up from a
    configuration (#990). The names are set after construction because they are not the
    parabola model's, which the configuration would refuse; the buffers are allocated by the
    first result that supplies them."""
    am.output_columns = list(columns)
    am.output_noise_columns = list(noise)
    am.output_noise_specs = None           # resolved for these columns on first use
    am.trajectory_suffixes = {model: list(suffixes)}
    for name in ('output_run_current', 'output_run_all', 'output_run_noise_current',
                 'output_run_noise_all'):
        setattr(am, name, {})


class TestConstraintSatisfactionWorkerScoring:
    """The default objective-scoring path (``local_objective_eval=0`` with
    ``parallelize_models=1``) scores on the worker and then nulls ``res.simdata``
    to save memory, leaving the full multi-model simulation dict on ``res.out``.
    The per-sample constraint-satisfaction bookkeeping must read that dict, not the
    ``None`` on ``res.simdata`` -- otherwise a job with cross-model ``.prop``
    constraints crashes on the first accepted move (lanl/PyBNF#480)."""

    def test_result_simdata_prefers_out_when_simdata_none(self, tmp_path):
        """_result_simdata returns res.out when the worker has nulled res.simdata."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        res = algorithms.Result(_uniform_pset('iter0run0'), None, 'iter0run0')
        payload = {'modelA': {'suf1': _data('A', [(1, 2), (2, 3)])}}
        res.out = payload
        assert am._result_simdata(res) is payload

    def test_result_simdata_uses_simdata_when_present(self, tmp_path):
        """When scoring stayed on the master, the data is on res.simdata and no
        res.out exists; _result_simdata returns simdata untouched."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        payload = {'modelA': {'suf1': _data('A', [(1, 2), (2, 3)])}}
        res = algorithms.Result(_uniform_pset('iter0run0'), payload, 'iter0run0')
        assert not hasattr(res, 'out')
        assert am._result_simdata(res) is payload

    def _run_worker_path(self, tmp_path, b_values):
        """Drive one accepted got_result on the worker-scoring path with a
        cross-model constraint ``A < suf2.B always`` (A from modelA/suf1, B from a
        *different* model modelB/suf2 -- the dotted reference whose resolution
        iterates the whole sim_data_dict). Returns the am after the move."""
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS)
        am = algorithms.Adaptive_MCMC(cfg)
        c = constraint.AlwaysConstraint('A', '<', 'suf2.B', 'modelA', 'suf1', 1.0)
        c.source_line = 'A < suf2.B always'
        am.all_constraints = [c]  # attach directly, bypassing .prop file wiring
        start = am.start_run()

        # Worker-scoring Result: simdata nulled, full multi-model dict on res.out.
        res = algorithms.Result(start[0], None, start[0].name)
        res.out = {'modelA': {'suf1': _data('A', [(1, 1), (2, 2)])},
                   'modelB': {'suf2': _data('B', list(enumerate(b_values, 1)))}}
        res.score = 5.0
        am.got_result(res)  # first move: ln_current_P is NaN, so always accepted
        assert am.accept is True
        return am

    def test_got_result_records_satisfied_cross_model_constraint(self, tmp_path):
        """A(max 2) < B(9) holds everywhere: the move is accepted and the
        constraint is recorded satisfied -- proving the data flowed from res.out
        into the (dotted, cross-model) constraint evaluation, not None."""
        am = self._run_worker_path(tmp_path, b_values=[9, 9])
        assert am.current_constraint_satisfied[0] == [1]

    def test_got_result_records_violated_cross_model_constraint(self, tmp_path):
        """A reaches 2 while B dips to 0: the constraint is violated and recorded
        as 0 (the evaluation ran against real data rather than crashing)."""
        am = self._run_worker_path(tmp_path, b_values=[0, 0])
        assert am.current_constraint_satisfied[0] == [0]


# --------------------------------------------------------------------------- #
# output_trajectory over edition-2 one-model + condition: cross-product suffixes
# (lanl/PyBNF#483)
# --------------------------------------------------------------------------- #
class TestOutputTrajectoryCrossProductSuffixes:
    """``output_run_current`` is allocated only for the *scored* diagonal keys
    (``time_length``: WT, KOko, ...), but under edition-2 Mechanism A (one model +
    ``condition:`` perturbations) the single model runs every action suffix under
    every mutant, so ``res.out[model]`` carries the full ``{action} x {mutant}``
    cross-product. ``got_result`` used to write every suffix it saw, hitting an
    uninitialized key on the first off-diagonal suffix (``KeyError: 'WTn78gMEK_pRDS'``);
    it must instead write only the scored diagonal it allocated (lanl/PyBNF#483)."""

    OBS = 'MEK_pRDS'

    @staticmethod
    def _diagonal_am(tmp_path, *, noise=False):
        """An ``am`` primed into the edition-2 diagonal-only trajectory state: two experiments
        (WT, KOko) of model ``m``, which is what ``__init__`` derives from a real edition-2
        one-model + condition configuration (#990), without needing that wiring."""
        overrides = {'output_trajectory': ['y']}
        if noise:
            overrides['output_noise_trajectory'] = ['y']
            overrides.update({'objfunc': 'neg_bin', 'neg_bin_r': 20.0})   # a noise trajectory needs a neg_bin noise model (#986)
        cfg = _make_config(tmp_path, 1, UNIFORM_VARS, **overrides)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        _prime(am, ['WT', 'KOko'], [TestOutputTrajectoryCrossProductSuffixes.OBS],
               noise=[TestOutputTrajectoryCrossProductSuffixes.OBS] if noise else ())
        return am

    def _cross_product_result(self, am):
        """One accepted worker-path Result whose single model emitted both scored
        diagonal suffixes (WT, KOko) and off-diagonal cross-product ones (WTn78g,
        KOn78g) -- the shape PyBNF produces for one model + condition: mutants."""
        start_name = am.current_pset[0].name if am.current_pset[0] else 'iter0run0'
        ps = _uniform_pset(start_name)
        res = algorithms.Result(ps, None, ps.name)
        res.out = {'m': {
            'WT':     _data(self.OBS, [(0, 1), (1, 2), (2, 3)]),   # scored diagonal
            'WTn78g': _data(self.OBS, [(0, 9), (1, 9), (2, 9)]),   # off-diagonal
            'KOko':   _data(self.OBS, [(0, 4), (1, 5), (2, 6)]),   # scored diagonal
            'KOn78g': _data(self.OBS, [(0, 7), (1, 8), (2, 9)]),   # off-diagonal
        }}
        res.score = 5.0
        return res

    def test_off_diagonal_suffix_does_not_crash(self, tmp_path):
        """The first off-diagonal suffix (``WTn78g``) used to raise
        ``KeyError: 'WTn78gMEK_pRDS'`` on the first accepted sample. It must now be
        skipped silently."""
        am = self._diagonal_am(tmp_path)
        am.got_result(self._cross_product_result(am))  # first move -> always accepted
        assert am.accept is True

    def test_scored_diagonal_trajectories_are_recorded(self, tmp_path):
        """The scored diagonal keys still receive their observable trajectory."""
        am = self._diagonal_am(tmp_path)
        am.got_result(self._cross_product_result(am))
        np.testing.assert_array_equal(am.output_run_current['WT' + self.OBS][0], [[1, 2, 3]])
        np.testing.assert_array_equal(am.output_run_current['KOko' + self.OBS][0], [[4, 5, 6]])

    def test_off_diagonal_keys_are_not_created(self, tmp_path):
        """The write must not fabricate buffers for unscored cross-product suffixes."""
        am = self._diagonal_am(tmp_path)
        am.got_result(self._cross_product_result(am))
        assert 'WTn78g' + self.OBS not in am.output_run_current
        assert 'KOn78g' + self.OBS not in am.output_run_current

    def test_noise_trajectory_block_has_the_same_guard(self, tmp_path):
        """The ``output_noise_trajectory`` write block has the identical structure and
        must also skip off-diagonal suffixes rather than KeyError."""
        am = self._diagonal_am(tmp_path, noise=True)
        am.got_result(self._cross_product_result(am))
        assert am.accept is True
        np.testing.assert_array_equal(am.output_run_noise_current['WT' + self.OBS][0], [[1, 2, 3]])
        np.testing.assert_array_equal(am.output_run_noise_current['KOko' + self.OBS][0], [[4, 5, 6]])
        assert 'WTn78g' + self.OBS not in am.output_run_noise_current


# --------------------------------------------------------------------------- #
# Adaptive-covariance seed file always carries a column-name header
# --------------------------------------------------------------------------- #
class TestAdaptiveSeedFileHeader:
    """``write_out_params`` must give ``params_<idx>.txt`` a parameter-name header
    row so the adaptive-covariance seed (``np.genfromtxt(..., names=True)`` at
    ``iteration == burn_in + adaptive``) can index it by name. The header used to be
    keyed to ``iteration == burn_in - 1``, which is unreachable when ``burn_in == 1``
    (iteration is already >= 1 by the time the write block runs in ``got_result``),
    leaving the file headerless and crashing ``pick_new_pset`` with
    ``ValueError: no field of name <first parameter>``. Surfaced while verifying the
    #480 fix on the MEK aMCMC example."""

    @staticmethod
    def _seed_file(am, idx=0):
        return (Path(am.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
                / f'params_{idx}.txt')

    def test_burn_in_1_writes_named_header(self, tmp_path):
        """burn_in == 1: the first write lands at iteration 1 (== burn_in), the branch
        that previously skipped the header. The file must still start with the names,
        and the exact field access pick_new_pset performs must not raise."""
        am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS, burn_in=1))
        am.start_run()
        names = [v.name for v in am.variables]
        am.parameter_index[0] = np.arange(1.0, len(names) + 1.0).reshape(1, len(names))
        # Populating the buffer by hand stands in for got_result's record step, which also
        # marks the slot as recorded -- what the write reads instead of the values (#758).
        am.parameter_index_recorded[0] = True

        am.iteration[0] = 1
        am.write_out_params(0)
        am.iteration[0] = 2
        am.write_out_params(0)

        seed = self._seed_file(am)
        lines = seed.read_text().splitlines()
        assert lines[0].split('\t') == names
        # The exact operation pick_new_pset performs on the seed file (#480 follow-up):
        arr = np.genfromtxt(seed, names=True)
        assert arr.dtype.names == tuple(names)
        for n in names:
            _ = arr[n]  # must not raise "no field of name ..."

    def test_normal_burn_in_header_then_data_preserved(self, tmp_path):
        """burn_in >= 2 behavior is unchanged: the burn_in-1 call writes the header
        only (no data row), and accepted-sample rows append under it afterward."""
        am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS, burn_in=3))
        am.start_run()
        names = [v.name for v in am.variables]
        am.parameter_index[0] = np.arange(1.0, len(names) + 1.0).reshape(1, len(names))
        # Populating the buffer by hand stands in for got_result's record step, which also
        # marks the slot as recorded -- what the write reads instead of the values (#758).
        am.parameter_index_recorded[0] = True
        seed = self._seed_file(am)

        am.iteration[0] = 2  # burn_in - 1: header initialized, no data row
        am.write_out_params(0)
        assert seed.read_text().splitlines() == ['\t'.join(names)]

        am.iteration[0] = 3  # burn_in onward: one accepted-sample row appended
        am.write_out_params(0)
        lines = seed.read_text().splitlines()
        assert lines[0].split('\t') == names
        assert len(lines) == 2
        arr = np.genfromtxt(seed, names=True)
        assert arr.dtype.names == tuple(names)


# --------------------------------------------------------------------------- #
# A configured trajectory name that no simulation produces (lanl/PyBNF#755, #990)
# --------------------------------------------------------------------------- #
class TestAnUnmatchedTrajectoryNameIsRefused:
    """``output_trajectory`` / ``output_noise_trajectory`` name columns of the simulation
    result. A name that matches no column is filled by nothing, so the user got a ``Runs/``
    directory with fewer ``traj_*.txt`` files than the conf asked for (#755). That was then
    reported as a warning at the first simulation, printed only at ``verbosity >= 1``, and
    the run still exited 0 without the file (#990). It is now refused: when the configuration
    loads, against the names the model files define, and at the first simulation for a name
    that slips past that, against the columns the experiments actually produced."""

    OBS = 'MEK_pRDS'
    TYPO = 'MEK_pRDZ'          # deliberately not a prefix of OBS, so assertions can tell

    @staticmethod
    def _am_wanting(tmp_path, wanted, noise_wanted=()):
        """An ``am`` asking for ``wanted`` trajectories of one experiment, ``WT``."""
        overrides = {'output_trajectory': ['y']}
        if noise_wanted:
            overrides['output_noise_trajectory'] = ['y']
            overrides.update({'objfunc': 'neg_bin', 'neg_bin_r': 20.0})   # a noise trajectory needs a neg_bin noise model (#986)
        cfg = _make_config(tmp_path, 1, UNIFORM_VARS, **overrides)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        _prime(am, ['WT'], wanted, noise=noise_wanted)
        return am

    def _result(self, am):
        """One accepted worker-path Result whose model emitted only ``OBS``."""
        start_name = am.current_pset[0].name if am.current_pset[0] else 'iter0run0'
        ps = _uniform_pset(start_name)
        res = algorithms.Result(ps, None, ps.name)
        res.out = {'m': {'WT': _data(self.OBS, [(0, 1), (1, 2), (2, 3)])}}
        res.score = 5.0
        return res

    @pytest.mark.parametrize('key', ['output_trajectory', 'output_noise_trajectory'])
    def test_a_name_no_model_defines_is_refused_when_the_configuration_loads(self, tmp_path,
                                                                             key):
        """parabola.bngl defines the observable x and the function y; Y is neither."""
        overrides = {key: ['y', 'Y']}
        if key == 'output_noise_trajectory':
            overrides.update({'objfunc': 'neg_bin', 'neg_bin_r': 20.0})
        with pytest.raises(PybnfError) as info:
            algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS, **overrides))
        assert info.value.log_message == (
            f'{key} names Y, which is not a column any simulation in this fit produces')
        assert info.value.message.endswith('The names it could match are: x, y.')

    def test_several_names_are_refused_together(self, tmp_path):
        with pytest.raises(PybnfError) as info:
            algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS,
                                                  output_trajectory=['X', 'y', 'Y']))
        assert info.value.log_message == ('output_trajectory names X, Y, which are not a column '
                                          'any simulation in this fit produces')

    def test_a_name_no_experiment_produces_is_refused_at_the_first_simulation(self, tmp_path):
        """A name can pass the load check without being an output column (the check accepts
        every identifier of the model files, so that it never refuses a real column)."""
        am = self._am_wanting(tmp_path, [self.OBS, self.TYPO])
        with pytest.raises(PybnfError) as info:
            am.got_result(self._result(am))
        assert info.value.log_message == (
            f"output_trajectory names {self.TYPO}, which is not a column of any experiment's "
            f"simulation")
        assert info.value.message.endswith(f'The simulations produce: {self.OBS}, time.')
        assert am.output_run_recorded == set()

    def test_a_model_identifier_that_is_not_a_column_passes_the_load_and_stops_the_run(
            self, tmp_path):
        """counter is the parabola model's molecule type: in its file, never a column."""
        am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS,
                                                   output_trajectory=['counter']))
        am.start_run()
        ps = _uniform_pset('iter0run0')
        res = algorithms.Result(ps, None, ps.name)
        res.out = {'parabola': {'par1': _data('y', [(0, 1), (1, 2)])}}
        res.score = 5.0
        with pytest.raises(PybnfError, match='output_trajectory names counter, which is not'):
            am.got_result(res)

    def test_a_fit_whose_names_all_match_is_recorded(self, tmp_path):
        am = self._am_wanting(tmp_path, [self.OBS])
        am.got_result(self._result(am))
        assert am.output_run_recorded == {('WT' + self.OBS, 0)}

    def test_the_noise_key_is_checked_under_its_own_name(self, tmp_path):
        am = self._am_wanting(tmp_path, [self.OBS], noise_wanted=[self.TYPO])
        with pytest.raises(PybnfError) as info:
            am.got_result(self._result(am))
        assert info.value.log_message.startswith(f'output_noise_trajectory names {self.TYPO},')

    def test_two_experiments_that_would_share_a_file_are_refused(self, tmp_path):
        """A file is named by experiment suffix and column with no separator, so ``epi`` +
        ``cases2`` and ``epic`` + ``ases2`` are one file, which both would append to."""
        am = self._am_wanting(tmp_path, ['cases2', 'ases2'])
        _prime(am, ['epi', 'epic'], ['cases2', 'ases2'])
        res = algorithms.Result(_uniform_pset('iter0run0'), None, 'iter0run0')
        res.out = {'m': {'epi': _data('cases2', [(0, 1), (1, 2)]),
                         'epic': _data('ases2', [(0, 3), (1, 4)])}}
        res.score = 5.0
        with pytest.raises(PybnfError) as info:
            am.got_result(res)
        assert info.value.log_message == ('output_trajectory would write two trajectories to '
                                          'one file, epicases2')
        assert ('cases2 of model m in experiment epi and ases2 of model m in experiment epic'
                in info.value.message)


# --------------------------------------------------------------------------- #
# A recorded sample is written whatever its values (lanl/PyBNF#758)
# --------------------------------------------------------------------------- #
class TestARecordedSampleIsWrittenWhateverItsValues:
    """The writers used to decide what to write by dropping every all-zero row.

    Each buffer holds one slot per chain (``arr_length`` is the constant 1 and ``factor``
    is never incremented), rewritten with the chain's current state each sampling
    iteration and appended to the file, so the filter ran per sample. It was standing in
    for "was this slot recorded", which it cannot distinguish from a recorded trajectory
    whose observable sits at zero across the window -- an absent species under a knockout,
    a ``_Cum`` counter over a quiet window. Those samples were dropped, leaving a file
    with fewer rows than the run sampled and nothing marking the gap, so a band or a mean
    taken from it covered only the samples where the observable was switched on (#758).

    The slot now records whether a result column supplied it, which also keeps a name that
    matched no column (#755) writing no file at all rather than a file of zeros."""

    OBS = 'OBS'

    @staticmethod
    def _am(tmp_path, wanted=(OBS,)):
        """An ``am`` past burn-in that samples on every iteration, with one experiment."""
        cfg = _make_config(tmp_path, 1, UNIFORM_VARS, output_trajectory=['y'],
                           burn_in=1, adaptive=0, sample_every=1, output_hist_every=10**9)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        _prime(am, ['WT'], wanted)
        am.iteration[0] = 1
        return am

    def _feed(self, am, values):
        """One result whose model emitted ``OBS`` with these values, driven through the
        real accept -> record -> write path."""
        ps = _uniform_pset(f'iter{am.iteration[0]}run0')
        res = algorithms.Result(ps, None, ps.name)
        res.out = {'m': {'WT': _data(self.OBS, list(enumerate(values)))}}
        res.score = 5.0
        am.got_result(res)

    @staticmethod
    def _rows(am, name='traj_WTOBS_chain_0.txt'):
        f = Path(am.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs' / name
        return f.read_text().splitlines() if f.exists() else []

    def test_a_trajectory_of_zeros_is_recorded_like_any_other(self, tmp_path):
        am = self._am(tmp_path)
        self._feed(am, [0.0, 0.0, 0.0])
        assert self._rows(am) == ['0.000000000000000000e+00 0.000000000000000000e+00 '
                                  '0.000000000000000000e+00']

    def test_the_row_count_matches_the_sampling_iterations(self, tmp_path):
        """The part that biases an ensemble: the zero samples were the ones dropped, so
        the file covered only the samples where the observable was switched on."""
        am = self._am(tmp_path)
        samples = [[1., 2., 3.], [0., 0., 0.], [4., 5., 6.], [0., 0., 0.]]
        for values in samples:
            self._feed(am, values)
        rows = self._rows(am)
        assert len(rows) == len(samples)
        assert [float(r.split()[0]) for r in rows] == [1.0, 0.0, 4.0, 0.0]

    def test_a_name_no_column_supplied_still_writes_no_file(self, tmp_path):
        """#755's case must not turn into a file of zeros now that zeros are written: the
        name is refused at the first simulation (#990), before anything is recorded."""
        am = self._am(tmp_path, wanted=('TYPO',))
        with pytest.raises(PybnfError):
            self._feed(am, [1.0, 2.0, 3.0])
        assert am.output_run_recorded == set()
        assert self._rows(am, 'traj_WTTYPO_chain_0.txt') == []

    def test_the_noise_writer_follows_the_same_rule(self, tmp_path):
        """``output_run_noise_all`` is only copied under a neg_bin objective, so the rule
        is exercised on the writer directly."""
        am = self._am(tmp_path)
        am.output_noise_columns = [self.OBS]
        am.output_run_noise_current = {'WTOBS': np.zeros((am.num_parallel, 1, 3))}
        am.output_run_noise_all = {'WTOBS': np.zeros((am.num_parallel, 1, 3))}
        am.write_out_trajactorys_noise(0)
        assert self._rows(am, 'traj_noise_WTOBS_chain_0.txt') == []   # never recorded
        am.output_run_noise_recorded.add(('WTOBS', 0))
        am.write_out_trajactorys_noise(0)
        assert len(self._rows(am, 'traj_noise_WTOBS_chain_0.txt')) == 1

    def test_an_all_zero_parameter_vector_is_written_once_recorded(self, tmp_path):
        """The same sentinel in ``write_out_params``: a vector of all zeros is a legal
        point of a box whose lower bound is zero, which the reflecting fold can return."""
        am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS, burn_in=1))
        am.start_run()
        am.iteration[0] = 1
        am.parameter_index[0] = np.zeros((1, len(am.variables)))
        am.write_out_params(0)
        seed = Path(am.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs' / 'params_0.txt'
        assert len(seed.read_text().splitlines()) == 1        # header only, not recorded
        am.parameter_index_recorded[0] = True
        am.write_out_params(0)
        assert len(seed.read_text().splitlines()) == 2        # the all-zero row is written


# --------------------------------------------------------------------------- #
# The combine step tolerates a per-chain file the run never wrote (lanl/PyBNF#760)
# --------------------------------------------------------------------------- #
class TestCombiningChainsWithAFileTheRunNeverWrote:
    """``combine_chains_traj`` loaded every per-chain file by name with no check that it
    exists, so the absence of an output the run itself declined to write ended the run in
    ``FileNotFoundError`` -- at one of the two run terminations, after all the sampling
    work was done and on the last step before ``'STOP'`` (#760).

    Two ways to get there, both needing ``population_size > 1`` and ``output_trajectory``:
    a name no simulation produced a column for is never written for any chain (#755), and
    a run that converges inside the adaptive window stops before ``valid_range``, which is
    the first iteration any trajectory is written at -- ``check_convergence`` needs only
    ``iteration > burn_in``. The second can leave chains in different states, since it
    fires on one chain's iteration count while the others are still behind it."""

    @staticmethod
    def _am(tmp_path, keys=('WTOBS',), wanted=('OBS',), noise=False):
        overrides = {'output_trajectory': ['y']}
        if noise:
            overrides['output_noise_trajectory'] = ['y']
            overrides.update({'objfunc': 'neg_bin', 'neg_bin_r': 20.0})   # a noise trajectory needs a neg_bin noise model (#986)
        cfg = _make_config(tmp_path, 2, UNIFORM_VARS, burn_in=2, adaptive=2,
                           max_iterations=10, **overrides)
        am = algorithms.Adaptive_MCMC(cfg)
        am.start_run()
        # A buffer key is an experiment suffix followed by its column, matched exactly (#986).
        _prime(am, [k[:-len(w)] for k in keys for w in wanted if k.endswith(w)], wanted,
               noise=wanted if noise else ())
        am.output_run_current = {k: np.zeros((am.num_parallel, 1, 3)) for k in keys}
        if noise:
            am.output_run_noise_current = dict(am.output_run_current)
        runs = Path(cfg.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        return am, runs

    def test_no_per_chain_file_at_all_does_not_end_the_run(self, tmp_path, capsys):
        am, runs = self._am(tmp_path, keys=('WTTYPO',), wanted=('TYPO',))
        am.combine_chains_traj()                      # must not raise
        out = capsys.readouterr().out
        assert 'WTTYPO chain 0, WTTYPO chain 1' in out
        assert 'missing 2 per-chain files' in out
        # and no empty combined file is left behind for a key with nothing in it
        assert list(runs.glob('combined_*')) == []

    def test_a_chain_that_wrote_is_still_combined(self, tmp_path, capsys):
        """A convergence stop can leave one chain with a file and another without."""
        am, runs = self._am(tmp_path)
        np.savetxt(runs / 'traj_WTOBS_chain_0.txt', np.array([[1., 2., 3.], [4., 5., 6.]]))
        am.combine_chains_traj()
        out = capsys.readouterr().out
        assert 'missing 1 per-chain file ' in out          # singular
        assert 'WTOBS chain 1' in out and 'WTOBS chain 0' not in out
        rows = [line.split() for line in (runs / 'combined_traj_WTOBS.txt').read_text().splitlines()]
        assert [[float(x) for x in r] for r in rows] == [[1., 2., 3.], [4., 5., 6.]]

    def test_both_chains_present_is_unchanged_and_silent(self, tmp_path, capsys):
        am, runs = self._am(tmp_path)
        np.savetxt(runs / 'traj_WTOBS_chain_0.txt', np.array([[1., 2., 3.], [4., 5., 6.]]))
        np.savetxt(runs / 'traj_WTOBS_chain_1.txt', np.array([[7., 8., 9.]]))
        am.combine_chains_traj()
        assert 'missing' not in capsys.readouterr().out
        rows = [line.split() for line in (runs / 'combined_traj_WTOBS.txt').read_text().splitlines()]
        assert [[float(x) for x in r] for r in rows] == [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]

    def test_a_single_sample_file_is_not_written_down_the_page(self, tmp_path):
        """``loadtxt`` drops a one-row file to 1-D and ``savetxt`` then writes that sample
        as one value per line. A run with exactly one sampling iteration past
        ``valid_range`` had its combined trajectory transposed."""
        am, runs = self._am(tmp_path)
        np.savetxt(runs / 'traj_WTOBS_chain_0.txt', np.array([[1., 2., 3.]]))
        np.savetxt(runs / 'traj_WTOBS_chain_1.txt', np.array([[4., 5., 6.]]))
        am.combine_chains_traj()
        rows = [line.split() for line in (runs / 'combined_traj_WTOBS.txt').read_text().splitlines()]
        assert [[float(x) for x in r] for r in rows] == [[1., 2., 3.], [4., 5., 6.]]

    def test_the_noise_trajectories_take_the_same_path(self, tmp_path, capsys):
        am, runs = self._am(tmp_path, noise=True)
        np.savetxt(runs / 'traj_WTOBS_chain_0.txt', np.array([[1., 2., 3.]]))
        np.savetxt(runs / 'traj_WTOBS_chain_1.txt', np.array([[4., 5., 6.]]))
        am.combine_chains_traj()                      # noise files absent -> must not raise
        out = capsys.readouterr().out
        assert 'missing 2 per-chain files' in out
        assert 'combined_traj_WTOBS.txt' in ' '.join(p.name for p in runs.glob('combined_*'))
        assert not (runs / 'combined_traj_noise_WTOBS.txt').exists()



def test_the_output_noise_draws_a_negative_prediction_as_the_mean_zero():
    """The sibling of the scoring fix: prob = r/(r + prediction) left (0, 1] for a prediction
    below -r, and nbinom then drew an enormous count into the noise trajectory."""
    from types import SimpleNamespace
    import numpy as np
    from pybnf.algorithms.samplers.adaptive_mcmc import Adaptive_MCMC
    from pybnf.noise import MEAN, ConstantSigma, NegBinomial
    stub = SimpleNamespace(output_noise_specs={
        'c': (NegBinomial(location=MEAN), ConstantSigma(2.0), False)})
    series = np.array([-1e6, -2.0, -1.0, 0.0, 4.0])
    draws = Adaptive_MCMC.generateBinomialNoise(stub, series, None, np.random.default_rng(1), 'c')
    clamped = Adaptive_MCMC.generateBinomialNoise(stub, np.maximum(series, 0.0), None,
                                                  np.random.default_rng(1), 'c')
    assert list(draws) == list(clamped)
    assert list(draws[:4]) == [0, 0, 0, 0]


# --------------------------------------------------------------------------------------------
# output_noise_trajectory draws from the column's own noise model (lanl/PyBNF#986)
# --------------------------------------------------------------------------------------------

def test_a_noise_trajectory_under_a_noise_model_that_is_not_neg_bin_is_refused(tmp_path):
    """Under the default chi_sq the draw never ran, and the noise files held zeros."""
    cfg = _make_config(tmp_path, 1, UNIFORM_VARS, output_noise_trajectory=['y'])
    with pytest.raises(PybnfError) as info:
        algorithms.Adaptive_MCMC(cfg)
    # main() prints the message and logs the summary: both name the column and its family.
    assert info.value.log_message.startswith(
        'output_noise_trajectory names y, whose noise model is Gaussian, not neg_bin')
    assert "y's is Gaussian. Declare it with a neg_bin noise model" in info.value.message


def test_a_median_located_noise_draw_is_around_the_mean_that_places_the_median():
    """An edition-2 neg_bin line defaults to the median location, so the draw's mean is the one
    whose median is the prediction. Oracle: scipy.stats.nbinom on the same seed at
    r / (r + _mean_for_median(prediction, r))."""
    from types import SimpleNamespace
    from scipy import stats
    from pybnf.algorithms.samplers.adaptive_mcmc import Adaptive_MCMC
    from pybnf.noise import MEDIAN, ConstantSigma, NegBinomial
    from pybnf.noise.negative_binomial import _mean_for_median
    r, series = 3.0, np.array([0.0, 2.0, 7.5, 40.0])
    stub = SimpleNamespace(output_noise_specs={
        'c': (NegBinomial(location=MEDIAN), ConstantSigma(r), False)})
    draws = Adaptive_MCMC.generateBinomialNoise(stub, series, None, np.random.default_rng(5), 'c')
    rng = np.random.default_rng(5)
    expected = [stats.nbinom.rvs(n=r, p=np.clip(r / (r + _mean_for_median(x, r)), 1e-10,
                                                1 - 1e-10), size=1, random_state=rng)[0]
                for x in series]
    np.testing.assert_array_equal(draws, expected)
    assert _mean_for_median(7.5, r) != 7.5


@pytest.mark.bngsim
def test_an_edition_2_noise_trajectory_holds_counts_around_the_predicted_increments(
        tmp_path, monkeypatch):
    """lanl/PyBNF#986: under edition 2 the draw was gated on the legacy objfunc, so every
    traj_noise_* file was all zeros. The counts are now negative binomial draws (fix_at 20)
    around the increments of `cases`, a column declared cumulative without the _Cum naming
    convention. Oracle: the plain trajectory written beside it, differenced the same way;
    over every recorded row the drawn total is within 10% of the predicted total."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    R.require_bng2pl()
    (tmp_path / 'sir.bngl').write_text(L.MODEL)
    L.write_data(tmp_path / 'cases.exp')
    (tmp_path / 'am.conf').write_text('\n'.join([
        'edition = 2', 'model: sir.bngl', 'bngl_backend = bngsim', 'job_type = am',
        'output_dir = out', 'experiment: epi, data: cases.exp',
        'noise_model = neg_bin, dispersion = fix_at 20, location = mean',
        'noise_model cases = neg_bin, dispersion = fix_at 20, location = mean, cumulative',
        'uniform_var = beta 0.3 1.2', 'uniform_var = gamma 0.1 0.5',
        'output_trajectory = cases', 'output_noise_trajectory = cases',
        'population_size = 2', 'max_iterations = 40', 'burn_in = 10', 'adaptive = 10',
        'sample_every = 1', 'step_size = 0.05', 'random_seed = 1', 'parallel_count = 2']) + '\n')
    assert L.run_main(monkeypatch, tmp_path, tmp_path / 'am.conf') == 0
    runs = tmp_path / 'out' / 'Results' / 'A_MCMC' / 'Runs'
    drawn, predicted = 0.0, 0.0
    for chain in (0, 1):
        noise = np.loadtxt(runs / ('traj_noise_epicases_chain_%d.txt' % chain), ndmin=2)
        plain = np.loadtxt(runs / ('traj_epicases_chain_%d.txt' % chain), ndmin=2)
        assert noise.shape == plain.shape and noise.shape[0] > 10
        increments = np.column_stack([plain[:, :1], np.diff(plain, axis=1)])
        recorded = ~(plain == 0).all(axis=1)          # #859's never-simulated first row
        drawn += noise[recorded].sum()
        predicted += increments[recorded].sum()
    assert predicted > 1000
    assert abs(drawn / predicted - 1.0) < 0.1, (drawn, predicted)


# --------------------------------------------------------------------------------------------
# output_noise_trajectory: the conf's noise model reaches the draw, column by column (#986 review)
# --------------------------------------------------------------------------------------------

def _edition_2_noise_am(tmp_path, monkeypatch, noise_lines, noise_columns, columns=('cases',)):
    """An edition-2 ``am`` over the Liu-West SIR model, built from its conf text by
    ``load_config`` (no simulation). ``columns`` are model functions, all ``rho*CumInf``, each
    a data column of ``cases.exp``."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    from pybnf.parse import load_config
    R.require_bng2pl()
    extra = ''.join('  %s() rho*CumInf\n' % c for c in columns if c != 'cases')
    (tmp_path / 'sir.bngl').write_text(
        L.MODEL.replace('  cases() rho*CumInf\n', '  cases() rho*CumInf\n' + extra))
    with open(tmp_path / 'cases.exp', 'w') as f:
        f.write('# time %s\n' % ' '.join(columns))
        for t, c in enumerate(L.COUNTS, 1):
            f.write('%d %s\n' % (t, ' '.join([str(c)] * len(columns))))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'edition = 2', 'model: sir.bngl', 'bngl_backend = bngsim', 'job_type = am',
        'output_dir = out', 'experiment: epi, data: cases.exp',
        'uniform_var = beta 0.3 1.2', 'uniform_var = gamma 0.1 0.5',
        'population_size = 1', 'max_iterations = 40', 'burn_in = 10', 'adaptive = 10',
        'sample_every = 1', 'output_noise_trajectory = %s' % ', '.join(noise_columns),
        *noise_lines]) + '\n')
    monkeypatch.chdir(tmp_path)
    return algorithms.Adaptive_MCMC(load_config(str(tmp_path / 'am.conf')))


def _median_mean(prediction, r):
    """The mean of NB(mean, r) whose continuous median I_p(r, x+1) = 1/2 sits at the prediction,
    solved here by bisection on scipy's regularized incomplete beta."""
    from scipy.special import betainc
    target = max(prediction, 0.0)
    lo, hi = 0.0, max(target, 1.0)
    while betainc(r, target + 1.0, r / (r + hi)) > 0.5:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if betainc(r, target + 1.0, r / (r + mid)) > 0.5 else (lo, mid)
    return 0.5 * (lo + hi)


_NOISE_LINES = [
    (line, dispersion, location, cumulative)
    for line in ('per_observable', 'whole_fit')
    for dispersion in ('fix_at 3', 'fit r')
    for location in ('mean', 'median', None)
    for cumulative in (False, True)
    if not (line == 'whole_fit' and cumulative)]    # a whole-fit line cannot be cumulative


@pytest.mark.bngsim
@pytest.mark.parametrize('line, dispersion, location, cumulative', _NOISE_LINES)
def test_an_edition_2_noise_draw_matches_scipy_at_the_declared_dispersion_and_location(
        tmp_path, monkeypatch, line, dispersion, location, cumulative):
    """From the conf text to the draw: the per-observable line wins over a whole-fit Gaussian,
    a whole-fit neg_bin line covers the column, ``fix_at`` is the constant and ``fit r`` the
    chain's current r, the location (edition 2 defaults to the median, ADR-0031) sets the
    mean, and ``cumulative`` decides the differencing. Oracle: scipy.stats.nbinom on a copy of
    the chain's Generator, at r / (r + mean), with the median's mean solved by bisection on
    scipy.special.betainc rather than by _mean_for_median."""
    from scipy import stats
    from pybnf.noise import MEAN, MEDIAN
    fields = 'neg_bin, dispersion = %s%s' % (
        dispersion, '' if location is None else ', location = %s' % location)
    lines = ['loguniform_var = r 1 100'] if dispersion == 'fit r' else []
    if line == 'per_observable':
        lines += ['noise_model = gaussian, sigma = fix_at 5, location = mean',
                  'noise_model cases = %s%s' % (fields, ', cumulative' if cumulative else '')]
    else:
        lines += ['noise_model = %s' % fields]
    am = _edition_2_noise_am(tmp_path, monkeypatch, lines, ['cases'])
    family, _source, differenced = am.output_noise_specs['cases']
    assert differenced is cumulative
    assert family.location is (MEAN if location == 'mean' else MEDIAN)

    r = 3.0 if dispersion == 'fix_at 3' else 7.5
    current = pset.PSet([v.set_value(7.5 if v.name == 'r' else v.value or v.lower_bound)
                         for v in am.variables])
    series = np.array([-2.0, 0.0, 1.0, 6.5, 40.0, 250.0])
    rng = np.random.default_rng(11)
    oracle_rng = np.random.default_rng(11)
    draws = am.generateBinomialNoise(series, current, rng, 'cases')
    means = [max(x, 0.0) if location == 'mean' else _median_mean(x, r) for x in series]
    expected = [stats.nbinom.rvs(n=r, p=np.clip(r / (r + m), 1e-10, 1 - 1e-10), size=1,
                                 random_state=oracle_rng)[0] for m in means]
    np.testing.assert_array_equal(draws, expected)


@pytest.mark.bngsim
def test_a_noise_column_whose_name_is_inside_another_key_is_drawn_with_its_own_model(
        tmp_path, monkeypatch):
    """`cases` is a substring of the buffer key `epicases2`, and the draw loop matches a column
    to a key with `if l in i`. Listing `cases2` before `cases` used to draw `epicases2` twice,
    the second time with `cases`'s noise model, which is the one that stuck. Oracle: each
    column's draw is tagged with the column it was drawn for."""
    am = _edition_2_noise_am(
        tmp_path, monkeypatch,
        ['noise_model = neg_bin, dispersion = fix_at 20, location = mean',
         'noise_model cases = neg_bin, dispersion = fix_at 0.5, location = mean',
         'noise_model cases2 = neg_bin, dispersion = fix_at 10000, location = mean'],
        ['cases2', 'cases'], columns=('cases', 'cases2'))
    assert am.output_noise_specs['cases2'][1].const == 10000.0
    am.start_run()
    tag = {'cases': 1.0, 'cases2': 2.0}
    monkeypatch.setattr(am, 'generateBinomialNoise',
                        lambda timeseries, pset, rng, column: np.full(len(timeseries), tag[column]))
    am.current_pset[0] = pset.PSet([v.set_value(0.5 * (v.lower_bound + v.upper_bound))
                                    for v in am.variables])
    am.current_pset[0].name = 'iter0run0'
    am.ln_current_P[0] = 0.0
    for key in ('epicases', 'epicases2'):      # as a recorded result allocates them (#990)
        am.output_run_noise_current[key] = np.zeros((1, 1, 13))
        am.output_run_noise_all[key] = np.zeros((1, 1, 13))
    am.iteration[0] = am.valid_range
    am._record_iteration(0)
    row = am.factor[0]
    assert set(am.output_run_noise_all['epicases'][0][row]) == {1.0}
    assert set(am.output_run_noise_all['epicases2'][0][row]) == {2.0}


@pytest.mark.bngsim
def test_a_neg_bin_noise_trajectory_whose_dispersion_follows_the_data_is_refused(
        tmp_path, monkeypatch):
    """A `relative` dispersion is a multiple of each observed count, so a simulated trajectory
    between and beyond the data rows has no dispersion to be drawn with. The refusal comes when
    the algorithm is built, and names the column and the source."""
    with pytest.raises(PybnfError) as info:
        _edition_2_noise_am(
            tmp_path, monkeypatch,
            ['noise_model = neg_bin, dispersion = fix_at 20, location = mean',
             'noise_model cases = neg_bin, dispersion = relative 0.5, location = mean, cumulative'],
            ['cases'])
    assert info.value.log_message == (
        'output_noise_trajectory names cases, whose neg_bin dispersion is RelativeSigma')
    assert ("draws cases's counts with a dispersion fixed (fix_at), fitted (fit <parameter>) "
            "or a formula of parameters" in info.value.message)
    assert "cases's is a RelativeSigma" in info.value.message


def test_a_run_resumed_from_a_backup_made_before_986_still_draws_its_noise(tmp_path):
    """`-r` unpickles the algorithm without running __init__, so a backup written before #986
    has no `output_noise_specs`; it must still record and draw, from the same legacy spec
    (objfunc = neg_bin, neg_bin_r = 20), not stop at its first result with an AttributeError.
    Oracle: scipy.stats.nbinom at r = 20, p = 20 / (20 + prediction), on a copy of the
    chain's Generator."""
    import copy
    import pickle
    from scipy import stats
    cfg = _make_config(tmp_path, 1, UNIFORM_VARS, output_noise_trajectory=['y'],
                       objfunc='neg_bin', neg_bin_r=20.0)
    am = algorithms.Adaptive_MCMC(cfg)
    am.start_run()
    key, n = 'par1y', 21          # parabola's par1 action: n_steps 20, so 21 output rows

    def result(name, prediction, score):
        res = algorithms.Result(_uniform_pset(name), None, name)
        res.out = {'parabola': {'par1': _data('y', list(enumerate(prediction)))}}
        res.score = score
        return res

    # One generation before the backup, as the run loop backs up once per generation.
    am.got_result(result('iter0run0', np.zeros(n), 5.0))
    del am.output_noise_specs                      # as a backup made before #986 holds it
    am = pickle.loads(pickle.dumps(am))
    assert 'output_noise_specs' not in vars(am)

    prediction = np.linspace(0.5, 40.0, n)
    am.got_result(result('iter1run0', prediction, 0.0))   # a better score: accepted, recorded
    assert am.accept is True
    np.testing.assert_array_equal(am.output_run_noise_current[key][0][0], prediction)

    am.iteration[0] = am.valid_range
    oracle_rng = copy.deepcopy(am.chain_rngs[0])
    am._record_iteration(0)
    expected = [stats.nbinom.rvs(n=20.0, p=20.0 / (20.0 + x), size=1, random_state=oracle_rng)[0]
                for x in prediction]
    np.testing.assert_array_equal(am.output_run_noise_all[key][0][am.factor[0]], expected)


_E1_SIR = """begin model
begin parameters
  beta__FREE   0.6
  gamma__FREE  0.25
  N      10000
  I0     10
  rho    0.5
  beta_N beta__FREE/N
end parameters
begin molecule types
  S()
  I()
  R()
  C()
end molecule types
begin seed species
  S() N-I0
  I() I0
  R() 0
  C() 0
end seed species
begin observables
  Molecules Inf I()
  Molecules CumInf C()
end observables
begin functions
  cases() rho*CumInf
  cases_Cum() rho*CumInf
end functions
begin reaction rules
  S() + I() -> I() + I() + C()  beta_N
  I() -> R()  gamma__FREE
end reaction rules
end model
generate_network({overwrite=>1})
simulate({method=>"ode",suffix=>"epi",t_end=>12,n_steps=>12,print_functions=>1})
"""


def _edition_1_noise_am(tmp_path, monkeypatch, lines, model_parameters=()):
    """An edition-1 ``am`` over an SIR model with `cases` and `cases_Cum` columns."""
    from . import lwf_harness as L
    from pybnf.parse import load_config
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'sir1.bngl').write_text(_E1_SIR.replace(
        '  N      10000\n', '  N      10000\n' + ''.join('  %s 1\n' % p for p in model_parameters)))
    (tmp_path / 'epi.exp').write_text('# time cases cases_Cum\n' + ''.join(
        '%d %d %d\n' % (t, c, c) for t, c in enumerate(L.COUNTS, 1)))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'model = sir1.bngl : epi.exp', 'fit_type = am', 'output_dir = out',
        'uniform_var = beta__FREE 0.3 1.2', 'uniform_var = gamma__FREE 0.1 0.5',
        'population_size = 1', 'max_iterations = 40', 'burn_in = 10', 'adaptive = 10',
        'output_noise_trajectory = cases_Cum', *lines]) + '\n')
    monkeypatch.chdir(tmp_path)
    return algorithms.Adaptive_MCMC(load_config(str(tmp_path / 'am.conf')))


def test_the_cum_name_differences_the_draw_only_under_the_legacy_objfuncs_own_spec(
        tmp_path, monkeypatch):
    """The legacy objfuncs have always drawn a `_Cum` column differenced, and still do, byte for
    byte; any other noise model is differenced as the scoring is, by its cumulative flag: a
    per-observable neg_bin on `cases_Cum` under chi_sq_dynamic is scored undifferenced, so it is
    drawn undifferenced."""
    am = _edition_1_noise_am(tmp_path / 'legacy', monkeypatch, ['objfunc = neg_bin', 'neg_bin_r = 20'])
    assert am.output_noise_specs['cases_Cum'][2] is True
    am = _edition_1_noise_am(tmp_path / 'override', monkeypatch, [
        'objfunc = chi_sq_dynamic', 'uniform_var = sigma__FREE 0.1 10',
        'noise_model cases_Cum = neg_bin, dispersion = fix_at 20, location = mean'],
        model_parameters=['sigma__FREE'])
    assert am.objective._is_cumulative('cases_Cum') is False
    assert am.output_noise_specs['cases_Cum'][2] is False


@pytest.mark.bngsim
def test_a_formula_dispersion_draws_at_its_value_at_the_chains_point(tmp_path, monkeypatch):
    """`dispersion = formula 2*r` has one value per parameter set. Oracle: scipy.stats.nbinom on
    the same seed at r = 2 * 7.5."""
    from scipy import stats
    am = _edition_2_noise_am(tmp_path, monkeypatch, [
        'loguniform_var = r 1 100',
        'noise_model = neg_bin, dispersion = formula 2*r, location = mean'], ['cases'])
    current = pset.PSet([v.set_value(7.5 if v.name == 'r' else v.value or v.lower_bound)
                         for v in am.variables])
    series = np.array([0.0, 3.0, 40.0])
    draws = am.generateBinomialNoise(series, current, np.random.default_rng(3), 'cases')
    rng = np.random.default_rng(3)
    expected = [stats.nbinom.rvs(n=15.0, p=np.clip(15.0 / (15.0 + x), 1e-10, 1 - 1e-10), size=1,
                                 random_state=rng)[0] for x in series]
    np.testing.assert_array_equal(draws, expected)


@pytest.mark.bngsim
def test_a_noise_trajectory_under_a_time_error_is_refused(tmp_path, monkeypatch):
    """The scoring marginalizes a timing error that the draw at the simulated times would
    ignore: one path honouring the option and the other not."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    from pybnf.parse import load_config
    R.require_bng2pl()
    (tmp_path / 'sir.bngl').write_text(L.MODEL)
    L.write_data(tmp_path / 'cases.exp')
    (tmp_path / 'am.conf').write_text('\n'.join([
        'edition = 2', 'model: sir.bngl', 'bngl_backend = bngsim', 'job_type = am',
        'output_dir = out', 'population_size = 1', 'max_iterations = 40',
        'experiment: epi, data: cases.exp, t_end: 14, n_steps: 140',
        'uniform_var = beta 0.3 1.2', 'uniform_var = gamma 0.1 0.5',
        'noise_model = neg_bin, dispersion = fix_at 20, location = mean, '
        'time_error = truncated_normal, sigma_t = fix_at 0.3',
        'output_noise_trajectory = cases']) + '\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        algorithms.Adaptive_MCMC(load_config(str(tmp_path / 'am.conf')))
    assert info.value.log_message == ('output_noise_trajectory names cases, but the fit has a '
                                      'time_error')


def test_the_output_noise_draws_the_predicted_mean_at_a_large_dispersion():
    """lanl/PyBNF#987: the draw clipped prob = r/(r + mean) at 1 - 1e-10 like the old score, so at
    r = 1e12 every predicted count below 100 was drawn as a count of about 100. There the negative
    binomial is Poisson to O(m / r): over 4000 draws the sample mean lies within 6 standard errors
    of m, and a prediction of 0 draws 0."""
    from types import SimpleNamespace
    from pybnf.algorithms.samplers.adaptive_mcmc import Adaptive_MCMC
    from pybnf.noise import MEAN, ConstantSigma, NegBinomial
    stub = SimpleNamespace(output_noise_specs={
        'c': (NegBinomial(location=MEAN), ConstantSigma(1e12), False)})
    series = np.array([0.0, 1.0, 5.0, 50.0])
    rng = np.random.default_rng(7)
    n = 4000
    draws = np.array([Adaptive_MCMC.generateBinomialNoise(stub, series, None, rng, 'c')
                      for _ in range(n)])
    assert draws[:, 0].max() == 0
    for j in (1, 2, 3):
        assert abs(draws[:, j].mean() - series[j]) < 6.0 * np.sqrt(series[j] / n), series[j]


# --------------------------------------------------------------------------------------------
# Trajectory outputs of every experiment, normalized once, and refused where wrong (#990)
# --------------------------------------------------------------------------------------------

def _parabola_result(name, values, score=5.0, *, master=False):
    """A scored Result of the parabola model's one experiment, ``par1``, carrying ``y``. On
    the worker-scoring path the data is on ``out``; on the master-scoring path
    (``local_objective_eval = 1``, ``smoothing > 1``) it stays on ``simdata`` and ``out`` is
    never set."""
    sim = {'parabola': {'par1': _data('y', list(enumerate(values)))}}
    res = algorithms.Result(_uniform_pset(name), sim if master else None, name)
    if not master:
        res.out = sim
    res.score = score
    return res


@pytest.mark.parametrize('wanted', [[], ['y']])
def test_a_chain_whose_first_simulation_failed_goes_on(tmp_path, wanted):
    """A chain's first result is always accepted, and a FailedSimulation has no ``out``, so a
    genuine failure of the first simulation stopped every am run with ``AttributeError:
    'FailedSimulation' object has no attribute 'out'``, trajectory outputs or not. Nothing is
    recorded for it; the chain's next, successful, result is."""
    cfg = _make_config(tmp_path, 1, UNIFORM_VARS,
                       **({'output_trajectory': wanted} if wanted else {}))
    am = algorithms.Adaptive_MCMC(cfg)
    am.start_run()
    failed = algorithms.FailedSimulation(_uniform_pset('iter0run0'), 'iter0run0', 1)
    failed.score = np.inf                          # as score_result leaves it
    proposals = am.got_result(failed)
    assert am.accept is True and am.iteration[0] == 1
    assert am.output_run_recorded == set() and am.output_run_current == {}
    assert [p.name for p in proposals] == ['iter1run0']
    am.got_result(_parabola_result('iter1run0', [1.0, 2.0, 3.0]))
    assert am.accept is True
    if wanted:
        np.testing.assert_array_equal(am.output_run_current['par1y'][0], [[1.0, 2.0, 3.0]])
        assert am.output_run_recorded == {('par1y', 0)}


def test_a_result_scored_on_the_master_is_recorded(tmp_path):
    """Scored on the master, a result keeps its data on ``simdata``, and only
    ``parallelize_models > 1`` copied it to ``out``: ``local_objective_eval = 1`` and
    ``smoothing > 1`` stopped at the first accepted result with ``AttributeError``."""
    am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS,
                                               output_trajectory=['y']))
    am.start_run()
    res = _parabola_result('iter0run0', [4.0, 5.0, 6.0], master=True)
    assert not hasattr(res, 'out')
    am.got_result(res)
    np.testing.assert_array_equal(am.output_run_current['par1y'][0], [[4.0, 5.0, 6.0]])


def test_an_experiment_whose_output_changes_length_is_refused(tmp_path):
    """A buffer is allocated at the length of the first output that supplies it, and a file
    row holds one sample of that length. A later output of another length used to fail inside
    numpy's broadcasting; it is refused, naming the experiment, the column and both lengths."""
    am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS,
                                               output_trajectory=['y']))
    am.start_run()
    am.got_result(_parabola_result('iter0run0', [1.0, 2.0, 3.0], score=5.0))
    with pytest.raises(PybnfError) as info:
        am.got_result(_parabola_result('iter1run0', [1.0, 2.0], score=1.0))
    assert info.value.log_message == ("Experiment par1's simulation gave 2 rows of y, where an "
                                      "earlier one gave 3")


def test_the_scored_series_is_recorded_as_it_was_scored_not_normalized_again(tmp_path):
    """The side that scored a result normalized its data in place before scoring it, so the
    trajectory is recorded as it arrives. It was normalized again here, with the whole-fit
    ``{data key: transforms}`` map passed as one experiment's transforms, which raised
    ``ValueError: too many values to unpack``. Oracle: the values handed in, which an ``init``
    normalization applied a second time would have divided by 2."""
    am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS, normalization='init',
                                               output_trajectory=['y']))
    assert am.config.config['normalization'] == {'par1': [('init', ['x', 'y'])]}
    am.start_run()
    am.got_result(_parabola_result('iter0run0', [2.0, 4.0, 8.0]))
    np.testing.assert_array_equal(am.output_run_current['par1y'][0], [[2.0, 4.0, 8.0]])


@pytest.mark.parametrize('key', ['output_trajectory', 'output_noise_trajectory'])
def test_a_trajectory_of_an_analytically_scaled_column_is_refused(tmp_path, key):
    """``scale`` is profiled against the data inside the objective, so the simulated series
    the file would hold is off the data's scale by the factor the fit applied."""
    with pytest.raises(PybnfError) as info:
        algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS, normalization='scale',
                                              **{key: ['y']}))
    assert info.value.log_message == f'{key} names y, which normalization scales analytically'
    assert 'normalization scales y analytically (scale) in experiment par1.' in info.value.message


def test_a_noise_trajectory_of_a_normalized_column_is_refused(tmp_path, monkeypatch):
    """The issue's edition-1 case, ``objfunc = neg_bin`` with ``normalization = init``, which
    stopped at the first accepted result with ``ValueError``. A count drawn around a normalized
    prediction is no count of anything, so the combination is refused when the configuration
    loads."""
    with pytest.raises(PybnfError) as info:
        _edition_1_noise_am(tmp_path, monkeypatch,
                            ['objfunc = neg_bin', 'neg_bin_r = 20', 'normalization = init'])
    assert info.value.log_message == ('output_noise_trajectory names cases_Cum, which '
                                      'normalization transforms')
    assert 'a normalized value in experiment epi, not an expected count' in info.value.message


def test_a_run_resumed_from_a_backup_made_before_990_records_every_experiment(tmp_path):
    """`-r` unpickles without running __init__, so a backup made before #990 has no
    ``trajectory_suffixes``: they are resolved from the configuration at the first result."""
    import pickle
    am = algorithms.Adaptive_MCMC(_make_config(tmp_path, 1, UNIFORM_VARS,
                                               output_trajectory=['y']))
    am.start_run()
    del am.trajectory_suffixes
    am = pickle.loads(pickle.dumps(am))
    assert 'trajectory_suffixes' not in vars(am)
    am.got_result(_parabola_result('iter0run0', [1.0, 2.0]))
    assert am.trajectory_suffixes == {'parabola': ['par1']}
    assert am.output_run_recorded == {('par1y', 0)}


@pytest.mark.parametrize('fit_type', ['mh', 'pt', 'dream', 'p_dream', 'de', 'check'])
def test_a_trajectory_output_outside_am_is_refused(tmp_path, fit_type):
    """Both keys are global, so every other job type accepted them and wrote nothing; the
    repository's own dream and p_dream benchmark configurations asked for them."""
    with pytest.raises(PybnfError) as info:
        _make_config(tmp_path, 1, UNIFORM_VARS, fit_type=fit_type, output_trajectory=['y'],
                     output_noise_trajectory=['y'])
    assert info.value.log_message == (
        'output_trajectory and output_noise_trajectory are written only by job_type = am, and '
        f'this is job_type = {fit_type}')


def _edition_2_config(tmp_path, monkeypatch, lines):
    """``load_config`` of an edition-2 job over the Liu-West SIR model (``cases`` is
    ``rho*CumInf``), with an unconditioned experiment ``epi`` and ``epi2`` under ``hi``."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    from pybnf.parse import load_config
    R.require_bng2pl()
    (tmp_path / 'sir.bngl').write_text(L.MODEL)
    L.write_data(tmp_path / 'cases.exp')
    (tmp_path / 'job.conf').write_text('\n'.join([
        'edition = 2', 'model: sir.bngl', 'job_type = am', 'output_dir = out',
        'experiment: epi, data: cases.exp', 'condition: hi, perturbations: rho = 0.6',
        'experiment: epi2, condition: hi, data: cases.exp',
        'uniform_var = beta 0.3 1.2', 'uniform_var = gamma 0.1 0.5',
        'population_size = 1', 'max_iterations = 40', 'burn_in = 10', 'adaptive = 10',
        *lines]) + '\n')
    monkeypatch.chdir(tmp_path)
    return load_config(str(tmp_path / 'job.conf'))


def test_the_recorded_experiments_are_the_configured_ones_conditions_included(
        tmp_path, monkeypatch):
    """The one model runs both actions under both conditions; the experiments are ``epi`` and
    ``epi2`` under ``hi``, data keys ``epi`` and ``epi2hi``. The buffers used to be keyed by the
    action suffixes ``epi`` and ``epi2`` (``time_length``), so ``epi2hi`` was never recorded."""
    from pybnf.algorithms.samplers.adaptive_mcmc import _experiment_suffixes
    cfg = _edition_2_config(tmp_path, monkeypatch,
                            ['noise_model = gaussian, sigma = fix_at 5, location = mean'])
    assert sorted(cfg.models['sir'].get_suffixes()) == ['epi', 'epi2', 'epi2hi', 'epihi']
    assert sorted(cfg.config['time_length']) == ['epi', 'epi2']
    assert _experiment_suffixes(cfg) == {'sir': ['epi', 'epi2hi']}


_FAMILY_LINES = {
    'gaussian': ('gaussian, sigma = fix_at 5', 'linear'),
    'lognormal': ('lognormal, sigma = fix_at 0.5', 'log'),
    'lnnormal': ('lnnormal, sigma = fix_at 0.5', 'log'),
    'laplace': ('laplace, scale = fix_at 5', 'linear'),
    'student_t': ('student_t, sigma = fix_at 5', 'linear'),
    'neg_bin': ('neg_bin, dispersion = fix_at 20, location = mean', None),
}


@pytest.mark.parametrize('family', sorted(_FAMILY_LINES))
def test_scale_is_refused_on_a_noise_family_with_no_additive_scale(tmp_path, monkeypatch,
                                                                   family):
    """``scale`` is profiled in the space the noise family adds its error in, which
    ``_scale_mode`` reads off the family's ``additive_on``. neg_bin has none, and every
    evaluation raised ``AttributeError`` in the objective, which the scoring turned into a
    failed simulation. Every other family loads, and its profile space resolves."""
    line, space = _FAMILY_LINES[family]
    lines = ['noise_model = gaussian, sigma = fix_at 5, location = mean',
             'noise_model cases = %s' % line, 'normalization cases = scale']
    if space is not None:
        cfg = _edition_2_config(tmp_path, monkeypatch, lines)
        assert cfg.obj._scale_mode('cases') == space
        return
    with pytest.raises(PybnfError) as info:
        _edition_2_config(tmp_path, monkeypatch, lines)
    assert info.value.log_message == ('normalization scales cases analytically, but its noise '
                                      'model NegBinomial has no additive scale')
    assert 'cases (experiment epi, epi2hi) is scored by NegBinomial' in info.value.message
    assert info.value.message.endswith('gaussian, laplace, lnnormal, lognormal, normal, '
                                       'student_t.')


def test_the_scale_refusal_reads_the_additive_scale_not_the_family_class():
    """Any family without an additive scale is refused, not only NegBinomial."""
    from types import SimpleNamespace

    class Counts:
        pass

    owner = SimpleNamespace(obj=SimpleNamespace(_spec_for=lambda column: (Counts(), {})))
    with pytest.raises(PybnfError) as info:
        config.Configuration._require_additive_scale(owner, {'e1': frozenset({'c'})})
    assert info.value.log_message == ('normalization scales c analytically, but its noise '
                                      'model Counts has no additive scale')


_DECAY = """begin model
begin parameters
  A0  100
  k   0.5
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() A0
end seed species
begin observables
  Molecules Atot A()
  Molecules Cnt A()
end observables
begin reaction rules
  A() -> 0  k
end reaction rules
end model
"""


def _trajectories(runs, name):
    """The rows of one per-chain trajectory file, without the never-simulated all-zero row
    the first write puts at the top (lanl/PyBNF#859, a separate issue)."""
    rows = np.loadtxt(runs / name, ndmin=2)
    return rows[~(rows == 0).all(axis=1)]


@pytest.mark.bngsim
@pytest.mark.parametrize('local_objective_eval', [0, 1])
def test_an_edition_2_run_writes_every_experiment_normalized_once(tmp_path, monkeypatch,
                                                                  local_objective_eval):
    """A first-order decay A(t) = A0 exp(-k t), with A0 the fitted parameter, measured in
    experiment ``slow`` (k = 0.5) and in ``quick`` under the condition ``fast`` (k = 1).

    Oracle, in closed form and the same for every row whatever A0 the chain was at: ``Atot``,
    normalized by ``floor 0.1, init``, is (exp(-k t) + 0.1) / 1.1, because the floor adds 0.1
    of the maximum, A0 at t = 0, and init divides by the t = 0 value. The floor is not
    idempotent, so a second normalization would show. ``Cnt``, the same species unnormalized,
    is A0 exp(-k t), a row divided by its first entry is exp(-k t), and its noise trajectory
    holds counts of the same shape. ``quick`` under ``fast`` (data key ``quickfast``) used to
    get no file at all, and the normalization used to stop the run. Both scoring paths: the
    workers (``local_objective_eval = 0``) and the master (1), which ``am`` could not run."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    R.require_bng2pl()
    (tmp_path / 'decay.bngl').write_text(_DECAY)
    (tmp_path / 'decay.exp').write_text('# time Atot Cnt\n' + ''.join(
        '%d %g %d\n' % (t, 100 * np.exp(-0.5 * t), round(100 * np.exp(-0.5 * t)))
        for t in range(1, 6)))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'edition = 2', 'model: decay.bngl', 'bngl_backend = bngsim', 'job_type = am',
        'output_dir = out', 'experiment: slow, data: decay.exp',
        'condition: fast, perturbations: k = 1',
        'experiment: quick, condition: fast, data: decay.exp',
        'noise_model = gaussian, sigma = fix_at 1, location = mean',
        'noise_model Cnt = neg_bin, dispersion = fix_at 50, location = mean',
        'normalization Atot = floor 0.1, init', 'uniform_var = A0 50 200',
        'output_trajectory = Atot, Cnt', 'output_noise_trajectory = Cnt',
        'population_size = 1', 'max_iterations = 10', 'burn_in = 2', 'adaptive = 2',
        'sample_every = 1', 'step_size = 20', 'random_seed = 1', 'parallel_count = 1',
        'local_objective_eval = %d' % local_objective_eval]) + '\n')
    assert L.run_main(monkeypatch, tmp_path, tmp_path / 'am.conf') == 0
    runs = tmp_path / 'out' / 'Results' / 'A_MCMC' / 'Runs'
    assert sorted(p.name for p in runs.glob('traj_*')) == [
        'traj_noise_quickfastCnt_chain_0.txt', 'traj_noise_slowCnt_chain_0.txt',
        'traj_quickfastAtot_chain_0.txt', 'traj_quickfastCnt_chain_0.txt',
        'traj_slowAtot_chain_0.txt', 'traj_slowCnt_chain_0.txt']
    times = np.arange(6.0)
    for experiment, k in (('slow', 0.5), ('quickfast', 1.0)):
        normalized = _trajectories(runs, 'traj_%sAtot_chain_0.txt' % experiment)
        assert normalized.shape[0] >= 5
        expected = (np.exp(-k * times) + 0.1) / 1.1
        np.testing.assert_allclose(normalized, np.tile(expected, (len(normalized), 1)),
                                   rtol=1e-5)
        counts = _trajectories(runs, 'traj_%sCnt_chain_0.txt' % experiment)
        assert len(set(counts[:, 0])) > 1          # the chain moved A0, and Cnt with it
        np.testing.assert_allclose(counts / counts[:, :1],
                                   np.tile(np.exp(-k * times), (len(counts), 1)), rtol=1e-5)
        drawn = np.loadtxt(runs / ('traj_noise_%sCnt_chain_0.txt' % experiment), ndmin=2)
        assert drawn.shape[1] == 6 and (drawn >= 0).all() and (drawn == np.round(drawn)).all()


@pytest.mark.bngsim
def test_an_edition_1_mutant_gets_its_trajectories_normalized_as_scored(tmp_path, monkeypatch):
    """The edition-1 sibling: a mutant ``hi`` (rho = 0.6 against 0.5) of the SIR model, whose
    output suffix ``epihi`` had no buffer, under the legacy per-file ``normalization = peak:
    epi.exp``, which stopped the run. Oracle, from ``cases`` = rho * CumInf with ``CumInf``
    recorded beside it (in no data file, so unnormalized, and the same in both experiments,
    since rho changes no dynamics): ``epi``'s ``cases``, peak-normalized as scored, is CumInf /
    max(CumInf), and ``epihi``'s, not normalized, is 0.6 * CumInf."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    R.require_bng2pl()
    (tmp_path / 'sir1.bngl').write_text(_E1_SIR)
    for name in ('epi.exp', 'epihi.exp'):
        (tmp_path / name).write_text('# time cases cases_Cum\n' + ''.join(
            '%d %d %d\n' % (t, c, c) for t, c in enumerate(L.COUNTS, 1)))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'model = sir1.bngl : epi.exp', 'mutant = sir1 hi rho=0.6 : epihi.exp',
        'fit_type = am', 'objfunc = sos', 'normalization = peak: epi.exp', 'output_dir = out',
        'uniform_var = beta__FREE 0.3 1.2', 'uniform_var = gamma__FREE 0.1 0.5',
        'output_trajectory = cases, CumInf', 'population_size = 1', 'max_iterations = 10',
        'burn_in = 2', 'adaptive = 2', 'sample_every = 1', 'step_size = 0.05',
        'random_seed = 1', 'parallel_count = 1']) + '\n')
    assert L.run_main(monkeypatch, tmp_path, tmp_path / 'am.conf') == 0
    runs = tmp_path / 'out' / 'Results' / 'A_MCMC' / 'Runs'
    assert sorted(p.name for p in runs.glob('traj_*')) == [
        'traj_epiCumInf_chain_0.txt', 'traj_epicases_chain_0.txt',
        'traj_epihiCumInf_chain_0.txt', 'traj_epihicases_chain_0.txt']
    cum, cum_hi = (_trajectories(runs, 'traj_%sCumInf_chain_0.txt' % s) for s in ('epi', 'epihi'))
    cases, cases_hi = (_trajectories(runs, 'traj_%scases_chain_0.txt' % s)
                       for s in ('epi', 'epihi'))
    assert cum.shape == cum_hi.shape == cases.shape == cases_hi.shape and cum.shape[0] >= 5
    assert (cum.max(axis=1) > 0).all()
    np.testing.assert_allclose(cum_hi, cum, rtol=1e-9)
    np.testing.assert_allclose(cases, cum / cum.max(axis=1, keepdims=True), rtol=1e-9)
    np.testing.assert_allclose(cases_hi, 0.6 * cum_hi, rtol=1e-9)
