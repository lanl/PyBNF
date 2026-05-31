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
``normal_var`` parameters are used for the proposal so reflection at the bounds
does not distort the Gaussian.
"""
import numpy as np
import pytest

from .context import algorithms, config, pset

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
        np.random.seed(20240517)
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
        np.random.seed(7)
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
# These tests freeze the Gaussian draw (monkeypatching np.random.multivariate_
# normal) so each contract is checked against its analytical oracle, not RNG.
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


def _freeze_draw_to_zero(monkeypatch):
    """Replace the Gaussian proposal draw with the zero vector so pick_new_pset's
    state updates (mean/cov/scale) are isolated and the trivial step lands in
    bounds on the first try."""
    monkeypatch.setattr(np.random, 'multivariate_normal',
                        lambda mean, cov: np.zeros(len(mean)))


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
        _freeze_draw_to_zero(monkeypatch)

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
        _freeze_draw_to_zero(monkeypatch)

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
        _freeze_draw_to_zero(monkeypatch)

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
        _freeze_draw_to_zero(monkeypatch)
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
        _freeze_draw_to_zero(monkeypatch)

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
        monkeypatch.setattr(np.random, 'multivariate_normal', fake_mvn)

        prop = am.pick_new_pset(0)

        names = list(base.keys())
        for i, n in enumerate(names):
            np.testing.assert_allclose(prop[n] - base[n], D * delta[i], rtol=1e-12)
        # The draw sampled from the adapted (post-update) covariance, once.
        np.testing.assert_allclose(rec['cov'], am.diffMatrix[0], rtol=1e-12)
        assert rec['calls'] == 1
