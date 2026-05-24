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
