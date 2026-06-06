"""Integration tests for the MCMC samplers against analytical posteriors.

With ``direct_pass`` and a Gaussian target, the negative-log-likelihood is the
Gaussian NLL, the priors are uniform (flat) and wide, so the posterior is
exactly ``N(mean, diag(variance))`` — closed-form moments to check against.

Two tiers (see ``integration_harness``):

  * **fast** (every change): cheap *sanity / directional* invariants on a short
    chain — the sampler runs, produces finite samples, the pooled chain mean has
    moved to the mode, and the acceptance rate is in a sane band. Short chains
    are statistically noisy, so tolerances are deliberately loose; these catch
    wiring regressions (a broken acceptance rule, proposal, or output path), not
    fine posterior accuracy.
  * **slow** (``-m slow``, opt-in): full moment recovery with tight tolerances —
    the gold-standard check to run before/after the critical algorithm patches.

NOTE on per-tier runtime: the samplers stream per-step output to disk, so full
posterior recovery is inherently slow (seconds–minutes). That is why recovery
lives in the slow tier and the fast tier asserts only directional invariants.
"""
import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms


SAMPLERS = {
    'am': algorithms.Adaptive_MCMC,
    'dream': algorithms.DreamAlgorithm,
    'p_dream': algorithms.PDreamAlgorithm,
}


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _am_keys(**extra):
    base = dict(output_hist_every=10 ** 9, hist_bins=10, num_bins=10,
                credible_intervals=[68, 95], rhat_threshold=0, step_size=0.6)
    base.update(extra)
    return base


def _short_config(tmp_path, fit_type, mean, var):
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    common = dict(burn_in=150, sample_every=2, rhat_threshold=0,
                  output_hist_every=10 ** 9, hist_bins=10)
    if fit_type == 'am':
        kw = dict(common, population_size=3, max_iterations=450, adaptive=100,
                  num_bins=10, credible_intervals=[68, 95], step_size=0.6)
    else:  # dream / p_dream
        kw = dict(common, population_size=5, max_iterations=450)
    return H.make_config(tmp_path, fit_type, tgt, exp, len(mean), **kw)


# --------------------------------------------------------------------------- #
# FAST: directional / sanity invariants on a short chain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('fit_type', list(SAMPLERS))
def test_sampler_moves_to_mode(tmp_path, fit_type):
    mean, var = [1.0, -1.0], [1.0, 1.0]
    conf = _short_config(tmp_path, fit_type, mean, var)
    alg = SAMPLERS[fit_type](conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], len(mean))
    assert len(samples) > 0, 'no samples written'
    assert np.all(np.isfinite(samples)), 'non-finite samples'
    # The chain has moved off the flat prior and concentrated near the mode.
    chain_mean = samples.mean(axis=0)
    assert np.allclose(chain_mean, mean, atol=0.7), \
        '%s chain mean %s not near mode %s' % (fit_type, chain_mean, mean)
    # And it is actually moving — not frozen at a single point (which is what a
    # broken proposal or all-reject acceptance rule looks like). Recorded samples
    # are post-thinning, so a healthy chain's consecutive-differ rate is ~1; only
    # the lower bound is meaningful here. Posterior *spread* is checked in the
    # slow recovery tier.
    rate = H.acceptance_rate(samples)
    assert rate > 0.3, '%s looks frozen (move rate %.3f)' % (fit_type, rate)


@pytest.mark.parametrize('fit_type', list(SAMPLERS))
def test_same_seed_reproduces_saved_samples(tmp_path, fit_type):
    """End-to-end reproducibility: a full fit re-run with the same ``random_seed``
    writes byte-identical samples and best fit. This is the workflow guarantee the
    default_rng migration must preserve -- same seed -> same saved data -- exercised
    through the real run loop and output files, not just the proposal math.
    Per-chain ``SeedSequence.spawn`` keeps it true regardless of the order results
    come back. (The transitive stochastic-sim case -- reproducible params ->
    reproducible derived sim seeds -> reproducible trajectories -- is locked by
    ``test_bngsim_bngl_e2e.test_bngsim_ssa_same_seed_reproduces_trajectory``.)"""
    mean, var = [1.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))

    def run(sub):
        common = dict(burn_in=80, sample_every=2, rhat_threshold=0,
                      output_hist_every=10 ** 9, hist_bins=10,
                      output_dir=str(tmp_path / sub), random_seed=4321)
        if fit_type == 'am':
            kw = dict(common, population_size=3, max_iterations=240, adaptive=60,
                      num_bins=10, credible_intervals=[68, 95], step_size=0.6)
        else:  # dream / p_dream
            kw = dict(common, population_size=5, max_iterations=240)
        conf = H.make_config(tmp_path, fit_type, tgt, exp, len(mean), **kw)
        alg = SAMPLERS[fit_type](conf)
        H.drive(alg)
        return (H.read_samples(conf.config['output_dir'], len(mean)),
                H.best_params(alg, len(mean)))

    samples1, best1 = run('repro_a')
    samples2, best2 = run('repro_b')
    assert samples1.size > 0, 'no samples written'
    assert samples1.shape == samples2.shape
    np.testing.assert_array_equal(samples1, samples2)
    np.testing.assert_array_equal(best1, best2)


# --------------------------------------------------------------------------- #
# SLOW: full posterior-moment recovery against the analytical truth
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize('fit_type', list(SAMPLERS))
def test_sampler_recovers_gaussian_moments(tmp_path, fit_type):
    # 2-D keeps the serial in-process run tractable (no dask parallelism here, so
    # cost is population_size * max_iterations evaluations, run one at a time).
    # Runtime is dominated by the convergence diagnostics, which fire every 10
    # iterations and rank-normalize the *full growing chain history* — so wall
    # time scales ~O(max_iterations^2), not linearly. max_iterations is therefore
    # the lever; these budgets still leave ample effective samples to recover a
    # 2-D Gaussian within the tolerances below.
    mean, var = [2.0, -1.0], [1.0, 4.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    common = dict(sample_every=2, rhat_threshold=0,
                  output_hist_every=10 ** 9, hist_bins=20)
    if fit_type == 'am':
        # am samples only after burn_in + adaptive; leave ~800 post-adaptation
        # iterations across 3 chains (~1200 thinned samples).
        kw = dict(common, population_size=3, burn_in=800, adaptive=800,
                  max_iterations=2400, num_bins=20,
                  credible_intervals=[68, 95], step_size=0.5)
    else:
        # dream/p_dream mix fast; ~1300 post-burn-in iterations across 4 chains.
        kw = dict(common, population_size=4, burn_in=700, max_iterations=2000)
    conf = H.make_config(tmp_path, fit_type, tgt, exp, len(mean), **kw)
    alg = SAMPLERS[fit_type](conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], len(mean))
    assert len(samples) > 200, 'too few samples: %d' % len(samples)
    rec_mean = samples.mean(axis=0)
    rec_std = samples.std(axis=0, ddof=1)
    true_std = np.sqrt(var)
    assert np.allclose(rec_mean, mean, atol=0.25), \
        '%s recovered mean %s vs %s' % (fit_type, rec_mean, mean)
    # Posterior spread within ~30% of the analytical standard deviation.
    assert np.allclose(rec_std, true_std, rtol=0.3), \
        '%s recovered std %s vs %s' % (fit_type, rec_std, true_std)
