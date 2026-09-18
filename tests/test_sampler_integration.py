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
from pathlib import Path

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


def test_output_inference_data_with_non_likelihood_records_no_sidecar(tmp_path):
    """The LOO/WAIC no-op gate, end to end through a real am run: output_inference_data=1
    with direct_pass (not a per-point likelihood) leaves _record_loglik False, so the run
    writes no log_likelihood.txt -- loo/waic are simply not offered where they would be
    invalid (ADR-0056). The InferenceData (sans log_likelihood group) is still emitted."""
    import os
    mean, var = [0.3, -0.7], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    kw = dict(burn_in=120, sample_every=2, rhat_threshold=0, output_hist_every=10 ** 9,
              hist_bins=10, population_size=3, max_iterations=240, adaptive=60,
              num_bins=10, credible_intervals=[68, 95], step_size=0.6,
              output_inference_data=1)
    conf = H.make_config(tmp_path, 'am', tgt, exp, len(mean), **kw)
    alg = SAMPLERS['am'](conf)
    assert alg._record_loglik is False  # direct_pass is not a per-point likelihood
    H.drive(alg)
    assert not os.path.exists(os.path.join(conf.config['output_dir'], 'Results', 'log_likelihood.txt'))


def test_am_samples_a_flat_posterior_evenly_up_to_the_walls(tmp_path):
    """The proposal and the acceptance rule held to one target together (#709).

    A flat likelihood over ``uniform_var [0, 1]`` makes the posterior exactly
    Uniform(0, 1), so nothing here is a tolerance on a fit: the variance is 1/12 and a
    fifth of the mass lies within 0.1 of a wall. The step is 0.3, long enough for a wall
    to be in reach from most of the box.

    am used to redraw a proposal until it landed in the box and then accept it with the
    plain Metropolis ratio. The redrawn proposal is not symmetric -- its density is the
    Gaussian renormalized by the share of it inside the box, Z(x), which falls toward a
    wall -- so the chain sampled Z(x) instead of a constant: variance 0.0712 and 0.149
    near the walls on this target, by quadrature, and 0.0707-0.0723 and 0.145-0.156 over
    eight seeds of this test. Rejecting the proposal that leaves the box gives
    0.0819-0.0844 and 0.191-0.214 over the same seeds. Each threshold sits between its
    two values, about five standard errors or more from both, so the pinned seed is not
    what passes it.

    The acceptance tests and the proposal tests in test_adaptive_mcmc.py each pin their
    half exactly, and both passed while this was wrong; only a known target sees whether
    the two fit together. Run through the real loop, with two chains, so both shapes of a
    rejected generation come up thousands of times: one chain rejected at the boundary
    while the other simulates, and both rejected with nothing to submit -- which, returned
    as an empty generation, the scheduler reads as a job pool run dry and ends the run on.
    """
    iterations = 5000
    # variance 1e12 on a unit box: the NLL moves by less than 1e-12 across it.
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.5], [1e12]))
    conf = H.make_config(
        tmp_path, 'am', tgt, exp, 1, bounds=(0.0, 1.0), population_size=2, step_size=0.3,
        max_iterations=iterations,
        # The fixed-step branch for the whole run: on a flat target the adaptive scale has
        # nothing to settle on, and the bias is the same kernel's either way.
        burn_in=1, adaptive=iterations - 3, sample_every=1, rhat_threshold=0,
        output_hist_every=10 ** 9, hist_bins=10, credible_intervals=[68, 95],
        # Off, because they are the run loop's cost and not the sampler's: the trajectory
        # rewrite and the pickle are each per result, and together several times the fit.
        diagnostics_every=10 ** 9, backup_every=10 ** 9, output_every=10 ** 9)
    alg = SAMPLERS['am'](conf)
    H.drive(alg)

    x = H.read_samples(conf.config['output_dir'], 1)[:, 0]
    near_a_wall = np.mean((x < 0.1) | (x > 0.9))
    assert abs(x.var() - 1 / 12) < 0.006, 'variance %.4f, uniform is 0.0833' % x.var()
    assert near_a_wall > 0.167, '%.3f of the samples within 0.1 of a wall, uniform is 0.200' % near_a_wall

    # The run went the distance: every iteration of both chains left its row, the ones
    # spent on a boundary rejection included, and those cost no simulation.
    assert len(x) == 2 * (iterations - 1)
    assert alg.boundary_rejections > 1000
    assert alg.total_evaluations == 2 * iterations - alg.boundary_rejections


@pytest.mark.parametrize('fit_type,extra', [
    ('p_dream', {}),
    ('dream', {'proposal': 'whitened'}),
], ids=['p_dream', 'dream+whitened'])
def test_whitened_proposal_runs_with_one_parameter(tmp_path, fit_type, extra):
    """A one-parameter fit with the whitened proposal, run to completion (#767).

    ``np.cov`` of a single column returns a 0-d array rather than a 1x1 matrix, and
    ``_update_covariance`` took its trace, so the run died with ``ValueError: diag
    requires an array of at least two dimensions``. Not at startup: the covariance
    refresh first fires at ``precondition_adapt``, which defaults to ``burn_in // 2``
    -- always before ``burn_in``, so the crash landed before the first sample was
    recorded and the whole run was lost. Both entry points to the proposal are
    covered, since ``dream`` can opt into it as well as ``p_dream`` defaulting to it.

    Running to completion is most of the point, but on its own it would also pass for
    a fix that skipped one-parameter preconditioning altogether, so ``_preconditioned``
    is asserted: the whitened path must actually be on. The target is #766's flat-box
    oracle -- a flat likelihood over ``uniform_var [0, 1]`` makes the posterior exactly
    Uniform(0, 1), variance 1/12 with a fifth of the mass within 0.1 of a wall -- which
    holds this sampler to the same bound handling as its siblings and fails if the 1x1
    preconditioner is built wrong rather than merely built.

    Thresholds are sized from a 20-seed sweep of this config, which gave variance
    0.0812-0.0896 and near-wall fraction 0.188-0.226; the bands below clear those and
    still exclude the pi(x)Z(x) signature #766 measured on this oracle (0.0707-0.0723
    and 0.145-0.156), so the pinned seed is not what passes them.
    """
    iterations, burn_in, chains = 1500, 200, 5
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.5], [1e12]))
    conf = H.make_config(
        tmp_path, fit_type, tgt, exp, 1, bounds=(0.0, 1.0), population_size=chains,
        max_iterations=iterations, burn_in=burn_in, sample_every=1, rhat_threshold=0,
        output_hist_every=10 ** 9, hist_bins=10,
        # The run loop's per-result pickle and O(n^2) trajectory rewrite cost several
        # times the sampler itself; off, as in the am flat-box test above.
        diagnostics_every=10 ** 9, backup_every=10 ** 9, output_every=10 ** 9, **extra)
    alg = SAMPLERS[fit_type](conf)
    H.drive(alg)

    assert alg._preconditioned, 'the whitened proposal never activated'
    x = H.read_samples(conf.config['output_dir'], 1)[:, 0]
    # Every post-burn-in iteration of every chain left a row: the run went the distance.
    assert len(x) == chains * (iterations - burn_in)
    near_a_wall = np.mean((x < 0.1) | (x > 0.9))
    assert abs(x.var() - 1 / 12) < 0.012, 'variance %.4f, uniform is 0.0833' % x.var()
    assert near_a_wall > 0.16, \
        '%.3f of the samples within 0.1 of a wall, uniform is 0.200' % near_a_wall


@pytest.mark.parametrize('fit_type', list(SAMPLERS))
def test_sampler_writes_final_histograms_and_credible_intervals(tmp_path, fit_type):
    """Every sampler ends its run by writing the marginal histogram and the credible
    intervals for each free parameter — the posterior summaries a Bayesian fit is run
    for, and what the docs present as its output (#771).

    ``am`` did neither. It overrode ``update_histograms`` with a bare ``pass``, so the
    stride call it already made was a no-op and the ``Results/Histograms`` directory it
    already created stayed empty, while it accepted ``credible_intervals``,
    ``hist_bins`` and ``output_hist_every`` without a word — they sit on the shared
    MCMC config, so nothing reported them as unused. Its stop path was also missing the
    ``update_histograms('_final')`` that sits next to ``report_constraint_satisfaction``
    in every other sampler, so removing the override alone would still have left no
    ``*_final`` files. This runs to ``max_iterations``, which is the path that needed
    that call.

    Stated over every sampler rather than for ``am`` alone, because it is a shared
    contract and a per-sampler test is exactly what was missing: nothing asserted that a
    given fit_type reaches this output path at all.

    The oracles are structural rather than statistical, so this stays a fast test: the
    bin counts must account for every recorded sample, and the 68% interval must nest
    inside the 95% one. Both fail on an empty or misread sample matrix.
    """
    n_params = 2
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.5] * n_params, [0.04] * n_params))
    common = dict(burn_in=100, sample_every=2, rhat_threshold=0, max_iterations=400,
                  output_hist_every=5, hist_bins=10, credible_intervals=[68, 95],
                  diagnostics_every=10 ** 9, backup_every=10 ** 9, output_every=10 ** 9)
    if fit_type == 'am':
        kw = dict(common, population_size=3, adaptive=100, num_bins=10, step_size=0.3)
    else:
        kw = dict(common, population_size=5)
    conf = H.make_config(tmp_path, fit_type, tgt, exp, n_params, bounds=(0.0, 1.0), **kw)
    alg = SAMPLERS[fit_type](conf)
    H.drive(alg)

    results = Path(conf.config['output_dir']) / 'Results'
    samples = H.read_samples(conf.config['output_dir'], n_params)
    assert len(samples) > 0, 'no samples written'

    for i in range(n_params):
        hist_file = results / 'Histograms' / ('p%d_final.txt' % (i + 1))
        assert hist_file.is_file(), '%s wrote no final histogram for p%d' % (fit_type, i + 1)
        hist = np.genfromtxt(hist_file)
        assert hist.shape == (10, 3)                       # lower edge, upper edge, count
        # Every recorded sample is accounted for -- so the summary describes the run's
        # own samples, not a truncated or empty read of them.
        assert hist[:, 2].sum() == len(samples)

    bounds = {}
    for interval in (68, 95):
        cred_file = results / ('credible%d_final.txt' % interval)
        assert cred_file.is_file(), \
            '%s wrote no final credible%d file' % (fit_type, interval)
        lines = cred_file.read_text().splitlines()
        assert lines[0] == '# param\tlower_bound\tupper_bound'
        assert len(lines) == n_params + 1
        bounds[interval] = {}
        for line in lines[1:]:
            name, lo, hi = line.split('\t')
            bounds[interval][name] = (float(lo), float(hi))

    for i in range(n_params):
        name = 'p%d' % (i + 1)
        lo68, hi68 = bounds[68][name]
        lo95, hi95 = bounds[95][name]
        assert lo68 < hi68 and lo95 < hi95
        # The wider interval contains the narrower one: both are order statistics of the
        # same sorted column, so this fails if the two were built from different data.
        assert lo95 <= lo68 and hi68 <= hi95, \
            '%s %s: 68%% [%g, %g] not inside 95%% [%g, %g]' % (fit_type, name, lo68, hi68, lo95, hi95)
        # And they bracket the column they summarize.
        assert lo68 >= samples[:, i].min() and hi68 <= samples[:, i].max()


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
