"""HMC (blackjax NUTS) reference sampler — oracle-anchored tests (ADR-0059, #425).

The first end-to-end slice: ``job_type = hmc`` drives blackjax NUTS on an analytical
target's JAX log-density, writes draws in the standard samples format, and must RECOVER the
**closed-form** posterior moments. With a wide ``uniform_var`` (flat) prior over a target
whose NLL is a Gaussian quadratic form, the posterior IS that Gaussian — exact mean /
variance / covariance to check against, so the test anchors on analytic truth, not on
another sampler's output.

The whole module is skipped when the optional ``pybnf[jax]`` extra (jax + blackjax) is
absent — mirroring the project's other optional-extra test modules (arviz / petab). The
diagnostics are PyBNF's own (rank-normalized split-R-hat, bulk/tail ESS, ``pybnf.diagnostics``
via the sampler), so HMC's output drops into the same comparison machinery the samplers it
benchmarks use.
"""
import importlib.util

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms

# Guard the whole module on the optional pybnf[jax] extra (ADR-0059): no jax/blackjax ->
# no gradient-based sampler to exercise. find_spec avoids importing the heavy stack just to
# decide whether to skip.
_HAS_JAX = all(importlib.util.find_spec(m) is not None for m in ('jax', 'blackjax'))
pytestmark = pytest.mark.skipif(
    not _HAS_JAX, reason='requires the optional pybnf[jax] extra (jax + blackjax)')


def _hmc_config(tmp_path, spec, n_params, *, num_chains=4, num_warmup=800,
                num_samples=1500, bounds=(-12.0, 12.0), **overrides):
    """A real ``Configuration`` for an ``hmc`` fit over ``spec`` with wide uniform priors.

    Wide ``uniform_var`` bounds keep the (concentrated) posterior far inside the box, so it
    is effectively a flat prior and the posterior equals the target Gaussian (the box walls
    are never reached — the constrained-boundary case is ADR-0059's deferred follow-on)."""
    tgt, exp = H.write_target(tmp_path, spec)
    # max_iterations is a globally-required config key; HMC drives off num_warmup/num_samples
    # instead and ignores it, so echo num_samples to satisfy the validator.
    kw = dict(population_size=num_chains, num_warmup=num_warmup, num_samples=num_samples,
              max_iterations=num_samples, random_seed=20260627)
    kw.update(overrides)
    return H.make_config(tmp_path, 'hmc', tgt, exp, n_params, bounds=bounds, **kw)


# --------------------------------------------------------------------------- #
# Closed-form posterior-moment recovery (the analytic-truth oracle)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('mean,variance', [
    ([0.0, 0.0], [1.0, 1.0]),     # standard 2-D normal
    ([2.0, -1.0], [1.0, 4.0]),    # shifted + anisotropic (diagonal)
])
def test_hmc_recovers_gaussian_moments(tmp_path, mean, variance):
    conf = _hmc_config(tmp_path, H.gaussian_spec(mean, variance), len(mean))
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], len(mean))
    assert samples.shape[0] == conf.config['population_size'] * conf.config['num_samples']
    assert np.all(np.isfinite(samples))

    rec_mean = samples.mean(axis=0)
    rec_var = samples.var(axis=0, ddof=1)
    np.testing.assert_allclose(rec_mean, mean, atol=0.1)
    np.testing.assert_allclose(rec_var, variance, rtol=0.12)

    # PyBNF's own diagnostics, on the NUTS draws, drop in unchanged: well-mixed chains.
    rhat = alg.compute_rhat()
    bulk_ess, _tail_ess = alg.compute_ess()
    assert rhat is not None and np.nanmax(rhat) < 1.05
    assert np.nanmin(bulk_ess) > 400   # ~near-independent NUTS draws -> healthy ESS


def test_hmc_binds_coordinates_by_name_not_declaration_order(tmp_path):
    """Bind-by-name (ADR-0034) through the HMC path: declaring the parameters in REVERSE
    coordinate order must not swap which coordinate each binds to.

    ``nll_jax`` consumes ``theta`` in the target's coordinate order, while the sampler builds
    ``u`` in declaration order, so HMC permutes ``u`` -> coordinate order by name
    (``_coordinate_permutation``). An asymmetric gaussian (distinct per-coordinate means) makes
    a mis-binding visible: with the fix ``p1`` recovers coordinate 1's mean and ``p2``
    coordinate 2's, despite ``p2`` being declared first; a positional binding would swap them."""
    mean, variance = [2.0, -1.0], [1.0, 4.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, variance))
    base = {
        'output_dir': str(tmp_path) + '/out',
        'models': {tgt}, tgt: [exp], 'exp_data': {exp},
        'objfunc': 'direct_pass', 'fit_type': 'hmc', 'initialization': 'lh',
        'delete_old_files': 1, 'verbosity': 0, 'wall_time_sim': 0, 'random_seed': 20260627,
        'population_size': 4, 'num_warmup': 800, 'num_samples': 1500, 'max_iterations': 1500,
    }
    base[('uniform_var', 'p2')] = [-12.0, 12.0]   # declared BEFORE p1 -> variables == [p2, p1]
    base[('uniform_var', 'p1')] = [-12.0, 12.0]
    conf = H.config.Configuration(base)
    assert [v.name for v in conf.variables] == ['p2', 'p1']   # declaration order really is reversed
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 2)   # columns ordered p1, p2 by name
    rec = samples.mean(axis=0)
    np.testing.assert_allclose(rec, mean, atol=0.15)          # p1 -> coord1 (2), p2 -> coord2 (-1)


def test_hmc_recovers_rotated_gaussian_covariance(tmp_path):
    """A non-trivial off-diagonal covariance: HMC must recover the full Sigma, not just the
    marginals — the discriminating check that the correlated geometry is sampled correctly."""
    mean = [0.0, 0.0]
    cov = H.rotated_cov([2.0, 0.5], angle=np.pi / 6)   # tilted -> nonzero off-diagonal
    assert abs(cov[0, 1]) > 0.4                          # the spec really is correlated
    conf = _hmc_config(tmp_path, H.rotated_gaussian_spec(mean, cov), len(mean))
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], len(mean))
    assert np.all(np.isfinite(samples))
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.1)
    rec_cov = np.cov(samples, rowvar=False)
    np.testing.assert_allclose(rec_cov, cov, atol=0.2)
    # The recovered off-diagonal has the right sign and is non-trivial (a diagonal sampler
    # would fail this), confirming the correlation was captured, not averaged away.
    assert np.sign(rec_cov[0, 1]) == np.sign(cov[0, 1])
    assert abs(rec_cov[0, 1]) > 0.5 * abs(cov[0, 1])


# --------------------------------------------------------------------------- #
# Closed-form banana posterior-moment recovery (the curved stress-geometry oracle)
# --------------------------------------------------------------------------- #
def _banana_moments(a, b):
    """Closed-form posterior moments of the 2-D banana ``0.5[(a-x1)^2 + b(x2-x1^2)^2]``.

    With a flat prior, ``p(x) ∝ exp(-NLL)`` factorizes as ``x1 ~ N(a, 1)`` and
    ``x2 | x1 ~ N(x1^2, 1/b)``. Hence ``E[x1]=a``, ``Var[x1]=1``; ``E[x2]=E[x1^2]=
    1+a^2``; and, using ``Var[X^2]=2σ⁴+4μ²σ²=2+4a²`` for ``X~N(a,1)`` together with
    the law of total variance, ``Var[x2]=E[1/b]+Var[x1^2]=1/b+2+4a²``. (Derivation
    checked against the factorization above, not assumed — ADR-0059 test brief.)"""
    mean = np.array([a, 1.0 + a ** 2])
    var = np.array([1.0, 1.0 / b + 2.0 + 4.0 * a ** 2])
    return mean, var


def test_hmc_recovers_banana_moments(tmp_path):
    """HMC recovers the closed-form moments of the curved, non-Gaussian banana — the
    canonical stress geometry the gradient-free samplers are scored against (ADR-0059).
    The analytic truth (``x1~N(a,1)``, ``x2|x1~N(x1²,1/b)``) is the oracle, not another
    sampler.

    A gentle ``b=8`` with a Stan-tight ``target_accept=0.95`` is what makes HMC a *clean*
    reference here: the small step size lets NUTS negotiate the sharp tip without a single
    divergent transition (``b=100`` is sharp enough that NUTS diverges and its own R-hat
    rejects it — exactly the honesty the reliability gate enforces). All three of HMC's own
    diagnostics — rank-normalized split-R-hat, bulk ESS, and the NUTS divergence count —
    must pass before its draws are trusted as a yardstick."""
    a, b = 1.0, 8.0
    true_mean, true_var = _banana_moments(a, b)
    conf = _hmc_config(tmp_path, H.banana_spec(a=a, b=b), 2, num_chains=4,
                       num_warmup=1200, num_samples=2500, target_accept=0.95,
                       bounds=(-25.0, 25.0))
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 2)
    assert samples.shape[0] == conf.config['population_size'] * conf.config['num_samples']
    assert np.all(np.isfinite(samples))

    rec_mean = samples.mean(axis=0)
    rec_var = samples.var(axis=0, ddof=1)
    # E[x1]=a=1, E[x2]=1+a²=2 ; Var[x1]=1, Var[x2]=1/b+2+4a²≈6.125. The x2 mean/variance
    # carry the heavy right tail of x1², so they get the looser (absolute / relative)
    # tolerances; x1 is a clean unit Gaussian and lands tight.
    np.testing.assert_allclose(rec_mean, true_mean, atol=0.25)
    np.testing.assert_allclose(rec_var, true_var, rtol=0.2)

    # The reliability gate: HMC's own diagnostics certify the reference. Well-mixed
    # (R-hat ≈ 1), healthy ESS (the curved ridge autocorrelates, so ESS/draw is below a
    # Gaussian's but still ample), and -- because target_accept is tight enough for the
    # tip's curvature -- exactly zero divergent transitions.
    rhat = alg.compute_rhat()
    bulk_ess, _tail = alg.compute_ess()
    assert rhat is not None and np.nanmax(rhat) < 1.05
    assert np.nanmin(bulk_ess) > 350
    assert sum(alg.divergences) == 0, 'clean reference must be divergence-free'


@pytest.mark.parametrize('var_type,loc,scale,true_mean,true_var', [
    # logistic: mean = loc, var = pi^2 s^2 / 3
    ('logistic_var', 2.0, 0.7, 2.0, np.pi ** 2 * 0.7 ** 2 / 3.0),
    # gumbel_r: mean = loc + euler_gamma * s, var = pi^2 s^2 / 6
    ('gumbel_var', -1.0, 0.9, -1.0 + np.euler_gamma * 0.9, np.pi ** 2 * 0.9 ** 2 / 6.0),
])
def test_hmc_recovers_informative_nonnormal_prior(tmp_path, var_type, loc, scale,
                                                  true_mean, true_var):
    """HMC composes the NEW per-family JAX prior densities into its target log-density
    (ADR-0059 item 4) and recovers a NON-normal prior's closed-form moments end to end.

    Construction of the oracle: a Gaussian target with a huge variance makes the NLL
    effectively flat over the prior's mass, so the posterior IS the prior --
    ``p(u) ∝ prior(u)`` -- and the recovered mean/variance must equal that prior's
    analytic moments (logistic: mean=loc, var=π²s²/3; gumbel: mean=loc+γs, var=π²s²/6).
    Both families are smooth, log-concave and real-support, so NUTS samples them
    divergence-free -- this exercises ``logpdf_jax`` through ``_build_logdensity`` and
    ``prior_logpdf_jax``, not just the unit oracle above. (Positive/bounded-support
    families would sample as densities here too, but HMC quality at their hard support
    edge awaits the item-5 bijection, so the recovery oracle uses real-support priors.)"""
    # 1-D near-flat likelihood (variance 1e6): the prior dominates the posterior.
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0], [1.0e6]))
    conf = H.make_config(tmp_path, 'hmc', tgt, exp, 1, var_type=var_type,
                         bounds=(loc, scale), population_size=4, num_warmup=1000,
                         num_samples=2500, max_iterations=2500, random_seed=20260627)
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 1)
    assert samples.shape[0] == conf.config['population_size'] * conf.config['num_samples']
    assert np.all(np.isfinite(samples))

    np.testing.assert_allclose(samples.mean(axis=0)[0], true_mean, atol=0.12)
    np.testing.assert_allclose(samples.var(axis=0, ddof=1)[0], true_var, rtol=0.15)

    # Smooth unimodal prior -> HMC's own reliability gate certifies the run: well-mixed
    # (R-hat ~ 1) with healthy ESS. A skewed tail (gumbel) can still trip a handful of
    # divergences at the default step size, so the gate here is a small divergent
    # FRACTION, not exactly zero (the strict zero-divergence certification is the
    # banana test's job, with its Stan-tight target_accept).
    rhat = alg.compute_rhat()
    bulk_ess, _tail = alg.compute_ess()
    assert rhat is not None and np.nanmax(rhat) < 1.05
    assert np.nanmin(bulk_ess) > 400
    assert sum(alg.divergences) < 0.01 * samples.shape[0]


def test_hmc_same_seed_reproduces_samples(tmp_path):
    """Reproducibility from the resolved seed — the per-chain Generator seeds the JAX PRNG,
    so the same ``random_seed`` writes byte-identical draws (the samplers' workflow guarantee)."""
    def run(sub):
        conf = _hmc_config(tmp_path, H.gaussian_spec([0.5, -0.5], [1.0, 2.0]), 2,
                           num_chains=2, num_warmup=300, num_samples=400,
                           output_dir=str(tmp_path / sub))
        alg = algorithms.HMCSampler(conf)
        H.drive(alg)
        return H.read_samples(conf.config['output_dir'], 2)

    a = run('repro_a')
    b = run('repro_b')
    assert a.size > 0
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Honesty: HMC's own diagnostics flag the geometry it cannot be trusted on
# --------------------------------------------------------------------------- #
def test_hmc_multimodal_is_flagged_by_its_own_rhat(tmp_path):
    """HMC NUTS follows the gradient into one basin and cannot hop the near-zero-density
    gap between well-separated modes, so it does NOT sample a multimodal posterior
    correctly. The point of the test is not that HMC fails (it must) but that HMC's *own*
    diagnostics catch the failure: independent chains park in different modes, so the
    rank-normalized cross-chain R-hat is large and rejects the run as unconverged. This is
    the ADR-0059 contract — HMC is the reference only *where its own R-hat/ESS/divergences
    pass*; here R-hat correctly refuses to certify it.

    Two well-separated equal-weight modes at ``±(4,4)`` (width ~0.7) make the inter-mode
    gap effectively zero-density, so the failure (and its detection) is unambiguous."""
    modes = [(0.5, [-4.0, -4.0], [0.5, 0.5]),
             (0.5, [4.0, 4.0], [0.5, 0.5])]
    conf = _hmc_config(tmp_path, H.multimodal_spec(modes), 2, num_chains=6,
                       num_warmup=600, num_samples=1200, bounds=(-12.0, 12.0))
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 2)
    assert np.all(np.isfinite(samples))

    # The failure is real: the chains split across the two modes (the *mechanism* behind
    # the high R-hat — it is mode-stranding, not noise). Each chain's center sits at one of
    # the two true modes ±(4,4), confirming within-a-single-mode sampling is itself fine.
    chain_x1_means = np.array([np.array(c)[:, 0].mean() for c in alg.chain_history])
    nearest_mode = np.where(chain_x1_means < 0.0, -4.0, 4.0)
    np.testing.assert_allclose(chain_x1_means, nearest_mode, atol=0.5)
    assert np.any(chain_x1_means < 0) and np.any(chain_x1_means > 0), \
        'both modes must be occupied for cross-chain R-hat to flag the stranding'

    # The honest verdict: HMC's own rank-normalized split-R-hat is large, so the run is
    # correctly rejected as unconverged. (Divergences stay near zero here — each mode is a
    # smooth Gaussian; the failure is *between-mode* mixing, which R-hat, not the divergence
    # count, is the diagnostic for. The reliability gate is the conjunction of the three.)
    rhat = alg.compute_rhat()
    assert rhat is not None and np.nanmax(rhat) > 1.3, \
        'R-hat must flag the multimodal stranding as unconverged'


# --------------------------------------------------------------------------- #
# Cross-check: HMC is the reference oracle the gradient-free samplers are scored against
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_hmc_reference_agrees_with_dream_on_banana(tmp_path, monkeypatch):
    """ADR-0059's stated purpose, end to end: HMC (the reference) and a gradient-free
    sampler (DREAM) run on the *same* banana and must both recover the closed-form moments,
    HMC being the yardstick DREAM is scored against.

    The reference is trusted only because its own reliability gate passes on this geometry —
    R-hat ≈ 1, healthy ESS, and zero divergent transitions — so "HMC agrees with the truth"
    is a *certified* statement, not an assumption. Against that certified reference, DREAM's
    differential evolution negotiates the curved ridge to MC tolerance. (Adaptive_MCMC, by
    contrast, mixes poorly on the banana at any tractable in-process budget — precisely the
    kind of gap this comparison exists to expose; DREAM is the gradient-free method that
    passes here.)

    This is a slow-tier test: full moment recovery for the gradient-free sampler is
    inherently many evaluations (mirrors ``test_sampler_integration``'s slow recovery tier).
    ``H.install`` patches the FakeClient dask layer that DREAM's per-pset dispatch needs;
    HMC ignores the client and runs NUTS in process, so the same patch is a harmless no-op
    for it."""
    H.install(monkeypatch)
    a, b = 1.0, 8.0
    true_mean, true_var = _banana_moments(a, b)
    spec = H.banana_spec(a=a, b=b)
    bounds = (-25.0, 25.0)
    tgt, exp = H.write_target(tmp_path, spec)

    # The reference: blackjax NUTS, certified on its own diagnostics before it is trusted.
    hmc_conf = H.make_config(
        tmp_path, 'hmc', tgt, exp, 2, bounds=bounds, population_size=4, num_warmup=1200,
        num_samples=2500, target_accept=0.95, max_iterations=2500, random_seed=20260627,
        output_dir=str(tmp_path / 'hmc'))
    hmc = algorithms.HMCSampler(hmc_conf)
    H.drive(hmc)
    hmc_s = H.read_samples(hmc_conf.config['output_dir'], 2)
    assert np.nanmax(hmc.compute_rhat()) < 1.05      # the gate: reference is trustworthy here
    assert sum(hmc.divergences) == 0
    np.testing.assert_allclose(hmc_s.mean(axis=0), true_mean, atol=0.25)
    np.testing.assert_allclose(hmc_s.var(axis=0, ddof=1), true_var, rtol=0.2)

    # The gradient-free sampler scored against that reference, on the same banana. A wide box
    # gives DREAM's differential-evolution proposals room to negotiate the curved ridge (a
    # tighter box actually mixes worse here); a single final diagnostics pass
    # (diagnostics_every = max_iterations) keeps the rank-normalization off the run's hot
    # loop -- the in-process recovery is already O(pop * max_iterations).
    dream_conf = H.make_config(
        tmp_path, 'dream', tgt, exp, 2, bounds=bounds, population_size=5, burn_in=350,
        max_iterations=1000, sample_every=2, rhat_threshold=0, output_hist_every=10**9,
        hist_bins=20, diagnostics_every=1000, random_seed=20260627,
        output_dir=str(tmp_path / 'dream'))
    dream = algorithms.DreamAlgorithm(dream_conf)
    H.drive(dream)
    dream_s = H.read_samples(dream_conf.config['output_dir'], 2)
    # MC tolerance: the gradient-free recovery is noisier than the certified reference (it
    # samples the heavy right tail of x1² less completely, so the x2 moments come in a touch
    # low), so the closed-form check is looser here than HMC's, but still pins both moments.
    np.testing.assert_allclose(dream_s.mean(axis=0), true_mean, atol=0.3)
    np.testing.assert_allclose(dream_s.var(axis=0, ddof=1), true_var, rtol=0.3)

    # ...and the two samplers agree with *each other* (HMC the reference), the comparison the
    # ADR is after -- not merely that each independently lands near the truth.
    np.testing.assert_allclose(dream_s.mean(axis=0), hmc_s.mean(axis=0), atol=0.3)


# --------------------------------------------------------------------------- #
# Pointed errors at the slice boundaries (fail clearly, never silently)
# --------------------------------------------------------------------------- #
def test_hmc_unsupported_target_raises_pointed_error(tmp_path):
    """A target with no JAX NLL yet (rotated_quartic) errors clearly, naming the supported
    set — not a silent wrong answer or a bare AttributeError (ADR-0059 deferred-work
    boundary). banana / multimodal moved *into* the supported set this slice, so the
    boundary check now stands on rotated_quartic, which is still a later slice."""
    from pybnf.printing import PybnfError
    spec = H.rotated_quartic_spec([0.0, 0.0], angle=np.pi / 6, coeff=[0.01, 1.0])
    conf = _hmc_config(tmp_path, spec, 2, num_chains=1, num_warmup=20, num_samples=20)
    alg = algorithms.HMCSampler(conf)
    with pytest.raises(PybnfError, match='rotated_quartic'):
        H.drive(alg)


# --------------------------------------------------------------------------- #
# Item 5: the unconstraining bijection + log-scale -- constrained-support and
# log-scaled priors now SAMPLE divergence-free (they could not before this slice)
# --------------------------------------------------------------------------- #
def _recover_prior(tmp_path, var_type, bounds, *, n_params=1, target_accept=0.9,
                   num_warmup=1000, num_samples=2500, num_chains=4, **overrides):
    """Run HMC on a near-flat (variance 1e6) Gaussian likelihood so the posterior IS the
    prior, and return ``(samples, alg)``.

    The 1-D Gaussian NLL is effectively constant over the prior's mass, so ``p(u) ~ prior(u)``
    and the recovered draws must match the prior's analytic moments -- the item-4 oracle
    pattern, now exercising the item-5 bijection (a positive/bounded/truncated/log-scaled prior
    that NUTS could not sample before)."""
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0], [1.0e6]))
    conf = H.make_config(tmp_path, 'hmc', tgt, exp, n_params, var_type=var_type, bounds=bounds,
                         population_size=num_chains, num_warmup=num_warmup,
                         num_samples=num_samples, max_iterations=num_samples,
                         target_accept=target_accept, random_seed=20260627, **overrides)
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)
    samples = H.read_samples(conf.config['output_dir'], n_params)
    assert samples.shape[0] == num_chains * num_samples
    assert np.all(np.isfinite(samples))
    return samples, alg


def _assert_clean_reference(alg, samples, *, min_ess=350):
    """HMC's own reliability gate: well-mixed (R-hat < 1.05), healthy ESS, and -- the whole
    point of the bijection -- exactly zero divergent transitions (the constrained-support
    families diverged at the -inf wall before item 5)."""
    rhat = alg.compute_rhat()
    bulk_ess, _tail = alg.compute_ess()
    assert rhat is not None and np.nanmax(rhat) < 1.05
    assert np.nanmin(bulk_ess) > min_ess
    assert sum(alg.divergences) == 0, 'the bijection must make the constrained prior divergence-free'


@pytest.mark.parametrize('var_type,bounds,true_mean,true_var', [
    # half_normal(scale s): mean = s*sqrt(2/pi), var = s^2 (1 - 2/pi). Support (0, inf) ->
    # log bijection u = exp(z). (p2 is ignored by the one-parameter family.)
    ('half_normal_var', (2.0, 2.0), 2.0 * np.sqrt(2.0 / np.pi), 4.0 * (1.0 - 2.0 / np.pi)),
    # gamma(shape k, scale theta): mean = k*theta, var = k*theta^2. Support (0, inf) -> log
    # bijection. A genuine right-skew the wall used to make NUTS diverge on.
    ('gamma_var', (2.5, 1.3), 2.5 * 1.3, 2.5 * 1.3 ** 2),
])
def test_hmc_recovers_positive_support_prior(tmp_path, var_type, bounds, true_mean, true_var):
    """A positive-support prior (gamma / half_normal) now samples divergence-free: the log
    bijection u = exp(z) puts the u=0 wall out of reach, so NUTS recovers the closed-form
    moments instead of diverging at the support edge (ADR-0059 item 5)."""
    samples, alg = _recover_prior(tmp_path, var_type, bounds)
    np.testing.assert_allclose(samples.mean(axis=0)[0], true_mean, rtol=0.06)
    np.testing.assert_allclose(samples.var(axis=0, ddof=1)[0], true_var, rtol=0.12)
    _assert_clean_reference(alg, samples)


def test_hmc_recovers_beta_prior(tmp_path):
    """A bounded [0, 1] prior (beta) now samples divergence-free via the logit bijection
    u = sigmoid(z): NUTS recovers the closed-form beta moments mean = a/(a+b),
    var = ab/((a+b)^2 (a+b+1)) without diverging at either wall (ADR-0059 item 5)."""
    a, b = 2.0, 3.0
    true_mean = a / (a + b)
    true_var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    samples, alg = _recover_prior(tmp_path, 'beta_var', (a, b))
    np.testing.assert_allclose(samples.mean(axis=0)[0], true_mean, atol=0.02)
    np.testing.assert_allclose(samples.var(axis=0, ddof=1)[0], true_var, rtol=0.12)
    _assert_clean_reference(alg, samples)


def test_hmc_samples_lognormal_log_scaled_param(tmp_path):
    """A log-scaled parameter now SAMPLES instead of raising (this slice removes the
    'log-scaled' error): lognormal_var places a normal prior on u = log10(theta) (real
    support -> identity bijection) and evaluates the likelihood at theta = 10**u through the
    JAX-traceable Scale.inverse_jax. The recovered log10(theta) must match the prior's
    normal moments, and -- no constrained support -- it is divergence-free (ADR-0059 item 5,
    half A)."""
    mu, sigma = 0.5, 0.3
    samples, alg = _recover_prior(tmp_path, 'lognormal_var', (mu, sigma))
    log10_theta = np.log10(samples[:, 0])
    np.testing.assert_allclose(log10_theta.mean(), mu, atol=0.02)
    np.testing.assert_allclose(log10_theta.var(ddof=1), sigma ** 2, rtol=0.12)
    _assert_clean_reference(alg, samples, min_ess=400)


def test_hmc_samples_loguniform_tight_box(tmp_path):
    """A log-scaled BOX prior (loguniform_var) over a TIGHT range now samples divergence-free:
    the box bijection u = lo + (hi-lo) sigmoid(z) operates in log10 space and theta = 10**u,
    composing the item-5 bijection with the log scale. log10(theta) must recover the uniform
    box's moments (mean = midpoint, var = width^2/12) with no divergences -- the tight box is
    the case the item-4 '-inf walls, mass far inside' caveat could not handle."""
    lo, hi = 0.5, 5.0                       # theta box; log10 box is (log10 lo, log10 hi)
    llo, lhi = np.log10(lo), np.log10(hi)
    samples, alg = _recover_prior(tmp_path, 'loguniform_var', (lo, hi))
    log10_theta = np.log10(samples[:, 0])
    np.testing.assert_allclose(log10_theta.mean(), 0.5 * (llo + lhi), atol=0.03)
    np.testing.assert_allclose(log10_theta.var(ddof=1), (lhi - llo) ** 2 / 12.0, rtol=0.12)
    # Every draw is strictly inside the box (the bijection cannot leave it).
    assert np.all((samples[:, 0] > lo) & (samples[:, 0] < hi))
    _assert_clean_reference(alg, samples, min_ess=400)


def test_hmc_recovers_tight_truncated_normal(tmp_path):
    """A TIGHT truncated normal -- N(0,1) confined to [1.5, 2.5], mass piled against the lower
    wall -- now samples divergence-free through the box bijection on the TruncatedPrior's
    support (ADR-0059 item 5). This is the case the item-4 caveat flagged: the retained mass
    leans on a bound, so the old -inf wall made NUTS diverge; the bijection puts the wall out
    of reach and NUTS recovers the truncated moments. Declared via the new-era parameter:
    record (the only surface that carries truncation bounds, ADR-0043/0020)."""
    from scipy import stats
    a, b = 1.5, 2.5
    truth = stats.truncnorm(a=a, b=b, loc=0.0, scale=1.0)   # standardized bounds == theta here
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0], [1.0e6]))
    # edition 2 is required for the parameter: record, so the run selector and objective use
    # the modern keys (job_type / objective = score); the internal models/exp_data form is the
    # harness's already-resolved shape (not edition-gated).
    base = {
        'output_dir': str(tmp_path) + '/out',
        'models': {tgt}, tgt: [exp], 'exp_data': {exp},
        'objective': 'score', 'job_type': 'hmc', 'initialization': 'lh',
        'delete_old_files': 1, 'verbosity': 0, 'wall_time_sim': 0, 'random_seed': 20260627,
        'edition': 2, 'population_size': 4, 'num_warmup': 1000, 'num_samples': 2500,
        'max_iterations': 2500, 'target_accept': 0.9,
        ('parameter', 'p1'): {'prior': 'normal', 'mean': '0', 'sd': '1',
                              'lower': str(a), 'upper': str(b)},
    }
    conf = H.config.Configuration(base)
    alg = algorithms.HMCSampler(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 1)
    assert np.all(np.isfinite(samples)) and np.all((samples[:, 0] > a) & (samples[:, 0] < b))
    np.testing.assert_allclose(samples.mean(axis=0)[0], truth.mean(), atol=0.03)
    np.testing.assert_allclose(samples.var(axis=0, ddof=1)[0], truth.var(), rtol=0.15)
    _assert_clean_reference(alg, samples, min_ess=400)


# --------------------------------------------------------------------------- #
# Bijection unit oracles: round-trip, Jacobian == finite difference, and a finite
# composed-target gradient exactly where the item-4 -inf wall used to be
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('lo,hi,u0', [
    (-np.inf, np.inf, 0.7),     # identity
    (0.0, np.inf, 1.3),         # lower-bounded (positive families)
    (-np.inf, 3.0, 1.1),        # upper-bounded
    (-2.0, 5.0, 1.9),           # finite box (uniform / loguniform / beta / truncated)
], ids=['identity', 'lower', 'upper', 'box'])
def test_bijector_round_trip_and_logdet(lo, hi, u0):
    """Each support-aware bijector is a clean inverse pair (``b(b^{-1}(u)) == u``) and its
    analytic ``log|b'(z)|`` matches a central finite difference of the constrained map -- the
    Jacobian the HMC target adds. The numpy and JAX log-determinants agree (float32 tol)."""
    import jax
    import jax.numpy as jnp
    from pybnf.priors.bijector import bijector_for_support

    b = bijector_for_support(lo, hi)
    z = b.to_unconstrained(u0)
    assert b.to_constrained(z) == pytest.approx(u0, abs=1e-6)        # round-trip

    h = 1e-4
    fd = np.log(abs((b.to_constrained(z + h) - b.to_constrained(z - h)) / (2 * h)))
    assert b.logdet(z) == pytest.approx(fd, abs=1e-3)               # logdet == |db/dz|
    assert float(b.logdet_jax(jnp.asarray(z))) == pytest.approx(b.logdet(z), abs=1e-4)
    assert float(b.to_constrained_jax(jnp.asarray(z))) == pytest.approx(u0, abs=1e-5)
    # The constrained map is differentiable everywhere (no NaN), so jax.grad of the target
    # composes.
    g = float(jax.grad(lambda zz: b.to_constrained_jax(zz))(jnp.asarray(z)))
    assert np.isfinite(g)


def test_bijection_removes_the_item4_support_wall(tmp_path):
    """The load-bearing item-5 guarantee: composing a constrained prior's ``logpdf_jax`` with
    its bijector gives a target whose ``jax.grad`` is FINITE for every finite z -- including the
    z that map u right onto the old support edge, where item 4's ``-inf`` wall produced the
    divergences this slice removes.

    Checks both a positive family (gamma, lower wall at u=0) and a tight TruncatedPrior (both
    walls): ``b(z)`` lands strictly inside the open support for all finite z, so
    ``prior.logpdf_jax(b(z))`` never reaches ``-inf`` and the gradient stays finite where the
    bare density's did not."""
    import jax
    import jax.numpy as jnp
    from pybnf.priors.bijector import bijector_for_support
    from pybnf.priors.gamma import Gamma
    from pybnf.priors.normal import Normal
    from pybnf.priors.truncated import TruncatedPrior

    for prior in (Gamma(shape=2.5, gamma_scale=1.3),
                  TruncatedPrior(Normal(0.0, 1.0), 1.5, 2.5)):
        lo, hi = prior.support()
        b = bijector_for_support(lo, hi)

        def target(z, _b=b, _p=prior):
            return _p.logpdf_jax(_b.to_constrained_jax(z)) + _b.logdet_jax(z)

        grad = jax.grad(target)
        # z spanning deep into both tails: large |z| drives u arbitrarily close to a support
        # edge (the item-4 wall) -- the composed gradient must stay finite throughout.
        for z in np.linspace(-12.0, 12.0, 41):
            val = float(target(jnp.asarray(z)))
            g = float(grad(jnp.asarray(z)))
            assert np.isfinite(val) and np.isfinite(g), \
                f'{type(prior).__name__}: non-finite target/grad at z={z}'


# --------------------------------------------------------------------------- #
# Prior logpdf_jax matches the scipy logpdf (the sampler-of-record oracle)
# --------------------------------------------------------------------------- #
def test_normal_logpdf_jax_matches_scipy():
    from pybnf.priors.normal import Normal
    p = Normal(loc=0.7, sigma=1.3)
    us = np.linspace(-4.0, 4.0, 25)
    got = np.array([float(p.logpdf_jax(float(u))) for u in us])
    want = np.array([p.logpdf(u) for u in us])
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_uniform_logpdf_jax_matches_scipy_inside_and_walls():
    from pybnf.priors.uniform import Uniform
    p = Uniform(lo=-2.0, hi=3.0)
    inside = np.linspace(-1.9, 2.9, 15)
    got = np.array([float(p.logpdf_jax(float(u))) for u in inside])
    want = np.array([p.logpdf(u) for u in inside])
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    # Outside the box the JAX density is -inf, matching scipy's out-of-support logpdf.
    assert float(p.logpdf_jax(5.0)) == -np.inf
    assert float(p.logpdf_jax(-5.0)) == -np.inf


def _all_prior_family_instances():
    """One concrete instance of every edition-2 prior family (ADR-0059 item 4).

    Construction is scipy-only (no jax), so this is safe to evaluate at parametrize
    collection time even when the pybnf[jax] extra is absent; the jax import lives in
    the test body, under the module skip."""
    from pybnf.priors.beta import Beta
    from pybnf.priors.cauchy import Cauchy
    from pybnf.priors.chisquare import ChiSquare
    from pybnf.priors.exponential import Exponential
    from pybnf.priors.gamma import Gamma
    from pybnf.priors.gumbel import Gumbel
    from pybnf.priors.half_cauchy import HalfCauchy
    from pybnf.priors.half_normal import HalfNormal
    from pybnf.priors.inv_gamma import InvGamma
    from pybnf.priors.laplace import Laplace
    from pybnf.priors.logistic import Logistic
    from pybnf.priors.normal import Normal
    from pybnf.priors.rayleigh import Rayleigh
    from pybnf.priors.student_t import StudentT
    from pybnf.priors.uniform import Uniform
    from pybnf.priors.weibull import Weibull
    return [
        Normal(loc=0.7, sigma=1.3),
        Uniform(lo=-2.0, hi=3.0),
        Laplace(loc=0.5, b=1.2),
        Cauchy(loc=-0.3, scale=0.8),
        Gamma(shape=2.5, gamma_scale=1.3),
        Exponential(exp_scale=0.7),
        ChiSquare(dof=3.0),
        Rayleigh(ray_scale=1.1),
        StudentT(df=4.0, loc=0.2, t_scale=1.5),
        HalfNormal(hn_scale=2.0),
        HalfCauchy(hc_scale=1.5),
        Beta(alpha=2.0, beta=3.0),
        InvGamma(shape=3.0, ig_scale=2.0),
        Weibull(shape=1.7, wb_scale=1.4),
        Gumbel(loc=0.3, scale=0.9),
        Logistic(loc=-0.2, scale=1.1),
    ]


@pytest.mark.parametrize('prior', _all_prior_family_instances(),
                         ids=lambda p: type(p).__name__)
def test_family_logpdf_jax_matches_scipy_and_grad_is_finite(prior):
    """Every edition-2 family's hand-written ``logpdf_jax`` is oracle-equal to its
    scipy ``logpdf`` (the sampler-of-record), returns ``-inf`` past any finite support
    edge, and has a FINITE ``jax.grad`` both inside and outside the support (ADR-0059
    item 4).

    The grad-outside check is the load-bearing one: a naive ``where(inside, real, -inf)``
    over a bounded/half-line support gives a NaN gradient outside it (the jax where-grad
    rule taints the masked branch), which derails NUTS leapfrog trajectories that step
    out of support. The families guard it with the safe-``u`` double-``where``; this test
    is what proves the guard holds for all of them, not just the two hand-checked ones."""
    import jax
    frozen = prior.frozen
    lo, hi = frozen.support()

    # In-support points spanning the bulk (scipy ppf of evenly-spaced quantiles).
    inside = frozen.ppf(np.linspace(0.05, 0.95, 11))
    got = np.array([float(prior.logpdf_jax(float(u))) for u in inside])
    want = np.array([float(prior.logpdf(float(u))) for u in inside])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    # Just past each FINITE support edge the JAX density is -inf (matches scipy).
    outside = []
    if np.isfinite(lo):
        outside.append(float(lo) - 1.0)
    if np.isfinite(hi):
        outside.append(float(hi) + 1.0)
    for u in outside:
        assert float(prior.logpdf_jax(float(u))) == -np.inf

    grad = jax.grad(lambda x: prior.logpdf_jax(x))
    for u in list(inside) + outside:
        assert np.isfinite(float(grad(float(u)))), \
            f'{type(prior).__name__} has a non-finite grad at u={u}'


# --------------------------------------------------------------------------- #
# JAX NLL == numpy NLL (the model's "one source of truth", ADR-0059 item 2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('spec', [
    H.gaussian_spec([2.0, -1.0], [1.0, 4.0]),
    H.rotated_gaussian_spec([0.0, 0.0], H.rotated_cov([2.0, 0.5], angle=np.pi / 6)),
    H.banana_spec(a=1.0, b=8.0),
    H.banana_spec(a=0.5, b=20.0),
    H.multimodal_spec([(0.5, [-4.0, -4.0], [0.5, 0.5]), (0.5, [4.0, 4.0], [1.0, 2.0])]),
    H.multimodal_spec([(0.3, [0.0, 0.0], [1.0, 1.0]), (0.7, [3.0, -2.0], [0.5, 2.0])]),
])
def test_nll_jax_matches_numpy_nll(tmp_path, spec):
    """Every supported HMC target's JAX NLL must equal its numpy ``_compute_nll`` peer at
    arbitrary points — the score path and the differentiable HMC log-density are one closed
    form, not two hand-maintained copies (ADR-0059's "one source of truth"). This is the NLL
    analog of the prior ``logpdf_jax``↔scipy oracle check above: it pins ``nll_jax`` to the
    sampler-of-record numpy form so the JAX branch cannot silently drift (e.g. a dropped
    ``logsumexp`` shift, a mis-paired banana index)."""
    import jax.numpy as jnp
    from pybnf.analytical_model import AnalyticalModel
    tgt, _ = H.write_target(tmp_path, spec)
    model = AnalyticalModel(tgt)
    nll = model.nll_jax()
    rng = np.random.default_rng(0)
    for _ in range(12):
        x = rng.normal(size=2) * 3.0
        got = float(nll(jnp.asarray(x)))
        want = float(model._compute_nll(x))
        assert got == pytest.approx(want, rel=1e-5, abs=1e-6)
