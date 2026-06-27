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


def test_hmc_log_scaled_param_raises_pointed_error(tmp_path):
    """A log-scaled parameter (loguniform_var) is out of this slice — it needs a
    JAX-traceable 10**u inverse + Jacobian — so HMC errors clearly rather than sampling a
    silently wrong target."""
    from pybnf.printing import PybnfError
    conf = _hmc_config(tmp_path, H.gaussian_spec([0.0], [1.0]), 1, num_chains=1,
                       num_warmup=20, num_samples=20,
                       var_type='loguniform_var', bounds=(0.01, 100.0))
    alg = algorithms.HMCSampler(conf)
    with pytest.raises(PybnfError, match='log-scaled'):
        H.drive(alg)


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
