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
# Pointed errors at the slice boundaries (fail clearly, never silently)
# --------------------------------------------------------------------------- #
def test_hmc_unsupported_target_raises_pointed_error(tmp_path):
    """A target with no JAX NLL yet (banana) errors clearly, naming the supported set —
    not a silent wrong answer or a bare AttributeError (ADR-0059 deferred-work boundary)."""
    from pybnf.printing import PybnfError
    conf = _hmc_config(tmp_path, H.banana_spec(), 2, num_chains=1,
                       num_warmup=20, num_samples=20)
    alg = algorithms.HMCSampler(conf)
    with pytest.raises(PybnfError, match='gaussian'):
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
