"""
Tests for probability distribution sampling and prior calculations.

These tests lock in the current behavior of:
1. FreeParameter sampling distributions (normal, lognormal, uniform, loguniform)
2. BayesianAlgorithm.ln_prior() calculations through scipy.stats distribution objects

These must survive the migration to scipy.stats (issue #5).
"""

import numpy as np
from scipy import stats

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pybnf.pset import FreeParameter, PSet
from pybnf.algorithms import BayesianAlgorithm


# ---------------------------------------------------------------------------
# Sampling distribution tests
# ---------------------------------------------------------------------------

class TestNormalVarSampling:
    """Tests for normal_var distribution sampling."""

    def setup_method(self):
        self.p = FreeParameter('x__FREE', 'normal_var', 5.0, 2.0)

    def test_sample_returns_freeparameter(self):
        s = self.p.sample_value()
        assert isinstance(s, FreeParameter)
        assert s.value is not None

    def test_sample_mean(self):
        """Samples should have mean close to p1."""
        vals = [self.p.sample_value().value for _ in range(50000)]
        assert abs(np.mean(vals) - 5.0) < 0.1

    def test_sample_std(self):
        """Samples should have std close to p2."""
        vals = [self.p.sample_value().value for _ in range(50000)]
        assert abs(np.std(vals) - 2.0) < 0.1

    def test_sample_can_be_negative(self):
        """Normal distribution centered at 0 should produce negative values."""
        p = FreeParameter('x__FREE', 'normal_var', 0, 1)
        vals = [p.sample_value().value for _ in range(1000)]
        assert any(v < 0 for v in vals)


class TestLognormalVarSampling:
    """Tests for lognormal_var distribution sampling.

    lognormal_var(p1, p2) samples in log10-space as normal(p1, p2),
    then transforms to real space as 10^sample.
    """

    def setup_method(self):
        self.p = FreeParameter('x__FREE', 'lognormal_var', 2.0, 0.5)

    def test_sample_positive(self):
        """All lognormal samples should be positive (10^x > 0)."""
        vals = [self.p.sample_value().value for _ in range(1000)]
        assert all(v > 0 for v in vals)

    def test_log_sample_mean(self):
        """log10 of samples should have mean close to p1."""
        vals = [np.log10(self.p.sample_value().value) for _ in range(50000)]
        assert abs(np.mean(vals) - 2.0) < 0.1

    def test_log_sample_std(self):
        """log10 of samples should have std close to p2."""
        vals = [np.log10(self.p.sample_value().value) for _ in range(50000)]
        assert abs(np.std(vals) - 0.5) < 0.05


class TestUniformVarSampling:
    """Tests for uniform_var distribution sampling."""

    def setup_method(self):
        self.p = FreeParameter('x__FREE', 'uniform_var', 2.0, 8.0)

    def test_sample_within_bounds(self):
        """All uniform samples should be within [p1, p2]."""
        vals = [self.p.sample_value().value for _ in range(10000)]
        assert all(2.0 <= v <= 8.0 for v in vals)

    def test_sample_mean(self):
        """Uniform samples should have mean close to (p1+p2)/2."""
        vals = [self.p.sample_value().value for _ in range(50000)]
        assert abs(np.mean(vals) - 5.0) < 0.1

    def test_sample_covers_range(self):
        """Samples should cover the full range."""
        vals = [self.p.sample_value().value for _ in range(10000)]
        assert min(vals) < 3.0
        assert max(vals) > 7.0


class TestLoguniformVarSampling:
    """Tests for loguniform_var distribution sampling.

    loguniform_var(p1, p2) samples uniformly in log10-space between
    log10(p1) and log10(p2), then transforms to real space as 10^sample.
    """

    def setup_method(self):
        self.p = FreeParameter('x__FREE', 'loguniform_var', 0.01, 100.0)

    def test_sample_within_bounds(self):
        """All loguniform samples should be within [p1, p2]."""
        vals = [self.p.sample_value().value for _ in range(10000)]
        assert all(0.01 <= v <= 100.0 for v in vals)

    def test_log_sample_mean(self):
        """log10 of samples should have mean close to (log10(p1)+log10(p2))/2."""
        vals = [np.log10(self.p.sample_value().value) for _ in range(50000)]
        expected_mean = (np.log10(0.01) + np.log10(100.0)) / 2.0  # = 0.0
        assert abs(np.mean(vals) - expected_mean) < 0.05

    def test_log_sample_uniform(self):
        """log10 of samples should be approximately uniform."""
        vals = [np.log10(self.p.sample_value().value) for _ in range(50000)]
        # Check that the distribution is roughly flat by comparing quartiles
        q25, q75 = np.percentile(vals, [25, 75])
        expected_q25 = np.log10(0.01) + 0.25 * (np.log10(100.0) - np.log10(0.01))  # = -1.0
        expected_q75 = np.log10(0.01) + 0.75 * (np.log10(100.0) - np.log10(0.01))  # = 1.0
        assert abs(q25 - expected_q25) < 0.1
        assert abs(q75 - expected_q75) < 0.1


# ---------------------------------------------------------------------------
# Prior (log-PDF) calculation tests
# ---------------------------------------------------------------------------

class TestNormalPrior:
    """Tests for normal prior log-PDF calculation.

    The implementation uses scipy.stats.norm.logpdf(), including the
    normalization constant.
    """

    def test_at_mean(self):
        """ln_prior should match scipy.stats at the mean."""
        prior = {'x__FREE': ('reg', 'n', 5.0, 2.0)}
        pset = _make_pset({'x__FREE': 5.0})
        result = _ln_prior(prior, pset)
        assert abs(result - stats.norm.logpdf(5.0, 5.0, 2.0)) < 1e-10

    def test_one_sigma_away(self):
        """ln_prior one sigma from mean should match scipy.stats."""
        prior = {'x__FREE': ('reg', 'n', 5.0, 2.0)}
        pset = _make_pset({'x__FREE': 7.0})  # 1 sigma away
        result = _ln_prior(prior, pset)
        assert abs(result - stats.norm.logpdf(7.0, 5.0, 2.0)) < 1e-10

    def test_two_sigma_away(self):
        """ln_prior two sigma from mean should match scipy.stats."""
        prior = {'x__FREE': ('reg', 'n', 5.0, 2.0)}
        pset = _make_pset({'x__FREE': 9.0})  # 2 sigma away
        result = _ln_prior(prior, pset)
        assert abs(result - stats.norm.logpdf(9.0, 5.0, 2.0)) < 1e-10

    def test_symmetric(self):
        """ln_prior should be symmetric around the mean."""
        prior = {'x__FREE': ('reg', 'n', 5.0, 2.0)}
        pset_above = _make_pset({'x__FREE': 7.0})
        pset_below = _make_pset({'x__FREE': 3.0})
        assert abs(_ln_prior(prior, pset_above) - _ln_prior(prior, pset_below)) < 1e-10

    def test_multiple_params(self):
        """ln_prior with multiple parameters should sum contributions."""
        prior = {
            'x__FREE': ('reg', 'n', 0.0, 1.0),
            'y__FREE': ('reg', 'n', 0.0, 1.0),
        }
        pset = _make_pset({'x__FREE': 1.0, 'y__FREE': 1.0})
        result = _ln_prior(prior, pset)
        expected = stats.norm.logpdf(1.0, 0.0, 1.0) + stats.norm.logpdf(1.0, 0.0, 1.0)
        assert abs(result - expected) < 1e-10

    def test_narrow_vs_wide(self):
        """Narrower prior should penalize a large deviation more."""
        prior_narrow = {'x__FREE': ('reg', 'n', 0.0, 0.5)}
        prior_wide = {'x__FREE': ('reg', 'n', 0.0, 5.0)}
        pset = _make_pset({'x__FREE': 10.0})
        assert _ln_prior(prior_narrow, pset) < _ln_prior(prior_wide, pset)


class TestLognormalPrior:
    """Tests for lognormal prior log-PDF calculation.

    For lognormal, the prior is computed in log10-space:
        ln_prior = scipy.stats.norm(mu, sigma).logpdf(log10(val))
    """

    def test_at_mean(self):
        """ln_prior should match scipy.stats when log10(val) == mu."""
        prior = {'x__FREE': ('log', 'n', 2.0, 0.5)}
        pset = _make_pset({'x__FREE': 100.0})  # log10(100) = 2.0
        result = _ln_prior(prior, pset)
        assert abs(result - stats.norm.logpdf(2.0, 2.0, 0.5)) < 1e-10

    def test_one_sigma_away(self):
        """ln_prior one sigma from mean in log-space should match scipy.stats."""
        prior = {'x__FREE': ('log', 'n', 2.0, 0.5)}
        pset = _make_pset({'x__FREE': 10**2.5})  # 1 sigma away in log space
        result = _ln_prior(prior, pset)
        assert abs(result - stats.norm.logpdf(2.5, 2.0, 0.5)) < 1e-10

    def test_nonpositive_value(self):
        """Log-space priors assign zero density to nonpositive regular-space values."""
        prior = {'x__FREE': ('log', 'n', 2.0, 0.5)}
        pset = _make_pset({'x__FREE': 0.0})
        assert _ln_prior(prior, pset) == -np.inf


class TestUniformPrior:
    """Tests for uniform (box) prior log-PDF calculation.

    The current implementation computes:
        ln_prior = -log(max - min)  if min <= val <= max
        ln_prior = -inf             otherwise
    """

    def test_inside_box(self):
        """ln_prior inside the box should be -log(width)."""
        prior = {'x__FREE': ('reg', 'b', 2.0, 8.0)}
        pset = _make_pset({'x__FREE': 5.0})
        result = _ln_prior(prior, pset)
        assert abs(result - (-np.log(6.0))) < 1e-10

    def test_constant_inside_box(self):
        """ln_prior should be the same everywhere inside the box."""
        prior = {'x__FREE': ('reg', 'b', 0.0, 10.0)}
        vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        results = [_ln_prior(prior, _make_pset({'x__FREE': v})) for v in vals]
        assert all(abs(r - results[0]) < 1e-10 for r in results)

    def test_outside_box(self):
        """ln_prior outside the box should be -inf."""
        prior = {'x__FREE': ('reg', 'b', 2.0, 8.0)}
        pset_below = _make_pset({'x__FREE': 1.0})
        pset_above = _make_pset({'x__FREE': 9.0})
        assert _ln_prior(prior, pset_below) == -np.inf
        assert _ln_prior(prior, pset_above) == -np.inf

    def test_at_boundary(self):
        """ln_prior at the boundary should be finite (inclusive bounds)."""
        prior = {'x__FREE': ('reg', 'b', 2.0, 8.0)}
        pset_low = _make_pset({'x__FREE': 2.0})
        pset_high = _make_pset({'x__FREE': 8.0})
        assert np.isfinite(_ln_prior(prior, pset_low))
        assert np.isfinite(_ln_prior(prior, pset_high))


class TestLoguniformPrior:
    """Tests for loguniform (log-space box) prior log-PDF calculation.

    For loguniform, the prior is a box in log10-space:
        ln_prior = -log(log10(max) - log10(min))  if log10(min) <= log10(val) <= log10(max)
        ln_prior = -inf                            otherwise
    """

    def test_inside_box(self):
        """ln_prior inside the box should be -log(log-width)."""
        prior = {'x__FREE': ('log', 'b', np.log10(0.01), np.log10(100.0))}  # log range = 4
        pset = _make_pset({'x__FREE': 1.0})  # log10(1) = 0, inside [-2, 2]
        result = _ln_prior(prior, pset)
        assert abs(result - (-np.log(4.0))) < 1e-10

    def test_outside_box(self):
        """ln_prior outside the box should be -inf."""
        prior = {'x__FREE': ('log', 'b', np.log10(0.01), np.log10(100.0))}
        pset = _make_pset({'x__FREE': 0.001})  # log10(0.001) = -3, outside [-2, 2]
        assert _ln_prior(prior, pset) == -np.inf


class TestPriorDifferenceForMCMC:
    """Tests that verify prior differences are correct for MCMC acceptance ratios.

    In MCMC, what matters is ln_prior(proposed) - ln_prior(current), so the
    omitted normalization constants must cancel. These tests verify this by
    comparing prior differences against scipy.stats reference values.
    """

    def test_normal_prior_difference_matches_scipy(self):
        """Prior difference should match scipy.stats.norm.logpdf difference."""
        mu, sigma = 5.0, 2.0
        prior = {'x__FREE': ('reg', 'n', mu, sigma)}
        val_a, val_b = 3.0, 7.0
        diff_ours = _ln_prior(prior, _make_pset({'x__FREE': val_b})) - _ln_prior(prior, _make_pset({'x__FREE': val_a}))
        diff_scipy = stats.norm.logpdf(val_b, mu, sigma) - stats.norm.logpdf(val_a, mu, sigma)
        assert abs(diff_ours - diff_scipy) < 1e-10

    def test_uniform_prior_difference_matches_scipy(self):
        """Prior difference inside the box should be 0 (matching scipy)."""
        lo, hi = 2.0, 8.0
        prior = {'x__FREE': ('reg', 'b', lo, hi)}
        val_a, val_b = 3.0, 6.0
        diff_ours = _ln_prior(prior, _make_pset({'x__FREE': val_b})) - _ln_prior(prior, _make_pset({'x__FREE': val_a}))
        diff_scipy = (stats.uniform.logpdf(val_b, lo, hi - lo)
                      - stats.uniform.logpdf(val_a, lo, hi - lo))
        assert abs(diff_ours - diff_scipy) < 1e-10


class TestBayesianPriorStorage:
    """Tests for the scipy-backed prior objects stored by BayesianAlgorithm."""

    def test_load_priors_stores_freeparameters(self):
        alg = object.__new__(BayesianAlgorithm)
        alg.variables = [
            FreeParameter('x__FREE', 'normal_var', 5.0, 2.0),
            FreeParameter('y__FREE', 'loguniform_var', 0.01, 100.0),
        ]

        BayesianAlgorithm.load_priors(alg)

        assert isinstance(alg.prior['x__FREE'], FreeParameter)
        assert isinstance(alg.prior['y__FREE'], FreeParameter)
        assert hasattr(alg.prior['x__FREE']._distribution, 'logpdf')
        assert hasattr(alg.prior['y__FREE']._distribution, 'rvs')

    def test_ln_prior_uses_freeparameter_logpdf(self):
        prior_var = FreeParameter('x__FREE', 'normal_var', 5.0, 2.0)
        prior = {'x__FREE': prior_var}
        pset = _make_pset({'x__FREE': 7.0})

        assert _ln_prior(prior, pset) == prior_var.prior_logpdf(7.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pset(param_dict):
    """Create a minimal PSet-like object for testing ln_prior.

    Keys in param_dict should match the keys used in the prior dict
    (i.e. include the __FREE suffix, matching FreeParameter.name).
    """
    params = []
    for name, val in param_dict.items():
        p = FreeParameter(name, 'uniform_var', -1e6, 1e6, value=val)
        params.append(p)
    return PSet(params)


def _ln_prior(prior, pset):
    """
    Evaluate BayesianAlgorithm.ln_prior() without constructing a full algorithm.

    Tests use the historical compact tuple notation for readability, then
    convert it to the FreeParameter objects now stored by load_priors().
    """
    alg = object.__new__(BayesianAlgorithm)
    alg.prior = {
        name: spec if isinstance(spec, FreeParameter) else _prior_param(name, spec)
        for name, spec in prior.items()
    }
    return BayesianAlgorithm.ln_prior(alg, pset)


def _prior_param(name, spec):
    """Convert a compact prior tuple into the FreeParameter used by ln_prior()."""
    space, dist, x1, x2 = spec
    if space == 'reg' and dist == 'n':
        return FreeParameter(name, 'normal_var', x1, x2)
    elif space == 'log' and dist == 'n':
        return FreeParameter(name, 'lognormal_var', x1, x2)
    elif space == 'reg' and dist == 'b':
        return FreeParameter(name, 'uniform_var', x1, x2)
    elif space == 'log' and dist == 'b':
        return FreeParameter(name, 'loguniform_var', 10**x1, 10**x2)
    raise ValueError('Unknown prior spec %r' % (spec,))
