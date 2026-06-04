"""Unit tests for the extracted ``pybnf.priors`` package (M2.3, Move 1a).

Two layers:

1. **Family/scale math vs scipy oracles** -- each family in the sampling space
   ``u``, plus the ``Scale`` transforms.
2. **Bit-exact equivalence to the current ``FreeParameter``** -- the composition
   ``prior.logpdf(scale.forward(theta))`` must equal today's
   ``FreeParameter.prior_logpdf(theta)``, and ``scale.inverse(prior.ppf(q))``
   must equal today's latin-hypercube rescale arithmetic. These oracle the
   behavior-preserving wiring done in Move 1b, before ``pset.py`` is touched.
"""

import numpy as np
import pytest
from scipy import stats

from .context import priors, pset

from pybnf.priors import LINEAR, LOG10, build_prior
from pybnf.priors.normal import Normal
from pybnf.priors.uniform import Uniform
from pybnf.priors.base import NoPrior
from pybnf.printing import PybnfError


# ---------------------------------------------------------------------------
# Scale transforms
# ---------------------------------------------------------------------------

class TestScale:
    def test_linear_is_identity(self):
        assert LINEAR.forward(3.7) == 3.7
        assert LINEAR.inverse(3.7) == 3.7
        assert not LINEAR.is_log

    def test_log10_forward_inverse(self):
        assert LOG10.forward(100.0) == 2.0
        assert LOG10.inverse(2.0) == 100.0
        assert LOG10.is_log

    def test_log10_inverse_matches_ten_to_the_power(self):
        # Must be 10.0 ** u, matching exp10/the inline 10** in proposal arithmetic.
        for u in (-3.0, -0.5, 0.0, 1.7, 4.2):
            assert LOG10.inverse(u) == 10.0 ** u

    @pytest.mark.parametrize("theta", [0.001, 0.5, 1.0, 42.0, 1e5])
    def test_log10_round_trip(self, theta):
        np.testing.assert_allclose(LOG10.inverse(LOG10.forward(theta)), theta, rtol=1e-12)


# ---------------------------------------------------------------------------
# Family math vs scipy
# ---------------------------------------------------------------------------

class TestNormalFamily:
    def test_logpdf_matches_scipy(self):
        p = Normal(loc=5.0, sigma=2.0)
        for u in (5.0, 7.0, 1.3):
            assert p.logpdf(u) == pytest.approx(stats.norm(5.0, 2.0).logpdf(u))

    def test_ppf_matches_scipy(self):
        p = Normal(loc=5.0, sigma=2.0)
        for q in (0.1, 0.5, 0.9):
            assert p.ppf(q) == pytest.approx(stats.norm(5.0, 2.0).ppf(q))

    def test_rvs_moments(self):
        np.random.seed(0)
        p = Normal(loc=5.0, sigma=2.0)
        xs = np.array([p.rvs() for _ in range(50000)])
        assert xs.mean() == pytest.approx(5.0, abs=0.05)
        assert xs.std() == pytest.approx(2.0, abs=0.05)

    def test_unbounded(self):
        p = Normal(loc=0.0, sigma=1.0)
        assert not p.has_bounded_support
        assert p.has_prior
        assert p.support() == (-np.inf, np.inf)


class TestUniformFamily:
    def test_logpdf_matches_scipy(self):
        p = Uniform(lo=2.0, hi=8.0)
        ref = stats.uniform(loc=2.0, scale=6.0)
        for u in (2.0, 5.0, 8.0):
            assert p.logpdf(u) == pytest.approx(ref.logpdf(u))
        assert p.logpdf(1.0) == -np.inf
        assert p.logpdf(9.0) == -np.inf

    def test_ppf_is_manual_linear_interpolation(self):
        # Must be lo + q*(hi-lo) to match the latin-hypercube rescale bit-for-bit.
        p = Uniform(lo=2.0, hi=8.0)
        for q in (0.0, 0.25, 0.5, 1.0):
            assert p.ppf(q) == 2.0 + q * (8.0 - 2.0)

    def test_bounded_support(self):
        p = Uniform(lo=2.0, hi=8.0)
        assert p.has_bounded_support
        assert p.support() == (2.0, 8.0)

    def test_build_transforms_bounds_into_scale(self):
        lin = Uniform.build(0.01, 100.0, LINEAR)
        assert lin.support() == (0.01, 100.0)
        log = Uniform.build(0.01, 100.0, LOG10)
        np.testing.assert_allclose(log.support(), (-2.0, 2.0), rtol=1e-12)


class TestNoPrior:
    def test_logpdf_zero(self):
        assert NoPrior().logpdf(123.0) == 0.0

    def test_rvs_and_ppf_raise(self):
        with pytest.raises(PybnfError):
            NoPrior().rvs()
        with pytest.raises(PybnfError):
            NoPrior().ppf(0.5)

    def test_flags(self):
        p = NoPrior()
        assert not p.has_prior
        assert not p.has_bounded_support
        assert p.frozen is None

    def test_build_ignores_values(self):
        assert isinstance(NoPrior.build(3.0, None, LOG10), NoPrior)


# ---------------------------------------------------------------------------
# Keyword map / build_prior resolution
# ---------------------------------------------------------------------------

class TestKeywordMap:
    @pytest.mark.parametrize("keyword,family,scale", [
        ('normal_var', Normal, LINEAR),
        ('lognormal_var', Normal, LOG10),
        ('uniform_var', Uniform, LINEAR),
        ('loguniform_var', Uniform, LOG10),
        ('var', NoPrior, LINEAR),
        ('logvar', NoPrior, LOG10),
    ])
    def test_map_entries(self, keyword, family, scale):
        fam, sc = priors.PRIOR_KEYWORD_MAP[keyword]
        assert fam is family
        assert sc is scale

    def test_build_prior_returns_prior_and_scale(self):
        prior, scale = build_prior('loguniform_var', 0.01, 100.0)
        assert isinstance(prior, Uniform)
        assert scale is LOG10
        np.testing.assert_allclose(prior.support(), (-2.0, 2.0), rtol=1e-12)

    def test_build_prior_unknown_keyword_is_noprior_linear(self):
        # Legacy back-compat: an unrecognised type (e.g. the test-only
        # 'random_var') is a linear, unbounded, no-prior value carrier --
        # matching _make_distribution's old None-for-unknown behavior.
        prior, scale = build_prior('random_var', 0.0, 1.0)
        assert isinstance(prior, NoPrior)
        assert scale is LINEAR


# ---------------------------------------------------------------------------
# Bit-exact equivalence to the current FreeParameter (oracles Move 1b)
# ---------------------------------------------------------------------------

# (keyword, p1, p2, sample values to probe the prior density at)
_CASES = [
    ('normal_var', 5.0, 2.0, [5.0, 7.0, 1.3, -4.0]),
    ('lognormal_var', 2.0, 0.5, [10.0, 100.0, 1.0, 0.01]),
    ('uniform_var', 2.0, 8.0, [2.0, 5.0, 8.0, 1.0, 9.0]),
    ('loguniform_var', 0.01, 100.0, [1.0, 0.01, 100.0, 0.001, 1000.0]),
]


@pytest.mark.parametrize("keyword,p1,p2,values", _CASES)
def test_prior_logpdf_matches_current_freeparameter(keyword, p1, p2, values):
    """prior.logpdf(scale.forward(theta)) == FreeParameter.prior_logpdf(theta)."""
    fp = pset.FreeParameter('x__FREE', keyword, p1, p2)
    prior, scale = build_prior(keyword, p1, p2)
    for v in values:
        expected = fp.prior_logpdf(v)
        if keyword in ('lognormal_var', 'loguniform_var') and v <= 0:
            got = -np.inf  # scale.forward(non-positive) is the FreeParameter guard's job (Move 1b)
        else:
            got = prior.logpdf(scale.forward(v))
        if np.isinf(expected):
            assert got == expected
        else:
            assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_noprior_logpdf_matches_current_freeparameter():
    fp = pset.FreeParameter('x__FREE', 'var', 3.0, None)
    prior, _ = build_prior('var', 3.0, None)
    assert prior.logpdf(42.0) == fp.prior_logpdf(42.0) == 0.0


@pytest.mark.parametrize("keyword,p1,p2", [
    ('uniform_var', 2.0, 8.0),
    ('loguniform_var', 0.01, 100.0),
])
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_lhc_quantile_matches_current_rescale(keyword, p1, p2, q):
    """scale.inverse(prior.ppf(q)) reproduces the existing LH rescale arithmetic."""
    prior, scale = build_prior(keyword, p1, p2)
    got = scale.inverse(prior.ppf(q))
    if keyword == 'uniform_var':
        expected = p1 + q * (p2 - p1)
    else:
        expected = 10.0 ** (np.log10(p1) + q * (np.log10(p2) - np.log10(p1)))
    assert got == expected
