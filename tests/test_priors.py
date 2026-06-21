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

from pybnf.priors import LINEAR, LOG10, TruncatedPrior, build_prior
from pybnf.priors.normal import Normal
from pybnf.priors.uniform import Uniform
from pybnf.priors.laplace import Laplace
from pybnf.priors.cauchy import Cauchy
from pybnf.priors.gamma import Gamma
from pybnf.priors.exponential import Exponential
from pybnf.priors.chisquare import ChiSquare
from pybnf.priors.rayleigh import Rayleigh
from pybnf.priors.base import NoPrior
from pybnf.printing import PybnfError


def _trapz(y, x):
    """Trapezoidal integral, version-agnostic (np.trapz is deprecated in numpy 2)."""
    y = np.asarray(y)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


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
        rng = np.random.default_rng(0)
        p = Normal(loc=5.0, sigma=2.0)
        xs = np.array([p.rvs(rng) for _ in range(50000)])
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


class TestCatalogFamilies:
    """The full v2 catalog families (#417), each oracled against its scipy distribution.

    The PEtab parameterizations are verified against petab's own ``v1.distributions``: gamma
    is shape+scale (not shape+rate), exponential's parameter is the scale (= 1/rate), and the
    one-parameter families ignore the unused ``p2``.
    """

    # (family, build-kwargs, scipy reference, sampling-space support)
    _FAMILIES = [
        (Cauchy, dict(loc=1.0, scale=2.0), stats.cauchy(loc=1.0, scale=2.0), (-np.inf, np.inf)),
        (Gamma, dict(shape=2.0, gamma_scale=3.0), stats.gamma(a=2.0, scale=3.0), (0.0, np.inf)),
        (Exponential, dict(exp_scale=0.5), stats.expon(scale=0.5), (0.0, np.inf)),
        (ChiSquare, dict(dof=4.0), stats.chi2(df=4.0), (0.0, np.inf)),
        (Rayleigh, dict(ray_scale=1.5), stats.rayleigh(scale=1.5), (0.0, np.inf)),
    ]

    @pytest.mark.parametrize("cls,kwargs,ref,support", _FAMILIES)
    def test_logpdf_ppf_match_scipy(self, cls, kwargs, ref, support):
        p = cls(**kwargs)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            u = float(ref.ppf(q))
            assert p.logpdf(u) == pytest.approx(float(ref.logpdf(u)))
            assert p.ppf(q) == pytest.approx(u)
        assert not p.has_bounded_support and p.has_prior
        np.testing.assert_allclose(p.support(), support)

    @pytest.mark.parametrize("cls,kwargs,ref,_support", _FAMILIES)
    def test_rvs_mean_matches_scipy(self, cls, kwargs, ref, _support):
        rng = np.random.default_rng(0)
        p = cls(**kwargs)
        xs = np.array([p.rvs(rng) for _ in range(40000)])
        # Cauchy has no finite mean; check the median instead (it has one).
        stat = np.median if cls is Cauchy else np.mean
        assert stat(xs) == pytest.approx(float(ref.median() if cls is Cauchy else ref.mean()),
                                         abs=0.1)

    @pytest.mark.parametrize("keyword,p1,p2,cls", [
        ('cauchy_var', 1.0, 2.0, Cauchy),
        ('gamma_var', 2.0, 3.0, Gamma),
        ('exponential_var', 0.5, None, Exponential),
        ('chisquare_var', 4.0, None, ChiSquare),
        ('rayleigh_var', 1.5, None, Rayleigh),
    ])
    def test_build_prior_resolves_keyword(self, keyword, p1, p2, cls):
        # The registry-derived keyword map resolves each family (linear), and one-parameter
        # families build from p1 alone (p2 absent, the ADR-0010/#417 one-number form).
        prior, scale = build_prior(keyword, p1, p2)
        assert isinstance(prior, cls)
        assert scale is LINEAR


class TestNoPrior:
    def test_logpdf_zero(self):
        assert NoPrior().logpdf(123.0) == 0.0

    def test_rvs_and_ppf_raise(self):
        with pytest.raises(PybnfError):
            NoPrior().rvs(np.random.default_rng(0))
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


# ---------------------------------------------------------------------------
# FreeParameter's prior-delegation surface (Move 3 -- the properties the
# algorithms ask instead of matching the *_var type string)
# ---------------------------------------------------------------------------

class TestFreeParameterPriorSurface:
    @pytest.mark.parametrize("keyword,bounded,has_prior", [
        ('normal_var', False, True),
        ('lognormal_var', False, True),
        ('uniform_var', True, True),
        ('loguniform_var', True, True),
        ('var', False, False),
        ('logvar', False, False),
    ])
    def test_flags_delegate_to_prior(self, keyword, bounded, has_prior):
        fp = pset.FreeParameter('x__FREE', keyword, 1.0, 2.0)
        assert fp.has_bounded_support is bounded
        assert fp.has_prior is has_prior

    @pytest.mark.parametrize("keyword,p1,p2", [
        ('uniform_var', 2.0, 8.0),
        ('loguniform_var', 0.01, 100.0),
    ])
    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_value_from_quantile_matches_lh_rescale(self, keyword, p1, p2, q):
        """FreeParameter.value_from_quantile(q) bit-matches the old LH rescale +
        set_value, the operation random_latin_hypercube_psets now calls."""
        fp = pset.FreeParameter('x__FREE', keyword, p1, p2)
        if keyword == 'uniform_var':
            expected = p1 + q * (p2 - p1)
        else:
            expected = 10.0 ** (np.log10(p1) + q * (np.log10(p2) - np.log10(p1)))
        assert fp.value_from_quantile(q).value == fp.set_value(expected).value


# ---------------------------------------------------------------------------
# Move 4 seam proof: one Laplace family file yields laplace_var (linear) and
# loglaplace_var (log10) end-to-end -- grammar, keyword map, prior, sampling --
# with no other code change.
# ---------------------------------------------------------------------------

class TestLaplaceSeam:
    def test_keyword_map_has_both_scales_for_free(self):
        assert priors.PRIOR_KEYWORD_MAP['laplace_var'] == (Laplace, LINEAR)
        assert priors.PRIOR_KEYWORD_MAP['loglaplace_var'] == (Laplace, LOG10)

    def test_grammar_recognizes_keywords(self):
        # Derived grammar (Move 2): an unbounded family lands in var_def_keys
        # (no b/u flag), so parse.py accepts laplace_var / loglaplace_var.
        import pybnf.parse as parse
        assert 'laplace_var' in parse.var_def_keys
        assert 'loglaplace_var' in parse.var_def_keys
        assert 'laplace_var' not in parse.b_var_def_keys

    def test_ploop_parses_a_laplace_var_line(self):
        import pybnf.parse as parse
        d = parse.ploop(['laplace_var = k__FREE 5 2'])
        assert d[('laplace_var', 'k__FREE')] == [5.0, 2.0]
        d2 = parse.ploop(['loglaplace_var = k__FREE 2 0.5'])
        assert d2[('loglaplace_var', 'k__FREE')] == [2.0, 0.5]

    @pytest.mark.parametrize("keyword,p1,p2,values", [
        ('laplace_var', 5.0, 2.0, [5.0, 8.0, 1.0, -3.0]),
        ('loglaplace_var', 2.0, 0.5, [100.0, 10.0, 1.0]),
    ])
    def test_prior_logpdf_matches_scipy_laplace(self, keyword, p1, p2, values):
        fp = pset.FreeParameter('x__FREE', keyword, p1, p2)
        ref = stats.laplace(loc=p1, scale=p2)
        for v in values:
            expected = ref.logpdf(np.log10(v) if keyword == 'loglaplace_var' else v)
            assert fp.prior_logpdf(v) == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_loglaplace_sampling_centered_in_log(self):
        rng = np.random.default_rng(0)
        fp = pset.FreeParameter('x__FREE', 'loglaplace_var', 2.0, 0.5)
        logs = np.array([np.log10(fp.sample_value(rng).value) for _ in range(40000)])
        assert logs.mean() == pytest.approx(2.0, abs=0.05)

    def test_laplace_is_unbounded_no_reflecting_box(self):
        # Unbounded family -> not box-bounded even with the default bounded flag.
        fp = pset.FreeParameter('x__FREE', 'laplace_var', 0.0, 1.0)
        assert not fp.has_bounded_support
        assert not fp.bounded
        assert fp.lower_bound == -np.inf and fp.upper_bound == np.inf


# ---------------------------------------------------------------------------
# TruncatedPrior: an unbounded family confined to a finite box in u (ADR-0020).
# Normal is oracled against scipy.stats.truncnorm (independent); both families
# are checked against the defining invariants -- the renormalized density
# integrates to 1 over the box, the inverse CDF inverts the truncated CDF and
# spans the box, and sampling lands inside the box.
# ---------------------------------------------------------------------------

def _truncnorm(loc, sigma, lo, hi):
    """scipy truncnorm with bounds given in the parameter's own space."""
    return stats.truncnorm((lo - loc) / sigma, (hi - loc) / sigma, loc=loc, scale=sigma)


class TestTruncatedPrior:
    def test_normal_logpdf_matches_truncnorm(self):
        tp = TruncatedPrior(Normal(loc=1.0, sigma=2.0), -1.0, 4.0)
        oracle = _truncnorm(1.0, 2.0, -1.0, 4.0)
        for u in (-1.0, 0.0, 1.0, 2.5, 4.0):
            assert tp.logpdf(u) == pytest.approx(oracle.logpdf(u), rel=1e-12)

    def test_normal_ppf_matches_truncnorm(self):
        tp = TruncatedPrior(Normal(loc=1.0, sigma=2.0), -1.0, 4.0)
        oracle = _truncnorm(1.0, 2.0, -1.0, 4.0)
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            assert tp.ppf(q) == pytest.approx(oracle.ppf(q), rel=1e-9)

    def test_logpdf_minus_inf_outside_box(self):
        tp = TruncatedPrior(Normal(0.0, 1.0), -2.0, 2.0)
        assert tp.logpdf(-2.0001) == -np.inf
        assert tp.logpdf(2.0001) == -np.inf
        assert np.isfinite(tp.logpdf(0.0))

    @pytest.mark.parametrize("inner,lo,hi", [
        (Normal(1.0, 2.0), -1.0, 4.0),
        (Laplace(0.0, 1.5), -2.0, 3.0),
    ])
    def test_density_integrates_to_one(self, inner, lo, hi):
        # Correct renormalization <=> the truncated density integrates to 1 over
        # the box. A direct check of Z, independent of the closed-form Z used.
        grid = np.linspace(lo, hi, 40001)
        tp = TruncatedPrior(inner, lo, hi)
        dens = np.exp(np.array([tp.logpdf(u) for u in grid]))
        assert _trapz(dens, grid) == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("inner,lo,hi", [
        (Normal(1.0, 2.0), -1.0, 4.0),
        (Laplace(0.0, 1.5), -2.0, 3.0),
    ])
    def test_ppf_spans_box_and_is_monotone(self, inner, lo, hi):
        tp = TruncatedPrior(inner, lo, hi)
        assert tp.ppf(0.0) == pytest.approx(lo, abs=1e-9)
        assert tp.ppf(1.0) == pytest.approx(hi, abs=1e-9)
        vals = np.array([tp.ppf(q) for q in np.linspace(0.0, 1.0, 21)])
        assert vals.min() >= lo - 1e-9 and vals.max() <= hi + 1e-9
        assert np.all(np.diff(vals) > 0)

    def test_rvs_in_box_and_moments_match_truncnorm(self):
        lo, hi, loc, sigma = 0.0, 6.0, 5.0, 3.0
        tp = TruncatedPrior(Normal(loc, sigma), lo, hi)
        oracle = _truncnorm(loc, sigma, lo, hi)
        rng = np.random.default_rng(0)
        xs = np.array([tp.rvs(rng) for _ in range(60000)])
        assert xs.min() >= lo and xs.max() <= hi
        assert xs.mean() == pytest.approx(oracle.mean(), abs=0.05)
        assert xs.std() == pytest.approx(oracle.std(), abs=0.05)

    @pytest.mark.parametrize("lo,hi", [
        (-1.0, np.inf),   # open above
        (-np.inf, 4.0),   # open below
    ])
    def test_half_bounded_matches_truncnorm(self, lo, hi):
        # One infinite bound: the decorator renormalizes over the half-line via
        # Z = cdf(hi) - cdf(lo) with cdf(+-inf) in {0, 1} (ADR-0047). Oracle against
        # scipy truncnorm with the matching infinite bound.
        tp = TruncatedPrior(Normal(loc=1.0, sigma=2.0), lo, hi)
        oracle = _truncnorm(1.0, 2.0, lo, hi)
        for u in (-0.5, 0.0, 1.0, 3.0):
            if lo <= u <= hi:
                assert tp.logpdf(u) == pytest.approx(oracle.logpdf(u), rel=1e-12)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            assert tp.ppf(q) == pytest.approx(oracle.ppf(q), rel=1e-9)
        assert tp.has_bounded_support and tp.support() == (lo, hi)

    def test_half_bounded_density_integrates_to_one(self):
        # Correct half-line renormalization: integrate over a wide finite proxy of
        # the open tail (the density is negligible far from the bulk).
        tp = TruncatedPrior(Normal(1.0, 2.0), 0.0, np.inf)
        grid = np.linspace(0.0, 60.0, 60001)
        dens = np.exp(np.array([tp.logpdf(u) for u in grid]))
        assert _trapz(dens, grid) == pytest.approx(1.0, abs=1e-4)

    def test_flags_and_support(self):
        tp = TruncatedPrior(Normal(0.0, 1.0), -2.0, 2.0)
        assert tp.has_bounded_support and tp.has_prior
        assert tp.support() == (-2.0, 2.0)
        assert tp.frozen is not None  # passthrough for FreeParameter._distribution

    def test_rejects_noprior_and_empty_box(self):
        with pytest.raises(ValueError):
            TruncatedPrior(NoPrior(), -1.0, 1.0)        # no scipy frozen to truncate
        with pytest.raises(ValueError):
            TruncatedPrior(Normal(0.0, 1.0), 2.0, 1.0)  # lo >= hi
