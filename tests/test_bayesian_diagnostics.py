"""
Oracle-anchored tests for the convergence-diagnostic machinery in
``BayesianAlgorithm`` (pybnf/algorithms.py): Brooks-Gelman split-R-hat,
rank-normalized split-R-hat (Vehtari et al. 2021), bulk/tail effective
sample size, the chain-splitting helper, and the sampling-space parameter
extractor.

These methods are pure numerical routines with strong oracles:

  * R-hat has an exact closed form (verifiable by hand on a tiny array) and is
    invariant under affine rescaling of the chains; the *rank-normalized*
    version is additionally invariant under any strictly monotonic transform.
  * ESS equals M*n for uncorrelated draws and collapses toward M*n*(1-rho)/(1+rho)
    for an AR(1) chain, monotonically decreasing in the autocorrelation rho.

The diagnostic methods only read ``self.chain_history``, ``self.num_parallel``
and ``self.variables``, so we exercise them on bare instances built with
``object.__new__`` rather than paying for the full (model-parsing,
directory-creating) constructor. ``_split_chain_rhat`` and ``_ess_from_chains``
are staticmethods and are called directly on the class.
"""
import numpy as np
import pytest
from hypothesis import given, assume, settings, strategies as st
from hypothesis.extra import numpy as hnp

from .context import algorithms, pset

BA = algorithms.BayesianAlgorithm


def _bare_ba(chain_history=None, num_parallel=None, variables=None):
    """A BayesianAlgorithm with only the attributes the diagnostics read set,
    bypassing the heavyweight constructor."""
    ba = object.__new__(BA)
    if chain_history is not None:
        ba.chain_history = chain_history
        ba.num_parallel = num_parallel if num_parallel is not None else len(chain_history)
    if variables is not None:
        ba.variables = variables
    return ba


def _chains_strategy(min_chains=2, max_chains=4, min_len=4, max_len=8, dims=1):
    """Strategy producing finite (N, n, d) chain arrays in a moderate range.

    Affine-invariance of R-hat is exact in real arithmetic but only *numerically*
    meaningful when each chain has within-chain variance comparable to its value
    scale: near-constant chains lose their sub-ULP variance to catastrophic
    cancellation under translation, giving a 0/0 = nan that is a float artifact,
    not a property violation. We therefore draw moderate-range values and the
    callers ``assume`` the chains are well-conditioned."""
    return hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_chains, max_chains),
            st.integers(min_len, max_len),
            st.just(dims),
        ),
        elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    )


def _well_conditioned(chains):
    """True when every chain carries enough within-chain spread that affine
    transforms preserve R-hat to within ordinary float tolerance."""
    return float(np.min(np.var(chains, axis=1, ddof=1))) > 0.5


# --------------------------------------------------------------------------- #
# _split_chain_rhat: the Brooks-Gelman potential scale reduction factor
# --------------------------------------------------------------------------- #
class TestSplitChainRhat:

    def test_matches_hand_computed_value(self):
        """Exact analytical oracle. For chains [0,1,2,3] and [1,2,3,4] (N=2, n=4):
        W = var_within = 5/3, B = n*var(means) = 4*0.5 = 2, sigma2 = (3/4)W + (1/4)B = 1.75,
        R-hat = sqrt((N+1)/N * sigma2/W - (n-1)/(N*n)) = sqrt(1.5*1.05 - 0.375) = sqrt(1.2)."""
        chains = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=float)[:, :, None]
        rhat = BA._split_chain_rhat(chains)
        np.testing.assert_allclose(rhat, np.sqrt(1.2), rtol=1e-12)

    def test_identical_chains_give_sqrt_ratio(self):
        """When all chains are identical the between-chain variance B=0, so
        sigma2 = ((n-1)/n) W and R-hat = sqrt((n-1)/n) exactly (the classic
        finite-n underestimate). Here n=4 -> sqrt(3/4)."""
        one = np.array([0.0, 1.0, 2.0, 3.0])
        chains = np.stack([one, one, one])[:, :, None]  # N=3, n=4, d=1
        rhat = BA._split_chain_rhat(chains)
        np.testing.assert_allclose(rhat, np.sqrt(3 / 4), rtol=1e-12)

    def test_well_separated_chains_exceed_one(self):
        """Chains drawn around widely separated means have B >> W, so R-hat must
        be substantially larger than 1 (the whole point of the diagnostic)."""
        rng = np.random.default_rng(0)
        n = 50
        chains = np.stack([
            rng.normal(loc=0.0, scale=1.0, size=n),
            rng.normal(loc=20.0, scale=1.0, size=n),
            rng.normal(loc=40.0, scale=1.0, size=n),
        ])[:, :, None]
        rhat = BA._split_chain_rhat(chains).item()
        assert rhat > 2.0

    def test_separation_increases_rhat(self):
        """Metamorphic: increasing only the between-chain separation (B grows,
        W fixed) must increase R-hat monotonically."""
        rng = np.random.default_rng(1)
        n = 80
        base = [rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)]
        rhats = []
        for sep in (1.0, 5.0, 25.0):
            chains = np.stack([base[0], base[1] + sep])[:, :, None]
            rhats.append(BA._split_chain_rhat(chains).item())
        assert rhats[0] < rhats[1] < rhats[2]

    @settings(max_examples=150)
    @given(chains=_chains_strategy(), c=st.floats(0.5, 5, allow_nan=False))
    def test_scale_invariance(self, chains, c):
        """R-hat is invariant under multiplying every chain value by a constant:
        B and W both scale by c^2, so their ratio (and thus R-hat) is unchanged."""
        assume(_well_conditioned(chains))
        base = BA._split_chain_rhat(chains)
        scaled = BA._split_chain_rhat(c * chains)
        np.testing.assert_allclose(scaled, base, rtol=1e-9)

    @settings(max_examples=150)
    @given(chains=_chains_strategy(), b=st.floats(-10, 10, allow_nan=False))
    def test_translation_invariance(self, chains, b):
        """R-hat is invariant under adding a constant to every chain value:
        variances (within and between) are unchanged by translation."""
        assume(_well_conditioned(chains))
        base = BA._split_chain_rhat(chains)
        shifted = BA._split_chain_rhat(chains + b)
        np.testing.assert_allclose(shifted, base, rtol=1e-6, atol=1e-9)

    def test_independent_per_dimension(self):
        """R-hat is computed per parameter; stacking two parameters must give the
        same answer as computing each alone."""
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, (3, 30))
        b = rng.normal(5, 2, (3, 30))
        stacked = np.stack([a, b], axis=-1)  # (3, 30, 2)
        rhat = BA._split_chain_rhat(stacked)
        np.testing.assert_allclose(rhat[0], BA._split_chain_rhat(a[:, :, None]).item(), rtol=1e-12)
        np.testing.assert_allclose(rhat[1], BA._split_chain_rhat(b[:, :, None]).item(), rtol=1e-12)


# --------------------------------------------------------------------------- #
# compute_rhat: rank-normalized, split, folded R-hat (Vehtari et al. 2021)
# --------------------------------------------------------------------------- #
def _make_chain_history(rng, num_parallel, length, n_dim, loc=0.0, scale=1.0):
    """Build a chain_history list-of-lists-of-vectors like the sampler records."""
    return [
        [rng.normal(loc, scale, n_dim) for _ in range(length)]
        for _ in range(num_parallel)
    ]


class TestComputeRhat:

    def test_none_when_insufficient_history(self):
        """Fewer than 20 recorded steps per chain -> no R-hat (returns None)."""
        rng = np.random.default_rng(3)
        ba = _bare_ba(_make_chain_history(rng, 4, 15, 2), num_parallel=4)
        assert ba.compute_rhat() is None

    def test_shape_matches_n_dim(self):
        rng = np.random.default_rng(4)
        ba = _bare_ba(_make_chain_history(rng, 4, 60, 3), num_parallel=4)
        rhat = ba.compute_rhat()
        assert rhat is not None and rhat.shape == (3,)

    def test_well_mixed_chains_near_one(self):
        """Independent chains from the same distribution are converged, so the
        rank-normalized R-hat should sit just above 1."""
        rng = np.random.default_rng(5)
        ba = _bare_ba(_make_chain_history(rng, 6, 400, 2), num_parallel=6)
        rhat = ba.compute_rhat()
        assert np.all(rhat < 1.1)

    def test_divergent_chains_exceed_one(self):
        """Chains parked at different locations are unconverged -> R-hat well
        above 1."""
        rng = np.random.default_rng(6)
        history = [
            [rng.normal(10.0 * j, 1.0, 1) for _ in range(60)]
            for j in range(4)
        ]
        ba = _bare_ba(history, num_parallel=4)
        assert np.nanmax(ba.compute_rhat()) > 1.5

    @pytest.mark.parametrize("transform", [np.exp, lambda x: x ** 3, lambda x: 2.0 * x + 7.0])
    def test_invariant_under_monotonic_transform(self, transform):
        """The defining property of *rank-normalized* R-hat: because it depends
        only on the ranks of the samples, any strictly increasing transform of
        every sample leaves it exactly unchanged. (Raw R-hat would change under
        a nonlinear transform like exp or x**3 — this is what distinguishes the
        rank-normalized variant.)"""
        rng = np.random.default_rng(7)
        history = _make_chain_history(rng, 4, 80, 2, loc=0.5, scale=0.3)
        ba = _bare_ba(history, num_parallel=4)
        base = ba.compute_rhat()

        transformed = [[transform(v) for v in chain] for chain in history]
        ba_t = _bare_ba(transformed, num_parallel=4)
        np.testing.assert_allclose(ba_t.compute_rhat(), base, rtol=1e-9)


# --------------------------------------------------------------------------- #
# _get_split_chains: last-50%-then-halve splitting
# --------------------------------------------------------------------------- #
class TestGetSplitChains:

    def test_none_below_minimum_length(self):
        rng = np.random.default_rng(8)
        ba = _bare_ba(_make_chain_history(rng, 3, 19, 2), num_parallel=3)
        assert ba._get_split_chains() is None

    def test_doubles_chain_count_and_shapes(self):
        """With num_parallel chains of length L, the splitter keeps the last
        ~L/2 steps and halves them, yielding 2*num_parallel sub-chains."""
        rng = np.random.default_rng(9)
        ba = _bare_ba(_make_chain_history(rng, 3, 40, 2), num_parallel=3)
        split = ba._get_split_chains()
        # min_len=40 -> start=20, usable=20, half=10
        assert split.shape == (6, 10, 2)

    def test_uses_second_half_split_in_two(self):
        """Content oracle: feed a per-chain ramp 0..39 so we know exactly which
        values land in each split. Last half starts at index 20; the two
        sub-chains are [20..29] and [30..39]."""
        history = [[np.array([float(i)]) for i in range(40)]]
        ba = _bare_ba(history, num_parallel=1)
        split = ba._get_split_chains()
        assert split.shape == (2, 10, 1)
        np.testing.assert_array_equal(split[0, :, 0], np.arange(20, 30))
        np.testing.assert_array_equal(split[1, :, 0], np.arange(30, 40))

    def test_uses_shortest_chain_length(self):
        """Ragged chains are truncated to the shortest; the split is driven by
        min length across chains."""
        rng = np.random.default_rng(10)
        history = [
            [rng.normal(0, 1, 1) for _ in range(40)],
            [rng.normal(0, 1, 1) for _ in range(100)],
        ]
        ba = _bare_ba(history, num_parallel=2)
        split = ba._get_split_chains()
        # min_len=40 -> half=10, 2 chains -> 4 sub-chains
        assert split.shape == (4, 10, 1)


# --------------------------------------------------------------------------- #
# _ess_from_chains: FFT autocovariance + Geyer initial positive sequence
# --------------------------------------------------------------------------- #
class TestEssFromChains:

    def test_nan_below_four_samples(self):
        assert np.isnan(BA._ess_from_chains(np.ones((2, 3))))

    def test_constant_chain_returns_full_count(self):
        """Zero variance short-circuits to the total sample count M*n."""
        assert BA._ess_from_chains(np.ones((4, 500))) == 4 * 500

    def test_white_noise_near_full_count(self):
        """Uncorrelated draws have autocorrelation ~0, so ESS ~ M*n."""
        rng = np.random.default_rng(11)
        M, n = 4, 500
        ess = BA._ess_from_chains(rng.standard_normal((M, n)))
        assert 0.7 * M * n < ess <= 1.05 * M * n

    @pytest.mark.parametrize("rho,frac", [(0.5, 0.6), (0.9, 0.2)])
    def test_autocorrelated_chain_loses_efficiency(self, rho, frac):
        """A positively autocorrelated AR(1) chain has ESS far below M*n; the
        theoretical value is M*n*(1-rho)/(1+rho)."""
        rng = np.random.default_rng(12)
        M, n = 4, 500
        x = np.zeros((M, n))
        for m in range(M):
            for t in range(1, n):
                x[m, t] = rho * x[m, t - 1] + rng.standard_normal()
        ess = BA._ess_from_chains(x)
        assert ess < frac * M * n

    def test_ess_matches_ar1_theory(self):
        """Quantitative oracle pinning the integrated-autocorrelation scaling.
        For an AR(1) chain the asymptotic ESS is M*n*(1-rho)/(1+rho); with
        M=8, n=2000, rho=0.6 the estimator lands within ~10% of it. The band is
        tight enough to catch a missing factor of 2 in the 1/(1 + 2*tau)
        normalization (which would inflate the estimate to ~1.5x theory)."""
        rng = np.random.default_rng(42)
        M, n, rho = 8, 2000, 0.6
        x = np.zeros((M, n))
        for m in range(M):
            for t in range(1, n):
                x[m, t] = rho * x[m, t - 1] + rng.standard_normal()
        ess = BA._ess_from_chains(x)
        theory = M * n * (1 - rho) / (1 + rho)
        assert 0.7 * theory < ess < 1.25 * theory

    def test_ess_decreases_with_autocorrelation(self):
        """Metamorphic: stronger autocorrelation -> fewer effective samples."""
        rng = np.random.default_rng(13)
        M, n = 4, 600

        def ar1(rho):
            x = np.zeros((M, n))
            r = np.random.default_rng(99)  # same innovations for both rho
            for m in range(M):
                for t in range(1, n):
                    x[m, t] = rho * x[m, t - 1] + r.standard_normal()
            return BA._ess_from_chains(x)

        assert ar1(0.3) > ar1(0.7) > ar1(0.95)

    def test_ess_at_least_one(self):
        """ESS is floored at 1 even for a pathologically anti-correlated chain."""
        x = np.tile([1.0, -1.0], (3, 50))  # period-2 oscillation
        assert BA._ess_from_chains(x) >= 1.0


# --------------------------------------------------------------------------- #
# compute_ess: bulk and tail ESS wrapper
# --------------------------------------------------------------------------- #
class TestComputeEss:

    def test_none_when_insufficient_history(self):
        rng = np.random.default_rng(14)
        ba = _bare_ba(_make_chain_history(rng, 3, 15, 2), num_parallel=3)
        assert ba.compute_ess() == (None, None)

    def test_shapes_and_positivity(self):
        rng = np.random.default_rng(15)
        ba = _bare_ba(_make_chain_history(rng, 6, 300, 2), num_parallel=6)
        bulk, tail = ba.compute_ess()
        assert bulk.shape == (2,) and tail.shape == (2,)
        assert np.all(bulk > 0) and np.all(tail > 0)

    def test_bulk_ess_substantial_for_mixed_chains(self):
        """For well-mixed chains the bulk ESS should be a sizeable fraction of
        the number of post-split samples, not collapsed."""
        rng = np.random.default_rng(16)
        ba = _bare_ba(_make_chain_history(rng, 6, 300, 1), num_parallel=6)
        split = ba._get_split_chains()
        n_samples = split.shape[0] * split.shape[1]
        bulk, _ = ba.compute_ess()
        assert bulk[0] > 0.3 * n_samples


# --------------------------------------------------------------------------- #
# _param_vec: extract parameters into the sampling space
# --------------------------------------------------------------------------- #
class TestParamVec:

    def test_linear_and_log_space_components(self):
        """Log-space parameters are returned as log10(value); linear-space
        parameters are returned unchanged; order follows self.variables."""
        v_lin = pset.FreeParameter('a__FREE', 'normal_var', 0.0, 1.0, 7.0)
        v_log = pset.FreeParameter('b__FREE', 'lognormal_var', 0.0, 1.0, 100.0)
        ba = _bare_ba(variables=[v_lin, v_log])
        ps = pset.PSet([
            pset.FreeParameter('a__FREE', 'normal_var', 0.0, 1.0, 7.0),
            pset.FreeParameter('b__FREE', 'lognormal_var', 0.0, 1.0, 100.0),
        ])
        vec = ba._param_vec(ps)
        np.testing.assert_allclose(vec, [7.0, np.log10(100.0)], rtol=1e-12)
