"""
Oracle-anchored tests for the *preconditioning* machinery that
``PDreamAlgorithm`` (the ``p_dream`` sampler, pybnf/algorithms.py) adds on top
of DREAM(ZS).  Everything DREAM- or Bayesian-generic (R-hat / ESS / acceptance,
the base DE proposal, ln_prior) is already covered by test_bayesian_diagnostics,
test_adaptive_mcmc and test_dream_class and is NOT re-tested here.

The "P" is preconditioning: an online Adaptive-Metropolis covariance estimate
C = Cov(chain) + eps*I is learned, factored C = L Lᵀ, and DE proposals are
computed in the covariance-whitened space z = L⁻¹ x (decorrelated coordinates),
then the jump is mapped back via dx = L dz.  The PDream-specific surface is:

  * ``_update_covariance`` — Haario-regularized sample covariance + Cholesky,
    with L and L⁻¹ stored.  Strong oracles: the exact algebraic reconstruction
    L Lᵀ = Cov + eps*I, lower-triangularity of L, L⁻¹L = I, and the *purpose*
    invariant that whitening the pooled samples decorrelates them (cov ≈ I).
  * ``_whiten`` / ``_unwhiten_diff`` — a matched (L⁻¹, L) pair, so
    unwhiten(whiten(x)) = x exactly (a round-trip oracle that pins the pairing).
  * ``calculate_new_pset`` — the whitened DE proposal.  Two oracles: before
    preconditioning it must delegate *identically* to the base DREAM proposal;
    once active, under a full crossover mask the whitening cancels
    (dx = L L⁻¹ D = D) so the jump equals the plain original-space DE difference
    ±(A−B) regardless of L.
  * ``got_result`` — the hook that fires ``_update_covariance`` once per synced
    generation, but only after ``precondition_adapt`` iterations.

Pure-numeric helpers are exercised on bare ``object.__new__`` instances (only
the attributes they read are set), following test_bayesian_diagnostics.  The
proposal / hook contracts need a real (model-parsing) instance, built from a
config.Configuration over the three v*__FREE params in bngl_files/parabola.bngl
exactly as test_dream_class / test_adaptive_mcmc do.
"""
import os

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from .context import algorithms, config, pset

PD = algorithms.PDreamAlgorithm
PARABOLA = 'bngl_files/parabola.bngl'


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bare_pd(**attrs):
    """A PDreamAlgorithm with only the attributes a method reads set, bypassing
    the heavyweight (model-parsing) constructor."""
    pd = object.__new__(PD)
    for k, v in attrs.items():
        setattr(pd, k, v)
    return pd


def _spd_cholesky(d, seed=0):
    """A non-trivial lower-triangular Cholesky factor L (and its inverse) of a
    well-conditioned SPD matrix, for round-trip / whitening tests."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    C = A @ A.T + d * np.eye(d)          # strictly SPD, well conditioned
    L = np.linalg.cholesky(C)
    return L, np.linalg.inv(L)


def _make_config(tmp_path, var_spec, **overrides):
    out = str(tmp_path) + '/'
    # start_run() writes the samples file and Histograms dir under output_dir.
    os.makedirs(out + 'Results/Histograms', exist_ok=True)
    base = {
        'population_size': 6, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 10,
        'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': out,
        'models': {PARABOLA}, 'exp_data': {'bngl_files/par1.exp'},
        'initialization': 'lh', PARABOLA: ['bngl_files/par1.exp'],
        'fit_type': 'p_dream',
    }
    base.update(var_spec)
    base.update(overrides)
    return config.Configuration(base)


# Three box-uniform params (bounded -> OOB proposals can be rejected).
UNIFORM_VARS = {
    ('uniform_var', 'v1__FREE'): [0.0, 10.0],
    ('uniform_var', 'v2__FREE'): [0.0, 10.0],
    ('uniform_var', 'v3__FREE'): [0.0, 10.0],
}
# Three unbounded normal params (proposals never rejected for bounds).
NORMAL_VARS = {
    ('normal_var', 'v1__FREE'): [0.0, 1.0],
    ('normal_var', 'v2__FREE'): [0.0, 1.0],
    ('normal_var', 'v3__FREE'): [0.0, 1.0],
}


def _normal_pset(values):
    ps = pset.PSet([
        pset.FreeParameter('v1__FREE', 'normal_var', 0.0, 1.0, values[0]),
        pset.FreeParameter('v2__FREE', 'normal_var', 0.0, 1.0, values[1]),
        pset.FreeParameter('v3__FREE', 'normal_var', 0.0, 1.0, values[2]),
    ])
    return ps


def _uniform_pset(values):
    ps = pset.PSet([
        pset.FreeParameter('v1__FREE', 'uniform_var', 0.0, 10.0, values[0]),
        pset.FreeParameter('v2__FREE', 'uniform_var', 0.0, 10.0, values[1]),
        pset.FreeParameter('v3__FREE', 'uniform_var', 0.0, 10.0, values[2]),
    ])
    return ps


# --------------------------------------------------------------------------- #
# __init__: precondition_adapt default
# --------------------------------------------------------------------------- #
class TestPreconditionAdaptDefault:

    @pytest.mark.parametrize("pa,burn_in,expected", [
        (None, 10, 5),     # None -> burn_in // 2
        (None, 7, 3),      # integer division
        (4, 10, 4),        # explicit value honored
        (0, 10, 0),        # explicit zero honored (not treated as "unset")
    ])
    def test_default_is_half_burn_in(self, tmp_path, pa, burn_in, expected):
        """precondition_adapt defaults to burn_in // 2 when unset (None), and is
        taken verbatim otherwise. Oracle: the documented fallback formula."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, burn_in=burn_in,
                           precondition_adapt=pa)
        pd = PD(cfg)
        assert pd.precondition_adapt == expected


# --------------------------------------------------------------------------- #
# _update_covariance: Haario-regularized covariance + Cholesky whitening
# --------------------------------------------------------------------------- #
def _pd_for_cov(chain_history, n_dim):
    return _bare_pd(chain_history=chain_history, n_dim=n_dim,
                    iteration=[0], _cov_L=None, _cov_L_inv=None,
                    _preconditioned=False)


def _pool(chain_history, n_dim):
    """Replicate the pooling the method does (the post-warmup last 50% of each
    chain of length > 1) so the test can build the oracle covariance independently."""
    samples = []
    for chain in chain_history:
        if len(chain) > 1:
            samples.extend(chain[len(chain) // 2:])
    return np.array(samples)


class TestUpdateCovariance:

    def _make_history(self, d=3, n=400, seed=1):
        """One long chain of n correlated d-vectors (a single list, length > 1)."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((d, d)) + np.eye(d)   # nontrivial mixing
        X = rng.standard_normal((n, d)) @ A.T + np.array([2.0, -1.0, 0.5])[:d]
        return [list(X)], X

    def test_reconstruction_identity(self):
        """Exact algebraic oracle: the stored Cholesky factor must satisfy
        L Lᵀ = sample_cov + eps*I, with eps = 1e-6 * trace(sample_cov)/d (the
        Haario regularization). This pins both the regularization and that L is
        the Cholesky factor of the *regularized* matrix."""
        d = 3
        hist, _ = self._make_history(d=d)
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()

        cov = np.cov(_pool(hist, d), rowvar=False)   # oracle over the post-warmup pool
        eps = 1e-6 * np.trace(cov) / d
        np.testing.assert_allclose(pd._cov_L @ pd._cov_L.T, cov + eps * np.eye(d),
                                   rtol=1e-10, atol=1e-12)

    def test_eps_is_trace_scaled(self):
        """The regularization added to the diagonal is exactly eps*I with
        eps = 1e-6 * trace(cov)/d — off-diagonals of (L Lᵀ - cov) are zero, the
        diagonal is the single scalar eps. (A constant 1e-6, or trace without
        the /d, would fail.)"""
        d = 3
        hist, _ = self._make_history(d=d, seed=2)
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()

        cov = np.cov(_pool(hist, d), rowvar=False)
        added = pd._cov_L @ pd._cov_L.T - cov
        eps = 1e-6 * np.trace(cov) / d
        np.testing.assert_allclose(added, eps * np.eye(d), atol=1e-12)

    def test_L_is_lower_triangular(self):
        """np.linalg.cholesky returns a lower-triangular factor; the stored
        _cov_L must be lower triangular (strictly upper part exactly zero)."""
        d = 3
        hist, _ = self._make_history(d=d, seed=3)
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()
        np.testing.assert_array_equal(pd._cov_L, np.tril(pd._cov_L))

    def test_cov_L_inv_is_inverse(self):
        """_cov_L_inv is L⁻¹: L⁻¹ L = I exactly (it is solved as solve(L, I))."""
        d = 3
        hist, _ = self._make_history(d=d, seed=4)
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()
        np.testing.assert_allclose(pd._cov_L_inv @ pd._cov_L, np.eye(d),
                                   rtol=1e-10, atol=1e-12)

    def test_whitening_decorrelates_pooled_samples(self):
        """Purpose oracle: the whole point of preconditioning is that whitening
        the chain history with L⁻¹ removes correlation. The pooled samples
        transformed by z = L⁻¹ x must have an (almost) identity covariance —
        unit diagonal, ~zero off-diagonal. The tiny residual is exactly
        -eps * (L⁻¹ L⁻¹ᵀ) (eps ~ 1e-6), so 1e-3 tolerances are far tighter than
        the O(1) off-diagonals of the un-whitened data."""
        d = 3
        hist, _ = self._make_history(d=d, seed=5)
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()

        Xp = _pool(hist, d)              # the post-warmup pool the method whitened
        Z = Xp @ pd._cov_L_inv.T         # z_i = L⁻¹ x_i, stacked
        cov_Z = np.cov(Z, rowvar=False)
        np.testing.assert_allclose(np.diag(cov_Z), np.ones(d), atol=1e-3)
        off = cov_Z[~np.eye(d, dtype=bool)]
        assert np.max(np.abs(off)) < 1e-3
        # Sanity: the un-whitened data really was correlated (else the test is vacuous).
        raw_off = np.cov(Xp, rowvar=False)[~np.eye(d, dtype=bool)]
        assert np.max(np.abs(raw_off)) > 0.1

    def test_regularization_enables_rank_deficient_cov(self):
        """Haario's eps*I is what lets a singular sample covariance be Cholesky-
        factored. Samples lying on a line in 2-D have a rank-1 (singular) sample
        covariance whose bare Cholesky raises; with regularization the update
        succeeds and still satisfies the reconstruction identity."""
        d = 2
        rng = np.random.default_rng(6)
        t = rng.standard_normal(80)
        X = np.column_stack([t, 2.0 * t + 1.0])      # exact line -> rank-1 cov
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(np.cov(X, rowvar=False))   # bare factorization fails

        hist = [list(X)]
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()
        assert pd._cov_L is not None
        cov = np.cov(_pool(hist, d), rowvar=False)   # post-warmup half (still collinear -> rank-1)
        eps = 1e-6 * np.trace(cov) / d
        np.testing.assert_allclose(pd._cov_L @ pd._cov_L.T, cov + eps * np.eye(d),
                                   rtol=1e-8, atol=1e-12)

    def test_no_update_below_sample_threshold(self):
        """Gate: with fewer than 2*n_dim pooled samples there is no covariance
        update — _cov_L stays None and _preconditioned stays False. (The pool is
        the post-warmup half: 10 raw -> 5 trimmed, still < the threshold of 6.)"""
        d = 3                                  # threshold = 2*d = 6 pooled samples
        rng = np.random.default_rng(7)
        hist = [list(rng.standard_normal((10, d)))]  # last-50% -> 5 < 6
        pd = _pd_for_cov(hist, d)
        assert pd._update_covariance() is None
        assert pd._cov_L is None and pd._cov_L_inv is None
        assert pd._preconditioned is False

    def test_covariance_uses_post_warmup_half_only(self):
        """PDREAM-1: _update_covariance discards the first 50% of each chain
        (warmup) so the burn-in transient does not inflate the preconditioner.
        Build a chain whose first half is a wide transient and second half a tight
        stationary block; the recovered covariance must match the second half, not
        the pooled full history (which the pre-fix code used). Mutation guard: the
        full-history trace is many times larger, so the pre-fix code fails this."""
        d = 3
        rng = np.random.default_rng(11)
        n = 400
        first = rng.standard_normal((n, d)) * 10.0 + 50.0   # wide transient
        second = rng.standard_normal((n, d)) * 1.0          # tight stationary
        X = np.vstack([first, second])                      # len 800 -> trim to X[400:] == second
        hist = [list(X)]
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()

        cov_second = np.cov(second, rowvar=False)
        eps = 1e-6 * np.trace(cov_second) / d
        np.testing.assert_allclose(pd._cov_L @ pd._cov_L.T, cov_second + eps * np.eye(d),
                                   rtol=1e-10, atol=1e-12)
        # Mutation guard: pre-fix full-history covariance is wildly inflated by the transient.
        assert np.trace(np.cov(X, rowvar=False)) > 10 * np.trace(cov_second)

    def test_pools_post_warmup_across_chains(self):
        """Claim B (retained behavior, not a defect): the preconditioner pools the
        post-warmup half across ALL chains — P-DREAM's global proposal scale for
        archive-based mode hopping — rather than a single chain. Two chains each
        contribute their last 50%."""
        d = 2
        rng = np.random.default_rng(12)
        c0 = list(rng.standard_normal((200, d)) * 2.0)
        c1 = list(rng.standard_normal((200, d)) * 2.0 + 5.0)
        hist = [c0, c1]
        pd = _pd_for_cov(hist, d)
        pd._update_covariance()

        Xp = _pool(hist, d)              # 100 + 100 pooled across both chains
        assert Xp.shape[0] == 200
        cov = np.cov(Xp, rowvar=False)
        eps = 1e-6 * np.trace(cov) / d
        np.testing.assert_allclose(pd._cov_L @ pd._cov_L.T, cov + eps * np.eye(d),
                                   rtol=1e-10, atol=1e-12)

    def test_length_one_chains_excluded_from_pool(self):
        """Only chains with length > 1 are pooled. Appending extra length-1
        chains (even with extreme values) must not change the estimated factor:
        they are dropped before covariance estimation."""
        d = 3
        hist, X = self._make_history(d=d, seed=8)
        pd_a = _pd_for_cov([list(X)], d)
        pd_a._update_covariance()

        outlier = [np.array([1e6, -1e6, 1e6])]       # a singleton chain
        pd_b = _pd_for_cov([list(X), outlier, outlier], d)
        pd_b._update_covariance()

        np.testing.assert_allclose(pd_b._cov_L, pd_a._cov_L, rtol=1e-12, atol=1e-12)

    def test_first_success_sets_preconditioned_flag(self):
        """The flag flips to True exactly once on the first successful update
        (it is the gate calculate_new_pset reads to switch to whitened space)."""
        d = 3
        hist, _ = self._make_history(d=d, seed=9)
        pd = _pd_for_cov(hist, d)
        assert pd._preconditioned is False
        pd._update_covariance()
        assert pd._preconditioned is True


# --------------------------------------------------------------------------- #
# _whiten / _unwhiten_diff: the matched (L⁻¹, L) transform pair
# --------------------------------------------------------------------------- #
class TestWhitenRoundTrip:

    @pytest.mark.parametrize("d,seed", [(2, 0), (3, 1), (4, 2)])
    def test_unwhiten_of_whiten_is_identity(self, d, seed):
        """unwhiten_diff(whiten(x)) = L (L⁻¹ x) = x. This pins that _whiten uses
        L⁻¹ and _unwhiten_diff uses L (the matched factor), not the same matrix
        twice: L L⁻¹ = I but L L ≠ I and L⁻¹ L⁻¹ ≠ I for a non-orthogonal L."""
        L, L_inv = _spd_cholesky(d, seed)
        pd = _bare_pd(_cov_L=L, _cov_L_inv=L_inv)
        rng = np.random.default_rng(seed + 100)
        x = rng.standard_normal(d) * 5.0
        np.testing.assert_allclose(pd._unwhiten_diff(pd._whiten(x)), x,
                                   rtol=1e-10, atol=1e-12)

    @settings(max_examples=100, deadline=None)
    @given(x=hnp.arrays(np.float64, (3,),
                        elements=st.floats(-1e3, 1e3, allow_nan=False)))
    def test_round_trip_property(self, x):
        """Property form of the round trip over a class of vectors, with a fixed
        non-trivial L. Catches any asymmetry between the forward/back maps."""
        L, L_inv = _spd_cholesky(3, seed=42)
        pd = _bare_pd(_cov_L=L, _cov_L_inv=L_inv)
        np.testing.assert_allclose(pd._unwhiten_diff(pd._whiten(x)), x,
                                   rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------- #
# calculate_new_pset: the whitened DE proposal
# --------------------------------------------------------------------------- #
class TestCalculateNewPset:

    def test_delegates_to_base_before_preconditioning(self, tmp_path):
        """Before preconditioning activates, calculate_new_pset must return the
        plain DREAM proposal. Oracle: with the same RNG seed and identical
        state, the PDream call and DreamAlgorithm.calculate_new_pset on the same
        instance produce the identical proposal — and it never touches the
        (None) whitening matrices."""
        cfg = _make_config(tmp_path, NORMAL_VARS)
        pd = PD(cfg)
        first = pd.start_run()                 # populates the ZS archive
        pd.current_pset = list(first)          # a real (non-None) current state
        assert pd._preconditioned is False
        assert pd._cov_L is None and pd._cov_L_inv is None   # nothing to dot with

        np.random.seed(2024)
        prop_pd, cr_pd = pd.calculate_new_pset(0)
        np.random.seed(2024)
        prop_base, cr_base = algorithms.DreamAlgorithm.calculate_new_pset(pd, 0)

        assert cr_pd == cr_base
        np.testing.assert_array_equal(pd._param_vec(prop_pd), pd._param_vec(prop_base))

    def test_whitening_cancels_under_full_mask(self, tmp_path):
        """Core preconditioning oracle. Under a full crossover mask (forced by
        gamma_prob=1 -> mode jump, with zeta=lambda=0 to silence the random
        perturbation), the proposed jump is

            dx = L (1 · (L⁻¹ D)) = (L L⁻¹) D = D,

        the plain original-space DE difference, *independent of the covariance
        L*. With delta=1 and an archive of exactly two donors, D = ±(A − B).
        So proposed − current must equal ±(A − B) for a non-identity L — which
        it can only do if donors are whitened (L⁻¹) and the jump unwhitened with
        the matched L. Using L⁻¹ for the back-transform, or skipping the donor
        whitening, would leave a residual L-dependent factor and fail."""
        cfg = _make_config(tmp_path, NORMAL_VARS, gamma_prob=1.0, delta=1,
                           zeta=0.0, **{'lambda': 0.0})
        pd = PD(cfg)

        L, L_inv = _spd_cholesky(3, seed=11)
        pd._preconditioned = True
        pd._cov_L = L
        pd._cov_L_inv = L_inv

        x0 = _normal_pset((0.1, 0.2, 0.3))
        A = _normal_pset((1.0, 0.5, -0.2))
        B = _normal_pset((0.3, 0.9, 0.4))
        pd.current_pset = [x0]
        pd.archive = [A, B]

        D = pd._param_vec(A) - pd._param_vec(B)
        x0_vec = pd._param_vec(x0)

        seen_plus = seen_minus = False
        for s in range(40):
            np.random.seed(s)
            prop, _ = pd.calculate_new_pset(0)
            jump = pd._param_vec(prop) - x0_vec
            # The jump equals +D or -D regardless of the (non-identity) L.
            close_plus = np.allclose(jump, D, atol=1e-9)
            close_minus = np.allclose(jump, -D, atol=1e-9)
            assert close_plus or close_minus, jump
            seen_plus |= close_plus
            seen_minus |= close_minus
        assert seen_plus and seen_minus    # both donor orderings actually occurred

    def test_out_of_bounds_proposal_rejected(self, tmp_path):
        """A whitened proposal that lands outside the box returns (None, cr_idx):
        the OOB rejection path. Mode jump + a huge donor difference drives
        current + D far past the [0, 10] bounds, so set_value(reflect=False)
        raises OutOfBoundsException and the method returns None (with the int
        crossover index still reported)."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, gamma_prob=1.0, delta=1,
                           zeta=0.0, **{'lambda': 0.0})
        pd = PD(cfg)
        L, L_inv = _spd_cholesky(3, seed=12)
        pd._preconditioned = True
        pd._cov_L = L
        pd._cov_L_inv = L_inv

        x0 = _uniform_pset((5.0, 5.0, 5.0))
        A = _uniform_pset((9.9, 9.9, 9.9))
        B = _uniform_pset((0.1, 0.1, 0.1))     # |A - B| ~ 9.8 per dim -> jump leaves box
        pd.current_pset = [x0]
        pd.archive = [A, B]

        np.random.seed(3)
        prop, cr_idx = pd.calculate_new_pset(0)
        assert prop is None
        assert isinstance(cr_idx, (int, np.integer))
        assert 0 <= cr_idx < pd.ncr_count


# --------------------------------------------------------------------------- #
# got_result: the covariance-update hook
# --------------------------------------------------------------------------- #
class TestGotResultHook:

    @pytest.mark.parametrize("precondition_adapt,expected_calls", [
        (0, 1),     # min(iteration)=1 >= 0  -> fires on the synced generation
        (1, 1),     # 1 >= 1                 -> fires
        (2, 0),     # 1 <  2                 -> does not fire yet
        (50, 0),    # well past              -> does not fire
    ])
    def test_update_fires_only_after_adapt_on_sync(self, tmp_path, monkeypatch,
                                                    precondition_adapt, expected_calls):
        """The hook runs _update_covariance exactly once per *synced* generation
        (super().got_result returns a non-empty list), and only once
        min(iteration) >= precondition_adapt. After one full generation every
        chain is at iteration 1, so the call happens iff precondition_adapt <= 1.
        Oracle: a counting spy on _update_covariance; partial (non-final)
        got_result calls return [] and must not trigger it."""
        cfg = _make_config(tmp_path, UNIFORM_VARS, population_size=6,
                           precondition_adapt=precondition_adapt)
        pd = PD(cfg)
        psets = pd.start_run()

        calls = {'n': 0}
        monkeypatch.setattr(pd, '_update_covariance',
                            lambda: calls.__setitem__('n', calls['n'] + 1))

        outputs = []
        for ps in psets:
            res = algorithms.Result(ps, {}, ps.name)
            res.score = 12.0
            outputs.append(pd.got_result(res))

        # Exactly one generation sync occurred (the final result returned a list).
        assert sum(isinstance(o, list) and len(o) > 0 for o in outputs) == 1
        assert calls['n'] == expected_calls
