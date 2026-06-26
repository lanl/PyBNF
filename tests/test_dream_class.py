from .context import algorithms, objective, data, config, pset

import os
import numpy as np
import pytest
import re
import shutil
import tempfile
from scipy import stats


class _PinnedChoiceRng:
    """Real Generator with choice() pinned to a fixed donor-index array, delegating
    all other draws. Stands in for a chain's Generator (whose methods are read-only)
    when a test needs to freeze archive donor selection."""

    def __init__(self, indices, seed=0):
        self._idx = np.array(indices)
        self._g = np.random.default_rng(seed)

    def choice(self, *a, **k):
        return self._idx

    def __getattr__(self, name):
        return getattr(self._g, name)



class TestDream:
    @classmethod
    def setup_class(cls):
        cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, r'\s+')

        cls.variables = ['v1__FREE', 'v2__FREE', 'v3__FREE']

        cls.chi_sq = objective.ChiSquareObjective()

        # Per-class temp output dirs (absolute + unique) instead of the shared
        # relative noseoutput1/noseoutput2 this class once shared with
        # test_bayes_mcmc -- under pytest-xdist the two classes raced on those
        # names, one teardown_class deleting a directory the other was mid-test
        # inside. A unique tempdir per class removes the collision.
        cls._tmpdir = tempfile.mkdtemp(prefix='test_dream_class_')
        cls.out1 = os.path.join(cls._tmpdir, 'noseoutput1') + os.sep
        cls.out2 = os.path.join(cls._tmpdir, 'noseoutput2') + os.sep
        os.makedirs(os.path.join(cls.out1, 'Results'), exist_ok=True)
        os.makedirs(os.path.join(cls.out2, 'Results'), exist_ok=True)

        cls.config = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 10, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': cls.out1,
            ('uniform_var', 'v1__FREE'): [0., 0.5], ('loguniform_var', 'v2__FREE'): [1., 10.], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'fit_type': 'dream'})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_start(self):
        dream = algorithms.DreamAlgorithm(self.config)
        start_psets = dream.start_run()
        assert len(dream.variables) == 3
        assert sorted([v.name for v in dream.variables]) == ['v1__FREE', 'v2__FREE', 'v3__FREE']
        assert len(start_psets) == 20
        assert dream.prior['v1__FREE'].type == 'uniform_var'
        assert dream.prior['v1__FREE'].p1 == 0.
        assert dream.prior['v1__FREE'].p2 == 0.5
        assert dream.prior['v2__FREE'].type == 'loguniform_var'
        assert dream.prior['v2__FREE'].p1 == 1.
        assert dream.prior['v2__FREE'].p2 == 10.
        assert dream.prior['v3__FREE'].type == 'uniform_var'
        assert dream.prior['v3__FREE'].p1 == 0
        assert dream.prior['v3__FREE'].p2 == 10
        assert np.isfinite(dream.prior['v1__FREE'].prior_logpdf(0.25))
        assert np.isfinite(dream.prior['v2__FREE'].prior_logpdf(5.))
        assert np.isfinite(dream.prior['v3__FREE'].prior_logpdf(5.))

    def test_update(self):
        dream = algorithms.DreamAlgorithm(self.config)
        start_psets = dream.start_run()
        for i, pset in enumerate(start_psets):
            res = algorithms.Result(pset, self.d1s, pset.name)
            res.score = 12
            if i == len(start_psets) - 1:
                next_gen = dream.got_result(res)
                assert len(next_gen) > 0  # unlikely that all updates are outside of the variable bounds
            else:
                empty = dream.got_result(res)
                assert empty == []

        for pset in next_gen:
            assert re.match(r'iter1run\d+', pset.name) is not None


# =========================================================================== #
# DREAM(ZS)-specific machinery (oracle-anchored).
#
# The Bayesian-generic diagnostics (R-hat / ESS / acceptance / _param_vec /
# ln_prior) are covered by test_bayesian_diagnostics / test_adaptive_mcmc /
# test_distributions and are NOT re-tested here. These classes target the
# pieces DreamAlgorithm adds: outlier detection (IQR + Grubbs) and the chain
# reset it drives, the snooker proposal with its Hastings correction, and the
# differential-evolution proposal.
# =========================================================================== #
PARABOLA = 'bngl_files/parabola.bngl'


def _make_dream_config(tmp_path, var_spec, **overrides):
    out = str(tmp_path) + '/'
    os.makedirs(out + 'Results/Histograms', exist_ok=True)
    base = {
        'population_size': 6, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 10,
        'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': out,
        'models': {PARABOLA}, 'exp_data': {'bngl_files/par1.exp'},
        'initialization': 'lh', PARABOLA: ['bngl_files/par1.exp'],
        'fit_type': 'dream',
    }
    base.update(var_spec)
    base.update(overrides)
    return config.Configuration(base)


NORMAL_VARS = {
    ('normal_var', 'v1__FREE'): [0.0, 1.0],
    ('normal_var', 'v2__FREE'): [0.0, 1.0],
    ('normal_var', 'v3__FREE'): [0.0, 1.0],
}
UNIFORM_VARS = {
    ('uniform_var', 'v1__FREE'): [0.0, 10.0],
    ('uniform_var', 'v2__FREE'): [0.0, 10.0],
    ('uniform_var', 'v3__FREE'): [0.0, 10.0],
}


def _normal_pset(values):
    return pset.PSet([
        pset.FreeParameter('v1__FREE', 'normal_var', 0.0, 1.0, values[0]),
        pset.FreeParameter('v2__FREE', 'normal_var', 0.0, 1.0, values[1]),
        pset.FreeParameter('v3__FREE', 'normal_var', 0.0, 1.0, values[2]),
    ])


def _uniform_pset(values):
    return pset.PSet([
        pset.FreeParameter('v1__FREE', 'uniform_var', 0.0, 10.0, values[0]),
        pset.FreeParameter('v2__FREE', 'uniform_var', 0.0, 10.0, values[1]),
        pset.FreeParameter('v3__FREE', 'uniform_var', 0.0, 10.0, values[2]),
    ])


# --------------------------------------------------------------------------- #
# _detect_outliers_iqr: chains with mean ln-posterior below Q25 - 2*IQR
# --------------------------------------------------------------------------- #
class TestOutlierDetectionIQR:

    def _bare(self):
        return object.__new__(algorithms.DreamAlgorithm)

    def test_matches_iqr_rule(self):
        """Reference oracle re-stating the rule independently: outliers are the
        chains whose mean ln-posterior is strictly below Q25 - 2*IQR, with
        Q25/Q75 the 25th/75th percentiles and IQR = Q75 - Q25. Pins the
        percentile choice (Q25 not Q75), the factor 2, and the < direction."""
        da = self._bare()
        arr = np.array([3.1, 2.9, 3.0, 3.2, 2.8, 3.05, 2.95, -12.0])
        q75, q25 = np.percentile(arr, 75), np.percentile(arr, 25)
        expected = np.where(arr < q25 - 2.0 * (q75 - q25))[0]
        np.testing.assert_array_equal(da._detect_outliers_iqr(arr), expected)
        assert 7 in expected            # the -12 chain really is flagged

    def test_no_outlier_in_tight_cluster(self):
        """A tightly clustered set has IQR small but no point 2*IQR below Q25,
        so no chain is flagged (empty result)."""
        da = self._bare()
        arr = np.array([5.0, 5.1, 4.9, 5.05, 4.95, 5.02])
        assert da._detect_outliers_iqr(arr).size == 0

    def test_factor_two_lower_fence(self):
        """Pin the 2x multiplier from both sides. A large two-level cluster gives
        Q25~=10, Q75~=20 (IQR~=10), so the fences are 0 (1x), -10 (2x), -20 (3x).
        Appending a single point barely shifts the quartiles, so:
          * x = -15 is below the 2x fence -> flagged (rules out a 3x fence);
          * x = -5  is above the 2x fence -> NOT flagged (rules out a 1x fence).
        Only the correct factor of 2 satisfies both."""
        da = self._bare()
        cluster = np.concatenate([np.full(50, 10.0), np.full(50, 20.0)])

        below = np.append(cluster, -15.0)
        assert (len(below) - 1) in da._detect_outliers_iqr(below)

        above = np.append(cluster, -5.0)
        assert (len(above) - 1) not in da._detect_outliers_iqr(above)


# --------------------------------------------------------------------------- #
# _detect_outliers_grubbs: Grubbs test for a single low outlier (alpha=0.01)
# --------------------------------------------------------------------------- #
class TestOutlierDetectionGrubbs:

    def _bare(self):
        return object.__new__(algorithms.DreamAlgorithm)

    @staticmethod
    def _grubbs_reference(arr):
        """Independent re-implementation of the one-sided (minimum) Grubbs test
        at alpha=0.01, used as the oracle."""
        N = len(arr)
        if N < 3:
            return np.array([], dtype=int)
        mu = np.mean(arr)
        sd = np.std(arr, ddof=1)
        if sd < 1e-20:
            return np.array([], dtype=int)
        G = (mu - np.min(arr)) / sd
        t2 = stats.t.ppf(0.01 / (2 * N), N - 2) ** 2
        Tc = (N - 1) / np.sqrt(N) * np.sqrt(t2 / (N - 2 + t2))
        return np.array([np.argmin(arr)]) if G > Tc else np.array([], dtype=int)

    @pytest.mark.parametrize("arr", [
        np.array([10.0, 10.1, 9.9, 10.05, 9.95, 3.0]),    # clear low outlier
        np.array([10.0, 10.1, 9.9, 10.05, 9.95, 10.02]),  # no outlier
        np.array([1.0, 2.0, -8.0, 1.5, 0.5, 1.2, 0.8]),   # moderate low point
        # Borderline: G=2.32 sits between the correct critical value (2.48) and
        # the value a dropped sqrt would give (2.16). Correct -> NOT flagged;
        # the no-sqrt mutation -> flagged. Pins the sqrt in the critical value.
        np.array([0.0, 0.3, -0.2, 0.1, -0.1, 0.25, -0.15, 0.05, 0.2, -0.70]),
    ])
    def test_matches_reference(self, arr):
        """Agreement with an independent Grubbs implementation (critical value
        from scipy's t quantile). Catches a wrong significance level, a dropped
        sqrt, or the wrong tail."""
        da = self._bare()
        np.testing.assert_array_equal(da._detect_outliers_grubbs(arr),
                                      self._grubbs_reference(arr))

    def test_flags_obvious_outlier(self):
        """Sanity that the reference isn't vacuously empty: a point far below a
        tight cluster is flagged at its argmin index."""
        da = self._bare()
        arr = np.array([20.0, 20.1, 19.9, 20.05, 19.95, 20.02, -50.0])
        out = da._detect_outliers_grubbs(arr)
        np.testing.assert_array_equal(out, [6])

    def test_too_few_points_returns_empty(self):
        """Grubbs is undefined for N < 3 (the critical-value formula divides by
        N-2); the method must short-circuit to an empty result."""
        da = self._bare()
        assert da._detect_outliers_grubbs(np.array([1.0, 2.0])).size == 0

    def test_zero_variance_returns_empty(self):
        """With all means equal the sample std is 0; the test would divide by
        zero, so the method returns empty rather than flagging."""
        da = self._bare()
        assert da._detect_outliers_grubbs(np.full(5, 7.0)).size == 0


# --------------------------------------------------------------------------- #
# detect_and_reset_outliers: reset a flagged chain to a healthy one
# --------------------------------------------------------------------------- #
class TestResetOutliers:

    def test_outlier_chain_reset_to_good_chain(self):
        """An outlier chain (chain 0, parked at a far-lower ln-posterior) is
        overwritten with a copy of one of the surviving chains: its current
        pset, current ln-posterior, and the last-50% history slice all take the
        donor's values. Oracle: after the reset, chain 0's ln_current_P equals a
        good chain's, and its current_pset is value-equal to that donor's."""
        da = object.__new__(algorithms.DreamAlgorithm)
        da.num_parallel = 6
        da.outlier_method = 'iqr'
        da.iteration = [40] * 6
        # Chain 0 is the outlier: a much lower mean over the (used) second half;
        # the other five cluster tightly so the IQR fence flags only chain 0.
        healthy_means = [5.0, 5.1, 4.9, 5.05, 4.95]
        da.ln_posterior_history = [[-1000.0] * 40] + [[m] * 40 for m in healthy_means]
        da.chain_history = [[np.zeros(3)] * 40] + [[np.ones(3)] * 40 for _ in range(5)]
        da.current_pset = [_normal_pset((0., 0., 0.))] \
            + [_normal_pset((1., 1., 1.)) for _ in range(5)]
        da.ln_current_P = [-1000.0] + healthy_means

        da.rng = np.random.default_rng(0)
        da.detect_and_reset_outliers()

        # Chain 0 was reset to one of the healthy chains (its ln-posterior and
        # current pset now match a donor's, not the outlier's).
        assert da.ln_current_P[0] in healthy_means
        assert da.current_pset[0]['v1__FREE'] == 1.0
        # The healthy chains are untouched.
        assert da.ln_current_P[1:] == healthy_means

    def test_no_reset_when_no_outlier(self):
        """When every chain has a comparable mean ln-posterior, no chain is an
        outlier and the state is left entirely unchanged."""
        da = object.__new__(algorithms.DreamAlgorithm)
        da.num_parallel = 3
        da.outlier_method = 'iqr'
        da.iteration = [40, 40, 40]
        da.ln_posterior_history = [[5.0] * 40, [5.1] * 40, [4.9] * 40]
        da.chain_history = [[np.zeros(3)] * 40 for _ in range(3)]
        original = [-3.0, 5.1, 4.9]
        da.ln_current_P = list(original)
        da.current_pset = [_normal_pset((0., 0., 0.)) for _ in range(3)]
        da.detect_and_reset_outliers()
        assert da.ln_current_P == original


# --------------------------------------------------------------------------- #
# calculate_snooker_pset: snooker proposal + Hastings correction
# --------------------------------------------------------------------------- #
class TestSnookerProposal:

    def _dream_with_archive(self, tmp_path, x0, donors):
        cfg = _make_dream_config(tmp_path, NORMAL_VARS, zeta=0.0, **{'lambda': 0.0})
        da = algorithms.DreamAlgorithm(cfg)
        da.current_pset = [x0]
        da.archive = donors
        return da

    def test_jump_is_parallel_to_reference_axis(self, tmp_path, monkeypatch):
        """The snooker proposal moves the chain along the line through the
        current point x0 and the reference donor zc. With zeta=lambda=0 the jump
        is gamma_s*(za_proj - zb_proj), and both projections lie on that line,
        so (xp - x0) is parallel to (x0 - zc) for any gamma_s. Oracle: the cross
        product of the two vectors is zero. Donor selection is pinned so zc is
        archive[0]."""
        x0 = _normal_pset((0.30, 0.10, -0.20))
        A = _normal_pset((1.00, 0.50, -0.40))     # zc (reference)
        B = _normal_pset((0.40, -0.30, 0.60))     # za
        C = _normal_pset((-0.50, 0.20, 0.10))     # zb
        da = self._dream_with_archive(tmp_path, x0, [A, B, C])
        da.chain_rngs[0] = _PinnedChoiceRng([0, 1, 2])   # pin donor selection

        x0_vec = da._param_vec(x0)
        zc_vec = da._param_vec(A)
        prop, _ = da.calculate_snooker_pset(0)
        jump = da._param_vec(prop) - x0_vec
        np.testing.assert_allclose(np.cross(jump, x0_vec - zc_vec),
                                   np.zeros(3), atol=1e-9)

    def test_hastings_correction_formula(self, tmp_path, monkeypatch):
        """The returned log-correction must equal (d-1)*log(||xp - zc|| /
        ||x0 - zc||) — the snooker Jacobian term (ter Braak & Vrugt 2008).
        Verified against the actually-proposed point. A dropped (d-1) factor or
        an inverted ratio fails."""
        x0 = _normal_pset((0.30, 0.10, -0.20))
        A = _normal_pset((1.00, 0.50, -0.40))
        B = _normal_pset((0.40, -0.30, 0.60))
        C = _normal_pset((-0.50, 0.20, 0.10))
        da = self._dream_with_archive(tmp_path, x0, [A, B, C])
        da.chain_rngs[0] = _PinnedChoiceRng([0, 1, 2])   # pin donor selection

        x0_vec, zc_vec = da._param_vec(x0), da._param_vec(A)
        prop, log_corr = da.calculate_snooker_pset(0)
        xp_vec = da._param_vec(prop)
        expected = (da.n_dim - 1) * np.log(np.linalg.norm(xp_vec - zc_vec)
                                           / np.linalg.norm(x0_vec - zc_vec))
        np.testing.assert_allclose(log_corr, expected, rtol=1e-10)

    def test_degenerate_axis_returns_none(self, tmp_path, monkeypatch):
        """When the reference donor coincides with the current point the snooker
        axis has zero length (axis_norm_sq < 1e-20); the method must bail out
        with (None, 0.0) rather than dividing by zero."""
        x0 = _normal_pset((0.30, 0.10, -0.20))
        zc = _normal_pset((0.30, 0.10, -0.20))     # identical to x0 -> zero axis
        B = _normal_pset((0.40, -0.30, 0.60))
        da = self._dream_with_archive(tmp_path, x0, [zc, B, B])
        da.chain_rngs[0] = _PinnedChoiceRng([0, 1, 2])   # pin donor selection
        prop, log_corr = da.calculate_snooker_pset(0)
        assert prop is None and log_corr == 0.0


# --------------------------------------------------------------------------- #
# calculate_new_pset: the differential-evolution proposal
# --------------------------------------------------------------------------- #
class TestDreamProposal:

    def _dream(self, tmp_path, x0, archive, **overrides):
        cfg = _make_dream_config(tmp_path, NORMAL_VARS, delta=1, zeta=0.0,
                                 **{'lambda': 0.0}, **overrides)
        da = algorithms.DreamAlgorithm(cfg)
        da.current_pset = [x0]
        da.archive = archive
        return da

    def test_mode_jump_reduces_to_de_difference(self, tmp_path):
        """With gamma_prob=1 every proposal is a mode jump: gamma=1, all
        dimensions active, and (with zeta=lambda=0) the move is exactly the sum
        of donor differences. For delta=1 and a 2-entry archive the jump is
        ±(A - B) (sign set by the random donor ordering). Oracle pins gamma=1
        and the difference-vector construction."""
        x0 = _normal_pset((0.1, 0.2, 0.3))
        A = _normal_pset((1.0, 0.5, -0.2))
        B = _normal_pset((0.3, 0.9, 0.4))
        da = self._dream(tmp_path, x0, [A, B], gamma_prob=1.0)
        D = da._param_vec(A) - da._param_vec(B)
        x0_vec = da._param_vec(x0)

        seen_plus = seen_minus = False
        for s in range(40):
            da.chain_rngs[0] = np.random.default_rng(s)
            prop, _ = da.calculate_new_pset(0)
            jump = da._param_vec(prop) - x0_vec
            cp, cm = np.allclose(jump, D, atol=1e-9), np.allclose(jump, -D, atol=1e-9)
            assert cp or cm, jump
            seen_plus |= cp
            seen_minus |= cm
        assert seen_plus and seen_minus

    def test_fixed_step_size_scales_difference(self, tmp_path):
        """With adaptive_step_size off (and gamma_prob=0 so no mode jump) the DE
        gain is the fixed step_size. Forcing the crossover to update all
        dimensions (crossover_number=1 -> cr=1) makes the jump exactly
        ±step_size*(A - B). Pins the non-adaptive gamma = step_size branch."""
        x0 = _normal_pset((0.1, 0.2, 0.3))
        A = _normal_pset((1.0, 0.5, -0.2))
        B = _normal_pset((0.3, 0.9, 0.4))
        step = 0.35
        da = self._dream(tmp_path, x0, [A, B], gamma_prob=0.0,
                         adaptive_step_size=False, step_size=step,
                         crossover_number=1)
        D = step * (da._param_vec(A) - da._param_vec(B))
        x0_vec = da._param_vec(x0)
        for s in range(20):
            da.chain_rngs[0] = np.random.default_rng(s)
            prop, _ = da.calculate_new_pset(0)
            jump = da._param_vec(prop) - x0_vec
            assert np.allclose(jump, D, atol=1e-9) or np.allclose(jump, -D, atol=1e-9)

    def test_out_of_bounds_proposal_rejected(self, tmp_path):
        """A mode jump with a huge donor difference drives the proposal past the
        [0, 10] box, so set_value(reflect=False) raises and calculate_new_pset
        returns (None, cr_idx) — the DE rejection path."""
        cfg = _make_dream_config(tmp_path, UNIFORM_VARS, delta=1, zeta=0.0,
                                 **{'lambda': 0.0}, gamma_prob=1.0)
        da = algorithms.DreamAlgorithm(cfg)
        da.current_pset = [_uniform_pset((5.0, 5.0, 5.0))]
        da.archive = [_uniform_pset((9.9, 9.9, 9.9)), _uniform_pset((0.1, 0.1, 0.1))]
        da.chain_rngs[0] = np.random.default_rng(0)
        prop, cr_idx = da.calculate_new_pset(0)
        assert prop is None
        assert isinstance(cr_idx, (int, np.integer))


# --------------------------------------------------------------------------- #
# Driven run: ZS archive growth and CR-probability adaptation
# --------------------------------------------------------------------------- #
class TestDreamDrivenRun:

    def test_archive_growth_and_cr_adaptation(self, tmp_path):
        """Drive DreamAlgorithm through enough synced generations to exercise the
        run-loop bookkeeping. Oracles:
          * the ZS archive grows in whole-population batches: every
            archive_thin_rate generations it appends one deepcopy per chain, so
            (len(archive) - initial) is a positive multiple of num_parallel;
          * the adapted crossover probabilities remain a valid distribution
            (non-negative, summing to 1) and freeze once past cr_adapt_end."""
        cfg = _make_dream_config(tmp_path, UNIFORM_VARS, population_size=4,
                                 max_iterations=200, burn_in=25,
                                 archive_thin_rate=10)
        da = algorithms.DreamAlgorithm(cfg)
        psets = da.start_run()
        initial_archive = len(da.archive)              # archive_m0 = 10 * n_dim
        assert initial_archive == 10 * da.n_dim

        d1s = data.Data()
        d1s.data = d1s._read_file_lines(
            ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

        score_rng = np.random.default_rng(12345)
        current = psets
        # Run until every chain has passed iteration 20 (two thin events at 10, 20).
        guard = 0
        while min(da.iteration) < 21 and guard < 2000:
            guard += 1
            nxt = None
            for ps in current:
                res = algorithms.Result(ps, d1s, ps.name)
                res.score = float(score_rng.uniform(5, 15))
                out = da.got_result(res)
                if isinstance(out, list) and out:
                    nxt = out
            assert nxt is not None, 'a generation should have synced'
            current = nxt

        grown = len(da.archive) - initial_archive
        assert grown > 0 and grown % da.num_parallel == 0

        assert da.cr_probs.shape == (da.ncr_count,)
        np.testing.assert_allclose(np.sum(da.cr_probs), 1.0, rtol=1e-9)
        assert np.all(da.cr_probs >= 0)
        assert da.cr_frozen is True            # past cr_adapt_end = burn_in // 2
