from .context import data, noise, objective, printing, raises
import numpy as np
import numpy.testing as npt
import pytest
from scipy import stats


def _mkdata(lines):
    """Build a Data object from BNGL-gdat-style header+rows (weights default to ones)."""
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


class _Param:
    """Minimal stand-in for a FreeParameter: evaluate_multiple only reads .name/.value."""

    def __init__(self, name, value):
        self.name = name
        self.value = value


class _FakeConstraintSet:
    """Stand-in for constraint.ConstraintSet exposing the two hooks the objective calls."""

    def __init__(self, penalty=0.0, failed=0):
        self._penalty = penalty
        self._failed = failed

    def total_penalty(self, sim_data_dict):
        return self._penalty

    def number_failed(self, sim_data_dict):
        return self._failed


class TestObjectiveFunctions:
    @classmethod
    def setup_class(cls):

        cls.data1e = [
            '# x    obs1    obs3\n',
            ' 0 3   5\n',
            ' 1 2   6\n',
            ' 2 4   10\n'
        ]
        cls.d1e = data.Data()
        cls.d1e.data = cls.d1e._read_file_lines(cls.data1e, r'\s+')

        cls.data1e_sd = [
            '# x    obs1    obs3  obs1_SD  obs3_SD\n',
            ' 0 3   5   0.1   0.3\n',
            ' 1 2   6   0.1   0.1\n',
            ' 2 4   10  0.3   1.0\n'
        ]
        cls.d1e_sd = data.Data()
        cls.d1e_sd.data = cls.d1e_sd._read_file_lines(cls.data1e_sd, r'\s+')

        cls.data1s = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 0.5 7   8\n',
            ' 1   2   6\n',
            ' 1.5 7   8\n',
            ' 2   4.2   10.2\n'
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, r'\s+')

        cls.data1round = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 1.1   2   6\n',
            ' 2.8   4.2   10.2\n'
        ]
        cls.d1round = data.Data()
        cls.d1round.data = cls.d1round._read_file_lines(cls.data1round, r'\s+')

        cls.data1s_nan = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 0.5 7   8\n',
            ' 1   NaN   6\n',
            ' 1.5 7   8\n',
            ' 2   4.2   10.2\n'
        ]
        cls.d1s_nan = data.Data()
        cls.d1s_nan.data = cls.d1s_nan._read_file_lines(cls.data1s_nan, r'\s+')

        cls.data1s_inf = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 0.5 7   8\n',
            ' 1   2   Inf\n',
            ' 1.5 7   8\n',
            ' 2   4.2   10.2\n'
        ]
        cls.d1s_inf = data.Data()
        cls.d1s_inf.data = cls.d1s_inf._read_file_lines(cls.data1s_inf, r'\s+')

        cls.data1e_extracol = [
            '# x    obs1    obs2    obs3\n',
            ' 0 3 1 5\n',
            ' 1 2 2 6\n',
            ' 2 4 3 10\n'
        ]
        cls.d1e_extracol = data.Data()
        cls.d1e_extracol.data = cls.d1e_extracol._read_file_lines(cls.data1e_extracol, r'\s+')

        cls.data1e_extrarow = [
            '# x    obs1    obs3\n',
            ' 0 3   5\n',
            ' 1 2   6\n',
            ' 2 4   10\n',
            ' 3 6   12\n'
        ]
        cls.d1e_extrarow = data.Data()
        cls.d1e_extrarow.data = cls.d1e_extrarow._read_file_lines(cls.data1e_extrarow, r'\s+')

        cls.chi_sq = objective.ChiSquareObjective()
        cls.sos = objective.SumOfSquaresObjective()
        cls.norm_sos = objective.NormSumOfSquaresObjective()
        cls.ave_norm_sos = objective.AveNormSumOfSquaresObjective()
        cls.kl = objective.KLLikelihood()

    def test_chi_square(self):
        npt.assert_almost_equal(self.chi_sq.evaluate(self.d1s, self.d1e_sd), 0.797777777777778)  # Value computed by hand

    def test_weighted_chi_square(self):
        self.d1e_sd.weights = np.array([[0, 0, 2, 0, 0], [0, 2, 0, 0, 0], [0, 1, 1, 0, 0]])
        chi_sq_eval = self.chi_sq.evaluate(self.d1s, self.d1e_sd)
        npt.assert_almost_equal(chi_sq_eval, 0.1111111 + 0.2222222 + 0.02)
        self.d1e_sd.weights = np.ones(self.d1e.data.shape)

    @raises(printing.PybnfError)
    def test_chi_square_no_sd(self):
        self.chi_sq.evaluate(self.d1s, self.d1s)

    def test_sum_of_squares(self):
        npt.assert_almost_equal(self.sos.evaluate(self.d1s, self.d1e), 0.1)  # Value computed by hand

    def test_norm_sum_of_squares(self):
        npt.assert_almost_equal(self.norm_sos.evaluate(self.d1s, self.d1e), 0.00441111111111)  # Value computed by hand

    def test_ave_norm_sum_of_squares(self):
        # Value computed by hand
        npt.assert_almost_equal(self.ave_norm_sos.evaluate(self.d1s, self.d1e), 0.00657963719, decimal=5)

    def test_obj_nan(self):
        assert self.chi_sq.evaluate(self.d1s_nan, self.d1e_sd) is None
        assert self.sos.evaluate(self.d1s_nan, self.d1e) is None
        assert self.norm_sos.evaluate(self.d1s_nan, self.d1e) is None
        assert self.ave_norm_sos.evaluate(self.d1s_nan, self.d1e) is None

    def test_obj_inf(self):
        assert self.chi_sq.evaluate(self.d1s_inf, self.d1e_sd) is None
        assert self.sos.evaluate(self.d1s_inf, self.d1e) is None
        assert self.norm_sos.evaluate(self.d1s_inf, self.d1e) is None
        assert self.ave_norm_sos.evaluate(self.d1s_inf, self.d1e) is None

    def test_round_ind_var(self):
        obj = objective.ChiSquareObjective(ind_var_rounding=1)
        npt.assert_almost_equal(obj.evaluate(self.d1round, self.d1e_sd), 0.797777777777778)  # Value computed by hand

    def test_kl_positive(self):
        """KL objective should return a non-negative value"""
        # Use d1e as both sim and exp (same shape required for KLLikelihood)
        result = self.kl.evaluate(self.d1e, self.d1e)
        assert result >= 0

    def test_kl_better_fit_lower_score(self):
        """A better fit should produce a lower KL score"""
        # Create "good fit" sim data (close to exp)
        data_good = [
            '# x    obs1    obs3\n',
            ' 0 3.1   5.1\n',
            ' 1 2.1   6.1\n',
            ' 2 4.1   10.1\n'
        ]
        d_good = data.Data()
        d_good.data = d_good._read_file_lines(data_good, r'\s+')

        # Create "bad fit" sim data (far from exp)
        data_bad = [
            '# x    obs1    obs3\n',
            ' 0 100   1\n',
            ' 1 1   100\n',
            ' 2 1   1\n'
        ]
        d_bad = data.Data()
        d_bad.data = d_bad._read_file_lines(data_bad, r'\s+')

        good_score = self.kl.evaluate(d_good, self.d1e)
        bad_score = self.kl.evaluate(d_bad, self.d1e)
        assert good_score < bad_score

    def test_kl_value(self):
        """KL objective should return the correct value (negative cross-entropy, negated)"""
        # exp = [3, 2, 4], sim = [3, 2, 4] => normalized sim = [3/9, 2/9, 4/9]
        # For obs1: -sum([3, 2, 4] * log([3/9, 2/9, 4/9]))
        exp_obs1 = np.array([3., 2., 4.])
        sim_norm = exp_obs1 / exp_obs1.sum()
        expected_obs1 = -np.sum(exp_obs1 * np.log(sim_norm))
        # For obs3: exp = [5, 6, 10], sim = [5, 6, 10]
        exp_obs3 = np.array([5., 6., 10.])
        sim_norm3 = exp_obs3 / exp_obs3.sum()
        expected_obs3 = -np.sum(exp_obs3 * np.log(sim_norm3))
        expected = expected_obs1 + expected_obs3
        npt.assert_almost_equal(self.kl.evaluate(self.d1e, self.d1e), expected)

    @raises(printing.PybnfError)
    def test_unused_col(self):
        self.sos.evaluate(self.d1s, self.d1e_extracol)

    @raises(printing.PybnfError)
    def test_unused_row(self):
        self.sos.evaluate(self.d1s, self.d1e_extrarow)


# ---------------------------------------------------------------------------
# Negative-binomial likelihoods (oracle: scipy.stats.nbinom.logpmf)
# ---------------------------------------------------------------------------

def _nbin_oracle(exp_counts, sim_vals, r):
    """Reference negative-binomial NLL: sum of |logpmf|, matching the code's clip + abs.

    The code computes loggamma(k+r) - loggamma(k+1) - loggamma(r) + r*log(p) + k*log(1-p),
    which is exactly scipy.stats.nbinom.logpmf(k, n=r, p=p) with p = r/(r+sim). It returns
    the absolute value, and contributes 0 for any negative experimental count.
    """
    total = 0.0
    for k, s in zip(exp_counts, sim_vals):
        if k < 0:
            continue
        p = np.clip(r / (r + s), 1e-10, 1 - 1e-10)
        total += abs(stats.nbinom.logpmf(k, r, p))
    return total


class TestNegBinLikelihood:
    """Static-r negative-binomial likelihood: agreement with the scipy nbinom reference."""

    def setup_method(self):
        self.r = 10.0
        # x grid shared by sim and exp so each exp row maps to the same sim row.
        self.exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  5\n', ' 2  8\n'])
        self.sim = _mkdata(['# x  obs1\n', ' 0  2.0\n', ' 1  5.0\n', ' 2  10.0\n'])

    def test_matches_scipy_nbinom(self):
        """Total NLL equals sum_i |nbinom.logpmf(exp_i, r, r/(r+sim_i))| (scipy reference)."""
        obj = objective.NegBinLikelihood(self.r, 0)
        expected = _nbin_oracle([3, 5, 8], [2.0, 5.0, 10.0], self.r)
        npt.assert_almost_equal(obj.evaluate(self.sim, self.exp), expected)

    def test_negative_count_contributes_zero(self):
        """A negative experimental count adds nothing (exp_val >= 0 guard)."""
        obj = objective.NegBinLikelihood(self.r, 0)
        exp_neg = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  -1\n', ' 2  8\n'])
        # The row with exp=-1 drops out, so only rows 0 and 2 contribute.
        expected = _nbin_oracle([3, 8], [2.0, 10.0], self.r)
        npt.assert_almost_equal(obj.evaluate(self.sim, exp_neg), expected)

    def test_perfect_mean_still_positive(self):
        """|logpmf| is a negative-log-likelihood: strictly > 0 even when sim is a plausible mean."""
        obj = objective.NegBinLikelihood(self.r, 0)
        assert obj.evaluate(self.sim, self.exp) > 0.0


class TestNegBinLikelihoodDynamic:
    """Free-r negative-binomial likelihood: r injected via pset, plus the _Cum differencing."""

    def test_r_free_drives_scipy_value(self):
        """r__FREE param sets self.r; evaluate_multiple then matches the scipy reference."""
        obj = objective.NegBinLikelihood_Dynamic()
        sim = _mkdata(['# x  obs1\n', ' 0  2.0\n', ' 1  5.0\n', ' 2  10.0\n'])
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  5\n', ' 2  8\n'])
        sim_dict = {'m': {'s': sim}}
        exp_dict = {'m': {'s': exp}}
        r = 7.0
        # The extra non-special param exercises the "ignore other params" branch.
        result = obj.evaluate_multiple(sim_dict, exp_dict, [_Param('r__FREE', r), _Param('k__FREE', 1.0)])
        expected = _nbin_oracle([3, 5, 8], [2.0, 5.0, 10.0], r)
        npt.assert_almost_equal(result, expected)

    def test_negative_count_contributes_zero(self):
        """A negative experimental count adds nothing (exp_val >= 0 guard)."""
        obj = objective.NegBinLikelihood_Dynamic()
        obj._pset_values = {'r__FREE': 10.0}  # normally built from the pset by evaluate_multiple
        sim = _mkdata(['# x  obs1\n', ' 0  2.0\n', ' 1  5.0\n', ' 2  10.0\n'])
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  -1\n', ' 2  8\n'])
        expected = _nbin_oracle([3, 8], [2.0, 10.0], 10.0)
        npt.assert_almost_equal(obj.evaluate(sim, exp), expected)

    def test_cumulative_column_uses_consecutive_difference(self):
        """For a _Cum column the effective sim value is the row-to-row increment (raw at row 0)."""
        obj = objective.NegBinLikelihood_Dynamic()
        obj._pset_values = {'r__FREE': 10.0}  # normally built from the pset by evaluate_multiple
        # cumulative sim [2, 7, 15] -> increments [2, 5, 8]; row 0 keeps the raw value.
        sim = _mkdata(['# x  obs1_Cum\n', ' 0  2.0\n', ' 1  7.0\n', ' 2  15.0\n'])
        exp = _mkdata(['# x  obs1_Cum\n', ' 0  3\n', ' 1  5\n', ' 2  8\n'])
        expected = _nbin_oracle([3, 5, 8], [2.0, 5.0, 8.0], 10.0)
        npt.assert_almost_equal(obj.evaluate(sim, exp), expected)


class TestNegBinMedianCentering:
    """The negative-binomial median location (issue #419, ADR-0031): the prediction is
    the **continuous** 0.5-quantile, realized by solving for the mean whose continuous
    NB median equals it. Oracle: scipy.stats.nbinom (its CDF is the betainc the solver
    inverts, at integer arguments)."""

    r = 10.0

    def test_default_location_is_median(self):
        """The constructor default is MEDIAN -- median is the universal centering default
        for every family (ADR-0031), true in code like Gaussian/Laplace."""
        assert noise.NegBinomial().location is noise.MEDIAN

    def test_mean_location_is_the_native_identity(self):
        """MEAN data_fit is the identity: the prediction IS the mean (the frozen legacy
        behavior the legacy objfuncs pin explicitly)."""
        nb = noise.NegBinomial(location=noise.MEAN)
        p = np.clip(self.r / (self.r + 4.0), 1e-10, 1 - 1e-10)
        npt.assert_almost_equal(nb.data_fit(4.0, 5, self.r), -stats.nbinom.logpmf(5, self.r, p))

    def test_legacy_objfuncs_pin_mean(self):
        """The legacy neg_bin / neg_bin_dynamic objfuncs stay frozen-mean by pinning
        MEAN explicitly, despite the family's modern median default."""
        assert objective.NegBinLikelihood(self.r).noise.location is noise.MEAN
        assert objective.NegBinLikelihood_Dynamic().noise.location is noise.MEAN

    def test_with_location_round_trips(self):
        nb = noise.NegBinomial()
        assert nb.with_location(noise.MEDIAN).location is noise.MEDIAN
        assert nb.with_location(noise.MEDIAN).with_location(noise.MEAN).location is noise.MEAN

    @pytest.mark.parametrize('prediction', [0, 1, 3, 8, 25, 100])
    def test_solved_mean_places_continuous_median_at_prediction(self, prediction):
        """For an integer prediction the solved mean mu satisfies nbinom.cdf(pred, r, p)
        == 0.5 with p = r/(r+mu) -- i.e. the continuous median sits exactly on the
        prediction (scipy.stats.nbinom oracle for the round-trip)."""
        nb = noise.NegBinomial(location=noise.MEDIAN)
        mu = nb._mean(float(prediction), self.r)
        p = self.r / (self.r + mu)
        npt.assert_almost_equal(stats.nbinom.cdf(prediction, self.r, p), 0.5, decimal=8)

    def test_median_data_fit_matches_scipy_at_solved_mean(self):
        """data_fit under MEDIAN scores NB(mu, r) at the observation, where mu is the
        solved mean -- equal to -nbinom.logpmf(obs, r, r/(r+mu))."""
        nb = noise.NegBinomial(location=noise.MEDIAN)
        pred, obs = 8.0, 5
        mu = nb._mean(pred, self.r)
        p = np.clip(self.r / (self.r + mu), 1e-10, 1 - 1e-10)
        npt.assert_almost_equal(nb.data_fit(pred, obs, self.r), -stats.nbinom.logpmf(obs, self.r, p))

    def test_median_diverges_from_mean(self):
        """The location axis is live: at a moderate count median != mean, so the two
        interpretations give different data fits for the same prediction."""
        pred, obs = 8.0, 5
        mean_fit = noise.NegBinomial(location=noise.MEAN).data_fit(pred, obs, self.r)
        med_fit = noise.NegBinomial(location=noise.MEDIAN).data_fit(pred, obs, self.r)
        assert mean_fit != pytest.approx(med_fit)

    def test_inversion_is_smooth_and_monotone(self):
        """The continuous median (unlike the discrete ppf step) is smooth and strictly
        increasing in the prediction -- what keeps the objective continuous for the
        optimizers."""
        nb = noise.NegBinomial(location=noise.MEDIAN)
        mus = [nb._mean(x, self.r) for x in [3.0, 3.5, 4.0, 4.5, 5.0]]
        assert all(b > a for a, b in zip(mus, mus[1:]))

    def test_negative_observation_contributes_zero_under_median(self):
        """The count-domain guard still drops a negative observation (no solve needed)."""
        assert noise.NegBinomial(location=noise.MEDIAN).data_fit(8.0, -1, self.r) == 0


# ---------------------------------------------------------------------------
# Dynamic chi-square and sum-of-diffs (closed-form values)
# ---------------------------------------------------------------------------

class TestChiSquareDynamic:
    """chi_sq with a free sigma: sum of Delta^2/(2 sigma^2) + log(sigma) per point."""

    def setup_method(self):
        self.sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.1  5.1\n', ' 1  2.0  6.0\n', ' 2  4.2  10.2\n'])
        self.exp = _mkdata(['# x  obs1  obs3\n', ' 0  3    5\n', ' 1  2    6\n', ' 2  4    10\n'])

    def test_closed_form_with_sigma_free(self):
        """sigma__FREE sets self.sigma; total = sum[ Delta^2/(2 sigma^2) + log sigma ] over all points."""
        obj = objective.ChiSquareObjective_Dynamic()
        sigma = 2.0
        deltas = [0.1, 0.1, 0.0, 0.0, 0.2, 0.2]  # obs1 then obs3 residuals across 3 rows
        expected = sum(d ** 2 / (2 * sigma ** 2) + np.log(sigma) for d in deltas)
        result = obj.evaluate_multiple(
            {'m': {'s': self.sim}}, {'m': {'s': self.exp}}, [_Param('sigma__FREE', sigma)])
        npt.assert_almost_equal(result, expected)

    def test_log_sigma_term_present(self):
        """The +log(sigma) term means a perfect fit is non-zero and grows with #points."""
        obj = objective.ChiSquareObjective_Dynamic()
        obj._pset_values = {'sigma__FREE': 3.0}  # normally built from the pset by evaluate_multiple
        # sim == exp -> all residuals zero, leaving only 6 * log(sigma).
        npt.assert_almost_equal(obj.evaluate(self.exp, self.exp), 6 * np.log(3.0))


class TestSumOfDiffs:
    def test_sum_of_absolute_diffs(self):
        """SumOfDiffs returns sum|sim-exp|: 0.1+0+0.2 (obs1) + 0.1+0+0.2 (obs3) = 0.6."""
        obj = objective.SumOfDiffsObjective()
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.1  5.1\n', ' 1  2.0  6.0\n', ' 2  4.2  10.2\n'])
        exp = _mkdata(['# x  obs1  obs3\n', ' 0  3    5\n', ' 1  2    6\n', ' 2  4    10\n'])
        npt.assert_almost_equal(obj.evaluate(sim, exp), 0.6)

    def test_zero_iff_identical(self):
        """SumOfDiffs == 0 exactly when sim == exp."""
        obj = objective.SumOfDiffsObjective()
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        assert obj.evaluate(exp, exp) == 0.0


# ---------------------------------------------------------------------------
# Invariants strengthening the already-pinned chi-square / KL value tests
# ---------------------------------------------------------------------------

class TestChiSquareInvariants:
    def test_variance_rescale_divides_by_c_squared(self):
        """Scaling every SD by c divides chi-square by c^2 exactly (the 1/(2 sigma^2) weight)."""
        sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.5\n', ' 2  4.2\n'])
        exp1 = _mkdata(['# x  obs1  obs1_SD\n', ' 0  3  0.1\n', ' 1  2  0.2\n', ' 2  4  0.5\n'])
        exp2 = _mkdata(['# x  obs1  obs1_SD\n', ' 0  3  0.3\n', ' 1  2  0.6\n', ' 2  4  1.5\n'])  # SD x3
        obj = objective.ChiSquareObjective()
        chi1 = obj.evaluate(sim, exp1)
        chi2 = obj.evaluate(sim, exp2)
        npt.assert_almost_equal(chi2, chi1 / 9.0)  # c = 3 -> 1/c^2 = 1/9

    def test_equals_sum_of_squares_when_two_sigma_sq_is_one(self):
        """With sigma = 1/sqrt(2) every weight is 1, so chi-square reduces to plain SoS."""
        sigma = 1.0 / np.sqrt(2.0)
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.1  5.1\n', ' 1  2.0  6.0\n', ' 2  4.2  10.2\n'])
        exp_sd = _mkdata(['# x  obs1  obs3  obs1_SD  obs3_SD\n',
                          ' 0  3  5  %g  %g\n' % (sigma, sigma),
                          ' 1  2  6  %g  %g\n' % (sigma, sigma),
                          ' 2  4  10  %g  %g\n' % (sigma, sigma)])
        exp_plain = _mkdata(['# x  obs1  obs3\n', ' 0  3  5\n', ' 1  2  6\n', ' 2  4  10\n'])
        chi = objective.ChiSquareObjective().evaluate(sim, exp_sd)
        sos = objective.SumOfSquaresObjective().evaluate(sim, exp_plain)
        npt.assert_almost_equal(chi, sos)


# ---------------------------------------------------------------------------
# Lognormal noise and the location/scale axes (ADR-0011 seam proof)
# ---------------------------------------------------------------------------

class TestLogNormalNoise:
    """Lognormal observation noise = the Gaussian family reconfigured onto the
    log10 scale with the prediction as the median (ADR-0022). The natural-log
    density (the scipy.stats.lognorm oracle) is the LN scale."""

    def setup_method(self):
        # All-positive sim/exp (the lognormal support); shared x grid.
        self.sim = _mkdata(['# x  obs1\n', ' 0  2.0\n', ' 1  5.0\n', ' 2  9.0\n'])
        self.exp_sd = _mkdata(['# x  obs1  obs1_SD\n',
                               ' 0  3  0.5\n', ' 1  5  0.5\n', ' 2  8  0.5\n'])

    def test_objfunc_is_log10_space_chi_square(self):
        """The lognormal objfunc sums (log10 sim - log10 exp)^2 / (2 sigma^2) --
        chi_sq in log10 space (sigma a log10-scale std fixed from _SD, so the
        normalizer and Jacobian drop; ADR-0022)."""
        obj = objective.LogNormalObjective()
        expected = sum((np.log10(s) - np.log10(e)) ** 2 / (2 * sd ** 2)
                       for s, e, sd in [(2.0, 3, 0.5), (5.0, 5, 0.5), (9.0, 8, 0.5)])
        npt.assert_almost_equal(obj.evaluate(self.sim, self.exp_sd), expected)

    def test_family_nll_matches_scipy_lognorm(self):
        """Gaussian(LN, MEDIAN).nll plus the dropped Jacobian + constant equals the
        full natural-log lognormal -logpdf (scipy oracle); median -> scipy scale =
        prediction. scipy.stats.lognorm is defined in natural log, so this exercises
        the LN scale (ADR-0022)."""
        g = noise.Gaussian(additive_on=noise.LN, location=noise.MEDIAN)
        pred, obs, sigma = 9.0, 8.0, 0.5
        full_nll = g.nll(pred, obs, sigma) + np.log(obs) + 0.5 * np.log(2 * np.pi)
        npt.assert_almost_equal(full_nll, -stats.lognorm.logpdf(obs, s=sigma, scale=pred))

    def test_perfect_prediction_is_zero(self):
        """sim == exp -> zero log-residual at every point."""
        obj = objective.LogNormalObjective()
        exp_only = _mkdata(['# x  obs1  obs1_SD\n', ' 0  3  0.5\n', ' 1  5  0.5\n', ' 2  8  0.5\n'])
        sim_match = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  5\n', ' 2  8\n'])
        assert obj.evaluate(sim_match, exp_only) == 0.0


class TestNoiseAxes:
    """The location and additive-noise-scale axes are real and orthogonal
    (ADR-0011): location only bites for asymmetric (log-scale) noise."""

    def test_location_is_a_no_op_on_the_linear_scale(self):
        """On the linear scale the Gaussian is symmetric, so mean and median give an
        identical data fit -- the location axis is trivial (why chi_sq is mean=median)."""
        mean = noise.Gaussian(additive_on=noise.LINEAR, location=noise.MEAN)
        median = noise.Gaussian(additive_on=noise.LINEAR, location=noise.MEDIAN)
        npt.assert_almost_equal(mean.data_fit(3.1, 3.0, 0.5), median.data_fit(3.1, 3.0, 0.5))

    def test_location_diverges_on_the_log10_scale(self):
        """On the log10 scale the lognormal is asymmetric: prediction-as-mean shifts
        the location parameter by ln10*sigma^2/2 from prediction-as-median, so the
        data fits differ -- the axis is live (ADR-0022)."""
        sigma = 0.5
        ln10 = np.log(10.0)
        mean = noise.Gaussian(additive_on=noise.LOG10, location=noise.MEAN)
        median = noise.Gaussian(additive_on=noise.LOG10, location=noise.MEDIAN)
        mean_fit = mean.data_fit(9.0, 8.0, sigma)
        assert mean_fit != median.data_fit(9.0, 8.0, sigma)
        # median: mu = log10(pred); mean: mu = log10(pred) - ln10*sigma^2/2
        expected_mean = (np.log10(9.0) - ln10 * sigma ** 2 / 2 - np.log10(8.0)) ** 2 / (2 * sigma ** 2)
        npt.assert_almost_equal(mean_fit, expected_mean)

    def test_ln_mean_offset_is_half_sigma_squared(self):
        """Natural-log (LN) mean interpretation subtracts sigma^2/2 -- the lognormal
        moment correction in natural-log space (ADR-0022)."""
        sigma = 0.5
        mean = noise.Gaussian(additive_on=noise.LN, location=noise.MEAN)
        expected = (np.log(9.0) - sigma ** 2 / 2 - np.log(8.0)) ** 2 / (2 * sigma ** 2)
        npt.assert_almost_equal(mean.data_fit(9.0, 8.0, sigma), expected)

    def test_gaussian_default_is_linear_median(self):
        """Gaussian() defaults to additive-on-linear, location-median (ADR-0031) --
        and since location is trivial on the linear scale, chi_sq's delegation is
        still byte-identical to the pre-refactor squared residual."""
        g = noise.Gaussian()
        npt.assert_almost_equal(g.data_fit(3.1, 3.0, 0.5), (3.1 - 3.0) ** 2 / (2 * 0.5 ** 2))


class TestLaplaceNoise:
    """Laplace observation noise (ADR-0021). Oracle: scipy.stats.laplace."""

    def test_kernel_matches_scipy_laplace(self):
        """data_fit = |pred - obs| / b, log_normalizer = log(2b); nll = -logpdf."""
        lap = noise.Laplace()
        pred, obs, b = 9.0, 8.0, 0.5
        npt.assert_almost_equal(lap.data_fit(pred, obs, b), abs(pred - obs) / b)
        npt.assert_almost_equal(lap.log_normalizer(b), np.log(2 * b))
        npt.assert_almost_equal(lap.nll(pred, obs, b), -stats.laplace.logpdf(obs, loc=pred, scale=b))

    def test_objfunc_sums_full_nll_with_free_scale(self):
        """The laplace objfunc fits b (b__FREE), so it keeps the log(2b) normalizer:
        total = sum |sim - exp| / b + log(2b) over points."""
        obj = objective.LaplaceObjective()
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.1  5.1\n', ' 1  2.0  6.0\n', ' 2  4.2  10.2\n'])
        exp = _mkdata(['# x  obs1  obs3\n', ' 0  3    5\n', ' 1  2    6\n', ' 2  4    10\n'])
        b = 2.0
        deltas = [0.1, 0.1, 0.0, 0.0, 0.2, 0.2]  # obs1 then obs3 residuals across 3 rows
        expected = sum(abs(d) / b + np.log(2 * b) for d in deltas)
        result = obj.evaluate_multiple({'m': {'s': sim}}, {'m': {'s': exp}}, [_Param('b__FREE', b)])
        npt.assert_almost_equal(result, expected)

    def test_free_scale_normalizer_penalizes_large_b(self):
        """A perfect fit is not zero: it leaves sum log(2b), which grows with b --
        the term that stops the fit driving b -> inf."""
        obj = objective.LaplaceObjective()
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        for b in (0.5, 2.0):
            result = obj.evaluate_multiple({'m': {'s': exp}}, {'m': {'s': exp}}, [_Param('b__FREE', b)])
            npt.assert_almost_equal(result, 3 * np.log(2 * b))


class TestDecoupledDefaults:
    """Each legacy likelihood objfunc == an exact decoupled (family, sigma-source)
    default, and the sigma-source's estimated-ness governs the normalizer
    (ADR-0021). This pins the strict-superset backward-compatibility contract."""

    # (objfunc instance, expected family class, expected sigma-source class, estimated)
    CASES = [
        (objective.ChiSquareObjective(), noise.Gaussian, noise.DataColumnSigma, False),
        (objective.LogNormalObjective(), noise.Gaussian, noise.DataColumnSigma, False),
        (objective.ChiSquareObjective_Dynamic(), noise.Gaussian, noise.FreeParameterSigma, True),
        (objective.LaplaceObjective(), noise.Laplace, noise.FreeParameterSigma, True),
        (objective.NegBinLikelihood(10.0, 0), noise.NegBinomial, noise.ConstantSigma, False),
        (objective.NegBinLikelihood_Dynamic(), noise.NegBinomial, noise.FreeParameterSigma, True),
    ]

    @pytest.mark.parametrize('obj,family,source,estimated', CASES)
    def test_default_spec(self, obj, family, source, estimated):
        assert isinstance(obj.noise, family)
        assert isinstance(obj.sigma_source, source)
        assert obj.sigma_source.estimated is estimated

    def test_estimated_source_keeps_normalizer_fixed_drops_it(self):
        """Same Gaussian family, only the source differs: a free sigma (estimated)
        adds exactly sum(log sigma) over the fixed-sigma chi_sq data fit."""
        sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        exp_sd = _mkdata(['# x  obs1  obs1_SD\n', ' 0  3  2.0\n', ' 1  2  2.0\n', ' 2  4  2.0\n'])
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        fixed = objective.ChiSquareObjective().evaluate(sim, exp_sd)             # data_fit only
        free = objective.ChiSquareObjective_Dynamic().evaluate_multiple(
            {'m': {'s': sim}}, {'m': {'s': exp}}, [_Param('sigma__FREE', 2.0)])   # + normalizer
        npt.assert_almost_equal(free - fixed, 3 * np.log(2.0))


class TestPerObservableNoise:
    """A likelihood objective carries a per-observable {col: (family, source)}
    override map, defaulting to the global objfunc's spec (ADR-0021)."""

    def setup_method(self):
        self.sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.1  5.1\n', ' 1  2.0  6.0\n', ' 2  4.2  10.2\n'])
        # obs1 carries an _SD column (Gaussian default); obs3 is overridden to Laplace.
        self.exp = _mkdata(['# x  obs1  obs1_SD  obs3\n',
                            ' 0  3  0.1  5\n', ' 1  2  0.1  6\n', ' 2  4  0.3  10\n'])

    def test_override_mixes_families_per_column(self):
        """chi_sq on obs1 (Gaussian x _SD), Laplace x free b on obs3 -- the total is
        the sum of the two columns' own specs."""
        overrides = {'obs3': (noise.Laplace(), noise.FreeParameterSigma('b__FREE'))}
        obj = objective.ChiSquareObjective(overrides=overrides)
        result = obj.evaluate_multiple({'m': {'s': self.sim}}, {'m': {'s': self.exp}}, [_Param('b__FREE', 2.0)])
        obs1 = sum((s - e) ** 2 / (2 * sd ** 2) for s, e, sd in [(3.1, 3, 0.1), (2.0, 2, 0.1), (4.2, 4, 0.3)])
        obs3 = sum(abs(s - e) / 2.0 + np.log(2 * 2.0) for s, e in [(5.1, 5), (6.0, 6), (10.2, 10)])
        npt.assert_almost_equal(result, obs1 + obs3)

    def test_empty_override_is_byte_identical_to_default(self):
        """An empty override map reproduces the plain objfunc exactly."""
        sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        exp_sd = _mkdata(['# x  obs1  obs1_SD\n', ' 0  3  0.1\n', ' 1  2  0.1\n', ' 2  4  0.3\n'])
        plain = objective.ChiSquareObjective().evaluate(sim, exp_sd)
        empty = objective.ChiSquareObjective(overrides={}).evaluate(sim, exp_sd)
        npt.assert_array_equal(plain, empty)

    def test_required_free_noise_params_unions_default_and_overrides(self):
        overrides = {'obs3': (noise.Laplace(), noise.FreeParameterSigma('b_obs3__FREE'))}
        obj = objective.ChiSquareObjective_Dynamic(overrides=overrides)
        assert obj.required_free_noise_params() == {'sigma__FREE', 'b_obs3__FREE'}


class TestKLInvariants:
    def setup_method(self):
        self.exp = _mkdata(['# x  obs1  obs3\n', ' 0  3  5\n', ' 1  2  6\n', ' 2  4  10\n'])
        self.kl = objective.KLLikelihood()

    def test_scale_invariance(self):
        """KL depends only on the normalized sim profile: scaling all sim by c leaves it unchanged."""
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.0  6.0\n', ' 1  2.0  5.0\n', ' 2  5.0  9.0\n'])
        sim_scaled = _mkdata(['# x  obs1  obs3\n', ' 0  21.0  42.0\n', ' 1  14.0  35.0\n', ' 2  35.0  63.0\n'])  # x7
        npt.assert_almost_equal(self.kl.evaluate(sim, self.exp), self.kl.evaluate(sim_scaled, self.exp))

    def test_gibbs_gap_is_M_times_kl_divergence(self):
        """objective(sim) - objective(exp) = sum_cols sum_i e_i log(p_i/q_i) >= 0 (Gibbs; KL reference)."""
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.0  6.0\n', ' 1  2.0  5.0\n', ' 2  5.0  9.0\n'])
        gap = self.kl.evaluate(sim, self.exp) - self.kl.evaluate(self.exp, self.exp)
        expected = 0.0
        for col in ('obs1', 'obs3'):
            e = self.exp[col]
            s = sim[col]
            p = e / e.sum()
            q = s / s.sum()
            expected += np.sum(e * np.log(p / q))
        npt.assert_almost_equal(gap, expected)
        assert gap >= 0.0  # Gibbs' inequality: cross-entropy minimized at sim proportional to exp

    def test_zero_sum_column_is_inf_not_nan(self):
        """CQ-7: an all-zero sim column can't be normalized -> inf, never nan
        (a nan would silently corrupt the minimization, since nan compares False)."""
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  0  6.0\n', ' 1  0  5.0\n', ' 2  0  9.0\n'])
        val = self.kl.evaluate(sim, self.exp)
        assert np.isinf(val) and val > 0
        assert not np.isnan(val)

    def test_negative_sim_column_is_inf_not_nan(self):
        """CQ-7: a negative sim entry makes log(sim) ill-defined -> inf, not nan."""
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.0  6.0\n', ' 1  -2.0  5.0\n', ' 2  5.0  9.0\n'])
        val = self.kl.evaluate(sim, self.exp)
        assert np.isinf(val) and val > 0

    def test_zero_entry_gives_finite_penalty_not_neg_inf(self):
        """CQ-7: a single zero-mass entry (column sum still positive) yields a
        large finite penalty via the 1e-10 floor, not log(0) == -inf."""
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  0.0  6.0\n', ' 1  2.0  5.0\n', ' 2  5.0  9.0\n'])
        val = self.kl.evaluate(sim, self.exp)
        assert np.isfinite(val)

    def test_well_behaved_value_unchanged_by_guard(self):
        """CQ-7: for strictly-positive sim the guard/floor is a no-op -- score is
        bit-identical to the original -sum(exp * log(sim / sum(sim)))."""
        sim = _mkdata(['# x  obs1  obs3\n', ' 0  3.0  6.0\n', ' 1  2.0  5.0\n', ' 2  5.0  9.0\n'])
        expected = 0.0
        for col in ('obs1', 'obs3'):
            s = sim[col]
            expected += -np.sum(self.exp[col] * np.log(s / s.sum()))
        npt.assert_array_equal(self.kl.evaluate(sim, self.exp), expected)


# ---------------------------------------------------------------------------
# evaluate_multiple: the shared dataflow over models/suffixes/constraints
# ---------------------------------------------------------------------------

class TestEvaluateMultiple:
    def setup_method(self):
        self.obj = objective.SumOfSquaresObjective()
        self.simA = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        self.expA = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        self.simB = _mkdata(['# x  obs1\n', ' 0  5.0\n', ' 1  6.0\n', ' 2  10.5\n'])
        self.expB = _mkdata(['# x  obs1\n', ' 0  5\n', ' 1  6\n', ' 2  10\n'])

    def test_empty_sim_returns_inf(self):
        """No simulation data -> objective is +inf."""
        assert self.obj.evaluate_multiple({}, {'m': {'s': self.expA}}, []) == np.inf

    def test_additive_over_suffixes(self):
        """Total over two suffixes equals the sum of the per-suffix evaluations."""
        sim_dict = {'m': {'s1': self.simA, 's2': self.simB}}
        exp_dict = {'m': {'s1': self.expA, 's2': self.expB}}
        total = self.obj.evaluate_multiple(sim_dict, exp_dict, [])
        expected = self.obj.evaluate(self.simA, self.expA) + self.obj.evaluate(self.simB, self.expB)
        npt.assert_almost_equal(total, expected)

    def test_suffix_without_exp_data_skipped(self):
        """A simulated suffix with no matching experimental data is ignored, not an error."""
        sim_dict = {'m': {'s1': self.simA, 's2': self.simB}}
        exp_dict = {'m': {'s1': self.expA}}  # no 's2'
        total = self.obj.evaluate_multiple(sim_dict, exp_dict, [])
        npt.assert_almost_equal(total, self.obj.evaluate(self.simA, self.expA))

    def test_none_propagates(self):
        """If a per-suffix evaluate returns None (nan/inf sim), the total is None."""
        sim_nan = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  NaN\n', ' 2  4.2\n'])
        result = self.obj.evaluate_multiple({'m': {'s': sim_nan}}, {'m': {'s': self.expA}}, [])
        assert result is None

    def test_constraint_penalty_added(self):
        """Constraint penalties passed via the constraints arg add to the data objective."""
        base = self.obj.evaluate_multiple({'m': {'s': self.simA}}, {'m': {'s': self.expA}}, [])
        total = self.obj.evaluate_multiple(
            {'m': {'s': self.simA}}, {'m': {'s': self.expA}}, [], constraints=[_FakeConstraintSet(penalty=2.5)])
        npt.assert_almost_equal(total, base + 2.5)

    def test_constraints_passed_as_pset_legacy_path(self):
        """Objects lacking .name passed as pset trigger the AttributeError branch -> treated as constraints."""
        base = self.obj.evaluate_multiple({'m': {'s': self.simA}}, {'m': {'s': self.expA}}, [])
        total = self.obj.evaluate_multiple(
            {'m': {'s': self.simA}}, {'m': {'s': self.expA}}, (_FakeConstraintSet(penalty=1.0),))
        npt.assert_almost_equal(total, base + 1.0)


# ---------------------------------------------------------------------------
# DirectPass, ConstraintCounter, and abstract-method guards
# ---------------------------------------------------------------------------

class TestObjectiveCalculator:
    def test_delegates_to_objective(self):
        """ObjectiveCalculator.evaluate_objective forwards to objective.evaluate_multiple."""
        sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        obj = objective.SumOfSquaresObjective()
        exp_dict = {'m': {'s': exp}}
        calc = objective.ObjectiveCalculator(obj, exp_dict, ())
        result = calc.evaluate_objective({'m': {'s': sim}}, [])
        expected = obj.evaluate_multiple({'m': {'s': sim}}, exp_dict, [], ())
        npt.assert_almost_equal(result, expected)


class TestDirectPassObjective:
    def test_passes_score_through(self):
        """Returns the single 'score' value from the simulated data verbatim."""
        obj = objective.DirectPassObjective()
        sim = _mkdata(['# score\n', ' 0.728\n'])
        assert obj.evaluate(sim, None) == 0.728

    @raises(printing.PybnfError)
    def test_missing_score_column_raises(self):
        obj = objective.DirectPassObjective()
        sim = _mkdata(['# x  obs1\n', ' 0  3\n'])
        obj.evaluate(sim, None)


class TestConstraintCounter:
    def test_counts_failed_constraints(self):
        """evaluate_multiple sums number_failed over all constraint sets, ignoring exp data."""
        obj = objective.ConstraintCounter()
        total = obj.evaluate_multiple(
            {'m': {'s': _mkdata(['# x  obs1\n', ' 0  3\n'])}}, {},
            [_FakeConstraintSet(failed=2), _FakeConstraintSet(failed=1)])
        assert total == 3

    @raises(NotImplementedError)
    def test_evaluate_not_implemented(self):
        objective.ConstraintCounter().evaluate(None, None)


class TestAbstractGuards:
    @raises(NotImplementedError)
    def test_base_evaluate(self):
        objective.ObjectiveFunction().evaluate(None, None)

    @raises(NotImplementedError)
    def test_summation_eval_point(self):
        objective.SummationObjective().eval_point(None, None, 0, 0, 'obs1')

    @raises(NotImplementedError)
    def test_column_summation_eval_column(self):
        objective.ColumnSummationObjective().eval_column(None, None, 'obs1')


# ---------------------------------------------------------------------------
# Remaining error / branch paths in the summation machinery
# ---------------------------------------------------------------------------

class TestSummationBranches:
    def setup_method(self):
        self.sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        self.exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])

    @raises(printing.PybnfError)
    def test_invalid_rounding_value(self):
        objective.SumOfSquaresObjective(ind_var_rounding=2).evaluate(self.sim, self.exp)

    def test_exp_nan_point_skipped(self):
        """A NaN in the experimental data skips that point rather than poisoning the sum."""
        exp_nan = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  NaN\n', ' 2  4\n'])
        obj = objective.SumOfSquaresObjective()
        # Only rows 0 and 2 contribute: 0.1^2 + 0.2^2.
        npt.assert_almost_equal(obj.evaluate(self.sim, exp_nan), 0.1 ** 2 + 0.2 ** 2)

    def test_missing_indvar_in_sim_raises(self):
        """If the sim lacks the exp independent variable, the summation objective errors out."""
        sim_no_x = _mkdata(['# t  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        obj = objective.SumOfSquaresObjective()
        with pytest.raises(printing.PybnfError):
            obj.evaluate(sim_no_x, self.exp, show_warnings=False)

    def test_rounding_uses_nearest_row_with_warning(self):
        """ind_var_rounding=1 maps each exp point to the nearest sim row; the value uses that row."""
        # exp x=8 has no nearby sim point (nearest is x=10, diff=2 > 1 and 2/8 > 0.1) -> warning branch.
        sim = _mkdata(['# x  obs1\n', ' 0   3.5\n', ' 5   2.5\n', ' 10  4.5\n'])
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 8  2\n', ' 10  4\n'])
        obj = objective.SumOfSquaresObjective(ind_var_rounding=1)
        # exp x=0->sim row0(3.5), x=8->nearest sim x=10(4.5), x=10->sim x=10(4.5).
        expected = (3.5 - 3) ** 2 + (4.5 - 2) ** 2 + (4.5 - 4) ** 2
        npt.assert_almost_equal(obj.evaluate(sim, exp), expected)


class TestCheckColumns:
    def test_chi_square_missing_column_raises(self):
        """chi-square errors when an exp observable (not an _SD column) is absent from the sim."""
        sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
        exp = _mkdata(['# x  obs1  obs2  obs1_SD\n',
                       ' 0  3  9  0.1\n', ' 1  2  9  0.1\n', ' 2  4  9  0.1\n'])
        with pytest.raises(printing.PybnfError):
            objective.ChiSquareObjective().evaluate(sim, exp)

    def test_kl_missing_column_raises(self):
        """KL (column-summation) errors when an exp column is absent from the sim."""
        sim = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        exp = _mkdata(['# x  obs1  obs2\n', ' 0  3  9\n', ' 1  2  9\n', ' 2  4  9\n'])
        with pytest.raises(printing.PybnfError):
            objective.KLLikelihood().evaluate(sim, exp)

    def test_kl_missing_indvar_in_sim_raises(self):
        """KL errors (via the indvar KeyError) when the sim lacks the exp independent variable."""
        sim_no_x = _mkdata(['# t  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        exp = _mkdata(['# x  obs1\n', ' 0  3\n', ' 1  2\n', ' 2  4\n'])
        with pytest.raises(printing.PybnfError):
            objective.KLLikelihood().evaluate(sim_no_x, exp, show_warnings=False)
