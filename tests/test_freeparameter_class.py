from .context import pset, raises

import numpy as np
import pytest
from scipy import stats

from pybnf.printing import PybnfError

# Shared Generator for the statistical sampling tests (reused across draws).
_RNG = np.random.default_rng(0)
from hypothesis import given, strategies as st


def _truncnorm(loc, sigma, lo, hi):
    return stats.truncnorm((lo - loc) / sigma, (hi - loc) / sigma, loc=loc, scale=sigma)


def _fold_reference(new, lb, ub):
    """Independent triangle-wave fold of `new` into [lb, ub] (the reflection map)."""
    w = ub - lb
    q = (new - lb) % (2.0 * w)
    return lb + q if q <= w else ub - (q - w)


class TestFreeParameter:
    @classmethod
    def setup_class(cls):
        cls.p0 = pset.FreeParameter('var0__FREE', 'normal_var', 0, 1)
        cls.p1 = pset.FreeParameter('var1__FREE', 'lognormal_var', 1, 2)
        cls.p2 = pset.FreeParameter('var2__FREE', 'loguniform_var', 0.01, 100)
        cls.p3 = pset.FreeParameter('var2__FREE', 'uniform_var', 0, 10)
        cls.p4 = pset.FreeParameter('var2__FREE', 'uniform_var', 0, 10, bounded=False)

    @classmethod
    def teardown_class(cls):
        pass

    def test_check_init(self):
        print(self.p0.value)
        assert self.p0.value is None
        assert self.p0.type == 'normal_var'
        assert not self.p0.bounded

        assert not self.p1.bounded
        assert self.p1.lower_bound == -np.inf
        assert np.isinf(self.p1.upper_bound)

        assert self.p2.upper_bound == 100

        assert self.p3.bounded
        print(self.p4.bounded)
        assert not self.p4.bounded

    @raises(pset.OutOfBoundsException)
    def test_check_erroneous_assignment(self):
        pset.FreeParameter('var2__FREE', 'loguniform_var', 0.01, 100, value=1000)

    def test_distribution(self):
        xs = [self.p3.sample_value(_RNG).value for x in range(100000)]
        for x in xs:
            assert self.p3.lower_bound <= x < self.p3.upper_bound
        ys = [self.p0.sample_value(_RNG).value for x in range(100000)]
        assert np.any(np.array(ys) < 0.0)  # normal_var centered at 0 should produce negative values

    def test_sample_value(self):
        p0s = self.p0.sample_value(_RNG)
        assert p0s.value is not None

    def test_freeparameter_equality(self):
        p6 = self.p0.sample_value(_RNG)
        p0s = self.p0.set_value(p6.value)
        print(p0s, p6)
        assert p6 == p0s

    def test_add(self):
        p7 = self.p0.set_value(1)
        p7a = p7.add(1)
        assert p7a.value == 2
        p8 = self.p2.set_value(1)
        p8a = p8.add(1)
        assert p8a.value == 10

    def test_diff(self):
        p9 = self.p0.set_value(1)
        p10 = self.p0.set_value(2)
        assert p9.diff(p10) == -1

        p11 = self.p2.set_value(10)
        p12 = self.p2.set_value(100)
        assert p12.diff(p11) == 1

    def test_reflect(self):
        assert self.p3.set_value(11).value == 9
        assert self.p3.set_value(12).value == 8
        assert self.p3.set_value(25).value == 5
        assert self.p2.set_value(1000).value == 10

    def test_set_value(self):
        p13 = self.p0.set_value(1)
        assert p13.lower_bound == self.p0.lower_bound
        assert p13.upper_bound == self.p0.upper_bound
        p14 = self.p4.set_value(100)
        assert p14.lower_bound == self.p4.lower_bound
        assert p14.upper_bound == self.p4.upper_bound

    @raises(pset.OutOfBoundsException)
    def test_no_reflect(self):
        self.p3.set_value(11, False)


class TestReflectFold:
    """The boundary reflection (FreeParameter._reflect) is the triangle-wave fold
    of the proposed value into the box. These pin the closed-form behavior and,
    in particular, that a step large enough to formerly exceed the 1000-reflection
    cap is now folded deterministically rather than replaced by a random value
    (which would have broken Metropolis detailed balance)."""

    def test_matches_existing_oracles(self):
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0, 10)
        assert p.set_value(11).value == 9
        assert p.set_value(25).value == 5

    def test_in_bounds_value_unchanged(self):
        """A value already inside the box is returned untouched (no reflection)."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0, 10)
        assert p.set_value(3.7).value == 3.7

    @pytest.mark.parametrize("new", [10.5, 19.0, 20.0, 21.0, -1.0, -11.0, 100.3])
    def test_triangle_wave_fold(self, new):
        """The reflected value equals the closed-form triangle-wave fold."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0, 10)
        np.testing.assert_allclose(p.set_value(float(new)).value,
                                   _fold_reference(new, 0, 10), atol=1e-9)

    def test_large_step_is_deterministic_and_in_bounds(self):
        """A step needing >1000 reflections (here ~50000) used to fall back to a
        random value; now it folds deterministically. Oracle: repeated calls
        agree, the result stays in the box, and it matches the closed-form fold."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0, 10)
        results = {p.set_value(1000000.7).value for _ in range(5)}
        assert len(results) == 1                      # deterministic, not random
        v = results.pop()
        assert 0.0 <= v <= 10.0
        np.testing.assert_allclose(v, _fold_reference(1000000.7, 0, 10), atol=1e-9)

    def test_log_space_reflection(self):
        """loguniform parameters reflect in log10 space: 1000 -> log10 = 3, folded
        into [log10(0.01), log10(100)] = [-2, 2] gives 1, i.e. 10."""
        p = pset.FreeParameter('x__FREE', 'loguniform_var', 0.01, 100)
        np.testing.assert_allclose(p.set_value(1000).value, 10.0, rtol=1e-12)

    @given(new=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_fold_always_lands_in_bounds(self, new):
        """For any finite proposal the reflected value lies within [lb, ub]."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', -3, 7)
        v = p.set_value(new).value
        assert -3.0 <= v <= 7.0


class TestSamplingSpaceTransform:
    """The public θ↔u transform pair (FreeParameter.to_sampling_space /
    from_sampling_space) the algorithm layer asks for instead of inlining
    np.log10 / 10** (#412). Linear is the identity; Log10 is base-10 log, and the
    inverse is the unguarded 10.0**u that matches the proposal arithmetic."""

    def setup_method(self):
        self.lin = pset.FreeParameter('x__FREE', 'normal_var', 0, 1)            # Linear
        self.log = pset.FreeParameter('x__FREE', 'loguniform_var', 0.01, 100)   # Log10

    def test_linear_is_identity(self):
        assert self.lin.to_sampling_space(3.7) == 3.7
        assert self.lin.from_sampling_space(3.7) == 3.7

    def test_log10_forward_and_inverse(self):
        assert self.log.to_sampling_space(100.0) == 2.0
        # Unguarded, bit-for-bit 10.0**u (the contract the proposal arithmetic relied on).
        assert self.log.from_sampling_space(2.0) == 10.0 ** 2.0

    @pytest.mark.parametrize("theta", [0.001, 0.5, 1.0, 42.0, 1e5])
    def test_round_trip(self, theta):
        for p in (self.lin, self.log):
            np.testing.assert_allclose(
                p.from_sampling_space(p.to_sampling_space(theta)), theta, rtol=1e-12)

    def test_forward_accepts_arrays(self):
        """The histogram path passes a whole data column through the forward map."""
        col = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(self.log.to_sampling_space(col), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(self.lin.to_sampling_space(col), col)

    def test_inverse_is_unguarded(self):
        """Unlike the guarded exp10 (which re-raises overflow as a PybnfError
        configuration hint), from_sampling_space is the bare scale inverse: a
        numpy-float overflow -- the type the proposal arithmetic produces, since
        to_sampling_space returns np.float64 -- yields inf, which the box clamp /
        reflection at the call site handles, never a mid-fit error."""
        with np.errstate(over='ignore'):
            assert np.isinf(self.log.from_sampling_space(np.float64(1000.0)))


class TestTruncatedFreeParameter:
    """Two finite bounds on an unbounded-support prior (normal/laplace/log-*)
    turn it into a truncated prior with a reflecting box (ADR-0020, #411). The
    box machinery (reflection, latin-hypercube) was already family-agnostic and
    only gated off for these families; truncation flips the gate."""

    def test_box_and_flags(self):
        fp = pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, lb=-1.0, ub=4.0)
        assert fp.bounded
        assert fp.has_bounded_support              # now latin-hypercube eligible
        assert fp.lower_bound == -1.0 and fp.upper_bound == 4.0

    def test_sampling_stays_in_box(self):
        fp = pset.FreeParameter('x__FREE', 'normal_var', 5.0, 3.0, lb=0.0, ub=6.0)
        rng = np.random.default_rng(0)
        xs = np.array([fp.sample_value(rng).value for _ in range(20000)])
        assert xs.min() >= 0.0 and xs.max() <= 6.0

    def test_prior_logpdf_matches_truncnorm(self):
        fp = pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, lb=-1.0, ub=4.0)
        oracle = _truncnorm(1.0, 2.0, -1.0, 4.0)
        for v in (-1.0, 0.0, 1.0, 3.9):
            assert fp.prior_logpdf(v) == pytest.approx(oracle.logpdf(v), rel=1e-12)

    def test_value_from_quantile_matches_truncnorm(self):
        fp = pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, lb=-1.0, ub=4.0)
        oracle = _truncnorm(1.0, 2.0, -1.0, 4.0)
        for q in (0.1, 0.5, 0.9):
            assert fp.value_from_quantile(q).value == pytest.approx(oracle.ppf(q), rel=1e-9)

    def test_reflection_folds_into_box(self):
        # The triangle-wave fold (gated off for normal_var before #411) is active:
        # box [0, 10], 11 -> 9, 25 -> 5, matching the uniform_var oracle.
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=0.0, ub=10.0)
        assert fp.set_value(11).value == pytest.approx(9.0)
        assert fp.set_value(25).value == pytest.approx(5.0)

    def test_set_value_preserves_truncation_box(self):
        # Reconstruction must carry the box through, else the rebuilt parameter
        # would silently re-widen to unbounded.
        fp = pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, lb=-1.0, ub=4.0)
        fp2 = fp.set_value(2.0)
        assert fp2.bounded and fp2.has_bounded_support
        assert fp2.lower_bound == -1.0 and fp2.upper_bound == 4.0

    def test_log_truncation_reflects_in_log_space(self):
        # lognormal_var truncated to [0.1, 100] -> box [-1, 2] in log10 u.
        fp = pset.FreeParameter('x__FREE', 'lognormal_var', 1.0, 0.5, lb=0.1, ub=100.0)
        assert fp.lower_bound == 0.1 and fp.upper_bound == 100.0
        rng = np.random.default_rng(0)
        xs = np.array([fp.sample_value(rng).value for _ in range(20000)])
        assert xs.min() >= 0.1 and xs.max() <= 100.0

    def test_nominal_value_outside_box_raises(self):
        with pytest.raises(pset.OutOfBoundsException):
            pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, value=99.0, lb=-1.0, ub=4.0)

    def test_one_sided_box_raises(self):
        # An infinite bound -> no finite width to fold into.
        with pytest.raises(PybnfError):
            pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=0.0, ub=np.inf)

    def test_only_one_bound_given_raises(self):
        # Passing exactly one of lb/ub is loud, not a silent unbounded prior.
        with pytest.raises(PybnfError):
            pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, ub=5.0)
        with pytest.raises(PybnfError):
            pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=-5.0)
