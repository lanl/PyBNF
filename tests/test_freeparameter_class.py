from .context import pset, raises

import logging

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


# Bounds as they are written in a conf: a short mantissa times a power of ten. Spelled as
# decimal literals, since that is what a parser hands the box -- 1.2e-06 is not 1.2 * 1e-06.
_ORDINARY_BOUNDS = [float(f'{m}e{e}')
                    for m in ('1', '1.2', '1.5', '2', '2.5', '3', '4', '5', '6', '7.5',
                              '8', '9')
                    for e in range(-6, 4)]


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


class TestReflectBesideAWall:
    """A proposal within floating-point rounding of a wall folds INSIDE the box.

    The fold used to reconstruct the descending leg from the far wall, so a value one
    ulp below ``lb`` came back as ``ub - width``: on [0.1, 5] that is
    0.09999999999999964, below the wall, and the constructor's bounds check then raised
    :class:`OutOfBoundsException` and ended the fit. It did end one: a differential
    evolution proposal for ``k_elim`` on tutorial lesson 25, through
    ``new_individual`` -> ``FreeParameter.add`` -> ``set_value``. On a log scale the
    theta<->u round trip does it without help, since ``10 ** log10(30)`` is
    30.000000000000004.

    Both walls, both scales, and the half-bounded boxes, since a one-wall fold reflects
    through its own arithmetic and lands on the same rounding."""

    # (label, parameter, values that sit within a rounding outside a wall)
    def _cases(self):
        lin = pset.FreeParameter('k_elim__FREE', 'uniform_var', 0.1, 5.0, 1.0)
        log = pset.FreeParameter('k__FREE', 'loguniform_var', 0.1, 30.0, 1.0)
        wide = pset.FreeParameter('w__FREE', 'uniform_var', 1e-9, 1e9, 1.0)
        open_above = pset.FreeParameter('h__FREE', 'normal_var', 0.0, 1.0, 0.5,
                                        lb=0.1, ub=np.inf)
        open_below = pset.FreeParameter('h__FREE', 'normal_var', 0.0, 1.0, 0.5,
                                        lb=-np.inf, ub=2.0)
        return [
            ('linear, lower wall', lin, [np.nextafter(0.1, 0), 0.1 - 1e-17, 0.1 - 1e-16,
                                         0.1 - 1e-13, 0.1 - 1e-12]),
            ('linear, upper wall', lin, [np.nextafter(5.0, 10), 5.0 + 1e-15, 5.0 + 1e-12]),
            ('log, lower wall', log, [np.nextafter(0.1, 0), 0.1 - 1e-17]),
            ('log, upper wall', log, [np.nextafter(30.0, 40), 30.0 + 1e-13]),
            ('wide box, lower wall', wide, [np.nextafter(1e-9, 0), 1e-9 - 1e-25]),
            ('open above, its wall', open_above, [np.nextafter(0.1, 0), 0.1 - 1e-17]),
            ('open below, its wall', open_below, [np.nextafter(2.0, 10), 2.0 + 1e-15]),
        ]

    def test_folds_inside_the_box(self):
        for label, p, values in self._cases():
            for new in values:
                folded = p.set_value(new).value       # must not raise
                assert p.lower_bound <= folded <= p.upper_bound, (
                    f'{label}: {new!r} folded to {folded!r}, outside '
                    f'[{p.lower_bound}, {p.upper_bound}]')

    def test_lands_beside_the_wall_it_came_from(self):
        """Not merely inside: a proposal a rounding outside a wall folds to that wall's
        own neighbourhood. On the wide box the old fold answered 1e-07 for a proposal at
        1e-09 -- inside the box, and a hundredfold off."""
        for label, p, values in self._cases():
            for new in values:
                wall = p.lower_bound if new < p.lower_bound else p.upper_bound
                folded = p.set_value(new).value
                assert abs(folded - wall) <= 2.0 * abs(new - wall) + 1e-300, (
                    f'{label}: {new!r} folded to {folded!r}, far from the wall {wall}')

    def test_de_proposal_beside_a_wall_survives(self):
        """The crash as differential evolution reached it: a member sitting on the wall
        takes a step that would leave the box by a rounding. ``add`` folds it back."""
        p = pset.FreeParameter('k_elim__FREE', 'uniform_var', 0.1, 5.0, 0.1)
        stepped = p.add(-np.finfo(float).eps / 8)     # must not raise
        assert 0.1 <= stepped.value <= 5.0

    def test_a_wall_is_a_legal_value(self):
        """The fold may now return a bound itself, so the constructor has to accept one
        (it always did -- its check is inclusive)."""
        for value in (0.1, 5.0):
            assert pset.FreeParameter('k__FREE', 'uniform_var', 0.1, 5.0, value).value == value

    @pytest.mark.parametrize('lb, ub', [(0.1, 5.0), (-3.0, 7.0), (1e-9, 1e9)])
    def test_every_ulp_around_both_walls(self, lb, ub):
        """Sweep the first few representable values on each side of each wall."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', lb, ub)
        for wall in (lb, ub):
            for direction in (-np.inf, np.inf):
                new = wall
                for _ in range(5):
                    new = np.nextafter(new, direction)
                    assert lb <= p.set_value(new).value <= ub

    def test_far_proposals_still_match_the_reference_fold(self):
        """The rearrangement is the same triangle wave away from the walls."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0, 10)
        for new in (10.5, 19.0, 20.0, 21.0, -1.0, -11.0, 100.3, 1000000.7):
            np.testing.assert_allclose(p.set_value(float(new)).value,
                                       _fold_reference(new, 0, 10), atol=1e-9)

    def test_a_nan_proposal_still_raises(self):
        """The clip is written so a NaN reaches the constructor's check as before,
        rather than being silently clipped onto a wall."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0.1, 5.0, 1.0)
        with pytest.raises(pset.OutOfBoundsException):
            p.set_value(float('nan'))


class TestADeclaredBoundSurvivesItsOwnRoundTrip:
    """A parameter resting ON a declared bound materializes back inside its box (#750).

    The bounded optimizers work in sampling space ``u`` over
    ``[to_sampling_space(lb), to_sampling_space(ub)]`` (``GradientOptimizer._u_bounds``)
    and project every iterate into it, so a bound that is *active at the optimum* reaches
    the PSet bridge as exactly that wall in ``u``. ``Algorithm._pset_from_u`` maps it back
    with ``10 ** u``, which lands on the far side of the wall it came from for 91 of the
    240 walls swept here: ``10 ** log10(20)`` is 20.000000000000004 and
    ``10 ** log10(5000)`` is 4999.999999999999. So the fold has to absorb the image of the
    box's own corner, and before #706 it did not -- ``job_type = profile_likelihood`` on
    ``loguniform_var Vm2 1.2 20`` ended at the first grid point whose re-optimization
    pressed ``Vm2`` against its bound, with "Free parameter Vm2 cannot be assigned the
    value 20.000000000000004" (#750). 86 of these 240 walls raise against the old fold.
    Both counts are from one platform: ``log10`` and ``pow`` are not required to be
    correctly rounded, so *which* walls overshoot is libm's business, which is why the
    assertions below are over the sweep rather than over named bounds.

    Distinct from :class:`TestReflectBesideAWall` above, which folds proposals *chosen* to
    sit a rounding outside a wall: a bounded gradient optimizer returning an iterate on an
    active bound is ordinary behaviour rather than a near-miss, so this sweeps the bounds
    people write instead of the walls of one box.
    """

    def _walls(self):
        """Each bound twice: as the upper wall of one box, as the lower wall of another."""
        for b in _ORDINARY_BOUNDS:
            yield b / 100.0, b, b
            yield b, b * 100.0, b

    def test_a_parameter_on_a_declared_bound_stays_in_its_box(self):
        for lo, hi, wall in self._walls():
            p = pset.FreeParameter('x__FREE', 'loguniform_var', lo, hi)
            # Exactly the value the bridge assigns when this wall is the active bound:
            # the optimizer's iterate is the wall in u, and 10 ** u brings it back.
            theta = p.from_sampling_space(p.to_sampling_space(wall))
            value = p.set_value(theta).value          # must not raise
            assert lo <= value <= hi, (
                f'box [{lo}, {hi}]: the wall {wall} came back as {value!r}, outside it')
            assert value == pytest.approx(wall, rel=1e-12), (
                f'box [{lo}, {hi}]: the wall {wall} came back as {value!r}')

    def test_the_round_trip_really_does_leave_the_box(self):
        """Without this the sweep above could pass vacuously -- a round trip that never
        left the box would exercise no fold at all."""
        outside = 0
        for lo, hi, wall in self._walls():
            p = pset.FreeParameter('x__FREE', 'loguniform_var', lo, hi)
            theta = p.from_sampling_space(p.to_sampling_space(wall))
            outside += not lo <= theta <= hi
        assert outside >= 40, (
            f'only {outside} of 240 walls came back outside their box, so the sweep above '
            f'is not reaching the fold (91 did where this was written)')

    def test_the_reported_values_fold_onto_their_wall(self):
        """The two values #750 reported, as literals rather than as libm's output: the
        first from its original box, the second from the widened one it tried next, which
        moved the abort rather than removing it."""
        for lo, hi, image in ((1.2, 20.0, 20.000000000000004),
                              (1.176, 20.4, 20.400000000000002)):
            p = pset.FreeParameter('Vm2__FREE', 'loguniform_var', lo, hi)
            assert image > hi                          # outside the box it was built from
            value = p.set_value(image).value           # must not raise
            assert lo <= value <= hi
            assert value == pytest.approx(hi, rel=1e-12)   # and back onto the wall


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

    def test_one_sided_box_is_half_bounded(self):
        # An infinite bound is the ub->inf limit of the fold: a single reflecting
        # wall, not an error (ADR-0047). Open above: wall at lb, reflect below it.
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=0.0, ub=np.inf)
        assert fp.bounded and fp.has_bounded_support
        assert fp.lower_bound == 0.0 and fp.upper_bound == np.inf
        assert fp.set_value(-3.0).value == pytest.approx(3.0)   # 2*0 - (-3)
        assert fp.set_value(7.0).value == pytest.approx(7.0)    # in-bounds, untouched
        # Open below: wall at ub, reflect above it.
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=-np.inf, ub=2.0)
        assert fp.lower_bound == -np.inf and fp.upper_bound == 2.0
        assert fp.set_value(5.0).value == pytest.approx(-1.0)   # 2*2 - 5

    def test_only_one_bound_given_is_half_bounded(self):
        # The constructor treats a missing (None) side as open (+-inf); the pairing
        # rule is a native-surface concern (ADR-0047), not a core constraint.
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, ub=5.0)
        assert fp.lower_bound == -np.inf and fp.upper_bound == 5.0
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=-5.0)
        assert fp.lower_bound == -5.0 and fp.upper_bound == np.inf

    def test_half_bounded_logpdf_matches_truncnorm(self):
        # A half-line truncated normal renormalizes over [lb, inf): oracle against
        # scipy truncnorm with an infinite upper bound (ADR-0047).
        fp = pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, lb=-1.0, ub=np.inf)
        oracle = _truncnorm(1.0, 2.0, -1.0, np.inf)
        for v in (-0.5, 0.0, 1.0, 3.0, 8.0):
            assert fp.prior_logpdf(v) == pytest.approx(oracle.logpdf(v))
        assert fp.prior_logpdf(-2.0) == -np.inf   # below the wall

    def test_half_bounded_value_from_quantile_matches_truncnorm(self):
        fp = pset.FreeParameter('x__FREE', 'normal_var', 1.0, 2.0, lb=-1.0, ub=np.inf)
        oracle = _truncnorm(1.0, 2.0, -1.0, np.inf)
        for q in (0.1, 0.5, 0.9, 0.99):
            assert fp.value_from_quantile(q).value == pytest.approx(oracle.ppf(q))

    def test_explicit_infinite_box_is_untruncated(self):
        # lb/ub both infinite is identical to omitting the bounds (the untruncated
        # prior), not a degenerate box (ADR-0047).
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0, lb=-np.inf, ub=np.inf)
        assert not fp.bounded and not fp.has_bounded_support
        plain = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0)
        assert fp.prior_logpdf(2.0) == pytest.approx(plain.prior_logpdf(2.0))


class TestInitializationDistribution:
    """Start-point sampling is a separate distribution from the objective prior
    (#413). ``sample_value`` stays the prior draw; algorithms use
    ``sample_initial_value`` / ``initial_value_from_quantile``."""

    def test_default_initialization_is_prior(self):
        fp = pset.FreeParameter('x__FREE', 'normal_var', 0.0, 1.0)
        assert not fp.has_bounded_initialization
        assert fp.initialization_distribution == pset.INITIALIZATION_PRIOR

    def test_bounds_initialization_uses_box_not_prior_quantile(self):
        fp = pset.FreeParameter(
            'x__FREE', 'normal_var', 0.0, 0.01, lb=-10.0, ub=10.0,
            initialization_distribution=pset.INITIALIZATION_BOUNDS)
        assert fp.has_bounded_initialization
        assert fp.initial_value_from_quantile(0.25).value == pytest.approx(-5.0)
        assert fp.initial_value_from_quantile(0.75).value == pytest.approx(5.0)
        assert abs(fp.value_from_quantile(0.75).value) < 0.01

    def test_explicit_initialization_bounds_are_independent(self):
        fp = pset.FreeParameter(
            'x__FREE', 'normal_var', 0.0, 0.01, lb=-10.0, ub=10.0,
            initialization_distribution=pset.INITIALIZATION_BOUNDS,
            initialization_lb=-5.0, initialization_ub=5.0)
        assert fp.initial_value_from_quantile(0.0).value == pytest.approx(-5.0)
        assert fp.initial_value_from_quantile(1.0).value == pytest.approx(5.0)
        assert abs(fp.value_from_quantile(0.75).value) < 0.1

    def test_set_value_preserves_initialization_distribution(self):
        fp = pset.FreeParameter(
            'x__FREE', 'normal_var', 0.0, 1.0, lb=-2.0, ub=2.0,
            initialization_distribution=pset.INITIALIZATION_BOUNDS,
            initialization_lb=-10.0, initialization_ub=10.0)
        got = fp.set_value(1.0)
        assert got.initialization_distribution == pset.INITIALIZATION_BOUNDS
        assert got.initialization_lb == -10.0
        assert got.initialization_ub == 10.0

    def test_bounds_initialization_requires_finite_box(self):
        with pytest.raises(PybnfError, match='initialization_distribution'):
            pset.FreeParameter(
                'x__FREE', 'normal_var', 0.0, 1.0,
                initialization_distribution=pset.INITIALIZATION_BOUNDS)


class TestTheNoInitializationBoxRefusalIsActionable:
    """``initialization_distribution = 'bounds'`` needs a finite box, and a parameter that
    has none has to be refused -- "uniformly over a half-line" is not a distribution. The
    refusal is a configuration error and now reads like one (#799).

    It used to read ``Parameter x1: initialization bounds must be finite and increasing in
    sampling space, got [-12.0, inf].`` for a parameter declared ``lower: 1e-12``: the
    number was the log10 of what the user wrote and appeared nowhere in their conf, the key
    that caused the refusal was never named, and no remedy was offered -- on the one
    declaration class (half-bounded, ADR-0047) where the right answer is usually just to
    drop the key."""

    @staticmethod
    def _refusal(**kwargs):
        """The PybnfError from building a parameter whose initialization box is unusable."""
        with pytest.raises(PybnfError) as excinfo:
            pset.FreeParameter(
                'kon__FREE', kwargs.pop('type', 'lognormal_var'),
                kwargs.pop('p1', -9.0), kwargs.pop('p2', 0.5),
                initialization_distribution=pset.INITIALIZATION_BOUNDS, **kwargs)
        return excinfo.value

    def test_the_bounds_are_printed_in_the_parameters_own_units(self):
        """The headline. A log-scaled parameter's box is stored in theta and sampled in
        log10; the refusal has to quote the one the user typed, or it names a number that
        is not in their file."""
        e = self._refusal(lb=1e-12, ub=np.inf)
        assert '1e-12' in e.log_message
        assert '-12.0' not in e.log_message      # log10(1e-12): what it used to print

    def test_it_names_the_key_that_caused_it(self):
        """Nothing else in the conf mentions "initialization bounds", so without the key
        the reader goes looking for a setting they never wrote."""
        e = self._refusal(lb=1e-12, ub=np.inf)
        assert "initialization_distribution = 'bounds'" in e.log_message

    def test_it_says_which_side_is_open_and_offers_only_that_side(self):
        """Telling someone who wrote ``lower: 1e-12, upper: inf`` to supply a lower and an
        upper is advice they have half-followed already."""
        e = self._refusal(lb=1e-12, ub=np.inf)
        assert 'open above' in e.log_message
        assert "'upper:'" in e.message and "'lower:'" not in e.message

    def test_the_mirrored_open_side_is_named_the_other_way(self):
        e = self._refusal(type='normal_var', p1=0.0, p2=1.0, lb=-np.inf, ub=5.0)
        assert 'open below' in e.log_message
        assert "'lower:'" in e.message and "'upper:'" not in e.message

    def test_both_remedies_ride_as_hints_and_leave_the_diagnosis_intact(self):
        """``hint`` appends; ``user_message`` would replace. A refusal carrying only a
        generic remedy discards its own reason (#527), and the log line should stay the
        bare diagnosis rather than repeating the advice."""
        e = self._refusal(lb=1e-12, ub=np.inf)
        assert len(e.hints) == 2
        assert e.message.startswith(e.log_message)
        assert '->' not in e.log_message
        assert "remove 'initialization_distribution = bounds'" in e.hints[1]

    def test_a_parameter_with_no_bounds_at_all_asks_for_both_sides(self):
        """The sibling refusal, for an untruncated prior. It already named the key; it now
        also prints in the parameter's units and carries the same two remedies, so the two
        failures of one key do not read as though they came from different programs."""
        e = self._refusal(type='normal_var', p1=0.0, p2=1.0)
        assert "initialization_distribution = 'bounds'" in e.log_message
        assert 'normal_var' in e.log_message         # the prior that has no finite support
        assert "'lower:' and 'upper:'" in e.message
        assert len(e.hints) == 2

    def test_a_positive_family_is_only_asked_for_the_side_it_lacks(self):
        """A ``gamma_var`` is floored at 0 by its own support, so only the ceiling is
        missing. Asking for a lower bound it already has would be noise."""
        e = self._refusal(type='gamma_var', p1=2.0, p2=1.0)
        assert "'upper:'" in e.message and "'lower:' and" not in e.message

    def test_a_usable_box_is_not_refused(self):
        """The guard against an over-eager refusal: the case this key exists for still
        builds, and initializes over the box rather than the prior."""
        fp = pset.FreeParameter('kon__FREE', 'lognormal_var', -9.0, 0.5,
                                lb=1e-12, ub=1e-6,
                                initialization_distribution=pset.INITIALIZATION_BOUNDS)
        assert fp._initialization_bounds_u() == (-12.0, -6.0)


class TestTheOutOfBoundsDebugLineIsReadable:
    """The fold's debug line prints its two values at a precision that can tell them
    apart (#753).

    It used to format both with ``%f``. The two routinely differ only in their last bits --
    the common case being a proposal a rounding past a wall, folded onto it -- so at six
    decimals the line printed the same number twice and refuted itself: "Assigned value
    20.000000 is out of defined bounds: [1.2, 20.0].  Adjusted to 20.000000". Below 1e-6
    both printed as 0.000000. The reader most likely to meet it is someone at DEBUG level
    working out why a bounded fit is behaving oddly, and it is emitted on every evaluation
    for as long as a log-scaled parameter rests on a bound whose corner leaves the box (see
    :class:`TestADeclaredBoundSurvivesItsOwnRoundTrip`)."""

    @staticmethod
    def _fold_at_the_upper_corner(caplog, lo, hi):
        """Fold the image of a log box's upper corner -- what an active bound hands
        ``set_value`` -- and return the emitted records."""
        p = pset.FreeParameter('Vm2__FREE', 'loguniform_var', lo, hi)
        theta = p.from_sampling_space(p.to_sampling_space(hi))
        assert theta > hi, 'this box does not exercise the fold; pick another'
        with caplog.at_level(logging.DEBUG, logger='pybnf.pset'):
            p.set_value(theta)
        return [r.getMessage() for r in caplog.records]

    def test_the_two_values_are_distinguishable(self, caplog):
        line = next(m for m in self._fold_at_the_upper_corner(caplog, 1.2, 20.0)
                    if 'out of defined bounds' in m)
        assert '20.000000000000004' in line      # the value that was assigned
        assert 'Adjusted to 20.0' in line        # the wall it was folded onto
        assert '20.000000 ' not in line          # the %f rendering of both

    def test_a_small_magnitude_value_is_not_printed_as_zero(self, caplog):
        line = next(m for m in self._fold_at_the_upper_corner(caplog, 1.2e-11, 1.2e-09)
                    if 'out of defined bounds' in m)
        assert '0.000000' not in line
        assert '1.2000000000000008e-09' in line and 'Adjusted to 1.2e-09' in line

    def test_a_numpy_scalar_prints_as_its_value(self, caplog):
        """``from_sampling_space`` returns np.float64, whose repr under numpy 2 is
        ``np.float64(...)`` -- the line converts to float first."""
        for line in self._fold_at_the_upper_corner(caplog, 1.2, 20.0):
            assert 'np.float64' not in line

    def test_the_log_space_line_carries_theta(self, caplog):
        """``log10`` rounds a proposal a rounding past the wall back onto the wall, so
        ``new`` equals ``ub`` and u alone cannot explain why a fold is happening."""
        line = next(m for m in self._fold_at_the_upper_corner(caplog, 1.2, 20.0)
                    if 'Reflecting in log space' in m)
        assert 'theta=20.000000000000004' in line

    def test_a_genuine_fold_still_reports_both_ends(self, caplog):
        """The ordinary case -- a proposal well outside a linear box -- is unchanged apart
        from the formatting."""
        p = pset.FreeParameter('x__FREE', 'uniform_var', 0.0, 10.0)
        with caplog.at_level(logging.DEBUG, logger='pybnf.pset'):
            assert p.set_value(25.0).value == 5.0
        line = next(m for m in (r.getMessage() for r in caplog.records)
                    if 'out of defined bounds' in m)
        assert 'Assigned value 25.0 is' in line and line.endswith('Adjusted to 5.0')
