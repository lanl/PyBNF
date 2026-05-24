from .context import pset, raises

import numpy as np
import pytest
from hypothesis import given, strategies as st


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
        xs = [self.p3.sample_value().value for x in range(100000)]
        for x in xs:
            assert self.p3.lower_bound <= x < self.p3.upper_bound
        ys = [self.p0.sample_value().value for x in range(100000)]
        assert np.any(np.array(ys) < 0.0)  # normal_var centered at 0 should produce negative values

    def test_sample_value(self):
        p0s = self.p0.sample_value()
        assert p0s.value is not None

    def test_freeparameter_equality(self):
        p6 = self.p0.sample_value()
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
