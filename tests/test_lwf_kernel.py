"""The Liu–West filter's arithmetic, with no run, simulator or bngsim.

Oracles: the move keeps each weighted mean and variance and scales each covariance by
``a_i a_j + h_i h_j``, exactly, under draws built to have weighted mean 0, covariance ``I`` and
no correlation with the population; each parameter's step, read off the output, is its own ``h``;
systematic resampling gives particle ``i`` ``floor(N w_i)`` or ``ceil(N w_i)`` copies, and a hand
count; the weight ESS is ``1 / sum(w**2)``; the fold into the box is
:meth:`FreeParameter.set_value`'s, value for value.
"""

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.pset import FreeParameter
from . import lwf_harness as C


def _weighted_moments(x, w):
    mean = w @ x
    dev = x - mean
    return mean, (dev * w[:, None]).T @ dev


@pytest.mark.parametrize('h', [(0.15, 0.15, 0.15), (0.3, 0.05, 0.1)])
def test_the_move_keeps_each_mean_and_variance_and_scales_each_covariance_exactly(h):
    rng = np.random.default_rng(3)
    n = 400
    x = rng.standard_normal((n, 3)) @ np.array([[1.0, 0.6, 0.2], [0.0, 0.8, -0.3], [0, 0, 0.5]])
    x += np.array([2.0, -1.0, 5.0])
    w = rng.random(n)
    w /= w.sum()
    z = C.unit_draws(x, w, rng)
    moved, cholesky = lw.liu_west_move(x, w, h, z)
    assert cholesky
    mean0, cov0 = _weighted_moments(x, w)
    mean1, cov1 = _weighted_moments(moved, w)
    h = np.asarray(h)
    a = np.sqrt(1.0 - h ** 2)
    np.testing.assert_allclose(mean1, mean0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov1, (np.outer(a, a) + np.outer(h, h)) * cov0, rtol=1e-10,
                               atol=1e-12)


def test_a_singular_covariance_moves_each_parameter_on_its_own_scale():
    """A parameter equal to 0 in every particle gives an exact zero pivot on any LAPACK. The
    move then uses independent noise at each parameter's own scale, ``a x + (1 - a) m + h sd z``
    by hand, and reports it (``cholesky`` False)."""
    rng = np.random.default_rng(5)
    n = 50
    first = rng.standard_normal(n)
    x = np.column_stack([first, np.zeros(n), 0.8 * first + 0.6 * rng.standard_normal(n)])
    w = rng.random(n)
    w /= w.sum()
    h = np.array([0.2, 0.3, 0.1])
    z = rng.standard_normal((n, 3))
    z -= w @ z
    moved, cholesky = lw.liu_west_move(x, w, h, z)
    assert not cholesky
    mean = w @ x
    sd = np.sqrt(w @ (x - mean) ** 2)
    a = np.sqrt(1.0 - h ** 2)
    np.testing.assert_allclose(moved, a * x + (1.0 - a) * mean + h * sd * z, rtol=1e-12,
                               atol=1e-14)
    assert np.array_equal(moved[:, 1], np.zeros(n))
    np.testing.assert_allclose(w @ moved, mean, atol=1e-12)


def test_a_row_without_weight_has_no_part_in_the_moments_whatever_it_holds():
    """A row without weight holding inf or nan moves the others exactly as a finite value
    would (0 times inf would be nan)."""
    rng = np.random.default_rng(13)
    n = 40
    x = rng.standard_normal((n, 3)) @ np.array([[1.0, 0.4, 0.1], [0.0, 0.9, -0.2], [0, 0, 0.6]])
    w = rng.random(n)
    w[[3, 17]] = 0.0
    w /= w.sum()
    z = rng.standard_normal((n, 3))
    finite = x.copy()
    finite[3], finite[17] = 1e6, -2.0
    reference, cholesky = lw.liu_west_move(finite, w, 0.2, z, np.full(3, 1e-9))
    assert cholesky
    held = x.copy()
    held[3], held[17] = np.inf, np.nan
    moved, cholesky = lw.liu_west_move(held, w, 0.2, z, np.full(3, 1e-9))
    assert cholesky
    live = w > 0.0
    assert np.array_equal(moved[live], reference[live])
    assert np.isfinite(moved[live]).all()



def test_a_parameter_the_kernel_does_not_move_keeps_its_values_and_scales_its_covariance_by_a():
    rng = np.random.default_rng(9)
    n = 300
    u = rng.standard_normal((n, 3)) @ np.array([[1, 0.5, 0.3], [0, 1, 0.4], [0, 0, 1.0]])
    w = rng.random(n)
    w /= w.sum()
    h = np.array([0.25, 0.0, 0.1])            # the middle one sets only the initial state
    alg = object.__new__(lw.LiuWestFilter)
    alg.jitter, alg._moved = h, h > 0
    alg._lo, alg._hi = np.full(3, -np.inf), np.full(3, np.inf)
    alg._logit, alg._var_floor = np.zeros(3, dtype=bool), np.zeros(2)
    pop = lw._Population(u=u.copy(), state=np.zeros((n, 1)), weights=w)
    # Orthogonal to the unmoved column as well, so its cross-covariance is exact too.
    z = C.unit_draws(u, w, rng, d=2)
    alg._move(pop, C.Draws(z))
    assert np.array_equal(pop.u[:, 1], u[:, 1])
    _, cov0 = _weighted_moments(u, w)
    _, cov1 = _weighted_moments(pop.u, w)
    a = np.sqrt(1 - h ** 2)
    assert cov1[0, 1] == pytest.approx(a[0] * cov0[0, 1], rel=1e-9)
    assert cov1[2, 1] == pytest.approx(a[2] * cov0[2, 1], rel=1e-9)
    assert cov1[0, 2] == pytest.approx((a[0] * a[2] + h[0] * h[2]) * cov0[0, 2], rel=1e-9)


@pytest.mark.parametrize('seed', range(5))
def test_systematic_resampling_gives_each_particle_floor_or_ceil_of_n_w_copies(seed):
    rng = np.random.default_rng(seed)
    n = 97
    w = rng.gamma(0.3, size=n)
    w /= w.sum()
    for uniform in (0.0, 0.3, rng.random(), 0.999999):
        copies = np.bincount(lw.systematic_resample(w, uniform), minlength=n)
        assert copies.sum() == n
        assert np.all((copies == np.floor(n * w)) | (copies == np.ceil(n * w)))


def test_systematic_resampling_never_selects_a_particle_of_weight_0():
    """Normalized weights can sum to a few ulps below 1; a position above that sum must not fall
    to the last particle when its weight is 0."""
    rng = np.random.default_rng(0)
    n = 4000
    for _ in range(300):
        w = rng.gamma(0.3, size=n)
        w[-1] = 0.0
        w /= w.sum()
        ancestors = lw.systematic_resample(w, np.nextafter(1.0, 0.0))
        assert np.all(w[ancestors] > 0.0)


def test_time_headers_keep_distinct_times_distinct():
    assert lw._time_headers([1.0, 2.5, 1e6, 1e6 + 1]) == ['1', '2.5', '1000000', '1000001']


def test_systematic_resampling_by_hand():
    """Positions 0.125, 0.375, 0.625, 0.875 fall in the cumulative weights of 1, 1, 2, 3."""
    assert list(lw.systematic_resample([0.1, 0.4, 0.25, 0.25], 0.5)) == [1, 1, 2, 3]


def test_the_weight_ess_is_one_over_the_sum_of_squares():
    assert lw.weight_ess([0.25, 0.25, 0.25, 0.25]) == pytest.approx(4.0)
    assert lw.weight_ess([0.5, 0.5, 0.0, 0.0]) == pytest.approx(2.0)
    assert lw.weight_ess([0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0 / 0.3)


@pytest.mark.parametrize('declaration', [
    ('uniform_var', 1.0, 5.0),
    ('loguniform_var', 0.01, 10.0),
    ('normal_var', 0.0, 1.0),
])
def test_the_fold_into_the_box_is_the_fold_a_free_parameter_makes(declaration):
    kind, p1, p2 = declaration
    lb, ub = (None, None) if kind != 'normal_var' else (-1.0, None)
    v = FreeParameter('p', kind, p1, p2, lb=lb, ub=ub)
    lo, hi = lw._box_in_sampling_space(v)
    rng = np.random.default_rng(2)
    width = (hi - lo) if np.isfinite(hi) else 4.0
    u = np.concatenate([lo - rng.random(50) * 3 * width, rng.random(20) * 0 + lo + 0.5,
                        (hi if np.isfinite(hi) else lo + 10) + rng.random(50) * 3 * width])
    folded = lw.fold_into_box(u[:, None], [lo], [hi])[:, 0]
    expected = [v.to_sampling_space(v.set_value(v.from_sampling_space(x), reflect=True).value)
                for x in u]
    np.testing.assert_allclose(folded, expected, rtol=1e-12, atol=1e-12)


def test_the_logit_working_space_round_trips_and_stays_in_the_box():
    lo, hi = np.array([0.0, -2.0]), np.array([1.0, 3.0])
    u = np.array([[0.2, -1.0], [0.9, 2.5], [0.5, 0.0]])
    x = lw.to_working(u, lo, hi, np.array([True, True]))
    np.testing.assert_allclose(lw.from_working(x, lo, hi, np.array([True, True])), u, rtol=1e-12)
    far = lw.from_working(np.array([[80.0, -80.0]]), lo, hi, np.array([True, True]))
    assert lo[0] <= far[0, 0] <= hi[0] and lo[1] <= far[0, 1] <= hi[1]


def test_a_stream_is_keyed_on_the_row_time_not_on_what_came_before():
    first = lw.stream(42, 0, lw._UPDATE, 7.0).random(5)
    lw.stream(42, 0, lw._UPDATE, 3.0).random(100)    # an earlier row's draws
    assert np.array_equal(lw.stream(42, 0, lw._UPDATE, 7.0).random(5), first)
    assert not np.array_equal(lw.stream(42, 0, lw._UPDATE, 14.0).random(5), first)
    assert not np.array_equal(lw.stream(42, 0, lw._FORECAST, 7.0).random(5), first)
    assert not np.array_equal(lw.stream(42, 1, lw._UPDATE, 7.0).random(5), first)
    assert not np.array_equal(lw.stream(43, 0, lw._UPDATE, 7.0).random(5), first)
    assert np.array_equal(lw.stream(42, 0, lw._UPDATE, -0.0).random(3),
                          lw.stream(42, 0, lw._UPDATE, 0.0).random(3))


def test_the_weighted_quantiles_follow_the_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.7, 0.1, 0.1, 0.1])
    assert lw._weighted_quantiles(values, weights, (0.5, 0.75, 0.95)) == [1.0, 2.0, 4.0]


def test_resampling_carries_each_particles_state_and_predicted_counts_with_its_parameters():
    pop = lw._Population(u=np.arange(8.0).reshape(4, 2), state=np.arange(4.0)[:, None] * 10,
                         weights=np.array([0.1, 0.2, 0.3, 0.4]),
                         predicted=[np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])])
    pop.resample(np.array([3, 3, 1, 0]))
    assert np.array_equal(pop.u[:, 0], [6.0, 6.0, 2.0, 0.0])
    assert np.array_equal(pop.state[:, 0], [30.0, 30.0, 10.0, 0.0])
    assert np.array_equal(pop.predicted[0], [4.0, 4.0, 2.0, 1.0])
    assert np.array_equal(pop.predicted[1], [8.0, 8.0, 6.0, 5.0])
    assert np.array_equal(pop.weights, [0.25] * 4)
