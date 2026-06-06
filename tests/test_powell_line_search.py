"""Unit tests for Powell's bracketing + Brent line search (#406, ADR-0016).

``_BrentLineSearch`` is the resumable, picklable 1-D minimizer that replaced the
fixed-step parabola. It is driven one evaluation at a time (``first`` then
``feed``), confined to a feasible ``t``-interval. These tests drive it against
closed-form 1-D minima and scipy's Brent as oracles, check the box-constraint
(boundary-as-minimum), and confirm it pickles mid-search (the backup/resume
contract, ADR-0015).
"""
import pickle

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pybnf.algorithms.optimizers.powell import _BrentLineSearch


def _minimize(func, t_lo=-50.0, t_hi=50.0, step=1.0, tol=1e-8, max_evals=200):
    """Drive a _BrentLineSearch over ``func`` to completion; return (ls, n_evals).

    Mirrors how PowellAlgorithm drives it: ``first()`` gives the first abscissa,
    each ``feed(t, f)`` the next, until ``('done', ...)``."""
    ls = _BrentLineSearch(func(0.0), t_lo, t_hi, step, tol, max_evals)
    t = ls.first()
    n = 0
    while t is not None:
        n += 1
        action = ls.feed(t, func(t))
        if action[0] == 'done':
            break
        t = action[1]
    return ls, n


@pytest.mark.parametrize('tmin', [0.7, -2.3, 4.1, -0.05])
def test_brent_finds_quadratic_minimum(tmin):
    """On a smooth quadratic slice the minimum is found to ~tol, matching the
    closed-form argmin from any starting bracket position."""
    def func(t):
        return 3.0 * (t - tmin) ** 2 + 1.0
    ls, _ = _minimize(func, tol=1e-9)
    assert ls.best_t == pytest.approx(tmin, abs=1e-5)
    assert ls.best_f == pytest.approx(1.0, abs=1e-8)


def test_brent_matches_scipy_on_nonquadratic_slice():
    """Oracle: on a non-quadratic (quartic+quadratic) slice, the recovered
    minimizer matches scipy's Brent to the requested tolerance."""
    def func(t):
        return 0.4 * (t - 1.3) ** 4 + 0.5 * (t - 1.3) ** 2
    ls, _ = _minimize(func, tol=1e-7)
    ref = minimize_scalar(func, method='brent').x
    assert ls.best_t == pytest.approx(ref, abs=1e-3)
    assert ls.best_t == pytest.approx(1.3, abs=1e-3)


def test_brent_brackets_minimum_straddling_zero():
    """When both +/- probes go uphill, the base is bracketed and the interior
    minimum near 0 is found (no spurious step off to a boundary)."""
    def func(t):
        return t ** 2
    ls, _ = _minimize(func, step=1.0, tol=1e-9)
    assert ls.best_t == pytest.approx(0.0, abs=1e-5)


def test_brent_box_constraint_puts_minimum_on_boundary():
    """If the unconstrained minimum lies outside the feasible interval, the line
    search returns the boundary (the constrained minimum) and never steps past
    it — the mechanism that makes the refine path bound-correct."""
    def func(t):
        return (t - 3.0) ** 2                 # unconstrained min at t=3
    ls, _ = _minimize(func, t_lo=-1.0, t_hi=1.0, step=0.5)
    assert ls.best_t == pytest.approx(1.0, abs=1e-9)   # clamped to the upper bound
    assert -1.0 <= ls.best_t <= 1.0                    # never evaluated past the box


def test_brent_pinned_interval_makes_no_move():
    """A degenerate feasible interval (the point is pinned) yields no move:
    first() is None and the best stays the base."""
    def func(t):
        return (t - 1.0) ** 2
    ls = _BrentLineSearch(func(0.0), 0.0, 0.0, 1.0, 1e-8, 100)
    assert ls.first() is None
    assert ls.best_t == 0.0
    assert ls.best_f == pytest.approx(func(0.0))


def test_brent_respects_eval_cap():
    """The combined bracket+Brent evaluation budget is honored; on exhaustion the
    best point seen is returned (a finite, in-interval abscissa)."""
    def func(t):
        return (t - 12.34) ** 2
    ls, n = _minimize(func, max_evals=5)
    assert ls.evals <= 5
    assert n <= 5
    assert np.isfinite(ls.best_t)


def test_brent_line_search_pickles_mid_search():
    """The line minimizer pickles and resumes mid-search to the same minimum —
    the per-step backup/resume contract for Powell (ADR-0015/0016). All state is
    plain float/int/bool."""
    def func(t):
        return 2.0 * (t - 0.6) ** 2 + 5.0
    ls = _BrentLineSearch(func(0.0), -10.0, 10.0, 1.0, 1e-9, 200)
    t = ls.first()
    action = ('eval', t)
    # Take two steps, then pickle-roundtrip and finish from the snapshot.
    for _ in range(2):
        action = ls.feed(t, func(t))
        if action[0] == 'done':
            break
        t = action[1]
    ls = pickle.loads(pickle.dumps(ls))          # resume from the snapshot
    while action[0] != 'done':
        action = ls.feed(t, func(t))
        if action[0] == 'done':
            break
        t = action[1]
    assert ls.best_t == pytest.approx(0.6, abs=1e-5)
    assert ls.best_f == pytest.approx(5.0, abs=1e-8)
