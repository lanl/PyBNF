"""Offline unit tests for the profile-likelihood driver (#446/#466).

These exercise the pure, backend-free pieces -- the chi-square threshold, the CI
extraction + identifiability classification, and the adaptive :class:`_ProfileTrack`
walk driving a reduced-dimension ``_TRFRunner`` -- against analytic linear-Gaussian
problems whose profile confidence intervals are known in closed form. No bngsim, no
dask: the track is driven directly by feeding it synthetic ``GradientResult``\\ s, exactly
as the sibling ``tests/test_gradient_runner.py`` drives the TRF/LBFGS runners against a
scipy oracle.

Linear-Gaussian reference. For residual ``r(theta) = A theta - y`` the objective
``F = 1/2 ||r||**2`` is an exact quadratic ``F = F_min + 1/2 (theta-theta*)^T (A^T A)
(theta-theta*)``, so the profile of ``theta_k`` is the parabola ``Delta chi2(theta_k) =
(theta_k - theta*_k)**2 / C_kk`` with ``C = (A^T A)^{-1}``. The chi-square threshold is
therefore crossed at ``theta*_k +- sqrt(threshold * C_kk)`` -- the analytic CI every test
below checks against. All parameters are on a LINEAR scale, so sampling space ``u`` equals
``theta`` and the u-space Jacobian is ``A`` itself.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.algorithms.optimizers.profile_likelihood import (
    ProfileLikelihoodAlgorithm,
    _FLAT_DCHI2,
    _ProfileTrack,
    _chi2_quantile_1dof,
    _classify,
    _coverage_notes,
    _extract_ci,
    _norm_ppf,
    _render_profile_plots,
    _resolve_profile_idxs,
)
from pybnf.config import Configuration
from pybnf.gradient import GradientResult
from pybnf.parse import ploop

from . import recovery_harness as H
# The Becker EpoR fast-2p fixtures live beside the gradient smoke tests; the profile-likelihood
# agreement test (below) profiles the SAME subset through the SAME sensitivity path.
from .test_gradient_optimizer import (
    _BECKER_FREE,
    _BECKER_NOMINAL,
    _simulate_becker,
    _write_becker_exp,
    SBML_DIR,
)

BNGL_DIR = Path(__file__).resolve().parent / 'bngl_files'
# Truth for the synthetic decay data (the decay model's own nominal values).
TRUE_K = 0.3
TRUE_S0 = 100.0


def _write_decay_exp(path, *, n=21, t_end=10.0, sd=2.0, with_sd=True):
    """A zero-noise analytic decay ``.exp`` (``time Stot [Stot_SD]``): ``Stot = S0*exp(-k*t)``
    at the truth. With ``with_sd`` a constant SD column makes the chi-square a fixed-scale
    Gaussian (an exact least-squares residual -- the profile driver's TRF re-optimization
    path); with ``with_sd=False`` the SD column is dropped, so an ``estimated`` noise scale
    (``chi_sq_dynamic``, sigma from a free parameter) reads no ``_SD`` column -- the
    non-exact objective the L-BFGS-B inner path profiles."""
    t = np.linspace(0.0, t_end, n)
    obs = TRUE_S0 * np.exp(-TRUE_K * t)
    if with_sd:
        lines = ['#\ttime\tStot\tStot_SD']
        lines += ['%.12g\t%.12g\t%.12g' % (ti, oi, sd) for ti, oi in zip(t, obs)]
    else:
        lines = ['#\ttime\tStot']
        lines += ['%.12g\t%.12g' % (ti, oi) for ti, oi in zip(t, obs)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _decay_model(tmp_path):
    return H.strip_actions_block(BNGL_DIR / 'e2e_ode_decay.bngl', tmp_path / 'decay_v2.bngl')


def _two_channel_decay_model(tmp_path):
    """The two-parallel-channel decay ``S(t) = S0*exp(-(k1+k2)*t)`` -- only ``k1+k2`` is
    observable, so ``k1`` and ``k2`` are structurally non-identifiable while ``S0`` stays
    identifiable (see ``tests/bngl_files/e2e_ode_two_channel_decay.bngl``)."""
    return H.strip_actions_block(BNGL_DIR / 'e2e_ode_two_channel_decay.bngl',
                                 tmp_path / 'two_channel_v2.bngl')


# --------------------------------------------------------------------------- #
# chi-square threshold + probit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('conf,expected', [
    (0.95, 3.841458820694124),      # scipy.stats.chi2.ppf(0.95, 1)
    (0.90, 2.705543454095404),
    (0.99, 6.634896601021214),
    (0.6826894921370859, 1.0),      # 1-sigma
])
def test_chi2_quantile_matches_scipy(conf, expected):
    assert _chi2_quantile_1dof(conf) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_norm_ppf_matches_known_quantiles():
    assert _norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
    assert _norm_ppf(0.001) == pytest.approx(-3.090232306167813, abs=1e-8)


def test_confidence_out_of_range_raises():
    from pybnf.printing import PybnfError
    with pytest.raises(PybnfError):
        _chi2_quantile_1dof(1.0)
    with pytest.raises(PybnfError):
        _chi2_quantile_1dof(0.0)


class _V:
    def __init__(self, name):
        self.name = name


def test_resolve_profile_idxs_selects_all_or_a_named_subset():
    from pybnf.printing import PybnfError
    variables = [_V('a'), _V('b'), _V('c')]
    assert _resolve_profile_idxs(variables, None) == [0, 1, 2]
    assert _resolve_profile_idxs(variables, []) == [0, 1, 2]
    # Subset is returned in variables order, not the order it was named.
    assert _resolve_profile_idxs(variables, ['c', 'a']) == [0, 2]
    with pytest.raises(PybnfError) as exc:
        _resolve_profile_idxs(variables, ['a', 'zzz'])
    # The name that was rejected reaches the user, not only the log file (#527): the hint
    # listing the valid ids is appended to that diagnosis, not printed in place of it.
    assert 'zzz' in exc.value.message
    assert 'a, b, c' in exc.value.message      # ... and the hint still lists the valid ids


def test_the_refusal_of_an_unbounded_parameter_names_it():
    """Profiling needs a bounded box for every parameter -- one to lay the grid in, and one to
    recognize a bound-limited CI against. The refusal names **which** parameters are unbounded
    on the user-facing message and then suggests the fix (#527); before, the user saw only
    'Declare each parameter with a bounded prior' and had to open the log to learn which of
    their parameters was at fault.

    The gate reads only ``self.variables``, so a headless stand-in exercises it without a
    config or a model."""
    import types
    from pybnf.printing import PybnfError

    def _var(name, bounded):
        return types.SimpleNamespace(name=name, has_bounded_support=bounded)

    stub = types.SimpleNamespace(variables=[_var('k', True), _var('S0', False)])
    with pytest.raises(PybnfError) as exc:
        ProfileLikelihoodAlgorithm._require_bounded_parameters(stub)

    assert 'S0 is unbounded' in exc.value.message       # the offending parameter, and only it
    assert 'bounded prior' in exc.value.message         # with the remedy appended
    assert exc.value.log_message in exc.value.message   # losing nothing the log has


# --------------------------------------------------------------------------- #
# CI extraction + classification on synthetic parabolas
# --------------------------------------------------------------------------- #
def test_extract_ci_on_a_parabola_recovers_the_crossings():
    # Delta chi2 = (u - 2)**2, threshold 3.84 -> crossings at 2 +- sqrt(3.84).
    u = np.linspace(-2.0, 6.0, 201)
    dchi2 = (u - 2.0) ** 2
    thr = 3.841458820694124
    lo, hi, lob, hib = _extract_ci(u, dchi2, 2.0, thr, -10.0, 10.0)
    assert lo == pytest.approx(2.0 - np.sqrt(thr), abs=1e-3)
    assert hi == pytest.approx(2.0 + np.sqrt(thr), abs=1e-3)
    assert not lob and not hib
    assert _classify(u, dchi2, 2.0, lo, hi, lob, hib) == 'identifiable'


def test_bound_limited_side_is_practical_not_identifiable():
    # Parabola centered at 0, but the grid (== box) only reaches +1.2 on the right,
    # below the threshold -> right side is open at the upper bound.
    u = np.linspace(-3.0, 1.2, 200)
    dchi2 = u ** 2
    thr = 3.841458820694124
    lo, hi, lob, hib = _extract_ci(u, dchi2, 0.0, thr, -3.0, 1.2)
    assert lo == pytest.approx(-np.sqrt(thr), abs=1e-3)
    assert hi == pytest.approx(1.2, abs=1e-9) and hib          # clamped at the bound
    assert _classify(u, dchi2, 0.0, lo, hi, lob, hib) == 'practically non-identifiable'


def test_flat_profile_is_structurally_non_identifiable():
    u = np.linspace(-2.0, 2.0, 81)
    dchi2 = np.zeros_like(u)
    lo, hi, lob, hib = _extract_ci(u, dchi2, 0.0, 3.84, -2.0, 2.0)
    # Flat + below threshold to both finite bounds -> both sides clamped.
    assert _classify(u, dchi2, 0.0, lo, hi, lob, hib) == 'structurally non-identifiable'


# --------------------------------------------------------------------------- #
# Coverage notes: a grid-point-capped open side is flagged, not read as settled (#467)
# --------------------------------------------------------------------------- #
def test_coverage_notes_flags_a_grid_point_capped_open_side():
    # The +1 direction ran out its point budget while its (upper) CI is still open (hi None):
    # that must be flagged; the -1 direction crossed the threshold cleanly (a normal stop),
    # so it contributes no note.
    stops = [(1, 'reached max grid points', 40), (-1, 'crossed Delta chi2 threshold', 12)]
    notes = _coverage_notes(stops, lo=-0.5, hi=None, lo_at_bound=False, hi_at_bound=False)
    assert len(notes) == 1 and 'upper side' in notes[0] and 'grid-point cap' in notes[0]


def test_coverage_notes_silent_when_the_capped_side_still_closed():
    # A direction that hit the cap but whose side DID cross (hi finite) is fully covered --
    # no caveat needed.
    stops = [(1, 'reached max grid points', 40)]
    assert _coverage_notes(stops, lo=None, hi=2.0, lo_at_bound=False, hi_at_bound=False) == []


# --------------------------------------------------------------------------- #
# Profile-curve plots (matplotlib, an optional extra -- #467)
# --------------------------------------------------------------------------- #
class _StubVar:
    """Minimal stand-in for a FreeParameter carrying just what the plotter reads:
    ``name``, ``log_space``, and the ``u -> theta`` map."""

    def __init__(self, name, log_space=False):
        self.name = name
        self.log_space = log_space

    def from_sampling_space(self, u):
        return 10.0 ** u if self.log_space else u


def _plot_summary():
    """A three-panel summary exercising each panel branch: an identifiable parameter (finite
    two-sided CI), a practically non-identifiable one (open upper CI), and one with no finite
    profile points (the empty-panel fallback)."""
    u = np.linspace(-1.5, 1.5, 31)
    ident = {
        'name': 'k', 'best': 0.0, 'ci_low': -0.7, 'ci_high': 0.7,
        'lo_at_bound': False, 'hi_at_bound': False, 'classification': 'identifiable',
        'notes': [], 'u': u, 'dchi2': u ** 2, 'success': np.ones_like(u, dtype=bool),
    }
    practical = {
        'name': 'S0', 'best': 0.0, 'ci_low': -0.7, 'ci_high': None,
        'lo_at_bound': False, 'hi_at_bound': False,
        'classification': 'practically non-identifiable', 'notes': ['upper side stopped'],
        'u': u, 'dchi2': np.maximum(u, 0.0) ** 2, 'success': np.ones_like(u, dtype=bool),
    }
    empty = {
        'name': 'kf', 'best': 1.0, 'ci_low': None, 'ci_high': None,
        'lo_at_bound': False, 'hi_at_bound': False,
        'classification': 'practically non-identifiable', 'notes': [],
        'u': np.array([0.0, 0.1]), 'dchi2': np.array([np.inf, np.inf]),
        'success': np.array([False, False]),
    }
    return [ident, practical, empty]


def test_render_profile_plots_writes_a_png(tmp_path):
    pytest.importorskip('matplotlib')
    path = str(tmp_path / 'profile_likelihood.png')
    variables = [_StubVar('k'), _StubVar('S0', log_space=True), _StubVar('kf')]
    ok = _render_profile_plots(path, _plot_summary(), variables,
                               threshold=3.841458820694124, confidence=0.95)
    assert ok is True
    assert Path(path).is_file() and Path(path).stat().st_size > 0
    with open(path, 'rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'   # a real PNG, not an empty/renamed file


def test_render_profile_plots_fails_soft_without_matplotlib(tmp_path, monkeypatch):
    # With matplotlib un-importable, the plotter returns False (the caller keeps the text
    # artifacts and reports the skip) rather than raising -- plots are strictly optional.
    import sys
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    path = str(tmp_path / 'profile_likelihood.png')
    ok = _render_profile_plots(path, _plot_summary(), [_StubVar('k'), _StubVar('S0'),
                               _StubVar('kf')], threshold=3.84, confidence=0.95)
    assert ok is False
    assert not Path(path).exists()


# --------------------------------------------------------------------------- #
# _ProfileTrack driven offline against an analytic linear-Gaussian model
# --------------------------------------------------------------------------- #
def _linear_gaussian(A, y):
    """theta*, F_min, and covariance C = (A^T A)^{-1} of F = 1/2 ||A theta - y||**2."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    AtA = A.T @ A
    theta_star = np.linalg.solve(AtA, A.T @ y)
    f_min = 0.5 * float((A @ theta_star - y) @ (A @ theta_star - y))
    return theta_star, f_min, np.linalg.inv(AtA)


def _run_track(track, A, y, names):
    """Drive a track to termination, feeding it synthetic GradientResults, and return the
    (fixed_u, cost, theta_free, nfev, success) grid points it recorded."""
    A = np.asarray(A, dtype=float)
    u = track.start()
    guard = 0
    while u is not None:
        guard += 1
        assert guard < 10000, 'profile track did not terminate'
        r = A @ u - y
        score = 0.5 * float(r @ r)
        full = GradientResult(residual=r, jacobian=A, gradient=A.T @ r,
                              param_names=list(names), least_squares_exact=True)
        reduced = ProfileLikelihoodAlgorithm._reduce_gradient(full, track.free_idx)
        u = track.got(u[track.free_idx], score, reduced)
    return track.points


def _profile_param(A, y, theta_star, f_min, idx, lower, upper, threshold, names,
                   *, runner_kind='trf'):
    """Assemble a full two-sided profile of parameter ``idx`` by running both directional
    tracks against the analytic model; return sorted (u, dchi2). ``runner_kind`` selects the
    inner re-optimizer (``'trf'`` or the scalar-gradient ``'lbfgs'``)."""
    n = len(theta_star)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    fixed_u = [float(theta_star[idx])]
    cost = [f_min]
    for direction in (1, -1):
        warm = np.array([theta_star[i] for i in range(n) if i != idx], dtype=float)
        track = _ProfileTrack(
            idx, direction, theta_star, lower, upper, warm, f_min,
            step=0.05, min_step=1e-3, max_step=0.5, dchi2_target=threshold / 10.0,
            threshold=threshold, max_points=400, reopt_max_iterations=100,
            grad_tol=1e-10, step_tol=1e-12, runner_kind=runner_kind)
        for fu, c, _theta_free, _nfev, _success in _run_track(track, A, y, names):
            fixed_u.append(fu)
            cost.append(c)
    u = np.array(fixed_u)
    order = np.argsort(u)
    u = u[order]
    dchi2 = 2.0 * (np.array(cost)[order] - f_min)
    return u, dchi2


@pytest.mark.parametrize('runner_kind', ['trf', 'lbfgs'])
def test_profile_track_pickles_mid_walk(runner_kind):
    """The track (and its inner runner -- TRF or L-BFGS-B) is plain numpy/float/list, so the
    optimizer that owns it checkpoints for backup (ADR-0007). Round-trip a track mid-walk and
    confirm it resumes to the same grid points, for both inner-runner kinds."""
    import pickle
    A = np.array([[1.0, 0.2], [0.1, 1.0], [0.4, 0.3]])
    y = np.array([1.0, 2.0, 0.5])
    theta_star, f_min, _ = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    track = _ProfileTrack(
        0, 1, theta_star, theta_star - 5.0, theta_star + 5.0,
        np.array([theta_star[1]]), f_min, step=0.05, min_step=1e-3, max_step=0.5,
        dchi2_target=0.4, threshold=3.84, max_points=50, reopt_max_iterations=100,
        grad_tol=1e-10, step_tol=1e-12, runner_kind=runner_kind)
    # Advance a couple of inner evaluations, then pickle mid-walk.
    u = track.start()
    for _ in range(3):
        r = A @ u - y
        full = GradientResult(residual=r, jacobian=A, gradient=A.T @ r,
                              param_names=names, least_squares_exact=True)
        reduced = ProfileLikelihoodAlgorithm._reduce_gradient(full, track.free_idx)
        u = track.got(u[track.free_idx], 0.5 * float(r @ r), reduced)
        if u is None:
            break
    restored = pickle.loads(pickle.dumps(track))
    # points are (fixed_u, cost, theta_free, nfev, success) -- compare field by field, since
    # the theta_free element is a numpy array (a bare tuple == would be ambiguous).
    assert len(restored.points) == len(track.points)
    for pr, pt in zip(restored.points, track.points):
        assert pr[0] == pt[0] and pr[1] == pt[1]
        assert np.array_equal(pr[2], pt[2])
        assert pr[3] == pt[3] and pr[4] == pt[4]
    assert restored.fixed_u == track.fixed_u


def test_track_records_per_point_theta_nfev_success():
    """Each recorded grid point carries the state #467 persists for resume/analysis: the
    re-optimized free coordinates, the inner iteration count, and a convergence flag. On the
    exactly-quadratic linear-Gaussian objective the reduced TRF re-optimization converges at
    every point, so every ``success`` is True and every ``theta_free`` has the free
    dimension."""
    A = np.array([[1.0, 0.2], [0.1, 1.0], [0.4, 0.3], [0.0, 0.7]])
    y = np.array([1.0, 2.0, 0.5, 1.5])
    theta_star, f_min, _ = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    track = _ProfileTrack(
        0, 1, theta_star, theta_star - 10.0, theta_star + 10.0,
        np.array([theta_star[1]]), f_min, step=0.05, min_step=1e-3, max_step=0.5,
        dchi2_target=_chi2_quantile_1dof(0.95) / 10.0, threshold=_chi2_quantile_1dof(0.95),
        max_points=400, reopt_max_iterations=100, grad_tol=1e-10, step_tol=1e-12)
    points = _run_track(track, A, y, names)
    assert points, 'the track recorded at least one grid point'
    for fixed_u, cost, theta_free, nfev, success in points:
        assert isinstance(theta_free, np.ndarray) and theta_free.shape == (1,)  # one free coord
        assert isinstance(nfev, int) and nfev >= 0
        assert success is True         # a quadratic re-opt always converges before its cap
        assert np.isfinite(cost) and np.isfinite(fixed_u)


def test_track_recovers_the_analytic_ci_on_an_identifiable_2d_model():
    # Two well-separated, correlated directions -> both parameters identifiable.
    A = np.array([[1.0, 0.2],
                  [0.1, 1.0],
                  [0.4, 0.3],
                  [0.0, 0.7]])
    y = np.array([1.0, 2.0, 0.5, 1.5])
    theta_star, f_min, C = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    thr = _chi2_quantile_1dof(0.95)
    lower = theta_star - 10.0
    upper = theta_star + 10.0

    for idx in (0, 1):
        u, dchi2 = _profile_param(A, y, theta_star, f_min, idx, lower, upper, thr, names)
        lo, hi, lob, hib = _extract_ci(u, dchi2, float(theta_star[idx]), thr,
                                       lower[idx], upper[idx])
        half = np.sqrt(thr * C[idx, idx])
        assert lo == pytest.approx(theta_star[idx] - half, rel=2e-2)
        assert hi == pytest.approx(theta_star[idx] + half, rel=2e-2)
        assert _classify(u, dchi2, float(theta_star[idx]), lo, hi, lob, hib) == 'identifiable'


def test_track_recovers_the_analytic_ci_via_the_lbfgs_inner_path():
    # The Gap-1 inner-runner split, offline: driving the SAME analytic linear-Gaussian
    # model with runner_kind='lbfgs' (the _LBFGSRunner consuming only grad.gradient -- what
    # _reduce_gradient slices out) must recover the SAME closed-form profile CI as the trf
    # path above. The CI is a property of the objective's shape, not the optimizer, so a
    # correct scalar-gradient re-optimization reproduces it. This is the backend-free proof
    # that profile likelihood's L-BFGS-B path re-optimizes the remaining parameters
    # correctly (the recovery tier proves it end to end on an estimated-scale fit TRF
    # refuses).
    A = np.array([[1.0, 0.2],
                  [0.1, 1.0],
                  [0.4, 0.3],
                  [0.0, 0.7]])
    y = np.array([1.0, 2.0, 0.5, 1.5])
    theta_star, f_min, C = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    thr = _chi2_quantile_1dof(0.95)
    lower = theta_star - 10.0
    upper = theta_star + 10.0

    for idx in (0, 1):
        u, dchi2 = _profile_param(A, y, theta_star, f_min, idx, lower, upper, thr, names,
                                  runner_kind='lbfgs')
        lo, hi, lob, hib = _extract_ci(u, dchi2, float(theta_star[idx]), thr,
                                       lower[idx], upper[idx])
        half = np.sqrt(thr * C[idx, idx])
        assert lo == pytest.approx(theta_star[idx] - half, rel=2e-2)
        assert hi == pytest.approx(theta_star[idx] + half, rel=2e-2)
        assert _classify(u, dchi2, float(theta_star[idx]), lo, hi, lob, hib) == 'identifiable'


def test_track_flags_a_structurally_non_identifiable_parameter():
    # Columns 0 and 1 are identical -> only theta0 + theta1 is constrained; fixing theta0
    # and re-optimizing theta1 always recovers the minimum -> a flat profile.
    A = np.array([[1.0, 1.0, 0.0],
                  [1.0, 1.0, 0.3],
                  [0.5, 0.5, 1.0],
                  [0.2, 0.2, 0.8]])
    y = np.array([2.0, 2.2, 1.0, 0.9])
    # theta* is not unique; pick the min-norm solution as the profiling center.
    AtA = A.T @ A
    theta_star = np.linalg.lstsq(A, y, rcond=None)[0]
    f_min = 0.5 * float((A @ theta_star - y) @ (A @ theta_star - y))
    names = ['p0', 'p1', 'p2']
    thr = _chi2_quantile_1dof(0.95)
    lower = theta_star - 5.0
    upper = theta_star + 5.0

    u, dchi2 = _profile_param(A, y, theta_star, f_min, 0, lower, upper, thr, names)
    # The profile never rises: fixing p0, p1 compensates exactly.
    assert np.nanmax(dchi2) < 1e-3
    lo, hi, lob, hib = _extract_ci(u, dchi2, float(theta_star[0]), thr,
                                   lower[0], upper[0])
    assert _classify(u, dchi2, float(theta_star[0]), lo, hi, lob, hib) == \
        'structurally non-identifiable'

    # p2 (independent column) stays identifiable.
    u2, dchi2_2 = _profile_param(A, y, theta_star, f_min, 2, lower, upper, thr, names)
    lo2, hi2, lob2, hib2 = _extract_ci(u2, dchi2_2, float(theta_star[2]), thr,
                                       lower[2], upper[2])
    C22 = np.linalg.pinv(AtA)[2, 2]
    assert lo2 == pytest.approx(theta_star[2] - np.sqrt(thr * C22), rel=3e-2)
    assert hi2 == pytest.approx(theta_star[2] + np.sqrt(thr * C22), rel=3e-2)
    assert _classify(u2, dchi2_2, float(theta_star[2]), lo2, hi2, lob2, hib2) == 'identifiable'


def test_track_reports_an_open_ci_at_a_bound_without_clamping_silently():
    # A single-parameter parabola whose box truncates the upper side below threshold.
    A = np.array([[1.0], [1.0], [1.0]])
    y = np.array([0.0, 0.1, -0.1])
    theta_star, f_min, C = _linear_gaussian(A, y)
    names = ['p0']
    thr = _chi2_quantile_1dof(0.95)
    half = float(np.sqrt(thr * C[0, 0]))
    # Upper bound sits inside the CI; lower bound is generous.
    lower = np.array([theta_star[0] - 10.0])
    upper = np.array([theta_star[0] + 0.5 * half])

    u, dchi2 = _profile_param(A, y, theta_star, f_min, 0, lower, upper, thr, names)
    lo, hi, lob, hib = _extract_ci(u, dchi2, float(theta_star[0]), thr, lower[0], upper[0])
    assert lo == pytest.approx(theta_star[0] - half, rel=3e-2)   # left side crosses
    assert hib and hi == pytest.approx(upper[0], abs=1e-9)       # right side open @ bound
    assert _classify(u, dchi2, float(theta_star[0]), lo, hi, lob, hib) == \
        'practically non-identifiable'


# --------------------------------------------------------------------------- #
# Cross-parameter parallel orchestration + resume, driven offline (#467)
#
# The optimizer's parallel driver (the phase machine that farms directional tracks across
# the scheduler and routes results back by PSet name) is exercised WITHOUT bngsim by
# subclassing it over the analytic linear-Gaussian model: real _begin_profiling /
# _fill_tracks / _profile_got / _merge_track / _finalize, with only the three model-coupled
# seams (gradient_at + the u<->PSet plumbing) overridden. Everything is picklable (module-
# level classes, no lambdas), so the same harness proves backup/resume mid-profiling.
# --------------------------------------------------------------------------- #
class _UPoint:
    """A picklable stand-in for both an evaluated PSet and its Result: a named point that
    carries its ``u``-vector and (once scored) its objective."""

    def __init__(self, name, u):
        self.name = name
        self.u = np.asarray(u, dtype=float)
        self.score = None
        # A real Result carries per-experiment simdata; a failed simulation returns it as
        # None (the #492 sentinel the gradient path now guards). Non-None here = a successful
        # evaluation; a failed-simulation test sets it to None.
        self.simdata = {}

    @property
    def pset(self):
        return self


class _LinVar:
    """A linear-scale bounded free-variable stub (identity sampling space)."""

    def __init__(self, name, lo, hi):
        self.name = name
        self.bounded = True
        self.lower_bound = float(lo)
        self.upper_bound = float(hi)
        self.log_space = False

    def to_sampling_space(self, x):
        return x

    def from_sampling_space(self, u):
        return u


class _LinTrajectory:
    """A trajectory stub returning the known analytic optimum + minimum objective."""

    def __init__(self, theta_star, f_min, names):
        self._theta = {n: float(t) for n, t in zip(names, theta_star)}
        self._f = float(f_min)

    def best_score(self):
        return self._f

    def best_fit(self):
        return self._theta


class _OfflineProfileAlg(ProfileLikelihoodAlgorithm):
    """``ProfileLikelihoodAlgorithm`` wired to the analytic ``F = 1/2||A theta - y||**2``
    model, bypassing the bngsim gradient path so the parallel profiling orchestration runs
    offline and deterministically. Only the model-coupled seams are overridden."""

    def __init__(self, A, y, theta_star, f_min, lower, upper, names, res_dir,
                 max_parallel=0):
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.variables = [_LinVar(n, lo, hi) for n, lo, hi in zip(names, lower, upper)]
        self.n = len(names)
        self._u_lower = np.asarray(lower, dtype=float)
        self._u_upper = np.asarray(upper, dtype=float)
        self.refine = False
        # Kept picklable so __setstate__ can rebuild the (non-pickled) trajectory on resume,
        # standing in for the base's reload of sorted_params_backup.txt.
        self._theta_star = np.asarray(theta_star, dtype=float)
        self._f_min = float(f_min)
        self._names = list(names)
        self.trajectory = _LinTrajectory(theta_star, f_min, names)
        self.fit_type = 'profile_likelihood'
        self.confidence = 0.95
        self.threshold = _chi2_quantile_1dof(0.95)
        self.pl_step, self.pl_min_step, self.pl_max_step = 0.05, 1e-3, 0.5
        self.pl_dchi2_target = self.threshold / 10.0
        self.pl_max_points = 400
        self.pl_max_parallel = max_parallel
        self.reopt_max_iterations = 100
        self.grad_tol, self.step_tol = 1e-10, 1e-12
        self.probe_counter = 0
        self._runner_kind = 'trf'
        self.res_dir = res_dir
        self._profile_idxs = list(range(self.n))
        self._profiles = {}
        self._track_queue = []
        self._active_tracks = {}
        self.profile_summary = None
        self.phase = 'profile'

    def __setstate__(self, state):
        # The base __getstate__ drops 'trajectory' (should_pickle); the real resume reloads
        # it from the backup params file. Here we rebuild the analytic trajectory instead --
        # everything else (_profiles / _active_tracks / _track_queue / the tracks) rode the
        # pickle through the base's real filter, which is what this test proves.
        self.__dict__.update(state)
        self.trajectory = _LinTrajectory(self._theta_star, self._f_min, self._names)

    def gradient_at(self, res):
        # Mirror the real gradient_at's very first act -- iterating res.simdata -- so a
        # failed-simulation Result (simdata is None) crashes *here* exactly as the production
        # path does (the #492 AttributeError), if a caller ever assembles a gradient for one
        # instead of guarding it. For a successful eval the analytic gradient is built from
        # res.u; simdata is just the presence marker the real assembly reads.
        res.simdata.items()
        r = self.A @ res.u - self.y
        return GradientResult(residual=r, jacobian=self.A, gradient=self.A.T @ r,
                              param_names=[v.name for v in self.variables],
                              least_squares_exact=True)

    def _pset_from_u(self, u, name=''):
        return _UPoint(name, u)

    def _u_from_pset(self, pset):
        return np.asarray(pset.u, dtype=float)


def _score(alg, p):
    r = alg.A @ p.u - alg.y
    return 0.5 * float(r @ r)


def _pump(alg, work, steps=None):
    """Drive the async run loop over a pending-PSet list (FIFO, emulating as_completed).
    Returns ``(finished, max_inflight, evaluated)``: whether the run reached ``'STOP'``, the
    peak number of simultaneously in-flight tracks observed, and the number of evaluations
    consumed. Stops early after ``steps`` results (for the resume test) with the run live."""
    max_inflight = len(alg._active_tracks)
    done = 0
    while work:
        p = work.pop(0)
        p.score = _score(alg, p)
        decision = alg._profile_got(p)
        done += 1
        max_inflight = max(max_inflight, len(alg._active_tracks))
        if decision == 'STOP':
            return True, max_inflight, done
        work.extend(decision)
        if steps is not None and done >= steps:
            return False, max_inflight, done
    return True, max_inflight, done


def _lin_model_2d():
    A = np.array([[1.0, 0.2], [0.1, 1.0], [0.4, 0.3], [0.0, 0.7]])
    y = np.array([1.0, 2.0, 0.5, 1.5])
    theta_star, f_min, C = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    lower = theta_star - 10.0
    upper = theta_star + 10.0
    return A, y, theta_star, f_min, C, names, lower, upper


def _summary_cis(alg):
    return {s['name']: (s['ci_low'], s['ci_high'], s['classification'])
            for s in alg.profile_summary}


def test_parallel_orchestration_runs_tracks_concurrently(tmp_path):
    """The default (``max_parallel = 0``) farms every directional track at once: 2
    parameters -> 4 tracks all launched by _begin_profiling, so multiple re-optimizations are
    in flight simultaneously (the #467 deliverable). The recovered CIs still match the
    closed-form linear-Gaussian answer -- concurrency changes scheduling, not results."""
    A, y, theta_star, f_min, C, names, lower, upper = _lin_model_2d()
    alg = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names, str(tmp_path))
    assert alg._effective_parallel() == 4          # 2 params x 2 directions, uncapped
    work = list(alg._begin_profiling(theta_star))
    assert len(work) == 4 and len(alg._active_tracks) == 4   # all four launched at once
    finished, max_inflight, _ = _pump(alg, work)
    assert finished and max_inflight >= 2          # genuinely concurrent, not serialized
    thr = _chi2_quantile_1dof(0.95)
    for idx, name in enumerate(names):
        lo, hi, klass = _summary_cis(alg)[name]
        half = np.sqrt(thr * C[idx, idx])
        assert klass == 'identifiable'
        assert lo == pytest.approx(theta_star[idx] - half, rel=3e-2)
        assert hi == pytest.approx(theta_star[idx] + half, rel=3e-2)
    # The deliverable artifacts landed in Results/.
    assert (tmp_path / 'profile_likelihood_summary.txt').is_file()
    assert (tmp_path / 'profile_p0.txt').is_file() and (tmp_path / 'profile_p1.txt').is_file()


def test_max_parallel_cap_serializes_without_truncating_coverage(tmp_path):
    """A cap only limits concurrency -- the excess tracks queue and run as slots free, never
    dropped -- so a fully serial (cap = 1) run visits every track and yields byte-identical
    CIs to the fully-parallel run. Proves the cap never silently truncates coverage (#467)."""
    A, y, theta_star, f_min, C, names, lower, upper = _lin_model_2d()
    par = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names,
                             str(tmp_path / 'par'), max_parallel=0)
    os.makedirs(tmp_path / 'par'); os.makedirs(tmp_path / 'ser')
    _pump(par, list(par._begin_profiling(theta_star)))

    ser = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names,
                             str(tmp_path / 'ser'), max_parallel=1)
    assert ser._effective_parallel() == 1
    work = list(ser._begin_profiling(theta_star))
    assert len(work) == 1 and len(ser._active_tracks) == 1   # only one at a time
    finished, max_inflight, _ = _pump(ser, work)
    assert finished and max_inflight == 1                     # never more than one in flight
    assert _summary_cis(ser) == _summary_cis(par)             # identical coverage + CIs


def test_resume_after_pickle_midway_completes_without_recompute(tmp_path):
    """The whole optimizer -- finished tracks in ``_profiles``, in-flight tracks in
    ``_active_tracks``, the pending PSet queue -- pickles mid-profiling and resumes, exactly
    how PyBNF's backup/resume re-enters ``run(resume=...)``. The resumed run re-submits only
    the in-flight PSets and never recomputes a finished track, reaching the same CIs as a
    straight run."""
    import pickle
    A, y, theta_star, f_min, C, names, lower, upper = _lin_model_2d()

    straight = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names,
                                  str(tmp_path / 'straight'))
    os.makedirs(tmp_path / 'straight')
    _, _, n_straight = _pump(straight, list(straight._begin_profiling(theta_star)))
    expected = _summary_cis(straight)

    alg = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names,
                             str(tmp_path / 'resumed'))
    os.makedirs(tmp_path / 'resumed')
    work = list(alg._begin_profiling(theta_star))
    # Pump until an in-flight track has committed real progress (its first accepted profile
    # point), then pause there -- the mid-walk state worth pickling. A hard-coded step count is
    # fragile: the first point lands only once a track's first re-optimization converges, whose
    # inner-iteration count depends on the (SciPy) least-squares solver, so we stop on the
    # condition itself rather than a fixed evaluation number.
    n_partial = 0
    inflight_points = 0
    while inflight_points == 0:
        finished_partial, _, done = _pump(alg, work, steps=1)
        n_partial += done
        assert not finished_partial, 'run finished before any in-flight point was committed'
        inflight_points = sum(len(tr.points) for _, tr in alg._active_tracks.values())
    assert alg._active_tracks or alg._track_queue    # the run is not yet finished
    # In-flight tracks hold real, un-merged progress that must survive the pickle.
    assert inflight_points > 0

    # Backup: pickle (algorithm, pending PSets) through the base's real __getstate__ filter
    # (drops the trajectory, keeps _profiles / _active_tracks / _track_queue), exactly as
    # Algorithm.backup pickles (self, pending_psets).
    restored, restored_work = pickle.loads(pickle.dumps((alg, work)))
    del alg, work
    restored_inflight = sum(len(tr.points) for _, tr in restored._active_tracks.values())
    assert restored_inflight == inflight_points      # in-flight progress rode the pickle

    finished, _, n_resume = _pump(restored, list(restored_work))
    assert finished
    # No finished point is recomputed and none is skipped: the pause + resume evaluates
    # exactly as many points as the straight run, and lands on the same CIs.
    assert n_partial + n_resume == n_straight
    assert _summary_cis(restored) == expected


# --------------------------------------------------------------------------- #
# End-to-end through the real bngsim forward-sensitivity path (recovery tier)
# --------------------------------------------------------------------------- #
@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_profile_likelihood_recovers_identifiable_cis(tmp_path, monkeypatch):
    """The core #466 deliverable end to end: ``job_type = profile_likelihood`` polishes to
    the optimum of an exponential decay through the real bngsim sensitivity path, then
    traces one re-optimized profile per parameter. With informative zero-noise data both
    ``k`` and ``S0`` are identifiable -- a finite two-sided CI that brackets the optimum --
    and the driver classifies them so, writing the summary + curves to ``Results/``."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp', sd=2.0)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'profile_likelihood', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=100,
        profile_likelihood_confidence=0.95, profile_likelihood_step=0.03,
        profile_likelihood_max_points=15)

    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)

    assert alg.polished is True   # no initial_value given -> the polish ran
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.03
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.03

    summary = {s['name']: s for s in alg.profile_summary}
    for name, truth in (('k', TRUE_K), ('S0', TRUE_S0)):
        s = summary[name]
        assert s['classification'] == 'identifiable', (name, s['classification'])
        assert s['ci_low'] is not None and s['ci_high'] is not None
        assert not s['lo_at_bound'] and not s['hi_at_bound']
        # A proper two-sided CI brackets the optimum (which sits at ~truth).
        assert s['ci_low'] < s['best'] < s['ci_high']
        assert s['ci_low'] < truth < s['ci_high']

    # The deliverable text outputs land in Results/, now carrying the #467 richer state:
    # the curve files report per-point nfev + success, and the summary a notes column.
    res = Path(conf.config['output_dir']) / 'Results'
    assert (res / 'profile_likelihood_summary.txt').is_file()
    assert (res / 'profile_k.txt').is_file() and (res / 'profile_S0.txt').is_file()
    assert '\tnfev\tsuccess' in (res / 'profile_k.txt').read_text().splitlines()[0]
    assert '\tnotes' in (res / 'profile_likelihood_summary.txt').read_text().splitlines()[1]

    # The persisted per-point state #467 adds (rides the backup pickle for resume).
    for name in ('k', 'S0'):
        prof = alg._profiles[name]
        assert set(prof) >= {'fixed_u', 'cost', 'nfev', 'success', 'theta_opt'}
        assert len(prof['theta_opt']) == len(prof['cost'])
        assert all(v.shape == (len(alg.variables),) for v in prof['theta_opt'])

    # The profile plots (matplotlib is an optional extra -- only assert when it is present;
    # a bare env legitimately skips them and writes only the text artifacts).
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pass
    else:
        assert (res / 'profile_likelihood.png').is_file()


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_profile_likelihood_flags_a_structurally_nonidentifiable_parameter(tmp_path, monkeypatch):
    """The second #468 deliverable end to end: correctly flag a *deliberately*
    non-identifiable parameter through the real bngsim sensitivity path. The two-channel
    decay ``S(t) = S0*exp(-(k1+k2)*t)`` makes only the sum ``k1+k2`` observable, so fixing
    either rate and re-optimizing the other slides freely along the ``k1+k2 = const``
    manifold at no objective cost -- each rate profiles FLAT (``Delta chi2`` stays under the
    structural-flatness floor on the side with room to compensate), the D2D signature of
    **structural** non-identifiability. The intercept ``S0`` is set by ``t = 0`` alone and
    stays **identifiable**.

    :math:`\\theta^\\*` is *supplied* (an ``initial_value:`` on each parameter, the canonical
    profile-likelihood workflow -- you profile around a fit you already have). That centers the
    profiles in the interior of the flat manifold, where a genuinely structural profile shows
    as a broad plateau; letting the polish choose :math:`\\theta^\\*` instead would land it at
    an arbitrary edge of the manifold (the objective is flat along it, so there is no unique
    optimum). This is the end-to-end sibling of the offline
    ``test_track_flags_a_structurally_non_identifiable_parameter`` (identical residual-Jacobian
    columns), here driven by genuine forward sensitivities rather than a synthetic Jacobian."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _two_channel_decay_model(tmp_path)
    # k1+k2 = 0.3 and S0 = 100, so the analytic decay oracle is byte-for-byte the single-channel
    # one (TRUE_K = 0.3): the model is degenerate, the data is not.
    exp = _write_decay_exp(tmp_path / 'decay.exp', sd=2.0)

    # theta* supplied at an interior split of the manifold (0.15 + 0.15 = 0.3): each rate then
    # has room to compensate the other in both directions, so the flat plateau is traced cleanly.
    lines = [
        f'model: {model}',
        'edition = 2', 'job_type = profile_likelihood', 'objective = chi_sq',
        f'output_dir = {Path(tmp_path) / "out"}',
        'bngl_backend = bngsim', 'initialization = lh', 'delete_old_files = 1',
        'verbosity = 0', 'wall_time_sim = 0', 'random_seed = 1234',
        'population_size = 1', 'max_iterations = 100',
        'profile_likelihood_confidence = 0.95', 'profile_likelihood_step = 0.03',
        'profile_likelihood_max_points = 20',
        'parameter: k1, lower: 0.001, upper: 1.0, initial_value: 0.15',
        'parameter: k2, lower: 0.001, upper: 1.0, initial_value: 0.15',
        f'parameter: S0, lower: 20.0, upper: 400.0, initial_value: {TRUE_S0}',
        f'experiment: decay, data: {exp}',
    ]
    conf = Configuration(ploop('\n'.join(lines).splitlines(keepends=True)))

    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)

    assert alg.polished is False   # theta* supplied -> profile around it, no re-fit

    summary = {s['name']: s for s in alg.profile_summary}
    for rate in ('k1', 'k2'):
        s = summary[rate]
        assert s['classification'] == 'structurally non-identifiable', (rate, s['classification'])
    # S0 is pinned by the intercept -> a proper finite two-sided CI bracketing the truth.
    s0 = summary['S0']
    assert s0['classification'] == 'identifiable', s0['classification']
    assert s0['ci_low'] < TRUE_S0 < s0['ci_high']
    assert not s0['lo_at_bound'] and not s0['hi_at_bound']

    # The structural label is not a lucky single-point dip: a broad plateau was traced -- most
    # of each rate's re-optimized profile sits on the flat floor (Delta chi2 < the structural
    # threshold), the direct numeric witness of the flat manifold.
    ref = alg.trajectory.best_score()
    for rate in ('k1', 'k2'):
        dchi2 = 2.0 * (np.asarray(alg._profiles[rate]['cost'], dtype=float) - ref)
        assert np.mean(dchi2 < _FLAT_DCHI2) >= 0.4, (rate, dchi2)


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_profile_likelihood_profiles_an_estimated_scale_fit_via_lbfgs(tmp_path, monkeypatch):
    """The Gap-1 deliverable end to end: ``job_type = profile_likelihood`` on an
    objective TRF *refuses* -- ``chi_sq_dynamic`` with a free ``sigma__FREE`` (an
    **estimated** noise scale, ``least_squares_exact == False``). The preflight reads that
    off the assembled gradient and routes both the polish and every per-grid-point
    re-optimization through the scalar-gradient L-BFGS-B inner runner (never TRF, which
    would raise). The polish recovers the rate ``k``, and profiling it (re-optimizing the
    nuisance ``S0`` + ``sigma__FREE`` at each grid point) yields a finite CI that brackets
    the truth -- the estimated-scale sibling of
    ``test_profile_likelihood_recovers_identifiable_cis`` (which takes the TRF path)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    # No _SD column: chi_sq_dynamic reads its sigma from the free parameter, not the data.
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sigma__FREE': ('uniform_var', 0.1, 50.0)},
        'decay', 'profile_likelihood', objective='chi_sq_dynamic', random_seed=1234,
        population_size=1, max_iterations=250,
        profile_likelihood_params='k',
        profile_likelihood_confidence=0.95, profile_likelihood_step=0.02,
        profile_likelihood_max_points=25)

    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)   # must NOT raise -- this is the objective TRF refuses

    # The auto-detection chose the scalar-gradient path, never TRF.
    assert alg._runner_kind == 'lbfgs'
    assert alg.polished is True    # no initial_value -> the polish ran (through lbfgs)

    rec = H.best_params(alg, ('k',))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.05

    s = {p['name']: p for p in alg.profile_summary}['k']
    # A finite, sensible CI for the rate that brackets both the optimum and the truth.
    assert s['ci_low'] is not None and s['ci_high'] is not None
    assert s['ci_low'] < s['best'] < s['ci_high']
    assert s['ci_low'] < TRUE_K < s['ci_high']

    # Only k was profiled (profile_likelihood_params), and its curve landed in Results/.
    assert [p['name'] for p in alg.profile_summary] == ['k']
    res = Path(conf.config['output_dir']) / 'Results'
    assert (res / 'profile_k.txt').is_file()


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_profile_likelihood_uses_supplied_initial_value_and_skips_polish(tmp_path, monkeypatch):
    """The explicit-override path (#466 open design question): every parameter declares an
    ``initial_value:`` (the optimum from a prior fit), so the job takes those as ``theta*``,
    evaluates that one point for the reference objective, and profiles without re-fitting.
    The supplied values are the truth, so both parameters come out identifiable."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp', sd=2.0)

    lines = [
        f'model: {model}',
        'edition = 2', 'job_type = profile_likelihood', 'objective = chi_sq',
        f'output_dir = {Path(tmp_path) / "out"}',
        'bngl_backend = bngsim', 'initialization = lh', 'delete_old_files = 1',
        'verbosity = 0', 'wall_time_sim = 0', 'random_seed = 1234',
        'population_size = 1', 'max_iterations = 100',
        'profile_likelihood_confidence = 0.95', 'profile_likelihood_step = 0.03',
        'profile_likelihood_max_points = 15',
        f'parameter: k, lower: 0.01, upper: 3.0, initial_value: {TRUE_K}',
        f'parameter: S0, lower: 20.0, upper: 400.0, initial_value: {TRUE_S0}',
        f'experiment: decay, data: {exp}',
    ]
    conf = Configuration(ploop('\n'.join(lines).splitlines(keepends=True)))

    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)

    assert alg.polished is False   # initial_value on every parameter -> polish skipped
    summary = {s['name']: s for s in alg.profile_summary}
    for name, truth in (('k', TRUE_K), ('S0', TRUE_S0)):
        s = summary[name]
        assert s['classification'] == 'identifiable', (name, s['classification'])
        assert s['ci_low'] < truth < s['ci_high']


# --------------------------------------------------------------------------- #
# Agreement with examples/becker_d2d_gradient/ on the fast 2-parameter subset (#468)
# --------------------------------------------------------------------------- #
@pytest.mark.bngsim
@pytest.mark.bngsim_antimony
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_profile_likelihood_agrees_with_becker_d2d_on_the_fast_2p_subset(tmp_path, monkeypatch):
    """The third #468 deliverable: agreement with ``examples/becker_d2d_gradient/`` on a small
    subset -- the D2D "fast 2-parameter" variant (``kon`` / ``koff`` of the Becker EpoR model,
    BioModels ``BIOMD0000000271``). ``job_type = profile_likelihood`` drives the SAME verbatim
    published model through the SAME real Antimony forward-sensitivity path the ``trf`` becker
    smoke test uses (``tests/test_gradient_optimizer.py``), but profiles rather than merely
    fitting: it polishes to the optimum, then traces a re-optimized profile per rate.

    The data is a zero-noise ``Epo_EpoR`` time course generated at the published nominal rates,
    so the polish recovers ``kon`` / ``koff`` to the nominal point (the "agreement" on the fit).
    The identifiability structure it then reports is the D2D lesson of this subset: from this one
    observable ``kon`` is **identifiable** (a finite, two-sided CI bracketing the optimum) while
    ``koff`` is **practically non-identifiable** -- its profile never rises to the threshold
    inside the prior box, so the CI is reported *open at the bounds* rather than silently pinned
    to a spurious finite interval. (A *smoke*-scoped recovery like the sibling ``trf`` becker
    tests: the point is the SBML/Antimony sensitivity path end to end, not a tight EpoR fit.)"""
    from pybnf.bngsim_antimony_model import BngsimAntimonyModelNoTimeout

    H.install(monkeypatch)
    model = SBML_DIR / 'becker_epor.ant'
    # Zero-noise Epo_EpoR time course at the published nominal kon/koff -> the oracle.
    data = _simulate_becker(tmp_path, model, BngsimAntimonyModelNoTimeout)
    exp = _write_becker_exp(tmp_path, data['time'], data['Epo_EpoR'], 'Epo_EpoR')
    conf = H.make_newera_config(
        tmp_path, str(model), exp, _BECKER_FREE, 'tc', 'profile_likelihood',
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=60,
        profile_likelihood_confidence=0.95, profile_likelihood_step=0.05,
        profile_likelihood_max_points=25)

    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)

    # The full SBML/Antimony sensitivity path ran to completion: an exact-least-squares fit
    # (fixed-SD Gaussian) -> the trust-region inner runner, then two profiles.
    assert alg.polished is True and alg._runner_kind == 'trf'
    summary = {s['name']: s for s in alg.profile_summary}
    assert set(summary) == {'kon', 'koff'}

    # Agreement on the fit: the polish recovers the nominal fast-2p rates from the zero-noise data.
    for name in ('kon', 'koff'):
        assert abs(summary[name]['best'] - _BECKER_NOMINAL[name]) / _BECKER_NOMINAL[name] < 1e-2

    # kon: identifiable -- a finite two-sided CI (both crossings genuine, neither at a bound)
    # that brackets the recovered optimum.
    kon = summary['kon']
    assert kon['classification'] == 'identifiable', kon['classification']
    assert kon['ci_low'] is not None and kon['ci_high'] is not None
    assert not kon['lo_at_bound'] and not kon['hi_at_bound']
    assert kon['ci_low'] < kon['best'] < kon['ci_high']

    # koff: practically non-identifiable -- the profile stays under the threshold across the whole
    # prior box, so the CI is reported open AT the bounds (never silently closed, #446), and the
    # box edges are what land in the summary (not None, not a spurious finite crossing).
    koff = summary['koff']
    kv = next(v for v in alg.variables if v.name == 'koff')
    assert koff['classification'] == 'practically non-identifiable', koff['classification']
    assert koff['lo_at_bound'] and koff['hi_at_bound']
    assert koff['ci_low'] == pytest.approx(kv.lower_bound)
    assert koff['ci_high'] == pytest.approx(kv.upper_bound)

    # The per-rate curves + the summary landed in Results/ (the #468 artifacts).
    res = Path(conf.config['output_dir']) / 'Results'
    assert (res / 'profile_kon.txt').is_file() and (res / 'profile_koff.txt').is_file()
    assert (res / 'profile_likelihood_summary.txt').is_file()


# --------------------------------------------------------------------------- #
# Failed-simulation robustness (#492). The profile-likelihood path shares the gradient
# path's `gradient_at`, so a non-integrable candidate point (a bngsim CVODE failure ->
# res.simdata is None) has the same crash exposure -- and profiling deliberately pushes the
# fixed parameter to extremes, so it is *more* likely to walk into one. Driven offline
# through the analytic linear-Gaussian harness (no backend), so the guards are exercised
# deterministically.
# --------------------------------------------------------------------------- #
def _pump_failing_at(alg, work, fail_at):
    """Like :func:`_pump`, but the ``fail_at``-th evaluation (0-indexed) arrives as a failed
    simulation (``simdata=None``, ``score=inf``) -- how the gradient path forwards a
    non-integrable point. Returns ``(finished, evaluated)``; the run must survive and reach
    ``'STOP'``."""
    done = 0
    while work:
        p = work.pop(0)
        if done == fail_at:
            p.simdata, p.score = None, float('inf')
        else:
            p.score = _score(alg, p)
        decision = alg._profile_got(p)
        done += 1
        if decision == 'STOP':
            return True, done
        work.extend(decision)
    return True, done


def test_profile_got_absorbs_a_failed_grid_point_simulation_without_crashing(tmp_path):
    """A failed simulation at a track's very first probe (``res.simdata is None``) must not
    crash ``_profile_got`` dereferencing the absent simdata (the #492 gradient-path bug,
    shared here). It is fed to the track's inner runner as a non-finite, gradient-less
    evaluation; the inner runner terminates that grid point at the ``inf`` penalty and the run
    continues with the other tracks (or STOPs) -- never raises."""
    A, y, theta_star, f_min, C, names, lower, upper = _lin_model_2d()
    alg = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names, str(tmp_path))
    work = list(alg._begin_profiling(theta_star))
    first = work[0]
    first.simdata, first.score = None, float('inf')     # a non-integrable first probe
    decision = alg._profile_got(first)                  # no AttributeError
    assert decision == 'STOP' or isinstance(decision, list)


def test_a_failed_simulation_mid_profile_does_not_abort_the_run(tmp_path):
    """A single failed simulation partway through profiling is absorbed (the track backs off,
    or ends that grid point at the ``inf`` penalty), the run completes, and a full profile
    summary is still produced for every parameter -- rather than aborting the whole fit with
    the #492 AttributeError."""
    A, y, theta_star, f_min, C, names, lower, upper = _lin_model_2d()
    alg = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names, str(tmp_path))
    finished, evaluated = _pump_failing_at(alg, list(alg._begin_profiling(theta_star)),
                                           fail_at=3)
    assert finished and evaluated > 3
    assert alg.profile_summary is not None
    assert {s['name'] for s in alg.profile_summary} == set(names)


def test_select_runner_kind_refuses_a_reference_point_that_failed_to_simulate(tmp_path):
    """The reference point (box center for a polish, or the supplied theta*) is the anchor
    the whole profile is built around. If it does not simulate (``res.simdata is None``),
    ``_select_runner_kind`` must refuse fast with an actionable :class:`PybnfError` -- naming
    the non-integrable reference point -- instead of dereferencing the absent simdata (#492)."""
    from pybnf.printing import PybnfError
    A, y, theta_star, f_min, C, names, lower, upper = _lin_model_2d()
    alg = _OfflineProfileAlg(A, y, theta_star, f_min, lower, upper, names, str(tmp_path))
    alg.phase = 'center'
    failed = _UPoint('pl_center', theta_star)
    failed.simdata = None
    with pytest.raises(PybnfError, match='(?i)reference point'):
        alg._select_runner_kind(failed)


def test_track_stops_cleanly_at_a_non_integrable_slice():
    """When the outward walk reaches a fixed-parameter value the model cannot integrate at (a
    failed simulation -> cost inf, no gradient), the track stops that direction *at* the
    non-integrable boundary instead of marching further into the region. The boundary point is
    recorded unsuccessful (inf cost, ``success=False``), and the finite filter in CI extraction
    drops it -- so the side reads as not cleanly crossed rather than crashing or looping to
    ``max_points`` (#492)."""
    A = np.array([[1.0, 0.2], [0.1, 1.0], [0.4, 0.3], [0.0, 0.7]])
    y = np.array([1.0, 2.0, 0.5, 1.5])
    theta_star, f_min, _ = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    fail_beyond = float(theta_star[0]) + 0.5     # model integrates only within 0.5 of theta*
    track = _ProfileTrack(
        0, 1, theta_star, theta_star - 10.0, theta_star + 10.0,
        np.array([theta_star[1]]), f_min, step=0.05, min_step=1e-3, max_step=0.5,
        dchi2_target=0.4, threshold=1e9,          # a threshold the finite profile never crosses
        max_points=400, reopt_max_iterations=100, grad_tol=1e-10, step_tol=1e-12)
    u = track.start()
    guard = 0
    while u is not None:
        guard += 1
        assert guard < 10000, 'profile track did not terminate'
        if u[track.param_idx] > fail_beyond:                       # a non-integrable slice
            u = track.got(u[track.free_idx], float('inf'), None)   # failed sim: inf, no grad
        else:
            r = A @ u - y
            full = GradientResult(residual=r, jacobian=A, gradient=A.T @ r,
                                  param_names=names, least_squares_exact=True)
            reduced = ProfileLikelihoodAlgorithm._reduce_gradient(full, track.free_idx)
            u = track.got(u[track.free_idx], 0.5 * float(r @ r), reduced)
    assert track.stop_reason == 'reached a non-integrable point (simulation failed)'
    assert not np.isfinite(track.points[-1][1])   # the boundary point's cost is inf
    assert track.points[-1][4] is False           # ... and it is marked unsuccessful
    assert len(track.points) < 400                # stopped at the wall, not at max_points
    # Every point strictly before the wall was a normal, successful, finite re-optimization.
    for _fu, cost, _theta, _nfev, success in track.points[:-1]:
        assert np.isfinite(cost) and success is True
