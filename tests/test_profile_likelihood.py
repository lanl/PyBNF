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

from pathlib import Path

import numpy as np
import pytest

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.algorithms.optimizers.profile_likelihood import (
    ProfileLikelihoodAlgorithm,
    _ProfileTrack,
    _chi2_quantile_1dof,
    _classify,
    _extract_ci,
    _norm_ppf,
    _resolve_profile_idxs,
)
from pybnf.config import Configuration
from pybnf.gradient import GradientResult
from pybnf.parse import ploop

from . import recovery_harness as H

BNGL_DIR = Path(__file__).resolve().parent / 'bngl_files'
# Truth for the synthetic decay data (the decay model's own nominal values).
TRUE_K = 0.3
TRUE_S0 = 100.0


def _write_decay_exp(path, *, n=21, t_end=10.0, sd=2.0):
    """A zero-noise analytic decay ``.exp`` (``time Stot Stot_SD``): ``Stot = S0*exp(-k*t)``
    at the truth, with a constant SD column so the chi-square is a fixed-scale Gaussian (an
    exact least-squares residual -- the profile driver's TRF re-optimization needs it)."""
    t = np.linspace(0.0, t_end, n)
    obs = TRUE_S0 * np.exp(-TRUE_K * t)
    lines = ['#\ttime\tStot\tStot_SD']
    lines += ['%.12g\t%.12g\t%.12g' % (ti, oi, sd) for ti, oi in zip(t, obs)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _decay_model(tmp_path):
    return H.strip_actions_block(BNGL_DIR / 'e2e_ode_decay.bngl', tmp_path / 'decay_v2.bngl')


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
    with pytest.raises(PybnfError):
        _resolve_profile_idxs(variables, ['a', 'zzz'])


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
    (fixed_u, cost) grid points it recorded."""
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


def _profile_param(A, y, theta_star, f_min, idx, lower, upper, threshold, names):
    """Assemble a full two-sided profile of parameter ``idx`` by running both directional
    tracks against the analytic model; return sorted (u, dchi2)."""
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
            grad_tol=1e-10, step_tol=1e-12)
        for fu, c in _run_track(track, A, y, names):
            fixed_u.append(fu)
            cost.append(c)
    u = np.array(fixed_u)
    order = np.argsort(u)
    u = u[order]
    dchi2 = 2.0 * (np.array(cost)[order] - f_min)
    return u, dchi2


def test_profile_track_pickles_mid_walk():
    """The track (and its inner runner) is plain numpy/float, so the optimizer that owns it
    checkpoints for backup (ADR-0007). Round-trip a track mid-walk and confirm it resumes to
    the same grid points."""
    import pickle
    A = np.array([[1.0, 0.2], [0.1, 1.0], [0.4, 0.3]])
    y = np.array([1.0, 2.0, 0.5])
    theta_star, f_min, _ = _linear_gaussian(A, y)
    names = ['p0', 'p1']
    track = _ProfileTrack(
        0, 1, theta_star, theta_star - 5.0, theta_star + 5.0,
        np.array([theta_star[1]]), f_min, step=0.05, min_step=1e-3, max_step=0.5,
        dchi2_target=0.4, threshold=3.84, max_points=50, reopt_max_iterations=100,
        grad_tol=1e-10, step_tol=1e-12)
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
    assert restored.points == track.points
    assert restored.fixed_u == track.fixed_u


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

    # The deliverable text outputs land in Results/.
    res = Path(conf.config['output_dir']) / 'Results'
    assert (res / 'profile_likelihood_summary.txt').is_file()
    assert (res / 'profile_k.txt').is_file() and (res / 'profile_S0.txt').is_file()


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
