"""Headless gradient runners: offline step-math validation (#386 multi-start).

The gradient leaves' step machines are factored into picklable, backend-free
:class:`~pybnf.algorithms.optimizers.gradient_base.GradientRunner`\\ s (``_TRFRunner`` /
``_LBFGSRunner``) so a fit can run ``N`` of them concurrently (local multi-start). Being
pure ``u``-space ``numpy`` -- no PSets, objective, routing, or bngsim -- a runner can be
driven directly from an analytic objective, which is what this module does:

* **runner parity** -- drive the *real* runners on bounded quadratics and cross-check the
  converged point against scipy (the test oracle; scipy is banned only from the production
  run loop, ADR-0007, not from tests). The L-BFGS-B runner is checked with **several
  bounds active at the optimum**, where its active-set machinery (generalized Cauchy point
  + subspace minimization) actually bites;
* **the multi-start win** -- a genuinely multimodal objective where a single start from the
  box center descends into a shallow local basin, but scattering several starts and keeping
  the global best recovers the deep basin. This is the case local multi-start exists for (a
  gradient method only ever descends into the basin its start lands in), proven here without
  a simulation backend; the bngsim end-to-end version is in ``test_gradient_optimizer.py``;
* **picklability** -- a runner round-trips through pickle mid-search (the backup/resume
  contract every start must honor);
* **failed-simulation robustness** (#492) -- a non-integrable candidate point (a bngsim
  CVODE failure) arrives at the runner as ``grad=None`` with a non-finite score. Mid-search
  the runner backs off (shrinks the step / trust region) and still converges; at the start
  point it terminates *that* start cleanly. Driven both directly and through the real
  ``GradientOptimizer._advance``, which must not dereference a failed result's ``simdata``.

These run in the default suite (no ``BNG2.pl`` / bngsim needed), unlike the end-to-end
recovery tier in ``test_gradient_optimizer.py``.
"""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares, minimize

from pybnf.algorithms.optimizers.gradient_base import DONE, GradientOptimizer
from pybnf.algorithms.optimizers.gntr import _GNTRRunner
from pybnf.algorithms.optimizers.lbfgs import _LBFGSRunner
from pybnf.algorithms.optimizers.trf import _TRFRunner


class _Grad:
    """Stand-in for an assembled GradientResult: carries exactly what each runner reads
    off it (``gradient`` for L-BFGS-B; ``residual`` / ``jacobian`` /
    ``least_squares_exact`` for TRF; ``gradient`` / ``hessian`` for GNTR)."""

    def __init__(self, gradient, residual=None, jacobian=None, least_squares_exact=True,
                 hessian=None):
        self.gradient = gradient
        self.residual = residual
        self.jacobian = jacobian
        self.least_squares_exact = least_squares_exact
        self.hessian = hessian


def _drive_scalar(runner, f_and_grad, max_evals=4000):
    """Drive a runner that consumes a scalar gradient (L-BFGS-B) from an analytic
    ``f_and_grad(u) -> (score, gradient)``."""
    u = runner.start()
    for _ in range(max_evals):
        score, grad = f_and_grad(u)
        nxt = runner.got(u, score, _Grad(grad))
        if nxt is DONE:
            return runner
        u = nxt
    raise AssertionError('runner did not terminate within %d evaluations' % max_evals)


def _drive_least_squares(runner, r_and_jac, max_evals=4000):
    """Drive a runner that consumes a residual + Jacobian (TRF) from an analytic
    ``r_and_jac(u) -> (residual, jacobian)``; the score is ``0.5||r||**2`` (the model
    F the assembly reports, GradientResult docstring)."""
    u = runner.start()
    for _ in range(max_evals):
        r, jac = r_and_jac(u)
        score = 0.5 * float(r @ r)
        nxt = runner.got(u, score, _Grad(jac.T @ r, residual=r, jacobian=jac,
                                         least_squares_exact=True))
        if nxt is DONE:
            return runner
        u = nxt
    raise AssertionError('runner did not terminate within %d evaluations' % max_evals)


def _drive_efim(runner, f_g_h, max_evals=4000):
    """Drive the GNTR (Fisher/Gauss-Newton trust-region) runner from an analytic
    ``f_g_h(u) -> (score, gradient, hessian)`` -- the general-objective EFIM it consumes: a
    scalar gradient + a PSD curvature Hessian, not a least-squares residual model."""
    u = runner.start()
    for _ in range(max_evals):
        score, grad, hess = f_g_h(u)
        nxt = runner.got(u, score, _Grad(grad, hessian=hess, least_squares_exact=False))
        if nxt is DONE:
            return runner
        u = nxt
    raise AssertionError('runner did not terminate within %d evaluations' % max_evals)


# A mildly coupled, well-conditioned least-squares quadratic F(u) = 0.5||D(u-u*)||**2.
_D = np.diag([1.0, 2.0, 0.5]) + 0.1 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
_LOWER = np.array([-3.0, -3.0, -3.0])
_UPPER = np.array([3.0, 3.0, 3.0])
_U0 = np.zeros(3)


def _ls_problem(ustar):
    A = _D.T @ _D

    def r_and_jac(u):
        return _D @ (u - ustar), _D

    def f_and_grad(u):
        d = u - ustar
        return 0.5 * float(d @ (A @ d)), A @ d

    return r_and_jac, f_and_grad


def test_trf_runner_matches_scipy_on_an_interior_least_squares_minimum():
    """The TRF (Trust-Region-Reflective) runner drives a bounded least-squares quadratic
    whose minimum is interior to the box to the same point ``scipy.least_squares`` finds
    -- in the interior the Coleman–Li scaling and reflection fall away, so this is an
    ordinary trust-region least-squares step, preserved by the headless extraction."""
    ustar = np.array([1.0, -0.5, 2.0])
    r_and_jac, _ = _ls_problem(ustar)
    runner = _drive_least_squares(
        _TRFRunner(_U0, _LOWER, _UPPER, 200, grad_tol=1e-10, step_tol=1e-12),
        r_and_jac)
    oracle = least_squares(lambda u: _D @ (u - ustar), _U0, jac=lambda u: _D,
                           bounds=(_LOWER, _UPPER))
    assert np.allclose(runner.point, ustar, atol=1e-5)
    assert np.allclose(runner.point, oracle.x, atol=1e-5)
    assert 'gradient is flat' in runner.stop_reason


def test_trf_runner_matches_scipy_with_several_bounds_active():
    """The case the Trust-Region-Reflective bound handling is for (#460): the
    unconstrained minimum lies far outside the box, so **all three** bounds are active at
    the constrained optimum (the box corner). The reflective transformation must slide the
    iterate cleanly onto that corner -- where plain clipping of the unconstrained step can
    stall -- matching ``scipy.optimize.least_squares(method='trf', bounds=…)``, the offline
    oracle for the bound-constrained least-squares step. The scaled-gradient optimality
    test ``‖v·Jᵀr‖∞`` reads as flat there (the Coleman–Li ``v`` vanishes on every active
    face), so the run stops on first-order optimality, not the budget."""
    ustar = np.array([10.0, -8.0, 9.0])     # every coordinate's optimum is outside the box
    r_and_jac, _ = _ls_problem(ustar)
    runner = _drive_least_squares(
        _TRFRunner(_U0, _LOWER, _UPPER, 500, grad_tol=1e-10, step_tol=1e-12),
        r_and_jac)
    oracle = least_squares(lambda u: _D @ (u - ustar), _U0, jac=lambda u: _D,
                           bounds=(_LOWER, _UPPER))
    assert np.allclose(runner.point, [3.0, -3.0, 3.0], atol=1e-6)   # the box corner
    assert np.allclose(runner.point, oracle.x, atol=1e-5)
    assert 'gradient is flat' in runner.stop_reason


# --------------------------------------------------------------------------- #
# GNTR (general-objective Fisher/Gauss-Newton trust region, #481). On a Gaussian
# least-squares problem its EFIM Hessian is H = J^T J and gradient g = J^T r, so the
# pseudo-Jacobian reduction reproduces trf's / scipy's step exactly. On a genuinely
# non-least-squares NLL (a curvature H that is not any J^T J) it takes a real trust-region
# Newton step, converging to the same minimum scipy.minimize finds.
# --------------------------------------------------------------------------- #
def _ls_efim(ustar):
    """The Gaussian least-squares EFIM: H = A = D^T D (= J^T J), g = A(u-ustar), the score
    the same 0.5||D(u-ustar)||^2 the residual form reports."""
    A = _D.T @ _D

    def f_g_h(u):
        d = u - ustar
        return 0.5 * float(d @ (A @ d)), A @ d, A

    return f_g_h


def test_gntr_runner_matches_trf_and_scipy_on_an_interior_least_squares_minimum():
    """On a Gaussian least-squares fit the EFIM Hessian is exactly ``J^T J``, so the GNTR runner
    -- built by feeding ``(g, H)`` through the pseudo-Jacobian into trf's Coleman-Li machinery --
    takes the SAME step as ``_TRFRunner`` and ``scipy.least_squares`` and reaches the same interior
    minimum. This is the reduction that anchors ``gntr`` to ``trf``'s step quality."""
    ustar = np.array([1.0, -0.5, 2.0])
    runner = _drive_efim(
        _GNTRRunner(_U0, _LOWER, _UPPER, 200, grad_tol=1e-10, step_tol=1e-12, ridge=1e-10),
        _ls_efim(ustar))
    oracle = least_squares(lambda u: _D @ (u - ustar), _U0, jac=lambda u: _D,
                           bounds=(_LOWER, _UPPER))
    assert np.allclose(runner.point, ustar, atol=1e-5)
    assert np.allclose(runner.point, oracle.x, atol=1e-5)
    assert 'gradient is flat' in runner.stop_reason


def test_gntr_runner_matches_scipy_with_several_bounds_active():
    """The bound-active corner (every coordinate's unconstrained optimum is outside the box): the
    inherited Coleman-Li reflective transformation slides the iterate cleanly onto the box corner,
    matching ``scipy.optimize.least_squares(method='trf')`` -- the same offline oracle ``trf`` uses,
    now reached through the EFIM ``(g, H)`` seam."""
    ustar = np.array([10.0, -8.0, 9.0])
    runner = _drive_efim(
        _GNTRRunner(_U0, _LOWER, _UPPER, 500, grad_tol=1e-10, step_tol=1e-12, ridge=1e-10),
        _ls_efim(ustar))
    oracle = least_squares(lambda u: _D @ (u - ustar), _U0, jac=lambda u: _D,
                           bounds=(_LOWER, _UPPER))
    assert np.allclose(runner.point, [3.0, -3.0, 3.0], atol=1e-6)   # the box corner
    assert np.allclose(runner.point, oracle.x, atol=1e-5)
    assert 'gradient is flat' in runner.stop_reason


def test_gntr_runner_converges_on_a_general_non_least_squares_nll():
    """The case ``gntr`` exists for: a genuinely non-least-squares objective whose Hessian is not
    any ``J^T J`` (a quadratic bowl plus an exponential ridge). The EFIM trust-region step drives
    it to the same interior minimum ``scipy.optimize.minimize`` finds -- the general-NLL trust
    region ``trf`` cannot do and ``lbfgs`` does only with a history Hessian."""
    A = np.diag([2.0, 1.0])
    a = np.array([0.5, -0.3])
    c = np.array([0.4, 0.2])
    lower, upper, u0 = np.array([-5.0, -5.0]), np.array([5.0, 5.0]), np.array([2.0, 2.0])

    def f_g_h(u):
        e = np.exp(c @ u)
        return (float(0.5 * (u - a) @ (A @ (u - a)) + e),
                A @ (u - a) + e * c,
                A + e * np.outer(c, c))     # a true PSD Hessian, != any J^T J

    runner = _drive_efim(
        _GNTRRunner(u0, lower, upper, 500, grad_tol=1e-11, step_tol=1e-13, ridge=1e-10), f_g_h)
    oracle = minimize(lambda u: f_g_h(u)[0], u0, jac=lambda u: f_g_h(u)[1],
                      method='L-BFGS-B', bounds=list(zip(lower, upper)),
                      options={'ftol': 1e-15, 'gtol': 1e-12})
    assert np.allclose(runner.point, oracle.x, atol=1e-5)


def test_gntr_runner_pickles_midway_through_a_search():
    """The GNTR runner is pure float/ndarray (inherited from ``_TRFRunner`` -- the point, the
    pseudo residual model, the trust radius + cached SVD), so it round-trips through pickle
    mid-search and resumes identically (the backup/resume contract, ADR-0007)."""
    ustar = np.array([1.0, -0.5, 2.0])
    f_g_h = _ls_efim(ustar)
    runner = _GNTRRunner(_U0, _LOWER, _UPPER, 200, grad_tol=1e-10, step_tol=1e-12, ridge=1e-10)
    u = runner.start()
    for _ in range(4):
        score, grad, hess = f_g_h(u)
        u = runner.got(u, score, _Grad(grad, hessian=hess, least_squares_exact=False))
        assert u is not DONE, 'expected the search to still be running here'
    revived = pickle.loads(pickle.dumps(runner))
    assert revived.iteration == runner.iteration
    assert revived.phase == runner.phase
    assert np.allclose(revived.point, runner.point)
    while u is not DONE:
        score, grad, hess = f_g_h(u)
        u = revived.got(u, score, _Grad(grad, hessian=hess, least_squares_exact=False))
    assert np.allclose(revived.point, ustar, atol=1e-4)


def test_gntr_refusal_points_at_lbfgs_not_a_metaheuristic():
    """A corner whose EFIM Hessian ``gntr`` cannot assemble (a MEDIAN count, a MEAN-on-log
    estimated scale, an estimated constraint scale) must refuse toward ``lbfgs`` -- which
    consumes the scalar gradient and fits it -- not toward a metaheuristic (the base's default
    hint). The leaf overrides ``_unsupported_gradient_error`` to redirect the pointer, which it
    appends to the diagnosis rather than printing in place of it (#527), so the message names
    both the corner that refused and the job_type that fits it."""
    from pybnf.algorithms.optimizers.gntr import GNTRAlgorithm
    from pybnf.gradient import GradientNotSupported
    from pybnf.printing import PybnfError
    err = GNTRAlgorithm._unsupported_gradient_error(
        object(), GradientNotSupported('a MEDIAN-centered negative-binomial ...'))
    assert isinstance(err, PybnfError)
    assert 'lbfgs' in err.message.lower()
    assert 'MEDIAN-centered negative-binomial' in err.message
    assert 'metaheuristic' not in err.message.lower()


def test_gntr_runner_fails_fast_without_an_attached_hessian():
    """The GNTR runner needs an assembled EFIM Hessian (GNTRAlgorithm attaches the data-fit
    block in _assemble_objective_gradient before the runner sees the gradient). If it is ever
    driven off the residual-form path with no hessian, it raises a clear PybnfError naming the
    wiring error rather than an opaque numpy failure deep in the eigen-factorisation."""
    from pybnf.printing import PybnfError
    runner = _GNTRRunner(_U0, _LOWER, _UPPER, 10, grad_tol=1e-8, step_tol=1e-8, ridge=1e-10)
    u = runner.start()
    with pytest.raises(PybnfError, match='(?i)hessian'):
        runner.got(u, 1.0, _Grad(np.zeros(3)))   # _Grad defaults hessian=None


def test_gntr_runner_accepts_any_objective_the_hessian_covers():
    """The GNTR runner overrides ``trf``'s exact-least-squares gate to a no-op: the curvature is
    the EFIM Hessian, not a residual, so a non-least-squares gradient (``least_squares_exact ==
    False``) is accepted (the real gate -- whether the Fisher Hessian could be assembled -- fired
    upstream). ``_require_exact`` returns the grad unchanged."""
    runner = _GNTRRunner(_U0, _LOWER, _UPPER, 10, grad_tol=1e-8, step_tol=1e-8, ridge=1e-10)
    grad = _Grad(np.zeros(3), hessian=np.eye(3), least_squares_exact=False)
    assert runner._require_exact(grad) is grad


def test_lbfgs_runner_matches_scipy_on_an_interior_minimum():
    """The L-BFGS-B runner reduces, in the box interior, to the unconstrained
    limited-memory quasi-Newton step, recovering the interior minimum (== scipy)."""
    ustar = np.array([1.0, -0.5, 2.0])
    _, f_and_grad = _ls_problem(ustar)
    runner = _drive_scalar(
        _LBFGSRunner(_U0, _LOWER, _UPPER, 200, grad_tol=1e-9, step_tol=1e-14,
                     history=10, c1=1e-4, backtrack=0.5),
        f_and_grad)
    assert np.allclose(runner.point, ustar, atol=1e-4)


def test_lbfgs_runner_matches_scipy_with_several_bounds_active():
    """The case the active-set machinery is for: the unconstrained minimum lies far
    outside the box so **all three** bounds are active at the constrained optimum. The
    L-BFGS-B runner's generalized Cauchy point + subspace minimization must drive each
    coordinate onto its bound, matching ``scipy.optimize.minimize(method='L-BFGS-B')``
    -- the offline oracle for the bound-constrained step."""
    ustar = np.array([10.0, -8.0, 9.0])     # every coordinate's optimum is outside the box
    _, f_and_grad = _ls_problem(ustar)
    runner = _drive_scalar(
        _LBFGSRunner(_U0, _LOWER, _UPPER, 500, grad_tol=1e-8, step_tol=1e-14,
                     history=10, c1=1e-4, backtrack=0.5),
        f_and_grad)
    oracle = minimize(lambda u: f_and_grad(u)[0], _U0, jac=lambda u: f_and_grad(u)[1],
                      method='L-BFGS-B', bounds=list(zip(_LOWER, _UPPER)),
                      options={'ftol': 1e-15, 'gtol': 1e-12})
    assert np.allclose(runner.point, [3.0, -3.0, 3.0], atol=1e-6)   # the box corner
    assert np.allclose(runner.point, oracle.x, atol=1e-4)


# --------------------------------------------------------------------------- #
# The multi-start win: a multimodal objective with a shallow basin at the box
# center and a deep basin toward a corner. Two inverted Gaussian wells + a gentle
# bowl; the box center sits in the shallow well, so a single (center) start traps
# there while scattered starts find the deep one.
# --------------------------------------------------------------------------- #
_MM_LOWER = np.array([-3.0, -3.0])
_MM_UPPER = np.array([3.0, 3.0])
_MM_LOCAL = np.array([0.0, 0.0])       # shallow well at the box center
_MM_GLOBAL = np.array([2.2, 2.2])      # deep well toward a corner


def _multimodal(u):
    bowl = 0.05 * float(u @ u)
    g_loc = -1.0 * np.exp(-((u - _MM_LOCAL) @ (u - _MM_LOCAL)) / 2.0)
    g_glob = -6.0 * np.exp(-((u - _MM_GLOBAL) @ (u - _MM_GLOBAL)) / 1.2)
    f = bowl + g_loc + g_glob
    grad = (0.1 * u
            - g_loc * (u - _MM_LOCAL)            # d/du of -A exp(-||.||^2 / 2)
            - g_glob * (u - _MM_GLOBAL) / 0.6)   # /(s^2/2), s^2 = 1.2
    return float(f), grad


def _converge_lbfgs_from(u0):
    return _drive_scalar(
        _LBFGSRunner(np.asarray(u0, float), _MM_LOWER, _MM_UPPER, 200, grad_tol=1e-7,
                     step_tol=1e-12, history=10, c1=1e-4, backtrack=0.5),
        _multimodal)


def test_multistart_escapes_the_basin_a_single_center_start_traps_in():
    """Local multi-start (keep the global best over ``N`` box-sampled starts) escapes a
    local basin that a single box-center start descends into and cannot leave. This is
    the headless analogue of the bngsim end-to-end multi-start recovery test, and proves
    the win is in the *method* (scatter + keep-best), not the backend."""
    single = _converge_lbfgs_from(_MM_LOCAL)   # box center -> shallow local well
    assert np.allclose(single.point, _MM_LOCAL, atol=0.1)

    # N starts: the box center (start 0) plus Latin-hypercube-style scatter from a
    # seeded rng (as GradientOptimizer draws them). Keep the global best.
    rng = np.random.default_rng(1234)
    starts = [_MM_LOCAL] + [_MM_LOWER + rng.random(2) * (_MM_UPPER - _MM_LOWER)
                            for _ in range(7)]
    finals = [_converge_lbfgs_from(s) for s in starts]
    best = min(finals, key=lambda r: r.fval)

    assert np.allclose(best.point, _MM_GLOBAL, atol=0.2), \
        'multi-start should find the deep (global) basin'
    assert best.fval < single.fval - 1.0, \
        'the global best across starts must clearly beat the single center start'


def test_runner_pickles_midway_through_a_search():
    """A runner is pure float/ndarray/list, so it round-trips through pickle mid-search
    and resumes identically -- the backup/resume contract every start must honor
    (ADR-0007). Uses the interior minimum, which takes enough iterations to freeze the
    runner well before it converges."""
    ustar = np.array([1.0, -0.5, 2.0])
    _, f_and_grad = _ls_problem(ustar)
    runner = _LBFGSRunner(_U0, _LOWER, _UPPER, 200, grad_tol=1e-9, step_tol=1e-14,
                          history=10, c1=1e-4, backtrack=0.5)
    # Take a few steps (still mid-search), then freeze and thaw.
    u = runner.start()
    for _ in range(4):
        score, grad = f_and_grad(u)
        u = runner.got(u, score, _Grad(grad))
        assert u is not DONE, 'expected the search to still be running here'
    revived = pickle.loads(pickle.dumps(runner))
    assert revived.iteration == runner.iteration
    assert revived.phase == runner.phase
    assert np.allclose(revived.point, runner.point)
    # Resume: feed the pending trial u (the in-flight evaluation) to the thawed runner
    # and continue to convergence -- exactly how run(resume=...) replays pending psets,
    # never re-calling start(). The thawed runner reaches the interior optimum.
    while u is not DONE:
        score, grad = f_and_grad(u)
        u = revived.got(u, score, _Grad(grad))
    assert np.allclose(revived.point, ustar, atol=1e-4)


# --------------------------------------------------------------------------- #
# Failed-simulation robustness (#492). A non-integrable candidate point (a bngsim CVODE
# integration failure, a NaN/Inf) returns from the gradient path with res.simdata is None
# and res.score = inf; GradientOptimizer._advance then feeds the runner grad=None with a
# non-finite score. The runner must treat that as a failed evaluation -- back off mid-search
# (never dereferencing the absent gradient), terminate the start when the start point itself
# fails -- rather than aborting the fit with the AttributeError of the issue.
# --------------------------------------------------------------------------- #
def _runner_factories():
    """One factory per gradient leaf, each seeded at the box center _U0."""
    return {
        'lbfgs': lambda u0: _LBFGSRunner(u0, _LOWER, _UPPER, 200, grad_tol=1e-9,
                                         step_tol=1e-14, history=10, c1=1e-4, backtrack=0.5),
        'trf': lambda u0: _TRFRunner(u0, _LOWER, _UPPER, 200, grad_tol=1e-10, step_tol=1e-12),
        'gntr': lambda u0: _GNTRRunner(u0, _LOWER, _UPPER, 200, grad_tol=1e-10,
                                       step_tol=1e-12, ridge=1e-10),
    }


@pytest.mark.parametrize('leaf', ['lbfgs', 'trf', 'gntr'])
def test_runner_terminates_the_start_when_the_start_point_fails_to_simulate(leaf):
    """A failed simulation at the START point (a non-integrable box center) arrives as
    ``grad=None`` with a non-finite score. With no gradient to model the local surface, the
    runner cannot descend, so it terminates *this* start cleanly with a clear stop_reason --
    instead of dereferencing the absent gradient (the #492 crash). Concurrent multi-start
    keeps every other start and the global best; a single-start fit ends here reporting no
    viable fit."""
    runner = _runner_factories()[leaf](_U0)
    u0 = runner.start()
    nxt = runner.got(u0, float('inf'), None)
    assert nxt is DONE
    assert 'start point failed to simulate' in runner.stop_reason
    assert runner.failure == 'simulation'      # distinguishable from an unusable model (#528)


def _drive_failing_first_trial(runner, evaluate, max_evals=4000):
    """Drive a runner where the FIRST proposed trial after the start point is a failed
    simulation (``grad=None``, ``score=inf`` -- exactly how ``_advance`` forwards it), then
    every later point evaluates normally via ``evaluate(u) -> (score, _Grad)``. Returns the
    converged runner; asserts it did not abort at the failed trial."""
    u = runner.start()
    for i in range(max_evals):
        if i == 1:                                  # the first trial after the start point
            nxt = runner.got(u, float('inf'), None)
        else:
            score, grad = evaluate(u)
            nxt = runner.got(u, score, grad)
        if nxt is DONE:
            assert i > 1, 'the runner aborted at the failed trial instead of backing off'
            return runner
        u = nxt
    raise AssertionError('runner did not terminate within %d evaluations' % max_evals)


def test_lbfgs_backs_off_a_failed_trial_and_still_converges():
    """A failed line-search trial (a non-integrable point met early in the search) makes the
    L-BFGS-B backtracking line search shrink the step -- its ``isfinite(score)`` guard rejects
    the trial without touching the (absent) gradient -- and the search still reaches the same
    interior minimum. The gradient method backs off, it does not abort (#492)."""
    ustar = np.array([1.0, -0.5, 2.0])
    _, f_and_grad = _ls_problem(ustar)
    runner = _drive_failing_first_trial(
        _runner_factories()['lbfgs'](_U0),
        lambda u: (f_and_grad(u)[0], _Grad(f_and_grad(u)[1])))
    assert np.allclose(runner.point, ustar, atol=1e-4)


def test_trf_shrinks_the_trust_region_on_a_failed_trial_and_still_converges():
    """A failed trust-region trial gives TRF its shrink-and-re-solve signal (the
    ``not isfinite(f_new)`` branch: ``Delta = 0.25*step_h_norm``, reject) without
    dereferencing the absent Jacobian, and the search still reaches the same interior
    minimum. Covers the inheriting ``gntr`` runner's reject path too (#492)."""
    ustar = np.array([1.0, -0.5, 2.0])
    r_and_jac, _ = _ls_problem(ustar)

    def evaluate(u):
        r, jac = r_and_jac(u)
        return 0.5 * float(r @ r), _Grad(jac.T @ r, residual=r, jacobian=jac,
                                         least_squares_exact=True)

    runner = _drive_failing_first_trial(_runner_factories()['trf'](_U0), evaluate)
    assert np.allclose(runner.point, ustar, atol=1e-5)


class _AdvanceHarness:
    """Drives the *real* :meth:`GradientOptimizer._advance` without a backend: supplies only
    the seams it touches -- the ``u``<->PSet identity and a ``_dispatch`` that echoes the
    routed point -- so a failed-simulation Result (``simdata=None``) exercises the real guard.
    ``gradient_at`` asserts if reached: a failed sim must never trigger gradient assembly."""

    fit_type = 'lbfgs'
    _advance = GradientOptimizer._advance

    def __init__(self):
        self.dispatched = []

    def _u_from_pset(self, pset):
        return np.asarray(pset, float)

    def _dispatch(self, idx, u):
        self.dispatched.append((idx, np.asarray(u, float)))
        return ('routed', idx)

    def gradient_at(self, res):
        raise AssertionError('gradient_at must not be called when res.simdata is None')


def test_advance_terminates_a_start_whose_start_point_failed_without_dereferencing_simdata():
    """Regression for #492: a failed simulation returns ``res.simdata=None`` (score inf). The
    real ``GradientOptimizer._advance`` must skip gradient assembly and feed the runner a
    non-finite, gradient-less evaluation -- terminating the start at the start point -- rather
    than dereferencing ``res.simdata.items()`` (the reported AttributeError)."""
    harness = _AdvanceHarness()
    runner = _runner_factories()['lbfgs'](_U0)
    runner.start()
    res = SimpleNamespace(pset=_U0, simdata=None, score=float('inf'))
    out = harness._advance(0, runner, res)
    assert out is DONE
    assert 'start point failed to simulate' in runner.stop_reason
    assert harness.dispatched == []            # the start terminated; nothing dispatched


def test_advance_backs_off_a_failed_mid_search_trial_and_dispatches_a_retry():
    """Mid-search, ``_advance`` forwards the failed evaluation to a runner already past its
    start; the runner backs off and proposes a shorter trial, which ``_advance`` dispatches --
    the fit continues rather than aborting (#492)."""
    harness = _AdvanceHarness()
    runner = _runner_factories()['lbfgs'](_U0)
    _, f_and_grad = _ls_problem(np.array([1.0, -0.5, 2.0]))
    u0 = runner.start()
    s0, g0 = f_and_grad(u0)
    trial = runner.got(u0, s0, _Grad(g0))      # advance past 'init' -> phase 'line'
    assert trial is not DONE
    res = SimpleNamespace(pset=np.asarray(trial), simdata=None, score=float('inf'))
    out = harness._advance(0, runner, res)
    assert isinstance(out, list) and len(out) == 1   # a (shorter) retry was dispatched
    assert len(harness.dispatched) == 1


# --------------------------------------------------------------------------- #
# Non-finite-model robustness (#528). A point can *score* finitely and still hand back
# non-finite derivatives -- a stiff parameter set whose ODE solve completes while its
# forward sensitivities diverge (a CVODE `flag=-3` on the sensitivity system), an overflow
# in the chain rule. That model is unusable, and unguarded it took down not the start that
# met it but the whole fit: LAPACK will not factorize a non-finite matrix (`LinAlgError: SVD
# did not converge` from the trust-region subproblem), and a quasi-Newton direction built
# from a NaN gradient is itself NaN, so the next proposed point is a NaN PSet
# (`OutOfBoundsException`). Either unwound out through got_result -- 19 healthy starts
# discarded because of one. Each leaf must instead treat an unusable model the way it already
# treats a failed simulation: back off mid-search, terminate just this start at the start
# point.
# --------------------------------------------------------------------------- #
_USTAR = np.array([1.0, -0.5, 2.0])


def _evaluators(leaf, ustar=_USTAR):
    """``(evaluate, spoil)`` for one leaf: ``evaluate(u)`` is the ordinary analytic
    ``(score, _Grad)`` of the shared quadratic; ``spoil(u)`` returns the SAME finite score
    with one non-finite entry in the model the leaf actually steps from -- the residual
    Jacobian for ``trf``, the EFIM gradient for ``gntr`` (the reported shape: a finite
    Hessian, a NaN gradient), the scalar gradient for ``lbfgs``."""
    r_and_jac, f_and_grad = _ls_problem(ustar)
    f_g_h = _ls_efim(ustar)

    def evaluate(u):
        if leaf == 'trf':
            r, jac = r_and_jac(u)
            return 0.5 * float(r @ r), _Grad(jac.T @ r, residual=r, jacobian=jac,
                                             least_squares_exact=True)
        if leaf == 'gntr':
            score, g, h = f_g_h(u)
            return score, _Grad(g, hessian=h, least_squares_exact=False)
        score, g = f_and_grad(u)
        return score, _Grad(g)

    def spoil(u):
        score, grad = evaluate(u)
        if leaf == 'trf':
            jac = np.array(grad.jacobian, dtype=float)
            jac[1, 1] = np.nan
            return score, _Grad(jac.T @ grad.residual, residual=grad.residual, jacobian=jac,
                                least_squares_exact=True)
        bad = np.array(grad.gradient, dtype=float)
        bad[1] = np.nan
        return score, _Grad(bad, hessian=grad.hessian,
                            least_squares_exact=grad.least_squares_exact)

    return evaluate, spoil


@pytest.mark.parametrize('leaf', ['lbfgs', 'trf', 'gntr'])
def test_runner_terminates_the_start_when_the_start_points_model_is_not_finite(leaf):
    """The reported crash (#528), at the runner: the START point scores finitely but its
    derivatives are not finite. There is no local surface to descend and no earlier iterate
    to fall back to, so the runner ends *this* start with a stop_reason naming the unusable
    model -- rather than letting the NaN into the step math, where it aborts the whole
    multi-start fit (``LinAlgError`` from the trust-region factorization, or the NaN point
    L-BFGS-B goes on to propose).

    The start reports the ``inf`` penalty, not the start point's own score: the point was
    scored but never *fitted*, and a consumer that reads a terminated runner's objective (the
    profile-likelihood grid point) must not take an unoptimized value for an optimized one.
    ``failure`` says which of the two failures it was, so such a consumer can name it."""
    _, spoil = _evaluators(leaf)
    runner = _runner_factories()[leaf](_U0)
    u0 = runner.start()
    score, grad = spoil(u0)
    assert np.isfinite(score), 'the fixture must score finitely; only the model is bad'
    nxt = runner.got(u0, score, grad)
    assert nxt is DONE
    assert 'not finite' in runner.stop_reason
    assert 'no usable local model' in runner.stop_reason
    assert not np.isfinite(runner.fval)    # scored, but not fitted
    assert runner.failure == 'model'       # ... and not a failed simulation


def _first_accepted_trial_index(runner, evaluate, max_evals=4000):
    """The evaluation index at which the clean search first *moves its iterate* -- accepts a
    trial. That is where injecting an unusable model bites: a trial the trust region or the
    line search would reject on its score alone never reaches the model at all."""
    u = runner.start()
    point = None
    for i in range(max_evals):
        score, grad = evaluate(u)
        nxt = runner.got(u, score, grad)
        if point is not None and not np.array_equal(runner.point, point):
            return i
        point = np.array(runner.point, dtype=float)
        assert nxt is not DONE, 'the clean search ended without ever accepting a trial'
        u = nxt
    raise AssertionError('runner did not terminate within %d evaluations' % max_evals)


def _drive_unusable_accepted_trial(runner, evaluate, spoil, spoil_at, max_evals=4000):
    """Drive a runner whose trial at index ``spoil_at`` (the one the clean search accepts)
    scores finitely but returns a non-finite model, with every other point evaluating
    normally. Returns the converged runner; asserts it did not terminate at that trial, and
    that no point it proposes is itself non-finite (a NaN model that leaks into the step
    machine produces NaN trial points, which the orchestrator dispatches to the cluster as
    NaN PSets)."""
    u = runner.start()
    for i in range(max_evals):
        score, grad = spoil(u) if i == spoil_at else evaluate(u)
        nxt = runner.got(u, score, grad)
        if nxt is DONE:
            assert i > spoil_at, 'the runner ended the start instead of backing off'
            return runner
        assert np.all(np.isfinite(nxt)), 'the runner proposed a non-finite point'
        u = nxt
    raise AssertionError('runner did not terminate within %d evaluations' % max_evals)


@pytest.mark.parametrize('leaf', ['lbfgs', 'trf', 'gntr'])
def test_runner_backs_off_a_mid_search_trial_whose_model_is_not_finite(leaf):
    """Mid-search the runner has somewhere to back off *to* -- the current iterate, whose
    model is fine -- so an unusable trial is rejected rather than ending the start: the trust
    region shrinks (``trf``/``gntr``) or the line search backtracks (``lbfgs``), and the
    search still reaches the same minimum. The trial is rejected even though its objective
    *improved* (it is the very trial the clean search accepts): stepping onto it would move
    the iterate to a point with no usable curvature and, for L-BFGS-B, fold a NaN pair into
    the limited-memory history, making every direction afterwards NaN."""
    evaluate, spoil = _evaluators(leaf)
    factory = _runner_factories()[leaf]
    spoil_at = _first_accepted_trial_index(factory(_U0), evaluate)
    runner = _drive_unusable_accepted_trial(factory(_U0), evaluate, spoil, spoil_at)
    assert np.allclose(runner.point, _USTAR, atol=1e-4)


@pytest.mark.parametrize('leaf', ['trf', 'gntr'])
def test_trust_region_start_terminates_when_lapack_cannot_factorize(leaf, monkeypatch):
    """Belt and braces for the same crash: ``gesdd`` can fail to converge on a *finite* but
    pathological matrix too, independently of any NaN, and it reports that the only way
    LAPACK can -- ``LinAlgError``. The augmented-Jacobian SVD is the one place in the step
    math that can raise it, so the runner converts it into the same clean per-start
    termination instead of letting it unwind into ``got_result`` (#528)."""
    def gesdd_fails(*args, **kwargs):
        raise np.linalg.LinAlgError('SVD did not converge')

    monkeypatch.setattr(np.linalg, 'svd', gesdd_fails)
    evaluate, _ = _evaluators(leaf)
    runner = _runner_factories()[leaf](_U0)
    u0 = runner.start()
    score, grad = evaluate(u0)
    assert runner.got(u0, score, grad) is DONE
    assert 'could not be factorized' in runner.stop_reason


def test_gntr_start_terminates_when_lapack_cannot_diagonalize_the_hessian(monkeypatch):
    """``gntr`` reaches LAPACK one step earlier than ``trf``: it eigen-decomposes the EFIM
    Hessian into its pseudo-Jacobian before any scaling happens, so ``eigh`` is a second
    place a per-start failure could have aborted the fit (#528)."""
    def eigh_fails(*args, **kwargs):
        raise np.linalg.LinAlgError('Eigenvalues did not converge')

    monkeypatch.setattr(np.linalg, 'eigh', eigh_fails)
    evaluate, _ = _evaluators('gntr')
    runner = _runner_factories()['gntr'](_U0)
    u0 = runner.start()
    score, grad = evaluate(u0)
    assert runner.got(u0, score, grad) is DONE
    assert 'could not be diagonalized' in runner.stop_reason


def test_gntr_terminates_the_start_on_a_non_finite_hessian_without_reaching_lapack():
    """A NaN in the EFIM Hessian itself (rather than in the gradient) is caught by the
    finiteness gate *before* the eigen-decomposition -- ``eigh`` on a non-finite matrix
    raises ``LinAlgError: Eigenvalues did not converge``, the same class of whole-fit abort
    as the reported SVD failure (#528)."""
    runner = _runner_factories()['gntr'](_U0)
    u0 = runner.start()
    hessian = _D.T @ _D
    hessian = hessian.copy()
    hessian[0, 2] = hessian[2, 0] = np.nan
    nxt = runner.got(u0, 7.0, _Grad(np.array([1.0, 1.0, 1.0]), hessian=hessian,
                                    least_squares_exact=False))
    assert nxt is DONE
    assert 'Fisher model' in runner.stop_reason and 'not finite' in runner.stop_reason


def test_gntr_start_still_fails_loudly_when_no_hessian_was_attached(monkeypatch):
    """The finiteness gate must not swallow the *internal wiring* error: a gntr runner driven
    without an EFIM Hessian is a bug in how the fit was assembled, not a bad point to skip
    over, so it still raises rather than quietly ending the start (#528 guarding #481)."""
    runner = _runner_factories()['gntr'](_U0)
    u0 = runner.start()
    with pytest.raises(Exception, match='requires an assembled EFIM Hessian'):
        runner.got(u0, 7.0, _Grad(np.array([1.0, 1.0, 1.0]), least_squares_exact=False))


class _NonFiniteGradientHarness(_AdvanceHarness):
    """The real ``_advance`` over a result that *did* simulate (``simdata`` present, a finite
    score) but whose assembled gradient is not finite -- the #528 arrival path."""

    def gradient_at(self, res):
        return _Grad(np.array([1.0, np.nan, 0.5]))


def test_advance_terminates_a_start_whose_assembled_gradient_is_not_finite():
    """End-to-end at the orchestration seam of the reported traceback
    (``got_result`` -> ``_advance`` -> ``got`` -> ... -> ``np.linalg.svd``): a scored point
    with a non-finite gradient now returns :data:`DONE` from ``_advance``, so the base ends
    that one start and the other concurrent starts keep running -- instead of an uncaught
    ``LinAlgError`` propagating out of the run loop (#528)."""
    harness = _NonFiniteGradientHarness()
    runner = _runner_factories()['lbfgs'](_U0)
    runner.start()
    res = SimpleNamespace(pset=_U0, simdata={'model': 'simdata'}, score=7.0)
    out = harness._advance(0, runner, res)
    assert out is DONE
    assert 'not finite' in runner.stop_reason
    assert harness.dispatched == []            # the start terminated; nothing dispatched
