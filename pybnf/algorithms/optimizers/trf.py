"""Trust-Region-Reflective least-squares optimizer (``trf`` fit type, #386 / #460).

The primary gradient-based local optimizer: the workhorse for Gaussian / sum-of-
squares objectives (the common case in PyBNF). It consumes the **residual vector +
residual-Jacobian** #385 assembles from bngsim's forward output sensitivities, and
approximates the objective Hessian as ``JᵀJ`` -- far better conditioned and faster-
converging than feeding a scalar gradient to a generic quasi-Newton method on a
least-squares problem. This is the D2D (Data2Dynamics) workhorse step; see
``examples/becker_d2d_gradient/`` for the standalone methodological reference.

Why native (not ``scipy.optimize.least_squares``): scipy is a *blocking* driver that
calls ``fun``/``jac`` synchronously, so it cannot farm its evaluations to PyBNF's
distributed propose/score loop (the same incompatibility ``powell.py`` documents).
The method is reimplemented as an explicit, *picklable* step machine -- here a headless
:class:`~pybnf.algorithms.optimizers.gradient_base.GradientRunner` (:class:`_TRFRunner`)
that :class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer` drives inside
the run-loop contract, no ``run()`` override (ADR-0007) -- so backup/resume work like
every other method and one evaluation is one scheduler job. Factoring the step machine
into a per-start runner is also what lets a fit run ``N`` of them concurrently (local
multi-start, the orchestration the base owns). scipy is the **test oracle** for the step
math (``tests/test_gradient_runner.py``), never called in the production loop -- the
Branch–Coleman–Li math below is ported in-process (the ``_trf_*`` helpers).

The method (Trust-Region-Reflective, Branch–Coleman–Li 1999)
------------------------------------------------------------
In sampling space ``u`` (``StartPointOptimizer`` / ``GradientOptimizer``), with the
residual ``r`` and residual-Jacobian ``J`` from #385 at the current point ``x``
(gradient ``g = Jᵀr``, Gauss–Newton Hessian ``A = JᵀJ``), the bounds are handled by the
Coleman–Li affine scaling, matching ``scipy.optimize.least_squares(method='trf')``:

* the **Coleman–Li scaling** ``v(x)``: ``vᵢ`` is the distance from ``xᵢ`` to the bound
  the anti-gradient points at (``ubᵢ-xᵢ`` when ``gᵢ<0``, ``xᵢ-lbᵢ`` when ``gᵢ>0``, else
  ``1``). With ``D = diag(√v)`` the first-order optimality condition is ``D²g = 0`` -- so
  the scaled gradient ``‖v·g‖∞`` reads as zero both in the interior (``g=0``) and on an
  active face (``v=0`` there). A bound-active optimum is therefore *recognized* as
  optimal, where a plain ``‖g‖∞`` never would be;
* the trust-region subproblem is solved in the scaled ("hat") space ``x = D x̂``, where
  the Newton model is ``½ x̂ᵀ(DAD + C)x̂ + (Dg)ᵀx̂`` with ``C = diag(g·dv)`` the curvature
  the scaling itself contributes (``dv ∈ {-1,0,1}``), via a single SVD of the augmented
  Jacobian ``[J D; C^{½}]`` and More's root-find on the trust radius ``Δ``;
* **reflective step selection**: a trust-region step that would leave the box is
  *reflected* off the first bound it crosses, and the best (by predicted model
  reduction) of {the constrained trust step, the reflected step, a scaled-gradient
  Cauchy step} is taken, kept strictly interior by a step-back factor ``θ`` (the
  Coleman–Li differentiability requirement). Reflection lets the iterate slide *along*
  an active face and converge cleanly onto a bound-active corner, where simply clipping
  the unconstrained step into the box can stall.

``Δ`` is the trust-region knob: it shrinks when a step is rejected or over-predicted
(``Δ ← ¼‖step‖`` when the gain ratio ``ρ < ¼``) and grows when a good step pushes against
the trust boundary (``Δ ← 2Δ`` when ``ρ > ¾`` and the step hit the boundary). The
initial ``Δ₀`` is derived from the (scaled) start point, not a tunable.

Each iteration costs **one** objective evaluation (the trial). The whole scaled-SVD
subproblem solve + reflective step selection is in-process linear algebra on the cached
``r``/``J`` -- no evaluations. On accept, the trial's own residual/Jacobian -- already
assembled, since master-side scoring returns the simdata -- become the next iterate's, so
an accepted step needs no re-evaluation; on reject, only ``Δ`` shrinks and the step is
re-solved from the same cached SVD. The run stops when the scaled gradient is flat
(``‖v·g‖∞ ≤ trf_grad_tol`` -- the first-order optimality test that respects active
bounds), the step is negligible (``‖δ‖ ≤ trf_step_tol·(‖x‖+trf_step_tol)``), or the
iteration budget is spent. In the box interior (no bound active at the solution) the
scaling and reflection fall away and the method is an ordinary trust-region least-squares
step, MINPACK-style.

Scope. TRF consumes the **exact least-squares residual** -- the Gaussian (any
scale/location) and the Student-t (#459). A fit whose objective is not an exact sum of
squares (an estimated noise scale, a Laplace / count family, active constraints;
``GradientResult.least_squares_exact == False``) has no faithful residual model, so this
optimizer refuses it with a pointer to the L-BFGS-B path (``fit_type = lbfgs``, #386's
fallback). Local multi-start is provided by :class:`GradientOptimizer` (the base runs
``N`` independent :class:`_TRFRunner` starts concurrently and keeps the global best).

All runner state is plain ``numpy`` / ``float`` (the point, residual, Jacobian, the
cached scaling + SVD, the trust radius) -- picklable, so ``Algorithm.backup`` checkpoints
the optimizer (and its list of runners) mid-run.
"""

import math
from typing import ClassVar

import numpy as np

from .gradient_base import DONE, GradientOptimizer, GradientRunner
from ...config_schema import PyBNFConfigModel
from ...printing import PybnfError
from ...registry import register_fit_type

_EPS = np.finfo(float).eps


class TRFConfig(PyBNFConfigModel):
    """TRF (Trust-Region-Reflective) config fields, co-located with the method (ADR-0006).

    ``trf_grad_tol`` ends the run when the largest component of the **scaled**
    least-squares gradient ``v·Jᵀr`` (in sampling space) falls below it -- a first-order
    optimality test that reads as the ordinary ``‖Jᵀr‖∞`` in the interior and as zero on
    a bound the gradient pushes against (the Coleman–Li scaling ``v`` vanishes there).
    ``trf_step_tol`` ends it when an accepted step ``δ`` becomes negligible relative to
    the point (``‖δ‖ ≤ trf_step_tol·(‖x‖+trf_step_tol)``). The initial trust radius is
    derived from the (scaled) start point -- there is no damping/radius tunable. Like
    Powell's cycle budget, ``trf_max_iterations`` is runtime-guarded -- it defaults to
    the global ``max_iterations`` when unset -- so it is a valid key but not a schema
    field. ``trf_start_point`` is internal (the refiner injects it), so it is not modeled
    here either."""

    trf_grad_tol: float = 1e-8
    trf_step_tol: float = 1e-8

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'trf_max_iterations'})


@register_fit_type('trf', family='optimizer', display_name='Trust-Region Least-Squares',
                   schema=TRFConfig, refiner=True, start_from_box=True)
class TRFAlgorithm(GradientOptimizer):
    """Bounded Trust-Region-Reflective least-squares: a method-agnostic multi-start
    orchestrator (:class:`GradientOptimizer`) over per-start :class:`_TRFRunner` step
    machines."""

    #: Message label + refiner start-point key (see StartPointOptimizer).
    fit_type = 'trf'
    START_POINT_KEY = 'trf_start_point'
    _method_label = 'TRF'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.grad_tol = config.config['trf_grad_tol']
        self.step_tol = config.config['trf_step_tol']
        if 'trf_max_iterations' in config.config:
            self.max_iterations = config.config['trf_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']

    def _start_banner(self):
        return ("Running trust-region-reflective least-squares for up to "
                "%i iterations from %i start point(s)" % (self.max_iterations, self.n_starts))

    def _make_runner(self, u0):
        """One Trust-Region-Reflective step machine seeded at ``u0`` (sampling space),
        carrying this fit's box + tunables. The orchestrator builds one per start."""
        return _TRFRunner(u0, self._u_lower, self._u_upper, self.max_iterations,
                          grad_tol=self.grad_tol, step_tol=self.step_tol)


# --------------------------------------------------------------------------- #
# Branch–Coleman–Li trust-region math, ported in-process (#460).
#
# Faithful ports of ``scipy.optimize._lsq.{common,trf}`` -- the test oracle -- kept
# here as pure-``numpy`` module functions so the production loop never imports scipy
# (ADR-0007). The runner below uses the ``x_scale = 1``, dense-``exact`` (SVD),
# linear-loss path of ``scipy``'s ``method='trf'``, which is exactly what the offline
# oracle (``tests/test_gradient_runner.py``) compares against.
# --------------------------------------------------------------------------- #
def _cl_scaling_vector(x, g, lower, upper):
    """Coleman–Li scaling vector ``v`` and its diagonal derivative ``dv``.

    ``vᵢ`` is the distance to the bound the anti-gradient points at (``ubᵢ-xᵢ`` when
    ``gᵢ<0`` and ``ubᵢ`` is finite, ``xᵢ-lbᵢ`` when ``gᵢ>0`` and ``lbᵢ`` is finite, else
    ``1``); ``dvᵢ ∈ {-1, 0, 1}`` is ``∂vᵢ/∂xᵢ``. ``v ≥ 0`` and ``g·dv ≥ 0`` everywhere."""
    v = np.ones_like(x)
    dv = np.zeros_like(x)
    mask = (g < 0) & np.isfinite(upper)
    v[mask] = upper[mask] - x[mask]
    dv[mask] = -1
    mask = (g > 0) & np.isfinite(lower)
    v[mask] = x[mask] - lower[mask]
    dv[mask] = 1
    return v, dv


def _in_bounds(x, lower, upper):
    return bool(np.all((x >= lower) & (x <= upper)))


def _step_size_to_bound(x, s, lower, upper):
    """Smallest ``t ≥ 0`` with ``x + t·s`` on a bound, and a ``{-1,0,1}`` hit indicator
    per coordinate (which bound, if any, is reached at that ``t``)."""
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over='ignore'):
        steps[non_zero] = np.maximum((lower - x)[non_zero] / s_non_zero,
                                     (upper - x)[non_zero] / s_non_zero)
    min_step = np.min(steps)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def _find_active_constraints(x, lower, upper, rtol=1e-10):
    """Per-coordinate ``{-1,0,1}`` active-bound indicator (``rtol=0`` uses exact equality)."""
    active = np.zeros_like(x, dtype=int)
    if rtol == 0:
        active[x <= lower] = -1
        active[x >= upper] = 1
        return active
    lower_dist = x - lower
    upper_dist = upper - x
    lower_threshold = rtol * np.maximum(1, np.abs(lower))
    upper_threshold = rtol * np.maximum(1, np.abs(upper))
    lower_active = (np.isfinite(lower)
                    & (lower_dist <= np.minimum(upper_dist, lower_threshold)))
    active[lower_active] = -1
    upper_active = (np.isfinite(upper)
                    & (upper_dist <= np.minimum(lower_dist, upper_threshold)))
    active[upper_active] = 1
    return active


def _make_strictly_feasible(x, lower, upper, rstep=1e-10):
    """Nudge ``x`` off any bound it sits on so the iterate stays strictly interior (the
    Coleman–Li differentiability requirement); ``rstep=0`` uses :func:`numpy.nextafter`."""
    x_new = x.copy()
    active = _find_active_constraints(x, lower, upper, rstep)
    lower_mask = np.equal(active, -1)
    upper_mask = np.equal(active, 1)
    if rstep == 0:
        x_new[lower_mask] = np.nextafter(lower[lower_mask], upper[lower_mask])
        x_new[upper_mask] = np.nextafter(upper[upper_mask], lower[upper_mask])
    else:
        x_new[lower_mask] = (lower[lower_mask]
                             + rstep * np.maximum(1, np.abs(lower[lower_mask])))
        x_new[upper_mask] = (upper[upper_mask]
                             - rstep * np.maximum(1, np.abs(upper[upper_mask])))
    tight = (x_new < lower) | (x_new > upper)
    x_new[tight] = 0.5 * (lower[tight] + upper[tight])
    return x_new


def _intersect_trust_region(x, s, Delta):
    """Roots ``t`` of ``‖x + t·s‖ = Δ`` (negative, positive), avoiding cancellation."""
    a = np.dot(s, s)
    if a == 0:
        raise ValueError('`s` is zero.')
    b = np.dot(x, s)
    c = np.dot(x, x) - Delta ** 2
    if c > 0:
        raise ValueError('`x` is not within the trust region.')
    d = np.sqrt(b * b - a * c)
    q = -(b + math.copysign(d, b))
    t1 = q / a
    t2 = c / q
    return (t1, t2) if t1 < t2 else (t2, t1)


def _solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
                            rtol=0.01, max_iter=10):
    """Solve the least-squares trust-region subproblem from the SVD of the (augmented)
    Jacobian (More's root-find on the Levenberg–Marquardt parameter ``alpha``). Returns
    ``(p, alpha, n_iter)``; ``n_iter == 0`` means the Gauss–Newton step was taken."""
    def phi_and_derivative(alpha, suf, s, Delta):
        denom = s ** 2 + alpha
        p_norm = np.linalg.norm(suf / denom)
        phi = p_norm - Delta
        phi_prime = -np.sum(suf ** 2 / denom ** 3) / p_norm
        return phi, phi_prime

    suf = s * uf
    if m >= n:
        threshold = _EPS * m * s[0]
        full_rank = s[-1] > threshold
    else:
        full_rank = False

    if full_rank:
        p = -V.dot(uf / s)
        if np.linalg.norm(p) <= Delta:
            return p, 0.0, 0

    alpha_upper = np.linalg.norm(suf) / Delta
    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    else:
        alpha = initial_alpha

    it = 0
    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)
        if phi < 0:
            alpha_upper = alpha
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta
        if np.abs(phi) < rtol * Delta:
            break

    p = -V.dot(suf / (s ** 2 + alpha))
    p *= Delta / np.linalg.norm(p)   # make ‖p‖ == Δ exactly (it barely moves)
    return p, alpha, it + 1


def _build_quadratic_1d(J, g, s, diag=None, s0=None):
    """Coefficients ``(a, b[, c])`` of the model ``½(s0+s·t)ᵀ(JᵀJ+diag)(s0+s·t) +
    gᵀ(s0+s·t)`` as a univariate quadratic in ``t`` (``c`` returned iff ``s0`` given)."""
    v = J.dot(s)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5
    b = np.dot(g, s)
    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v)
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c
    return a, b


def _minimize_quadratic_1d(a, b, lower, upper, c=0):
    """Minimize ``a·t² + b·t + c`` over ``[lower, upper]`` (both finite); returns
    ``(argmin, min)``."""
    t = [lower, upper]
    if a != 0:
        extremum = -0.5 * b / a
        if lower < extremum < upper:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    i = np.argmin(y)
    return t[i], y[i]


def _evaluate_quadratic(J, g, s, diag=None):
    """The model value ``½ sᵀ(JᵀJ + diag)s + gᵀs`` at step ``s``."""
    if s.ndim == 1:
        Js = J.dot(s)
        q = np.dot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = np.sum(Js ** 2, axis=0)
        if diag is not None:
            q += np.sum(diag * s ** 2, axis=1)
    return 0.5 * q + np.dot(s, g)


def _update_tr_radius(Delta, actual_reduction, predicted_reduction, step_norm, bound_hit):
    """New trust radius + gain ratio: shrink to ``¼‖step‖`` on a poor step (``ρ<¼``),
    double when a good step (``ρ>¾``) pushes against the trust boundary."""
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0
    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0
    return Delta, ratio


def _select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lower, upper, theta):
    """Pick the Trust-Region-Reflective step: the best (by predicted model reduction) of
    the constrained trust step, the reflected step, and the scaled-gradient Cauchy step,
    each kept strictly interior. Returns ``(step, step_h, predicted_reduction)``."""
    if _in_bounds(x + p, lower, upper):
        p_value = _evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value

    p_stride, hits = _step_size_to_bound(x, p, lower, upper)

    # Reflect the trust step off the first bound it crosses.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict the trust step so it lands on that bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # The reflected ray exits the feasible region or the trust region first.
    _, to_tr = _intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = _step_size_to_bound(x_on_bound, r, lower, upper)

    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    if r_stride_l <= r_stride_u:
        a, b, c = _build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = _minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    # Step the constrained trust step back to strictly interior.
    p *= theta
    p_h *= theta
    p_value = _evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

    # Scaled-gradient (Cauchy) step.
    ag_h = -g_h
    ag = d * ag_h
    to_tr = Delta / np.linalg.norm(ag_h)
    to_bound, _ = _step_size_to_bound(x, ag, lower, upper)
    ag_stride = theta * to_bound if to_bound < to_tr else to_tr
    a, b = _build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = _minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


class _TRFRunner(GradientRunner):
    """One Trust-Region-Reflective start: the picklable step machine, in sampling space
    ``u``.

    Holds the iterate, its residual model (``r``, ``J``, ``g = Jᵀr``, ``cost = ½‖r‖²``),
    the trust radius ``Δ``, and -- cached for the current outer iteration -- the
    Coleman–Li scaling and the augmented-Jacobian SVD (so a rejected step re-solves the
    trust-region subproblem without re-evaluating). Consumes ``(u_point, score, grad)``
    and returns the next ``u`` to evaluate (or :data:`DONE`). Pure ``numpy`` -- no PSets,
    objective, or backend (see :class:`GradientRunner`). The step math is the Branch–
    Coleman–Li bound handling described in the module docstring (the ``_trf_*`` helpers);
    the orchestrator (:class:`TRFAlgorithm` / :class:`GradientOptimizer`) supplies the
    assembled :class:`GradientResult` and does the reporting, and this runner requires it
    to be an **exact** least-squares residual (:meth:`_require_exact`)."""

    def __init__(self, u0, lower, upper, max_iterations, *, grad_tol, step_tol):
        super().__init__(u0, lower, upper, max_iterations)
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        # Current-iterate residual model.
        self.r = None              # residual vector f at point (m,)
        self.J = None              # residual Jacobian at point (m, n)
        self.g = None              # Jᵀr at point (n,)
        self.m = 0                 # residual count
        self.cost = None           # ½‖r‖² (== fval)
        self.Delta = None          # trust-region radius
        self.alpha = 0.0           # LM parameter, carried across solves (scipy's `alpha`)
        # Scaling + augmented-Jacobian SVD for the current outer iteration (rebuilt per
        # accepted point; reused while rejecting). All plain ndarray/float -> picklable.
        self._d = None             # √v (the Coleman–Li scale, scale = 1)
        self._diag_h = None        # g·dv (the C-matrix diagonal in hat space)
        self._g_h = None           # d·g (hat-space gradient)
        self._J_h = None           # J·d (hat-space Jacobian, m×n)
        self._s = None             # singular values of the augmented Jacobian
        self._V = None             # right singular vectors (n×n)
        self._uf = None            # Uᵀ·[r; 0]
        self._theta = None         # strict-feasibility step-back factor
        # Trial currently out for evaluation.
        self.predicted_reduction = None
        self.step_h_norm = None    # ‖step‖ in hat space (drives the Δ update)
        self.step_norm = None      # ‖step‖ in original space (drives the step-tol test)

    def progress_detail(self):
        return 'trust radius %g' % self.Delta

    def got(self, u_point, score, grad):
        if self.phase == 'init':
            return self._after_init(u_point, score, grad)
        if self.phase == 'step':
            return self._after_step(u_point, score, grad)
        raise RuntimeError(f'Internal error in _TRFRunner: phase {self.phase!r}')

    # --- state machine ----------------------------------------------------- #
    def _after_init(self, u_point, score, grad):
        """Seed the state from the start-point evaluation: residual/Jacobian model and
        the initial trust radius ``Δ₀`` (derived from the scaled start point)."""
        if grad is None:
            # The start point did not simulate (a non-integrable point): no residual model
            # to seed the trust region from, so this start is not viable (#492). Shared by
            # the inheriting ``gntr`` runner.
            return self._failed_start()
        gr = self._require_exact(grad)
        self.point = np.array(u_point, dtype=float)
        self.fval = score
        self.cost = score
        self._set_model(gr)
        if not self.n:
            self.stop_reason = 'no free parameters to optimize'
            return DONE
        v, _ = _cl_scaling_vector(self.point, self.g, self._u_lower, self._u_upper)
        self.Delta = float(np.linalg.norm(self.point / np.sqrt(v)))
        if self.Delta == 0.0:
            self.Delta = 1.0
        if self._gradient_converged():
            self.stop_reason = 'gradient already flat at the start point'
            return DONE
        return self._begin_outer()

    def _after_step(self, u_point, score, grad):
        """Score the trial by its gain ratio, update the trust radius, and accept (step
        + rebuild the scaling/SVD at the new point) or reject (re-solve from the cached
        SVD at a smaller radius)."""
        self.iteration += 1
        f_new = score
        step_h_norm = self.step_h_norm

        if not np.isfinite(f_new):
            # A non-finite trial gives the trust region its signal to shrink and re-solve
            # (no accept), exactly like scipy's `Delta = 0.25*step_h_norm; continue`.
            self.Delta = 0.25 * step_h_norm
            return self._reject_or_budget()

        cost_new = f_new
        actual_reduction = self.cost - cost_new
        Delta_new, _ = _update_tr_radius(
            self.Delta, actual_reduction, self.predicted_reduction,
            step_h_norm, step_h_norm > 0.95 * self.Delta)
        step_negligible = self._step_negligible(self.step_norm)

        if actual_reduction > 0.0:
            # Accept: the trial's own residual/Jacobian (assembled by the orchestrator)
            # become the next iterate's, so no re-evaluation is needed.
            gr = self._require_exact(grad)
            self.point = np.array(u_point, dtype=float)
            self.cost = cost_new
            self.fval = cost_new
            self._set_model(gr)
            self.alpha *= self.Delta / Delta_new
            self.Delta = Delta_new
            stop = self._stop_reason(step_negligible)
            if stop is not None:
                self.stop_reason = stop
                return DONE
            return self._begin_outer()

        # Reject: shrink the trust region (a shorter, more gradient-like step) and
        # re-solve from the same point + cached SVD. A step that became negligible
        # without reducing the cost means the run has stalled at a minimum.
        self.alpha *= self.Delta / Delta_new
        self.Delta = Delta_new
        if step_negligible:
            self.stop_reason = self._step_negligible_reason()
            return DONE
        return self._reject_or_budget()

    def _reject_or_budget(self):
        """Re-solve the trust-region subproblem (cached SVD, current ``Δ``), or stop on a
        spent iteration budget so a stalled run cannot loop forever."""
        if self.iteration >= self.max_iterations:
            self.stop_reason = 'reached max_iterations (%i)' % self.max_iterations
            return DONE
        return self._solve_and_propose()

    # --- trust-region step ------------------------------------------------- #
    def _begin_outer(self):
        """Start a new outer iteration at the current point: rebuild the Coleman–Li
        scaling + augmented-Jacobian SVD, then solve the first trust-region step."""
        self._build_scaling()
        return self._solve_and_propose()

    def _build_scaling(self):
        """Cache the Coleman–Li scaling (``d = √v``, ``diag_h = g·dv``, ``g_h = d·g``,
        ``J_h = J·d``) and the SVD of the augmented Jacobian ``[J_h; diag(√diag_h)]`` for
        the current point (``x_scale = 1``, dense `exact` path of scipy's TRF)."""
        x, g, r, J = self.point, self.g, self.r, self.J
        m, n = self.m, self.n
        v, dv = _cl_scaling_vector(x, g, self._u_lower, self._u_upper)
        g_scaled_norm = float(np.max(np.abs(g * v)))
        d = np.sqrt(v)
        diag_h = g * dv
        J_h = J * d                       # column scaling (d broadcasts over columns)
        j_aug = np.zeros((m + n, n))
        j_aug[:m] = J_h
        j_aug[m:] = np.diag(np.sqrt(diag_h))
        u, s, vt = np.linalg.svd(j_aug, full_matrices=False)
        f_aug = np.zeros(m + n)
        f_aug[:m] = r
        self._d = d
        self._diag_h = diag_h
        self._g_h = d * g
        self._J_h = J_h
        self._s = s
        self._V = vt.T
        self._uf = u.T @ f_aug
        # theta controls the strict-feasibility step-back from the bounds.
        self._theta = max(0.995, 1.0 - g_scaled_norm)

    def _solve_and_propose(self):
        """Solve the (scaled) trust-region subproblem at the current ``Δ`` from the cached
        SVD, pick the reflective step, and return the strictly-feasible trial point."""
        x = self.point
        lower, upper = self._u_lower, self._u_upper
        p_h, self.alpha, _ = _solve_lsq_trust_region(
            self.n, self.m, self._uf, self._s, self._V, self.Delta,
            initial_alpha=self.alpha)
        p = self._d * p_h
        step, step_h, predicted_reduction = _select_step(
            x, self._J_h, self._diag_h, self._g_h, p, p_h, self._d, self.Delta,
            lower, upper, self._theta)
        trial = _make_strictly_feasible(x + step, lower, upper, rstep=0.0)
        self.predicted_reduction = predicted_reduction
        self.step_h_norm = float(np.linalg.norm(step_h))
        self.step_norm = float(np.linalg.norm(step))
        self.phase = 'step'
        return trial

    def _set_model(self, grad):
        """Cache the residual model ``r``, ``J``, ``g = Jᵀr`` (and the residual count
        ``m``) at the current point from an assembled :class:`GradientResult`."""
        self.J = grad.jacobian
        self.r = grad.residual
        self.g = self.J.T @ self.r
        self.m = self.J.shape[0]

    def _require_exact(self, grad):
        """Require an **exact** least-squares residual from the assembled gradient. TRF
        models the objective as ``½‖r‖²``; an objective that is not an exact sum of
        squares (estimated scale, Laplace/count family, constraints) has no faithful
        residual, so refuse it with a pointer to the L-BFGS-B fallback rather than
        silently optimizing the wrong surface."""
        if not grad.least_squares_exact:
            raise PybnfError(
                "fit_type = trf needs an exact least-squares residual (a Gaussian or "
                "Student-t objective with a fixed noise scale and no constraints), but "
                "this fit's objective is not an exact sum of squares.",
                "Use the gradient quasi-Newton fallback 'fit_type = lbfgs', which "
                "consumes the scalar gradient and handles estimated noise scales, the "
                "Laplace / count families, and constraint penalties.")
        return grad

    # --- convergence ------------------------------------------------------- #
    def _scaled_gradient_norm(self):
        """The first-order optimality measure ``‖v·g‖∞``: the ordinary ``‖g‖∞`` in the
        box interior, and zero on a bound the gradient pushes against (the Coleman–Li
        scaling ``v`` vanishes there)."""
        v, _ = _cl_scaling_vector(self.point, self.g, self._u_lower, self._u_upper)
        return float(np.max(np.abs(self.g * v)))

    def _gradient_converged(self):
        return bool(self.n) and self._scaled_gradient_norm() <= self.grad_tol

    def _step_negligible(self, step_norm):
        """Whether the step is negligible relative to the (pre-step) point."""
        point_norm = float(np.linalg.norm(self.point))
        return step_norm < self.step_tol * (self.step_tol + point_norm)

    def _step_negligible_reason(self):
        point_norm = float(np.linalg.norm(self.point))
        return 'step is negligible (‖δ‖ ≤ %g)' % (self.step_tol * (point_norm + self.step_tol))

    def _stop_reason(self, step_negligible):
        """A termination string after an accepted step, or None to keep going."""
        if self._gradient_converged():
            return 'gradient is flat (‖v·Jᵀr‖∞ ≤ %g)' % self.grad_tol
        if step_negligible:
            return self._step_negligible_reason()
        if self.iteration >= self.max_iterations:
            return 'reached max_iterations (%i)' % self.max_iterations
        return None
