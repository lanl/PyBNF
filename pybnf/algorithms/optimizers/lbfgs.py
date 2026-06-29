"""Projected-gradient limited-memory BFGS optimizer (``lbfgs`` fit type, #386).

The scalar-gradient sibling of the trust-region least-squares method (``trf.py``):
where TRF consumes the residual vector + residual-Jacobian and is the workhorse for
exact-least-squares (Gaussian / Student-t, fixed scale) objectives, this method consumes
only the **scalar** objective value (``res.score``) and the **scalar** gradient
(``gradient_at(res).gradient``) that #385 assembles from bngsim's forward output
sensitivities. That makes it the fallback for precisely the objectives TRF *refuses* --
the ones whose ``GradientResult.least_squares_exact`` is ``False`` and so have no
faithful sum-of-squares residual: an **estimated** noise scale (the retained ``+log σ``
normalizer is not a square), the **Laplace / count** families, and a fit with active
**constraint penalties**. The assembly returns a finite scalar gradient for all of
these, so the same gradient seam drives both methods; only the step math differs.

Why native (not ``scipy.optimize.minimize(method='L-BFGS-B')``): scipy is a *blocking*
driver that calls ``fun``/``jac`` synchronously, so it cannot farm its evaluations to
PyBNF's distributed propose/score loop (the same incompatibility ``powell.py`` and
``trf.py`` document). The method is reimplemented as an explicit, *picklable* state
machine inside the run-loop contract -- ``start_run`` / ``got_result`` only, no
``run()`` override (ADR-0007) -- so backup/resume work like every other method and one
evaluation is one scheduler job.

The method (projected-gradient L-BFGS + backtracking line search)
-----------------------------------------------------------------
In sampling space ``u`` (``StartPointOptimizer`` / ``GradientOptimizer``), with the
scalar gradient ``g = ∇F`` from #385 at the current point ``x``:

* keep the ``m`` most recent curvature pairs ``(s, y)`` (``s = xₖ₊₁ - xₖ``,
  ``y = gₖ₊₁ - gₖ``) and form the quasi-Newton search direction
  ``d = -Hₖ g`` by the **two-loop recursion** (the implicit limited-memory inverse
  Hessian, scaled by ``γ = (sᵀy)/(yᵀy)`` on its diagonal seed). With no history this is
  steepest descent, ``d = -g``;
* **backtracking (Armijo) line search**: start at step length ``α = 1`` and project each
  trial ``P[x + α d]`` onto the box (:meth:`_u_bounds`); accept the first ``α`` whose
  *projected* sufficient-decrease holds,
  ``F(P[x+αd]) ≤ F(x) + c₁ · gᵀ(P[x+αd] - x)`` (Bertsekas' projected Armijo, testing the
  **actual** displacement so a clamped coordinate is handled correctly; it reduces to the
  usual ``α gᵀd`` when no bound is active), else shrink ``α ← backtrack·α`` and retry;
* on accept, fold the realized ``(s, y)`` into the history (skipping a non-positive-
  curvature pair, which would break the BFGS positive-definite invariant) and step.

Each line-search trial costs **one** objective evaluation; an accepted step's gradient
is assembled from that trial's own simdata (master-side scoring returns it), so the step
needs no re-evaluation. The run stops when the **projected** gradient is flat
(``‖P[x-g]-x‖∞ ≤ lbfgs_grad_tol`` -- the first-order optimality test that respects active
bounds), the accepted step is negligible (``‖s‖ ≤ lbfgs_step_tol·(‖x‖+lbfgs_step_tol)``),
the line search cannot reduce the objective even along steepest descent (a stalled
minimum), or the iteration budget is spent.

Scope (this cut). This is **projected-gradient L-BFGS** -- the limited-memory direction
with a bound-projected backtracking line search -- not full L-BFGS-B, whose generalized
Cauchy point + subspace minimization identifies the active set more aggressively. The
projected line search keeps every iterate feasible and converges on the bounded problems
PyBNF poses; the generalized-Cauchy-point refinement is a tracked #386 follow-up.

All state is plain ``numpy`` / ``float`` / ``list`` (the point, gradient, the ``(s, y)``
history, the line-search scratch) -- picklable, so ``Algorithm.backup`` checkpoints the
optimizer mid-run, exactly like Powell / CMA-ES / TRF (ADR-0007).
"""

import logging
from typing import ClassVar

import numpy as np

from .gradient_base import GradientOptimizer
from ...config_schema import PyBNFConfigModel
from ...printing import print1, print2
from ...registry import register_fit_type

logger = logging.getLogger('pybnf.algorithms')


class LBFGSConfig(PyBNFConfigModel):
    """Projected-gradient L-BFGS config fields, co-located with the method (ADR-0006).

    ``lbfgs_grad_tol`` ends the run when the largest component of the **projected**
    gradient ``P[x-g]-x`` (in sampling space) falls below it -- a first-order optimality
    test that reads as the ordinary ``‖g‖∞`` in the interior and as zero on a bound the
    gradient pushes against. ``lbfgs_step_tol`` ends it when an accepted step ``s``
    becomes negligible relative to the point (``‖s‖ ≤ lbfgs_step_tol·(‖x‖+lbfgs_step_tol)``).
    ``lbfgs_history`` is ``m``, the number of recent curvature pairs the limited-memory
    inverse Hessian retains (more pairs ⇒ a richer Hessian model at higher per-step cost).
    ``lbfgs_c1`` is the Armijo sufficient-decrease constant (``0 < c₁ < 1``, conventionally
    ``1e-4``) and ``lbfgs_backtrack`` the step-length reduction factor on a rejected trial
    (``0 < β < 1``). Like Powell's / TRF's cycle budget, ``lbfgs_max_iterations`` is
    runtime-guarded -- it defaults to the global ``max_iterations`` when unset -- so it is
    a valid key but not a schema field. ``lbfgs_start_point`` is internal (the refiner
    injects it), so it is not modeled here either."""

    lbfgs_grad_tol: float = 1e-6
    lbfgs_step_tol: float = 1e-8
    lbfgs_history: int = 10
    lbfgs_c1: float = 1e-4
    lbfgs_backtrack: float = 0.5

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'lbfgs_max_iterations'})


@register_fit_type('lbfgs', family='optimizer', display_name='Projected-Gradient L-BFGS',
                   schema=LBFGSConfig, refiner=True, start_from_box=True)
class LBFGSAlgorithm(GradientOptimizer):
    """Projected-gradient limited-memory BFGS as a picklable reactor state machine."""

    #: Message label + refiner start-point key (see StartPointOptimizer).
    fit_type = 'lbfgs'
    START_POINT_KEY = 'lbfgs_start_point'

    #: Safety cap on objective evaluations per backtracking line search. With a
    #: descent direction the Armijo step is found in a handful of backtracks, so this
    #: is essentially never reached; on exhaustion the line search falls back to
    #: steepest descent and, failing that, the run stops at the (stalled) minimum.
    _MAX_LINE_EVALS = 30

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.n = len(self.variables)
        self.grad_tol = config.config['lbfgs_grad_tol']
        self.step_tol = config.config['lbfgs_step_tol']
        self.history = config.config['lbfgs_history']
        self.c1 = config.config['lbfgs_c1']
        self.backtrack = config.config['lbfgs_backtrack']
        if 'lbfgs_max_iterations' in config.config:
            self.max_iterations = config.config['lbfgs_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']
        self._u_lower, self._u_upper = self._u_bounds()
        self.start_pset = self._resolve_start_pset()
        self._init_state()

    def _init_state(self):
        """(Re)initialize the mutable L-BFGS state. Filled in by start_run / got_result;
        all plain float / ndarray / list so the optimizer pickles for backup/resume."""
        self.point = None          # current accepted iterate (u-space)
        self.fval = None           # objective F(point) (== res.score)
        self.grad = None           # scalar gradient ∇F(point) (u-space), from #385
        self.s_list = []           # recent s = xₖ₊₁ - xₖ (limited-memory history)
        self.y_list = []           # recent y = gₖ₊₁ - gₖ
        self.iteration = 0
        self.phase = None          # 'init' | 'line'
        # Backtracking line-search scratch (one 1-D search at a time).
        self.ls_base = None        # the iterate the line search departs from
        self.ls_fbase = None       # F(ls_base)
        self.ls_grad = None        # ∇F(ls_base)
        self.ls_dir = None         # the (descent) direction d
        self.ls_alpha = None       # current trial step length α
        self.ls_evals = 0          # backtracks taken in this line search
        self.ls_steepest = False   # is this line search the steepest-descent fallback?
        self.trial_u = None        # the u-vector currently out for evaluation
        self.probe_counter = 0

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._u_lower, self._u_upper = self._u_bounds()
        self.start_pset = self._resolve_start_pset()
        self._init_state()

    def add_iterations(self, n):
        self.max_iterations += n

    # --- batch plumbing ---------------------------------------------------- #
    def _submit(self, u, phase):
        """Queue the single objective evaluation at ``u`` and advance to ``phase``
        once it returns. L-BFGS is serial -- one evaluation (the line trial) per
        scheduler batch."""
        self.phase = phase
        self.trial_u = np.array(u, dtype=float)
        self.probe_counter += 1
        name = 'lbfgs_%i' % self.probe_counter
        return [self._pset_from_u(u, name=name)]

    def got_result(self, res):
        if self.phase == 'init':
            return self._after_init(res)
        if self.phase == 'line':
            return self._after_line(res)
        raise RuntimeError(f'Internal error in LBFGSAlgorithm: phase {self.phase!r}')

    # --- state machine ----------------------------------------------------- #
    def start_run(self):
        print2("Running projected-gradient L-BFGS for up to %i iterations"
               % self.max_iterations)
        # Activate the gradient path (enable sensitivities + build routings) before
        # the model scatter; start from the resolved start point / box center.
        self._setup_gradient_path()
        self.point = self._u_from_pset(self.start_pset)
        return self._submit(self.point, 'init')

    def _after_init(self, res):
        """Seed the state from the start-point evaluation: objective + scalar gradient,
        empty curvature history (so the first step is steepest descent)."""
        self.point = self._u_from_pset(res.pset)
        self.fval = float(res.score)
        self.grad = self.gradient_at(res).gradient
        if self._gradient_converged():
            logger.info('L-BFGS converged at the start point (projected gradient already flat)')
            return 'STOP'
        return self._begin_line_search()

    def _begin_line_search(self, steepest=False):
        """Form the search direction (two-loop recursion, or steepest descent) and
        start a backtracking line search along it from the current point.

        Guards descent: a quasi-Newton direction with ``gᵀd ≥ 0`` (a stale curvature
        model) is not a descent direction, so drop the history and retry along steepest
        descent, which always descends unless the gradient is itself flat (⇒ stop)."""
        d = -self.grad if (steepest or not self.s_list) else self._direction()
        if self.n and float(self.grad @ d) >= 0.0:
            if not steepest:
                self.s_list, self.y_list = [], []
                return self._begin_line_search(steepest=True)
            logger.info('L-BFGS stopping: no descent direction (gradient flat)')
            return 'STOP'
        self.ls_base = self.point.copy()
        self.ls_fbase = self.fval
        self.ls_grad = self.grad.copy()
        self.ls_dir = d
        self.ls_alpha = 1.0
        self.ls_evals = 0
        self.ls_steepest = steepest or not self.s_list
        return self._submit_trial()

    def _submit_trial(self):
        """Project the current trial step onto the box and submit its evaluation."""
        trial = np.clip(self.ls_base + self.ls_alpha * self.ls_dir,
                        self._u_lower, self._u_upper)
        return self._submit(trial, 'line')

    def _after_line(self, res):
        """Apply the projected Armijo test to the trial; accept it or backtrack."""
        f_new = float(res.score)
        trial_point = self._u_from_pset(res.pset)
        disp = trial_point - self.ls_base
        # Projected Armijo sufficient-decrease (Bertsekas): test the *actual*
        # displacement gᵀ(P[x+αd]-x), so a coordinate clamped to a bound is handled
        # correctly; reduces to α·gᵀd when no bound is active. The strict f_new <
        # f_base guard rejects the degenerate case where projection flips the
        # displacement out of descent.
        armijo_rhs = self.ls_fbase + self.c1 * float(self.ls_grad @ disp)
        if np.isfinite(f_new) and f_new <= armijo_rhs and f_new < self.ls_fbase:
            return self._accept(res, trial_point, f_new, disp)
        self.ls_evals += 1
        self.ls_alpha *= self.backtrack
        if (self.ls_evals >= self._MAX_LINE_EVALS or self.ls_alpha <= 1e-20
                or not np.any(disp != 0.0)):
            return self._line_search_failed()
        return self._submit_trial()

    def _line_search_failed(self):
        """No feasible Armijo step along this direction. Retry the iterate along
        steepest descent (dropping a possibly-stale curvature model) before concluding
        the run has stalled at a minimum."""
        if not self.ls_steepest:
            self.s_list, self.y_list = [], []
            return self._begin_line_search(steepest=True)
        logger.info('L-BFGS stopping: line search could not reduce the objective '
                    '(converged or stalled at a minimum)')
        return 'STOP'

    def _accept(self, res, trial_point, f_new, disp):
        """Accept the trial: fold the realized curvature pair into the history, step to
        it, and either propose the next line search or stop. The accepted trial's own
        gradient (assembled from its simdata) becomes the next iterate's, so no
        re-evaluation is needed."""
        new_grad = self.gradient_at(res).gradient
        self._store_pair(disp, new_grad - self.ls_grad)
        step_norm = float(np.linalg.norm(disp))
        self.point = trial_point
        self.fval = f_new
        self.grad = new_grad
        self.iteration += 1
        self._report()
        stop = self._stop_reason(step_norm)
        if stop is not None:
            logger.info('L-BFGS stopping: %s', stop)
            return 'STOP'
        return self._begin_line_search()

    # --- limited-memory linear algebra ------------------------------------- #
    def _store_pair(self, s, y):
        """Append the curvature pair ``(s, y)``, capped at ``lbfgs_history``. Skips a
        non-positive-curvature pair (``sᵀy`` not safely positive) -- storing it would
        break the BFGS positive-definite invariant and the ``ρ = 1/(sᵀy)`` the two-loop
        recursion divides by; the limited-memory model simply keeps its existing pairs."""
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        if float(s @ y) <= 1e-10 * float(y @ y):
            return
        self.s_list.append(s)
        self.y_list.append(y)
        if len(self.s_list) > self.history:
            self.s_list.pop(0)
            self.y_list.pop(0)

    def _direction(self):
        """The L-BFGS search direction ``d = -Hₖ g`` by the two-loop recursion (Nocedal &
        Wright Alg. 7.4): the implicit limited-memory inverse-Hessian product, with the
        diagonal seed scaled by ``γ = (sᵀy)/(yᵀy)`` of the most recent pair."""
        q = self.grad.copy()
        m = len(self.s_list)
        rho = [1.0 / float(self.y_list[i] @ self.s_list[i]) for i in range(m)]
        alpha = [0.0] * m
        for i in range(m - 1, -1, -1):
            alpha[i] = rho[i] * float(self.s_list[i] @ q)
            q = q - alpha[i] * self.y_list[i]
        s_last, y_last = self.s_list[-1], self.y_list[-1]
        gamma = float(s_last @ y_last) / float(y_last @ y_last)
        z = gamma * q
        for i in range(m):
            beta = rho[i] * float(self.y_list[i] @ z)
            z = z + (alpha[i] - beta) * self.s_list[i]
        return -z

    # --- convergence / reporting ------------------------------------------- #
    def _projected_gradient_norm(self):
        """The first-order optimality measure ``‖P[x-g]-x‖∞``: the ordinary ``‖g‖∞`` in
        the box interior, and zero along a bound the gradient pushes against (a
        constrained stationary point)."""
        pg = self.point - np.clip(self.point - self.grad, self._u_lower, self._u_upper)
        return float(np.max(np.abs(pg)))

    def _gradient_converged(self):
        return bool(self.n) and self._projected_gradient_norm() <= self.grad_tol

    def _stop_reason(self, step_norm):
        """A termination string after an accepted step, or None to keep going."""
        if self._gradient_converged():
            return 'projected gradient is flat (‖P[x-g]-x‖∞ ≤ %g)' % self.grad_tol
        point_norm = float(np.linalg.norm(self.point))
        if step_norm <= self.step_tol * (point_norm + self.step_tol):
            return 'step is negligible (‖s‖ ≤ %g)' % (self.step_tol * (point_norm + self.step_tol))
        if self.iteration >= self.max_iterations:
            return 'reached max_iterations (%i)' % self.max_iterations
        return None

    def _report(self):
        if self.iteration % self.config.config['output_every'] == 0:
            self.output_results()
        msg = 'Completed %i of %i L-BFGS iterations' % (self.iteration, self.max_iterations)
        (print1 if self.iteration % 10 == 0 else print2)(msg)
        print2('Current best objective: %f, %i curvature pair(s)' % (self.fval, len(self.s_list)))
