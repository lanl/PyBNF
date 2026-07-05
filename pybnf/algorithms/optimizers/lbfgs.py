"""Bounded limited-memory BFGS optimizer -- L-BFGS-B (``lbfgs`` fit type, #386).

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
``trf.py`` document). The method is reimplemented as an explicit, *picklable* step
machine -- here a headless :class:`~pybnf.algorithms.optimizers.gradient_base.GradientRunner`
(:class:`_LBFGSRunner`) that :class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer`
drives inside the run-loop contract, no ``run()`` override (ADR-0007) -- so backup/resume
work like every other method and one evaluation is one scheduler job. Factoring the step
machine into a per-start runner is also what lets a fit run ``N`` of them concurrently
(local multi-start, the orchestration the base owns); the runner itself is pure ``u``-space
``numpy``, so its math is unit-testable offline against a scipy oracle with no backend.

The method (L-BFGS-B: Cauchy point + subspace min + line search)
----------------------------------------------------------------
The Byrd–Lu–Nocedal–Zhu (1995) algorithm. In sampling space ``u``
(``StartPointOptimizer`` / ``GradientOptimizer``), with the scalar gradient
``g = ∇F`` from #385 at the current point ``x``:

* keep the ``m`` most recent curvature pairs ``(s, y)`` (``s = xₖ₊₁ - xₖ``,
  ``y = gₖ₊₁ - gₖ``) and form the limited-memory Hessian approximation ``B`` in its
  *direct* (not inverse) form -- here a **dense** ``θI`` seed updated by the cached
  pairs (:meth:`_form_b`), since PyBNF fits have small ``n`` (the paper's compact
  ``θI − W M Wᵀ`` products only pay off for large ``n``);
* **generalized Cauchy point** (:meth:`_generalized_cauchy_point`, their Algorithm CP):
  walk the piecewise-linear projected steepest-descent path ``P[x − t·g, l, u]``,
  sort the per-coordinate breakpoints, and stop at the first local minimizer of the
  quadratic model ``m`` along it -- the Cauchy point ``xᶜ`` and the **active set** (the
  coordinates pinned to a bound on the way);
* **subspace minimization** (:meth:`_subspace_min`, their §5 dense primal): minimize the
  quadratic model over the *free* variables from ``xᶜ``, holding the active coordinates
  at their bounds, truncated to stay feasible -- the subspace minimizer ``x̄``. The
  search direction is ``d = x̄ − x``. Both stages are pure in-process linear algebra on
  the cached pairs + gradient -- **no objective evaluations**;
* **backtracking (Armijo) line search** along ``d``: start at step length ``α = 1``
  (the full step lands at ``x̄``) and project each trial ``P[x + α d]`` onto the box
  (:meth:`_u_bounds`); accept the first ``α`` whose *projected* sufficient-decrease
  holds, ``F(P[x+αd]) ≤ F(x) + c₁ · gᵀ(P[x+αd] - x)`` (Bertsekas' projected Armijo,
  testing the **actual** displacement so a clamped coordinate is handled correctly; it
  reduces to the usual ``α gᵀd`` when no bound is active), else shrink
  ``α ← backtrack·α`` and retry;
* on accept, fold the realized ``(s, y)`` into the history (skipping a non-positive-
  curvature pair, which would break the BFGS positive-definite invariant) and step.

Each line-search trial costs **one** objective evaluation; an accepted step's gradient
is assembled from that trial's own simdata (master-side scoring returns it), so the step
needs no re-evaluation. The run stops when the **projected** gradient is flat
(``‖P[x-g]-x‖∞ ≤ lbfgs_grad_tol`` -- the first-order optimality test that respects active
bounds), the accepted step is negligible (``‖s‖ ≤ lbfgs_step_tol·(‖x‖+lbfgs_step_tol)``),
the line search cannot reduce the objective even along steepest descent (a stalled
minimum), or the iteration budget is spent.

In the box interior (no bound active at the solution) the Cauchy point falls short of
its first breakpoint and the subspace minimization runs over every variable, so the step
reduces exactly to the unconstrained limited-memory quasi-Newton step ``−B⁻¹g``; the
active-set machinery only bites when bounds are active at the optimum.

All runner state is plain ``numpy`` / ``float`` / ``list`` (the point, gradient, the
``(s, y)`` history, the line-search scratch) -- picklable, so ``Algorithm.backup``
checkpoints the optimizer (and its list of runners) mid-run, exactly like Powell /
CMA-ES / TRF (ADR-0007). ``B`` and the Cauchy / subspace work are recomputed per
iteration from the cached pairs, so no dense matrix joins the persisted state.
"""

from typing import ClassVar

import numpy as np

from .gradient_base import DONE, GradientOptimizer, GradientRunner
from ...config_schema import PyBNFConfigModel
from ...registry import register_fit_type


class LBFGSConfig(PyBNFConfigModel):
    """L-BFGS-B config fields, co-located with the method (ADR-0006).

    ``lbfgs_grad_tol`` ends the run when the largest component of the **projected**
    gradient ``P[x-g]-x`` (in sampling space) falls below it -- a first-order optimality
    test that reads as the ordinary ``‖g‖∞`` in the interior and as zero on a bound the
    gradient pushes against. ``lbfgs_step_tol`` ends it when an accepted step ``s``
    becomes negligible relative to the point (``‖s‖ ≤ lbfgs_step_tol·(‖x‖+lbfgs_step_tol)``).
    ``lbfgs_history`` is ``m``, the number of recent curvature pairs the limited-memory
    Hessian retains (more pairs ⇒ a richer Hessian model at higher per-step cost).
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


@register_fit_type('lbfgs', family='optimizer', display_name='L-BFGS-B',
                   schema=LBFGSConfig, refiner=True, start_from_box=True)
class LBFGSAlgorithm(GradientOptimizer):
    """Bounded limited-memory BFGS (L-BFGS-B): a method-agnostic multi-start
    orchestrator (:class:`GradientOptimizer`) over per-start :class:`_LBFGSRunner`
    step machines."""

    #: Message label + refiner start-point key (see StartPointOptimizer).
    fit_type = 'lbfgs'
    START_POINT_KEY = 'lbfgs_start_point'
    _method_label = 'L-BFGS-B'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.grad_tol = config.config['lbfgs_grad_tol']
        self.step_tol = config.config['lbfgs_step_tol']
        self.history = config.config['lbfgs_history']
        self.c1 = config.config['lbfgs_c1']
        self.backtrack = config.config['lbfgs_backtrack']
        if 'lbfgs_max_iterations' in config.config:
            self.max_iterations = config.config['lbfgs_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']

    def _start_banner(self):
        return ("Running L-BFGS-B for up to %i iterations from %i start point(s)"
                % (self.max_iterations, self.n_starts))

    def _make_runner(self, u0):
        """One L-BFGS-B step machine seeded at ``u0`` (sampling space), carrying this
        fit's box + tunables. The orchestrator builds one per start."""
        return _LBFGSRunner(u0, self._u_lower, self._u_upper, self.max_iterations,
                            grad_tol=self.grad_tol, step_tol=self.step_tol,
                            history=self.history, c1=self.c1, backtrack=self.backtrack)


class _LBFGSRunner(GradientRunner):
    """One L-BFGS-B start: the picklable step machine, in sampling space ``u``.

    Holds the iterate, the limited-memory curvature history, and the backtracking
    line-search scratch; consumes ``(u_point, score, grad)`` and returns the next ``u``
    to evaluate (or :data:`DONE`). Pure ``numpy`` -- no PSets, objective, or backend (see
    :class:`GradientRunner`). The step math is the full Byrd–Lu–Nocedal–Zhu (1995)
    L-BFGS-B described in the module docstring; the orchestrator
    (:class:`LBFGSAlgorithm` / :class:`GradientOptimizer`) supplies the assembled scalar
    gradient and does the reporting."""

    #: Safety cap on objective evaluations per backtracking line search. With a
    #: descent direction the Armijo step is found in a handful of backtracks, so this
    #: is essentially never reached; on exhaustion the line search falls back to
    #: steepest descent and, failing that, the run stops at the (stalled) minimum.
    _MAX_LINE_EVALS = 30

    def __init__(self, u0, lower, upper, max_iterations, *,
                 grad_tol, step_tol, history, c1, backtrack):
        super().__init__(u0, lower, upper, max_iterations)
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        self.history = history
        self.c1 = c1
        self.backtrack = backtrack
        self.s_list = []           # recent s = xₖ₊₁ - xₖ (limited-memory history)
        self.y_list = []           # recent y = gₖ₊₁ - gₖ
        # Backtracking line-search scratch (one 1-D search at a time).
        self.ls_base = None        # the iterate the line search departs from
        self.ls_fbase = None       # F(ls_base)
        self.ls_grad = None        # ∇F(ls_base)
        self.ls_dir = None         # the (descent) direction d
        self.ls_alpha = None       # current trial step length α
        self.ls_evals = 0          # backtracks taken in this line search
        self.ls_steepest = False   # is this line search the steepest-descent fallback?

    def progress_detail(self):
        return '%i curvature pair(s)' % len(self.s_list)

    def got(self, u_point, score, grad):
        if self.phase == 'init':
            return self._after_init(u_point, score, grad)
        if self.phase == 'line':
            return self._after_line(u_point, score, grad)
        raise RuntimeError(f'Internal error in _LBFGSRunner: phase {self.phase!r}')

    # --- state machine ----------------------------------------------------- #
    def _after_init(self, u_point, score, grad):
        """Seed the state from the start-point evaluation: objective + scalar gradient,
        empty curvature history (so the first step is steepest descent)."""
        self.point = np.array(u_point, dtype=float)
        self.fval = score
        self.grad = grad.gradient
        if not self.n:
            # No free coordinates to optimize (e.g. a reduced-dimension profile-likelihood
            # re-optimization that fixed the sole free parameter) -- trivially converged.
            self.stop_reason = 'no free parameters to optimize'
            return DONE
        if self._gradient_converged():
            self.stop_reason = 'projected gradient already flat at the start point'
            return DONE
        return self._begin_line_search()

    def _begin_line_search(self, steepest=False):
        """Form the search direction (the L-BFGS-B step, or steepest descent) and start
        a backtracking line search along it from the current point.

        Guards descent: an L-BFGS-B direction with ``gᵀd ≥ 0`` (a stale curvature model,
        or a degenerate active set) is not a descent direction, so drop the history and
        retry along steepest descent, which always descends unless the projected gradient
        is itself flat (⇒ stop)."""
        d = -self.grad if steepest else self._direction()
        if self.n and float(self.grad @ d) >= 0.0:
            if not steepest:
                self.s_list, self.y_list = [], []
                return self._begin_line_search(steepest=True)
            self.stop_reason = 'no descent direction (gradient flat)'
            return DONE
        self.ls_base = self.point.copy()
        self.ls_fbase = self.fval
        self.ls_grad = self.grad.copy()
        self.ls_dir = d
        self.ls_alpha = 1.0
        self.ls_evals = 0
        self.ls_steepest = steepest
        return self._submit_trial()

    def _submit_trial(self):
        """Project the current trial step onto the box and return it for evaluation."""
        self.phase = 'line'
        return np.clip(self.ls_base + self.ls_alpha * self.ls_dir,
                       self._u_lower, self._u_upper)

    def _after_line(self, u_point, score, grad):
        """Apply the projected Armijo test to the trial; accept it or backtrack."""
        f_new = score
        trial_point = np.array(u_point, dtype=float)
        disp = trial_point - self.ls_base
        # Projected Armijo sufficient-decrease (Bertsekas): test the *actual*
        # displacement gᵀ(P[x+αd]-x), so a coordinate clamped to a bound is handled
        # correctly; reduces to α·gᵀd when no bound is active. The strict f_new <
        # f_base guard rejects the degenerate case where projection flips the
        # displacement out of descent.
        armijo_rhs = self.ls_fbase + self.c1 * float(self.ls_grad @ disp)
        if np.isfinite(f_new) and f_new <= armijo_rhs and f_new < self.ls_fbase:
            return self._accept(grad, trial_point, f_new, disp)
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
        self.stop_reason = ('line search could not reduce the objective '
                            '(converged or stalled at a minimum)')
        return DONE

    def _accept(self, grad, trial_point, f_new, disp):
        """Accept the trial: fold the realized curvature pair into the history, step to
        it, and either propose the next line search or stop. The accepted trial's own
        gradient (assembled from its simdata) becomes the next iterate's, so no
        re-evaluation is needed."""
        new_grad = grad.gradient
        self._store_pair(disp, new_grad - self.ls_grad)
        step_norm = float(np.linalg.norm(disp))
        self.point = trial_point
        self.fval = f_new
        self.grad = new_grad
        self.iteration += 1
        stop = self._stop_reason(step_norm)
        if stop is not None:
            self.stop_reason = stop
            return DONE
        return self._begin_line_search()

    # --- limited-memory linear algebra (L-BFGS-B) -------------------------- #
    def _store_pair(self, s, y):
        """Append the curvature pair ``(s, y)``, capped at ``lbfgs_history``. Skips a
        non-positive-curvature pair (``sᵀy`` not safely positive) -- storing it would
        break the BFGS positive-definite invariant the dense Hessian ``B`` (and the
        ``1/(sᵀy)`` its rank-two update divides by) relies on; the limited-memory model
        simply keeps its existing pairs."""
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        if float(s @ y) <= 1e-10 * float(y @ y):
            return
        self.s_list.append(s)
        self.y_list.append(y)
        if len(self.s_list) > self.history:
            self.s_list.pop(0)
            self.y_list.pop(0)

    def _form_b(self):
        """The limited-memory BFGS Hessian approximation ``B`` as a dense ``n×n`` matrix
        -- the *direct* form the generalized Cauchy point and subspace minimization need
        (the two-loop recursion gave the *inverse* ``H``).

        Built by applying the ``m`` cached curvature pairs as dense rank-two BFGS updates
        to the scaled seed ``B₀ = θI``, ``θ = (yᵀy)/(sᵀy)`` of the most recent pair
        (Oren–Luenberger). This is exactly the paper's compact form ``θI − W M Wᵀ`` but
        formed densely: PyBNF fits have small ``n``, so the recursion's clarity beats the
        compact-form products (which only pay off for large ``n``). The curvature guard in
        :meth:`_store_pair` keeps every stored pair positive, so each update preserves the
        positive-definite invariant (the ``sᵀBs`` / ``sᵀy`` guards are defensive). With no
        history ``B = I``, so the step degenerates to projected steepest descent."""
        n = self.n
        if not self.s_list:
            return np.eye(n)
        s_last, y_last = self.s_list[-1], self.y_list[-1]
        theta = float(y_last @ y_last) / float(s_last @ y_last)
        b = theta * np.eye(n)
        for s, y in zip(self.s_list, self.y_list):
            bs = b @ s
            s_bs = float(s @ bs)
            sy = float(s @ y)
            if s_bs <= 0.0 or sy <= 0.0:
                continue
            b = b - np.outer(bs, bs) / s_bs + np.outer(y, y) / sy
        return 0.5 * (b + b.T)     # symmetrize against round-off drift

    def _generalized_cauchy_point(self, b):
        """The generalized Cauchy point ``xᶜ`` and the free-variable mask (Byrd–Lu–
        Nocedal–Zhu Algorithm CP, dense form).

        Walks the piecewise-linear projected steepest-descent path
        ``x(t) = P[x − t·g, l, u]`` and returns its first local minimizer of the quadratic
        model ``m(x) = F + gᵀ(x−xₖ) + ½(x−xₖ)ᵀB(x−xₖ)``. Each coordinate reaches its bound
        at a breakpoint ``tᵢ``; processed in increasing order, ``m`` is a convex parabola
        in ``t`` on each segment, so the Cauchy point is where a segment's interior
        minimizer (``−f′/f″``) first lands inside the segment. Coordinates pinned to a
        bound on the way out form the active set; the rest stay free. Returns
        ``(xᶜ, free_mask)`` with ``free_mask[i]`` true for a free (non-active) coordinate."""
        x, g = self.point, self.grad
        lower, upper = self._u_lower, self._u_upper
        n = self.n
        # Breakpoint tᵢ for each coordinate (∞ where the path never hits a bound).
        t_bp = np.full(n, np.inf)
        for i in range(n):
            if g[i] < 0.0 and np.isfinite(upper[i]):
                t_bp[i] = (x[i] - upper[i]) / g[i]
            elif g[i] > 0.0 and np.isfinite(lower[i]):
                t_bp[i] = (x[i] - lower[i]) / g[i]
        t_bp[t_bp < 0.0] = 0.0       # a coord already at/past its bound is fixed at once
        fixed = t_bp == 0.0
        d = np.where(fixed, 0.0, -g)  # path direction (0 on the immediately-fixed coords)
        z = np.zeros(n)               # displacement xᶜ − x accumulated so far
        order = sorted((i for i in range(n) if 0.0 < t_bp[i] < np.inf),
                       key=lambda i: t_bp[i])
        t_old = 0.0
        f1 = float(g @ d)             # model directional derivative on the current segment
        f2 = float(d @ (b @ d))       # model curvature on the current segment
        pos = 0
        while True:
            dt_star = (-f1 / f2) if f2 > 1e-30 else (np.inf if f1 < -1e-30 else 0.0)
            dt_star = max(dt_star, 0.0)
            if pos < len(order):
                t_next = t_bp[order[pos]]
                dt_seg = t_next - t_old
            else:
                t_next, dt_seg = np.inf, np.inf
            if dt_star <= dt_seg:     # interior minimizer reached within this segment
                z = z + dt_star * d
                break
            # Advance to the next breakpoint and fix every coordinate that hits it.
            z = z + dt_seg * d
            while pos < len(order) and t_bp[order[pos]] == t_next:
                i = order[pos]
                z[i] = -t_next * g[i]   # pin to the exact bound displacement
                d[i] = 0.0
                fixed[i] = True
                pos += 1
            t_old = t_next
            if not np.any(d):         # every coordinate now fixed
                break
            gz = g + b @ z            # model gradient at the new segment's start
            f1 = float(gz @ d)
            f2 = float(d @ (b @ d))
        xc = np.clip(x + z, lower, upper)
        return xc, ~fixed

    def _subspace_min(self, b, xc, free_mask):
        """Subspace minimization from the Cauchy point ``xᶜ`` (BLNZ §5, dense primal).

        Minimizes the quadratic model over the **free** variables, holding the active
        variables at their Cauchy-point bounds, then truncates the move so it stays inside
        the box (their feasibility step). The free block solves
        ``B_FF r_F = −(g_F + B_FA r_A)`` with ``r_A = xᶜ_A − x_A`` the active displacement;
        ``x̄_F = x_F + r_F``. With every variable free this is the unconstrained
        limited-memory step ``−B⁻¹g``; with all variables active it leaves ``xᶜ``
        unchanged. Returns the subspace minimizer ``x̄``."""
        x, g = self.point, self.grad
        lower, upper = self._u_lower, self._u_upper
        free = np.flatnonzero(free_mask)
        if free.size == 0:
            return xc.copy()
        active = np.flatnonzero(~free_mask)
        rhs = -g[free]
        if active.size:
            rhs = rhs - b[np.ix_(free, active)] @ (xc[active] - x[active])
        b_ff = b[np.ix_(free, free)]
        try:
            r_free = np.linalg.solve(b_ff, rhs)
        except np.linalg.LinAlgError:
            r_free = np.linalg.lstsq(b_ff, rhs, rcond=None)[0]
        # Move from xᶜ toward the unconstrained subspace minimizer, truncated to the box.
        du = np.zeros(self.n)
        du[free] = (x[free] + r_free) - xc[free]
        alpha = 1.0
        for i in free:
            if du[i] > 0.0 and np.isfinite(upper[i]):
                alpha = min(alpha, (upper[i] - xc[i]) / du[i])
            elif du[i] < 0.0 and np.isfinite(lower[i]):
                alpha = min(alpha, (lower[i] - xc[i]) / du[i])
        alpha = min(max(alpha, 0.0), 1.0)
        return np.clip(xc + alpha * du, lower, upper)

    def _direction(self):
        """The L-BFGS-B search direction ``d = x̄ − x``: the generalized Cauchy point
        ``xᶜ`` (active-set identification) followed by subspace minimization to the free
        minimizer ``x̄``, on the dense limited-memory Hessian ``B``. Pure in-process linear
        algebra on the cached ``(s, y)`` pairs + current gradient; no objective evals."""
        b = self._form_b()
        xc, free_mask = self._generalized_cauchy_point(b)
        xbar = self._subspace_min(b, xc, free_mask)
        return xbar - self.point

    # --- convergence ------------------------------------------------------- #
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
