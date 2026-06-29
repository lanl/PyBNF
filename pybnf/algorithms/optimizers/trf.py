"""Trust-region / Levenberg–Marquardt least-squares optimizer (``trf`` fit type, #386).

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
multi-start, the orchestration the base owns).

The method (Levenberg–Marquardt, Madsen–Nielsen damping)
--------------------------------------------------------
In sampling space ``u`` (``StartPointOptimizer`` / ``GradientOptimizer``), with the
residual ``r`` and residual-Jacobian ``J`` from #385 at the current point:

* gradient ``g = Jᵀr`` and Gauss–Newton Hessian ``A = JᵀJ``;
* solve the damped normal equations ``(A + μ I) δ = -g`` for the step ``δ``;
* project ``point + δ`` into the box (bounded ``uniform_var`` priors give a finite
  box; an unbounded ``var`` start gives ``±inf`` and never clamps);
* accept the step when the gain ratio ``ρ = (F(x) - F(x+δ)) / (L(0) - L(δ))`` is
  positive, and adapt ``μ`` by Nielsen's rule (``μ ← μ·max(⅓, 1-(2ρ-1)³)``, ``ν←2``
  on accept; ``μ ← μ·ν``, ``ν ← 2ν`` on reject). ``μ`` is the trust-region knob: a
  large ``μ`` is a short gradient-descent step, a small ``μ`` a full Gauss–Newton
  step.

Each iteration costs **one** objective evaluation (the trial). On accept, the trial's
own residual/Jacobian -- already assembled, since master-side scoring returns the
simdata -- become the next iterate's, so an accepted step needs no re-evaluation; on
reject, only ``μ`` grows and the step is re-solved from the same ``r``/``J``. The run
stops when the gradient is flat (``‖g‖∞ ≤ trf_grad_tol``), the step is tiny
(``‖δ‖ ≤ trf_step_tol·(‖x‖+trf_step_tol)``), or the iteration budget is spent.

Scope (this cut). TRF consumes the **exact least-squares residual** -- the Gaussian
(any scale/location) and the Student-t (#459). A fit whose objective is not an exact
sum of squares (an estimated noise scale, a Laplace / count family, active
constraints; ``GradientResult.least_squares_exact == False``) has no faithful
residual model, so this optimizer refuses it with a pointer to the L-BFGS-B path
(``fit_type = lbfgs``, #386's fallback). Bounds are handled by **clipping** the LM step
into the box (``_propose_step``); the full Trust-Region-Reflective bound handling
(Branch–Coleman–Li, matching ``scipy``'s ``method='trf'``), which converges cleanly at a
bound-active optimum where clipping can stall, is tracked in #460. Local multi-start is
provided by :class:`GradientOptimizer` (the base runs ``N`` independent
:class:`_TRFRunner` starts concurrently and keeps the global best).

All runner state is plain ``numpy`` / ``float`` (the point, residual, Jacobian, damping)
-- picklable, so ``Algorithm.backup`` checkpoints the optimizer (and its list of runners)
mid-run.
"""

from typing import ClassVar

import numpy as np

from .gradient_base import DONE, GradientOptimizer, GradientRunner
from ...config_schema import PyBNFConfigModel
from ...printing import PybnfError
from ...registry import register_fit_type


class TRFConfig(PyBNFConfigModel):
    """TRF / Levenberg–Marquardt config fields, co-located with the method (ADR-0006).

    ``trf_grad_tol`` ends the run when the largest component of the least-squares
    gradient ``Jᵀr`` (in sampling space) falls below it -- a first-order optimality
    test. ``trf_step_tol`` ends it when an accepted step ``δ`` becomes negligible
    relative to the point (``‖δ‖ ≤ trf_step_tol·(‖x‖+trf_step_tol)``). ``trf_tau``
    scales the initial damping ``μ₀ = trf_tau · max(diag JᵀJ)``: a small value (≈1e-3)
    starts near a full Gauss–Newton step for a good initial guess, a larger one a more
    cautious gradient-descent start. Like Powell's cycle budget, ``trf_max_iterations``
    is runtime-guarded -- it defaults to the global ``max_iterations`` when unset -- so
    it is a valid key but not a schema field. ``trf_start_point`` is internal (the
    refiner injects it), so it is not modeled here either."""

    trf_grad_tol: float = 1e-8
    trf_step_tol: float = 1e-8
    trf_tau: float = 1e-3

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'trf_max_iterations'})


@register_fit_type('trf', family='optimizer', display_name='Trust-Region Least-Squares',
                   schema=TRFConfig, refiner=True, start_from_box=True)
class TRFAlgorithm(GradientOptimizer):
    """Bounded Levenberg–Marquardt least-squares: a method-agnostic multi-start
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
        self.tau = config.config['trf_tau']
        if 'trf_max_iterations' in config.config:
            self.max_iterations = config.config['trf_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']

    def _start_banner(self):
        return ("Running trust-region least-squares (Levenberg–Marquardt) for up to "
                "%i iterations from %i start point(s)" % (self.max_iterations, self.n_starts))

    def _make_runner(self, u0):
        """One Levenberg–Marquardt step machine seeded at ``u0`` (sampling space),
        carrying this fit's box + tunables. The orchestrator builds one per start."""
        return _TRFRunner(u0, self._u_lower, self._u_upper, self.max_iterations,
                          grad_tol=self.grad_tol, step_tol=self.step_tol, tau=self.tau)


class _TRFRunner(GradientRunner):
    """One trust-region/Levenberg–Marquardt start: the picklable step machine, in
    sampling space ``u``.

    Holds the iterate, the Gauss–Newton model (``A = JᵀJ``, ``g = Jᵀr``), and the LM
    damping; consumes ``(u_point, score, grad)`` and returns the next ``u`` to evaluate
    (or :data:`DONE`). Pure ``numpy`` -- no PSets, objective, or backend (see
    :class:`GradientRunner`). The orchestrator (:class:`TRFAlgorithm` /
    :class:`GradientOptimizer`) supplies the assembled :class:`GradientResult` and does
    the reporting; this runner requires it to be an **exact** least-squares residual
    (:meth:`_require_exact`)."""

    def __init__(self, u0, lower, upper, max_iterations, *, grad_tol, step_tol, tau):
        super().__init__(u0, lower, upper, max_iterations)
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        self.tau = tau
        self.A = None              # JᵀJ at point (n, n)
        self.g = None              # Jᵀr at point (n,)
        self.mu = None             # LM damping
        self.nu = 2.0              # Nielsen reject-acceleration factor
        self.trial_delta = None    # the (box-projected) step currently out for evaluation

    def progress_detail(self):
        return 'damping mu %g' % self.mu

    def got(self, u_point, score, grad):
        if self.phase == 'init':
            return self._after_init(u_point, score, grad)
        if self.phase == 'step':
            return self._after_step(u_point, score, grad)
        raise RuntimeError(f'Internal error in _TRFRunner: phase {self.phase!r}')

    # --- state machine ----------------------------------------------------- #
    def _after_init(self, u_point, score, grad):
        """Seed the LM state from the start-point evaluation: residual/Jacobian,
        Gauss–Newton model, and the initial damping ``μ₀``."""
        gr = self._require_exact(grad)
        self.point = np.array(u_point, dtype=float)
        self.fval = score
        self._set_model(gr)
        self.mu = self.tau * float(np.max(np.diag(self.A))) if self.n else 0.0
        if self._gradient_converged():
            self.stop_reason = 'gradient already flat at the start point'
            return DONE
        return self._propose_step()

    def _after_step(self, u_point, score, grad):
        """Accept or reject the trial by its gain ratio, adapt the damping, and either
        propose the next step or stop."""
        f_new = score
        delta = self.trial_delta
        predicted = self._predicted_reduction(delta)
        actual = self.fval - f_new
        rho = actual / predicted if predicted > 0.0 else (1.0 if actual > 0.0 else -1.0)

        if rho > 0.0:
            # Accept: the trial's own residual/Jacobian (assembled by the orchestrator)
            # become the next iterate's, so no re-evaluation is needed.
            gr = self._require_exact(grad)
            step_norm = float(np.linalg.norm(delta))
            self.point = np.array(u_point, dtype=float)
            self.fval = f_new
            self._set_model(gr)
            self.mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            self.nu = 2.0
            self.iteration += 1
            stop = self._stop_reason(step_norm)
            if stop is not None:
                self.stop_reason = stop
                return DONE
            return self._propose_step()

        # Reject: grow the damping (shorter, more gradient-like step) and re-solve
        # from the same point. Counts against the iteration budget so a stalled run
        # cannot loop forever.
        self.mu *= self.nu
        self.nu *= 2.0
        self.iteration += 1
        if not np.isfinite(self.mu) or self.iteration >= self.max_iterations:
            self.stop_reason = ('damping diverged' if not np.isfinite(self.mu)
                                else 'reached max_iterations (%i)' % self.max_iterations)
            return DONE
        return self._propose_step()

    def _propose_step(self):
        """Solve the damped normal equations, project the step into the box, and return
        the trial point for evaluation."""
        delta = self._solve_lm_step()
        trial = np.clip(self.point + delta, self._u_lower, self._u_upper)
        self.trial_delta = trial - self.point   # the actually-taken (clamped) step
        self.phase = 'step'
        return trial

    # --- linear algebra ---------------------------------------------------- #
    def _solve_lm_step(self):
        """Solve ``(A + μ I) δ = -g`` for the LM step. Falls back to a least-squares
        solve if the damped Hessian is singular (μ keeps it positive-definite in
        practice, so this is a guard, not the usual path)."""
        if not self.n:
            return np.zeros(0)
        damped = self.A + self.mu * np.eye(self.n)
        try:
            return np.linalg.solve(damped, -self.g)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(damped, -self.g, rcond=None)[0]

    def _predicted_reduction(self, delta):
        """The LM model's predicted objective decrease for step ``δ``:
        ``L(0) - L(δ) = ½ δᵀ(μ δ - g)`` (Madsen–Nielsen) -- positive for a genuine LM
        step, the denominator of the gain ratio."""
        return 0.5 * float(delta @ (self.mu * delta - self.g))

    def _set_model(self, grad):
        """Cache the Gauss–Newton model ``A = JᵀJ``, ``g = Jᵀr`` at the current point
        from an assembled :class:`GradientResult` (residual / Jacobian in sampling
        space)."""
        J = grad.jacobian
        r = grad.residual
        self.A = J.T @ J
        self.g = J.T @ r

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
    def _gradient_converged(self):
        return bool(self.n) and float(np.max(np.abs(self.g))) <= self.grad_tol

    def _stop_reason(self, step_norm):
        """A termination string after an accepted step, or None to keep going."""
        if self._gradient_converged():
            return 'gradient is flat (‖Jᵀr‖∞ ≤ %g)' % self.grad_tol
        point_norm = float(np.linalg.norm(self.point))
        if step_norm <= self.step_tol * (point_norm + self.step_tol):
            return 'step is negligible (‖δ‖ ≤ %g)' % (self.step_tol * (point_norm + self.step_tol))
        if self.iteration >= self.max_iterations:
            return 'reached max_iterations (%i)' % self.max_iterations
        return None
