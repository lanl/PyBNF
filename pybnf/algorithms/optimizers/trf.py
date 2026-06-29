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
The method is reimplemented as an explicit, *picklable* state machine inside the
run-loop contract -- ``start_run`` / ``got_result`` only, no ``run()`` override
(ADR-0007) -- so backup/resume work like every other method and one evaluation is
one scheduler job.

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
(``fit_type = lbfgs``, #386's fallback). Multi-start orchestration and full TRF
reflective transformations are tracked as the follow-ups in #386.

All state is plain ``numpy`` / ``float`` (the point, residual, Jacobian, damping) --
picklable, so ``Algorithm.backup`` checkpoints the optimizer mid-run.
"""

import logging
from typing import ClassVar

import numpy as np

from .gradient_base import GradientOptimizer
from ...config_schema import PyBNFConfigModel
from ...printing import PybnfError, print1, print2
from ...registry import register_fit_type

logger = logging.getLogger('pybnf.algorithms')


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
    """Bounded Levenberg–Marquardt least-squares as a picklable reactor state machine."""

    #: Message label + refiner start-point key (see StartPointOptimizer).
    fit_type = 'trf'
    START_POINT_KEY = 'trf_start_point'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.n = len(self.variables)
        self.grad_tol = config.config['trf_grad_tol']
        self.step_tol = config.config['trf_step_tol']
        self.tau = config.config['trf_tau']
        if 'trf_max_iterations' in config.config:
            self.max_iterations = config.config['trf_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']
        self._u_lower, self._u_upper = self._u_bounds()
        self.start_pset = self._resolve_start_pset()
        self._init_state()

    def _init_state(self):
        """(Re)initialize the mutable LM state. Filled in by start_run / got_result;
        all plain float / ndarray so the optimizer pickles for backup/resume."""
        self.point = None          # current accepted iterate (u-space)
        self.fval = None           # objective F(point) (== res.score)
        self.A = None              # JᵀJ at point (n, n)
        self.g = None              # Jᵀr at point (n,)
        self.mu = None             # LM damping
        self.nu = 2.0              # Nielsen reject-acceleration factor
        self.iteration = 0
        self.phase = None          # 'init' | 'step'
        self.trial_u = None        # the u-vector currently out for evaluation
        self.trial_delta = None    # the (box-projected) step that produced trial_u
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
        once it returns. LM is serial -- one evaluation (the trial) per iteration."""
        self.phase = phase
        self.trial_u = np.array(u, dtype=float)
        self.probe_counter += 1
        name = 'trf_%i' % self.probe_counter
        return [self._pset_from_u(u, name=name)]

    def got_result(self, res):
        if self.phase == 'init':
            return self._after_init(res)
        if self.phase == 'step':
            return self._after_step(res)
        raise RuntimeError(f'Internal error in TRFAlgorithm: phase {self.phase!r}')

    # --- state machine ----------------------------------------------------- #
    def start_run(self):
        print2("Running trust-region least-squares (Levenberg–Marquardt) for up to "
               "%i iterations" % self.max_iterations)
        # Activate the gradient path (enable sensitivities + build routings) before
        # the model scatter; start from the resolved start point / box center.
        self._setup_gradient_path()
        self.point = self._u_from_pset(self.start_pset)
        return self._submit(self.point, 'init')

    def _after_init(self, res):
        """Seed the LM state from the start-point evaluation: residual/Jacobian,
        Gauss–Newton model, and the initial damping ``μ₀``."""
        grad = self._least_squares_gradient(res)
        self.point = self._u_from_pset(res.pset)
        self.fval = float(res.score)
        self._set_model(grad)
        self.mu = self.tau * float(np.max(np.diag(self.A))) if self.n else 0.0
        if self._gradient_converged():
            logger.info('TRF converged at the start point (gradient already flat)')
            return 'STOP'
        return self._propose_step()

    def _after_step(self, res):
        """Accept or reject the trial by its gain ratio, adapt the damping, and either
        propose the next step or stop."""
        f_new = float(res.score)
        delta = self.trial_delta
        predicted = self._predicted_reduction(delta)
        actual = self.fval - f_new
        rho = actual / predicted if predicted > 0.0 else (1.0 if actual > 0.0 else -1.0)

        if rho > 0.0:
            # Accept: the trial's own residual/Jacobian (assembled here) become the
            # next iterate's, so no re-evaluation is needed.
            grad = self._least_squares_gradient(res)
            step_norm = float(np.linalg.norm(delta))
            self.point = self._u_from_pset(res.pset)
            self.fval = f_new
            self._set_model(grad)
            self.mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            self.nu = 2.0
            self.iteration += 1
            self._report()
            stop = self._stop_reason(step_norm)
            if stop is not None:
                logger.info('TRF stopping: %s', stop)
                return 'STOP'
            return self._propose_step()

        # Reject: grow the damping (shorter, more gradient-like step) and re-solve
        # from the same point. Counts against the iteration budget so a stalled run
        # cannot loop forever.
        self.mu *= self.nu
        self.nu *= 2.0
        self.iteration += 1
        if not np.isfinite(self.mu) or self.iteration >= self.max_iterations:
            logger.info('TRF stopping: %s', 'damping diverged' if not np.isfinite(self.mu)
                        else 'reached max_iterations (%i)' % self.max_iterations)
            return 'STOP'
        return self._propose_step()

    def _propose_step(self):
        """Solve the damped normal equations, project the step into the box, and
        submit the trial evaluation."""
        delta = self._solve_lm_step()
        trial = np.clip(self.point + delta, self._u_lower, self._u_upper)
        self.trial_delta = trial - self.point   # the actually-taken (clamped) step
        return self._submit(trial, 'step')

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

    def _least_squares_gradient(self, res):
        """Assemble the gradient at ``res`` and require an **exact** least-squares
        residual. TRF models the objective as ``½‖r‖²``; an objective that is not an
        exact sum of squares (estimated scale, Laplace/count family, constraints) has
        no faithful residual, so refuse it with a pointer to the L-BFGS-B fallback
        rather than silently optimizing the wrong surface."""
        grad = self.gradient_at(res)
        if not grad.least_squares_exact:
            raise PybnfError(
                "fit_type = trf needs an exact least-squares residual (a Gaussian or "
                "Student-t objective with a fixed noise scale and no constraints), but "
                "this fit's objective is not an exact sum of squares.",
                "Use the gradient quasi-Newton fallback 'fit_type = lbfgs', which "
                "consumes the scalar gradient and handles estimated noise scales, the "
                "Laplace / count families, and constraint penalties.")
        return grad

    # --- convergence / reporting ------------------------------------------- #
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

    def _report(self):
        if self.iteration % self.config.config['output_every'] == 0:
            self.output_results()
        msg = 'Completed %i of %i TRF iterations' % (self.iteration, self.max_iterations)
        (print1 if self.iteration % 10 == 0 else print2)(msg)
        print2('Current best objective: %f, damping mu %g' % (self.fval, self.mu))
