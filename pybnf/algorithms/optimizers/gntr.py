"""General-objective Fisher/Gauss-Newton trust-region optimizer (``gntr`` fit type, #481).

The missing cell of the (objective x curvature-model) matrix. ``trf`` gives a trust-region
step with a ``J^T J`` (Gauss-Newton / empirical-Fisher) Hessian, but **only** for an exact
least-squares objective; the moment the objective stops being a pure sum of squares -- an
estimated noise scale, a Laplace / count likelihood, or an active constraint -- the gradient
path drops to ``lbfgs``, a limited-memory quasi-Newton method whose Hessian is built from
gradient differences. On the ill-conditioned NLL landscapes typical of those problems that is
a real downgrade. ``gntr`` closes the gap: a native, picklable trust-region optimizer whose
Hessian is the **expected-Fisher / Gauss-Newton information**

    H = sum_i kappa_i s_i s_i^T   (+ estimated-noise and constraint blocks)

built entirely from the #385 forward-sensitivity plumbing (``s_i = d(prediction_i)/d theta``,
``kappa_i`` the per-observation location Fisher) plus small analytic per-family factors
(:func:`~pybnf.gradient.assembly.assemble_gradient_and_fisher_hessian`), extending ``trf``'s step
quality to any NLL objective ``lbfgs`` handles. It consumes the **same scalar gradient** ``lbfgs``
does (``GradientResult.gradient``); only the curvature model differs.

Why native (not a scipy/pip trust-region driver): the same reason ``trf`` / ``lbfgs`` /
``powell`` document -- a blocking driver calls ``fun`` / ``jac`` synchronously and cannot farm
its evaluations to PyBNF's distributed propose/score loop. The method is an explicit, *picklable*
step machine driven by :class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer`
inside the run-loop contract (no ``run()`` override, ADR-0007), so backup/resume and local
multi-start work exactly as for every other ``fit_type``.

The step math -- reusing ``trf``'s Coleman-Li reflective core via a pseudo-Jacobian
--------------------------------------------------------------------------------------
``trf``'s bound-constrained trust-region-reflective machinery (the Branch-Coleman-Li affine
scaling, the augmented-Jacobian SVD subproblem solve, the reflective step selection) is written
in terms of a residual ``r`` and residual-Jacobian ``J`` with Hessian ``J^T J`` and gradient
``J^T r``. For a general PSD Hessian ``H`` and gradient ``g`` this runner **reuses that whole
apparatus unchanged** by building a *pseudo* residual model: ridge-regularise ``H <- H + lambda
I`` (``lambda`` proportional to ``trace(H)/n``, making ``H`` strictly positive definite so no
gradient direction is projected onto a flat curvature direction and lost -- the key robustness
fix for the constraint / estimated-noise blocks whose curvature can floor to zero), eigen-
decompose ``H = Q diag(w) Q^T``, and set

    J_pseudo = diag(sqrt(w)) Q^T          (so J_pseudo^T J_pseudo == H)
    r_pseudo = diag(1/sqrt(w)) Q^T g       (so J_pseudo^T r_pseudo == g)

Feeding ``(jacobian=J_pseudo, residual=r_pseudo)`` into :class:`~pybnf.algorithms.optimizers.trf._TRFRunner`
reproduces the exact Coleman-Li-scaled Newton step ``p = -(D H D + C)^{-1} D g`` for the
quadratic model ``1/2 s^T H s + g^T s``. Nothing else in the ``trf`` runner assumes the residual
norm equals the objective -- its accept/reject and trust-radius updates use the **real** objective
score (``self.cost`` / ``self.fval`` set from ``score``), and the predicted reduction comes from
the quadratic model, so ``1/2 ||r_pseudo||^2 != score`` is harmless. For a pure Gaussian
least-squares fit (``H = J^T J``, ``g = J^T r``) the step reduces to exactly ``trf``'s / scipy's,
the offline oracle in ``tests/test_gradient_runner.py``.

Scope (this cut -- the tractable configurations; the rest cleanly refuse to ``lbfgs``). Supports
an estimated-sigma Gaussian (``chi_sq_dynamic``), a fixed-scale Laplace, a fixed-dispersion
negative-binomial (MEAN), and a Gaussian fit with (static-hinge) constraints. Refuses -- with a
:class:`~pybnf.printing.PybnfError` pointing at ``job_type = lbfgs`` -- the coupled corners whose
Fisher this cut does not assemble: a MEAN-on-log estimated scale, the count family's free
dispersion or MEDIAN centering, the Student-t estimated-df 2x2 block, and an estimated constraint
scale. Local multi-start is provided by :class:`GradientOptimizer` (``N`` concurrent starts,
global best).

All runner state is plain ``numpy`` / ``float`` (inherited from ``_TRFRunner`` -- the point, the
pseudo residual model, the trust radius, the cached scaling + SVD), so ``Algorithm.backup``
checkpoints the optimizer mid-run like every other method (ADR-0007).
"""

from typing import ClassVar

import numpy as np

from .gradient_base import GradientOptimizer
from .trf import _TRFRunner, _UnusableModel
from ...config_schema import PyBNFConfigModel
from ...gradient import (
    assemble_constraint_hessian,
    assemble_gradient_and_fisher_hessian,
    assemble_marginal_time_gradient,
)
from ...printing import PybnfError
from ...registry import register_fit_type


class GNTRConfig(PyBNFConfigModel):
    """GNTR (Fisher/Gauss-Newton trust-region) config fields, co-located with the method
    (ADR-0006).

    ``gntr_grad_tol`` and ``gntr_step_tol`` are the first-order-optimality and negligible-step
    stopping tolerances, identical in meaning to ``trf``'s (the runner is a ``trf`` runner fed a
    pseudo-Jacobian): the run ends when the largest component of the **scaled** gradient
    ``v * g`` falls below ``gntr_grad_tol`` (the Coleman-Li optimality test that reads as zero on
    an active bound) or an accepted step becomes negligible relative to the point. ``gntr_ridge``
    is the relative Levenberg ridge added to the EFIM Hessian before the pseudo-Jacobian
    factorisation (``lambda = gntr_ridge * trace(H)/n``) -- large enough to keep ``H`` strictly
    positive definite (so every gradient direction is representable and the pseudo residual is
    bounded), small enough not to perturb the Newton step; the default ``1e-10`` is negligible on
    a well-conditioned fit. Like ``trf`` / ``lbfgs``, ``gntr_max_iterations`` is runtime-guarded
    (it defaults to the global ``max_iterations`` when unset), so it is a valid key but not a
    schema field, and ``gntr_start_point`` is internal (the refiner injects it)."""

    gntr_grad_tol: float = 1e-8
    gntr_step_tol: float = 1e-8
    gntr_ridge: float = 1e-10

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'gntr_max_iterations'})


@register_fit_type('gntr', family='optimizer',
                   display_name='Fisher/Gauss-Newton Trust Region',
                   schema=GNTRConfig, refiner=True, start_from_box=True)
class GNTRAlgorithm(GradientOptimizer):
    """General-objective Fisher/Gauss-Newton trust-region: a method-agnostic multi-start
    orchestrator (:class:`GradientOptimizer`) over per-start :class:`_GNTRRunner` step machines.
    It differs from ``trf`` in two assembly seams -- it builds the gradient and data-fit EFIM in
    one scored-point pass (:meth:`_assemble_objective_gradient`), then attaches any constraint
    curvature (:meth:`_attach_curvature`) -- and its runner consumes ``(gradient, hessian)``
    instead of a residual model. Everything else (multi-start, routing, gates, reporting) is
    inherited."""

    #: Message label + refiner start-point key (see StartPointOptimizer).
    fit_type = 'gntr'
    START_POINT_KEY = 'gntr_start_point'
    _method_label = 'GNTR'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.grad_tol = config.config['gntr_grad_tol']
        self.step_tol = config.config['gntr_step_tol']
        self.ridge = config.config['gntr_ridge']
        if 'gntr_max_iterations' in config.config:
            self.max_iterations = config.config['gntr_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']

    def _start_banner(self):
        return ("Running Fisher/Gauss-Newton trust-region (general-objective EFIM) for up to "
                "%i iterations from %i start point(s)" % (self.max_iterations, self.n_starts))

    def _make_runner(self, u0):
        """One Fisher/Gauss-Newton trust-region step machine seeded at ``u0`` (sampling space),
        carrying this fit's box + tunables. The orchestrator builds one per start."""
        return _GNTRRunner(u0, self._u_lower, self._u_upper, self.max_iterations,
                           grad_tol=self.grad_tol, step_tol=self.step_tol, ridge=self.ridge)

    def _assemble_objective_gradient(self, experiments, free_params):
        """Build the scalar/residual gradient and data-fit EFIM in one point walk (#488).

        A **marginal-time** objective (``time_error``, ADR-0113) has no matched-row Fisher; its
        Gauss-Newton curvature is the per-datum-score outer product ``Σ_k w_k g_k g_kᵀ`` (the
        empirical Fisher), assembled alongside the scalar gradient by the sensitivity-chaining
        assembler with ``include_fisher=True``."""
        if getattr(self.objective, 'marginalizes_time', False):
            return assemble_marginal_time_gradient(
                self.objective, experiments, free_params, include_fisher=True)
        return assemble_gradient_and_fisher_hessian(self.objective, experiments, free_params)

    def _attach_curvature(self, grad, res, experiments, free_params, routings):
        """Add constraint curvature to the already-attached data-fit EFIM (#481/#488).

        :meth:`_assemble_objective_gradient` builds the objective gradient and data-fit Hessian
        together. For a constrained fit this hook adds the constraint Gauss-Newton block
        (:func:`~pybnf.gradient.assembly.assemble_constraint_hessian`). It runs inside
        ``gradient_at``'s ``GradientNotSupported`` guard, so an unsupported-curvature corner
        refuses cleanly to ``lbfgs`` (:meth:`_unsupported_gradient_error`)."""
        if self.config.constraints:
            grad.hessian = grad.hessian + assemble_constraint_hessian(
                self.config.constraints, res.simdata, routings, free_params)

    def _unsupported_gradient_error(self, exc):
        """Point the refusal at ``lbfgs`` (which consumes the scalar gradient and needs no Fisher
        Hessian), not at a metaheuristic: the corners ``gntr`` cannot build an EFIM Hessian for --
        a MEAN-on-log estimated scale, the count family's free dispersion / MEDIAN centering, the
        Student-t 2-parameter block, an estimated constraint scale -- all fit under ``lbfgs``."""
        return PybnfError(
            "Fisher/Gauss-Newton trust-region fitting (job_type = gntr) cannot build the EFIM "
            "Hessian for this fit's objective: %s" % exc,
            hint="Use the gradient quasi-Newton fallback 'job_type = lbfgs', which consumes the "
                 "scalar gradient and needs no Fisher Hessian -- it handles the estimated noise "
                 "scales, the Laplace / count families, and the constraint penalties whose "
                 "curvature the EFIM trust-region step does not (yet) assemble.")


class _GNTRRunner(_TRFRunner):
    """One Fisher/Gauss-Newton trust-region start: a :class:`~pybnf.algorithms.optimizers.trf._TRFRunner`
    fed a **pseudo-Jacobian** built from the general EFIM ``(gradient, hessian)`` rather than a
    least-squares residual model (#481).

    It overrides exactly two seams of the ``trf`` runner: :meth:`_require_exact` (a no-op -- the
    curvature is the EFIM, not an exact residual, so any objective the assembly could build a
    Hessian for is accepted; that gate is upstream in
    :meth:`GNTRAlgorithm._assemble_objective_gradient` / :meth:`GNTRAlgorithm._attach_curvature`),
    and :meth:`_set_model` (ridge-regularise ``H`` and eigen-factor ``(g, H)`` into the pseudo
    ``(J, r)`` the inherited step machine consumes). Everything else -- the Coleman-Li scaling,
    the augmented-Jacobian SVD trust-region subproblem, the reflective step selection, the
    accept/reject + trust-radius state machine, the convergence tests, picklability -- is
    inherited unchanged. See the module docstring for why the reduction is exact."""

    #: What this runner steps from, for the terminated-start message (#528).
    _model_label = 'Fisher model (gradient + EFIM Hessian)'

    def __init__(self, u0, lower, upper, max_iterations, *, grad_tol, step_tol, ridge):
        super().__init__(u0, lower, upper, max_iterations, grad_tol=grad_tol, step_tol=step_tol)
        self.ridge = ridge

    def progress_detail(self):
        return 'trust radius %g (EFIM)' % self.Delta

    def _model_is_usable(self, grad):
        """GNTR steps from ``(gradient, EFIM Hessian)``, not from a residual model, so both of
        those are what must be finite -- overriding both the base's gradient-only check and
        ``trf``'s residual/Jacobian one (the pseudo residual model is *derived* here, in
        :meth:`_set_model`, so checking it would be checking the symptom). A **missing**
        Hessian is deliberately not this predicate's business: that is an internal wiring
        error :meth:`_set_model` raises loudly, not a bad point to be skipped over (#528)."""
        if grad is None:
            return False
        hessian = getattr(grad, 'hessian', None)
        if hessian is None:
            return True                      # -> _set_model raises the wiring error
        return self._all_finite(grad.gradient, hessian)

    def _require_exact(self, grad):
        """Accept any assembled gradient: the EFIM Hessian is the curvature model, not an exact
        least-squares residual, so the ``trf`` runner's exact-residual gate does not apply. The
        real gate -- whether the Fisher Hessian could be assembled at all -- fired upstream in
        :meth:`GNTRAlgorithm._assemble_objective_gradient` or, for constraints,
        :meth:`GNTRAlgorithm._attach_curvature` (raising :class:`GradientNotSupported` for an
        out-of-scope corner)."""
        return grad

    def _set_model(self, grad):
        """Build the pseudo residual model ``(J, r)`` from the general ``(gradient, hessian)`` so
        the inherited ``trf`` step machine computes the EFIM-scaled Newton step.

        Ridge-regularise ``H <- H + lambda I`` (``lambda = ridge * trace(H)/n``, an absolute floor
        of ``ridge`` when the trace is zero) to make ``H`` strictly positive definite -- so no
        gradient component lands in a flat curvature direction that the factorisation would project
        away (the constraint / estimated-noise blocks can floor to zero curvature), and the pseudo
        residual stays bounded. Then eigen-decompose the symmetric ``H = Q diag(w) Q^T`` and set
        ``J = diag(sqrt(w)) Q^T`` (so ``J^T J == H``) and ``r = diag(1/sqrt(w)) Q^T g`` (so
        ``J^T r == g``). ``self.g`` recomputed as ``J^T r`` is exactly ``g`` -- the value the
        Coleman-Li scaling and the convergence test read."""
        if getattr(grad, 'hessian', None) is None:
            # The EFIM leaf attaches the data-fit Hessian during combined objective assembly
            # before the runner ever sees the gradient; a None here means this runner was driven
            # off the residual-form (trf/lbfgs) path by mistake. Fail fast with the cause rather
            # than an opaque numpy error deep in the eigen-factorisation.
            raise PybnfError(
                "The GNTR (Fisher/Gauss-Newton trust-region) runner requires an assembled "
                "EFIM Hessian, but the gradient carried none. This is an internal wiring "
                "error -- a gntr runner must be driven by GNTRAlgorithm, which attaches the "
                "Hessian in _assemble_objective_gradient.")
        g = np.asarray(grad.gradient, dtype=float)
        hessian = np.asarray(grad.hessian, dtype=float)
        n = self.n
        lam = self.ridge * (float(np.trace(hessian)) / n) if n else 0.0
        if lam <= 0.0:
            lam = self.ridge
        hessian = 0.5 * (hessian + hessian.T) + lam * np.eye(n)
        try:
            w, q = np.linalg.eigh(hessian)
        except np.linalg.LinAlgError:
            # A finite but pathological Hessian LAPACK cannot diagonalize. Terminating this
            # start beats unwinding out through got_result and aborting the fit (#528); the
            # non-finite case never reaches here (_model_is_usable rejects it upstream).
            raise _UnusableModel('the EFIM Hessian at this point could not be diagonalized '
                                 '(LAPACK failed to converge)')
        # Positive definite after the ridge; the floor only mops up eigen roundoff.
        w = np.clip(w, lam * 1e-12, None)
        sqrt_w = np.sqrt(w)
        self.J = (q * sqrt_w).T              # rows sqrt(w_i) * q_i^T  =>  J^T J == H
        self.r = (q.T @ g) / sqrt_w          # J^T r == Q Q^T g == g
        self.g = self.J.T @ self.r
        self.m = n
