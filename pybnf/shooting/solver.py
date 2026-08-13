"""The inner solver: ``gntr``'s step machine, driven against an augmented subproblem (#563).

ADR-0109's layer ships no inner solver, deliberately -- shipping one would state a
preference the optimizer-agnostic contract exists to avoid. A *consumer* has to choose one,
and for multiple shooting the choice is measured rather than assumed. Over 30 data seeds x 2
starts on the layer's offline shooting problem (120 runs), a trust-region least-squares
solver converged **60/60** while a quasi-Newton one converged 36/60 and stalled out on the
rest. The reason is structural: the KKT stop needs the scaled defect and the first-order
optimality below tolerance in *one* iterate, and a method built from gradient differences
handles an augmented Lagrangian whose penalty term carries a large ``rho`` far less well
than one that sees ``rho J_c^T J_c`` explicitly. So the MVP steps from the Gauss-Newton
form, and a consumer should not treat the inner optimizer as free.

Reusing the fit type rather than reimplementing it
--------------------------------------------------
The Gauss-Newton form is ``(gradient, PSD hessian)`` -- exactly what ``job_type = gntr``
already consumes, and :class:`~pybnf.algorithms.optimizers.gntr._GNTRRunner` is already a
headless, backend-free step machine over that pair: ridge-regularise ``H``, eigen-factor it
into the pseudo-Jacobian that reproduces the Coleman-Li-scaled Newton step, and run ``trf``'s
bound-constrained trust-region-reflective accept/reject state machine unchanged. So this
module is a *driver*, not a method: it feeds that runner the augmented Lagrangian instead of
the fit's own objective, and nothing about the step math is new or separately tuned.

Driving it synchronously is the one difference from ``gntr``'s own use, and it is forced by
the interface above rather than chosen: ADR-0109's contract is
``solve(subproblem, u0, tolerance) -> InnerOutcome``, a blocking call, because an inner
solver "never calls back into the outer loop". Since a segment simulation is not a
:class:`~pybnf.pset.PSet` evaluation either (:mod:`pybnf.shooting.backend`), nothing is lost:
the propose/score loop was never available to this path.

Two tolerances, and why the outer loop's is only a floor away from being obeyed
-------------------------------------------------------------------------------
``tolerance`` is ``omega_k``, the outer loop's inner-optimality target: loose at first and
tightening as the penalty rises. It becomes the runner's ``grad_tol``, so an early
subproblem is solved roughly and a late one tightly -- which is the entire economic argument
for the augmented-Lagrangian frame. It is floored (:attr:`GaussNewtonSolver.grad_tol_floor`)
because ``omega`` decays geometrically and will eventually pass below what any solver can
demonstrate on a finite-precision Hessian; past that point the runner would simply spend its
whole iteration budget every outer iteration, which is the same waste ADR-0109 finding 5.1
measured on a too-loose penalty, arrived at from the other side.
"""

import numpy as np

from ..transcription import InnerOutcome


def _step_machine():
    """``(runner class, DONE sentinel)`` from the ``gntr`` fit type, imported lazily.

    The import is deferred because the dependency runs *against* the usual direction: a fit
    type imports the libraries it needs, and here a library reaches back into a fit type for
    its step machine. Importing :mod:`pybnf.algorithms.optimizers.gntr` at module scope
    executes ``pybnf.algorithms.__init__``, which registers every fit type -- including
    ``ms``, which imports this package -- so the cycle would close at import time. Deferring
    it to the first solve breaks the cycle and keeps :mod:`pybnf.shooting` importable on its
    own, which is what lets the whole package be exercised against a closed-form backend
    with no fit type in the picture.
    """
    from ..algorithms.optimizers.gntr import _GNTRRunner
    from ..algorithms.optimizers.gradient_base import DONE
    return _GNTRRunner, DONE


class _FisherModel:
    """The duck-typed local model :class:`~pybnf.algorithms.optimizers.gntr._GNTRRunner`
    reads: a scalar gradient and a PSD curvature matrix.

    The runner consumes a :class:`~pybnf.gradient.assembly.GradientResult` by attribute
    (``gradient``, ``hessian``) and never by type, so the augmented Lagrangian's own
    ``(grad f + J_c^T(lambda + rho c), H_f + rho J_c^T J_c)`` is handed over as-is. That is
    what "optimizer-agnostic" buys in practice: the fit type needs no knowledge that a
    multiplier exists.
    """

    __slots__ = ('gradient', 'hessian')

    def __init__(self, gradient, hessian):
        self.gradient = gradient
        self.hessian = hessian


class GaussNewtonSolver:
    """An inner solver on ADR-0109's contract, stepping from the Gauss-Newton form.

    :param max_iterations: Trust-region iterations per inner solve. Bounded per *outer*
        iteration rather than per run: an approximate inner minimisation is what the
        augmented-Lagrangian method is designed around, and the outer loop's stall detector
        is what notices a solver that stops achieving anything.
    :param ridge: The relative Levenberg ridge added to the curvature before the
        pseudo-Jacobian factorisation, as in ``gntr_ridge``. It matters more here than in an
        ordinary fit: with one observed state of three, the auxiliary states of the
        unobserved two carry **no data term at all** and their data-fit curvature is exactly
        zero, so the constraint block ``rho J_c^T J_c`` is the only curvature they have.
    :param step_tol: Negligible-step tolerance, as in ``gntr_step_tol``.
    :param grad_tol_floor: Floor under the outer loop's ``omega_k`` (see the module
        docstring).
    :param stop_check: Zero-argument callable; ``True`` truncates the inner solve at
        whichever iterate it has reached. The wall-clock-budget seam (ADR-0093/0107) --
        a truncated inner solve is a normal outcome here, not a failure, and the outer loop
        keeps the iterate.
    """

    def __init__(self, max_iterations=50, ridge=1e-10, step_tol=1e-10, grad_tol_floor=1e-10,
                 stop_check=None):
        self.max_iterations = int(max_iterations)
        self.ridge = float(ridge)
        self.step_tol = float(step_tol)
        self.grad_tol_floor = float(grad_tol_floor)
        self.stop_check = stop_check
        #: Model evaluations spent across every inner solve this object has driven -- the
        #: cost accounting the #563 acceptance benchmark reports, and the quantity the
        #: prototype's paired sweep measured multiple shooting's 2-7x overhead in.
        self.n_evaluations = 0

    def __call__(self, subproblem, u0, tolerance):
        runner_class, done = _step_machine()
        lower, upper = subproblem.lower, subproblem.upper
        runner = runner_class(np.clip(np.asarray(u0, dtype=float), lower, upper),
                              lower, upper, self.max_iterations,
                              grad_tol=max(float(tolerance), self.grad_tol_floor),
                              step_tol=self.step_tol, ridge=self.ridge)
        point = runner.start()
        spent = 0
        truncated = False
        while True:
            if self.stop_check is not None and self.stop_check():
                truncated = True
                break
            model = subproblem.at(point)
            spent += 1
            self.n_evaluations += 1
            nxt = runner.got(point, *_local_model(model))
            if nxt is done:
                break
            point = np.clip(np.asarray(nxt, dtype=float), lower, upper)

        if truncated:
            return InnerOutcome(runner.point, converged=False, n_evaluations=spent,
                                message='inner solve truncated by the run\'s stop check')
        return InnerOutcome(runner.point, converged=_met_its_tolerance(runner),
                            n_evaluations=spent, message=runner.stop_reason or '')

    def __repr__(self):
        return 'GaussNewtonSolver(max_iterations=%i, evaluations=%i)' % (
            self.max_iterations, self.n_evaluations)


def _local_model(model):
    """``(score, local model)`` for one visited point.

    A point whose segments did not integrate -- or whose curvature could not be assembled --
    is handed over as a non-finite score with **no** model, which is the signal the runner
    already understands: shrink the trust region and propose a shorter step, and at the
    start point end this solve rather than stepping from a surface that is not there
    (#492/#528). Multiple shooting meets that case far more often than an ordinary fit does,
    because it deliberately visits states the model was never integrated from.
    """
    if not model.is_finite():
        return float('inf'), None
    hessian = model.hessian()
    if hessian is None:
        return float('inf'), None
    return model.value, _FisherModel(model.gradient, hessian)


def _met_its_tolerance(runner):
    """Whether an inner solve stopped because it *finished*, rather than ran out.

    :class:`~pybnf.transcription.outer.InnerOutcome`'s ``converged`` means "met the
    tolerance it was given, as opposed to running out of iterations or budget". A flat
    scaled gradient and a negligible step both qualify -- the second is a solver that cannot
    improve its point further, which is a finished solve of this subproblem however loose
    ``omega_k`` was. An exhausted iteration budget and a failed start do not.

    Nothing load-bearing rests on the answer: the outer loop **measures** the
    projected-gradient optimality itself at each iterate rather than trusting this flag
    (ADR-0109), precisely so an inner solver's own opinion cannot certify a KKT point.
    """
    if runner.failure is not None or runner.stop_reason is None:
        return False
    return 'max_iterations' not in runner.stop_reason
