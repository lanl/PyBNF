"""The optimizer-agnostic augmented-Lagrangian outer loop (#563).

The outer loop owns the multipliers and the penalty; an *inner solver* -- any callable
matching the contract below -- owns the search. Separating them is the point: the
transcription layer has to work with ``gntr`` (the #563 MVP's inner optimizer), with
``trf`` and ``lbfgs``, and offline with neither, and none of those may need to know that a
multiplier exists.

The inner-solver contract
-------------------------
::

    outcome = inner_solver(subproblem, u0, tolerance)

``subproblem`` is an :class:`~pybnf.transcription.augmented.AugmentedSubproblem` -- fixed
multipliers, a box, and ``at(u)`` giving the scalar / least-squares / Gauss-Newton forms of
the augmented Lagrangian. ``tolerance`` is the outer loop's current inner-optimality target
``omega_k``, which starts loose and tightens; a solver that ignores it is correct but
slower. The return is an :class:`InnerOutcome`: where it stopped, whether it converged, how
much it spent. The solver never calls back into the outer loop, and the outer loop never
inspects the solver.

The schedule, and why it starts tight
-------------------------------------
The update is the classical Hestenes-Powell first-order rule inside the
Conn-Gould-Toint / LANCELOT test-and-tighten frame (Nocedal & Wright, Algorithm 17.4):
solve the subproblem to ``omega_k``; if the scaled defect met its target ``eta_k``, accept
the step and update ``lambda <- lambda + rho c``, then tighten both targets; otherwise keep
``lambda``, raise ``rho``, and reset the targets from the new ``rho``.

The defaults depart from the obvious reading of the multiple-shooting literature, on
measurement. Balsa-Canto et al. argue that the method's benefit comes from *allowing*
discontinuity, which suggests starting the penalty loose. On the motivating problem the
#563 prototype measured the opposite (issue #563, finding 5.1): from one start,
``rho_0 = 0.1, gamma = 3`` reached ``-178.38`` in 124 s while ``rho_0 = 10, gamma = 5``
reached ``-200.70`` in 62 s -- better *and* at half the cost. Too loose is not merely
ineffective; it is expensive, because the inner solve on a nearly-unconstrained subproblem
never converges and burns its whole budget every outer iteration. So
:class:`PenaltySchedule` starts at ``rho_0 = 10`` and grows by ``5``.

Stalling
--------
A run has two ways of going nowhere, and the guard against them has to span **both**
branches of the schedule.

On the accepting branch: the convergence test needs the defect and the first-order
optimality below tolerance in *one* iterate, and once a run is feasible the schedule's
inner tolerance is already floored -- so an inner solver that cannot drive the augmented
Lagrangian's optimality lower re-solves a near-identical subproblem every remaining outer
iteration.

On the penalty-raising branch the failure is worse, because raising ``rho`` is only
justified if the previous inner solve *did* something. An inner solver that fails on an
ill-conditioned subproblem and returns its own start point leaves the defect exactly where
it was -- which reads as "not feasible enough", raises ``rho`` by ``gamma``, and hands the
same solver a strictly harder problem. Measured on the offline shooting problem, that death
spiral runs the penalty from ``1.25e3`` to the ``1e8`` ceiling over ~15 outer iterations
during which the point never moves at all and the augmented gradient grows to ``2e6``. So a
penalty raise does **not** reset the stall counter.

Progress is the scaled defect improving or the point moving -- deliberately not the
optimality improving, which is not comparable across a change of ``rho`` (a raise scales
the augmented gradient by ``gamma``).

Stalling is a state to *report*, not a failure: a feasible, stalled run has found whatever
it found and could not certify a KKT residual for it, which is different from failing and
different again from having converged.

Best-iterate certification
--------------------------
The loop certifies **every** outer iterate, not just the last, and reports the best
(issue #563, finding 5.3: on one prototype start the final stage held ``-147.0`` while an
earlier iterate certified at ``-196.3``). Certification is the transcription's honesty
mechanism -- reconstruct the reported parameters through the fit's *ordinary* unsegmented
path and score them there -- so it is also the only ranking key that is comparable with an
ordinary fit's. The augmented objective at an infeasible point is computed on trajectories
that do not join up and is not a fit result; :class:`OuterResult` says so explicitly when
the problem could not certify (:attr:`OuterResult.certified`).
"""

import numpy as np

from .augmented import AugmentedSubproblem, Multipliers
from .errors import TranscriptionError


class PenaltySchedule:
    """The penalty / tolerance schedule of the outer loop.

    :param initial_penalty: ``rho_0``. Default **10.0** -- tight, per the finding in the
        module docstring, not the loose start the literature's motivation suggests.
    :param growth: ``gamma``, the factor ``rho`` is multiplied by when an outer iteration
        fails its feasibility target. Default **5.0**.
    :param max_penalty: The ceiling. Reaching it without meeting the feasibility target
        stops the loop (``stop_reason = 'penalty_ceiling'``) rather than grinding on an
        ill-conditioned subproblem: a penalty that large means the constraints are
        infeasible at this transcription, which is information, not a reason to keep going.
    :param optimality_tol: The projected-gradient first-order tolerance the run must reach
        to declare convergence. Default **1e-6**, deliberately looser than ``trf`` /
        ``gntr``'s ``1e-8``: those measure the *fit's* gradient, while this measures the
        augmented Lagrangian's, whose penalty term carries a factor of ``rho``. Grinding
        that to 1e-8 needs penalty raises that make the subproblem worse conditioned than
        the answer requires -- and the answer is certified by **reconstruction**, not by a
        KKT residual, so the extra digits buy nothing a certificate does not already
        establish.
    :param feasibility_tol: The scaled-defect target for convergence. Dimensionless,
        because the constraints are scaled (:mod:`pybnf.transcription.equality`). It also
        floors the schedule's own feasibility target: a point already this feasible is
        never a reason to raise the penalty.
    :param multiplier_clamp: Componentwise bound on ``lambda``. A multiplier that runs away
        is the signature of an infeasible or badly scaled constraint; clamping keeps the
        subproblem solvable so the loop can reach its ceiling and *report* that, rather
        than failing inside an inner solve.

    ``eta`` (the feasibility target a step must meet to earn a multiplier update) and
    ``omega`` (the inner-optimality target) follow Algorithm 17.4 with ``mu = 1/rho``:
    ``omega = mu`` and ``eta = mu**0.1`` on a reset, ``eta *= mu**0.9`` and ``omega *= mu``
    on a success.
    """

    def __init__(self, initial_penalty=10.0, growth=5.0, max_penalty=1e8,
                 optimality_tol=1e-6, feasibility_tol=1e-6, multiplier_clamp=1e10):
        self.initial_penalty = float(initial_penalty)
        self.growth = float(growth)
        self.max_penalty = float(max_penalty)
        self.optimality_tol = float(optimality_tol)
        self.feasibility_tol = float(feasibility_tol)
        self.multiplier_clamp = float(multiplier_clamp)
        if not np.isfinite(self.initial_penalty) or self.initial_penalty <= 0.0:
            raise TranscriptionError('initial_penalty must be finite and positive; got %r.'
                                     % initial_penalty)
        if self.growth <= 1.0:
            raise TranscriptionError(
                'The penalty growth factor must exceed 1 -- a factor of %r never tightens the '
                'constraints.' % growth)
        if self.max_penalty < self.initial_penalty:
            raise TranscriptionError('max_penalty (%r) is below initial_penalty (%r).'
                                     % (max_penalty, initial_penalty))
        if self.optimality_tol <= 0.0 or self.feasibility_tol <= 0.0:
            raise TranscriptionError('The convergence tolerances must be positive.')

    #: Floors under the two targets, so a large penalty cannot drive either to a value no
    #: inner solver can meet and stall the loop at "not converged" forever.
    TARGET_FLOOR = 1e-14

    def reset_targets(self, penalty):
        """``(eta, omega)`` for a freshly raised penalty."""
        mu = 1.0 / penalty
        return max(mu ** 0.1, self.TARGET_FLOOR), max(mu, self.TARGET_FLOOR)

    def tighten(self, penalty, eta, omega):
        """``(eta, omega)`` after an outer iteration that met its feasibility target."""
        mu = 1.0 / penalty
        return max(eta * mu ** 0.9, self.TARGET_FLOOR), max(omega * mu, self.TARGET_FLOOR)

    def raised(self, penalty):
        """The next penalty, capped at :attr:`max_penalty`."""
        return min(penalty * self.growth, self.max_penalty)

    def __repr__(self):
        return 'PenaltySchedule(rho0=%g, gamma=%g, rho_max=%g)' % (
            self.initial_penalty, self.growth, self.max_penalty)


class InnerOutcome:
    """Where an inner solver stopped.

    :param point: The augmented vector it stopped at.
    :param converged: Whether it met the tolerance it was given (as opposed to running out
        of iterations or budget). The outer loop will not declare convergence off an inner
        solve that did not converge, but it *will* keep going -- an approximate inner
        minimisation is what the method is designed around.
    :param n_evaluations: Model evaluations spent, for the cost accounting the #563
        acceptance benchmark reports.
    :param message: Free text for the run log.
    """

    __slots__ = ('point', 'converged', 'n_evaluations', 'message')

    def __init__(self, point, converged=True, n_evaluations=0, message=''):
        self.point = np.asarray(point, dtype=float).reshape(-1)
        self.converged = bool(converged)
        self.n_evaluations = int(n_evaluations)
        self.message = str(message)

    def __repr__(self):
        return 'InnerOutcome(converged=%s, evals=%i)' % (self.converged, self.n_evaluations)


class Certificate:
    """The verdict of reconstructing a parameter vector through the fit's ordinary path.

    :param objective: The score the reconstruction produced -- comparable with any ordinary
        PyBNF fit's, which is the whole point.
    :param accepted: Whether that score is usable. A reconstruction that fails to simulate,
        or whose objective is not finite, is rejected: it is not a fit result, and ranking
        it would let a transcription report a number no single-shoot run can reproduce.
    :param certified: Whether the score came from the ordinary unsegmented path. ``False``
        marks a score taken from the augmented problem itself -- legitimate only where the
        transcription *is* the ordinary problem (the one-segment stage of a homotopy).
    :param detail: Free text for the log (a rejection reason, a defect norm, a simulation
        error).
    """

    __slots__ = ('objective', 'accepted', 'certified', 'detail')

    def __init__(self, objective, accepted=True, certified=True, detail=''):
        self.objective = float(objective)
        self.accepted = bool(accepted) and np.isfinite(self.objective)
        self.certified = bool(certified)
        self.detail = str(detail)

    @classmethod
    def accept(cls, objective, detail=''):
        """A reconstruction that reproduced a finite objective."""
        return cls(objective, accepted=True, certified=True, detail=detail)

    @classmethod
    def reject(cls, detail):
        """A reconstruction that did not."""
        return cls(np.inf, accepted=False, certified=True, detail=detail)

    @classmethod
    def uncertified(cls, objective, detail=''):
        """The augmented problem's own objective, standing in for a reconstruction that was
        not performed."""
        return cls(objective, accepted=True, certified=False, detail=detail)

    def __repr__(self):
        return 'Certificate(objective=%.6g, accepted=%s, certified=%s)' % (
            self.objective, self.accepted, self.certified)


def projected_gradient_norm(point, gradient, lower, upper):
    """First-order optimality of a bound-constrained problem at ``point``:
    ``||P_[l,u](x - g) - x||_inf``.

    The standard projected-gradient stationarity measure -- zero exactly at a KKT point of
    the box-constrained subproblem, and (unlike a raw gradient norm) correctly reading as
    zero for a coordinate pinned at a bound by a gradient pushing outward, which is what
    ``trf`` / ``gntr``'s Coleman-Li optimality test measures too.

    The outer loop **measures** this rather than trusting the inner solver's own converged
    flag. An inner solver that stopped on its iteration cap at a point that happens to be
    stationary should end the run; one that reports success against a loose internal
    tolerance should not.
    """
    point = np.asarray(point, dtype=float)
    if point.size == 0:
        return 0.0
    step = np.clip(point - np.asarray(gradient, dtype=float), lower, upper) - point
    return float(np.max(np.abs(step)))


class CertifiedIterate:
    """One outer iterate, with the certificate earned by its reported parameters."""

    __slots__ = ('stage', 'iteration', 'reported', 'point', 'certificate', 'defect_norm',
                 'objective_value', 'augmented_value', 'penalty', 'optimality')

    def __init__(self, stage, iteration, reported, point, certificate, defect_norm,
                 objective_value, augmented_value, penalty, optimality):
        self.stage = stage
        self.iteration = int(iteration)
        self.reported = np.array(reported, dtype=float, copy=True)
        self.point = np.array(point, dtype=float, copy=True)
        self.certificate = certificate
        self.defect_norm = float(defect_norm)
        self.objective_value = float(objective_value)
        self.augmented_value = float(augmented_value)
        self.penalty = float(penalty)
        self.optimality = float(optimality)

    @property
    def score(self):
        """The ranking key: the certified objective."""
        return self.certificate.objective

    def __repr__(self):
        return 'CertifiedIterate(%s #%i, certified=%.6g, defect=%.3g, opt=%.3g)' % (
            self.stage, self.iteration, self.certificate.objective, self.defect_norm,
            self.optimality)


class CertifiedBest:
    """The best certified iterate seen so far -- across outer iterations, and across
    homotopy stages (:mod:`pybnf.transcription.homotopy` shares one of these over the whole
    ladder).

    Only :attr:`Certificate.accepted` records compete. Ties keep the earlier one, so a
    later iterate has to be strictly better to displace an established result.
    """

    def __init__(self):
        self.record = None

    def offer(self, record):
        """Consider ``record``; return ``True`` if it became the best."""
        if not record.certificate.accepted:
            return False
        if self.record is None or record.score < self.record.score:
            self.record = record
            return True
        return False

    @property
    def found(self):
        return self.record is not None

    @property
    def score(self):
        return self.record.score if self.record is not None else np.inf

    def __repr__(self):
        return 'CertifiedBest(%r)' % (self.record,)


class OuterResult:
    """What one augmented-Lagrangian run produced."""

    def __init__(self, stage, iterates, best, final_point, multipliers, converged, stop_reason,
                 certified, n_inner_evaluations, n_outer_evaluations, defect_norm, optimality):
        self.stage = stage
        self.iterates = tuple(iterates)
        self.best = best
        self.final_point = np.array(final_point, dtype=float, copy=True)
        self.multipliers = multipliers
        self.converged = bool(converged)
        self.stop_reason = str(stop_reason)
        self.certified = bool(certified)
        self.n_inner_evaluations = int(n_inner_evaluations)
        self.n_outer_evaluations = int(n_outer_evaluations)
        self.defect_norm = float(defect_norm)
        #: Projected-gradient first-order optimality at :attr:`final_point`, measured under
        #: the last multipliers -- the other half of the KKT test the defect norm starts.
        self.optimality = float(optimality)

    @property
    def n_evaluations(self):
        return self.n_inner_evaluations + self.n_outer_evaluations

    @property
    def best_score(self):
        return self.best.score if self.best is not None else np.inf

    def summary(self):
        """One line for the run log."""
        best = ('%.6g' % self.best_score) if self.best is not None else 'none'
        return ('%s: %i outer iteration(s), %s, scaled defect %.3g, optimality %.3g, best '
                'certified objective %s (%i evaluations)%s'
                % (self.stage, len(self.iterates), self.stop_reason, self.defect_norm,
                   self.optimality, best, self.n_evaluations,
                   '' if self.certified else ' [UNCERTIFIED]'))

    def __repr__(self):
        return 'OuterResult(%s, %s, best=%.6g)' % (self.stage, self.stop_reason, self.best_score)


class AugmentedLagrangian:
    """The outer loop.

    :param problem: The :class:`~pybnf.transcription.augmented.TranscriptionProblem` --
        one homotopy stage.
    :param inner_solver: The inner-solver callable (see the module docstring).
    :param schedule: The :class:`PenaltySchedule`; the defaults carry finding 5.1.
    :param max_outer: Cap on outer iterations.
    :param max_stall: Consecutive outer iterations that neither improve the scaled defect
        nor move the point before the run stops with ``stop_reason = 'stalled'``. See
        `Stalling` in the module docstring for what this guards and why it spans both
        branches of the schedule.
    :param shared_best: An optional :class:`CertifiedBest` that every iterate is *also*
        offered to, so a homotopy tracks one best across all its stages. Each
        :meth:`run` keeps its own local best regardless, which is what
        :attr:`OuterResult.best` reports.
    :param stop_check: Optional zero-argument callable returning ``True`` when the run must
        stop -- the seam a wall-clock budget (``wall_time_fit``, ADR-0093) plugs into
        without this module importing it.
    :param on_iterate: Optional callback given each :class:`CertifiedIterate` as it is
        produced, for progress logging.
    """

    #: Relative improvement in the scaled defect that counts as progress for the stall
    #: detector. Deliberately generous: the point is to catch a loop that has stopped
    #: moving, not to police the rate at which it moves.
    STALL_FACTOR = 0.9

    #: Relative step below which an outer iterate counts as not having moved.
    STALL_STEP = 1e-10

    def __init__(self, problem, inner_solver, schedule=None, max_outer=25, shared_best=None,
                 stop_check=None, on_iterate=None, max_stall=3):
        self.problem = problem
        self.inner_solver = inner_solver
        self.schedule = schedule or PenaltySchedule()
        self.max_outer = int(max_outer)
        if self.max_outer < 1:
            raise TranscriptionError('max_outer must be at least 1; got %r.' % max_outer)
        self.max_stall = int(max_stall)
        if self.max_stall < 1:
            raise TranscriptionError('max_stall must be at least 1; got %r.' % max_stall)
        self.shared_best = shared_best
        self.stop_check = stop_check
        self.on_iterate = on_iterate

    def run(self, u0, multipliers=None):
        """Solve from augmented start point ``u0``; return an :class:`OuterResult`."""
        layout = self.problem.layout
        best = CertifiedBest()
        u = np.asarray(u0, dtype=float).reshape(-1)
        if len(u) != layout.size:
            raise TranscriptionError(
                'The start point is %i wide but this transcription\'s layout is %i wide.'
                % (len(u), layout.size))
        u = np.clip(u, layout.lower, layout.upper)

        # Prefer the declared count (EqualitySystem.n_constraints, static within a stage)
        # over linearising: for a simulator-backed consumer, equality_at is a pass of
        # segment simulations, and asking it how many constraints exist would spend one
        # before the loop has taken a step.
        declared = getattr(self.problem, 'n_constraints', None)
        m = int(declared) if declared is not None else self.problem.equality_at(u).n_constraints
        if multipliers is None:
            multipliers = Multipliers.zeros(m, self.schedule.initial_penalty)
        elif len(multipliers.values) != m:
            raise TranscriptionError('Supplied %i multipliers for %i constraints.'
                                     % (len(multipliers.values), m))

        eta, omega = self.schedule.reset_targets(multipliers.penalty)
        iterates = []
        inner_evals = 0
        outer_evals = 0
        converged = False
        stop_reason = 'max_outer'
        defect_norm = np.inf
        optimality = np.inf
        certified = True
        prev_defect = np.inf
        stalled_for = 0

        for iteration in range(1, self.max_outer + 1):
            if self.stop_check is not None and self.stop_check():
                stop_reason = 'stopped'
                break

            subproblem = AugmentedSubproblem(self.problem, multipliers)
            outcome = self.inner_solver(subproblem, u, omega)
            if not isinstance(outcome, InnerOutcome):
                raise TranscriptionError(
                    'An inner solver must return an InnerOutcome; got %r.'
                    % type(outcome).__name__)
            if len(outcome.point) != layout.size:
                raise TranscriptionError(
                    'The inner solver returned a length-%i point for a %i-wide layout.'
                    % (len(outcome.point), layout.size))
            inner_evals += outcome.n_evaluations

            if not np.all(np.isfinite(outcome.point)):
                stop_reason = 'inner_failed'
                break
            previous_point = u
            u = np.clip(outcome.point, layout.lower, layout.upper)
            step = float(np.max(np.abs(u - previous_point))) / (
                1.0 + float(np.max(np.abs(previous_point))))

            model = subproblem.at(u)
            outer_evals += 1
            if not model.is_finite():
                stop_reason = 'inner_failed'
                break
            defect_norm = model.defect_norm
            optimality = projected_gradient_norm(u, model.gradient, layout.lower,
                                                 layout.upper)

            record = self._certify(iteration, u, model, multipliers.penalty, optimality)
            certified = certified and record.certificate.certified
            iterates.append(record)
            best.offer(record)
            if self.shared_best is not None:
                self.shared_best.offer(record)
            if self.on_iterate is not None:
                self.on_iterate(record)

            if m == 0:
                # No constraints: the transcription already *is* the ordinary problem, and
                # every further outer iteration would re-solve the identical subproblem.
                # This is the last stage of a segment homotopy, not a degenerate case.
                converged = optimality <= self.schedule.optimality_tol or outcome.converged
                stop_reason = 'unconstrained'
                break

            # Progress is the defect improving or the point moving. It is deliberately NOT
            # the optimality improving: a penalty raise scales the augmented gradient by
            # gamma, so optimality is not comparable across one.
            if defect_norm < self.STALL_FACTOR * prev_defect or step > self.STALL_STEP:
                stalled_for = 0
            else:
                stalled_for += 1
                if stalled_for >= self.max_stall:
                    stop_reason = 'stalled'
                    break
            # The running *best*, not the previous value: progress means beating what the
            # run has already achieved. The two coincide for a deterministic inner solver
            # (an unmoved point gives an unchanged defect); they differ for one that can
            # return a worse point than it found, which is why this is the conservative form.
            prev_defect = min(prev_defect, defect_norm)

            # A point already feasible to the run's own tolerance is accepted whatever the
            # schedule's current target says. Without this floor the geometrically
            # tightening `eta` eventually drops below any achievable defect and the loop
            # raises `rho` on a point that is feasible by every standard that matters --
            # buying nothing and making the next subproblem worse conditioned.
            if defect_norm <= max(eta, self.schedule.feasibility_tol):
                # The KKT test, measured at this point: the defects are feasible and the
                # augmented Lagrangian is stationary in the box, so `lambda + rho c` is a
                # multiplier certifying the constrained solution.
                if (defect_norm <= self.schedule.feasibility_tol
                        and optimality <= self.schedule.optimality_tol):
                    converged = True
                    stop_reason = 'converged'
                    break
                multipliers = multipliers.updated(model.defect,
                                                  clamp=self.schedule.multiplier_clamp)
                eta, omega = self.schedule.tighten(multipliers.penalty, eta, omega)
            else:
                if multipliers.penalty >= self.schedule.max_penalty:
                    stop_reason = 'penalty_ceiling'
                    break
                multipliers = multipliers.with_penalty(
                    self.schedule.raised(multipliers.penalty))
                eta, omega = self.schedule.reset_targets(multipliers.penalty)

        return OuterResult(self.problem.name, iterates, best.record, u, multipliers,
                           converged, stop_reason, certified, inner_evals, outer_evals,
                           defect_norm, optimality)

    def _certify(self, iteration, u, model, penalty, optimality):
        """Reconstruct this iterate's reported parameters and wrap it as a
        :class:`CertifiedIterate`.

        A problem that returns no certificate gets an *uncertified* one carrying the
        augmented problem's own objective, and the flag travels all the way out to
        :attr:`OuterResult.certified` -- so a run whose score never went through the
        ordinary path says so rather than looking like one that did.
        """
        reported = self.problem.layout.reported_of(u)
        certificate = self.problem.certify(reported)
        if certificate is None:
            certificate = Certificate.uncertified(
                model.objective_value, detail='the transcription supplied no reconstruction')
        elif not isinstance(certificate, Certificate):
            raise TranscriptionError(
                'TranscriptionProblem.certify must return a Certificate or None; got %r.'
                % type(certificate).__name__)
        return CertifiedIterate(self.problem.name, iteration, reported, u, certificate,
                                model.defect_norm, model.objective_value, model.value, penalty,
                                optimality)
