"""The augmented Lagrangian at a point: the smooth problem an inner optimizer walks (#563).

The outer loop (:mod:`pybnf.transcription.outer`) fixes the multipliers ``lambda`` and the
penalty ``rho``; what is left is an ordinary bound-constrained smooth minimisation,

    ``L_A(u) = f(u) + lambda^T c(u) + rho/2 ||c(u)||^2``

over the augmented vector, with ``c`` the *scaled* equality defects
(:mod:`pybnf.transcription.equality`). This module builds that problem's local model, in
every form PyBNF's existing optimizers consume, so that "optimizer-agnostic" is a property
of the interface rather than an aspiration.

The three forms, and why the least-squares one is exact
-------------------------------------------------------
* **scalar** -- ``value`` and ``gradient``, for a quasi-Newton method (``lbfgs``);
* **least-squares** -- a stacked residual and Jacobian, for a trust-region least-squares
  method (``trf``);
* **Gauss-Newton** -- ``gradient`` and a PSD ``hessian``, for the EFIM trust-region method
  (``gntr``).

The least-squares form is the one worth spelling out. Completing the square on the two
constraint terms,

    ``lambda^T c + rho/2 ||c||^2  ==  rho/2 ||c + lambda/rho||^2  -  ||lambda||^2/(2 rho)``

so with an objective that carries an exact least-squares residual (``0.5||r_f||^2 == f``,
which is what :attr:`~pybnf.gradient.assembly.GradientResult.least_squares_exact` certifies)
the whole augmented Lagrangian is a sum of squares up to a **constant**:

    ``r_aug = [ r_f ; sqrt(rho) (c + lambda/rho) ]``,
    ``J_aug = [ J_f ; sqrt(rho) J_c ]``,
    ``0.5 ||r_aug||^2 == L_A + ||lambda||^2/(2 rho)``.

The offset does not depend on ``u``, so it changes no step, no gradient, and no accept
test -- but it is reported (:attr:`AugmentedModel.residual_offset`) rather than left for a
caller to rediscover when ``0.5||r||^2`` does not equal the value it was told. The shifted
form is also the numerically better one: it keeps the multiplier inside the square instead
of adding a large linear term to a large quadratic one.

The Gauss-Newton curvature is ``H_f + rho J_c^T J_c``. It drops the exact augmented
Lagrangian's ``sum_i (lambda_i + rho c_i) grad^2 c_i`` term, which needs constraint second
derivatives nobody assembles -- the same Gauss-Newton omission ``trf`` and ``gntr`` already
make on the data term, and the standard one for augmented-Lagrangian SQP.

The invariant the noise scale depends on
----------------------------------------
**The constraint terms never enter the likelihood.** ``f`` is the fit's own objective and
nothing this module does is folded back into it: :attr:`AugmentedModel.objective_value` is
``f`` alone, and it is the only quantity a certification or a reported score is allowed to
read. This is not tidiness. 13 of the 23 slugs in the motivating benchmark corpus estimate
at least one noise scale, and an estimated ``sigma`` is fitted *to the residuals it is given*
-- so a ``sigma`` that could see continuity defects would absorb constraint violation as
measurement noise, and the reported objective would stop being comparable to a
single-shoot one. Keeping the penalty strictly outside ``f`` is what makes the
certification step meaningful; with ``noise_profiling = 1`` (ADR-0108) the profiled scale is
defined by the data residuals alone and the separation is structural.
"""

from abc import ABC, abstractmethod

import numpy as np

from .errors import TranscriptionError
from .equality import EqualityModel


class Multipliers:
    """The outer loop's state: one multiplier per (scaled) constraint, and the penalty.

    Immutable -- an update produces a new instance -- so an outer iterate can record the
    multipliers it was solved under without defensive copying.
    """

    __slots__ = ('values', 'penalty')

    def __init__(self, values, penalty):
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.penalty = float(penalty)
        if not np.isfinite(self.penalty) or self.penalty <= 0.0:
            raise TranscriptionError(
                'The augmented-Lagrangian penalty must be finite and strictly positive; got %r.'
                % penalty)
        if self.values.size and not np.all(np.isfinite(self.values)):
            raise TranscriptionError('The augmented-Lagrangian multipliers must be finite.')

    @classmethod
    def zeros(cls, n_constraints, penalty):
        """The start of an outer loop: no multiplier information yet, so the first inner
        solve is a plain quadratic-penalty solve."""
        return cls(np.zeros(int(n_constraints), dtype=float), penalty)

    def updated(self, defect, penalty=None, clamp=None):
        """The first-order multiplier update ``lambda <- lambda + rho c``.

        ``clamp`` bounds the result componentwise; a multiplier that runs away is the
        classic symptom of an infeasible or badly scaled constraint, and letting it do so
        turns the inner problem into an unrecoverable one rather than a slow one.
        """
        defect = np.asarray(defect, dtype=float).reshape(-1)
        if len(defect) != len(self.values):
            raise TranscriptionError('A multiplier update needs %i defects; got %i.'
                                     % (len(self.values), len(defect)))
        values = self.values + self.penalty * defect
        if clamp is not None:
            values = np.clip(values, -abs(clamp), abs(clamp))
        return Multipliers(values, self.penalty if penalty is None else penalty)

    def with_penalty(self, penalty):
        """The same multipliers under a raised penalty."""
        return Multipliers(self.values, penalty)

    @property
    def norm(self):
        return float(np.max(np.abs(self.values))) if self.values.size else 0.0

    def __repr__(self):
        return 'Multipliers(m=%i, rho=%.4g, |lambda|_inf=%.4g)' % (
            len(self.values), self.penalty, self.norm)


class ObjectiveModel:
    """The fit's own objective, linearised at one augmented point.

    This is deliberately the same shape as
    :class:`~pybnf.gradient.assembly.GradientResult`, because for #563's first consumer it
    *is* one: the #563 prototype established that a segment-start state enters the data fit
    as an ``IC`` route with chain-rule factor 1, so
    :func:`~pybnf.gradient.assembly.assemble_gradient_and_fisher_hessian` builds a segment's
    gradient column and Fisher block for an auxiliary variable with no new residual math.
    Use :meth:`from_gradient_result` for that path.

    :param value: ``f(u)`` -- the fit's objective. The certified quantity; see the module
        docstring on why the penalty never enters it.
    :param gradient: ``df/du_aug``, length ``layout.size``.
    :param residual: The objective's own least-squares residual, or ``None`` if it has
        none.
    :param jacobian: The matching ``(n_obs, layout.size)`` residual Jacobian.
    :param hessian: A PSD curvature model (the EFIM), or ``None``.
    :param least_squares_exact: Whether ``0.5||residual||^2 == value``. Only then does the
        stacked least-squares form model the whole augmented Lagrangian.
    """

    __slots__ = ('value', 'gradient', 'residual', 'jacobian', 'hessian', 'least_squares_exact')

    def __init__(self, value, gradient, residual=None, jacobian=None, hessian=None,
                 least_squares_exact=False):
        self.value = float(value)
        self.gradient = np.asarray(gradient, dtype=float).reshape(-1)
        n = len(self.gradient)
        self.residual = None if residual is None else np.asarray(residual, dtype=float).reshape(-1)
        self.jacobian = None if jacobian is None else np.atleast_2d(np.asarray(jacobian, dtype=float))
        self.hessian = None if hessian is None else np.asarray(hessian, dtype=float)
        self.least_squares_exact = bool(least_squares_exact)
        if (self.residual is None) != (self.jacobian is None):
            raise TranscriptionError(
                'An objective model carries a residual and its Jacobian together, or neither.')
        if self.jacobian is not None and self.jacobian.shape != (len(self.residual), n):
            raise TranscriptionError(
                'The objective residual Jacobian is %s; expected (%i, %i).'
                % (self.jacobian.shape, len(self.residual), n))
        if self.hessian is not None and self.hessian.shape != (n, n):
            raise TranscriptionError('The objective Hessian is %s; expected (%i, %i).'
                                     % (self.hessian.shape, n, n))
        if self.least_squares_exact and self.residual is None:
            raise TranscriptionError(
                'An objective model flagged least_squares_exact must carry the residual that '
                'makes it exact.')

    @classmethod
    def from_gradient_result(cls, value, grad, layout=None):
        """Adapt an assembled :class:`~pybnf.gradient.assembly.GradientResult`.

        Duck-typed, so this module stays free of the gradient package (and of everything it
        pulls in) -- which is what makes the whole transcription layer importable and
        testable with no simulation backend present. ``value`` comes from the objective's
        own ``evaluate``; a ``GradientResult`` carries derivatives, not a score.

        If ``layout`` is given and the assembled columns are the *reported* parameters only,
        the gradient and Jacobian are zero-padded into augmented space. A gradient already
        assembled over the augmented free-parameter list is used as-is.
        """
        gradient = np.asarray(grad.gradient, dtype=float).reshape(-1)
        residual = getattr(grad, 'residual', None)
        jacobian = getattr(grad, 'jacobian', None)
        hessian = getattr(grad, 'hessian', None)
        if layout is not None and len(gradient) == layout.n_reported < layout.size:
            gradient = layout.embed_gradient(gradient)
            if jacobian is not None:
                jacobian = layout.embed_jacobian(jacobian)
            if hessian is not None:
                padded = np.zeros((layout.size, layout.size), dtype=float)
                padded[:layout.n_reported, :layout.n_reported] = hessian
                hessian = padded
        return cls(value, gradient, residual=residual, jacobian=jacobian, hessian=hessian,
                   least_squares_exact=bool(getattr(grad, 'least_squares_exact', False)))

    def is_finite(self):
        for array in (np.array([self.value]), self.gradient, self.residual, self.jacobian,
                      self.hessian):
            if array is not None and not np.all(np.isfinite(array)):
                return False
        return True

    def __repr__(self):
        return 'ObjectiveModel(value=%.6g, n=%i, exact=%s)' % (
            self.value, len(self.gradient), self.least_squares_exact)


class AugmentedModel:
    """The augmented Lagrangian and its derivatives at one point, in all three forms.

    Built by :meth:`AugmentedSubproblem.at`; an inner optimizer asks for one per point it
    visits and reads whichever form it steps from.
    """

    def __init__(self, objective, equality, multipliers):
        if not isinstance(equality, EqualityModel):
            raise TranscriptionError('An augmented model takes an EqualityModel; got %r.'
                                     % type(equality).__name__)
        if equality.n_constraints != len(multipliers.values):
            raise TranscriptionError(
                'There are %i constraints but %i multipliers.'
                % (equality.n_constraints, len(multipliers.values)))
        n = len(objective.gradient)
        if equality.n_constraints and equality.jacobian.shape[1] != n:
            raise TranscriptionError(
                'The constraint Jacobian has %i columns but the objective gradient has %i '
                'entries -- the objective and the constraints must be assembled against the '
                'same augmented layout.' % (equality.jacobian.shape[1], n))
        self.objective = objective
        self.equality = equality
        self.multipliers = multipliers

    # -- the quantities ---------------------------------------------------------

    @property
    def objective_value(self):
        """``f(u)`` alone -- the fit's own objective, with no constraint term in it. The
        only value a certification or a reported score may read."""
        return self.objective.value

    @property
    def defect(self):
        """The scaled equality defects ``c(u)``."""
        return self.equality.scaled_residual

    @property
    def defect_norm(self):
        return self.equality.defect_norm

    @property
    def value(self):
        """``L_A = f + lambda^T c + rho/2 ||c||^2``."""
        c = self.defect
        if c.size == 0:
            return self.objective.value
        lam = self.multipliers.values
        rho = self.multipliers.penalty
        return float(self.objective.value + lam @ c + 0.5 * rho * (c @ c))

    @property
    def gradient(self):
        """``grad f + J_c^T (lambda + rho c)``."""
        c = self.defect
        if c.size == 0:
            return self.objective.gradient.copy()
        weight = self.multipliers.values + self.multipliers.penalty * c
        return self.objective.gradient + self.equality.scaled_jacobian.rmatvec(weight)

    @property
    def least_squares_exact(self):
        """Whether the stacked residual models the whole augmented Lagrangian (up to
        :attr:`residual_offset`). Inherited from the objective: the constraint rows are
        exact squares by construction, so the only question is whether ``f`` is."""
        return self.objective.least_squares_exact

    @property
    def residual_offset(self):
        """``||lambda||^2/(2 rho)``, the constant by which ``0.5||r_aug||^2`` exceeds
        :attr:`value`. Zero on the first outer iteration and whenever there are no
        constraints."""
        lam = self.multipliers.values
        if lam.size == 0:
            return 0.0
        return float(lam @ lam / (2.0 * self.multipliers.penalty))

    def residual_model(self):
        """The stacked ``(r_aug, J_aug)`` a least-squares inner optimizer consumes, or
        ``None`` when the objective carries no residual of its own.

        ``0.5||r_aug||^2 == value + residual_offset`` when
        :attr:`least_squares_exact`; when it is ``False`` the constraint rows are still
        exact and the *whole* model is not, which is the same signal
        :class:`~pybnf.gradient.assembly.GradientResult` already carries -- a caller that
        needs an exact model must step from the scalar or Gauss-Newton form instead.
        """
        if self.objective.residual is None:
            return None
        c = self.defect
        if c.size == 0:
            return self.objective.residual.copy(), self.objective.jacobian.copy()
        rho = self.multipliers.penalty
        root = np.sqrt(rho)
        shifted = root * (c + self.multipliers.values / rho)
        rows = self.equality.scaled_jacobian.to_dense() * root
        return (np.concatenate([self.objective.residual, shifted]),
                np.vstack([self.objective.jacobian, rows]))

    def hessian(self):
        """The Gauss-Newton curvature ``H_f + rho J_c^T J_c``, or ``None`` when the
        objective supplies neither a Hessian nor a residual to build one from."""
        c = self.defect
        base = self.objective.hessian
        if base is None and self.objective.jacobian is not None:
            base = self.objective.jacobian.T @ self.objective.jacobian
        if base is None:
            return None
        if c.size == 0:
            return np.array(base, dtype=float, copy=True)
        return base + self.multipliers.penalty * self.equality.scaled_jacobian.gram()

    def is_finite(self):
        return self.objective.is_finite() and self.equality.is_finite()

    def __repr__(self):
        return 'AugmentedModel(L_A=%.6g, f=%.6g, defect=%.3g)' % (
            self.value, self.objective_value, self.defect_norm)


class TranscriptionProblem(ABC):
    """What a constrained transcription implements for the outer loop.

    One homotopy stage is one of these. It owns the augmented layout, can score and
    differentiate the fit's objective at an augmented point, can linearise its equality
    constraints there, and -- the step that keeps the whole method honest -- can
    **certify** a reported parameter vector by reconstructing it through the fit's ordinary
    unsegmented path.

    Everything the layer does is expressed through these four methods, so an offline
    implementation with closed-form dynamics exercises the same code path a simulator-backed
    one does.
    """

    @property
    @abstractmethod
    def layout(self):
        """This stage's :class:`~pybnf.transcription.layout.AugmentedLayout`."""

    @property
    def name(self):
        """A short label for the stage trace (multiple shooting uses ``'m=4'``)."""
        return type(self).__name__

    @abstractmethod
    def objective_at(self, u):
        """The :class:`ObjectiveModel` at augmented point ``u``."""

    @abstractmethod
    def equality_at(self, u):
        """The :class:`~pybnf.transcription.equality.EqualityModel` at ``u``."""

    def certify(self, reported):
        """Reconstruct ``reported`` through the fit's ordinary single-shoot path and return
        a :class:`~pybnf.transcription.outer.Certificate`.

        Returning ``None`` (the default) means this problem cannot certify, and the outer
        loop marks its whole result uncertified. That is a legitimate state for the
        *last* stage of a segment homotopy -- coarsened to one segment the transcription
        already **is** the single-shoot problem, so its objective is its own certificate --
        but for any stage with constraints an uncertified score is not a fit result: the
        objective at an infeasible augmented point is computed on trajectories that do not
        join up.
        """
        del reported
        return None

    def augmented_at(self, u, multipliers):
        """The :class:`AugmentedModel` at ``u`` -- one objective linearisation, one
        constraint linearisation, combined. The single entry point an inner optimizer's
        evaluation goes through."""
        return AugmentedModel(self.objective_at(u), self.equality_at(u), multipliers)


class AugmentedSubproblem:
    """The augmented Lagrangian at **fixed** multipliers: a plain bound-constrained smooth
    minimisation, and the object handed to an inner optimizer.

    This is the whole of the optimizer-agnostic contract. An inner solver is any callable

        ``solve(subproblem, u0, tolerance) -> InnerOutcome``

    that reads :attr:`lower` / :attr:`upper` / :attr:`size`, calls :meth:`at` at the points
    it visits, and returns where it stopped. It may step from the scalar form, the stacked
    least-squares form, or the Gauss-Newton form; the subproblem does not know or care
    which, and never calls back into the outer loop.
    """

    def __init__(self, problem, multipliers):
        self.problem = problem
        self.multipliers = multipliers
        self.layout = problem.layout

    @property
    def size(self):
        return self.layout.size

    @property
    def lower(self):
        return self.layout.lower

    @property
    def upper(self):
        return self.layout.upper

    @property
    def penalty(self):
        return self.multipliers.penalty

    def at(self, u):
        """The :class:`AugmentedModel` at ``u``. One call per visited point: the objective
        and the constraints are linearised together, which for a simulator-backed consumer
        is one pass of segment simulations rather than two."""
        return self.problem.augmented_at(u, self.multipliers)

    def value_and_gradient(self, u):
        """Convenience for a scalar inner optimizer (``lbfgs``-shaped)."""
        model = self.at(u)
        return model.value, model.gradient

    def __repr__(self):
        return 'AugmentedSubproblem(%s, %r)' % (self.problem.name, self.multipliers)
