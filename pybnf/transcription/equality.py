"""The equality residual / Jacobian interface (#563).

What a constrained transcription adds to a fit is a vector of equality constraints
``c(u_aug) = 0`` on the augmented variables. For multiple shooting they are the continuity
defects ``c_j = Phi_j(z_j, theta) - z_{j+1}``; for direct collocation they would be the
per-node collocation equations; for latent-state estimation, the state-transition
residuals. This module fixes the interface all of them present to the outer loop, and
nothing else -- it contains no dynamics, no simulator, and no optimizer.

Three decisions are worth naming.

**The Jacobian is block-sparse, not dense.** A transcription's constraint Jacobian is
mostly zeros with a strong, known structure: a continuity row for segment ``j`` reads
``theta``, ``z_j``, and ``z_{j+1}`` and nothing else, so ``dc/du`` is (constraint group x
variable block) blocks against a background of exact structural zeros. :class:`BlockJacobian`
stores exactly those blocks -- one dense ``(rows x cols)`` array per non-zero region, both
ranges contiguous because :class:`~pybnf.transcription.layout.AugmentedLayout` lays every
block out contiguously -- and implements :meth:`~BlockJacobian.matvec`,
:meth:`~BlockJacobian.rmatvec`, and :meth:`~BlockJacobian.gram` block-wise. Dense assembly
(:meth:`~BlockJacobian.to_dense`) exists because today's inner optimizers consume dense
linear algebra (``gntr`` eigen-decomposes its Hessian), but the structure is preserved
rather than discarded on the way in, which is what leaves room for the **condensing seam**:
eliminating the ``z`` block-by-block to recover a dense system of the fit's own dimension
``k`` instead of ``k + sum_j dim(z_j)``. That is the standard multiple-shooting
condensation and the reason the representation is block-structured now, before there is a
model big enough to need it. Nothing in this module assumes condensing exists; nothing in
it prevents adding it.

**Blocks accumulate.** Two blocks covering the same region *add*, exactly as
:func:`pybnf.gradient.routing.route_experiment` folds two chain-rule paths reaching one
sensitivity column. Every operation here is linear in the block list -- ``to_dense``,
``matvec``, ``rmatvec``, ``gram`` -- so additive is the only semantics that makes them
agree with each other, and a consumer that reaches one region by two paths writes two
blocks rather than pre-summing them.

**The constraints are scaled, and the outer loop sees only the scaled ones.** A continuity
defect is a difference of *states*, so its units are the state's, and a model whose species
span six orders of magnitude would otherwise hand the penalty term a condition number for
free. Each constraint carries a strictly positive scale (``s_i``, typically the state's own
magnitude), and :class:`EqualityModel` exposes ``c_i / s_i``. One penalty parameter then
means the same thing for every constraint, the feasibility tolerance is dimensionless, and
the defect report the issue asks for is comparable across states. Scaling a constraint is
an exact reparameterisation -- ``lambda`` absorbs ``s`` -- so nothing downstream has to know
it happened. It matters most in the corner the #563 thread flags as the hard part of the
motivating problem: with one observed state of three, the *unobserved* segment-start states
are determined by continuity alone, so the conditioning of the constraint block is the
conditioning of the whole inner problem.
"""

from abc import ABC, abstractmethod

import numpy as np

from .errors import TranscriptionError


class JacobianBlock:
    """One dense, non-zero region of a constraint Jacobian.

    :param rows: Contiguous constraint rows this region covers (a ``slice``).
    :param cols: Contiguous augmented-variable columns it covers (a ``slice``).
    :param values: The ``(len(rows), len(cols))`` derivatives.

    For multiple shooting a segment contributes three blocks per knot: ``dPhi_j/dtheta``
    over the reported columns (from the ``PARAM`` sensitivity route), ``dPhi_j/dz_j`` over
    that segment's own state block (the ``IC`` route -- which the #563 prototype found is
    an ``IC`` contribution with chain-rule factor 1, so it needs no new residual math), and
    the constant ``-I`` over the next knot's state block.
    """

    __slots__ = ('rows', 'cols', 'values')

    def __init__(self, rows, cols, values):
        self.rows = _contiguous(rows, 'rows')
        self.cols = _contiguous(cols, 'cols')
        self.values = np.atleast_2d(np.asarray(values, dtype=float))
        want = (self.rows.stop - self.rows.start, self.cols.stop - self.cols.start)
        if self.values.shape != want:
            raise TranscriptionError(
                'A Jacobian block covering rows %i:%i x cols %i:%i needs a %s array; got %s.'
                % (self.rows.start, self.rows.stop, self.cols.start, self.cols.stop,
                   want, self.values.shape))

    @property
    def n_rows(self):
        return self.rows.stop - self.rows.start

    @property
    def n_cols(self):
        return self.cols.stop - self.cols.start

    def __repr__(self):
        return 'JacobianBlock(rows=%i:%i, cols=%i:%i)' % (
            self.rows.start, self.rows.stop, self.cols.start, self.cols.stop)


class BlockJacobian:
    """A constraint Jacobian ``dc/du_aug`` held as its non-zero blocks.

    :param shape: ``(n_constraints, layout.size)``.
    :param blocks: The :class:`JacobianBlock`\\ s. Regions that overlap **add**.

    Everything outside a block is a structural zero -- not a small number, a zero -- which
    is what makes the block-wise operations exact rather than approximate.
    """

    def __init__(self, shape, blocks):
        self.shape = (int(shape[0]), int(shape[1]))
        if self.shape[0] < 0 or self.shape[1] <= 0:
            raise TranscriptionError('A block Jacobian needs a non-negative row count and a '
                                     'positive column count; got %s.' % (self.shape,))
        self.blocks = tuple(blocks)
        m, n = self.shape
        for block in self.blocks:
            if not isinstance(block, JacobianBlock):
                raise TranscriptionError(
                    'A block Jacobian takes JacobianBlock objects; got %r.'
                    % type(block).__name__)
            if block.rows.stop > m or block.cols.stop > n:
                raise TranscriptionError(
                    'Jacobian block %r falls outside the declared %s shape.' % (block, self.shape))

    @property
    def nnz(self):
        """Stored entries -- the structural zeros are not among them."""
        return sum(b.n_rows * b.n_cols for b in self.blocks)

    @property
    def density(self):
        """Stored fraction of the dense shape. The number a condensing seam would act on."""
        total = self.shape[0] * self.shape[1]
        return (self.nnz / total) if total else 0.0

    def to_dense(self):
        """The dense ``(m, n)`` array. For the inner optimizers that consume dense linear
        algebra; the block structure is kept on the way in so a future condensation does
        not have to rediscover it."""
        out = np.zeros(self.shape, dtype=float)
        for block in self.blocks:
            out[block.rows, block.cols] += block.values
        return out

    def matvec(self, v):
        """``J @ v``, touching only the stored blocks."""
        v = np.asarray(v, dtype=float).reshape(-1)
        if len(v) != self.shape[1]:
            raise TranscriptionError('matvec needs a length-%i vector; got %i.'
                                     % (self.shape[1], len(v)))
        out = np.zeros(self.shape[0], dtype=float)
        for block in self.blocks:
            out[block.rows] += block.values @ v[block.cols]
        return out

    def rmatvec(self, y):
        """``J.T @ y`` -- how the constraint gradient ``J^T (lambda + rho c)`` is formed."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != self.shape[0]:
            raise TranscriptionError('rmatvec needs a length-%i vector; got %i.'
                                     % (self.shape[0], len(y)))
        out = np.zeros(self.shape[1], dtype=float)
        for block in self.blocks:
            out[block.cols] += block.values.T @ y[block.rows]
        return out

    def gram(self):
        """``J.T @ J`` -- the Gauss-Newton curvature of the penalty term ``rho/2 ||c||^2``.

        Accumulated over pairs of blocks whose row ranges intersect, since only those
        contribute: two blocks on disjoint rows are orthogonal by construction. For the
        block-diagonal-plus-arrowhead structure a transcription actually produces, almost
        every pair is disjoint.
        """
        n = self.shape[1]
        out = np.zeros((n, n), dtype=float)
        for a in self.blocks:
            for b in self.blocks:
                lo = max(a.rows.start, b.rows.start)
                hi = min(a.rows.stop, b.rows.stop)
                if lo >= hi:
                    continue
                va = a.values[lo - a.rows.start:hi - a.rows.start, :]
                vb = b.values[lo - b.rows.start:hi - b.rows.start, :]
                out[a.cols, b.cols] += va.T @ vb
        return out

    def scaled(self, row_scales):
        """This Jacobian with row ``i`` divided by ``row_scales[i]`` -- the Jacobian of the
        scaled constraints ``c_i / s_i``."""
        scales = _check_scales(row_scales, self.shape[0])
        blocks = [JacobianBlock(b.rows, b.cols,
                                b.values / scales[b.rows][:, None]) for b in self.blocks]
        return BlockJacobian(self.shape, blocks)

    def __repr__(self):
        return 'BlockJacobian(shape=%s, blocks=%i, density=%.3g)' % (
            self.shape, len(self.blocks), self.density)


class EqualityModel:
    """The equality constraints linearised at one augmented point.

    :param residual: The raw defect ``c(u)``, in the constraints' own units.
    :param jacobian: ``dc/du_aug`` as a :class:`BlockJacobian`.
    :param scales: Strictly positive per-constraint scales; ``None`` means all ones.
    :param names: Per-constraint labels for the defect report (for multiple shooting,
        ``'<knot block>::<state>'``).

    The outer loop reads :attr:`scaled_residual` and :attr:`scaled_jacobian`, never the raw
    pair -- see the module docstring. The raw pair is kept because it is what the consumer
    reports in the model's own units when a user asks how far from continuous the fit was.
    """

    def __init__(self, residual, jacobian, scales=None, names=None):
        self.residual = np.asarray(residual, dtype=float).reshape(-1)
        m = len(self.residual)
        if not isinstance(jacobian, BlockJacobian):
            raise TranscriptionError('An EqualityModel takes a BlockJacobian; got %r.'
                                     % type(jacobian).__name__)
        if jacobian.shape[0] != m:
            raise TranscriptionError(
                'The constraint Jacobian has %i rows but the residual has %i entries.'
                % (jacobian.shape[0], m))
        self.jacobian = jacobian
        self.scales = np.ones(m, dtype=float) if scales is None else _check_scales(scales, m)
        if names is None:
            self.names = tuple('c[%i]' % i for i in range(m))
        else:
            self.names = tuple(str(x) for x in names)
            if len(self.names) != m:
                raise TranscriptionError(
                    'The constraint model has %i residual entries but %i names.'
                    % (m, len(self.names)))
        self._scaled_residual = None
        self._scaled_jacobian = None

    @property
    def n_constraints(self):
        return len(self.residual)

    @property
    def scaled_residual(self):
        """``c_i / s_i`` -- the dimensionless defect the outer loop and its feasibility test
        are defined on."""
        if self._scaled_residual is None:
            self._scaled_residual = self.residual / self.scales
        return self._scaled_residual

    @property
    def scaled_jacobian(self):
        """The Jacobian of :attr:`scaled_residual`."""
        if self._scaled_jacobian is None:
            self._scaled_jacobian = self.jacobian.scaled(self.scales)
        return self._scaled_jacobian

    @property
    def defect_norm(self):
        """``max_i |c_i / s_i|`` -- the scaled infinity norm. The infinity norm rather than
        the 2-norm so the feasibility tolerance is a statement about the *worst* constraint
        and does not loosen as segments are added."""
        if self.n_constraints == 0:
            return 0.0
        return float(np.max(np.abs(self.scaled_residual)))

    @property
    def defect_rms(self):
        """Root-mean-square scaled defect -- the aggregate companion to :attr:`defect_norm`."""
        if self.n_constraints == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.scaled_residual ** 2)))

    def worst(self, count=5):
        """The ``count`` largest scaled defects as ``(name, value)``, worst first -- the
        "report scaled continuity defects" the issue asks a converged run to print."""
        if self.n_constraints == 0:
            return []
        order = np.argsort(-np.abs(self.scaled_residual))[:count]
        return [(self.names[i], float(self.scaled_residual[i])) for i in order]

    def is_finite(self):
        """Whether this linearisation can be stepped from at all."""
        return bool(np.all(np.isfinite(self.residual))
                    and np.all(np.isfinite(self.jacobian.to_dense())))

    def __repr__(self):
        return 'EqualityModel(m=%i, defect=%.3g)' % (self.n_constraints, self.defect_norm)


class EqualitySystem(ABC):
    """What a transcription implements to declare its equality constraints.

    Two obligations: say how many constraints there are and what they are called (once, at
    construction -- the layout and the constraint list are both static within a homotopy
    stage), and linearise them at a point. Everything else -- the penalty, the multipliers,
    the inner optimizer, the certification -- is somebody else's.

    :meth:`equality_at` is the only place a consumer touches a simulator, which is exactly
    why this layer is testable without one: an offline system implements it in closed form.
    The method is named to match
    :meth:`~pybnf.transcription.augmented.TranscriptionProblem.equality_at`, so a class that
    implements this ABC *is* the constraint half of a transcription problem and the two
    compose by inheritance rather than by an adapter.
    """

    @property
    @abstractmethod
    def layout(self):
        """The :class:`~pybnf.transcription.layout.AugmentedLayout` these constraints are
        written against."""

    @property
    @abstractmethod
    def constraint_names(self):
        """One label per constraint, in residual order."""

    @property
    def n_constraints(self):
        return len(self.constraint_names)

    @abstractmethod
    def equality_at(self, u):
        """The :class:`EqualityModel` at augmented point ``u``."""

    def empty_model(self):
        """The zero-constraint model -- what a system with nothing to enforce returns.

        The final stage of a segment-count homotopy has exactly this shape: coarsened to one
        segment there are no knots, hence no continuity constraints, and the augmented
        problem *is* the ordinary single-shoot problem. That stage is not a special case to
        be branched around; it is this model, and the outer loop reduces to one inner solve
        on it.
        """
        return EqualityModel(np.zeros(0), BlockJacobian((0, self.layout.size), ()),
                             names=())


def _contiguous(value, what):
    """Normalise a slice/range to a plain contiguous ``slice`` with non-negative bounds."""
    if isinstance(value, range):
        value = slice(value.start, value.stop, value.step)
    if not isinstance(value, slice):
        raise TranscriptionError('A Jacobian block needs a slice for its %s; got %r.'
                                 % (what, type(value).__name__))
    if value.step not in (None, 1):
        raise TranscriptionError(
            'A Jacobian block covers a contiguous range of %s (step 1); got step %r. A '
            'scattered region is written as several blocks.' % (what, value.step))
    start = 0 if value.start is None else int(value.start)
    if value.stop is None:
        raise TranscriptionError('A Jacobian block needs an explicit %s stop.' % what)
    stop = int(value.stop)
    if start < 0 or stop < start:
        raise TranscriptionError('A Jacobian block has an empty or negative %s range %i:%i.'
                                 % (what, start, stop))
    return slice(start, stop)


def _check_scales(scales, m):
    scales = np.asarray(scales, dtype=float).reshape(-1)
    if len(scales) != m:
        raise TranscriptionError('Expected %i constraint scales; got %i.' % (m, len(scales)))
    if m and not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise TranscriptionError(
            'Constraint scales must be finite and strictly positive -- they divide the defect, '
            'and a zero or negative scale would silently change which constraints the penalty '
            'weights.')
    return scales
