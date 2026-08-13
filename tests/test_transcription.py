"""Offline tests for the constrained-transcription layer (#563, ADR-0109).

Every test here runs with **no simulator**: the layer contains no dynamics, so the two
consumers it is exercised against are written in closed form.

* An equality-constrained quadratic whose KKT point -- primal *and dual* -- is known
  analytically, which is what pins the multiplier update rather than just the primal path.
* A scalar linear-ODE **multiple-shooting** problem, ``y' = theta y``, whose flow
  ``Phi(z, theta, dt) = z exp(theta dt)`` and both its sensitivities
  (``dPhi/dz = exp(theta dt)`` -- the initial-condition route the #563 prototype found to be
  a chain-rule-factor-1 ``IC`` contribution -- and ``dPhi/dtheta = z dt exp(theta dt)``) are
  elementary. It has the exact structure the simulator-backed consumer will have: knots,
  segment-start states as auxiliary variables, continuity defects, a data term that reads
  the auxiliary state of its own segment, and a single-shoot reconstruction for
  certification. The claim it lets us check is the load-bearing one from the issue --
  *at convergence the constrained transcription is equivalent to the uninterrupted fit* --
  measured against the single-shoot optimum computed independently.

Two different inner solvers drive the same outer loop (a scalar quasi-Newton one and a
least-squares trust-region one, both from scipy), because "optimizer-agnostic" is a claim
about the interface and one solver cannot demonstrate it.
"""

import numpy as np
import pytest
from scipy.optimize import least_squares, minimize

from pybnf.transcription import (
    AugmentedLagrangian,
    AugmentedLayout,
    AugmentedModel,
    AugmentedSubproblem,
    BlockJacobian,
    Certificate,
    CertifiedBest,
    CertifiedIterate,
    EqualityModel,
    EqualitySystem,
    InnerOutcome,
    JacobianBlock,
    Multipliers,
    ObjectiveModel,
    PenaltySchedule,
    TranscriptionError,
    TranscriptionProblem,
    VariableBlock,
    coarsening_ladder,
    run_homotopy,
)


# ---------------------------------------------------------------------------
# Inner solvers -- the optimizer-agnostic contract, twice
# ---------------------------------------------------------------------------

def scalar_inner_solver(subproblem, u0, tolerance):
    """A quasi-Newton inner solver stepping from the **scalar** form (``lbfgs``-shaped)."""
    counter = {'n': 0}

    def fun(u):
        counter['n'] += 1
        model = subproblem.at(u)
        return model.value, model.gradient

    result = minimize(fun, np.asarray(u0, dtype=float), jac=True, method='L-BFGS-B',
                      bounds=list(zip(subproblem.lower, subproblem.upper)),
                      options={'gtol': max(tolerance, 1e-12), 'ftol': 1e-16, 'maxiter': 500})
    return InnerOutcome(result.x, converged=bool(result.success), n_evaluations=counter['n'],
                        message=str(result.message))


def least_squares_inner_solver(subproblem, u0, tolerance):
    """A trust-region inner solver stepping from the stacked **least-squares** form
    (``trf``-shaped). It never sees a multiplier."""
    counter = {'n': 0}
    cache = {}

    def model_at(u):
        key = tuple(u)
        if key not in cache:
            counter['n'] += 1
            cache.clear()
            cache[key] = subproblem.at(u).residual_model()
        return cache[key]

    tol = float(min(max(tolerance, 1e-14), 1e-4))
    result = least_squares(lambda u: model_at(u)[0], np.asarray(u0, dtype=float),
                           jac=lambda u: model_at(u)[1],
                           bounds=(subproblem.lower, subproblem.upper),
                           xtol=tol, ftol=tol, gtol=tol, max_nfev=500)
    return InnerOutcome(result.x, converged=result.status > 0, n_evaluations=counter['n'],
                        message=str(result.message))


INNER_SOLVERS = pytest.mark.parametrize(
    'inner_solver', [scalar_inner_solver, least_squares_inner_solver],
    ids=['scalar-quasi-newton', 'least-squares-trust-region'])


# ---------------------------------------------------------------------------
# Consumer 1: an equality-constrained quadratic with a known KKT point
# ---------------------------------------------------------------------------

class ConstrainedQuadratic(TranscriptionProblem):
    """``min 0.5[(x-1)^2 + (y-2)^2]`` s.t. ``x + y = 1``.

    Stationarity gives ``x = 1 - lam``, ``y = 2 - lam``; feasibility then gives
    ``lam* = 1``, ``(x*, y*) = (0, 1)``, ``f* = 1.0``. Both the primal solution and the
    multiplier are checked, because an outer loop can reach the right primal point with a
    wrong multiplier (a pure penalty method does exactly that).

    ``y`` is laid out as an *internal* auxiliary block so the layout's reported/internal
    split is exercised on the path everything else uses.
    """

    OPTIMUM = np.array([0.0, 1.0])
    MULTIPLIER = 1.0

    def __init__(self, scale=1.0):
        self._layout = AugmentedLayout(
            ['x'], [-10.0], [10.0],
            [VariableBlock('aux', ['y'], [-10.0], [10.0], [0.0])])
        self.scale = float(scale)

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return 'quadratic'

    def objective_at(self, u):
        residual = np.array([u[0] - 1.0, u[1] - 2.0])
        return ObjectiveModel(0.5 * residual @ residual, residual, residual=residual,
                              jacobian=np.eye(2), least_squares_exact=True)

    def equality_at(self, u):
        # c = x + y - 1, deliberately written in units the caller then scales away.
        residual = np.array([self.scale * (u[0] + u[1] - 1.0)])
        jac = BlockJacobian((1, 2), [JacobianBlock(slice(0, 1), slice(0, 2),
                                                   self.scale * np.ones((1, 2)))])
        return EqualityModel(residual, jac, scales=np.array([self.scale]),
                             names=('sum',))

    def certify(self, reported):
        # The feasible reconstruction: y is determined by x through the constraint.
        x = float(reported[0])
        y = 1.0 - x
        return Certificate.accept(0.5 * ((x - 1.0) ** 2 + (y - 2.0) ** 2))


# ---------------------------------------------------------------------------
# Consumer 2: multiple shooting on y' = theta y, in closed form
# ---------------------------------------------------------------------------

class ShootingProblem(TranscriptionProblem, EqualitySystem):
    """Multiple shooting for ``y' = theta y``, ``y(0) = y0``, with ``m`` equal-time segments.

    Segment ``s`` starts at ``tau_s`` from state ``zeta_s`` -- ``y0`` for ``s = 0`` (fixed,
    not a variable) and an auxiliary variable otherwise. Prediction at ``t`` in segment
    ``s``: ``zeta_s exp(theta (t - tau_s))``. Continuity:
    ``c_s = zeta_s exp(theta dt_s) - zeta_{s+1}``.

    Knot blocks are named by their **fraction of the horizon**, so a coarser stage's knots
    are a subset of a finer one's *by name* and
    :meth:`~pybnf.transcription.layout.AugmentedLayout.carry_over` carries the solved
    auxiliary states down the ladder. ``m = 1`` has no knots, no constraints, and is exactly
    the single-shoot problem.
    """

    def __init__(self, m, times, observations, theta_seed, y0=1.0, sigma=0.1, horizon=None):
        self.m = int(m)
        self.times = np.asarray(times, dtype=float)
        self.observations = np.asarray(observations, dtype=float)
        self.y0 = float(y0)
        self.sigma = float(sigma)
        self.horizon = float(horizon if horizon is not None else self.times.max())
        self.fractions = tuple((i + 1) / self.m for i in range(self.m - 1))
        self.knot_times = tuple(f * self.horizon for f in self.fractions)
        self.starts = (0.0,) + self.knot_times
        self.block_names = tuple('knot@%.4f' % f for f in self.fractions)
        # Which segment each observation belongs to (last segment owns the endpoint).
        self.segment_of = np.clip(np.searchsorted(np.array(self.starts), self.times,
                                                  side='right') - 1, 0, self.m - 1)
        blocks = [VariableBlock(name, ['y'], [1e-8], [1e8],
                                [self.y0 * np.exp(theta_seed * t)])
                  for name, t in zip(self.block_names, self.knot_times)]
        self._layout = AugmentedLayout(['theta'], [-5.0], [5.0], blocks)
        self.certifications = []

    # -- layout / identity ------------------------------------------------

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return 'm=%i' % self.m

    @property
    def constraint_names(self):
        return tuple('%s::continuity' % n for n in self.block_names)

    # -- the closed-form pieces -------------------------------------------

    def _states(self, u):
        """Segment-start states, ``y0`` first then the auxiliary variables."""
        return np.array([self.y0] + [u[self._layout.slice_of(n)][0]
                                     for n in self.block_names])

    def objective_at(self, u):
        theta = u[0]
        states = self._states(u)
        n_obs = len(self.times)
        residual = np.zeros(n_obs)
        jac = np.zeros((n_obs, self._layout.size))
        for i, (t, obs) in enumerate(zip(self.times, self.observations)):
            s = int(self.segment_of[i])
            dt = t - self.starts[s]
            growth = np.exp(theta * dt)
            pred = states[s] * growth
            residual[i] = (pred - obs) / self.sigma
            jac[i, 0] = states[s] * dt * growth / self.sigma
            if s > 0:
                jac[i, self._layout.slice_of(self.block_names[s - 1]).start] = \
                    growth / self.sigma
        return ObjectiveModel(0.5 * residual @ residual, jac.T @ residual, residual=residual,
                              jacobian=jac, least_squares_exact=True)

    def equality_at(self, u):
        if self.m == 1:
            return self.empty_model()
        theta = u[0]
        states = self._states(u)
        m = self.m - 1
        residual = np.zeros(m)
        blocks = []
        for s in range(m):
            dt = self.starts[s + 1] - self.starts[s]
            growth = np.exp(theta * dt)
            residual[s] = states[s] * growth - states[s + 1]
            rows = slice(s, s + 1)
            blocks.append(JacobianBlock(rows, slice(0, 1),
                                        np.array([[states[s] * dt * growth]])))
            if s > 0:                                   # d/d zeta_s (the IC route)
                col = self._layout.slice_of(self.block_names[s - 1])
                blocks.append(JacobianBlock(rows, col, np.array([[growth]])))
            col_next = self._layout.slice_of(self.block_names[s])
            blocks.append(JacobianBlock(rows, col_next, np.array([[-1.0]])))
        scales = np.maximum(np.abs(states[1:]), 1e-12)
        return EqualityModel(residual, BlockJacobian((m, self._layout.size), blocks),
                             scales=scales, names=self.constraint_names)

    # -- certification: the ordinary, unsegmented path ---------------------

    def single_shoot_objective(self, theta):
        pred = self.y0 * np.exp(float(theta) * self.times)
        residual = (pred - self.observations) / self.sigma
        return float(0.5 * residual @ residual)

    def certify(self, reported):
        value = self.single_shoot_objective(reported[0])
        self.certifications.append(value)
        return Certificate.accept(value, detail='single shoot')


def shooting_data(theta_true=0.7, n=41, horizon=3.0, y0=1.0, sigma=0.1, seed=4):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, horizon, n)
    clean = y0 * np.exp(theta_true * times)
    return times, clean + rng.normal(0.0, sigma, size=n)


def single_shoot_optimum(times, observations, y0=1.0, sigma=0.1, start=0.2):
    """The uninterrupted fit's optimum, computed independently of the layer."""
    result = least_squares(
        lambda p: (y0 * np.exp(p[0] * times) - observations) / sigma, [start],
        bounds=([-5.0], [5.0]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(result.x[0])


# ---------------------------------------------------------------------------
# The augmented variable layout
# ---------------------------------------------------------------------------

class TestAugmentedLayout:

    def make(self):
        return AugmentedLayout(
            ['k1', 'k2'], [-1.0, -2.0], [1.0, 2.0],
            [VariableBlock('seg1', ['A', 'B'], [0.0, 0.0], [10.0, 10.0], [1.0, 2.0]),
             VariableBlock('seg2', ['A', 'B'], [0.0, 0.0], [10.0, 10.0], [3.0, 4.0])])

    def test_reported_block_leads_and_is_contiguous(self):
        layout = self.make()
        assert layout.size == 6
        assert layout.n_reported == 2
        assert layout.n_internal == 4
        assert layout.reported_slice == slice(0, 2)
        u = layout.initial_point([0.5, -0.5])
        # The reported parameters are the *leading* slice, so a caller that forgets to
        # unpack gets the fit's own coordinates rather than a misaligned mixture.
        assert np.allclose(u[:2], [0.5, -0.5])
        assert np.allclose(layout.reported_of(u), [0.5, -0.5])

    def test_internal_names_are_qualified_and_flagged(self):
        layout = self.make()
        assert layout.names == ('k1', 'k2', 'seg1::A', 'seg1::B', 'seg2::A', 'seg2::B')
        assert [layout.is_internal(i) for i in range(6)] == [False, False, True, True,
                                                             True, True]
        assert len(set(layout.names)) == 6

    def test_pack_unpack_round_trip(self):
        layout = self.make()
        u = layout.pack([0.1, 0.2], {'seg1': [5.0, 6.0], 'seg2': [7.0, 8.0]})
        reported, internals = layout.unpack(u)
        assert np.allclose(reported, [0.1, 0.2])
        assert np.allclose(internals['seg1'], [5.0, 6.0])
        assert np.allclose(internals['seg2'], [7.0, 8.0])
        assert np.allclose(layout.internal_of(u, 'seg2'), [7.0, 8.0])

    def test_bounds_stack_in_layout_order(self):
        layout = self.make()
        assert np.allclose(layout.lower, [-1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
        assert np.allclose(layout.upper, [1.0, 2.0, 10.0, 10.0, 10.0, 10.0])

    def test_an_internal_name_may_not_collide_with_a_reported_one(self):
        # The invariant that keeps an auxiliary state out of the reported fit results.
        with pytest.raises(TranscriptionError, match='disjoint from the reported'):
            AugmentedLayout(['seg1::A'], [0.0], [1.0],
                            [VariableBlock('seg1', ['A'], [0.0], [1.0], [0.5])])

    def test_duplicate_block_names_refuse(self):
        block = VariableBlock('seg', ['A'], [0.0], [1.0], [0.5])
        with pytest.raises(TranscriptionError, match='Duplicate internal variable block'):
            AugmentedLayout(['p'], [0.0], [1.0], [block, block])

    def test_embedding_pads_a_reported_space_term(self):
        layout = self.make()
        assert np.allclose(layout.embed_gradient([1.0, 2.0]), [1.0, 2.0, 0, 0, 0, 0])
        embedded = layout.embed_jacobian([[1.0, 2.0], [3.0, 4.0]])
        assert embedded.shape == (2, 6)
        assert np.allclose(embedded[:, 2:], 0.0)

    def test_wrong_width_vectors_refuse(self):
        layout = self.make()
        with pytest.raises(TranscriptionError, match='6 wide'):
            layout.reported_of(np.zeros(4))
        with pytest.raises(TranscriptionError, match='no internal variable block'):
            layout.slice_of('nope')


class TestCarryOver:
    """The homotopy transfer: survive by name, seed what is new, drop what is gone."""

    def fine(self):
        return AugmentedLayout(['p'], [-1.0], [1.0],
                               [VariableBlock('a', ['x'], [0.0], [9.0], [1.0]),
                                VariableBlock('b', ['x'], [0.0], [9.0], [2.0]),
                                VariableBlock('c', ['x'], [0.0], [9.0], [3.0])])

    def coarse(self):
        return AugmentedLayout(['p'], [-1.0], [1.0],
                               [VariableBlock('b', ['x'], [0.0], [9.0], [99.0])])

    def test_surviving_blocks_carry_their_solved_values(self):
        fine, coarse = self.fine(), self.coarse()
        u = fine.pack([0.25], {'a': [4.0], 'b': [5.0], 'c': [6.0]})
        moved = fine.carry_over(u, coarse)
        assert len(moved) == coarse.size
        assert np.allclose(coarse.reported_of(moved), [0.25])
        # 'b' survived with its solved value, not with the coarse layout's seed of 99.
        assert np.allclose(coarse.internal_of(moved, 'b'), [5.0])

    def test_new_blocks_take_their_own_initial(self):
        fine, coarse = self.fine(), self.coarse()
        u = coarse.pack([0.25], {'b': [5.0]})
        moved = coarse.carry_over(u, fine)
        assert np.allclose(fine.internal_of(moved, 'b'), [5.0])
        assert np.allclose(fine.internal_of(moved, 'a'), [1.0])
        assert np.allclose(fine.internal_of(moved, 'c'), [3.0])

    def test_carried_values_are_clipped_into_the_target_box(self):
        source = AugmentedLayout(['p'], [-1.0], [1.0],
                                 [VariableBlock('a', ['x'], [0.0], [100.0], [1.0])])
        target = AugmentedLayout(['p'], [-1.0], [1.0],
                                 [VariableBlock('a', ['x'], [0.0], [10.0], [1.0])])
        moved = source.carry_over(source.pack([0.0], {'a': [50.0]}), target)
        assert np.allclose(target.internal_of(moved, 'a'), [10.0])

    def test_a_block_that_changed_width_refuses(self):
        source = self.fine()
        target = AugmentedLayout(['p'], [-1.0], [1.0],
                                 [VariableBlock('b', ['x', 'y'], [0, 0], [9, 9], [1, 1])])
        with pytest.raises(TranscriptionError, match='one block name must mean one thing'):
            source.carry_over(source.initial_point([0.0]), target)

    def test_a_different_fit_refuses(self):
        source = self.fine()
        target = AugmentedLayout(['q'], [-1.0], [1.0], [])
        with pytest.raises(TranscriptionError, match='re-transcribes one fit'):
            source.carry_over(source.initial_point([0.0]), target)


# ---------------------------------------------------------------------------
# The block-sparse equality Jacobian
# ---------------------------------------------------------------------------

class TestBlockJacobian:

    def make(self):
        blocks = [JacobianBlock(slice(0, 2), slice(0, 3), np.arange(6.0).reshape(2, 3) + 1),
                  JacobianBlock(slice(1, 4), slice(3, 5), np.arange(6.0).reshape(3, 2) - 2)]
        return BlockJacobian((4, 5), blocks)

    def test_dense_assembly_places_blocks_and_leaves_structural_zeros(self):
        dense = self.make().to_dense()
        assert dense.shape == (4, 5)
        assert np.allclose(dense[0, 3:], 0.0)
        assert np.allclose(dense[2:, :3], 0.0)
        assert np.allclose(dense[:2, :3], np.arange(6.0).reshape(2, 3) + 1)

    def test_matvec_rmatvec_gram_agree_with_dense(self):
        jac = self.make()
        dense = jac.to_dense()
        rng = np.random.default_rng(0)
        v = rng.normal(size=5)
        y = rng.normal(size=4)
        assert np.allclose(jac.matvec(v), dense @ v)
        assert np.allclose(jac.rmatvec(y), dense.T @ y)
        # The block-wise Gram must match exactly, including the cross terms between two
        # blocks whose row ranges overlap -- the case a naive per-block sum gets wrong.
        assert np.allclose(jac.gram(), dense.T @ dense)

    def test_overlapping_blocks_accumulate(self):
        jac = BlockJacobian((1, 1), [JacobianBlock(slice(0, 1), slice(0, 1), [[2.0]]),
                                     JacobianBlock(slice(0, 1), slice(0, 1), [[3.0]])])
        assert np.allclose(jac.to_dense(), [[5.0]])
        assert np.allclose(jac.matvec([1.0]), [5.0])
        assert np.allclose(jac.gram(), [[25.0]])

    def test_row_scaling_matches_dense_row_scaling(self):
        jac = self.make()
        scales = np.array([1.0, 2.0, 4.0, 0.5])
        assert np.allclose(jac.scaled(scales).to_dense(),
                           jac.to_dense() / scales[:, None])

    def test_density_reports_the_structure_a_condensation_would_act_on(self):
        jac = self.make()
        assert jac.nnz == 2 * 3 + 3 * 2
        assert jac.density == pytest.approx(12 / 20)

    def test_a_block_outside_the_declared_shape_refuses(self):
        with pytest.raises(TranscriptionError, match='outside the declared'):
            BlockJacobian((2, 2), [JacobianBlock(slice(0, 3), slice(0, 2), np.zeros((3, 2)))])

    def test_a_strided_block_refuses(self):
        with pytest.raises(TranscriptionError, match='contiguous'):
            JacobianBlock(slice(0, 4, 2), slice(0, 1), np.zeros((2, 1)))

    def test_a_mis_shaped_block_refuses(self):
        with pytest.raises(TranscriptionError, match='needs a'):
            JacobianBlock(slice(0, 2), slice(0, 2), np.zeros((2, 3)))


class TestEqualityModel:

    def make(self, scales=None):
        jac = BlockJacobian((3, 3), [JacobianBlock(slice(0, 3), slice(0, 3), np.eye(3))])
        return EqualityModel(np.array([1.0, -20.0, 3.0]), jac, scales=scales,
                             names=('a', 'b', 'c'))

    def test_unscaled_defect_is_the_raw_infinity_norm(self):
        assert self.make().defect_norm == pytest.approx(20.0)

    def test_scaling_makes_one_penalty_mean_one_thing_across_states(self):
        # b's raw defect is 20x a's, but it lives on a scale 100x larger; scaled, a is worse.
        model = self.make(scales=np.array([1.0, 100.0, 1.0]))
        assert np.allclose(model.scaled_residual, [1.0, -0.2, 3.0])
        assert model.defect_norm == pytest.approx(3.0)
        assert np.allclose(model.scaled_jacobian.to_dense(),
                           np.diag([1.0, 0.01, 1.0]))

    def test_worst_reports_scaled_defects_worst_first(self):
        worst = self.make(scales=np.array([1.0, 100.0, 1.0])).worst(2)
        assert [name for name, _ in worst] == ['c', 'a']
        assert worst[0][1] == pytest.approx(3.0)

    def test_defect_rms_is_the_aggregate_companion(self):
        model = self.make(scales=np.array([1.0, 100.0, 1.0]))
        assert model.defect_rms == pytest.approx(
            np.sqrt(np.mean(np.array([1.0, -0.2, 3.0]) ** 2)))

    def test_a_non_positive_scale_refuses(self):
        with pytest.raises(TranscriptionError, match='finite and strictly positive'):
            self.make(scales=np.array([1.0, 0.0, 1.0]))

    def test_an_empty_system_is_a_first_class_model(self):
        # The last stage of a segment homotopy, not a degenerate case.
        problem = ShootingProblem(1, [0.0, 1.0], [1.0, 2.0], theta_seed=0.5)
        model = problem.equality_at(problem.layout.initial_point([0.5]))
        assert model.n_constraints == 0
        assert model.defect_norm == 0.0
        assert model.worst() == []


# ---------------------------------------------------------------------------
# The augmented Lagrangian at a point
# ---------------------------------------------------------------------------

class TestAugmentedModel:

    def model(self, u=(0.3, 1.4), lam=(0.7,), rho=3.0, scale=1.0):
        problem = ConstrainedQuadratic(scale=scale)
        u = np.asarray(u, dtype=float)
        return problem, u, problem.augmented_at(u, Multipliers(np.asarray(lam), rho))

    def test_value_is_f_plus_the_lagrangian_and_penalty_terms(self):
        _, u, model = self.model()
        c = u[0] + u[1] - 1.0
        expected = 0.5 * ((u[0] - 1) ** 2 + (u[1] - 2) ** 2) + 0.7 * c + 0.5 * 3.0 * c ** 2
        assert model.value == pytest.approx(expected)

    def test_gradient_matches_a_central_difference_of_the_value(self):
        problem, u, _ = self.model()
        mult = Multipliers(np.array([0.7]), 3.0)
        analytic = problem.augmented_at(u, mult).gradient
        step = 1e-6
        numeric = np.empty_like(u)
        for i in range(len(u)):
            plus, minus = u.copy(), u.copy()
            plus[i] += step
            minus[i] -= step
            numeric[i] = ((problem.augmented_at(plus, mult).value
                           - problem.augmented_at(minus, mult).value) / (2 * step))
        assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8)

    def test_the_stacked_residual_reproduces_the_value_up_to_the_reported_offset(self):
        _, _, model = self.model()
        residual, _ = model.residual_model()
        assert 0.5 * residual @ residual == pytest.approx(model.value + model.residual_offset)
        assert model.residual_offset == pytest.approx(0.7 ** 2 / (2 * 3.0))

    def test_the_stacked_jacobian_reproduces_the_gradient_and_the_gauss_newton_hessian(self):
        _, _, model = self.model()
        residual, jacobian = model.residual_model()
        assert np.allclose(jacobian.T @ residual, model.gradient)
        assert np.allclose(jacobian.T @ jacobian, model.hessian())

    def test_a_scaled_constraint_gives_the_identical_augmented_problem(self):
        # Scaling is an exact reparameterisation: lambda absorbs s, so the value, gradient
        # and step are unchanged. Written in units 1000x larger, with lambda 1000x smaller.
        _, u, plain = self.model(lam=(0.7,), scale=1.0)
        _, _, scaled = self.model(lam=(0.7,), scale=1000.0)
        assert scaled.value == pytest.approx(plain.value)
        assert np.allclose(scaled.gradient, plain.gradient)

    def test_no_constraints_reduces_to_the_plain_objective(self):
        problem = ShootingProblem(1, *shooting_data(n=9), theta_seed=0.5)
        u = problem.layout.initial_point([0.6])
        model = problem.augmented_at(u, Multipliers.zeros(0, 10.0))
        objective = problem.objective_at(u)
        assert model.value == pytest.approx(objective.value)
        assert model.defect_norm == 0.0
        assert model.residual_offset == 0.0
        assert np.allclose(model.gradient, objective.gradient)
        residual, jacobian = model.residual_model()
        assert np.allclose(residual, objective.residual)
        assert np.allclose(jacobian, objective.jacobian)

    def test_the_constraint_term_never_enters_the_objective_value(self):
        """The invariant an estimated noise scale depends on (issue #563, formulation
        point 4): a sigma fitted to the residuals it is given must never be shown a
        continuity defect, or it absorbs constraint violation as measurement noise and the
        reported objective stops being comparable to a single-shoot one."""
        problem, u, _ = self.model()
        objective = problem.objective_at(u)
        for rho in (1.0, 10.0, 1e6):
            for lam in (0.0, 5.0, -5.0):
                model = problem.augmented_at(u, Multipliers(np.array([lam]), rho))
                assert model.objective_value == objective.value
                # ... and the objective's own residual is untouched by the penalty rows.
                assert np.allclose(model.objective.residual, objective.residual)

    def test_a_mismatched_layout_refuses_rather_than_broadcasting(self):
        problem = ConstrainedQuadratic()
        objective = ObjectiveModel(0.0, np.zeros(3))
        equality = problem.equality_at(np.zeros(2))
        with pytest.raises(TranscriptionError, match='same augmented layout'):
            AugmentedModel(objective, equality, Multipliers(np.zeros(1), 1.0))


class TestObjectiveModelAdapter:

    class FakeGradientResult:
        """Duck-typed stand-in for a GradientResult, which is how the layer stays free of
        the gradient package (and of everything that needs a backend)."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def test_a_reported_space_gradient_is_embedded(self):
        layout = AugmentedLayout(['p'], [-1.0], [1.0],
                                 [VariableBlock('z', ['x'], [0.0], [1.0], [0.5])])
        grad = self.FakeGradientResult(gradient=np.array([2.0]),
                                       residual=np.array([1.0, 1.0]),
                                       jacobian=np.array([[3.0], [4.0]]),
                                       hessian=np.array([[25.0]]),
                                       least_squares_exact=True)
        model = ObjectiveModel.from_gradient_result(7.0, grad, layout)
        assert model.value == 7.0
        assert np.allclose(model.gradient, [2.0, 0.0])
        assert model.jacobian.shape == (2, 2)
        assert np.allclose(model.jacobian[:, 1], 0.0)
        assert model.hessian.shape == (2, 2)
        assert model.hessian[1, 1] == 0.0
        assert model.least_squares_exact

    def test_an_augmented_space_gradient_is_used_as_is(self):
        layout = AugmentedLayout(['p'], [-1.0], [1.0],
                                 [VariableBlock('z', ['x'], [0.0], [1.0], [0.5])])
        grad = self.FakeGradientResult(gradient=np.array([2.0, 5.0]), residual=None,
                                       jacobian=None, hessian=None,
                                       least_squares_exact=False)
        model = ObjectiveModel.from_gradient_result(1.0, grad, layout)
        assert np.allclose(model.gradient, [2.0, 5.0])

    def test_a_residual_without_its_jacobian_refuses(self):
        with pytest.raises(TranscriptionError, match='or neither'):
            ObjectiveModel(1.0, np.zeros(2), residual=np.zeros(3))


# ---------------------------------------------------------------------------
# The penalty schedule -- the defaults carry finding 5.1
# ---------------------------------------------------------------------------

class TestPenaltySchedule:

    def test_the_shipped_defaults_start_tight(self):
        """Finding 5.1: rho_0 = 10 / gamma = 5 beat rho_0 = 0.1 / gamma = 3 on the
        motivating problem, at half the cost. A loose start is not merely ineffective -- the
        inner solve on a nearly-unconstrained subproblem burns its whole budget."""
        schedule = PenaltySchedule()
        assert schedule.initial_penalty == 10.0
        assert schedule.growth == 5.0

    def test_targets_reset_from_the_penalty_and_tighten_on_success(self):
        schedule = PenaltySchedule()
        eta, omega = schedule.reset_targets(10.0)
        assert eta == pytest.approx(0.1 ** 0.1)
        assert omega == pytest.approx(0.1)
        tight_eta, tight_omega = schedule.tighten(10.0, eta, omega)
        assert tight_eta < eta and tight_omega < omega
        assert tight_omega == pytest.approx(omega * 0.1)

    def test_the_penalty_is_capped(self):
        schedule = PenaltySchedule(initial_penalty=10.0, growth=5.0, max_penalty=100.0)
        assert schedule.raised(10.0) == 50.0
        assert schedule.raised(50.0) == 100.0
        assert schedule.raised(100.0) == 100.0

    def test_a_growth_factor_that_never_tightens_refuses(self):
        with pytest.raises(TranscriptionError, match='never tightens'):
            PenaltySchedule(growth=1.0)

    def test_a_non_positive_penalty_refuses(self):
        with pytest.raises(TranscriptionError, match='finite and positive'):
            PenaltySchedule(initial_penalty=0.0)


class TestMultipliers:

    def test_the_first_order_update_is_lambda_plus_rho_c(self):
        mult = Multipliers(np.array([1.0, -2.0]), 4.0)
        updated = mult.updated(np.array([0.5, 0.25]))
        assert np.allclose(updated.values, [1.0 + 2.0, -2.0 + 1.0])
        assert updated.penalty == 4.0

    def test_a_runaway_multiplier_is_clamped(self):
        mult = Multipliers(np.array([0.0]), 1e6)
        assert mult.updated(np.array([1e9]), clamp=1e10).values[0] == 1e10

    def test_zeros_start_the_loop_as_a_pure_penalty_solve(self):
        mult = Multipliers.zeros(3, 10.0)
        assert np.allclose(mult.values, 0.0)
        assert mult.penalty == 10.0


# ---------------------------------------------------------------------------
# The outer loop
# ---------------------------------------------------------------------------

class TestAugmentedLagrangian:

    @INNER_SOLVERS
    def test_it_finds_the_known_kkt_point_with_either_inner_optimizer(self, inner_solver):
        """The optimizer-agnostic claim, measured: two inner solvers with nothing in common
        but the subproblem interface reach the same analytic solution."""
        problem = ConstrainedQuadratic()
        loop = AugmentedLagrangian(problem, inner_solver)
        result = loop.run(problem.layout.initial_point([0.9]))
        assert result.converged, result.stop_reason
        assert np.allclose(result.final_point, ConstrainedQuadratic.OPTIMUM, atol=1e-6)
        assert result.defect_norm < 1e-6

    @INNER_SOLVERS
    def test_it_recovers_the_analytic_multiplier(self, inner_solver):
        """A pure quadratic-penalty method reaches the primal point with a wrong (or absent)
        multiplier; recovering lambda* = 1 is what distinguishes the augmented Lagrangian."""
        problem = ConstrainedQuadratic()
        result = AugmentedLagrangian(problem, inner_solver).run(
            problem.layout.initial_point([0.9]))
        assert result.multipliers.values[0] == pytest.approx(
            ConstrainedQuadratic.MULTIPLIER, abs=1e-5)

    def test_the_multiplier_is_recovered_without_an_enormous_penalty(self):
        problem = ConstrainedQuadratic()
        result = AugmentedLagrangian(problem, scalar_inner_solver).run(
            problem.layout.initial_point([0.9]))
        assert result.multipliers.penalty <= 1e3

    def test_a_constraint_scale_does_not_change_where_it_lands(self):
        results = [AugmentedLagrangian(ConstrainedQuadratic(scale=s),
                                       scalar_inner_solver).run(np.array([0.9, 0.0]))
                   for s in (1.0, 1e3)]
        assert np.allclose(results[0].final_point, results[1].final_point, atol=1e-6)

    def test_an_unconstrained_transcription_takes_exactly_one_inner_solve(self):
        problem = ShootingProblem(1, *shooting_data(n=21), theta_seed=0.4)
        result = AugmentedLagrangian(problem, least_squares_inner_solver).run(
            problem.layout.initial_point([0.4]))
        assert result.stop_reason == 'unconstrained'
        assert len(result.iterates) == 1
        assert result.converged

    def test_an_infeasible_system_stops_at_the_penalty_ceiling(self):
        class Infeasible(ConstrainedQuadratic):
            def equality_at(self, u):
                # x + y = 1 AND x + y = 2: no point satisfies both.
                jac = BlockJacobian((2, 2), [JacobianBlock(slice(0, 2), slice(0, 2),
                                                           np.ones((2, 2)))])
                return EqualityModel(np.array([u[0] + u[1] - 1.0, u[0] + u[1] - 2.0]), jac,
                                     names=('a', 'b'))

        result = AugmentedLagrangian(Infeasible(), scalar_inner_solver,
                                     schedule=PenaltySchedule(max_penalty=1e4),
                                     max_outer=40).run(np.array([0.9, 0.0]))
        assert not result.converged
        assert result.stop_reason == 'penalty_ceiling'
        # The run reports the infeasibility rather than presenting a converged-looking fit.
        assert result.defect_norm > 0.1

    def test_the_penalty_is_not_raised_on_an_already_feasible_point(self):
        """The schedule's feasibility target ``eta`` tightens geometrically and will
        eventually drop below any achievable defect. Without a floor at ``feasibility_tol``
        the loop then raises ``rho`` on a point that is feasible by every standard that
        matters, buying nothing and leaving the next subproblem worse conditioned.

        Driven directly rather than through a real solve: the inner solver here always
        lands the same small feasible defect and never reaches the optimality target, so
        ``eta`` is guaranteed to tighten past it. It jitters the point *along* the
        constraint (which leaves the defect exactly unchanged) so the stall detector does
        not end the run before the floor is exercised."""
        defect = 1e-5
        problem = ConstrainedQuadratic()
        base = ConstrainedQuadratic.OPTIMUM + np.array([defect / 2, defect / 2])
        calls = {'n': 0}

        def wobbling_solver(subproblem, u0, tolerance):
            calls['n'] += 1
            jitter = 1e-6 * (calls['n'] % 2)
            return InnerOutcome(base + np.array([jitter, -jitter]), converged=False,
                                n_evaluations=1)

        schedule = PenaltySchedule(feasibility_tol=1e-4, optimality_tol=1e-30)
        result = AugmentedLagrangian(problem, wobbling_solver, schedule=schedule,
                                     max_outer=12).run(base)
        assert result.defect_norm == pytest.approx(defect, rel=1e-6)
        assert result.defect_norm < schedule.feasibility_tol
        # eta starts at rho**-0.1 = 0.79 and tightens by rho**-0.9 each accepted step, so
        # it passes below the achieved defect partway through -- and the penalty must not
        # move when it does.
        assert len(result.iterates) == 12
        assert {it.penalty for it in result.iterates} == {schedule.initial_penalty}

    def test_a_feasible_run_that_stops_improving_stalls_out_rather_than_spinning(self):
        """Once a run is feasible the schedule's inner tolerance is already floored, so an
        inner solver that cannot drive the optimality lower re-solves a near-identical
        subproblem every remaining outer iteration. For a simulator-backed consumer each of
        those is a full inner solve costing hundreds of simulations, so the loop stops and
        reports instead. The best certified iterate survives the stall."""
        problem = ConstrainedQuadratic()

        def stuck_solver(subproblem, u0, tolerance):
            # Exactly feasible (the offset is along x + y = 1) but stationary only to
            # ~1e-4, and unable to do better -- the shape of a quasi-Newton method on a
            # high-penalty augmented Lagrangian.
            return InnerOutcome(ConstrainedQuadratic.OPTIMUM + np.array([1e-4, -1e-4]),
                                converged=False, n_evaluations=1)

        result = AugmentedLagrangian(problem, stuck_solver, max_outer=50,
                                     max_stall=3).run(np.array([0.9, 0.0]))
        assert result.stop_reason == 'stalled'
        assert not result.converged
        assert len(result.iterates) <= 6
        assert result.defect_norm <= PenaltySchedule().feasibility_tol
        assert result.best is not None

    def test_a_penalty_raise_on_a_point_that_never_moves_stops_instead_of_escalating(self):
        """The death spiral the stall detector exists to cut, and the reason it must span
        the penalty-raising branch too.

        Raising ``rho`` is only justified if the previous inner solve did something. An
        inner solver that fails on an ill-conditioned subproblem and returns its own start
        point leaves the defect exactly where it was — which reads as "not feasible
        enough", raises ``rho`` by ``gamma``, and hands the same solver a strictly harder
        problem. Left alone the penalty runs to its ceiling, the augmented gradient grows
        with it, and the point never moves at all."""
        problem = ConstrainedQuadratic()
        infeasible_and_frozen = np.array([0.9, 0.9])       # x + y - 1 = 0.8, never moves

        def frozen_solver(subproblem, u0, tolerance):
            return InnerOutcome(infeasible_and_frozen, converged=False, n_evaluations=1)

        schedule = PenaltySchedule()
        result = AugmentedLagrangian(problem, frozen_solver, max_outer=50,
                                     max_stall=3).run(infeasible_and_frozen)
        assert result.stop_reason == 'stalled'
        # The first iterate sets the baseline it could fail to improve on, so a run that
        # goes nowhere from the start costs 1 + max_stall inner solves, not max_outer.
        assert len(result.iterates) == 4
        # Three raises, not the two dozen it takes to escalate to the ceiling.
        assert result.multipliers.penalty == pytest.approx(
            schedule.initial_penalty * schedule.growth ** 3)
        assert result.multipliers.penalty < schedule.max_penalty
        assert result.defect_norm == pytest.approx(0.8)

    def test_a_run_that_keeps_moving_is_not_stalled_by_a_slow_defect(self):
        """Progress is the defect improving *or* the point moving, so an outer iteration
        that barely improves feasibility while the point is still travelling does not count
        against the stall budget."""
        problem = ConstrainedQuadratic()
        result = AugmentedLagrangian(problem, scalar_inner_solver, max_outer=30,
                                     max_stall=1).run(np.array([0.9, 0.0]))
        assert result.converged, result.stop_reason

    def test_the_max_outer_cap_is_honoured(self):
        problem = ConstrainedQuadratic()
        result = AugmentedLagrangian(problem, scalar_inner_solver, max_outer=1).run(
            np.array([0.9, 0.0]))
        assert len(result.iterates) == 1
        assert result.stop_reason == 'max_outer'

    def test_a_stop_check_ends_the_run_cleanly(self):
        """The wall-clock-budget seam (ADR-0093), without this layer importing FitBudget."""
        calls = {'n': 0}

        def stop_after_two():
            calls['n'] += 1
            return calls['n'] > 2

        problem = ConstrainedQuadratic()
        result = AugmentedLagrangian(problem, scalar_inner_solver, max_outer=50,
                                     stop_check=stop_after_two).run(np.array([0.9, 0.0]))
        assert result.stop_reason == 'stopped'
        assert len(result.iterates) == 2
        assert result.best is not None       # the work already done is still reported

    def test_a_non_finite_inner_point_stops_rather_than_propagating(self):
        def broken_solver(subproblem, u0, tolerance):
            return InnerOutcome(np.full(subproblem.size, np.nan), converged=False)

        result = AugmentedLagrangian(ConstrainedQuadratic(), broken_solver).run(
            np.array([0.9, 0.0]))
        assert result.stop_reason == 'inner_failed'
        assert result.best is None

    def test_an_inner_solver_that_breaks_the_contract_refuses_loudly(self):
        result = AugmentedLagrangian(ConstrainedQuadratic(),
                                     lambda sub, u0, tol: np.zeros(2))
        with pytest.raises(TranscriptionError, match='must return an InnerOutcome'):
            result.run(np.array([0.9, 0.0]))

    def test_evaluation_counts_are_reported(self):
        problem = ConstrainedQuadratic()
        result = AugmentedLagrangian(problem, scalar_inner_solver).run(np.array([0.9, 0.0]))
        assert result.n_inner_evaluations > 0
        assert result.n_outer_evaluations == len(result.iterates)
        assert result.n_evaluations == (result.n_inner_evaluations
                                        + result.n_outer_evaluations)


class TestCertification:

    def test_every_outer_iterate_is_certified_through_the_ordinary_path(self):
        problem = ShootingProblem(2, *shooting_data(n=21), theta_seed=0.3)
        result = AugmentedLagrangian(problem, least_squares_inner_solver).run(
            problem.layout.initial_point([0.3]))
        assert len(problem.certifications) == len(result.iterates)
        assert result.certified

    def test_the_best_iterate_is_reported_not_the_last(self):
        """Finding 5.3: on one prototype start the final stage held -147.0 while an earlier
        iterate certified at -196.3. The loop must keep the best."""
        scores = iter([-10.0, -196.3, -50.0, -147.0])

        class DriftingCertifier(ConstrainedQuadratic):
            def certify(self, reported):
                return Certificate.accept(next(scores, 0.0))

        problem = DriftingCertifier()
        result = AugmentedLagrangian(problem, scalar_inner_solver, max_outer=4).run(
            np.array([0.9, 0.0]))
        assert len(result.iterates) == 4
        assert result.iterates[-1].score == pytest.approx(-147.0)
        assert result.best.score == pytest.approx(-196.3)
        assert result.best.iteration == 2

    def test_a_rejected_reconstruction_never_becomes_the_answer(self):
        verdicts = iter([Certificate.reject('did not simulate'),
                         Certificate.accept(3.0),
                         Certificate.reject('non-finite objective')])

        class FlakyCertifier(ConstrainedQuadratic):
            def certify(self, reported):
                return next(verdicts, Certificate.accept(100.0))

        result = AugmentedLagrangian(FlakyCertifier(), scalar_inner_solver,
                                     max_outer=3).run(np.array([0.9, 0.0]))
        assert result.best.score == pytest.approx(3.0)

    def test_a_problem_that_cannot_certify_says_so(self):
        class NoCertifier(ConstrainedQuadratic):
            def certify(self, reported):
                return None

        result = AugmentedLagrangian(NoCertifier(), scalar_inner_solver, max_outer=2).run(
            np.array([0.9, 0.0]))
        assert not result.certified
        assert 'UNCERTIFIED' in result.summary()
        # It still ranks something -- the augmented problem's own objective -- but the flag
        # travels with it, so the score is never mistaken for a single-shoot one.
        assert result.best is not None
        assert not result.best.certificate.certified

    def test_certified_best_keeps_the_earlier_of_two_ties(self):
        best = CertifiedBest()
        first = self.record('a', 1, 5.0)
        second = self.record('b', 2, 5.0)
        assert best.offer(first)
        assert not best.offer(second)
        assert best.record is first

    @staticmethod
    def record(stage, iteration, score):
        return CertifiedIterate(stage, iteration, np.zeros(1), np.zeros(1),
                                Certificate.accept(score), 0.0, score, score, 10.0, 0.0)


# ---------------------------------------------------------------------------
# Multiple shooting, offline: the equivalence claim
# ---------------------------------------------------------------------------

class TestOfflineMultipleShooting:

    def setup_method(self):
        self.times, self.observations = shooting_data()
        self.optimum = single_shoot_optimum(self.times, self.observations)

    def problem(self, m, theta_seed):
        return ShootingProblem(m, self.times, self.observations, theta_seed=theta_seed,
                               horizon=float(self.times.max()))

    def test_the_continuity_jacobian_matches_a_central_difference(self):
        """The block-sparse constraint Jacobian, measured rather than asserted -- including
        the initial-condition column, which is the block the whole formulation rests on."""
        problem = self.problem(4, theta_seed=0.55)
        u = problem.layout.initial_point([0.55]) * (1.0 + np.linspace(0.02, 0.09, 4))
        analytic = problem.equality_at(u).jacobian.to_dense()
        step = 1e-7
        numeric = np.empty_like(analytic)
        for j in range(len(u)):
            plus, minus = u.copy(), u.copy()
            plus[j] += step
            minus[j] -= step
            numeric[:, j] = ((problem.equality_at(plus).residual
                              - problem.equality_at(minus).residual) / (2 * step))
        assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    def test_the_augmented_gradient_matches_a_central_difference(self):
        problem = self.problem(4, theta_seed=0.55)
        mult = Multipliers(np.array([0.3, -0.4, 0.9]), 7.0)
        u = problem.layout.initial_point([0.55]) * (1.0 + np.linspace(0.02, 0.09, 4))
        analytic = problem.augmented_at(u, mult).gradient
        step = 1e-7
        numeric = np.empty_like(u)
        for j in range(len(u)):
            plus, minus = u.copy(), u.copy()
            plus[j] += step
            minus[j] -= step
            numeric[j] = ((problem.augmented_at(plus, mult).value
                           - problem.augmented_at(minus, mult).value) / (2 * step))
        assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-5)

    def test_a_feasible_augmented_point_scores_exactly_the_single_shoot_objective(self):
        """The equivalence the formulation rests on, at the point it is claimed: when the
        continuity defects are zero, the segmented objective *is* the uninterrupted one."""
        problem = self.problem(4, theta_seed=0.55)
        theta = 0.55
        feasible = problem.layout.pack(
            [theta], {name: [problem.y0 * np.exp(theta * t)]
                      for name, t in zip(problem.block_names, problem.knot_times)})
        assert problem.equality_at(feasible).defect_norm < 1e-12
        assert (problem.objective_at(feasible).value
                == pytest.approx(problem.single_shoot_objective(theta), rel=1e-12))

    def test_the_converged_transcription_reproduces_the_uninterrupted_fit(self):
        """The claim the whole formulation rests on, against an independently computed
        optimum. Driven by the least-squares inner solver, which is the shape the MVP's
        own inner optimizer (``gntr``) has; the quasi-Newton solver's weaker behaviour on
        this problem is measured separately below."""
        problem = self.problem(4, theta_seed=0.4)
        result = AugmentedLagrangian(problem, least_squares_inner_solver,
                                     max_outer=30).run(problem.layout.initial_point([0.4]))
        assert result.converged, result.stop_reason
        assert result.defect_norm < 1e-5
        assert result.best.reported[0] == pytest.approx(self.optimum, abs=1e-4)
        # The certified objective is the single-shoot one, by construction and in fact.
        assert result.best.score == pytest.approx(
            problem.single_shoot_objective(self.optimum), rel=1e-6)

    def test_it_closes_the_segments_from_a_start_whose_knot_states_are_badly_stale(self):
        """The realistic start, not the flattering one. Seeding the knots from a nominal
        trajectory makes the transcription *feasible* at iteration zero, which is not the
        state a fit is in after theta has moved: the segments start disjoint and the outer
        loop has to close them. Here they start off by an order of magnitude."""
        problem = self.problem(4, theta_seed=0.4)
        stale = problem.layout.pack([0.4], {name: [0.1 * problem.y0]
                                            for name in problem.block_names})
        assert problem.equality_at(stale).defect_norm > 1.0
        result = AugmentedLagrangian(problem, least_squares_inner_solver,
                                     max_outer=40).run(stale)
        assert result.converged, result.stop_reason
        assert result.defect_norm < 1e-5
        assert result.best.reported[0] == pytest.approx(self.optimum, abs=1e-4)

    @INNER_SOLVERS
    def test_a_run_that_reports_convergence_has_the_uninterrupted_fit(self, inner_solver):
        """The safety property, and the honest form of the optimizer-agnostic claim.

        Whether a given inner solver *reaches* the KKT stop on this problem is a property
        of that solver, not of the layer: the test needs the defect and the first-order
        optimality below tolerance in one iterate, and a quasi-Newton method built from
        gradient differences handles an augmented Lagrangian whose penalty term carries a
        large ``rho`` less well than a Gauss-Newton one that sees ``rho J^T J`` explicitly.
        Over 20 data seeds x 2 starts the least-squares solver converges 40/40; the
        quasi-Newton one converges 23/40 and stalls out honestly on the rest.

        What must hold for *both*, on every seed and every start, is that a run reporting
        `converged` has actually found the uninterrupted fit — no false positives. A loop
        that certified a wrong answer would be far worse than one that gives up."""
        for seed in range(5):
            times, observations = shooting_data(seed=seed)
            optimum = single_shoot_optimum(times, observations)
            problem = ShootingProblem(4, times, observations, theta_seed=0.4,
                                      horizon=float(times.max()))
            for start in (problem.layout.initial_point([0.4]),
                          problem.layout.pack([0.4], {name: [0.1 * problem.y0]
                                                      for name in problem.block_names})):
                result = AugmentedLagrangian(problem, inner_solver, max_outer=40).run(start)
                if not result.converged:
                    # Giving up is allowed; the run says so and reports what it has.
                    assert result.stop_reason in ('stalled', 'max_outer'), result.stop_reason
                    continue
                assert result.defect_norm < 1e-5, (seed, result.stop_reason)
                assert result.best.reported[0] == pytest.approx(optimum, abs=1e-4)

    def test_the_optimality_measure_is_the_other_half_of_the_kkt_test(self):
        problem = self.problem(4, theta_seed=0.4)
        result = AugmentedLagrangian(problem, least_squares_inner_solver, max_outer=30).run(
            problem.layout.initial_point([0.4]))
        assert result.converged, result.stop_reason
        assert result.optimality <= PenaltySchedule().optimality_tol
        assert result.iterates[0].optimality > result.iterates[-1].optimality

    def test_the_default_ladder_starts_in_the_middle(self):
        """Finding 5.2: many-short-segments is the wrong end of the homotopy. The default
        ladder is 4-2-1, not 8-4-2-1."""
        assert coarsening_ladder() == (4, 2, 1)
        assert coarsening_ladder(8) == (8, 4, 2, 1)

    @INNER_SOLVERS
    def test_the_homotopy_carries_knot_states_down_the_ladder_and_lands_on_the_fit(
            self, inner_solver):
        stages = [lambda reported, m=m: self.problem(m, theta_seed=float(reported[0]))
                  for m in coarsening_ladder()]
        result = run_homotopy(stages, inner_solver, [0.4], max_outer=20)
        assert [s.name for s in result.stages] == ['m=4', 'm=2', 'm=1']
        assert result.certified
        assert result.reported[0] == pytest.approx(self.optimum, abs=1e-4)
        # The final rung has no constraints, so it *is* the single-shoot problem: its
        # certified score is the fit's own objective at the optimum.
        assert result.best_score == pytest.approx(
            ShootingProblem(1, self.times, self.observations,
                            theta_seed=0.4).single_shoot_objective(self.optimum), rel=1e-6)

    def test_the_ladder_transfers_solved_knot_states_rather_than_reseeding(self):
        """A knot shared by two rungs (t = T/2 is a knot at m = 4 and at m = 2) must arrive
        at the coarser rung carrying its solved value."""
        seen = {}

        def stage(reported, m):
            problem = self.problem(m, theta_seed=float(reported[0]))
            seen.setdefault(m, problem)
            return problem

        starts = {}

        def recording_solver(subproblem, u0, tolerance):
            starts.setdefault(subproblem.problem.name, np.array(u0, dtype=float))
            return least_squares_inner_solver(subproblem, u0, tolerance)

        result = run_homotopy([lambda r, m=m: stage(r, m) for m in (4, 2)],
                              recording_solver, [0.4], max_outer=10)
        fine, coarse = seen[4], seen[2]
        shared = 'knot@0.5000'
        assert shared in fine.block_names and shared in coarse.block_names
        solved = fine.layout.internal_of(result.stages[0].final_point, shared)
        arrived = coarse.layout.internal_of(starts['m=2'], shared)
        assert np.allclose(arrived, solved)
        # ... and the seed the coarse stage would have used on its own is a different value.
        assert not np.allclose(arrived, coarse.layout.block(shared).initial)

    def test_the_trace_is_the_stage_by_stage_mechanism(self):
        stages = [lambda reported, m=m: self.problem(m, theta_seed=float(reported[0]))
                  for m in coarsening_ladder()]
        result = run_homotopy(stages, least_squares_inner_solver, [0.4], max_outer=20)
        trace = result.trace()
        assert trace.startswith('m=4: ')
        assert 'm=2: ' in trace and 'm=1: ' in trace
        assert len(result.stages) == 3

    def test_the_ladder_reports_the_best_stage_not_the_last(self):
        """The asymmetry the driver is built around: continue from the last point, report
        the best certified one -- even when a later stage ends worse."""
        class Stage(ConstrainedQuadratic):
            def __init__(self, label, score):
                super().__init__()
                self.label = label
                self.score = score

            @property
            def name(self):
                return self.label

            def certify(self, reported):
                return Certificate.accept(self.score)

        result = run_homotopy([Stage('fine', -20.0), Stage('mid', -300.0),
                               Stage('coarse', -5.0)],
                              scalar_inner_solver, [0.9], max_outer=1)
        assert result.stages[-1].best_score == pytest.approx(-5.0)
        assert result.best_score == pytest.approx(-300.0)
        assert result.best.stage == 'mid'
        assert 'from mid' in result.summary()

    def test_a_homotopy_needs_at_least_one_stage(self):
        with pytest.raises(TranscriptionError, match='at least one stage'):
            run_homotopy([], scalar_inner_solver, [0.4])

    def test_a_stage_that_is_not_a_transcription_problem_refuses(self):
        with pytest.raises(TranscriptionError, match='not a TranscriptionProblem'):
            run_homotopy([lambda reported: 'nope'], scalar_inner_solver, [0.4])

    def test_a_stop_check_ends_the_ladder_between_stages(self):
        problem = self.problem(2, theta_seed=0.4)
        calls = {'n': 0}

        def stop_soon():
            calls['n'] += 1
            return calls['n'] > 3

        result = run_homotopy([problem, self.problem(1, theta_seed=0.4)],
                              least_squares_inner_solver, [0.4], max_outer=2,
                              stop_check=stop_soon)
        assert result.stop_reason == 'stopped'
        assert result.best is not None


class TestLadder:

    def test_a_ladder_always_ends_at_the_ordinary_problem(self):
        for start in (1, 2, 3, 4, 5, 8, 16):
            assert coarsening_ladder(start)[-1] == 1
            assert list(coarsening_ladder(start)) == sorted(coarsening_ladder(start),
                                                            reverse=True)

    def test_a_ladder_of_one_is_the_ordinary_problem(self):
        assert coarsening_ladder(1) == (1,)

    def test_a_factor_of_three_coarsens_faster(self):
        assert coarsening_ladder(9, factor=3) == (9, 3, 1)

    def test_a_degenerate_ladder_refuses(self):
        with pytest.raises(TranscriptionError, match='factor of at least 2'):
            coarsening_ladder(4, factor=1)
        with pytest.raises(TranscriptionError, match='end at 1 or above'):
            coarsening_ladder(4, stop=0)


# ---------------------------------------------------------------------------
# The interfaces themselves
# ---------------------------------------------------------------------------

class TestInterfaces:

    def test_an_equality_system_is_the_constraint_half_of_a_transcription_problem(self):
        """The two ABCs compose by inheritance rather than through an adapter: EqualitySystem
        declares the same equality_at a TranscriptionProblem requires."""
        problem = ShootingProblem(4, *shooting_data(n=13), theta_seed=0.5)
        assert isinstance(problem, EqualitySystem)
        assert isinstance(problem, TranscriptionProblem)
        assert problem.n_constraints == 3
        model = problem.equality_at(problem.layout.initial_point([0.5]))
        assert isinstance(model, EqualityModel)
        assert model.names == problem.constraint_names

    def test_the_loop_reads_the_declared_constraint_count_rather_than_linearising(self):
        """For a simulator-backed consumer ``equality_at`` is a pass of segment simulations,
        so the loop must not spend one just to ask how many constraints exist. A problem
        that declares ``n_constraints`` (every EqualitySystem does) is asked; one that does
        not is linearised once, which is the fallback, not the path."""
        problem = ShootingProblem(2, *shooting_data(n=13), theta_seed=0.5)
        calls = {'n': 0, 'at_first_solve': None}
        linearise = problem.equality_at
        problem.equality_at = lambda u: (calls.__setitem__('n', calls['n'] + 1)
                                         or linearise(u))

        def watching_solver(subproblem, u0, tolerance):
            if calls['at_first_solve'] is None:
                calls['at_first_solve'] = calls['n']
            return least_squares_inner_solver(subproblem, u0, tolerance)

        AugmentedLagrangian(problem, watching_solver, max_outer=1).run(
            problem.layout.initial_point([0.5]))
        assert calls['at_first_solve'] == 0
        assert calls['n'] > 0                       # the loop does linearise, just later

    def test_the_subproblem_exposes_the_box_and_nothing_about_multipliers(self):
        problem = ConstrainedQuadratic()
        subproblem = AugmentedSubproblem(problem, Multipliers(np.array([0.5]), 10.0))
        assert subproblem.size == 2
        assert np.allclose(subproblem.lower, [-10.0, -10.0])
        assert np.allclose(subproblem.upper, [10.0, 10.0])
        value, gradient = subproblem.value_and_gradient(np.array([0.3, 1.4]))
        assert value == pytest.approx(subproblem.at(np.array([0.3, 1.4])).value)
        assert len(gradient) == 2

    def test_the_layer_pulls_in_no_backend_configuration_or_gradient_module(self):
        """The property that makes every test in this file offline, checked structurally
        rather than asserted: importing the layer must not drag in a simulation backend, the
        configuration machinery, or the gradient assembly. Its only PyBNF dependency is
        ``printing`` (for :class:`~pybnf.printing.PybnfError`)."""
        import ast
        import pathlib

        import pybnf.transcription as layer

        allowed = {'pybnf.printing', 'pybnf.transcription'}
        package = pathlib.Path(layer.__file__).parent
        for path in sorted(package.glob('*.py')):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    # '.errors' -> pybnf.transcription.errors; '..printing' -> pybnf.printing
                    root = 'pybnf.transcription' if node.level == 1 else 'pybnf'
                    resolved = '%s.%s' % (root, node.module) if node.module else root
                    assert any(resolved.startswith(a) for a in allowed), \
                        '%s imports %s' % (path.name, resolved)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith('pybnf'), \
                            '%s imports %s' % (path.name, alias.name)
