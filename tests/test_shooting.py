"""The multiple-shooting consumer (#563, ADR-0110), offline.

:mod:`pybnf.shooting` narrows its simulator coupling to one method
(:class:`~pybnf.shooting.backend.SegmentBackend.simulate`) precisely so everything above it
can be verified without a simulator -- the same strategy that let ADR-0109's layer be
verified against a closed-form transcription. This module implements that one method for a
problem whose flow and both sensitivities are elementary, and then holds the consumer to the
claims that matter:

* **knot identity** -- a coarser stage's knots are a subset of a finer stage's *by name*, so
  the ``4 -> 2 -> 1`` ladder continues rather than reseeds. If this breaks, the homotopy
  silently becomes three independent solves, which is the mechanism ADR-0109 finding 5.2
  says the method rests on;
* **the ``IC`` route with factor 1** -- the objective's gradient and the continuity
  Jacobian, both pinned against central differences of the quantities they claim to
  differentiate, including the corner where a reported free parameter *is* an initial
  condition and therefore contributes on segment 0 and nowhere else;
* **feasibility at iteration zero** -- seeding the knots from a nominal trajectory leaves
  every continuity defect at zero, so a discontinuity the run holds later is the optimizer's
  own;
* **the load-bearing claim** -- at convergence the transcription is equivalent to the
  uninterrupted fit, measured against a single-shoot optimum computed **independently** by
  ``scipy.least_squares`` rather than against this package's own arithmetic written twice.

The offline problem
-------------------
Two states, ``y' = k y`` and ``w' = -k w``, with only ``y`` observed. The flow is
``y(t) = z_y e^{k dt}``, ``w(t) = z_w e^{-k dt}``, and both sensitivity axes are elementary.
It has the shape the simulator-backed consumer has -- knots, auxiliary segment-start states,
continuity defects, a data term reading its own segment's auxiliary state, and a
single-shoot reconstruction -- and it has the motivating problem's hard corner: ``w`` carries
no data term at all, so half the auxiliary variables are determined by continuity alone.
"""

import threading

import numpy as np
import pytest
from scipy.optimize import least_squares

from pybnf.data import Data, OutputSensitivities
from pybnf.gradient import IC, PARAM, ExperimentRouting, ParamRoute
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, PSet
from pybnf.shooting import (
    EQUAL_OBSERVATIONS,
    EQUAL_TIME,
    EXPLICIT,
    GaussNewtonSolver,
    SegmentBackend,
    SegmentGrid,
    SegmentPool,
    SegmentSimulationFailed,
    ShootingExperiment,
    feasible_ladder,
    max_segments,
    run_multiple_shooting,
    seed_stage,
)
from pybnf.shooting.grid import KNOT
from pybnf.transcription import AugmentedLagrangian, Multipliers, PenaltySchedule

W0 = 2.0           # the unobserved state's (fixed) initial value
SIGMA = 0.05


# ---------------------------------------------------------------------------
# The closed-form backend
# ---------------------------------------------------------------------------

class TwoStateBackend(SegmentBackend):
    """``y' = k y``, ``w' = -k w`` in closed form, with both sensitivity axes.

    Deliberately the same *shape* a bngsim segment run returns: a
    :class:`~pybnf.data.Data` whose columns are the states, carrying an
    :class:`~pybnf.data.OutputSensitivities` with ``species:<name>`` selectors and both a
    parameter and an initial-condition axis. Everything in :mod:`pybnf.shooting` above
    :meth:`simulate` therefore runs unmodified.
    """

    def __init__(self, n_lanes=1):
        self.n_simulations = 0
        self.fail_beyond = None    # |k| above which this "model" refuses to integrate
        self._n_lanes = int(n_lanes)
        #: Lanes currently mid-integration, so a test can assert that a scheduler never puts
        #: two segments in one lane at once -- the invariant a stateful simulator needs and
        #: this closed-form one cannot notice being violated.
        self.busy = set()
        self.max_concurrent = 0
        self.collisions = 0
        #: An optional :class:`threading.Barrier` every simulation waits on, which turns
        #: "did these actually overlap?" from a timing observation into a deterministic one:
        #: a pass that runs its segments one at a time cannot get past it.
        self.barrier = None
        self._lock = threading.Lock()

    @property
    def state_names(self):
        return ('y', 'w')

    @property
    def nominal_state(self):
        return np.array([1.0, W0])

    def open_lanes(self, pset, n_lanes):
        return min(int(n_lanes), self._n_lanes)

    def simulate(self, pset, sample_times, initial_state=None, lane=0):
        with self._lock:
            self.n_simulations += 1
            if lane in self.busy:
                self.collisions += 1
            self.busy.add(lane)
            self.max_concurrent = max(self.max_concurrent, len(self.busy))
        try:
            if self.barrier is not None:
                self.barrier.wait()
            return self._simulate(pset, sample_times, initial_state)
        finally:
            with self._lock:
                self.busy.discard(lane)

    def _simulate(self, pset, sample_times, initial_state=None):
        k = float(pset['k'])
        if self.fail_beyond is not None and abs(k) > self.fail_beyond:
            raise SegmentSimulationFailed('the closed-form backend refuses |k| > %g'
                                          % self.fail_beyond)
        state = ({'y': float(pset['y0']), 'w': W0} if initial_state is None
                 else {name: float(value) for name, value in initial_state.items()})
        times = np.asarray(sample_times, dtype=float)
        dt = times - times[0]
        grow, decay = np.exp(k * dt), np.exp(-k * dt)
        y, w = state['y'] * grow, state['w'] * decay

        data = Data.from_columns(np.column_stack([times, y, w]), ['time', 'y', 'w'])
        d_param = np.zeros((len(times), 2, 1))
        d_param[:, 0, 0] = state['y'] * dt * grow          # dy/dk
        d_param[:, 1, 0] = -state['w'] * dt * decay        # dw/dk
        d_ic = np.zeros((len(times), 2, 2))
        d_ic[:, 0, 0] = grow                               # dy/dz_y
        d_ic[:, 1, 1] = decay                              # dw/dz_w
        data.output_sensitivities = OutputSensitivities(
            selectors=['species:y', 'species:w'], param_names=['k'], ic_species=['y', 'w'],
            d_param=d_param, d_ic=d_ic)
        return data


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------

def observations(k_true=0.7, y0_true=1.0, n=25, horizon=3.0, seed=4):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, horizon, n)
    return times, y0_true * np.exp(k_true * times) + rng.normal(0.0, SIGMA, size=n)


def variables(k=0.4, y0=0.8, log_space=False):
    """The fit's reported free parameters.

    ``log_space`` makes both ``loguniform_var``, so the search coordinate is ``log10(theta)``
    and every column of the objective Jacobian *and* of the continuity block carries a
    ``d theta/d u = ln(10) * theta`` factor. That factor is applied in two different places
    (the assembly's own, and this package's for the constraint rows), so the log variant is
    what stops the second from being a second implementation of the first.
    """
    kind = 'loguniform_var' if log_space else 'uniform_var'
    lower = 0.01 if log_space else -2.0
    return [FreeParameter('k', kind, lower, 2.0, value=k),
            FreeParameter('y0', kind, 0.01, 10.0, value=y0)]


def pset_from_u_for(free_params):
    def pset_from_u(u):
        return PSet([v.set_value(v.from_sampling_space(u[i]))
                     for i, v in enumerate(free_params)])
    return pset_from_u


def make_spec(times, obs, backend=None):
    exp_data = Data.from_columns(
        np.column_stack([times, obs, np.full(len(obs), SIGMA)]), ['time', 'y', 'y_SD'])
    routing = ExperimentRouting(routes={
        'k': ParamRoute.single('k', PARAM, 'k', 1.0),
        'y0': ParamRoute.single('y0', IC, 'y', 1.0),
    })
    return ShootingExperiment(('model', 'exp1'), backend or TwoStateBackend(), exp_data,
                              routing, label='exp1', start=0.0)


def build_stage(n_segments, start_u=(0.4, 0.8), seed=4, backend=None, log_space=False,
                pool=None):
    """One rung, seeded at ``start_u``, plus the pieces a caller needs to poke at it.

    ``start_u`` is in **sampling space**, so under ``log_space`` it is ``log10(theta)``.
    """
    times, obs = observations(seed=seed)
    free = variables(log_space=log_space)
    spec = make_spec(times, obs, backend=backend)
    problem = seed_stage([spec], n_segments, ChiSquareObjective(), free,
                         pset_from_u_for(free), np.asarray(start_u, dtype=float), pool=pool)
    return problem, spec, free


def single_shoot_optimum(times, obs):
    """The uninterrupted fit's optimum, computed independently of this package."""
    result = least_squares(
        lambda p: (p[1] * np.exp(p[0] * times) - obs) / SIGMA, [0.4, 0.8],
        bounds=([-2.0, 0.01], [2.0, 10.0]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return np.asarray(result.x, dtype=float), float(0.5 * result.fun @ result.fun)


# ---------------------------------------------------------------------------
# Knot placement
# ---------------------------------------------------------------------------

class TestSegmentGrid:

    def test_knot_names_nest_down_the_ladder(self):
        """The load-bearing naming property: a coarser stage's knots are a *subset by name*
        of a finer one's, which is the only thing ``carry_over`` matches on."""
        times = np.linspace(0.0, 3.0, 13)
        fine = SegmentGrid(times, 4, label='exp1')
        mid = SegmentGrid(times, 2, label='exp1')
        assert fine.block_names == ('exp1@1/4', 'exp1@1/2', 'exp1@3/4')
        assert mid.block_names == ('exp1@1/2',)
        assert set(mid.block_names) < set(fine.block_names)
        assert SegmentGrid(times, 1, label='exp1').block_names == ()

    def test_a_knot_is_named_by_its_exact_fraction_not_a_rounded_float(self):
        times = np.linspace(0.0, 1.0, 7)
        assert SegmentGrid(times, 3, label='e').block_names == ('e@1/3', 'e@2/3')
        # Exact rationals, so 1/3 at m=3 and 2/6 at m=6 are one knot, not two names.
        assert 'e@1/3' in SegmentGrid(times, 6, label='e').block_names

    def test_a_point_on_a_knot_belongs_to_the_later_segment(self):
        """Half-open ``[start_j, start_{j+1})``: a point exactly on a knot is read at
        ``dt = 0`` from that knot's own auxiliary state, which is the one row where the data
        sees a segment-start state directly."""
        grid = SegmentGrid(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 2, label='e', start=0.0,
                           horizon=4.0)
        assert grid.knot_times == (2.0,)
        assert list(grid.segment_of) == [0, 0, 1, 1, 1]

    def test_every_segment_outputs_its_own_start_and_end_knot(self):
        grid = SegmentGrid(np.linspace(0.0, 3.0, 7), 3, label='e', start=0.0)
        for segment in range(3):
            pts, rows = grid.sample_times(segment)
            assert pts[0] == pytest.approx(grid.starts[segment])
            assert pts[-1] == pytest.approx(grid.ends[segment])
            positions, same_rows = grid.row_positions(segment)
            assert list(same_rows) == list(rows)
            np.testing.assert_allclose(pts[positions], grid.times[rows])

    def test_a_label_may_not_contain_the_knot_separator(self):
        with pytest.raises(PybnfError, match='knot-name separator'):
            SegmentGrid(np.linspace(0, 1, 3), 2, label='a%sb' % KNOT)


# ---------------------------------------------------------------------------
# Where the knots go (#563: "a segment count or explicit knots; default to generic
# equal-time or equal-observation segments")
# ---------------------------------------------------------------------------

#: An unevenly sampled course: 8 points crowded into the first fifth of the horizon, 2 in
#: the rest. Equal *time* leaves a segment with nothing to fit; equal *observations* does
#: not. That difference is the whole reason the second placement exists.
_BURSTY = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.0, 10.0])


class TestKnotPlacement:

    def test_equal_time_is_the_default_and_cuts_the_horizon_evenly(self):
        grid = SegmentGrid(np.linspace(0.0, 8.0, 17), 4, label='e', start=0.0)
        assert grid.placement == EQUAL_TIME
        np.testing.assert_allclose(grid.knot_times, (2.0, 4.0, 6.0))

    def test_equal_observations_balances_the_data_where_equal_time_does_not(self):
        """The measured difference, on one unevenly sampled course rather than in the
        abstract: equal spans leave a segment empty, equal observations do not."""
        by_time = SegmentGrid(_BURSTY, 2, label='e', start=0.0, placement=EQUAL_TIME)
        by_data = SegmentGrid(_BURSTY, 2, label='e', start=0.0,
                              placement=EQUAL_OBSERVATIONS)
        assert [len(by_time.rows_in(j)) for j in range(2)] == [8, 2]
        assert [len(by_data.rows_in(j)) for j in range(2)] == [5, 5]

    def test_every_placement_names_a_knot_the_same_way_so_the_ladder_carries_it(self):
        """The property the homotopy rests on, which is *not* automatic once the knot time
        stops being a fraction of the horizon: the coarse grid's knot must be the same knot,
        by name and by time, as the fine grid's."""
        for placement in (EQUAL_TIME, EQUAL_OBSERVATIONS):
            fine = SegmentGrid(_BURSTY, 4, label='e', start=0.0, placement=placement)
            mid = SegmentGrid(_BURSTY, 2, label='e', start=0.0, placement=placement)
            assert mid.block_names == ('e@1/2',)
            assert set(mid.block_names) < set(fine.block_names)
            # Same name *and* same time -- a name that carried over onto a different knot
            # would reseed the ladder with a state belonging somewhere else.
            assert mid.knot_times[0] == fine.knot_times[fine.block_names.index('e@1/2')]

    def test_explicit_knots_are_used_as_given(self):
        grid = SegmentGrid(np.linspace(0.0, 8.0, 17), 4, label='e', start=0.0,
                           knots=[0.5, 1.0, 7.0])
        assert grid.placement == EXPLICIT
        np.testing.assert_allclose(grid.knot_times, (0.5, 1.0, 7.0))
        assert grid.block_names == ('e@1/4', 'e@1/2', 'e@3/4')

    def test_a_coarser_rung_keeps_the_explicit_knot_its_fraction_names(self):
        times = np.linspace(0.0, 8.0, 17)
        fine = SegmentGrid(times, 4, label='e', start=0.0, knots=[0.5, 1.0, 7.0])
        mid = SegmentGrid(times, 2, label='e', start=0.0, knots=[0.5, 1.0, 7.0])
        assert mid.block_names == ('e@1/2',)
        assert mid.knot_times == (1.0,)     # the fine grid's e@1/2, not its first or last
        assert mid.knot_times[0] == fine.knot_times[fine.block_names.index('e@1/2')]

    def test_out_of_order_explicit_knots_are_refused_by_name(self):
        with pytest.raises(PybnfError, match='not strictly increasing'):
            SegmentGrid(np.linspace(0.0, 8.0, 17), 3, label='e', start=0.0,
                        knots=[5.0, 2.0])

    def test_an_explicit_knot_outside_the_horizon_is_refused(self):
        with pytest.raises(PybnfError, match='not strictly increasing'):
            SegmentGrid(np.linspace(0.0, 8.0, 17), 2, label='e', start=0.0, knots=[9.0])

    def test_an_unknown_placement_is_refused(self):
        with pytest.raises(PybnfError, match='knot placement'):
            SegmentGrid(np.linspace(0.0, 8.0, 17), 2, label='e', placement='at_the_peaks')

    def test_each_placement_declares_the_segment_count_its_data_supports(self):
        """The ceiling ``feasible_ladder`` drops rungs against, and it is not one number:
        equal time needs a measurement per segment, equal observations two, and an explicit
        list *is* the finest rung."""
        times = np.linspace(0.0, 1.0, 10)
        assert max_segments(times, EQUAL_TIME) == 10
        assert max_segments(times, EQUAL_OBSERVATIONS) == 5
        assert max_segments(times, EXPLICIT, knots=[0.3, 0.6]) == 3

    def test_equal_observations_refuses_a_count_the_sampling_cannot_support(self):
        with pytest.raises(PybnfError, match='equal observations'):
            SegmentGrid(np.linspace(0.0, 1.0, 4), 4, label='e', start=0.0,
                        placement=EQUAL_OBSERVATIONS)


# ---------------------------------------------------------------------------
# The layout the transcription declares
# ---------------------------------------------------------------------------

class TestStageLayout:

    def test_auxiliary_states_are_never_reported_parameters(self):
        problem, _spec, free = build_stage(4)
        layout = problem.layout
        assert layout.reported_names == ('k', 'y0')
        assert layout.n_internal == 3 * 2                      # 3 knots x 2 states
        assert layout.names[:2] == ('k', 'y0')
        assert all('::' in name for name in layout.names[2:])
        u = layout.initial_point([0.4, 0.8])
        np.testing.assert_allclose(layout.reported_of(u), [0.4, 0.8])

    def test_the_seeded_start_point_is_feasible(self):
        """Seeding each knot from a nominal trajectory at the incoming parameters makes the
        transcription feasible at iteration zero, so every discontinuity the run holds later
        is the optimizer's own choice rather than an artifact of the stage."""
        problem, _spec, _free = build_stage(4)
        u = problem.layout.initial_point([0.4, 0.8])
        assert problem.equality_at(u).defect_norm == pytest.approx(0.0, abs=1e-9)

    def test_a_coarser_stage_carries_the_surviving_knot_and_drops_the_rest(self):
        fine, spec, free = build_stage(4)
        coarse = seed_stage([spec], 2, ChiSquareObjective(), free, pset_from_u_for(free),
                            np.array([0.4, 0.8]))
        u = fine.layout.initial_point([0.4, 0.8])
        u[fine.layout.slice_of('exp1@1/2')] = [0.25, -0.75]     # a "solved" knot state
        moved = fine.layout.carry_over(u, coarse.layout)
        np.testing.assert_allclose(coarse.layout.internal_of(moved, 'exp1@1/2'),
                                   [0.25, -0.75])
        assert coarse.layout.block_names == ('exp1@1/2',)

    def test_the_one_segment_stage_is_the_unsegmented_problem(self):
        problem, _spec, _free = build_stage(1)
        assert problem.layout.n_internal == 0
        assert problem.constraint_names == ()
        assert problem.equality_at(problem.layout.initial_point([0.4, 0.8])).n_constraints == 0


# ---------------------------------------------------------------------------
# The derivatives -- against central differences
# ---------------------------------------------------------------------------

def central_difference(fun, u, step=1e-6):
    """Jacobian of ``fun`` at ``u`` by central differences, one column per coordinate."""
    base = np.atleast_1d(np.asarray(fun(u), dtype=float))
    out = np.zeros((len(base), len(u)))
    for j in range(len(u)):
        h = step * max(1.0, abs(u[j]))
        up, down = np.array(u, dtype=float), np.array(u, dtype=float)
        up[j] += h
        down[j] -= h
        out[:, j] = (np.atleast_1d(np.asarray(fun(up), dtype=float))
                     - np.atleast_1d(np.asarray(fun(down), dtype=float))) / (2.0 * h)
    return out


def perturbed(problem, base=(0.55, 0.9), knot_shift=0.12):
    """A point that is *not* the feasible seed: the knots are stale, the defects nonzero.

    Differentiating only at a feasible point would leave the whole ``-z_{j+1}`` half of the
    continuity Jacobian untested against anything but zero.
    """
    u = problem.layout.initial_point(list(base))
    u[problem.layout.n_reported:] += knot_shift
    return u


#: Both parameterisations of the reported block. The log variant is not a repeat: the
#: ``d theta/d u`` factor is applied by the gradient assembly for the objective's columns and
#: by this package for the continuity block's, and only a log-scaled parameter tells the two
#: apart (the factor is 1 for a linear one). ``loguniform_var`` is also how a rate constant is
#: normally declared, so it is the common case rather than the exotic one.
LOG_SPACE = [
    pytest.param(False, (0.55, 0.9), id='linear'),
    pytest.param(True, (np.log10(0.55), np.log10(0.9)), id='log10'),
]


class TestDerivatives:

    @pytest.mark.parametrize('log_space,base', LOG_SPACE)
    def test_objective_gradient_matches_central_differences(self, log_space, base):
        """The ``IC``-routed data term: ``d f/d u`` over the reported parameters *and* over
        every auxiliary state, against differences of the objective the fit actually scores."""
        problem, _spec, _free = build_stage(4, start_u=base, log_space=log_space)
        u = perturbed(problem, base=base)
        model = problem.objective_at(u)
        numeric = central_difference(lambda x: problem.objective_at(x).value, u)[0]
        np.testing.assert_allclose(model.gradient, numeric, rtol=2e-5, atol=1e-7)

    def test_the_least_squares_residual_reproduces_the_objective(self):
        """The invariant that makes the assembled model the fit's own: ``0.5||rho||^2`` is
        the value ``evaluate`` reports, over the segmented trajectory."""
        problem, _spec, _free = build_stage(4)
        model = problem.objective_at(perturbed(problem))
        assert model.least_squares_exact
        assert 0.5 * model.residual @ model.residual == pytest.approx(model.value)
        np.testing.assert_allclose(model.jacobian.T @ model.residual, model.gradient,
                                   rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize('log_space,base', LOG_SPACE)
    def test_constraint_jacobian_matches_central_differences(self, log_space, base):
        problem, _spec, _free = build_stage(4, start_u=base, log_space=log_space)
        u = perturbed(problem, base=base)
        model = problem.equality_at(u)
        numeric = central_difference(lambda x: problem.equality_at(x).residual, u)
        np.testing.assert_allclose(model.jacobian.to_dense(), numeric, rtol=2e-5, atol=1e-7)

    @pytest.mark.parametrize('log_space,base', LOG_SPACE)
    def test_augmented_gradient_matches_central_differences(self, log_space, base):
        """The whole augmented Lagrangian, at nonzero multipliers and a raised penalty --
        the quantity the inner solver actually steps from."""
        problem, _spec, _free = build_stage(4, start_u=base, log_space=log_space)
        u = perturbed(problem, base=base)
        multipliers = Multipliers(np.linspace(-0.4, 0.6, len(problem.constraint_names)), 25.0)
        numeric = central_difference(
            lambda x: problem.augmented_at(x, multipliers).value, u)[0]
        np.testing.assert_allclose(problem.augmented_at(u, multipliers).gradient, numeric,
                                   rtol=2e-5, atol=1e-6)

    def test_an_unobserved_state_still_carries_continuity_columns(self):
        """The motivating problem's hard corner: ``w`` has no data term, so its auxiliary
        states get *no* gradient from the objective and are determined by continuity alone.
        Both halves of that have to be true, or the fit is silently unconstrained in them."""
        problem, _spec, _free = build_stage(4)
        u = perturbed(problem)
        layout = problem.layout
        w_columns = [i for i, name in enumerate(layout.names) if name.endswith('::w')]
        objective = problem.objective_at(u)
        assert np.allclose(objective.gradient[w_columns], 0.0)
        constraint = problem.equality_at(u).jacobian.to_dense()
        assert np.all(np.abs(constraint[:, w_columns]).sum(axis=0) > 0.0)


class TestSegmentRouting:
    """A fitted initial condition affects segment 0 and no other segment, because every
    later segment's initial state was overridden by an auxiliary variable."""

    def test_an_ic_route_is_dropped_beyond_the_first_segment(self):
        problem, _spec, _free = build_stage(4)
        experiment = problem.experiments[0]
        first = problem._segment_routing(experiment, 0)
        later = problem._segment_routing(experiment, 2)
        assert [c.target for c in first.routes['y0'].contributions] == [IC]
        assert later.routes['y0'].contributions == ()
        # ...and the parameter route is untouched on both.
        assert [c.target for c in later.routes['k'].contributions] == [PARAM]

    def test_only_the_segments_own_block_routes_to_its_states(self):
        problem, _spec, _free = build_stage(4)
        routing = problem._segment_routing(problem.experiments[0], 2)
        own = [name for name in routing.routes if name.startswith('exp1@1/2::')]
        other = [name for name in routing.routes if name.startswith('exp1@1/4::')]
        assert all(routing.routes[n].contributions for n in own)
        assert all(routing.routes[n].contributions == () for n in other)

    def test_a_fitted_initial_condition_reaches_the_first_continuity_row(self):
        """Segment 0 keeps the ``IC`` route, so ``y0`` has a nonzero column in the first
        knot's continuity block and a zero one in every later knot's."""
        problem, _spec, _free = build_stage(4)
        dense = problem.equality_at(perturbed(problem)).jacobian.to_dense()
        y0 = problem.layout.reported_names.index('y0')
        rows = {name: i for i, name in enumerate(problem.constraint_names)}
        assert abs(dense[rows['exp1@1/4::y'], y0]) > 0.0
        assert dense[rows['exp1@1/2::y'], y0] == pytest.approx(0.0)
        assert dense[rows['exp1@3/4::y'], y0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

class TestCertification:

    def test_certification_scores_the_unsegmented_reconstruction(self):
        """The certificate is the ordinary single-shoot score of the reported parameters --
        computed here independently of the transcription -- and *not* the augmented
        objective, which at an infeasible point is computed on trajectories that do not
        join up."""
        times, obs = observations()
        problem, _spec, _free = build_stage(4)
        u = perturbed(problem)
        reported = problem.layout.reported_of(u)
        certificate = problem.certify(reported)
        residual = (reported[1] * np.exp(reported[0] * times) - obs) / SIGMA
        assert certificate.accepted and certificate.certified
        assert certificate.objective == pytest.approx(0.5 * residual @ residual)
        assert certificate.objective != pytest.approx(problem.objective_at(u).value)

    def test_a_reconstruction_that_does_not_simulate_is_rejected(self):
        backend = TwoStateBackend()
        problem, _spec, _free = build_stage(4, backend=backend)
        backend.fail_beyond = 0.5
        certificate = problem.certify(np.array([1.5, 0.8]))
        assert not certificate.accepted
        assert 'did not simulate' in certificate.detail


# ---------------------------------------------------------------------------
# The inner solver, and the run
# ---------------------------------------------------------------------------

class TestGaussNewtonSolver:

    def test_a_failed_segment_backs_the_search_off_rather_than_ending_the_fit(self):
        """A non-integrable point is a property of the point: the local model comes back
        non-finite, the trust region shrinks, and the solve returns an iterate."""
        backend = TwoStateBackend()
        problem, _spec, _free = build_stage(2, backend=backend)
        backend.fail_beyond = 0.45
        loop = AugmentedLagrangian(problem, GaussNewtonSolver(max_iterations=20), max_outer=4)
        result = loop.run(problem.layout.initial_point([0.4, 0.8]))
        assert np.all(np.isfinite(result.final_point))
        assert result.stop_reason in ('converged', 'stalled', 'max_outer', 'unconstrained')

    def test_the_outer_tolerance_reaches_the_inner_solver(self):
        problem, _spec, _free = build_stage(2)
        solver = GaussNewtonSolver(max_iterations=30)
        subproblem_tolerances = []

        class Recording(GaussNewtonSolver):
            def __call__(self, subproblem, u0, tolerance):
                subproblem_tolerances.append(tolerance)
                return GaussNewtonSolver.__call__(self, subproblem, u0, tolerance)

        loop = AugmentedLagrangian(problem, Recording(max_iterations=30), max_outer=3)
        loop.run(problem.layout.initial_point([0.4, 0.8]))
        assert subproblem_tolerances
        assert all(t > 0 for t in subproblem_tolerances)
        assert solver.n_evaluations == 0          # the shared counter is per instance


class TestEquivalence:
    """At convergence the constrained transcription is the uninterrupted fit.

    Measured against a single-shoot optimum computed by ``scipy.least_squares`` on the
    unsegmented residual -- an independent oracle, not this package's arithmetic run twice.
    """

    @pytest.mark.parametrize('n_segments', [2, 4])
    def test_a_converged_stage_recovers_the_single_shoot_optimum(self, n_segments):
        times, obs = observations()
        target, target_objective = single_shoot_optimum(times, obs)
        problem, _spec, _free = build_stage(n_segments, start_u=(0.55, 0.9))
        loop = AugmentedLagrangian(problem, GaussNewtonSolver(max_iterations=60),
                                   schedule=PenaltySchedule(), max_outer=25)
        result = loop.run(problem.layout.initial_point([0.55, 0.9]))
        assert result.best is not None
        np.testing.assert_allclose(result.best.reported, target, rtol=1e-4, atol=1e-5)
        assert result.best_score == pytest.approx(target_objective, rel=1e-6)

    def test_a_log_scaled_fit_recovers_the_same_optimum(self):
        """The realistic parameterisation -- a rate constant is normally a
        ``loguniform_var`` -- so the search coordinate is ``log10(theta)`` and every column
        of both the objective Jacobian and the continuity block carries a chain factor."""
        times, obs = observations()
        target, target_objective = single_shoot_optimum(times, obs)
        base = (np.log10(0.55), np.log10(0.9))
        problem, _spec, _free = build_stage(4, start_u=base, log_space=True)
        loop = AugmentedLagrangian(problem, GaussNewtonSolver(max_iterations=60), max_outer=25)
        result = loop.run(problem.layout.initial_point(list(base)))
        assert result.best is not None
        np.testing.assert_allclose(10.0 ** result.best.reported, target, rtol=1e-4, atol=1e-5)
        assert result.best_score == pytest.approx(target_objective, rel=1e-6)

    def test_it_converges_from_a_start_whose_knots_are_stale(self):
        """The realistic case: seeding knots from a nominal trajectory makes the
        transcription feasible at iteration zero, which is *not* the state a fit is in after
        theta has moved."""
        times, obs = observations()
        target, _objective = single_shoot_optimum(times, obs)
        problem, _spec, _free = build_stage(4, start_u=(0.55, 0.9))
        u = perturbed(problem, base=(0.55, 0.9), knot_shift=0.8)
        loop = AugmentedLagrangian(problem, GaussNewtonSolver(max_iterations=60), max_outer=25)
        result = loop.run(u)
        assert result.best is not None
        np.testing.assert_allclose(result.best.reported, target, rtol=1e-3, atol=1e-4)


class TestLadder:

    def test_the_ladder_runs_and_reports_its_best_certified_iterate(self):
        times, obs = observations()
        target, target_objective = single_shoot_optimum(times, obs)
        free = variables()
        spec = make_spec(times, obs)
        result = run_multiple_shooting([spec], ChiSquareObjective(), free,
                                       pset_from_u_for(free), np.array([0.55, 0.9]))
        assert [stage.name for stage in result.stages] == ['m=4', 'm=2', 'm=1']
        assert result.certified
        np.testing.assert_allclose(result.reported, target, rtol=1e-3, atol=1e-4)
        assert result.best_score <= target_objective * (1 + 1e-6)
        assert 'm=4' in result.trace() and 'm=1' in result.trace()

    def test_a_rung_the_data_cannot_support_is_dropped_and_reported(self):
        """No silent caps: a segment count above the experiment's own measurement count
        would leave every segment determined by continuity alone."""
        times, obs = observations(n=3)
        spec = make_spec(times, obs)
        rungs, dropped = feasible_ladder([spec], ladder=(8, 4, 2, 1))
        assert rungs == (2, 1)
        assert dropped == (8, 4)

    def test_the_ladder_always_ends_unsegmented(self):
        times, obs = observations(n=2)
        spec = make_spec(times, obs)
        rungs, _dropped = feasible_ladder([spec], ladder=(8,))
        assert rungs[-1] == 1


# ---------------------------------------------------------------------------
# The segment pass: serial, or across lanes (#563, "segment simulations can run in
# parallel")
# ---------------------------------------------------------------------------

class TestSegmentPool:

    def test_lanes_change_when_a_segment_runs_and_nothing_else(self):
        """The claim the scheduler has to earn: the objective, its gradient, the continuity
        residual and its Jacobian come back **exactly** equal, not merely close.

        Exactly, because a segment pass is not an approximation of another segment pass --
        the same spans are integrated from the same states, only on different threads. A
        tolerance here would hide a lane mix-up, which is the failure this parallelisation
        can actually have.
        """
        serial, _spec, _free = build_stage(4, backend=TwoStateBackend(n_lanes=4))
        u = _stale(serial, serial.layout.initial_point([0.45, 0.85]))
        want = (serial.objective_at(u), serial.equality_at(u))

        pool = SegmentPool(4)
        try:
            parallel, _s, _f = build_stage(4, backend=TwoStateBackend(n_lanes=4), pool=pool)
            got = (parallel.objective_at(u), parallel.equality_at(u))
        finally:
            pool.close()

        assert got[0].value == want[0].value
        np.testing.assert_array_equal(got[0].gradient, want[0].gradient)
        np.testing.assert_array_equal(got[1].residual, want[1].residual)
        np.testing.assert_array_equal(got[1].jacobian.to_dense(), want[1].jacobian.to_dense())

    def test_two_segments_never_share_a_lane_at_once(self):
        """A lane is a stateful simulator restarted at its segment's knot, so two segments
        in one lane at one time is silent corruption rather than an error. With more
        segments than lanes the scheduler must make a segment *wait*."""
        backend = TwoStateBackend(n_lanes=2)
        pool = SegmentPool(4)
        try:
            problem, _spec, _free = build_stage(8, backend=backend, pool=pool)
            problem.objective_at(problem.layout.initial_point([0.45, 0.85]))
        finally:
            pool.close()
        assert backend.collisions == 0
        assert backend.max_concurrent <= 2      # capped by what the backend offered

    def test_a_parallel_pass_really_overlaps(self):
        """The whole point, asserted rather than assumed -- and deterministically.

        A barrier of four is the proof: all four segments of one point must be inside
        :meth:`simulate` at the same moment for any of them to return. A pass that ran them
        one at a time would block there and fail on the barrier's own timeout, not on a
        timing heuristic that happens to be true on a fast machine.
        """
        backend = TwoStateBackend(n_lanes=4)
        pool = SegmentPool(4)
        try:
            problem, _spec, _free = build_stage(4, backend=backend, pool=pool)
            # Armed only after seeding, which runs one unsegmented simulation on the calling
            # thread -- a party of four would have nothing to wait with.
            backend.barrier = threading.Barrier(4, timeout=30)
            problem.objective_at(problem.layout.initial_point([0.45, 0.85]))
        finally:
            pool.close()
        assert backend.max_concurrent == 4

    def test_a_failed_segment_still_makes_the_whole_point_unusable(self):
        """Parallel gives up the serial pass's short-circuit, not its verdict."""
        backend = TwoStateBackend(n_lanes=4)
        backend.fail_beyond = 0.5
        pool = SegmentPool(4)
        try:
            problem, _spec, _free = build_stage(4, backend=backend, pool=pool)
            model = problem.objective_at(problem.layout.initial_point([1.5, 0.85]))
        finally:
            pool.close()
        assert not np.isfinite(model.value)

    def test_a_serial_pool_opens_no_threads(self):
        pool = SegmentPool(1)
        assert not pool.parallel
        problem, _spec, _free = build_stage(4, pool=pool)
        problem.objective_at(problem.layout.initial_point([0.45, 0.85]))
        assert pool._executor is None
        pool.close()

    def test_a_backend_that_offers_one_lane_is_still_correct(self):
        """The default :meth:`SegmentBackend.open_lanes` says "one", and a pool asked for
        more must honour that rather than run two segments in the one context."""
        backend = TwoStateBackend(n_lanes=1)
        pool = SegmentPool(4)
        try:
            problem, _spec, _free = build_stage(4, backend=backend, pool=pool)
            model = problem.objective_at(problem.layout.initial_point([0.45, 0.85]))
        finally:
            pool.close()
        assert np.isfinite(model.value)
        assert backend.collisions == 0
        assert backend.max_concurrent == 1


def _stale(problem, u, factor=1.6):
    """Knots pushed off the seeded (feasible) trajectory, so the defects are nonzero."""
    u = np.array(u, dtype=float, copy=True)
    for block in problem.layout.blocks:
        u[problem.layout.slice_of(block.name)] += np.log10(factor)
    return u
