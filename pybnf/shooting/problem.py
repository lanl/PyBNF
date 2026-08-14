"""The multiple-shooting transcription: #563's first consumer of ADR-0109's layer.

:class:`MultipleShootingProblem` implements the two abstract methods
:class:`~pybnf.transcription.augmented.TranscriptionProblem` declares, plus ``certify``,
and gets the augmented-Lagrangian outer loop, the penalty schedule, the homotopy, the
best-iterate certification, and the reporting for free. Everything specific to multiple
shooting is here:

* which data point belongs to which segment (:mod:`pybnf.shooting.grid`);
* how a segment is simulated from a state the transcription supplies
  (:mod:`pybnf.shooting.backend`);
* how the fit's own objective is assembled over the *pieces*; and
* the continuity constraints ``c_j = Phi_j(z_j, theta) - z_{j+1}`` and their Jacobian.

The objective half needs no new residual math
----------------------------------------------
This is the structural finding the #563 prototype established, and it is what keeps this
module small. A segment-start state enters the data fit as an ``IC`` route with chain-rule
factor **1** -- ``sensitivity_ic`` is ``dy(t)/dy0``, and the transcription sets ``y0``
directly -- so an auxiliary variable is, to
:func:`~pybnf.gradient.assembly.assemble_gradient_and_fisher_hessian`, an ordinary free
parameter with an ordinary route. Each segment is presented to the assembly as its own
*experiment*: its own simulated ``Data``, its own slice of the observations, and its own
:class:`~pybnf.gradient.routing.ExperimentRouting`. The assembly already sums across
experiments, so the segmented data fit is the unsegmented one rearranged, and its gradient
and Fisher block come out over the augmented column list in one pass.

Two things that rearrangement has to get right, and both are structural rather than
cosmetic:

**Segment ``j > 0`` does not read the model's own initial conditions.** They were
overridden by ``z_j``. A reported free parameter that is a fitted initial condition
therefore has *no* effect on that segment, and its ``IC`` contribution is dropped from that
segment's routing. Keeping it would credit ``init_Z_state`` with a derivative it does not
have on ``m - 1`` of the ``m`` segments -- a wrong column the fit cannot detect from its own
objective. Segment 0 keeps it, which is exactly how a fitted initial condition reaches the
first continuity row.

**A quantity profiled or normalised across a whole series cannot be cut.** An analytic
per-series scale (ADR-0066) and a ``Data``-level normalization (ADR-0053/0102) are functions
of the series they are computed over, so splitting the series changes them; a
cumulative-to-incident transform (ADR-0051) is a difference between neighbouring rows, and
the row before a knot is in another piece. Those are refused by the ``ms`` fit type's gates
rather than silently rearranged. An analytically profiled **noise scale** (ADR-0108) is the
opposite case and needs no gate: it is profiled over the pooled residuals of every supplied
experiment, so cutting one series into ``m`` pieces pools exactly the same residuals and
gives exactly the same ``sigma_hat``.

The constraint terms never enter ``f``
---------------------------------------
:attr:`~pybnf.transcription.augmented.AugmentedModel.objective_value` is the fit's own
objective, scored by the fit's own objective function on the segmented trajectory. The
penalty lives strictly outside it (ADR-0109), which is what keeps an estimated noise scale
from absorbing continuity violation as measurement noise -- and 13 of the 23 slugs in the
motivating benchmark corpus estimate at least one.
"""

import numpy as np

from ..data import Data
from ..gradient import IC, NONE, PARAM, ExperimentRouting, ParamRoute
from ..gradient.assembly import assemble_gradient_and_fisher_hessian
from ..printing import PybnfError
from ..pset import FreeParameter
from ..transcription import (
    AugmentedLayout,
    BlockJacobian,
    Certificate,
    EqualityModel,
    EqualitySystem,
    JacobianBlock,
    ObjectiveModel,
    TranscriptionProblem,
    VariableBlock,
)
from .backend import SegmentSimulationFailed
from .grid import EQUAL_TIME, SegmentGrid, max_segments
from .parallel import SERIAL, SegmentTask

#: Floor under a state magnitude, so a species that is identically zero over the whole
#: horizon still gets a strictly positive constraint scale and a finite auxiliary box. The
#: layer requires strictly positive scales precisely so this decision is made once, here,
#: rather than being discovered as a divide-by-zero inside the penalty term.
STATE_FLOOR = 1e-12

#: Half-width, in decades, of an auxiliary state's box around its own magnitude. Wide
#: enough that the box is not a constraint on the search (the #563 prototype used a fixed
#: ``1e-6 .. 1e3`` window and never reported a knot pinned at a bound), finite because the
#: inner optimizers this layer feeds are bound-constrained and a segment-start
#: concentration that is allowed to go negative is not a state the simulator can restart
#: from.
AUX_DECADES = 6.0


class ShootingExperiment:
    """One scored ``(model, condition)`` pair, at the fit level: everything about it that
    does **not** depend on the segment count.

    Built once per fit and shared by every rung of the ladder, which is what makes a stage
    cheap to construct: a homotopy builds one :class:`SegmentedExperiment` per rung around
    the same backend, observations, and routing.

    :param key: ``(model name, suffix)`` -- the identity this experiment has in the
        simulated/experimental data dictionaries the objective consumes.
    :param backend: Its :class:`~pybnf.shooting.backend.SegmentBackend`.
    :param exp_data: Its observations, as an ordinary PyBNF :class:`~pybnf.data.Data`.
    :param routing: The experiment's prebuilt
        :class:`~pybnf.gradient.routing.ExperimentRouting` over the **reported** free
        parameters, exactly as the gradient optimizers build it once per fit.
    :param times: Its measurement times. Defaults to ``exp_data``'s independent-variable
        column.
    :param label: The label its knots are named under. Defaults to the suffix.
    :param start: Where the simulation starts. Defaults to the first measurement time; a
        time course whose first measurement is after ``t = 0`` should pass ``0.0``.
    :param placement: How this experiment's knots are placed -- one of
        :data:`~pybnf.shooting.grid.PLACEMENTS`. Fixed for the whole fit, so every rung of
        the ladder places its knots by the same rule and their fractions mean one thing.
    :param knots: Explicit knot times for the finest rung, which force
        ``placement = 'explicit'``.
    """

    def __init__(self, key, backend, exp_data, routing, times=None, label=None, start=None,
                 horizon=None, placement=EQUAL_TIME, knots=None):
        self.key = (str(key[0]), str(key[1]))
        self.backend = backend
        self.exp_data = exp_data
        self.routing = routing
        if times is None:
            times = np.asarray(exp_data.data, dtype=float)[:, exp_data.cols[exp_data.indvar]]
        self.times = np.asarray(times, dtype=float).reshape(-1)
        self.label = str(label if label is not None else self.key[1])
        self.start = start
        self.horizon = horizon
        self.placement = str(placement)
        self.knots = tuple(float(t) for t in (knots or ()))

    def grid(self, n_segments):
        """This experiment's knots at one segment count."""
        return SegmentGrid(self.times, n_segments, label=self.label, start=self.start,
                           horizon=self.horizon, placement=self.placement,
                           knots=self.knots or None)

    @property
    def max_segments(self):
        """The finest rung this experiment's data supports under its own placement."""
        return max_segments(self.times, self.placement, self.knots or None)

    @property
    def state_names(self):
        return tuple(self.backend.state_names)


class SegmentedExperiment:
    """One :class:`ShootingExperiment` cut at the knots of one stage."""

    def __init__(self, spec, grid):
        self.spec = spec
        self.key = spec.key
        self.backend = spec.backend
        self.exp_data = spec.exp_data
        self.routing = spec.routing
        self.grid = grid
        #: The ``m = 1`` grid of the same experiment -- what :meth:`certify` simulates.
        #: Not a special case: the unsegmented problem is the one-segment transcription.
        self.certification_grid = spec.grid(1)
        self._subsets = {}

    @property
    def state_names(self):
        return tuple(self.backend.state_names)

    def segment_key(self, segment):
        """The dictionary key one segment is scored under.

        A synthetic suffix, distinct per segment, because the objective scores a
        ``{model: {suffix: Data}}`` mapping and the ``m`` pieces of one experiment must be
        ``m`` entries rather than one that overwrites the others.
        """
        return '%s#seg%i' % (self.key[1], segment)

    def segment_exp_data(self, segment):
        """This experiment's observations restricted to one segment's rows.

        Cached: the slice depends only on the grid, never on the fit point, so it is built
        once per stage rather than once per evaluation.
        """
        segment = int(segment)
        if segment not in self._subsets:
            self._subsets[segment] = _subset_rows(self.exp_data, self.grid.rows_in(segment))
        return self._subsets[segment]

    def has_data(self, segment):
        return len(self.grid.rows_in(segment)) > 0


class MultipleShootingProblem(TranscriptionProblem, EqualitySystem):
    """One rung of the segment ladder: the fit, transcribed at a fixed segment count.

    :param experiments: The :class:`SegmentedExperiment`\\ s, all at the same segment
        count.
    :param objective: The fit's own objective function.
    :param variables: The fit's reported free parameters, in ``Configuration.variables``
        order -- the same order every existing PyBNF seam uses.
    :param pset_from_u: ``(u_reported) -> PSet``: the algorithm's own sampling-space to
        parameter-set bridge (:meth:`~pybnf.algorithms.base.Algorithm._pset_from_u`), so
        this module never re-derives the ``theta <-> u`` transform.
    :param blocks: The auxiliary :class:`~pybnf.transcription.layout.VariableBlock`\\ s,
        one per knot of every experiment, already seeded (see :func:`seed_stage`).
    :param scales: ``{experiment key: per-state constraint scale}`` -- the magnitude each
        continuity defect is measured against.
    :param name: The stage label for the trace (``'m=4'``).
    :param pool: The :class:`~pybnf.shooting.parallel.SegmentPool` that runs this stage's
        segment passes. Defaults to the serial one, so a caller that does not ask for lanes
        gets exactly the behaviour that shipped with ADR-0110.
    """

    def __init__(self, experiments, objective, variables, pset_from_u, blocks, scales,
                 name=None, pool=None):
        self.experiments = list(experiments)
        self._pool = pool or SERIAL
        self.objective = objective
        self.variables = list(variables)
        self._pset_from_u = pset_from_u
        self._scales = dict(scales)
        self._name = name or ('m=%i' % (self.experiments[0].grid.n_segments
                                        if self.experiments else 1))

        lower = np.array([v.to_sampling_space(v.lower_bound) if v.bounded else -np.inf
                          for v in self.variables], dtype=float)
        upper = np.array([v.to_sampling_space(v.upper_bound) if v.bounded else np.inf
                          for v in self.variables], dtype=float)
        self._layout = AugmentedLayout([v.name for v in self.variables], lower, upper, blocks)

        # Constraint identity, fixed for the stage: n_state rows per knot, in the layout's
        # own block order, so a Jacobian block's row range is a plain slice.
        names, rows_of = [], {}
        for experiment in self.experiments:
            for block_name in experiment.grid.block_names:
                rows_of[block_name] = slice(len(names), len(names) + len(experiment.state_names))
                names.extend('%s::%s' % (block_name, state)
                             for state in experiment.state_names)
        self._constraint_names = tuple(names)
        self._constraint_rows = rows_of
        self._constraint_scales = np.array(
            [self._scales[experiment.key][i]
             for experiment in self.experiments
             for _block in experiment.grid.block_names
             for i in range(len(experiment.state_names))], dtype=float)

        self._cache_key = None
        self._cache = None
        # The assembled objective at the last point asked for. Separate from the trace cache
        # above because it must survive an interleaved `equality_at` at another point, and
        # because it is what guarantees each simulated trajectory is scored once (#578; see
        # :meth:`objective_at`).
        self._objective_key = None
        self._objective_model = None
        self.n_certifications = 0

    # -- identity ---------------------------------------------------------------

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return self._name

    @property
    def constraint_names(self):
        return self._constraint_names

    def describe(self):
        """The stage in one line: what it added, and where."""
        return '%s -- %s; %s' % (self._name, self._layout.describe(),
                                 '; '.join(e.grid.describe() for e in self.experiments))

    # -- the segment pass -------------------------------------------------------

    def _knot_state(self, u, experiment, segment):
        """Segment ``segment``'s start state as ``{state name: value}``, or ``None`` for
        segment 0 (which starts from the model's own initial conditions)."""
        if segment == 0:
            return None
        block = experiment.grid.block_names[segment - 1]
        values = 10.0 ** self._layout.internal_of(u, block)
        return dict(zip(experiment.state_names, values))

    def _traces(self, u):
        """Every segment of every experiment, simulated once at ``u``.

        Cached on the point, because the outer loop asks for the objective and the
        constraints at the *same* point (``AugmentedSubproblem.at`` linearises both
        together) and a segment simulation is the expensive thing here. ``None`` when any
        segment failed to integrate: the caller turns that into a non-finite local model,
        which is the signal the inner solver's trust region backs off on.

        The pass itself belongs to :class:`~pybnf.shooting.parallel.SegmentPool` -- the
        segments of one point are independent (each is one span from a state the
        transcription already knows, with no data flowing between them), so whether they run
        one at a time or several at once is a scheduling decision and not a property of the
        transcription. This method's job is to say *which* spans, from *which* states.
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        key = u.tobytes()
        if self._cache_key == key:
            return self._cache
        pset = self._pset_from_u(self._layout.reported_of(u))
        tasks, shape = [], []
        for experiment in self.experiments:
            shape.append(experiment.grid.n_segments)
            for segment in range(experiment.grid.n_segments):
                times, _rows = experiment.grid.sample_times(segment)
                tasks.append(SegmentTask(experiment.backend, times,
                                         self._knot_state(u, experiment, segment),
                                         experiment.state_names))
        flat, ok = self._pool.run(pset, tasks)
        self._cache_key = key
        self._cache = (pset, _regroup(flat, shape)) if ok else None
        return self._cache

    def _free_params(self, u, pset):
        """The augmented free-parameter list, in layout order.

        The reported block is read straight off the evaluated PSet -- so every
        ``d theta/d u`` factor is taken at the point actually simulated, exactly as
        :meth:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer.gradient_at`
        does -- and each auxiliary coordinate is a ``loguniform_var``
        :class:`~pybnf.pset.FreeParameter` carrying its own box. Presenting an auxiliary
        state as a genuine free parameter is what lets the assembly treat it as one: it
        supplies the ``ln(10) * 10**u`` chain factor from the same
        :mod:`pybnf.priors.scale` code the reported parameters use, rather than from a
        second implementation of log10.
        """
        params = [pset.get_param(v.name) for v in self.variables]
        for block in self._layout.blocks:
            values = self._layout.internal_of(u, block.name)
            for name, lo, hi, value in zip(block.qualified_names, block.lower, block.upper,
                                           values):
                params.append(FreeParameter(name, 'loguniform_var', 10.0 ** lo, 10.0 ** hi,
                                            value=10.0 ** value))
        return params

    # -- the objective ----------------------------------------------------------

    def objective_at(self, u):
        """The fit's objective and its derivatives at ``u``, memoised on the point.

        The memo is not an optimisation first — it is what keeps this method **scoring each
        simulated trajectory exactly once** (#578). Scoring goes through
        ``objective.evaluate_multiple``, which asks the measurement layer to materialise
        every ``observable: <id>, formula: ...`` column *into the trajectory in place*
        (ADR-0036), and that materialisation deliberately refuses a column that already
        exists. Every ordinary fit satisfies it for free: the propose/score loop scores a
        freshly simulated :class:`~pybnf.data.Data` each time. Multiple shooting does not,
        because :meth:`_traces` caches the segment trajectories per point so that one
        augmented-model evaluation costs one pass of segment simulations rather than two --
        and the outer loop then re-evaluates at the point the inner solver finished at, which
        is a cache hit on the very same objects.

        Memoising the assembled model rather than copying the trajectories fixes that at the
        cause: the second call returns the first call's answer instead of re-scoring
        anything. It also removes a redundant gradient/Fisher assembly per outer iteration,
        which is the larger of the two costs. Sound because this method does not depend on
        the multipliers -- only
        :class:`~pybnf.transcription.augmented.AugmentedModel` combines them -- so the
        objective at a point is a property of the point alone.
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        key = u.tobytes()
        if self._objective_key == key:
            return self._objective_model
        model = self._build_objective(u)
        self._objective_key, self._objective_model = key, model
        return model

    def _build_objective(self, u):
        cached = self._traces(u)
        if cached is None:
            return self._unusable_objective()
        pset, traces = cached
        sim_dict, exp_dict, items = {}, {}, []
        for experiment, per_segment in zip(self.experiments, traces):
            model_name = experiment.key[0]
            for segment, trace in enumerate(per_segment):
                if not experiment.has_data(segment):
                    # A segment with no observations contributes no residual; its
                    # auxiliary state is determined by continuity alone, which is the
                    # normal case for an unobserved species and the corner the constraint
                    # scaling exists for.
                    continue
                key = experiment.segment_key(segment)
                sim_dict.setdefault(model_name, {})[key] = trace.data
                exp_dict.setdefault(model_name, {})[key] = experiment.segment_exp_data(segment)
                items.append((trace.data, experiment.segment_exp_data(segment),
                              self._segment_routing(experiment, segment), key))
        # Score first, assemble second -- the same order the gradient optimizers use. The
        # measurement layer materialises its expression columns during scoring, and the
        # assembly reads those columns.
        value = self.objective.evaluate_multiple(sim_dict, exp_dict, pset, show_warnings=False)
        if value is None or not np.isfinite(value):
            return self._unusable_objective()
        grad = assemble_gradient_and_fisher_hessian(
            self.objective, items, self._free_params(u, pset))
        return ObjectiveModel.from_gradient_result(value, grad)

    def _unusable_objective(self):
        """The local model at a point that did not simulate, or did not score.

        Non-finite by construction, which is the signal
        :meth:`~pybnf.transcription.augmented.AugmentedModel.is_finite` carries to the
        inner solver: shrink the trust region and try a shorter step, rather than end the
        run. A segment that fails is a property of the point.
        """
        return ObjectiveModel(np.inf, np.full(self._layout.size, np.nan))

    def _segment_routing(self, experiment, segment):
        """This segment's routing over the **augmented** free-parameter list.

        Three populations, and the whole of the ``IC``-route finding is in the second:

        * every reported free parameter keeps its route -- minus, for ``segment > 0``, any
          ``IC`` contribution to a state this segment overrode (see the module docstring);
        * this segment's own auxiliary block routes to ``IC`` on its states with factor 1;
        * every other auxiliary block gets an empty route, because this segment's
          trajectory does not depend on another segment's start state. An empty route is a
          structural zero column, not an omission: the assembly indexes routes by name into
          the augmented column list, so a block that is absent from the routing would leave
          an uninitialised column rather than a zero one.
        """
        overridden = set(experiment.state_names) if segment > 0 else set()
        routes = {}
        for name, route in experiment.routing.routes.items():
            kept = tuple(c for c in route.contributions
                         if not (c.target == IC and c.key in overridden))
            routes[name] = ParamRoute(free_param=name, contributions=kept)
        own = (experiment.grid.block_names[segment - 1] if segment > 0 else None)
        for block in self._layout.blocks:
            states = (experiment.state_names if block.name == own else ())
            for qualified, state in zip(block.qualified_names, block.labels):
                routes[qualified] = (
                    ParamRoute.single(qualified, IC, state, 1.0) if state in states
                    else ParamRoute(free_param=qualified, contributions=()))
        return ExperimentRouting(routes=routes,
                                 nominal_values=experiment.routing.nominal_values,
                                 condition=experiment.routing.condition)

    # -- the constraints --------------------------------------------------------

    def equality_at(self, u):
        if not self._constraint_names:
            return self.empty_model()
        cached = self._traces(u)
        if cached is None:
            return self._unusable_equality()
        _pset, traces = cached
        u = np.asarray(u, dtype=float).reshape(-1)   # this method indexes u by block slice
        m = len(self._constraint_names)
        residual = np.zeros(m)
        blocks = []
        reported = slice(0, self._layout.n_reported)
        factors = self._reported_scale_factors(u)
        for experiment, per_segment in zip(self.experiments, traces):
            n_state = len(experiment.state_names)
            for knot, block_name in enumerate(experiment.grid.block_names):
                segment = knot                      # segment `knot` ends at knot `knot + 1`
                trace = per_segment[segment]
                rows = self._constraint_rows[block_name]
                next_slice = self._layout.slice_of(block_name)
                z_next = 10.0 ** u[next_slice]
                residual[rows] = trace.end_state - z_next

                # d(end state)/d(reported parameters), folded from this segment's routing
                # exactly as the objective's columns are, then taken into sampling space.
                # The same routing the data rows of this segment were assembled through, so
                # a fitted initial condition contributes here on segment 0 and nowhere else.
                routing = self._segment_routing(experiment, segment)
                d_reported = np.zeros((n_state, self._layout.n_reported))
                for j, v in enumerate(self.variables):
                    d_reported[:, j] = _fold_route(routing.routes[v.name], trace)
                blocks.append(JacobianBlock(rows, reported, d_reported * factors[np.newaxis, :]))

                # d(end state)/d(this segment's own start state) -- the IC route, chain-ruled
                # through the auxiliary variable's own log10 space.
                if segment > 0:
                    own = experiment.grid.block_names[segment - 1]
                    own_slice = self._layout.slice_of(own)
                    z_own = 10.0 ** u[own_slice]
                    columns = _ic_columns(trace, experiment.state_names)
                    blocks.append(JacobianBlock(
                        rows, own_slice, columns * (np.log(10.0) * z_own)[np.newaxis, :]))

                # d(-z_{j+1})/d(its own coordinates): the constant -I of the transcription,
                # in the auxiliary variable's log10 space.
                blocks.append(JacobianBlock(
                    rows, next_slice, -np.diag(np.log(10.0) * z_next)))

        return EqualityModel(residual, BlockJacobian((m, self._layout.size), blocks),
                             scales=self._constraint_scales, names=self._constraint_names)

    def _unusable_equality(self):
        """The constraint linearisation at a point that did not simulate: non-finite, so
        the augmented model is, so the inner solver backs off."""
        m = len(self._constraint_names)
        return EqualityModel(np.full(m, np.inf),
                             BlockJacobian((m, self._layout.size), ()),
                             scales=self._constraint_scales, names=self._constraint_names)

    def _reported_scale_factors(self, u):
        """``d theta/d u`` for the reported block at ``u``.

        The constraint rows are in the model's own state units but their columns are in the
        space the optimizer walks, so the reported half of every continuity block takes the
        same chain factor the objective's Jacobian does.
        """
        reported = self._layout.reported_of(u)
        return np.array([v.d_from_sampling_space(reported[j]) if v.log_space else 1.0
                         for j, v in enumerate(self.variables)], dtype=float)

    # -- certification ----------------------------------------------------------

    def certify(self, reported):
        """Reconstruct ``reported`` through the fit's ordinary unsegmented path.

        Every auxiliary state is discarded and each experiment is simulated once over its
        whole horizon from the model's own initial conditions -- which *is* the single-shoot
        problem -- and scored by the fit's own objective. That score is the only one
        comparable with an ordinary PyBNF fit's, and under ADR-0109 it is the only one this
        run may report: the augmented objective at an infeasible point is computed on
        trajectories that do not join up.
        """
        self.n_certifications += 1
        pset = self._pset_from_u(np.asarray(reported, dtype=float))
        sim_dict, exp_dict = {}, {}
        for experiment in self.experiments:
            model_name, suffix = experiment.key
            times, _rows = experiment.certification_grid.sample_times(0)
            try:
                data = experiment.backend.simulate(pset, times, None)
            except SegmentSimulationFailed as exc:
                return Certificate.reject('the reconstructed trajectory did not '
                                          'simulate (%s)' % exc)
            sim_dict.setdefault(model_name, {})[suffix] = data
            exp_dict.setdefault(model_name, {})[suffix] = experiment.exp_data
        value = self.objective.evaluate_multiple(sim_dict, exp_dict, pset, show_warnings=False)
        if value is None or not np.isfinite(value):
            return Certificate.reject('the reconstructed trajectory scored non-finite')
        return Certificate.accept(value, detail='single-shoot reconstruction')


# ---------------------------------------------------------------------------
# Stage construction
# ---------------------------------------------------------------------------

def seed_stage(specs, n_segments, objective, variables, pset_from_u, reported,
               aux_decades=AUX_DECADES, name=None, pool=None):
    """Build one rung of the ladder, with its auxiliary states seeded from ``reported``.

    Each experiment is simulated **once**, unsegmented, at the incoming parameters, and the
    state is read off at each knot. Seeding this way makes the transcription feasible at
    iteration zero: every continuity defect is exactly zero at the start point, so any
    discontinuity the run subsequently holds is the optimizer's own choice rather than an
    artifact of how the stage was built. It is also why a homotopy stage is a *callable*
    in :func:`~pybnf.transcription.homotopy.run_homotopy` -- the seeds are not knowable
    until the previous stage has finished moving ``theta``.

    A start point that does not simulate is not a failure: the knots fall back to the
    model's own declared state magnitudes, the first inner solve sees a large defect, and
    the run proceeds (or stops with a stated reason) rather than dying at construction.

    ``aux_decades`` sets each auxiliary variable's box: ``+/- aux_decades`` around the
    state's own magnitude, in log10. Wide enough not to constrain the search, finite
    because the inner optimizers are bound-constrained and a segment-start concentration
    that is allowed to go negative is not a state a simulator can restart from.
    """
    reported = np.asarray(reported, dtype=float).reshape(-1)
    pset = pset_from_u(reported)
    staged, blocks, scales = [], [], {}
    for spec in specs:
        experiment = SegmentedExperiment(spec, spec.grid(n_segments))
        nominal = np.maximum(np.asarray(spec.backend.nominal_state, dtype=float), STATE_FLOOR)
        seeds, magnitude = _seed_knots(experiment, pset, nominal)
        scales[experiment.key] = magnitude
        centre = np.log10(magnitude)
        lower, upper = centre - aux_decades, centre + aux_decades
        for block_name, seed in zip(experiment.grid.block_names, seeds):
            initial = np.clip(np.log10(np.maximum(seed, STATE_FLOOR)), lower, upper)
            blocks.append(VariableBlock(block_name, experiment.state_names, lower, upper,
                                        initial))
        staged.append(experiment)
    return MultipleShootingProblem(staged, objective, variables, pset_from_u, blocks, scales,
                                   name=name or ('m=%i' % n_segments), pool=pool)


def _seed_knots(experiment, pset, nominal):
    """``(per-knot seed states, per-state magnitude)`` from one unsegmented simulation.

    The magnitude is the larger of the model's declared nominal and the largest value the
    seeding trajectory actually reaches, per state. Declared nominals alone understate a
    species that starts empty and grows -- which on an oscillator is most of them -- and a
    constraint scale that understates its state hands the penalty term a condition number
    for free, which is the corner ADR-0109's scaling exists to close.
    """
    grid = experiment.grid
    if grid.n_knots == 0:
        return [], nominal
    try:
        data = experiment.backend.simulate(pset, grid.seed_times(), None)
    except SegmentSimulationFailed:
        return [nominal.copy() for _ in range(grid.n_knots)], nominal
    columns = [data.cols[name] for name in experiment.state_names]
    trajectory = np.abs(np.asarray(data.data, dtype=float)[:, columns])
    if not np.all(np.isfinite(trajectory)):
        return [nominal.copy() for _ in range(grid.n_knots)], nominal
    times = np.asarray(data.data, dtype=float)[:, data.cols[data.indvar]]
    seeds = []
    for knot_time in grid.knot_times:
        row = int(np.argmin(np.abs(times - knot_time)))
        seeds.append(np.asarray(data.data, dtype=float)[row, columns])
    magnitude = np.maximum(nominal, np.max(trajectory, axis=0))
    return seeds, np.maximum(magnitude, STATE_FLOOR)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _regroup(flat, shape):
    """A flat per-segment list back into one list per experiment."""
    out, start = [], 0
    for n in shape:
        out.append(flat[start:start + n])
        start += n
    return out


def _fold_route(route, trace):
    """``d(end state)/d(one free parameter)``: the route's contributions, summed.

    The same fold :func:`pybnf.gradient.assembly._raw_sensitivity_accessor` performs for a
    scored column, applied to the end-knot row instead -- one place, so a condition's
    chain-rule factor and a multi-column route reach the continuity block exactly as they
    reach the data rows. A :data:`~pybnf.gradient.routing.NONE` term (a free noise scale)
    and a pinned zero-factor term contribute nothing, by the same test the assembly uses.
    """
    total = np.zeros(len(trace.end_state))
    for contribution in route.contributions:
        if contribution.target == NONE or contribution.factor == 0.0:
            continue
        if contribution.target == PARAM:
            axis, tensor = trace.param_axis, trace.d_end_param
        else:
            axis, tensor = trace.ic_axis, trace.d_end_ic
        if tensor is None or contribution.key not in axis:
            raise PybnfError(
                "Multiple shooting needs the %s sensitivity column '%s' at a segment's end "
                "knot, and the segment simulation did not return it. The continuity "
                "Jacobian is built from the same forward-sensitivity request the data rows "
                "use, so a missing column is an internal wiring error."
                % (contribution.target, contribution.key))
        total = total + contribution.factor * tensor[:, axis.index(contribution.key)]
    return total


def _ic_columns(trace, state_names):
    """``d(end state)/d(start state)`` as an ``(n_state, n_state)`` block.

    The ``IC`` route with chain-rule factor 1, read off the tensor the segment already
    returned: column ``s`` is ``d(end state)/d(z_s)``. Reordered to ``state_names`` so the
    block's columns line up with the auxiliary block's own component order rather than with
    whatever order the backend requested its sensitivity axis in.
    """
    if trace.d_end_ic is None:
        raise PybnfError(
            'Multiple shooting needs initial-condition sensitivities at a segment end knot '
            '(the continuity Jacobian is d(end state)/d(start state)), and the segment '
            'simulation returned none.')
    missing = [name for name in state_names if name not in trace.ic_axis]
    if missing:
        raise PybnfError(
            'A multiple-shooting segment simulation returned no initial-condition '
            'sensitivity axis for state(s) %s.' % ', '.join(sorted(missing)))
    return trace.d_end_ic[:, [trace.ic_axis.index(name) for name in state_names]]


def _subset_rows(data, rows):
    """One :class:`~pybnf.data.Data` restricted to a row subset, weights included.

    Bootstrap weights are per-point (``sqrt(w)``-folded into both the residual and the
    Jacobian), so a segment must carry its own rows' weights or a bootstrap refit would
    silently reweight the fit it is resampling.
    """
    rows = np.asarray(rows, dtype=int)
    out = Data(arr=np.asarray(data.data, dtype=float)[rows, :].copy())
    out.cols = dict(data.cols)
    out.headers = dict(data.headers)
    out.indvar = data.indvar
    if data.weights is not None:
        out.weights = np.asarray(data.weights, dtype=float)[rows, :].copy()
    return out
