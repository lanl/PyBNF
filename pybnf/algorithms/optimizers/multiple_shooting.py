"""Multiple shooting (``job_type = ms``, #563/ADR-0110): the fit type over ADR-0109's layer.

What the method does is documented in :mod:`pybnf.shooting`; this module is the *fit type*
-- the gates, the configuration surface, the ladder's tunables, and the run.

Why this one overrides ``run()``
--------------------------------
Every other optimizer plugs into the shared propose/score loop: propose a
:class:`~pybnf.pset.PSet`, have the cluster simulate and score it, consume the result,
propose the next. Multiple shooting's unit of work is not a PSet evaluation. A segment is
one span of one experiment integrated **from a state that is not in any parameter set** --
the auxiliary variable ``z_j``, which ADR-0109 keeps structurally out of the reported fit
results -- and one augmented-model evaluation is ``m`` such spans whose forward
sensitivities have to be assembled together, on one machine, before a single step can be
taken. The layer's inner-solver contract is a blocking call for the same reason
(``solve(subproblem, u0, tolerance)``; "the solver never calls back into the outer loop").

So this fit type drives its own search on the master and calls
:meth:`~pybnf.algorithms.base.Algorithm._finalize_run` for the end-of-fit path, exactly as
``job_type = hmc`` does and for the same class of reason (ADR-0059: "the gradient cannot
survive the per-pset dask round-trip"). What it does **not** do is fork the tail: every
certified iterate is entered in the ordinary trajectory at its ordinary single-shoot score,
so ``sorted_params``, the best-fit simulations, the information criteria, the profiled-noise
report, and the inference-data sidecar are produced by the same code every other fit type
uses, from numbers that mean the same thing.

The corollary, stated rather than discovered: this cut runs its segments serially on the
master. Segment simulation is embarrassingly parallel and the interface does not prevent
scheduling it, but nothing here schedules it.

The gates
---------
Beyond the gradient path's own (edition 2, a forward-sensitivity backend, differentiable
dynamics), multiple shooting refuses three further classes of fit, each because the
transcription would otherwise change the quantity being fitted rather than the way it is
searched:

* **a model with no enumerated state to restart from** -- the state at a knot is the ODE
  state vector, which a network-free (NFsim) model never enumerates and a non-bngsim backend
  does not expose. Both bngsim paths are supported, through two backends that differ only in
  what a simulation *returns*: :mod:`pybnf.shooting.bngsim_backend` for SBML/Antimony, whose
  trajectory columns already are the species, and :mod:`pybnf.shooting.net_backend` for
  ``.net``, which asks for the observable and species selector families together so one
  integration serves both the data terms and the continuity block (#577);
* **an experiment that is not a plain measured time course** -- a dose-response scan has no
  time axis to cut, and a pre-equilibration protocol's measured phase already begins from a
  carried state that is not the model's own;
* **a quantity that is a function of a whole series** -- an analytic per-series scale
  (ADR-0066), a ``Data``-level normalization (ADR-0053), or a cumulative-to-incident
  difference (ADR-0051). Cutting the series changes each of them, so the segmented fit would
  not be a transcription of the fit that was requested. An analytically profiled **noise
  scale** (ADR-0108) is deliberately not in this list: it is profiled over the pooled
  residuals of every scored experiment, so cutting one series into ``m`` pieces pools the
  same residuals and yields the same ``sigma_hat`` -- which is also what keeps continuity
  violation out of the reported likelihood.
"""

import logging
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from .gradient_base import GradientOptimizer
from ..multistart_report import NOT_STARTED, StartRecord
from ...config_schema import PyBNFConfigModel
from ...printing import PybnfError, print0, print1, print2
from ...registry import register_fit_type
from ...shooting import (
    EQUAL_TIME,
    EXPLICIT,
    BngsimSegmentBackend,
    GaussNewtonSolver,
    NetSegmentBackend,
    SegmentPool,
    ShootingExperiment,
    feasible_ladder,
    run_multiple_shooting,
)
from ...shooting.grid import KNOT
from ...transcription import PenaltySchedule, coarsening_ladder

logger = logging.getLogger('pybnf.algorithms')


def _is_sbml_path(model):
    """Whether this model is on the bngsim SBML/Antimony backend, whose trajectory columns
    *are* its species (:mod:`pybnf.shooting.bngsim_backend`)."""
    return (hasattr(model, '_run_simulation') and hasattr(model, '_result_to_data')
            and hasattr(model, 'species_names'))


def _is_net_path(model):
    """Whether this model is on the bngsim ``.net`` backend -- an expanded reaction network
    whose species this cut reads alongside its observables
    (:mod:`pybnf.shooting.net_backend`, #577).

    A network-free (NFsim) model reuses the same ``_build_data`` but enumerates no species,
    so the engine model's species list is what tells the two apart rather than the class.
    """
    engine = getattr(model, '_engine_model', None)
    if engine is None or not hasattr(model, '_build_data'):
        return False
    try:
        return len(list(engine.species_names)) > 0
    except Exception:
        return False


def _state_names(model):
    """The species a knot carries, whichever bngsim path this model is on.

    The SBML/Antimony wrapper exposes them as a property; the ``.net`` wrapper keeps them on
    its engine model. One accessor so the request-widening and the gate agree on what "the
    state" is.
    """
    if _is_sbml_path(model):
        return list(model.species_names)
    return list(model._engine_model.species_names)


def _flag(value):
    try:
        return bool(int(float(str(value).strip().strip('"\''))))
    except (TypeError, ValueError):
        return False


class MSConfig(PyBNFConfigModel):
    """Multiple-shooting config fields, co-located with the method (ADR-0006).

    Three of these carry measurements rather than preferences, and their defaults are the
    #563 prototype's findings rather than round numbers (ADR-0109):

    ``ms_segments`` is the **finest** rung of the ladder, and it defaults to 4 rather than
    to the largest affordable number. Starting with many short segments -- the easiest
    landscape -- is the wrong end on the motivating problem: at ``m = 8``, with one observed
    state of three, the segmented problem is under-determined and the stage routinely
    certifies worse than its own start. ``ms_coarsening`` is the factor between rungs, so
    the default ladder is ``4 -> 2 -> 1`` and always ends at the ordinary unsegmented fit.

    ``ms_penalty`` and ``ms_penalty_growth`` start the augmented Lagrangian **tight**, which
    is the opposite of what the multiple-shooting literature's motivation suggests. Measured
    from one start: ``rho_0 = 0.1, gamma = 3`` reached ``-178.38`` in 124 s while
    ``rho_0 = 10, gamma = 5`` reached ``-200.70`` in 62 s -- better *and* at half the cost,
    because the inner solve on a nearly-unconstrained subproblem never converges and burns
    its whole budget every outer iteration.

    ``ms_optimality_tol`` measures the *augmented Lagrangian's* projected gradient, whose
    penalty term carries a factor of ``rho``, so it is deliberately looser than ``gntr``'s
    ``1e-8`` on the fit's own gradient: the answer is certified by reconstruction, not by a
    KKT residual, so the extra digits buy nothing a certificate does not already establish.

    ``ms_aux_decades`` is the half-width, in decades, of an auxiliary state's box around its
    own magnitude; ``ms_inner_iterations`` bounds each inner solve (an approximate inner
    minimisation is what the method is designed around, and the outer loop's stall detector
    is what notices a solver that has stopped achieving anything). Like every other local
    method's, ``ms_max_iterations`` is runtime-guarded -- it defaults to the global
    ``max_iterations`` when unset -- and bounds the outer iterations per rung.

    ``ms_parallel_segments`` is how many of a point's segments are integrated at once, and
    it defaults to **1** on a measurement rather than on caution. bngsim releases the GIL
    inside CVODE -- four threads on four warm lanes integrate Borghans segments 2.7x faster
    than one, bit-identically -- but a lane is an engine+simulator pair that has to be built
    at every parameter point, and on that model preparing one (~4.1 ms) costs *more* than
    the integration it saves (~1-2.3 ms). Parallel segments pay when a segment's integration
    is the expensive thing, which is a property of the model's state count rather than of
    the fit; :mod:`pybnf.shooting.parallel` carries the cost model and the numbers.

    ``ms_knot_placement`` and ``ms_knots`` are the "a segment count **or** explicit knots;
    default to generic equal-time **or** equal-observation segments" the issue asks for.
    ``equal_time`` is the default because it uses only the experiment's own time axis;
    ``equal_observations`` uses only its sampling; and ``ms_knots`` names the times outright,
    which *replaces* ``ms_segments`` (the finest rung is then ``len(ms_knots) + 1``). None of
    the three reads a trajectory, which is deliberate: a placement derived from nominal
    dynamics would put the transcription's structure where the fit has not established
    anything, and on the motivating problem those dynamics are exactly what is in question.
    """

    ms_segments: int = 4
    ms_coarsening: int = 2
    ms_penalty: float = 10.0
    ms_penalty_growth: float = 5.0
    ms_max_penalty: float = 1e8
    ms_feasibility_tol: float = 1e-6
    ms_optimality_tol: float = 1e-6
    ms_inner_iterations: int = 50
    ms_aux_decades: float = 6.0
    ms_knot_placement: Literal['equal_time', 'equal_observations'] = 'equal_time'
    ms_knots: list = Field(default_factory=list)
    ms_parallel_segments: int = 1

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'ms_max_iterations'})


# The same pair of flags `gntr` carries, and for the same reasons.
#
# `refiner=True` makes `refine = 1, refine_method = ms` a search followed by a
# multiple-shooting polish -- the fourth arm of #563's acceptance benchmark, and the shape
# the issue's own motivation argues for: a global search finds a basin, and the
# transcription is what converts it. Nothing in the seam needed adding. `_refine_best_fit`
# injects the search's best fit under `START_POINT_KEY`, which `_resolve_start_pset` already
# prefers over every other source, and `_resolve_n_starts` already returns 1 for an injected
# start (a refine polishes one point; it does not re-scatter). Beyond passing
# `_check_refine_method`, the flag buys the coherent-group config pull-in: an `ms` refiner's
# keys are validated against `MSConfig` on the searching fit's config rather than sitting in
# it as unrecognised extras.
#
# `start_from_box=True` is then required rather than optional, and the first cut of this
# registration was wrong to omit it: `refiner` is what `_load_variables` reads to classify a
# fit_type as *start-point*, and a start-point fit_type that is not also `start_from_box`
# may not be given bounded priors at all. `ms` has always drawn its starts from the box
# (`_is_box_start` reads the priors), so without the second flag adding the first would have
# made every standalone `uniform_var` / `loguniform_var` multiple-shooting fit -- which is
# every one that exists -- a configuration error.
@register_fit_type('ms', family='optimizer', display_name='Multiple Shooting',
                   schema=MSConfig, refiner=True, start_from_box=True)
class MultipleShootingAlgorithm(GradientOptimizer):
    """The multiple-shooting fit type.

    Inherits the gradient path's pre-flight gates, its forward-sensitivity activation and
    per-experiment routing (:meth:`~GradientOptimizer._setup_gradient_path`), its
    ``u`` <-> PSet plumbing, and its start-point resolution (the box centre, then
    ``population_size - 1`` Latin-hypercube starts). It replaces only the search: the
    propose/score loop is not the granularity a segment lives at, so :meth:`run` drives the
    ladder directly (see the module docstring).
    """

    fit_type = 'ms'
    START_POINT_KEY = 'ms_start_point'
    _method_label = 'multiple shooting'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.max_outer = config.config.get('ms_max_iterations',
                                           config.config['max_iterations'])
        self.knots = tuple(float(t) for t in (config.config.get('ms_knots') or ()))
        self.placement = (EXPLICIT if self.knots
                          else str(config.config.get('ms_knot_placement', EQUAL_TIME)))
        # Explicit knots *are* the finest rung, so they set the segment count rather than
        # sitting alongside it. Refusing a contradictory ms_segments would be pedantic (the
        # knots are unambiguous); silently ignoring it would not be, so it is reported.
        self.n_segments = (len(self.knots) + 1 if self.knots
                           else int(config.config['ms_segments']))
        self.coarsening = int(config.config['ms_coarsening'])
        self.inner_iterations = int(config.config['ms_inner_iterations'])
        self.aux_decades = float(config.config['ms_aux_decades'])
        self.pool = SegmentPool(config.config.get('ms_parallel_segments', 1))
        self.schedule = PenaltySchedule(
            initial_penalty=config.config['ms_penalty'],
            growth=config.config['ms_penalty_growth'],
            max_penalty=config.config['ms_max_penalty'],
            optimality_tol=config.config['ms_optimality_tol'],
            feasibility_tol=config.config['ms_feasibility_tol'])
        self.homotopies = []

    def reset(self, bootstrap=None):
        """Clear the per-start ladder results along with the rest of the run's state.

        A bootstrap replicate reuses the algorithm object, and everything that reports on
        the ladder reads this list: the reported best start (:meth:`_best_homotopy`, behind
        ``continuity_defects.txt`` and the best stage trace) and the per-start summary. Left
        uncleared, a replicate could report a start belonging to the replicate before it,
        fitted to different resampled data.
        """
        super().reset(bootstrap)
        self.homotopies = []

    def _start_banner(self):
        return ('Running multiple shooting: coarsening ladder from %i segment(s) placed by '
                '%s, up to %i outer iteration(s) per rung, from %i start point(s)'
                % (self.n_segments, self.placement.replace('_', ' '), self.max_outer,
                   self.n_starts))

    def _make_runner(self, u0):
        """Never called: this fit type does not drive per-start step machines through the
        propose/score loop (see the module docstring)."""
        raise PybnfError('job_type = ms drives its own search; it builds no step machine '
                         'for the propose/score loop. This is an internal wiring error.')

    # --- the run ----------------------------------------------------------- #
    def run(self, client=None, resume=None, debug=False):
        """Drive the coarsening ladder from every start point, then finalize.

        ``client`` is accepted and ignored: a segment simulation is not a job the cluster
        can be handed (see the module docstring). ``resume`` is refused rather than
        silently ignored -- there is no checkpointed step machine to resume from, and a
        resumed run that quietly restarted from scratch would be worse than one that says
        it cannot.
        """
        del client, debug
        if resume:
            raise PybnfError(
                'job_type = ms cannot resume a stopped run: it drives its own search '
                'rather than the checkpointed propose/score loop, so there is no partial '
                'state to continue from.',
                hint='Start the fit again, seeding it from the best fit the stopped run '
                     'wrote (Results/sorted_params_final.txt).')
        self.stop_reason = None
        self.completed_simulations = 0
        print2(self._start_banner())

        self._setup_gradient_path()
        if self.knots and 'ms_segments' in self.config.config \
                and int(self.config.config['ms_segments']) != self.n_segments:
            # No silent override: ms_knots fixes the finest rung, so an ms_segments that
            # disagrees with it did not take effect and the run says which one won.
            print1('ms_knots supplies %i explicit knot(s), which fixes the finest rung at '
                   '%i segment(s); the configured ms_segments = %s was not used.'
                   % (len(self.knots), self.n_segments,
                      self.config.config['ms_segments']))
        specs = self._build_specs()
        rungs, dropped = feasible_ladder(
            specs, coarsening_ladder(start=self.n_segments, factor=self.coarsening))
        if dropped:
            # No silent caps: a dropped rung changes what ran, so it is reported.
            print1('Segment count(s) %s exceed the shortest experiment\'s measurement count '
                   'and were dropped; running the ladder %s.'
                   % (', '.join(str(m) for m in dropped),
                      '-'.join(str(m) for m in rungs)))
        print1('Multiple-shooting ladder: %s' % '-'.join(str(m) for m in rungs))
        # The transcription's width, stated up front. It scales with the model's *state*
        # (the auxiliary block is (m-1) x n_species per experiment), not with the number of
        # fitted parameters, so on an expanded reaction network it can dwarf the fit's own
        # dimension -- which a user should hear from the run rather than infer from its
        # duration (#577).
        added = sum((spec.grid(rungs[0]).n_knots) * len(spec.backend.state_names)
                    for spec in specs)
        print1('  finest rung adds %i internal auxiliary variable(s) to this fit\'s %i free '
               'parameter(s)' % (added, len(self.variables)))
        for spec in specs:
            print2(spec.grid(rungs[0]).describe())

        print1('  %s' % self.pool.describe())
        try:
            self._run_starts(specs, rungs)
        finally:
            # The thread pool outlives a single start, so it is closed once the ladder is
            # done -- including when a budget or an error ends the run early, since a live
            # pool would keep the interpreter's non-daemon threads alive past the fit.
            self.pool.close()

        if self._budget_spent():
            self.stop_reason = self._wall_time_stop_reason(self.completed_simulations)
        self._finalize_run()
        self._emit_continuity_defects()

    def _run_starts(self, specs, rungs):
        """Drive the ladder from every start point, newest result first in the log."""
        for index, start in enumerate(self.start_psets):
            if self._budget_spent():
                break
            solver = GaussNewtonSolver(max_iterations=self.inner_iterations,
                                       stop_check=self._budget_spent)
            result = run_multiple_shooting(
                specs, self.objective, self.variables, self._pset_from_u,
                self._param_vec(start), ladder=rungs, schedule=self.schedule,
                inner_solver=solver, max_outer=self.max_outer,
                aux_decades=self.aux_decades, stop_check=self._budget_spent,
                on_iterate=self._record_iterate, on_stage=self._report_stage,
                pool=self.pool)
            # Segment integrations, not augmented-model evaluations: one evaluation is m
            # integrations plus the certification's, and reporting evaluations would
            # understate the cost by the factor the method is judged on.
            self.completed_simulations = sum(spec.backend.n_simulations for spec in specs)
            self.homotopies.append(result)
            print1('Start %i of %i: %s' % (index + 1, len(self.start_psets),
                                           result.summary()))
            print1('  stage trace: %s' % result.trace())
            if not result.certified:
                # The ladder always ends unsegmented, so this should not happen; say so
                # loudly rather than let an unreconstructed score read as a fit result.
                print0('Warning: some of this run\'s scores did not go through the '
                       'ordinary single-shoot path and are not comparable with an '
                       'ordinary fit\'s.')

    def multistart_records(self):
        """One row per start for ``Results/multistart_summary.txt`` (#658).

        This fit type drives its own search rather than the per-start step machines the
        base's summary reads, so the numbers come from the homotopy result each start
        produced: its best certified objective, the outer iterations it took over the whole
        ladder, and why the ladder stopped. A start the run never reached -- the loop
        breaks when the wall-time budget goes -- is listed as one that never ran, so the
        table does not read as a complete set of starts when it is not.
        """
        rows = []
        for i in range(max(len(self.start_psets), len(self.homotopies))):
            if i >= len(self.homotopies):
                rows.append(StartRecord(start=i + 1, stop_reason=NOT_STARTED))
                continue
            result = self.homotopies[i]
            rows.append(StartRecord(
                start=i + 1,
                objective=result.best_score,
                iterations=sum(len(stage.outer.iterates) for stage in result.stages),
                evaluations=result.n_evaluations,
                stop_reason=result.stop_reason))
        return rows

    def _record_iterate(self, record):
        """Enter one certified outer iterate in the ordinary trajectory.

        The certificate *is* an ordinary single-shoot score of the reported parameters, so
        the trajectory holds the same kind of number every other fit type puts there and
        ``trajectory.best_fit()`` is this run's answer -- which is also how the run reports
        its **best certified** iterate rather than its last (ADR-0109 finding 5.3) without
        a ranking rule of its own. An unaccepted certificate (a reconstruction that did not
        simulate, or scored non-finite) is not a fit result and is not entered.
        """
        if not record.certificate.accepted:
            return
        self.probe_counter += 1
        name = '%s_%i' % (self.fit_type, self.probe_counter)
        self.trajectory.add(self._pset_from_u(record.reported, name=name),
                            record.certificate.objective, name)
        if self.probe_counter % self.config.config['output_every'] == 0:
            self.output_results()

    def _report_stage(self, stage):
        print2('  %s' % stage.outer.summary())
        report = stage.outer.defect_report()
        if report:
            print2('    worst scaled continuity defect(s): %s' % report)

    # --- the experiments --------------------------------------------------- #
    def _build_specs(self):
        """One :class:`~pybnf.shooting.problem.ShootingExperiment` per scored experiment.

        Built once for the whole fit -- the backend, the observations, and the routing do
        not depend on the segment count, so every rung of the ladder reuses them.
        """
        self._require_series_are_cuttable()
        specs = []
        for model in self.model_list:
            self._require_carryable_state(model)
            self._request_state_sensitivities(model)
            for suffix, exp_data in self.exp_data.get(model.name, {}).items():
                backend = self._make_backend(model, suffix)
                label = str(suffix)
                if KNOT in label:
                    raise PybnfError(
                        "Multiple shooting names each knot '<experiment>%s<fraction>', and "
                        "experiment suffix '%s' already contains %r."
                        % (KNOT, label, KNOT))
                routing = self._routings[(model.name, suffix)]
                if routing.is_point_dependent:
                    raise PybnfError(
                        "Multiple shooting (job_type = ms) does not yet support a fit whose "
                        "chain-rule factors are point-dependent -- experiment '%s' has a "
                        "seed derivative that reads other model symbols, so its routing has "
                        "to be re-evaluated at every fit point (#530)." % suffix,
                        hint='Fit this model with job_type = gntr, which resolves the '
                             'routing per evaluation.')
                specs.append(ShootingExperiment((model.name, suffix), backend, exp_data,
                                                routing, label=label, start=0.0,
                                                placement=self.placement,
                                                knots=self.knots or None))
        if not specs:
            raise PybnfError('Multiple shooting found no scored experiment to segment.')
        return specs

    def _request_state_sensitivities(self, model):
        """Widen this model's forward-sensitivity request to **every carried state**.

        :meth:`~GradientOptimizer._setup_gradient_path` asks for exactly the columns the
        routings read: the parameter axis a free parameter binds, and the initial-condition
        axis only for a species some free parameter *is* an initial condition of. That is
        the right request for an ordinary gradient fit and the wrong one here. Multiple
        shooting reads ``d(anything)/d(z_j)`` for every state it carries across a knot --
        both for the continuity block (which is exactly ``d(end state)/d(start state)``) and
        for the data rows of every segment after the first, whose predictions depend on
        their own segment-start state through the same axis. A state no free parameter
        happens to seed would otherwise come back with no ``ic`` column at all, and the
        assembly would refuse mid-run.

        Widening rather than replacing: the parameter axis and any initial-condition column
        the routings already asked for are kept, so an ordinary route is unaffected. The
        cost is real and inherent -- a full initial-condition axis is one sensitivity system
        per state -- which is the price of carrying the state at a knot.
        """
        request = getattr(model, '_sensitivity_request', None)
        params = list(getattr(request, 'params', None) or [])
        ic = list(getattr(request, 'ic', None) or [])
        for state in _state_names(model):
            if state not in ic:
                ic.append(state)
        model.enable_output_sensitivities(params=params, ic=ic)

    def _make_backend(self, model, suffix):
        """The segment simulator for one scored ``(model, condition)`` pair.

        Two backends, because the two model paths differ in *what a simulation returns*,
        not in whether they have a state (#577). The SBML/Antimony path reports its species
        as the trajectory's columns with ``species:`` sensitivity selectors on both axes; the
        ``.net`` path reports observables and expressions, so its backend asks for both
        selector families on one run and appends the species columns itself.
        """
        timeout = self.config.config['wall_time_sim']
        if _is_sbml_path(model):
            action, mutant = self._resolve_sbml_action(model, suffix)
            self._require_simple_time_course(model, action, suffix)
            return BngsimSegmentBackend(model, action, mutant, suffix, timeout=timeout)
        sim_params, mutant = self._resolve_net_action(model, suffix)
        self._require_simple_net_action(sim_params, suffix)
        return NetSegmentBackend(model, sim_params, mutant, suffix, timeout=timeout)

    def _resolve_sbml_action(self, model, suffix):
        """The ``(action, condition)`` pair whose output is scored under ``suffix``.

        The same pairing :meth:`~pybnf.bngsim_sbml_model.BngsimSbmlModelNoTimeout.execute`
        makes -- a mutant simulation's output suffix is ``action.suffix + mutant.suffix`` --
        read back so a segment is simulated under exactly the condition the experiment was
        measured under.
        """
        for mutant in getattr(model, 'mutants', []) or []:
            for action in getattr(model, 'actions', []) or []:
                if getattr(action, 'suffix', None) is None:
                    continue
                if action.suffix + mutant.suffix == suffix:
                    return action, mutant
        raise PybnfError(
            "Multiple shooting could not resolve which simulation of model '%s' produces "
            "the scored output '%s'." % (getattr(model, 'name', '?'), suffix))

    def _resolve_net_action(self, model, suffix):
        """The ``(parsed simulate action, condition)`` pair scored under ``suffix``.

        The ``.net`` peer of :meth:`_resolve_sbml_action`. A net model's actions are raw
        BNGL ``simulate()`` lines rather than objects, so each is parsed the way
        ``_execute_actions`` parses it and matched on the same
        ``action suffix + mutant suffix`` key.

        The wildtype is appended rather than assumed present: ``BngsimModel.execute`` runs
        the base actions *before* its mutant loop rather than as an empty-suffix member of
        it (which is where the SBML path puts it), so a model with conditions declared still
        scores its unperturbed run under the bare action suffix.
        """
        from ...bngsim_model.parsing import _parse_simulate_action
        from ...pset import MutationSet
        conditions = list(getattr(model, 'mutants', []) or [])
        if not any(getattr(m, 'suffix', '') == '' for m in conditions):
            conditions.append(MutationSet())
        for mutant in conditions:
            for line in getattr(model, 'actions', []) or []:
                sim_params = _parse_simulate_action(str(line).strip())
                if sim_params is None:
                    continue
                if sim_params.get('suffix', 'time_course') + mutant.suffix == suffix:
                    return sim_params, mutant
        raise PybnfError(
            "Multiple shooting could not resolve which simulate() action of model '%s' "
            "produces the scored output '%s'." % (getattr(model, 'name', '?'), suffix))

    # --- gates ------------------------------------------------------------- #
    def _require_carryable_state(self, model):
        """Refuse a model whose state no segment backend can surface.

        Both bngsim paths qualify (#577): the SBML/Antimony one reports species columns and
        ``species:`` sensitivity selectors directly, and the ``.net`` one is a reaction
        network with the same kind of ODE state whose species trajectory and
        ``d(species)/d(species_0)`` bngsim returns when asked -- which
        :mod:`pybnf.shooting.net_backend` does. What is left over is a backend with no
        expanded network to carry at all (network-free NFsim) or no bngsim seam (a
        RoadRunner/SBML model), and the message names that rather than implying the model
        has no state.
        """
        if _is_sbml_path(model) or _is_net_path(model):
            return
        raise PybnfError(
            "Multiple shooting (job_type = ms) restarts a simulation from the model's own "
            "state at each knot, and model '%s' uses a backend that has no such state to "
            "restart from -- a network-free (NFsim) model never enumerates one, and a "
            "non-bngsim backend exposes neither the state nor its forward sensitivities."
            % getattr(model, 'name', '?'),
            hint=['Simulate the model through bngsim -- a generated network (.net) or an '
                  'SBML model with \'sbml_backend = bngsim\' -- whose species and '
                  'initial-condition sensitivities are what a knot carries.',
                  'Or fit with job_type = gntr, which needs no segment transcription.'])

    def _require_simple_net_action(self, sim_params, suffix):
        """Refuse a ``.net`` ``simulate()`` action that is not a plain measured time course.

        The ``.net`` peer of :meth:`_require_simple_time_course`, on the parsed action rather
        than an action object.
        """
        method = str(sim_params.get('method', 'ode')).strip().strip('"\'')
        reason = None
        if method != 'ode':
            reason = ('it requests method=%r, which is not deterministic ODE integration '
                      'and so carries no forward sensitivities' % method)
        elif _flag(sim_params.get('continue', 0)):
            reason = ("it continues from a previous action's end state rather than from "
                      "the model's own initial conditions")
        elif sim_params.get('stop_if'):
            reason = ('a stop_if condition can end it before its horizon, so its knots are '
                      'not placed on a span it is guaranteed to reach')
        if reason is None:
            return
        raise PybnfError(
            "Multiple shooting (job_type = ms) segments a measured time course, and "
            "experiment '%s' cannot be segmented: %s." % (suffix, reason),
            hint='Fit with job_type = gntr, which needs no segment transcription.')

    def _require_simple_time_course(self, model, action, suffix):
        """Refuse an experiment that is not a plain measured time course."""
        reason = None
        if type(action).__name__ != 'TimeCourse':
            reason = 'it is a %s, which has no time axis to cut' % type(action).__name__
        elif getattr(action, 'preequilibrate', False):
            reason = ('its measured phase already begins from a carried, equilibrated state '
                      'rather than from the model\'s own initial conditions')
        elif getattr(action, 'steady_state', False):
            reason = 'a relaxation to steady state has no fixed horizon to place knots on'
        elif getattr(action, 'initial_state_only', False):
            reason = 'it measures only t = 0, so there is nothing to segment'
        elif getattr(model, '_resolve_method', None) is not None \
                and model._resolve_method(action) != 'ode':
            reason = 'it is not integrated as an ODE, so it carries no forward sensitivities'
        if reason is None:
            return
        raise PybnfError(
            "Multiple shooting (job_type = ms) segments a measured time course, and "
            "experiment '%s' cannot be segmented: %s." % (suffix, reason),
            hint='Fit with job_type = gntr, which needs no segment transcription.')

    def _require_series_are_cuttable(self):
        """Refuse a fit whose scored quantity is a function of a whole series.

        Each of these is computed *over* the series it belongs to, so splitting the series
        changes it -- and a transcription that changes the objective is not a transcription
        of the requested fit. An analytically profiled noise scale is deliberately absent
        from this list; see the module docstring.
        """
        if self.config.config.get('normalization'):
            raise PybnfError(
                "Multiple shooting (job_type = ms) cuts each experiment's time course into "
                "segments, and this fit normalizes its data (normalization = %s) -- a "
                "normalizer is computed over the whole series, so the segments would be "
                "normalized differently from the fit that was requested."
                % self.config.config['normalization'],
                hint='Fit with job_type = gntr, which scores each series whole.')
        if getattr(self.objective, '_analytic_scale', None):
            raise PybnfError(
                "Multiple shooting (job_type = ms) cuts each experiment's time course into "
                "segments, and this fit profiles an analytic per-series scale (ADR-0066) -- "
                "the scale is a property of the whole series, so each segment would be "
                "scaled on its own.",
                hint='Fit with job_type = gntr, which profiles the scale over the whole '
                     'series.')
        for by_suffix in self.exp_data.values():
            for suffix, exp_data in by_suffix.items():
                for column in exp_data.cols:
                    if self.objective._is_cumulative(column):
                        raise PybnfError(
                            "Multiple shooting (job_type = ms) cuts each experiment's time "
                            "course into segments, and column '%s' of experiment '%s' is "
                            "declared cumulative -- its prediction is the difference from "
                            "the previous row, and at a knot that row is in another "
                            "segment." % (column, suffix),
                            hint='Fit with job_type = gntr, which differences the whole '
                                 'series.')
        if self.config.constraints:
            raise PybnfError(
                "Multiple shooting (job_type = ms) does not support a fit with qualitative "
                "or inequality constraints (.con / .prop): a constraint is stated over a "
                "trajectory, and a segmented trajectory does not join up until the run "
                "converges.",
                hint='Fit with job_type = gntr, which evaluates the constraints on the '
                     'whole trajectory.')

    # --- reporting --------------------------------------------------------- #
    def _best_homotopy(self):
        """The start whose ladder produced this run's best certified fit, or ``None``."""
        finished = [h for h in self.homotopies if h.best is not None]
        if not finished:
            return None
        return min(finished, key=lambda h: h.best_score)

    def best_stage_trace(self):
        """The stage trace of the start that produced the run's best certified fit.

        The single most informative artifact a homotopy produces -- it shows whether the
        coarsening is converting the segmented stages, which is the mechanism the method
        rests on. ``None`` before any start has run.
        """
        homotopy = self._best_homotopy()
        return None if homotopy is None else homotopy.trace()

    def _emit_continuity_defects(self):
        """Write ``Results/continuity_defects.txt``: how nearly the transcription that
        produced the reported fit joined up, per knot.

        The issue asks a multiple-shooting run to "report scaled continuity defects", and
        the plural is the point. The aggregate norm says how far from continuous the run
        ended; this says *which knot* did not close and *in which state* -- which is the
        difference between a fit whose one unobserved species drifts at a single knot and
        one whose whole transcription never converged.

        The numbers are **scaled** (ADR-0109): each defect is divided by its state's own
        magnitude, so one number means the same thing across states spanning orders of
        magnitude, and the norm is comparable across models. The row this describes is the
        run's best *certified* iterate -- the reported fit -- so the file pairs with
        ``sorted_params_final.txt`` rather than describing some other point.

        The unsegmented rung of a ladder has no constraints, and that is where a converged
        run usually finishes; the file says so rather than being absent, because "no
        constraints at the reported fit" and "this run never wrote a report" are different
        facts. Every failure is logged and swallowed: the run has completed, and a report
        must never abort it.
        """
        homotopy = self._best_homotopy()
        if homotopy is None:
            return
        best = homotopy.best
        lines = [
            '# Scaled continuity defects at the best certified iterate (job_type = ms).',
            '# A continuity defect is c_j = Phi_j(z_j, theta) - z_{j+1}, the gap between the',
            '#   state a segment integrates to and the state the next segment starts from,',
            '#   divided by that state\'s own magnitude. Scaling is what makes one number',
            '#   mean one thing across states of different size (ADR-0109), so the norms',
            '#   below are dimensionless and comparable across models.',
            '# The iterate described here is the fit reported in sorted_params_final.txt:',
            '#   its objective is an ordinary single-shoot reconstruction, and these defects',
            '#   say how nearly the transcription it came from joined up.',
            'stage\t%s' % best.stage,
            'outer_iteration\t%i' % best.iteration,
            'certified_objective\t%.10g' % best.certificate.objective,
            'n_constraints\t%i' % best.n_constraints,
            'scaled_defect_norm_inf\t%.10g' % best.defect_norm,
            'scaled_defect_rms\t%.10g' % best.defect_rms,
        ]
        if not best.n_constraints:
            lines.append('# This iterate came from the unsegmented rung (m = 1), which has no')
            lines.append('#   knots and therefore no continuity constraints to violate.')
        else:
            lines.append('# The %i largest of %i, worst first.'
                         % (min(len(best.worst_defects), best.n_constraints),
                            best.n_constraints))
            lines.append('# knot\tstate\tscaled_defect')
            for name, value in best.worst_defects:
                knot, _, state = name.partition('::')
                lines.append('%s\t%s\t%.10g' % (knot, state or '-', value))
        path = str(Path(self.res_dir) / 'continuity_defects.txt')
        try:
            Path(self.res_dir).mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            logger.exception('Failed to write continuity_defects.txt')
            return
        logger.info('Wrote scaled continuity defects %s' % path)
        report = best.defect_report()
        print1('Scaled continuity defect at the best certified fit (%s): norm %.3g, rms %.3g'
               % (best.stage, best.defect_norm, best.defect_rms))
        if report:
            print1('  worst knot(s): %s' % report)

    def start_run(self):
        raise PybnfError('job_type = ms drives its own search and does not use the '
                         'propose/score loop. This is an internal wiring error.')

    def got_result(self, res):
        del res
        raise PybnfError('job_type = ms drives its own search and does not use the '
                         'propose/score loop. This is an internal wiring error.')
