"""Shared scaffolding for the gradient-based local optimizers (TRF/LM + L-BFGS-B, #386).

The metaheuristic fit types (de, pso, ss, cmaes, …) only ever ask each evaluated
``PSet`` for its scalar objective value. A **gradient** optimizer instead consumes
the residual vector + residual-Jacobian (TRF / Levenberg–Marquardt) or the scalar
gradient (L-BFGS-B) that #385 assembles from bngsim's forward output-sensitivity
tensor. :class:`GradientOptimizer` factors out everything a new gradient method
needs so a leaf (``trf.py``, ``lbfgs.py``) implements only its step math --
mirroring how :class:`StartPointOptimizer` factors the start-point / ``u`` ↔ ``PSet``
plumbing out of Powell and CMA-ES.

What this base provides
-----------------------
* **The edition + capability gate** (:meth:`_gate_gradient_supported`). Gradient
  fitting consumes the edition-2 surface (bind-by-id routing ADR-0034, the
  ``noise_model`` / measurement layer) and bngsim's forward sensitivities, so the
  fit is refused on a legacy (edition < 2) config, a non-bngsim model, or a bngsim
  build without the ``output_sensitivities`` feature -- with a message that names the
  condition that fired and *then* points at a metaheuristic ``job_type`` (#527).
  Never a silent finite-difference fallback.
* **The gradient path activation** (:meth:`_setup_gradient_path`). Builds each
  experiment's :class:`~pybnf.gradient.routing.ExperimentRouting` **once** (it
  depends only on model structure, conditions, and free-parameter ids -- never on
  the parameter values, #449) and ``apply_routing``\\ s the union request onto every
  model, so each simulated :class:`~pybnf.data.Data` carries its sensitivity tensor.
  Run before the model scatter (from ``start_run``); the request rides the pickle to
  the workers (``BngsimModel.__getstate__`` keeps it, rebuilding only the engine).
* **Master-side scoring** (``requires_master_scoring = True``). The worker scoring
  path nulls ``res.simdata`` after scoring (#385/#388), which would discard the
  sensitivity tensors; the flag makes ``Algorithm.run`` keep scoring on the master so
  every ``Result`` returns with its full simdata for :meth:`gradient_at`.
* **The per-evaluation assembly** (:meth:`gradient_at`). Aligns a Result's
  ``simdata`` with ``exp_data`` and the prebuilt routings, and returns the assembled
  :class:`~pybnf.gradient.assembly.GradientResult` (objective gradient + residual
  Jacobian, in sampling space ``u``), folding in any constraint-penalty gradient.
* **The ``u``-space box** (:meth:`_u_bounds`). The finite reflecting box for bounded
  (``uniform_var`` / ``loguniform_var``) priors, ``±inf`` for an unbounded point
  start -- the bounds the leaf's step projects/reflects into.

The search runs in sampling space ``u`` (``StartPointOptimizer``), and #385 already
delivers the gradient transformed into ``u`` once (ADR-0029), so a leaf never
re-transforms. Leaves own their ``start_run`` / ``got_result`` state machine and
must be picklable for backup/resume, exactly like Powell and CMA-ES (ADR-0007).
"""

import logging

import numpy as np

from .concurrent_multistart import DONE, ConcurrentMultiStartOptimizer
from ... import _bngsim_caps
from ...gradient import (
    GradientNotSupported,
    apply_routings,
    assemble_constraint_gradient,
    assemble_gaussian_gradient,
    assemble_marginal_time_gradient,
    route_for_model,
)
from ...printing import PybnfError, print0, print1, print2

# ``DONE`` is the shared multi-start sentinel, re-exported here so a gradient leaf's
# ``from .gradient_base import DONE`` keeps resolving to the one object the base's
# ``got_result`` identity-checks (#500).
__all__ = ['DONE', 'GradientRunner', 'GradientOptimizer']

logger = logging.getLogger(__name__)


class GradientRunner:
    """Headless, picklable per-start step machine in sampling space ``u`` (#386).

    A gradient leaf's step math, factored out of the optimizer so a single fit can
    run ``N`` of them **concurrently** -- local multi-start, the diversity a purely
    local gradient method otherwise lacks (it only ever descends into the one basin
    its start lands in). A runner owns one start's entire mutable state -- the
    iterate, the curvature / trust-region model, the reflecting box, the tunables --
    and is pure ``numpy``: it knows nothing about :class:`~pybnf.pset.PSet`\\ s, the
    objective, routing, backup, or dask. :class:`GradientOptimizer` drives it::

        u0 = runner.start()                       # the first point to evaluate
        nxt = runner.got(u_point, score, grad)    # consume one completed evaluation

    where ``u_point`` is the realized (box-projected) ``u``-vector of the completed
    evaluation, ``score`` its objective value, and ``grad`` the assembled
    :class:`~pybnf.gradient.assembly.GradientResult` at it; ``got`` returns the next
    ``u`` to evaluate, or the :data:`DONE` sentinel. Because a runner holds only
    plain ``float`` / ``ndarray`` / ``list``, the optimizer that owns the list of
    runners pickles for backup/resume exactly like the single-start machine did
    (ADR-0007). Being backend-free, a runner is also unit-testable offline by feeding
    it scores + gradients from an analytic function (no bngsim) -- how the step math
    is validated against a scipy oracle and how the multi-start win is demonstrated.

    A leaf subclass (``trf.py`` / ``lbfgs.py``) sets up its own model state in
    ``__init__`` and implements :meth:`got`; the orchestrator only ever calls
    :meth:`start`, :meth:`got`, and reads :attr:`iteration` / :attr:`fval` /
    :attr:`stop_reason` / :meth:`progress_detail` for reporting.
    """

    #: How a leaf names the local model it steps from, quoted in the :meth:`_failed_model`
    #: stop reason (``trf``: the residual model; ``gntr``: the Fisher model; ``lbfgs``: the
    #: gradient) so a terminated start says which object was unusable (#528).
    _model_label = 'local model'

    def __init__(self, u0, lower, upper, max_iterations):
        self.point = np.array(u0, dtype=float)   # current iterate (u-space)
        self._u_lower = lower                    # reflecting box (constant per fit)
        self._u_upper = upper
        self.n = len(self.point)
        self.max_iterations = max_iterations
        self.iteration = 0          # accepted steps so far (drives reporting / budget)
        self.phase = 'init'         # 'init' until the start point is evaluated
        self.fval = None            # objective F(point)
        self.grad = None            # scalar gradient dF/du at point (u-space)
        self.stop_reason = None     # set to a human string when got() returns DONE
        # Why this start produced no fit, for a consumer that has to say so in its own words
        # (the profile-likelihood track): None if it ran normally, 'simulation' if its start
        # point did not evaluate (_failed_start), 'model' if it evaluated but its derivatives
        # were unusable (_failed_model). Both leave ``fval`` at the inf penalty (#492/#528).
        self.failure = None

    def start(self):
        """The first ``u``-point to evaluate (the start point)."""
        return self.point

    def got(self, u_point, score, grad):
        """Consume one completed evaluation and return the next ``u`` or :data:`DONE`.

        The orchestrator passes ``grad = None`` with a non-finite ``score`` for a **failed
        simulation** (a non-integrable candidate point). A leaf's ``got`` must tolerate that:
        mid-search its ``isfinite(score)`` guard already rejects the trial without
        dereferencing the gradient (the fit backs off); at the start point (``phase ==
        'init'``, ``grad is None``) it must terminate via :meth:`_failed_start`. See
        :meth:`GradientOptimizer._advance` (#492).

        A point that *did* evaluate can still hand back an unusable **model**
        (:meth:`_model_is_usable`); a leaf handles that the same two ways -- back off
        mid-search, :meth:`_failed_model` at the start point (#528)."""
        raise NotImplementedError

    @staticmethod
    def _all_finite(*arrays):
        """True when every supplied array is present and entirely finite."""
        return all(a is not None and bool(np.all(np.isfinite(a))) for a in arrays)

    def _model_is_usable(self, grad):
        """Whether the local model assembled at an evaluated point is one this method can
        actually step from -- i.e. every array the leaf reads off it is finite (#528).

        A point can score finitely and still hand back non-finite *derivatives*: a stiff
        parameter set whose ODE solve completes while its forward sensitivities diverge, an
        overflow in the chain rule, a vanishing prediction in a denominator. There is no step
        to take from such a model, and taking one anyway does not degrade gracefully -- it
        aborts the **whole** fit, not just the start that met the bad point. LAPACK refuses to
        factorize a non-finite matrix (``LinAlgError: SVD did not converge`` out of the
        trust-region subproblem, the reported crash of #528), and a quasi-Newton direction
        built from a NaN gradient is itself NaN, so the next proposed point is NaN and dies as
        an ``OutOfBoundsException`` when the orchestrator builds its PSet. Either exception
        unwinds through ``got_result`` and out of the run loop, discarding every *other*
        concurrent start's progress with it.

        This base implementation checks the scalar gradient every assembled
        :class:`~pybnf.gradient.assembly.GradientResult` carries (what ``lbfgs`` consumes); a
        leaf whose model is built from other fields -- ``trf``'s residual + Jacobian,
        ``gntr``'s Fisher Hessian -- overrides to check those instead."""
        return grad is not None and self._all_finite(grad.gradient)

    def _failed_model(self, detail):
        """Terminate this start: the start point evaluated and scored, but the local model
        assembled there is not one this method can descend from (:meth:`_model_is_usable`),
        and at the start point there is no earlier iterate to back off to. The sibling of
        :meth:`_failed_start`, which handles the point that did not evaluate at all (#528).

        ``fval`` is set to the same ``inf`` penalty :meth:`_failed_start` records even though
        this point *did* score. What the start-point score is not is a **fit**: no step was
        ever taken from it. Reporting it as this start's objective would let a consumer read
        an unoptimized value as an optimized one -- concretely, the profile-likelihood grid
        point in :meth:`ProfileLikelihoodAlgorithm._profile_got`, where an un-minimized upper
        bound entered as the profile inflates that point's Δχ², which can fabricate a
        threshold crossing and report a confidence interval narrower than the data supports.
        :attr:`failure` records *which* of the two failures this was, so a consumer that has
        to explain the stop can say the accurate thing rather than "simulation failed".

        Only this start stops; concurrent multi-start keeps every other start running, and
        the trajectory (which holds every evaluated point, including this one, at its real
        score) keeps the global best."""
        self.fval = float('inf')
        self.failure = 'model'
        self.stop_reason = '%s; no usable local model to descend from' % detail
        return DONE

    def _failed_start(self):
        """Terminate this start: its start point did not simulate (a non-integrable point --
        a bngsim CVODE failure, a NaN/Inf), so there is no finite objective or gradient to
        model the local surface from and descend. A gradient method needs a viable start;
        with none, this start ends. Concurrent multi-start keeps every *other* start's
        progress and the global best, so only this start stops (a single-start fit ends here,
        the failed point left in the trajectory at the ``inf`` penalty). Fed by the
        orchestrator as ``grad is None`` at ``phase == 'init'``; see
        :meth:`GradientOptimizer._advance` (#492). ``fval`` is set to the ``inf`` penalty so a
        consumer that reads the terminated runner's objective (e.g. the profile-likelihood
        grid point in :meth:`ProfileLikelihoodAlgorithm._profile_got`) sees a non-finite value
        rather than the ``None`` a never-evaluated runner starts with."""
        self.fval = float('inf')
        self.failure = 'simulation'
        self.stop_reason = ('start point failed to simulate (a non-integrable point); '
                            'no objective/gradient to descend from')
        return DONE

    def progress_detail(self):
        """A short method-specific status suffix for the per-iteration report."""
        return ''


# The one-line suggestion every gradient-path refusal ends with, so a user whose
# model/config cannot be differentiated is pointed straight at a working job_type
# rather than left to guess. Gradient fitting is strictly opt-in (job_type = trf /
# lbfgs); a metaheuristic always works on the same config. Passed as PybnfError's
# ``hint=``, never as its ``user_message``: the hint is a *suffix* to the specific
# diagnosis, not a substitute for it (#527) -- four unrelated conditions refuse here,
# and which one fired is the whole of what the user needs to know.
_FALLBACK_HINT = (
    "Use a metaheuristic job_type instead (e.g. job_type = de, the default, or "
    "pso / ss / cmaes), which needs no gradient."
)


class GradientOptimizer(ConcurrentMultiStartOptimizer):
    """The gradient-based leg of the concurrent multi-start base (#386/#500).

    A leaf subclass supplies only its per-start step math as a :class:`GradientRunner`
    (Levenberg–Marquardt for ``trf``, L-BFGS-B for ``lbfgs``) via :meth:`_make_runner`; the
    shared
    :class:`~pybnf.algorithms.optimizers.concurrent_multistart.ConcurrentMultiStartOptimizer`
    owns the ``start_run`` / ``got_result`` orchestration -- seeding the runners, the name
    routing, reporting, and the multi-start ``STOP`` coordination -- and this class fills in
    what the gradient path does differently: the pre-flight gates + master scoring, the
    ``u`` <-> PSet plumbing, the sensitivity-path activation (:meth:`_setup_gradient_path`,
    hung on :meth:`_pre_seed`), the :meth:`gradient_at` assembly, and the gradient-consuming
    :meth:`_advance`. The leaf must set :attr:`START_POINT_KEY` like any
    :class:`StartPointOptimizer`, plus :attr:`_method_label` and :meth:`_start_banner` for
    its messages.

    Local multi-start (#386). A box-start gradient fit runs ``N`` independent starts
    concurrently (``N`` reuses ``population_size`` -- the gradient path predates the
    ``n_starts`` field, hence :attr:`_n_starts_key`) -- start 0 from the box center
    (preserving the deterministic single-start behavior), the rest from Latin-hypercube
    samples across the prior box -- and keeps the global best. Every evaluated PSet across
    all starts lands in the trajectory (``add_to_trajectory`` runs before ``got_result``),
    so ``trajectory.best_fit()`` is the global best for free -- each runner only tracks its
    own best for its own convergence test.
    """

    #: Keep objective scoring on the master so every Result returns with its
    #: simdata (the sensitivity tensors the gradient assembly reads); see
    #: ``Algorithm.run``. Without this the worker path nulls ``res.simdata``.
    requires_master_scoring = True

    #: Human label for the method in the per-iteration progress messages; set by
    #: each leaf (e.g. ``'L-BFGS-B'`` / ``'TRF'``).
    _method_label = 'gradient'

    #: The gradient path predates the ``n_starts`` field and reuses ``population_size`` as
    #: the box-fit start count (consistent with the metaheuristics, where it is the
    #: parallel-population size, and ``population_size = 1`` reproduces the historical
    #: single start).
    _n_starts_key = 'population_size'

    #: The verb the base logs when a start terminates (a gradient start "stops"; a
    #: derivative-free start "finishes") -- cosmetic, preserved verbatim.
    _stop_verb = 'stopping'

    # --- construction / reset hooks ---------------------------------------- #
    def _check_config_supported(self, config):
        """Refuse a legacy (edition < 2) config before the base builds a single model --
        the cheapest gate, before the expensive network generation in ``Algorithm.__init__``
        (a legacy-edition config can never carry the gradient surface)."""
        self._require_edition_2(config)

    def _after_init(self):
        """The gradient path's construction extras, run after the models are built and
        before start resolution: the sensitivity-backend and differentiability gates, and
        the reflecting box + (empty) per-experiment routings."""
        # Per-experiment routing, keyed by (model_name, suffix); built lazily in
        # _setup_gradient_path (needs the initialized models). None until then, and
        # restored as None by reset() so a bootstrap refit rebuilds it.
        self._routings = None
        # Whether _report_sensitivity_rhs has already spoken. Deliberately NOT reset by
        # _after_reset: the routings are rebuilt per bootstrap refit, but which
        # sensitivity RHS each model runs on is not a function of the resampled data
        # (#606).
        self._sens_rhs_reported = False
        # Backend gate: every model must expose bngsim's forward-sensitivity hooks
        # (the capability gate itself fires later, at apply_routing).
        self._require_sensitivity_backend()
        # Differentiability gate: a discrete-event model needs a bngsim that
        # differentiates the jump; on a build that does not, refuse now rather than
        # run to completion on a silently-wrong gradient (#461/#536).
        self._require_differentiable_dynamics()
        # The reflecting box in sampling space u (the leaf's step projects/reflects into it).
        self._u_lower, self._u_upper = self._u_bounds()

    def _after_reset(self):
        """Rebuild the reflecting box and drop the routings so a bootstrap refit rebuilds
        them; the gates already passed at construction and never regress on a refit."""
        self._routings = None
        self._u_lower, self._u_upper = self._u_bounds()

    # --- run-loop hooks ---------------------------------------------------- #
    def _pre_seed(self):
        """Activate the gradient path (enable sensitivities + build routings) before the
        runners are seeded -- and so before the model scatter, so the request rides the
        pickle to the workers."""
        self._setup_gradient_path()

    def _build_runners(self):
        """One :class:`GradientRunner` per start, seeded at each start PSet's ``u``-vector.
        The gradient step is deterministic, so (unlike the local path) no per-start rng is
        provisioned -- keeping the base rng-agnostic (#500)."""
        return [self._make_runner(self._u_from_pset(p)) for p in self.start_psets]

    def _seed(self, idx, runner):
        """Start ``idx``'s single opening evaluation (its start point)."""
        return [self._dispatch(idx, runner.start())]

    def _advance(self, idx, runner, res):
        """Assemble the gradient at the completed ``res``, feed ``(u, score, grad)`` to
        start ``idx``'s runner, and return its next PSet -- or :data:`DONE` once it
        terminates. The realized (box-projected) ``u`` of the evaluated point is read back
        off the PSet so the runner's internal iterate is a genuinely evaluated point.

        A **failed simulation** (a non-integrable candidate point: a bngsim CVODE failure, a
        NaN/Inf, ...) returns with ``res.simdata is None`` and ``res.score`` already the
        ``inf`` penalty (set in ``add_to_trajectory``); there is no trajectory data to
        assemble a gradient from. Feed the runner that non-finite evaluation with **no
        gradient** (``grad = None``): mid-search the runner's own ``isfinite(score)`` guard
        rejects the step / shrinks its trust region and proposes a shorter one -- it never
        dereferences the gradient on a rejected trial -- so the fit *backs off* rather than
        aborting; at the start point there is no basin to descend from, so the runner
        terminates that start (:meth:`GradientRunner._failed_start`), leaving every other
        concurrent start and the trajectory's global best untouched. Mirrors the scalar
        path's ``inf`` penalty for a failed simulation and the ``res.simdata is None`` guard
        added for the sampler / constraint-tracking path in #480 -- this is the gradient
        path's analogous unguarded case (#492)."""
        u_point = self._u_from_pset(res.pset)
        if res.simdata is None:
            grad, score = None, float('inf')
        else:
            grad, score = self.gradient_at(res), float(res.score)
        nxt = runner.got(u_point, score, grad)
        if nxt is DONE:
            return DONE
        return [self._dispatch(idx, nxt)]

    def _dispatch(self, idx, u):
        """Wrap a runner's proposed ``u``-point as a uniquely named PSet bound to its
        owning start in :attr:`pending`, and return it for submission. The name carries a
        single global counter (``<fit_type>_<k>``), so a single-start fit reproduces the
        historical ``<fit_type>_1``, ``<fit_type>_2``, … sequence exactly while every
        name stays unique across concurrent starts (the routing key)."""
        self.probe_counter += 1
        name = '%s_%i' % (self.fit_type, self.probe_counter)
        return self._route(idx, self._pset_from_u(u, name=name))

    def _report(self, runner):
        """Per-iteration progress for one start (mirrors the single-start report); the
        method-specific suffix comes from the runner."""
        if runner.iteration % self.config.config['output_every'] == 0:
            self.output_results()
        msg = 'Completed %i of %i %s iterations' % (
            runner.iteration, runner.max_iterations, self._method_label)
        (print1 if runner.iteration % 10 == 0 else print2)(msg)
        print2('Current best objective: %f, %s' % (runner.fval, runner.progress_detail()))

    # --- gates ------------------------------------------------------------- #
    # The gradient path is gated in four places, each as early as it can be (a fifth
    # check, :meth:`_report_sensitivity_rhs`, sits beside them and is a *report* unless
    # the user asks it to be a gate -- see its own docstring):
    #
    # * **edition** (:meth:`_require_edition_2`, before model build) -- the gradient
    #   consumes the edition-2 surface (bind-by-id routing, the noise-model /
    #   measurement layer), absent under legacy edition 1;
    # * **backend** (:meth:`_require_sensitivity_backend`, after model build) -- every
    #   model must expose bngsim's forward-sensitivity hooks; a non-bngsim (e.g.
    #   RoadRunner/SBML) model has no sensitivity tensor here;
    # * **differentiability** (:meth:`_require_differentiable_dynamics`, after model
    #   build, #461/#536) -- a discrete-event model needs a bngsim whose forward
    #   sensitivities survive the jump *and* which refuses the event subclasses it
    #   cannot cross; on an older build the refusal is blanket and fires here rather
    #   than mid-run, and on a current one the model passes straight through;
    # * **capability** (deferred to :meth:`_setup_gradient_path`'s ``apply_routing``,
    #   #447) -- raises if the bngsim build lacks the ``output_sensitivities`` feature.
    #
    # Every one of those is a property of the BUILD or of the CONFIG, which is why each
    # can be decided from a module-level flag or a config read. The fifth is not: whether
    # bngsim supplies an analytic ``∂f/∂p`` for a given model is a property of the
    # (build, model) pair, decided at codegen, so :meth:`_report_sensitivity_rhs` has to
    # build a Simulator to find out (#606, ADR-0121). It warns by default rather than
    # refusing, because the fallback is correct -- just N times the cost -- and only
    # refuses under ``sensitivity_fallback = error``.
    #
    # The per-evaluation gate (an unsupported *objective* -- Laplace residual,
    # estimated scale, … raising :class:`GradientNotSupported`) is caught at the first
    # assembly in :meth:`gradient_at`. A non-ODE simulation *method* (SSA / NFsim), and a
    # carried-state pre-equilibration ``parameter_scan`` (#474), are likewise
    # non-differentiable, but the method is an action-level property (a model can mix
    # actions) rather than a model-structure one, so they are not hoisted here. They keep
    # a per-evaluation refusal in the backend (``_sensitivity_request_kwargs`` /
    # ``_scan_carried_state`` raise a clean :class:`PybnfError`, not a raw backend
    # traceback) -- but *only when that action's output is a scored gradient target*
    # (#475): an incidental/unscored non-ODE or carried-state action needs no
    # sensitivities, so it runs sensitivity-free instead of aborting a fit whose scored
    # objective is fully differentiable. :meth:`_setup_gradient_path` declares each
    # model's scored suffixes so the backend can make that per-action distinction. Events,
    # by contrast, are a build-time structural signal and so *can* be a pre-flight gate --
    # which is why the one subclass of them a current bngsim still declines (a delayed or
    # non-relational trigger) keeps its refusal in the backend, where the whole event is
    # in view, instead of being re-derived here (#536).
    def _require_edition_2(self, config):
        """Refuse a legacy (edition < 2) config before any model is built."""
        edition = config.config.get('edition')
        if not edition or edition < 2:
            raise PybnfError(
                "Gradient-based fitting (job_type = %s) requires the edition-2 "
                "config surface, but this fit is %s." % (
                    self._fit_type_label(),
                    "edition 1 (legacy)" if not edition else "edition %d" % edition),
                hint=["Opt into edition 2 ('edition = 2') and declare the fit on the "
                      "new-era surface (experiment: / data: / noise_model, bind-by-id "
                      "parameters).",
                      _FALLBACK_HINT])

    def _require_sensitivity_backend(self):
        """Refuse a model whose backend has no forward-sensitivity hooks."""
        for model in self.model_list:
            if not hasattr(model, 'enable_output_sensitivities'):
                raise PybnfError(
                    "Gradient-based fitting (job_type = %s) requires the bngsim "
                    "backend's forward sensitivities, but model '%s' uses a backend "
                    "that does not provide them." % (
                        self._fit_type_label(), getattr(model, 'name', '?')),
                    hint=["Simulate the model through bngsim (an SBML model needs "
                          "'sbml_backend = bngsim'), which provides them.",
                          _FALLBACK_HINT])

    def _require_differentiable_dynamics(self):
        """Refuse a discrete-event model on a build that cannot differentiate one (#461/#536).

        A discrete event is a discrete jump in the dynamics: it reinitialises the
        integrator state discontinuously, so a forward-sensitivity vector carried
        across it is right only if the solver applies the event's own jump

        .. math::

            s^+ = \\frac{\\partial h}{\\partial x}
                  \\left(s^- + f^-\\frac{\\partial t^*}{\\partial p}\\right)
                  + \\frac{\\partial h}{\\partial p}
                  - f^+\\frac{\\partial t^*}{\\partial p}

        at each fire. Originally it never did -- the vectors were carried straight
        through and went silently stale, so bngsim refused sensitivities on any
        event-bearing model and #461 hoisted that refusal here, as a **blanket**
        pre-flight gate, rather than let it surface mid-run at the first
        sensitivity-bearing ``simulate()``.

        bngsim applies the jump now. On a build
        :data:`~pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS` reports it also *classifies*
        each event honestly -- differentiating the subclasses it covers (a fixed
        trigger time; a trigger thresholding a fitted constant, lanl/bngsim#49; a
        state-dependent trigger whose crossing it differentiates in flight,
        lanl/bngsim#144) and refusing the rest (an execution delay; a trigger that
        does not reduce to a single relational comparison). On such a build this
        stops being a gate: the model is allowed through and bngsim's own per-model
        refusal covers what it cannot cross, re-raised as a clean
        :class:`~pybnf.printing.PybnfError` by
        ``BngsimSbmlModelNoTimeout.execute`` -- the SBML/Antimony backend being the
        only one an event can reach PyBNF through, since a ``.net`` model cannot
        author one.

        Otherwise the refusal stays, and stays blanket, because such a build does
        not merely lack a subclass -- it answers one *wrongly and quietly*: a
        trigger reading the state came back as a finite tensor missing the event's
        contribution instead of being refused (lanl/bngsim#52), and an event
        assignment that reads the state dropped its carried term altogether
        (lanl/bngsim#144). Refusing up front beats a fit that runs to completion on
        a wrong gradient.

        The message says both what to install and *how the flag decided*
        (:func:`~pybnf._bngsim_caps.event_sens_probe`), because since #558 the two
        can disagree: the flag reads a capability rather than a version, so a
        reader whose bngsim already reports a new enough number needs to be told
        that the number was not the evidence -- otherwise the refusal reads as a
        version complaint they have already answered.

        Models whose backend exposes no event count (``has_discrete_events``
        absent) pass through untouched."""
        if _bngsim_caps.BNGSIM_HAS_EVENT_SENS:
            return
        for model in self.model_list:
            if getattr(model, 'has_discrete_events', False):
                raise PybnfError(
                    "Gradient-based fitting (job_type = %s) needs forward "
                    "sensitivities that survive a discrete event (a discrete jump in "
                    "the dynamics), and model '%s' contains one. The installed bngsim "
                    "(%s) can still answer such an event wrongly without saying so -- "
                    "a tensor missing the event's contribution rather than a refusal "
                    "-- so the gradient there would be silently wrong." % (
                        self._fit_type_label(), getattr(model, 'name', '?'),
                        _bngsim_caps.BNGSIM_VERSION or 'version unknown'),
                    hint=["Install bngsim >= %s, which differentiates the event "
                          "subclasses it supports and refuses the rest. This gate "
                          "reads a capability, not a version -- it decided from: %s "
                          "-- so a build whose version already reads new enough is "
                          "one that does not publish the capability, which a "
                          "from-source build ahead of (or behind) its own release "
                          "number can be."
                          % (_bngsim_caps.event_sens_min_version(),
                             _bngsim_caps.event_sens_probe()),
                          _FALLBACK_HINT])

    def _fit_type_label(self):
        """The fit_type code for messages (the leaf's registered name, best-effort)."""
        return getattr(self, 'fit_type', type(self).__name__)

    # --- gradient-path activation ------------------------------------------ #
    def _setup_gradient_path(self):
        """Enable forward sensitivities on every model and build the per-experiment
        routings -- idempotent, called once from the leaf's ``start_run`` (before the
        model scatter, so the request rides the pickle to the workers).

        For each model: build one :class:`ExperimentRouting` per scored
        ``(model, suffix)`` (carrying that condition's chain-rule factors for the
        assembly), then ``apply_routings`` the **union** of their sensitivity
        requests -- plus the wildtype's. The wildtype alone is not a superset: a
        condition can route a free parameter to a column no other experiment binds (a
        per-condition estimated initial condition, ADR-0076), so that column is
        reached only through the condition and must be unioned in (an extra requested
        column is harmless; a missing one aborts the assembly). Raises (capability
        gate) if the bngsim build lacks ``output_sensitivities``."""
        if self._routings is not None:
            return
        names = [v.name for v in self.variables]
        routings = {}
        for model in self.model_list:
            # Declare which of this model's outputs are scored gradient targets so
            # an incidental/unscored action (a stochastic diagnostic, a
            # carried-state pre-equilibration scan) runs sensitivity-free instead
            # of aborting the whole fit at a differentiability guard (#475). Rides
            # the scatter alongside the sensitivity request.
            model.set_scored_suffixes(self.exp_data.get(model.name, {}))
            # Apply the UNION sensitivity request over the wildtype and every scored
            # condition -> sets _sensitivity_request, which survives the scatter and is
            # applied at every simulate(). The wildtype alone is NOT a superset once a
            # condition routes a free parameter to a column no other experiment binds --
            # a per-condition estimated initial condition (ADR-0076): its species-IC /
            # multiplier column is reached only through that condition, so the union must
            # include every scored routing (an extra column is harmless; a missing one aborts).
            wildtype = route_for_model(model, names, condition=None)
            model_routings = {}
            for suffix in self.exp_data.get(model.name, {}):
                condition = self._condition_for_suffix(model, suffix)
                model_routings[suffix] = route_for_model(model, names, condition)
            apply_routings(model, [wildtype, *model_routings.values()])
            for suffix, routing in model_routings.items():
                routings[(model.name, suffix)] = routing
        self._routings = routings
        self._report_sensitivity_rhs()

    def _report_sensitivity_rhs(self):
        """Say which sensitivity right-hand side each model's gradient will run on (#606).

        ``CVodeSensInit1`` takes one sensitivity-RHS callback for every column, so a
        single rate law bngsim cannot differentiate declines the analytic ``∂f/∂p`` for
        the **whole** model and CVODES' internal difference quotient carries every
        column instead. That is a correctness-preserving substitution and a
        cost-multiplying one: an extra RHS evaluation per column per step, so an
        N-column request pays roughly N times the sensitivity cost. On a fit measured
        in hours that is not a slower answer, it is no answer -- on
        ``Smith_BMCSystBiol2013`` all 25 columns fell back, every start timed out to
        ``inf``, and thirteen hours produced nothing, with the only signal a bngsim log
        line on a worker that nobody had a reason to look for (#558).

        This is the only pre-flight check here that is a property of the **(build,
        model)** pair rather than of the build or the config, so unlike the four gates
        above it cannot be answered from a module-level flag or a config read: it
        builds one sensitivity-bearing Simulator per model and reads the verdict off
        the codegen artifact that Simulator installs
        (:func:`~pybnf._bngsim_caps.probe_sens_rhs`). Running it here rather than on a
        worker is what makes it useful -- the answer arrives before the fit has spent
        anything, which is the whole complaint the log line could not answer.

        The verdict is what policy keys off, because the verdict is stable. bngsim's
        own *reason* for a decline is captured too, but only ever as prose: it is
        emitted during codegen source generation, which a warm structural cache skips
        entirely, so it is present on the first run of a fit and absent on the second.
        A model reporting no opinion (``None`` -- no codegen artifact to read) is
        logged and not warned about, in either direction.

        Reported **once per run**, not once per pass: ``reset()`` drops the routings so
        a bootstrap refit rebuilds them, and a model's differentiability does not change
        with resampled data, so ``bootstrap = 100`` would otherwise print the same
        warning a hundred times. The refusal is once-only for the same reason -- it has
        already ended the run the first time.

        Never raises except under ``sensitivity_fallback = error``, which is a user
        asking to be stopped."""
        policy = str(self.config.config.get('sensitivity_fallback', 'warn')).lower()
        if policy == 'ignore' or getattr(self, '_sens_rhs_reported', False):
            return
        self._sens_rhs_reported = True
        declined = []
        for model in self.model_list:
            probe = getattr(model, 'analytic_sens_rhs_status', None)
            if not callable(probe):
                # A backend with no opinion to give (a test double, a non-bngsim
                # model). The sensitivity-backend gate above already refused anything
                # that cannot supply a gradient at all, so this is not a failure.
                continue
            status = probe()
            if status.analytic is None:
                logger.info(
                    "Model %s: cannot tell whether the forward sensitivities run on "
                    "bngsim's analytic df/dp -- %s.", model.name, status.route)
            elif status.analytic:
                logger.info(
                    "Model %s: forward sensitivities run on bngsim's analytic df/dp "
                    "(read from %s).", model.name, status.route)
            else:
                declined.append((model, status))
                self._warn_sensitivity_fallback(model, status)
        if declined and policy == 'error':
            raise PybnfError(
                "Gradient-based fitting (job_type = %s) was asked to require the "
                "analytic sensitivity right-hand side (sensitivity_fallback = error), "
                "and bngsim declined it for %s." % (
                    self._fit_type_label(),
                    ', '.join("model '%s'" % m.name for m, _ in declined)),
                hint=["Re-encode the declined rate law in a form bngsim can "
                      "differentiate. Its own reason is in the warning above when it "
                      "gave one -- it reports the reason while generating codegen "
                      "source, so a warm codegen cache has none to give.",
                      "Or accept the fallback with 'sensitivity_fallback = warn' (the "
                      "default) and expect roughly one extra right-hand-side "
                      "evaluation per sensitivity column per step.",
                      _FALLBACK_HINT])

    def _warn_sensitivity_fallback(self, model, status):
        """One model's difference-quotient warning, to the log and to the console.

        Console rather than log-only on purpose, and at verbosity 0 rather than 1. The
        decline already reaches ``<prefix>.log`` today -- bngsim's logger propagates to
        root and PyBNF puts a FileHandler there -- and that is exactly the channel that
        failed: a shared, noisy file written from N worker processes, one line per
        model, arriving mid-run. It is discoverable by someone who already suspects the
        problem, which is the wrong order. A reader who turned the verbosity down is
        still a reader who would rather not spend the next thirteen hours, which is the
        same call ``_report_bngsim_build`` makes for a stale compiled core (#558).
        """
        columns = status.columns or len(self.variables)
        cost = ("each of this model's %d sensitivity columns costs an extra "
                "right-hand-side evaluation per step, so expect roughly %dx the "
                "sensitivity cost of the analytic path" % (columns, columns))
        reason = '; '.join(reason for reason, _ in status.reasons)
        detail = (' bngsim declined it because %s.' % reason if reason else
                  ' bngsim did not say why in this run: it reports the reason while '
                  'generating the codegen source, which a warm codegen cache skips.')
        logger.warning(
            "Model %s: bngsim declined the analytic sensitivity RHS, so CVODES' "
            "internal difference quotient carries every column (%s; read from %s).%s",
            model.name, cost, status.route, detail)
        print0("WARNING: model '%s' has no analytic sensitivity right-hand side, so "
               "this gradient fit runs on CVODES' internal difference quotient -- %s. "
               "The gradient stays correct.%s" % (model.name, cost, detail))
        print1('  -> Read from %s.' % status.route)
        if status.fallback_is_wrong:
            # The half of the decline space where "correct, but slower" is FALSE. This
            # is a statement about the NUMBERS rather than the cost, so it goes to
            # print0, past the verbosity the cost line respects.
            #
            # From 0.14.0 bngsim refuses such a run outright rather than returning the
            # gradient it has flagged as wrong (lanl/bngsim#414/#416) -- and it decides
            # that from its own ground truth, re-scanning the model rather than trusting
            # the codegen warning, so its refusal survives a warm cache where this line
            # does not. The wording therefore has to be true on both sides of that
            # line: on a carrying build the fit is about to stop at the first
            # simulation, and on an older one it is about to run.
            print0("WARNING: model '%s' also branches at a crossing whose time moves, "
                   "and the difference quotient integrates straight through it, so "
                   "every sensitivity column is wrong at and after that crossing by "
                   "the jump it drops. bngsim refuses this case outright from 0.14.0; "
                   "if this fit proceeds, validate against a finite difference of the "
                   "trajectory before relying on it." % model.name)

    def _condition_for_suffix(self, model, suffix):
        """Resolve a scored ``suffix`` to the condition (``MutationSet``) it was
        simulated under, or ``None`` for the wildtype.

        An edition-2 ``condition:`` is a named :class:`~pybnf.pset.MutationSet` added
        to the model as a mutant (its name is the suffix); a mutant simulation's
        output suffix carries the mutant's own suffix (``net_model.execute``), so a
        scored suffix that ends with a known mutant suffix was simulated under that
        condition. The wildtype experiment touches no mutant and maps to ``None`` (the
        unperturbed routing, all factors 1)."""
        best = None
        for mut in getattr(model, 'mutants', []) or []:
            ms = getattr(mut, 'suffix', '')
            if ms and suffix.endswith(ms) and (best is None or len(ms) > len(best.suffix)):
                best = mut
        return best

    # --- per-evaluation assembly ------------------------------------------- #
    def gradient_at(self, res):
        """Assemble the objective gradient + residual-Jacobian at ``res``'s point.

        ``res`` is a master-scored Result, so ``res.simdata`` carries each
        experiment's forward-sensitivity tensor. Aligns it with ``exp_data`` over the
        scored ``(model, suffix)`` pairs (the same intersection the objective scores),
        attaches each one's prebuilt routing, and returns the assembled
        :class:`~pybnf.gradient.assembly.GradientResult` -- residual / Jacobian /
        scalar gradient in sampling space ``u`` (#385 transformed it once; the
        optimizer never re-transforms). Any constraint-penalty gradient is added to
        the scalar ``gradient`` (and clears ``least_squares_exact``, since a penalty
        is not a sum of squares).

        The free-parameter list (column order + current values) is read straight off
        the evaluated PSet, so the ``d theta/d u`` scale factors are taken at the
        point actually simulated -- and, for the same reason, each routing is taken
        ``at_point``: a chain-rule factor that reads other model symbols
        (``d(beta_N)/d(R0_) = gamma_/N_``, #530) is only a number once the fit vector
        is known. A routing whose factors are all constants returns itself, so every
        other fit is untouched. Converts a :class:`GradientNotSupported` (an
        objective the assembly cannot differentiate) into a clear, fail-fast
        :class:`PybnfError` pointing at a metaheuristic job_type."""
        free_params = [res.pset.get_param(v.name) for v in self.variables]
        try:
            routings = self._routings_at(res.pset)
        except GradientNotSupported as e:
            raise self._unsupported_gradient_error(e) from e
        experiments = []
        for model_name, by_suffix in res.simdata.items():
            model_exp = self.exp_data.get(model_name, {})
            for suffix, sim_data in by_suffix.items():
                if suffix in model_exp:
                    # The suffix travels with the experiment as its ``data_key`` -- the same key
                    # ``evaluate`` resolves a per-series analytic scale against (ADR-0066, #533),
                    # so the gradient profiles the scale over exactly the series scoring does.
                    experiments.append(
                        (sim_data, model_exp[suffix], routings[(model_name, suffix)], suffix))
        try:
            grad = self._assemble_objective_gradient(experiments, free_params)
            if self.config.constraints:
                cgrad = assemble_constraint_gradient(
                    self.config.constraints, res.simdata, routings, free_params)
                grad.gradient = grad.gradient + cgrad
                grad.least_squares_exact = False
            self._attach_curvature(grad, res, experiments, free_params, routings)
        except GradientNotSupported as e:
            raise self._unsupported_gradient_error(e) from e
        return grad

    def _routings_at(self, evaluated_pset):
        """The prebuilt routings with every point-dependent chain-rule factor resolved.

        The sensitivity *request* is fixed for the whole fit, but a seed derivative may
        read other model symbols and so is only a number at an evaluated point (#530).
        Returns ``self._routings`` itself when nothing is point-dependent -- the common
        case, and byte-identical to the pre-#530 path."""
        if not any(r.is_point_dependent for r in self._routings.values()):
            return self._routings
        values = {p.name: p.value for p in evaluated_pset}
        return {key: routing.at_point(values)
                for key, routing in self._routings.items()}

    def _assemble_objective_gradient(self, experiments, free_params):
        """Assemble the data-fit objective derivatives needed by this optimizer leaf.

        The base ``trf`` / ``lbfgs`` path needs only the scalar gradient and residual model.
        ``gntr`` overrides this seam to assemble those values and its Fisher Hessian in one
        scored-point pass (#488).

        A **marginal-time** objective (``time_error``, ADR-0113) scores each datum by an integral
        over the trajectory rather than at a matched row, so it assembles its own scalar gradient
        by sensitivity-chaining over the stored trajectory (``assemble_marginal_time_gradient``)
        instead of the matched-row Gaussian path. It is never a sum of squares, so the result is
        ``least_squares_exact = False`` -- ``lbfgs`` consumes its scalar gradient; ``trf`` (which
        needs an exact residual) refuses it and points at ``lbfgs``.
        """
        if getattr(self.objective, 'marginalizes_time', False):
            return assemble_marginal_time_gradient(self.objective, experiments, free_params)
        return assemble_gaussian_gradient(self.objective, experiments, free_params)

    def _attach_curvature(self, grad, res, experiments, free_params, routings):
        """Hook for a curvature-consuming leaf to attach its Hessian to the assembled
        gradient. A **no-op on the base**, so the residual-form (``trf``) and scalar-gradient
        (``lbfgs``) leaves -- which never form a Hessian -- are byte-identical; the EFIM
        trust-region leaf (``job_type = gntr``, #481) receives the data-fit Hessian from its
        combined :meth:`_assemble_objective_gradient` override, then uses this hook to add, for a
        constrained fit, :func:`~pybnf.gradient.assembly.assemble_constraint_hessian`. Called **inside**
        :meth:`gradient_at`'s :class:`GradientNotSupported` guard, so an unsupported-curvature
        corner (a MEDIAN-count Fisher, a MEAN-on-log estimated scale, an estimated constraint
        scale, ...) converts to the same fail-fast :class:`PybnfError`. ``routings`` are the
        per-experiment routings **at this point** (:meth:`_routings_at`), the same objects the
        objective assembly saw, so a point-dependent chain-rule factor (#530) reaches the
        constraint block too."""

    def _unsupported_gradient_error(self, exc):
        """Wrap a :class:`GradientNotSupported` as the leaf's fail-fast :class:`PybnfError`
        with an actionable fallback hint. The base points at a metaheuristic ``job_type``;
        the EFIM leaf (``gntr``) overrides the hint to point at ``lbfgs`` -- which consumes the
        scalar gradient and needs no Fisher Hessian, so it fits the very corners ``gntr``
        refuses."""
        return PybnfError(
            "Gradient-based fitting (job_type = %s) cannot differentiate this "
            "fit's objective: %s" % (self._fit_type_label(), exc),
            hint=_FALLBACK_HINT)

    # --- u-space box ------------------------------------------------------- #
    def _u_bounds(self):
        """The reflecting box in sampling space ``u`` as ``(lower, upper)`` arrays,
        ordered by ``self.variables``.

        Finite ``[to_sampling_space(lower_bound), to_sampling_space(upper_bound)]`` for
        a bounded (``uniform_var`` / ``loguniform_var``) parameter; ``(-inf, +inf)``
        for the unbounded ``var`` / ``logvar`` of a point start. The same box Powell
        confines its line search to (#412); a leaf projects or reflects its proposed
        step into it."""
        lower, upper = [], []
        for v in self.variables:
            if v.bounded:
                lower.append(v.to_sampling_space(v.lower_bound))
                upper.append(v.to_sampling_space(v.upper_bound))
            else:
                lower.append(-np.inf)
                upper.append(np.inf)
        return np.array(lower, dtype=float), np.array(upper, dtype=float)
