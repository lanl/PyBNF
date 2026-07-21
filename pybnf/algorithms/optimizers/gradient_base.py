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
  fit is refused -- with a clear message pointing at a metaheuristic ``fit_type`` --
  on a legacy (edition < 2) config, a non-bngsim model, or a bngsim build without
  the ``output_sensitivities`` feature. Never a silent finite-difference fallback.
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

import numpy as np

from .concurrent_multistart import DONE, ConcurrentMultiStartOptimizer
from ...gradient import (
    GradientNotSupported,
    apply_routing,
    assemble_constraint_gradient,
    assemble_gaussian_gradient,
    route_for_model,
)
from ...printing import PybnfError, print1, print2

# ``DONE`` is the shared multi-start sentinel, re-exported here so a gradient leaf's
# ``from .gradient_base import DONE`` keeps resolving to the one object the base's
# ``got_result`` identity-checks (#500).
__all__ = ['DONE', 'GradientRunner', 'GradientOptimizer']


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
        :meth:`GradientOptimizer._advance` (#492)."""
        raise NotImplementedError

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
        self.stop_reason = ('start point failed to simulate (a non-integrable point); '
                            'no objective/gradient to descend from')
        return DONE

    def progress_detail(self):
        """A short method-specific status suffix for the per-iteration report."""
        return ''


# The one-line suggestion every gradient-path refusal ends with, so a user whose
# model/config cannot be differentiated is pointed straight at a working fit_type
# rather than left to guess. Gradient fitting is strictly opt-in (fit_type = trf /
# lbfgs); a metaheuristic always works on the same config.
_FALLBACK_HINT = (
    "Gradient-based fitting is not available for this fit; use a metaheuristic "
    "fit_type instead (e.g. fit_type = de, the default, or pso / ss / cmaes), "
    "which needs no gradient."
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
        # Backend gate: every model must expose bngsim's forward-sensitivity hooks
        # (the capability gate itself fires later, at apply_routing).
        self._require_sensitivity_backend()
        # Differentiability gate: a discrete-event model has no smooth forward
        # sensitivity (bngsim refuses one), so refuse it now rather than mid-run (#461).
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
    # The gradient path is gated in four places, each as early as it can be:
    #
    # * **edition** (:meth:`_require_edition_2`, before model build) -- the gradient
    #   consumes the edition-2 surface (bind-by-id routing, the noise-model /
    #   measurement layer), absent under legacy edition 1;
    # * **backend** (:meth:`_require_sensitivity_backend`, after model build) -- every
    #   model must expose bngsim's forward-sensitivity hooks; a non-bngsim (e.g.
    #   RoadRunner/SBML) model has no sensitivity tensor here;
    # * **differentiability** (:meth:`_require_differentiable_dynamics`, after model
    #   build, #461) -- a discrete-event model has no smooth forward sensitivity
    #   (bngsim refuses one, GH #205), so refuse it here rather than fail mid-run;
    # * **capability** (deferred to :meth:`_setup_gradient_path`'s ``apply_routing``,
    #   #447) -- raises if the bngsim build lacks the ``output_sensitivities`` feature.
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
    # by contrast, are a build-time structural signal and so *can* be a pre-flight gate.
    def _require_edition_2(self, config):
        """Refuse a legacy (edition < 2) config before any model is built."""
        edition = config.config.get('edition')
        if not edition or edition < 2:
            raise PybnfError(
                "Gradient-based fitting (fit_type = %s) requires the edition-2 "
                "config surface, but this fit is %s." % (
                    self._fit_type_label(),
                    "edition 1 (legacy)" if not edition else "edition %d" % edition),
                "Opt into edition 2 ('edition = 2') and declare the fit on the "
                "new-era surface (experiment: / data: / noise_model, bind-by-id "
                "parameters). " + _FALLBACK_HINT)

    def _require_sensitivity_backend(self):
        """Refuse a model whose backend has no forward-sensitivity hooks."""
        for model in self.model_list:
            if not hasattr(model, 'enable_output_sensitivities'):
                raise PybnfError(
                    "Gradient-based fitting (fit_type = %s) requires the bngsim "
                    "backend's forward sensitivities, but model '%s' uses a backend "
                    "that does not provide them." % (
                        self._fit_type_label(), getattr(model, 'name', '?')),
                    _FALLBACK_HINT)

    def _require_differentiable_dynamics(self):
        """Refuse a discrete-event model before the run (#461).

        A discrete event is a state-dependent discrete jump in the dynamics; it
        reinitialises the integrator state discontinuously, but bngsim's CVODES
        forward-sensitivity vectors are not reinitialised across the jump, so the
        sensitivities go silently stale -- bngsim refuses forward output
        sensitivities outright on such a model (GH #205). Without this gate that
        refusal would surface only mid-run, at the first sensitivity-bearing
        ``simulate()`` (caught and re-raised in ``BngsimModel.execute``). Fired here
        next to :meth:`_require_sensitivity_backend`, it gives the discrete-event
        model the same clean pre-flight "use a metaheuristic fit_type" refusal the
        other gates give. Models whose backend exposes no event count
        (``has_discrete_events`` absent) pass through untouched."""
        for model in self.model_list:
            if getattr(model, 'has_discrete_events', False):
                raise PybnfError(
                    "Gradient-based fitting (fit_type = %s) requires smooth, "
                    "differentiable dynamics, but model '%s' contains discrete "
                    "events (a state-dependent discrete jump). Forward output "
                    "sensitivities go stale across such a jump, so bngsim cannot "
                    "supply the gradient there." % (
                        self._fit_type_label(), getattr(model, 'name', '?')),
                    _FALLBACK_HINT)

    def _fit_type_label(self):
        """The fit_type code for messages (the leaf's registered name, best-effort)."""
        return getattr(self, 'fit_type', type(self).__name__)

    # --- gradient-path activation ------------------------------------------ #
    def _setup_gradient_path(self):
        """Enable forward sensitivities on every model and build the per-experiment
        routings -- idempotent, called once from the leaf's ``start_run`` (before the
        model scatter, so the request rides the pickle to the workers).

        For each model: ``apply_routing`` the **wildtype** routing, whose request
        lists (``sensitivity_params`` / ``sensitivity_ic``) are the union over
        conditions -- the wildtype pins nothing, so every parameter-/IC-bound free
        parameter is requested, a superset of any single condition's (a ``=``-pinned
        condition only ever *drops* columns). Then build one
        :class:`ExperimentRouting` per scored ``(model, suffix)``, carrying that
        condition's chain-rule factors for the assembly. Raises (capability gate) if
        the bngsim build lacks ``output_sensitivities``."""
        if self._routings is not None:
            return
        names = [v.name for v in self.variables]
        routings = {}
        for model in self.model_list:
            # Union sensitivity request (wildtype) -> sets _sensitivity_request,
            # which survives the scatter and is applied at every simulate().
            apply_routing(model, route_for_model(model, names, condition=None))
            # Declare which of this model's outputs are scored gradient targets so
            # an incidental/unscored action (a stochastic diagnostic, a
            # carried-state pre-equilibration scan) runs sensitivity-free instead
            # of aborting the whole fit at a differentiability guard (#475). Rides
            # the scatter alongside the sensitivity request.
            model.set_scored_suffixes(self.exp_data.get(model.name, {}))
            for suffix in self.exp_data.get(model.name, {}):
                condition = self._condition_for_suffix(model, suffix)
                routings[(model.name, suffix)] = route_for_model(model, names, condition)
        self._routings = routings

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
        point actually simulated. Converts a :class:`GradientNotSupported` (an
        objective the assembly cannot differentiate) into a clear, fail-fast
        :class:`PybnfError` pointing at a metaheuristic fit_type."""
        free_params = [res.pset.get_param(v.name) for v in self.variables]
        experiments = []
        for model_name, by_suffix in res.simdata.items():
            model_exp = self.exp_data.get(model_name, {})
            for suffix, sim_data in by_suffix.items():
                if suffix in model_exp:
                    experiments.append(
                        (sim_data, model_exp[suffix], self._routings[(model_name, suffix)]))
        try:
            grad = self._assemble_objective_gradient(experiments, free_params)
            if self.config.constraints:
                cgrad = assemble_constraint_gradient(
                    self.config.constraints, res.simdata, self._routings, free_params)
                grad.gradient = grad.gradient + cgrad
                grad.least_squares_exact = False
            self._attach_curvature(grad, res, experiments, free_params)
        except GradientNotSupported as e:
            raise self._unsupported_gradient_error(e) from e
        return grad

    def _assemble_objective_gradient(self, experiments, free_params):
        """Assemble the data-fit objective derivatives needed by this optimizer leaf.

        The base ``trf`` / ``lbfgs`` path needs only the scalar gradient and residual model.
        ``gntr`` overrides this seam to assemble those values and its Fisher Hessian in one
        scored-point pass (#488).
        """
        return assemble_gaussian_gradient(self.objective, experiments, free_params)

    def _attach_curvature(self, grad, res, experiments, free_params):
        """Hook for a curvature-consuming leaf to attach its Hessian to the assembled
        gradient. A **no-op on the base**, so the residual-form (``trf``) and scalar-gradient
        (``lbfgs``) leaves -- which never form a Hessian -- are byte-identical; the EFIM
        trust-region leaf (``fit_type = gntr``, #481) receives the data-fit Hessian from its
        combined :meth:`_assemble_objective_gradient` override, then uses this hook to add, for a
        constrained fit, :func:`~pybnf.gradient.assembly.assemble_constraint_hessian`. Called **inside**
        :meth:`gradient_at`'s :class:`GradientNotSupported` guard, so an unsupported-curvature
        corner (a MEDIAN-count Fisher, a MEAN-on-log estimated scale, an estimated constraint
        scale, ...) converts to the same fail-fast :class:`PybnfError`."""

    def _unsupported_gradient_error(self, exc):
        """Wrap a :class:`GradientNotSupported` as the leaf's fail-fast :class:`PybnfError`
        with an actionable fallback hint. The base points at a metaheuristic ``fit_type``;
        the EFIM leaf (``gntr``) overrides the hint to point at ``lbfgs`` -- which consumes the
        scalar gradient and needs no Fisher Hessian, so it fits the very corners ``gntr``
        refuses."""
        return PybnfError(
            "Gradient-based fitting (fit_type = %s) cannot differentiate this "
            "fit's objective: %s" % (self._fit_type_label(), exc),
            _FALLBACK_HINT)

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
