"""Shared scaffolding for the gradient-based local optimizers (TRF/LM + L-BFGS-B, #386).

The metaheuristic fit types (de, pso, ss, cmaes, …) only ever ask each evaluated
``PSet`` for its scalar objective value. A **gradient** optimizer instead consumes
the residual vector + residual-Jacobian (TRF / Levenberg–Marquardt) or the scalar
gradient (L-BFGS-B) that #385 assembles from bngsim's forward output-sensitivity
tensor. :class:`GradientOptimizer` factors out everything a new gradient method
needs so a leaf (``trf.py``, ``lbfgs.py``) implements only its step math --
mirroring how :class:`StartPointOptimizer` factors the start-point / ``u``↔PSet
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

import logging

import numpy as np

from .local_base import StartPointOptimizer
from ...gradient import (
    GradientNotSupported,
    apply_routing,
    assemble_constraint_gradient,
    assemble_gaussian_gradient,
    route_for_model,
)
from ...printing import PybnfError, print1, print2

logger = logging.getLogger('pybnf.algorithms')


#: Sentinel a :class:`GradientRunner`'s ``got`` returns when that start has
#: converged or otherwise terminated (no further evaluation to propose). It is
#: only ever returned synchronously into :meth:`GradientOptimizer.got_result` and
#: never stored in pickled state, so a plain module-level object (identity-checked)
#: is enough.
DONE = object()


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
        """Consume one completed evaluation and return the next ``u`` or :data:`DONE`."""
        raise NotImplementedError

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


class GradientOptimizer(StartPointOptimizer):
    """Base for the gradient-based local optimizers (#386).

    A leaf subclass supplies only its per-start step math as a :class:`GradientRunner`
    (Levenberg–Marquardt for ``trf``, L-BFGS-B for ``lbfgs``) via :meth:`_make_runner`;
    this base owns the whole ``start_run`` / ``got_result`` orchestration -- seeding the
    runners, the ``u``<->PSet plumbing, :meth:`gradient_at` assembly, name-tagging /
    routing, reporting, and the multi-start ``STOP`` coordination. The leaf must set
    :attr:`START_POINT_KEY` like any :class:`StartPointOptimizer`, plus
    :attr:`_method_label` and :meth:`_start_banner` for its messages.

    Local multi-start (#386). A box-start gradient fit runs ``N`` independent starts
    concurrently (``N`` reuses ``population_size``) -- start 0 from the box center
    (preserving the deterministic single-start behavior), the rest from Latin-hypercube
    samples across the prior box -- and keeps the global best. The async run loop already
    supports this without change: ``start_run`` returns a list of ``N`` initial PSets,
    each ``got_result`` advances just the start that owns the returned Result (routed by
    PSet name through :attr:`pending`) and returns that start's next PSet (or ``[]`` once
    it converges), and ``'STOP'`` fires only when the **last** start finishes. Every
    evaluated PSet across all starts lands in the trajectory (``add_to_trajectory`` runs
    before ``got_result``), so ``trajectory.best_fit()`` is the global best for free --
    each runner only tracks its own best for its own convergence test.
    """

    #: Keep objective scoring on the master so every Result returns with its
    #: simdata (the sensitivity tensors the gradient assembly reads); see
    #: ``Algorithm.run``. Without this the worker path nulls ``res.simdata``.
    requires_master_scoring = True

    #: Human label for the method in the per-iteration progress messages; set by
    #: each leaf (e.g. ``'L-BFGS-B'`` / ``'TRF'``).
    _method_label = 'gradient'

    def __init__(self, config, refine=False):
        # Cheapest gate first, before the (expensive) network generation in
        # Algorithm.__init__: a legacy-edition config can never carry the gradient
        # surface, so refuse it before building a single model.
        self._require_edition_2(config)
        super().__init__(config)
        self.refine = refine
        # Per-experiment routing, keyed by (model_name, suffix); built lazily in
        # _setup_gradient_path (needs the initialized models). None until then, and
        # restored as None by reset() so a bootstrap refit rebuilds it.
        self._routings = None
        # Backend gate: every model must expose bngsim's forward-sensitivity hooks
        # (the capability gate itself fires later, at apply_routing).
        self._require_sensitivity_backend()
        # Multi-start setup: the reflecting box, the start-point count + the start
        # PSets, and the (empty) orchestration state. The per-start runners are built
        # lazily in start_run (they need the leaf's tunables, read after this returns).
        self.n = len(self.variables)
        self._u_lower, self._u_upper = self._u_bounds()
        self.n_starts = self._resolve_n_starts()
        self.start_psets = self._resolve_start_psets()
        self._init_orchestration()

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._routings = None
        self._u_lower, self._u_upper = self._u_bounds()
        self.n_starts = self._resolve_n_starts()
        self.start_psets = self._resolve_start_psets()
        self._init_orchestration()

    # --- multi-start orchestration ----------------------------------------- #
    def _resolve_n_starts(self):
        """The number of independent start points for this fit.

        Local multi-start (#386) reuses ``population_size`` as the start count --
        consistent with the metaheuristics, where it is the parallel-population size,
        and ``population_size = 1`` reproduces the historical single start. A
        point-start or refiner-injected start has no prior box to scatter across, so it
        always runs a single start (the refiner polishes the one best fit, it does not
        re-scatter). :meth:`_is_box_start` is exactly that test (no injected start +
        every variable bounded)."""
        if not self._is_box_start():
            return 1
        return max(1, int(self.config.config.get('population_size', 1)))

    def _resolve_start_psets(self):
        """The ``n_starts`` start PSets: the box center first (start 0, preserving the
        deterministic single-start behavior and the parity tests), then ``n_starts - 1``
        Latin-hypercube samples across the prior box drawn from the seeded ``self.rng``
        (so the scatter reproduces from ``random_seed``). With ``n_starts == 1`` no
        sample is drawn -- the rng is untouched -- so a single-start fit is byte-for-byte
        unchanged."""
        start0 = self._resolve_start_pset()
        if self.n_starts <= 1:
            return [start0]
        return [start0] + self.random_latin_hypercube_psets(self.n_starts - 1)

    def _init_orchestration(self):
        """(Re)initialize the multi-start bookkeeping -- all plain list/dict/int, so the
        optimizer pickles for backup/resume (ADR-0007). The per-start runners are built
        lazily in :meth:`start_run` (they need the leaf's tunables, set after
        ``super().__init__``); until then ``runners`` is empty, so a freshly constructed
        optimizer still round-trips through pickle."""
        self.runners = []        # one GradientRunner per start (built in start_run)
        self.pending = {}        # pset name -> owning runner index (the routing map)
        self.probe_counter = 0   # global submission counter -> unique pset names
        self.active = 0          # starts not yet converged/terminated

    def add_iterations(self, n):
        """Extend every start's per-start iteration budget by ``n`` (the ``-r`` resume
        path). Runners already exist when this is called on a resumed run (they ride the
        backup pickle), so bump each alongside the template budget."""
        self.max_iterations += n
        for runner in self.runners:
            runner.max_iterations += n

    def start_run(self):
        """Seed the ``n_starts`` runners and return their initial PSets (one evaluation
        each), so all starts run concurrently from the first scheduler batch."""
        print2(self._start_banner())
        # Activate the gradient path (enable sensitivities + build routings) before the
        # model scatter, so the request rides the pickle to the workers.
        self._setup_gradient_path()
        self.runners = [self._make_runner(self._u_from_pset(p)) for p in self.start_psets]
        self.active = len(self.runners)
        self.probe_counter = 0
        self.pending = {}
        return [self._dispatch(idx, runner.start())
                for idx, runner in enumerate(self.runners)]

    def got_result(self, res):
        """Route a completed Result to the start that owns it, advance just that start's
        runner, and return its next PSet -- ``[]`` once it converges (other starts keep
        going), or ``'STOP'`` only when the last live start finishes."""
        idx = self.pending.pop(res.name)
        runner = self.runners[idx]
        u_point = self._u_from_pset(res.pset)
        grad = self.gradient_at(res)
        prev_iter = runner.iteration
        nxt = runner.got(u_point, float(res.score), grad)
        if runner.iteration > prev_iter:
            self._report(runner)
        if nxt is DONE:
            logger.info('%s start %d/%d stopping: %s', self._method_label,
                        idx + 1, len(self.runners), runner.stop_reason)
            self.active -= 1
            if self.active == 0:
                return 'STOP'
            return []
        return [self._dispatch(idx, nxt)]

    def _dispatch(self, idx, u):
        """Wrap a runner's proposed ``u``-point as a uniquely named PSet bound to its
        owning start in :attr:`pending`, and return it for submission. The name carries a
        single global counter (``<fit_type>_<k>``), so a single-start fit reproduces the
        historical ``<fit_type>_1``, ``<fit_type>_2``, … sequence exactly while every
        name stays unique across concurrent starts (the routing key)."""
        self.probe_counter += 1
        name = '%s_%i' % (self.fit_type, self.probe_counter)
        self.pending[name] = idx
        return self._pset_from_u(u, name=name)

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
    # The gradient path is gated in three places, each as early as it can be:
    #
    # * **edition** (:meth:`_require_edition_2`, before model build) -- the gradient
    #   consumes the edition-2 surface (bind-by-id routing, the noise-model /
    #   measurement layer), absent under legacy edition 1;
    # * **backend** (:meth:`_require_sensitivity_backend`, after model build) -- every
    #   model must expose bngsim's forward-sensitivity hooks; a non-bngsim (e.g.
    #   RoadRunner/SBML) model has no sensitivity tensor here;
    # * **capability** (deferred to :meth:`_setup_gradient_path`'s ``apply_routing``,
    #   #447) -- raises if the bngsim build lacks the ``output_sensitivities`` feature.
    #
    # The per-evaluation gate (an unsupported *objective* -- Laplace residual,
    # estimated scale, … raising :class:`GradientNotSupported`) is caught at the first
    # assembly in :meth:`gradient_at`.
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
            grad = assemble_gaussian_gradient(self.objective, experiments, free_params)
            if self.config.constraints:
                cgrad = assemble_constraint_gradient(
                    self.config.constraints, res.simdata, self._routings, free_params)
                grad.gradient = grad.gradient + cgrad
                grad.least_squares_exact = False
        except GradientNotSupported as e:
            raise PybnfError(
                "Gradient-based fitting (fit_type = %s) cannot differentiate this "
                "fit's objective: %s" % (self._fit_type_label(), e),
                _FALLBACK_HINT) from e
        return grad

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
