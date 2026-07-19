"""Shared concurrent local multi-start scaffolding (#500, ADR-0009).

Both the gradient-based local optimizers (``trf`` / ``lbfgs`` / ``gntr``,
:class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer`, #386) and the
derivative-free local optimizers (``powell`` / ``sim``,
:class:`~pybnf.algorithms.optimizers.local_multistart.LocalMultiStartOptimizer`, #498)
solve the same problem the same way: a single local search descends into whichever basin
its start lands in, so on a multimodal objective it returns only a *local* minimum. The
remedy is identical for both families -- run ``N`` independent starts **concurrently**
(the way that saturates the cluster, not sequential restarts), keep the global best -- and
so is the run-loop machinery that drives it: seed ``N`` headless per-start step machines,
emit every start's opening job(s) on the first scheduler batch, route each returned Result
to the start that owns it (by PSet name), advance only that start, and ``STOP`` only when
the *last* live start terminates.

The two families were shipped as standalone siblings (``GradientOptimizer`` first, then
``LocalMultiStartOptimizer`` deliberately kept separate so the heavily-exercised gradient
path was untouched for #498). They ended up sharing ~40 lines of near-identical
orchestration -- the ">=2-member event" ADR-0009 says earns a shared base, the same
rationale that produced :class:`~pybnf.algorithms.optimizers.local_base.StartPointOptimizer`
(which factored the start-point / ``u`` <-> :class:`~pybnf.pset.PSet` plumbing out of Powell
and CMA-ES). :class:`ConcurrentMultiStartOptimizer` is that base.

What the base owns
------------------
* **Box-gated start resolution** (:meth:`_resolve_n_starts` / :meth:`_resolve_start_psets`):
  a point-start / refiner fit runs a single start; a standalone box fit runs ``N`` starts --
  start 0 the box center (so the single-start behavior, and the parity tests, are preserved
  byte-for-byte), the rest Latin-hypercube samples across the prior box, drawn from the
  seeded ``self.rng`` so the scatter reproduces from ``random_seed``.
* **The orchestration state** (:meth:`_init_orchestration`): the ``runners`` list, the
  ``pending`` name->start routing map, the global ``probe_counter``, and the ``active``
  count -- all plain list/dict/int, so the optimizer pickles for backup/resume (ADR-0007).
* **The run loop** (:meth:`start_run` / :meth:`got_result` / :meth:`_route`): seeding,
  routing, per-iteration reporting, and the ``STOP``-on-last-start coordination.
* **Resume plumbing** (:meth:`add_iterations`) and the ``__init__`` / :meth:`reset`
  skeleton.

The family differences, behind hooks
------------------------------------
A subclass (:class:`GradientOptimizer` / :class:`LocalMultiStartOptimizer`) fills in only
what genuinely differs:

* :meth:`_build_runners` -- how ``start_psets`` become the per-start step machines. The
  gradient path converts each PSet to a ``u``-vector and is deterministic; the local path
  spawns one :class:`numpy.random.Generator` per start (Simplex's degeneracy perturbation
  draws from it) so the base stays rng-agnostic.
* :meth:`_seed` ``(idx, runner) -> list[PSet]`` -- one start's initial job(s), named + routed.
* :meth:`_advance` ``(idx, runner, res) -> list[PSet] | DONE`` -- feed the completed ``res``
  to its runner and return the next job(s), or :data:`DONE` when that start has terminated.
  This is where the families diverge most: a gradient runner consumes an assembled gradient
  (``runner.got(u, score, grad)``), a local runner only ``(u, score)`` / ``(pset, score)``.
* :meth:`_report` ``(runner)`` -- the per-iteration progress line for one start.
* :meth:`_make_runner` / :meth:`_start_banner` -- the leaf's step machine + "running ..."
  banner.
* :meth:`_pre_seed` (no-op default) -- a pre-seed side effect (the gradient path enables
  forward sensitivities + builds routings here; the local path has none).
* :meth:`_announce_starts` (no-op default) -- the optional "concurrent multi-start" status
  line (local only; the gradient path stays silent, byte-for-byte).
* :meth:`_check_config_supported` / :meth:`_after_init` / :meth:`_after_reset` (no-op
  defaults) -- the gradient path's pre-flight gates, sensitivity routings, and reflecting
  box, hung off the construction / reset skeleton without duplicating it.

Two small differences are class attributes rather than hooks: :attr:`_n_starts_key` (the
gradient path predates the ``n_starts`` field and reuses ``population_size``) and
:attr:`_stop_verb` (the log phrasing).

A single-start fit (the default, or any point-start / refiner fit) builds one runner whose
PSet names -- and therefore search -- are byte-identical to the pre-multi-start behavior.
The global best is kept for free: every evaluated PSet across every start lands in the
trajectory (``add_to_trajectory`` runs on the master before ``got_result``), so
``trajectory.best_fit()`` is the best over all starts; each runner only tracks its own best
for its own convergence test.
"""

import logging

from .local_base import StartPointOptimizer
from ...printing import print2

logger = logging.getLogger('pybnf.algorithms')


#: Sentinel a runner's ``got`` returns (surfaced through :meth:`_advance`) when that start
#: has terminated -- converged or hit its per-start iteration budget. Only ever returned
#: synchronously into :meth:`got_result`, never stored in pickled state, so a plain
#: identity-checked module object is enough. Re-exported by both ``gradient_base`` and
#: ``local_multistart`` so every leaf's ``from .<family> import DONE`` resolves to this one
#: object (the ``out is DONE`` check in :meth:`got_result` is identity-based).
DONE = object()


class ConcurrentMultiStartOptimizer(StartPointOptimizer):
    """Base for the concurrent local multi-start optimizers (#500).

    Owns the whole ``start_run`` / ``got_result`` orchestration -- seeding the runners, the
    name routing, the ``STOP``-on-last-start coordination, and the box-gated box-center +
    Latin-hypercube start-point scheme. A subclass supplies its per-start step math + PSet
    plumbing through the hooks documented on the module. See the module docstring for the
    full contract; this class is never instantiated directly (both ``__init__`` and the
    run loop delegate to hooks a leaf must implement)."""

    #: Human label for the method in progress / log messages; set by each leaf.
    _method_label = 'local'

    #: The config key the box-fit start count is read from. Defaults to the newer
    #: ``n_starts`` field (:class:`~pybnf.algorithms.optimizers.multistart.MultiStartConfig`,
    #: #498); the gradient path predates it and overrides this to ``'population_size'`` (#386).
    _n_starts_key = 'n_starts'

    #: The verb :meth:`got_result` logs when a start terminates (``'finished'`` for the
    #: local path, ``'stopping'`` for the gradient path) -- cosmetic, preserved verbatim.
    _stop_verb = 'finished'

    def __init__(self, config, refine=False):
        # A subclass gate that must run *before* the (expensive) network generation in
        # Algorithm.__init__ -- the gradient path refuses a legacy-edition config here, so
        # it never builds a model it cannot differentiate. No-op by default.
        self._check_config_supported(config)
        super().__init__(config)
        self.refine = refine
        self.n = len(self.variables)
        # Subclass construction extras (the gradient path's sensitivity gates + reflecting
        # box); no-op by default. Runs before start resolution -- neither depends on the
        # other, but the gates are cheapest-first.
        self._after_init()
        # Multi-start setup: the start count + the start PSets, and the (empty)
        # orchestration state. The per-start runners are built lazily in start_run (they
        # need the leaf's tunables, read after this returns), so a freshly constructed
        # optimizer round-trips through pickle with an empty runner list.
        self._resolve_starts()

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._after_reset()
        self._resolve_starts()

    # --- construction / reset hooks (no-op defaults) ----------------------- #
    def _check_config_supported(self, config):
        """Pre-flight gate run *before* ``super().__init__`` (before any model is built).
        No-op on the base; the gradient path refuses a legacy-edition config here."""

    def _after_init(self):
        """Subclass construction extras, run after ``super().__init__`` + ``self.n`` and
        before start resolution. No-op on the base; the gradient path runs its
        sensitivity-backend / differentiability gates and builds its reflecting box here."""

    def _after_reset(self):
        """Subclass reset extras, run after ``super().reset`` and before start resolution.
        No-op on the base; the gradient path clears its routings and rebuilds its box here."""

    # --- start resolution -------------------------------------------------- #
    def _resolve_starts(self):
        """(Re)compute the start count + start PSets and (re)initialize the orchestration
        bookkeeping. Shared by ``__init__`` and :meth:`reset`."""
        self.n_starts = self._resolve_n_starts()
        self.start_psets = self._resolve_start_psets()
        self._init_orchestration()

    def _resolve_n_starts(self):
        """The number of independent starts for this fit. A point-start or refiner-injected
        start has no prior box to scatter across, so it always runs a single start (the
        refiner polishes the one best fit, it does not re-scatter). Only a standalone box
        fit (bounded priors, :meth:`_is_box_start`) reads the start count, from
        :attr:`_n_starts_key` (floored at 1)."""
        if not self._is_box_start():
            return 1
        return max(1, int(self.config.config.get(self._n_starts_key, 1)))

    def _resolve_start_psets(self):
        """The ``n_starts`` start PSets: the box center first (start 0, preserving the
        deterministic single-start behavior and the parity tests), then ``n_starts - 1``
        Latin-hypercube samples across the prior box drawn from the seeded ``self.rng`` (so
        the scatter reproduces from ``random_seed``). With ``n_starts == 1`` no sample is
        drawn -- the rng is untouched -- so a single-start fit is byte-for-byte unchanged."""
        start0 = self._resolve_start_pset()
        if self.n_starts <= 1:
            return [start0]
        return [start0] + self.random_latin_hypercube_psets(self.n_starts - 1)

    def _init_orchestration(self):
        """(Re)initialize the multi-start bookkeeping -- all plain list/dict/int, so the
        optimizer pickles for backup/resume (ADR-0007). The per-start runners are built
        lazily in :meth:`start_run`; until then ``runners`` is empty, so a freshly
        constructed optimizer still round-trips through pickle, and a white-box unit test
        that drives ``got_result`` after ``start_run`` finds every attribute in place."""
        self.runners = []        # one per-start step machine (built in start_run)
        self.pending = {}        # dispatched pset name -> owning runner index (routing map)
        self.probe_counter = 0   # global submission counter -> unique pset names
        self.active = 0          # starts not yet terminated

    def add_iterations(self, n):
        """Extend every start's per-start iteration budget by ``n`` (the ``-r`` resume path).
        Runners already exist when this is called on a resumed run (they ride the backup
        pickle), so bump each alongside the template budget."""
        self.max_iterations += n
        for runner in self.runners:
            runner.max_iterations += n

    # --- run-loop orchestration -------------------------------------------- #
    def start_run(self):
        """Seed the ``n_starts`` runners and return every start's initial job(s), so all
        starts run concurrently from the first scheduler batch."""
        print2(self._start_banner())
        self._announce_starts()
        # Pre-seed side effect (the gradient path enables forward sensitivities + builds
        # routings before the model scatter, so the request rides the pickle to the
        # workers); no-op on the local path.
        self._pre_seed()
        self.runners = self._build_runners()
        self.active = len(self.runners)
        self.probe_counter = 0
        self.pending = {}
        out = []
        for idx, runner in enumerate(self.runners):
            out.extend(self._seed(idx, runner))
        return out

    def got_result(self, res):
        """Route a completed Result to the start that owns it (by PSet name), advance just
        that start's runner, and return its next job(s) -- ``[]`` once it terminates (other
        starts keep going), or ``'STOP'`` only when the last live start finishes."""
        idx = self.pending.pop(res.name)
        runner = self.runners[idx]
        prev_iter = runner.iteration
        out = self._advance(idx, runner, res)
        if runner.iteration > prev_iter:
            self._report(runner)
        if out is DONE:
            logger.info('%s start %d/%d %s: %s', self._method_label,
                        idx + 1, len(self.runners), self._stop_verb, runner.stop_reason)
            self.active -= 1
            if self.active == 0:
                return 'STOP'
            return []
        return out

    def _route(self, idx, pset):
        """Record ``pset`` (already uniquely named by the leaf) as owned by start ``idx`` and
        return it for submission. The name is the routing key ``got_result`` reads back."""
        self.pending[pset.name] = idx
        return pset

    # --- run-loop hooks ---------------------------------------------------- #
    def _announce_starts(self):
        """Optional status line printed after the banner. No-op on the base; the local path
        announces its concurrent multi-start when ``n_starts > 1``."""

    def _pre_seed(self):
        """Side effect run once before the runners are seeded (before the model scatter).
        No-op on the base; the gradient path activates its sensitivity routings here."""

    def _build_runners(self):
        """Build the list of per-start step machines from :attr:`start_psets`. The rng /
        ``u``-conversion seam -- kept in the subclass so the base stays rng-agnostic."""
        raise NotImplementedError

    def _seed(self, idx, runner):
        """The initial job(s) for start ``idx``'s ``runner``, each named + routed to it."""
        raise NotImplementedError

    def _advance(self, idx, runner, res):
        """Feed the completed ``res`` to start ``idx``'s ``runner`` and return its next
        job(s) (named + routed), or :data:`DONE` when that start has terminated."""
        raise NotImplementedError

    def _report(self, runner):
        """The per-iteration progress line for one start (called when the runner's
        iteration count advances)."""
        raise NotImplementedError

    def _make_runner(self, *args, **kwargs):
        """Build one start's headless, picklable step machine (called from
        :meth:`_build_runners`). Signature is family-specific."""
        raise NotImplementedError

    def _start_banner(self):
        """The one-line "running ..." message printed when the run starts."""
        raise NotImplementedError
