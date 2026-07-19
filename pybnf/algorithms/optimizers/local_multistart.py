"""Concurrent local multi-start for the derivative-free local optimizers (``n_starts``, #498).

A single Powell or Simplex run descends into whichever basin its start lands in, so on a
multimodal objective it returns only a local minimum -- the same gap #386 closed for the
gradient methods. :class:`LocalMultiStartOptimizer` gives the *derivative-free* local
optimizers (``powell`` / ``sim``) the identical remedy, and does it the way that respects
PyBNF's ethos of saturating the cluster: **concurrent** starts, not sequential restarts.

Why concurrent, not the sequential ``MultiStartOptimizer`` mixin (ADR-0071)
---------------------------------------------------------------------------
The metaheuristics (``de`` / ``ss`` / ``pso``) already fan a whole population across the
cluster every generation, so running their ``n_starts`` restarts *sequentially* wastes
nothing -- one search already saturates the workers. A local method is the opposite:
Powell is strictly serial (one line-search probe in flight at a time = **one worker**), so
running ``n_starts`` Powell searches one after another would pin the cluster at a single
worker for ``n_starts x`` the wall-clock. Simplex fans out only ``parallel_count`` (roughly
``n_variables - 1``) probes per generation. So the local methods must run their starts
**concurrently** -- exactly what the gradient optimizers already do (``GradientOptimizer``,
#386): seed ``N`` independent per-start step machines, emit every start's first job on the
opening batch, advance only the start that owns each returned result, and stop only when the
*last* start terminates. Powell then uses ``N`` workers instead of one; Simplex uses
``N x parallel_count``.

This base is a standalone sibling of :class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer`;
they share ~40 lines of runner-list orchestration that a follow-up will factor into a common
``ConcurrentMultiStartOptimizer`` base (the >=2-member event, ADR-0009, that earned
:class:`~pybnf.algorithms.optimizers.local_base.StartPointOptimizer` in the first place). For
now it is kept separate so the shipped, heavily-exercised gradient path is untouched.

The leaf contract
-----------------
A leaf (``powell`` / ``sim``) mixes this in *before* :class:`StartPointOptimizer`
(``class PowellAlgorithm(LocalMultiStartOptimizer, StartPointOptimizer)``) so these
``start_run`` / ``got_result`` overrides win the MRO, sets :attr:`START_POINT_KEY` and
:attr:`_method_label`, and supplies its per-start step math + PSet plumbing through five
hooks:

* :meth:`_make_runner` ``(start_pset) -> runner`` -- build one start's headless, picklable
  step machine seeded at ``start_pset``.
* :meth:`_seed` ``(idx, runner) -> list[PSet]`` -- the runner's initial job(s), named + routed
  to start ``idx`` (via :meth:`_route`).
* :meth:`_advance` ``(idx, runner, res) -> list[PSet] | DONE`` -- feed the completed ``res``
  to the runner and return its next job(s) (named + routed), or :data:`DONE` when that start
  has terminated.
* :meth:`_report` ``(runner)`` -- the per-iteration progress line for one start (called when
  the runner's iteration count advances).
* :meth:`_start_banner` ``() -> str`` -- the one-line "running ..." message.

The global best is kept for free: every evaluated PSet across every start lands in the
trajectory (``add_to_trajectory`` runs on the master before ``got_result``), so
``trajectory.best_fit()`` is the best over all starts. ``n_starts == 1`` (the default, or
any point-start / refiner fit) is a single run whose PSet names -- and therefore search --
are byte-identical to the pre-multi-start behavior.
"""

import logging

from .local_base import StartPointOptimizer
from ...printing import print2

logger = logging.getLogger('pybnf.algorithms')


#: Sentinel a runner's ``got`` returns (surfaced through :meth:`_advance`) when that start
#: has terminated -- converged or hit its per-start iteration budget. Only ever returned
#: synchronously into :meth:`got_result`, never stored in pickled state, so a plain
#: identity-checked module object is enough (mirrors ``gradient_base.DONE``).
DONE = object()


class LocalMultiStartOptimizer(StartPointOptimizer):
    """Base for the derivative-free local optimizers with concurrent multi-start (#498).

    Owns the whole ``start_run`` / ``got_result`` orchestration -- seeding the runners, the
    name routing, the ``STOP``-on-last-start coordination, and the box-gated box-center +
    Latin-hypercube start-point scheme (lifted from ``GradientOptimizer``). A leaf supplies
    only its per-start step math + PSet plumbing through the hooks above.
    """

    #: Human label for the method in progress messages; set by each leaf (``'Powell'`` /
    #: ``'Simplex'``).
    _method_label = 'local'

    def __init__(self, config, refine=False):
        super().__init__(config)
        self.refine = refine
        self.n = len(self.variables)
        # Multi-start setup: the start count + the start PSets, and the (empty)
        # orchestration state. The per-start runners are built lazily in start_run (they
        # need the leaf's tunables, read after this returns), so a freshly constructed
        # optimizer round-trips through pickle with an empty runner list.
        self.n_starts = self._resolve_n_starts()
        self.start_psets = self._resolve_start_psets()
        self._init_orchestration()

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self.n_starts = self._resolve_n_starts()
        self.start_psets = self._resolve_start_psets()
        self._init_orchestration()

    # --- multi-start resolution (mirrors gradient_base) -------------------- #
    def _resolve_n_starts(self):
        """The number of independent starts for this fit. A point-start or refiner-injected
        start has no prior box to scatter across, so it always runs a single start (the
        refiner polishes the one best fit, it does not re-scatter). Only a standalone box
        fit (bounded priors, :meth:`_is_box_start`) reads ``n_starts`` (floored at 1)."""
        if not self._is_box_start():
            return 1
        return max(1, int(self.config.config.get('n_starts', 1)))

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
        if self.n_starts > 1:
            print2('Concurrent multi-start: %i independent starts (box center + '
                   'Latin-hypercube), keeping the global best' % self.n_starts)
        # One independent Generator per start (spawned deterministically from the run seed,
        # like the parallel samplers' per-chain rngs): a runner's rng-dependent behavior
        # (Simplex's degeneracy perturbation) then reproduces regardless of the order dask
        # returns concurrent starts' results, and spawn child 0 is identical whether we
        # spawn 1 or N -- so start 0 is byte-identical between a single- and a multi-start
        # run (the never-worse guarantee). Powell is deterministic and ignores its rng.
        rngs = self.spawn_chain_rngs(len(self.start_psets))
        self.runners = [self._make_runner(p, rngs[i]) for i, p in enumerate(self.start_psets)]
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
            logger.info('%s start %d/%d finished: %s', self._method_label,
                        idx + 1, len(self.runners), runner.stop_reason)
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
