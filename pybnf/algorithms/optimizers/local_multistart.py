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

The runner-list orchestration this needs is shared with the gradient path (the >=2-member
event, ADR-0009) and lives on the common
:class:`~pybnf.algorithms.optimizers.concurrent_multistart.ConcurrentMultiStartOptimizer`
base (#500) -- seeding, name routing, ``STOP``-on-last-start coordination, and the
box-gated box-center + Latin-hypercube start scheme. This class is the *local* leg of that
base: it adds only what the derivative-free path does differently from the gradient path --
the per-start rng spawning (:meth:`_build_runners`) and the concurrent-multi-start
announcement (:meth:`_announce_starts`).

The leaf contract
-----------------
A leaf (``powell`` / ``sim``) mixes this in *before* :class:`StartPointOptimizer`
(``class PowellAlgorithm(LocalMultiStartOptimizer, StartPointOptimizer)``) so the base's
``start_run`` / ``got_result`` overrides win the MRO, sets :attr:`START_POINT_KEY` and
:attr:`_method_label`, and supplies its per-start step math + PSet plumbing through the
hooks the base documents:

* :meth:`~ConcurrentMultiStartOptimizer._make_runner` ``(start_pset, rng) -> runner`` --
  build one start's headless, picklable step machine seeded at ``start_pset`` (with its own
  ``rng`` for any stochastic step, e.g. Simplex's degeneracy perturbation).
* :meth:`~ConcurrentMultiStartOptimizer._seed` ``(idx, runner) -> list[PSet]`` -- the
  runner's initial job(s), named + routed to start ``idx`` (via
  :meth:`~ConcurrentMultiStartOptimizer._route`).
* :meth:`~ConcurrentMultiStartOptimizer._advance` ``(idx, runner, res) -> list[PSet] |
  DONE`` -- feed the completed ``res`` to the runner and return its next job(s), or
  :data:`DONE` when that start has terminated.
* :meth:`~ConcurrentMultiStartOptimizer._report` ``(runner)`` -- the per-iteration
  progress line for one start.
* :meth:`~ConcurrentMultiStartOptimizer._start_banner` ``() -> str`` -- the one-line
  "running ..." message.

The global best is kept for free: every evaluated PSet across every start lands in the
trajectory (``add_to_trajectory`` runs on the master before ``got_result``), so
``trajectory.best_fit()`` is the best over all starts. ``n_starts == 1`` (the default, or
any point-start / refiner fit) is a single run whose PSet names -- and therefore search --
are byte-identical to the pre-multi-start behavior.
"""

from .concurrent_multistart import DONE, ConcurrentMultiStartOptimizer
from ...printing import print2

__all__ = ['DONE', 'LocalMultiStartOptimizer']


class LocalMultiStartOptimizer(ConcurrentMultiStartOptimizer):
    """The derivative-free leg of the concurrent multi-start base (#498/#500).

    Inherits the entire ``start_run`` / ``got_result`` orchestration and the box-gated
    box-center + Latin-hypercube start scheme from
    :class:`~pybnf.algorithms.optimizers.concurrent_multistart.ConcurrentMultiStartOptimizer`,
    and adds only the two things the derivative-free path does differently from the gradient
    path: one independent :class:`numpy.random.Generator` per start (so a stochastic step
    reproduces regardless of dask's result order), and the concurrent-multi-start
    announcement. A leaf (``powell`` / ``sim``) supplies its per-start step math + PSet
    plumbing through the base's hooks."""

    def _build_runners(self):
        """One runner per start, each with its own independent Generator (spawned
        deterministically from the run seed, like the parallel samplers' per-chain rngs): a
        runner's rng-dependent behavior (Simplex's degeneracy perturbation) then reproduces
        regardless of the order dask returns concurrent starts' results, and spawn child 0
        is identical whether we spawn 1 or N -- so start 0 is byte-identical between a
        single- and a multi-start run (the never-worse guarantee). Powell is deterministic
        and ignores its rng. Keeping the rng here (not in the base) keeps the base
        rng-agnostic, shared with the deterministic gradient path (#500)."""
        rngs = self.spawn_chain_rngs(len(self.start_psets))
        return [self._make_runner(p, rngs[i]) for i, p in enumerate(self.start_psets)]

    def _announce_starts(self):
        """Announce the concurrent multi-start (after the banner) when running more than one
        start; a single-start fit stays silent, byte-identical to the pre-multi-start run."""
        if self.n_starts > 1:
            # Name start 0 honestly: it is the declared start point when there is one, and
            # the box center otherwise (#583). A banner that always said "box center" became
            # a lie the moment a start point could be pinned.
            first = ('declared start point'
                     if (getattr(self.config, 'start_point', None) or {}) else 'box center')
            scatter = ('Latin-hypercube'
                       if self.config.config.get('initialization') == 'lh' else 'random')
            print2('Concurrent multi-start: %i independent starts (%s + %s), '
                   'keeping the global best' % (self.n_starts, first, scatter))
