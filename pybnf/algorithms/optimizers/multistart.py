"""Fit-type-agnostic multi-start for the non-gradient optimizers (``n_starts``, #498).

A single optimizer run descends into whichever basin its start lands in, so on a
multimodal objective it returns only a local minimum -- the gap #386 already closed
for the gradient methods (which run ``population_size`` independent starts
concurrently and keep the global best). :class:`MultiStartOptimizer` lifts that
"``n_starts`` independent runs, keep the best" idea into a shared layer over the
metaheuristics (``de`` / ``ss`` / ``pso`` / ``ade``), where the LHS start-generation
primitive (``random_latin_hypercube_psets``) already lives on the base
:class:`~pybnf.algorithms.base.Algorithm` -- only the *orchestration* was
gradient-specific.

The strategy is **sequential restart** rather than #386's concurrent starts: a
population method already parallelizes every generation across the cluster, so it
saturates the workers without extra starts. Each start runs the underlying search to
its own termination (convergence or ``max_iterations``); on that stop, if starts
remain, the mixin reinitializes the method from scratch (a fresh random / Latin
hypercube population) and continues, up to ``n_starts`` starts. The global best is
kept for free -- every evaluated ``PSet`` across every start lands in the trajectory
(``add_to_trajectory`` runs on the master before ``got_result``), so
``trajectory.best_fit()`` is the best over all starts. ``n_starts == 1`` (the default)
is a single run -- byte-identical to the pre-multi-start behavior.

How it plugs in (the leaf contract)
-----------------------------------
A leaf opts in by (1) mixing this class in *before* its Algorithm base
(``class DifferentialEvolution(MultiStartOptimizer, DifferentialEvolutionBase)``) so
the mixin's ``start_run`` / ``got_result`` win the MRO, and (2) exposing its search
under two renamed hooks:

* :meth:`_search_start_run` ``() -> list[PSet]`` -- (re)initialize the search from
  scratch (a fresh population, all counters reset) and return the initial PSets.
  **Idempotent**: the mixin calls it once per start, and each call must begin a fully
  independent search *without* touching the trajectory (the cross-start global best).
* :meth:`_search_got_result` ``(res) -> list[PSet] | 'STOP'`` -- the method's own
  ``got_result`` state machine, unchanged.

Name-spacing + draining (the two subtleties)
--------------------------------------------
Every start regenerates the same internal PSet names (``gen0ind0`` ...), which would
collide across starts and clobber each other's sim folders. The mixin is the sole
translator at the run-loop boundary: it **tags** every outgoing PSet with the current
start's prefix (``s1_`` ... ; start 0 is untagged, so a single-start run's names are
byte-identical) and **strips** that prefix off every returning result before handing it
to the inner method -- so the leaf only ever sees the clean names it generated, and the
run loop only ever sees unique ones. Because the leaf sees clean names, this works
whether the leaf routes completed results by PSet identity (``de`` / ``ss`` / ``pso``)
or by parsing the name (``ade`` / ``pso``).

The mixin also tracks its own **in-flight** set (tagged names emitted but not yet
returned). When an inner search returns ``'STOP'`` it may still have jobs pending -- a
one-in-one-out async method (``pso`` / ``ade``) keeps a full population in flight,
whereas a generation-synchronized method (``de`` / ``ss``) has none. So the mixin
**drains**: it stops feeding the (now-finished) inner search, discards its stragglers
(already recorded in the trajectory), and seeds the next start only once the in-flight
set empties. For a synchronized method the set is already empty, so the next start
begins immediately.

All added state is plain ``int`` / ``set`` / ``list`` (the start index, the in-flight
names, the draining flag, the per-start tallies), so the optimizer pickles for
backup/resume exactly as before (ADR-0007).

Reporting the starts (#658)
---------------------------
Keeping the global best is not the whole job: a user also needs to know whether the
starts agreed, since a run whose starts all reached the same value and a run whose starts
all landed somewhere different otherwise print the same single number. The mixin is the
only place that knows where one start ends and the next begins, so it tallies each start's
best objective value and evaluation count as the results come back and hands them to
``Results/multistart_summary.txt`` through :meth:`~MultiStartOptimizer.multistart_records`
(see :mod:`pybnf.algorithms.multistart_report`).
"""

from ..multistart_report import NOT_STARTED, StartRecord
from ...config_schema import PyBNFConfigModel
from ...printing import print2

import logging

logger = logging.getLogger('pybnf.algorithms')

#: Why a start of a metaheuristic ended, for the end-of-run per-start table (#658). These
#: methods return a bare stop signal without saying which of their two stopping conditions
#: fired, so the row states both rather than guessing at one.
_SEARCH_ENDED = 'the search ended on its own (converged or reached max_iterations)'


class MultiStartConfig(PyBNFConfigModel):
    """The shared ``n_starts`` config field (#498), mixed into the schema of every
    optimizer that supports multi-start (``de`` / ``ss`` / ``pso`` / ``ade``) so the
    key is defined once and appears only in those methods' effective configs -- not,
    say, in a sampler's or in ``cmaes`` (which has its own ``cmaes_restarts`` /
    IPOP-BIPOP restart, ADR-0070). ``n_starts`` is the number of independent starts to
    run, keeping the global best; ``1`` (the default) is a single run, byte-identical
    to the pre-multi-start behavior. Each start runs the underlying search to its own
    termination (convergence or ``max_iterations``)."""

    n_starts: int = 1


class MultiStartOptimizer:
    """Mixin adding ``n_starts`` sequential-restart multi-start to a metaheuristic
    optimizer. See the module docstring for the leaf contract and the name-spacing /
    draining design. Mix in *before* the Algorithm base so these ``start_run`` /
    ``got_result`` overrides win the MRO and delegate to the leaf's ``_search_*`` hooks.
    """

    def __init__(self, *args, **kwargs):
        # Cooperative: the leaf's __init__ calls super().__init__(config), which lands
        # here first (mixin precedes the Algorithm base in the MRO); delegate onward to
        # build self.config, then install the multi-start bookkeeping. Doing it here --
        # not only in start_run -- means the attributes exist for the object's whole
        # life, so a caller that drives got_result() directly without the run loop's
        # start_run() (white-box unit tests do) still finds them. start_run()/reset()
        # re-install the same per-run values, so a real run is byte-identical.
        super().__init__(*args, **kwargs)
        self._start_index = 0
        self._inflight = set()
        self._draining = False
        self._start_stats = []
        self.n_starts = self._resolve_n_starts()

    def _resolve_n_starts(self):
        """The number of independent starts, from ``n_starts`` (floored at 1). Every
        start of a metaheuristic is a fresh random / Latin-hypercube population, so --
        unlike the local optimizers' box-center-then-LHS scheme -- there is no
        distinguished start 0 and no box-start gate: the method re-randomizes regardless
        of whether its priors are bounded."""
        return max(1, int(self.config.config.get('n_starts', 1)))

    # --- run-loop overrides ------------------------------------------------- #
    def start_run(self):
        self.n_starts = self._resolve_n_starts()
        self._start_index = 0
        self._inflight = set()
        self._draining = False
        self._start_stats = []
        if self.n_starts > 1:
            print2('Multi-start: up to %i independent starts, keeping the global best'
                   % self.n_starts)
        return self._emit(self._search_start_run())

    def got_result(self, res):
        # res.pset.name is still the *tagged* name here (start_run/got_result tagged it
        # on emission, and add_to_trajectory already recorded it); untrack it before any
        # stripping.
        self._inflight.discard(res.pset.name)
        self._record_for_summary(res)
        if self._draining:
            # A straggler from the just-finished start: its score is already in the
            # trajectory (recorded before got_result), and the inner search has been
            # abandoned, so discard it. Seed the next start once the last one drains.
            if not self._inflight:
                return self._begin_next_start()
            return []
        self._strip_prefix(res)
        response = self._search_got_result(res)
        if response == 'STOP':
            self._current_start_stats()['stop_reason'] = _SEARCH_ENDED
            if self._start_index + 1 < self.n_starts:
                logger.info('Multi-start: start %d/%d finished; %d job(s) to drain '
                            'before the next start', self._start_index + 1, self.n_starts,
                            len(self._inflight))
                self._draining = True
                if not self._inflight:            # synchronized method: nothing pending
                    return self._begin_next_start()
                return []
            return 'STOP'
        return self._emit(response)

    def _begin_next_start(self):
        """Seed a fresh independent search after the previous start drained."""
        self._start_index += 1
        self._draining = False
        print2('Multi-start: beginning start %d of %d' % (self._start_index + 1, self.n_starts))
        return self._emit(self._search_start_run())

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._start_index = 0
        self._inflight = set()
        self._draining = False
        self._start_stats = []

    # --- the end-of-run per-start table (#658) ------------------------------- #
    def _current_start_stats(self):
        """The running record of how the start now in progress is doing, created on
        first use. A plain list of dicts, so it rides the backup pickle like the rest of
        the mixin's state."""
        while len(self._start_stats) <= self._start_index:
            self._start_stats.append({'objective': None, 'evaluations': 0,
                                      'stop_reason': None})
        return self._start_stats[self._start_index]

    def _record_for_summary(self, res):
        """Charge one completed evaluation to the start it belongs to, and keep the best
        objective value that start has reached (#658).

        A metaheuristic keeps no single "final objective" of its own -- its answer is the
        best member it ever produced -- so the mixin takes the best objective value seen
        while that start was running. Every result passes through here, including the
        stragglers of a start that has already finished, which belong to that start and
        not to the next one.
        """
        stats = self._current_start_stats()
        stats['evaluations'] += 1
        try:
            # Defensive: this runs on every completed result, in the run loop, purely to
            # fill in a report. A result that somehow carries no usable objective costs
            # the fit a row of a table, not the fit.
            score = float(getattr(res, 'score', None))
        except (TypeError, ValueError):
            return
        if stats['objective'] is None or score < stats['objective']:
            stats['objective'] = score

    def multistart_records(self):
        """One row per start for ``Results/multistart_summary.txt`` (#658).

        These starts run one after another, so a fit that stops early (a wall-time budget,
        say) may never reach some of them. Those are listed too, as starts that never ran,
        because a table that quietly showed six rows for a twenty-start fit would read as
        a twenty-start fit that agreed with itself.

        The iteration count is left out. Each of these methods counts its own progress
        differently -- generations per island, unproductive iterations, and so on -- and
        there is no shared number to put in the column.
        """
        rows = []
        for i in range(max(self.n_starts, len(self._start_stats))):
            if i < len(self._start_stats):
                stats = self._start_stats[i]
                rows.append(StartRecord(start=i + 1, objective=stats['objective'],
                                        iterations=None, evaluations=stats['evaluations'],
                                        stop_reason=stats['stop_reason']))
            else:
                rows.append(StartRecord(start=i + 1, stop_reason=NOT_STARTED))
        return rows

    # --- name-space boundary ------------------------------------------------ #
    def _prefix(self):
        """The current start's PSet-name prefix -- empty for start 0 (so a single-start
        run's names are byte-identical), ``s<k>_`` for start ``k >= 1`` (so restarts get
        unique sim folders and a unique in-flight key)."""
        return '' if self._start_index == 0 else 's%d_' % self._start_index

    def _emit(self, psets):
        """Tag each outgoing PSet with the current start's prefix and record it as
        in-flight. Returns the tagged list for the run loop to submit."""
        prefix = self._prefix()
        for p in psets:
            if prefix:
                p.name = prefix + p.name
            self._inflight.add(p.name)
        return psets

    def _strip_prefix(self, res):
        """Remove the current start's prefix from ``res.pset.name`` so the inner search
        sees exactly the clean name it generated. In-flight results always belong to the
        current start (the previous start fully drained before this one began), so the
        current prefix is the right one to strip."""
        prefix = self._prefix()
        if prefix and res.pset.name.startswith(prefix):
            res.pset.name = res.pset.name[len(prefix):]
