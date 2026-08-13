"""The transcription homotopy: solve a ladder of transcriptions, not one (#563).

A constrained transcription has a knob -- how many segments, how many collocation nodes,
how much of the state is made auxiliary -- and the whole method's difficulty runs along it.
The obvious MVP fixes the knob and solves once. The #563 prototype measured that that is
the wrong shape, twice over, and this module is the consequence.

**The ladder is the mechanism, not a refinement.** On the motivating problem the stage
trace *is* the result (issue #563, finding 5.2)::

    m=8: -132.32   m=4: -166.52   m=2: -221.91   m=1: -248.07

Every segmented stage scores worse than a flat line (``-164.68``); the *coarsening* is what
converts them, and the run that produced this trace is the first solve of a problem fifteen
independent global searches did not solve. Fixing the segment count and solving once
reaches ``m=4`` and stops.

**And it starts in the middle, not at the far end.** The original formulation proposed
starting with many short segments -- the easiest landscape -- and coarsening toward one.
Measured, ``m=8`` routinely certifies *worse than its own start*: with one observed state
of three and ~14 points per segment, the segmented problem is under-determined, so the data
term is satisfiable without correct dynamics and the auxiliary states absorb the rest. Over
eight paired starts, ``4-2-1`` had the best tail at moderate cost and ``8-4-2-1`` did not.
:func:`coarsening_ladder` therefore defaults to starting at **4**.

Two rules the driver follows, and the asymmetry between them
------------------------------------------------------------
*Continue from the last point; report the best certified one.* A stage seeds the next from
where it **finished** -- that is what continuation means, and the trace above is what it
buys. But the answer the run reports is the best **certified** iterate across every stage
(:class:`~pybnf.transcription.outer.CertifiedBest`), because a stage's final point is not
reliably its best one: on one prototype start the final stage held ``-147.0`` while an
earlier iterate certified at ``-196.3`` (finding 5.3). The two rules disagree on purpose.

Multipliers are **not** carried between stages. Coarsening changes the constraint set --
different knots, different count, different meaning -- so a multiplier estimated for a
constraint that no longer exists is not an estimate of anything. Each stage restarts from
``lambda = 0`` at ``rho_0``. The auxiliary *variables* do carry over, by name
(:meth:`~pybnf.transcription.layout.AugmentedLayout.carry_over`), which is where the
continuation information actually lives.
"""

import numpy as np

from .augmented import TranscriptionProblem
from .errors import TranscriptionError
from .outer import AugmentedLagrangian, CertifiedBest


def coarsening_ladder(start=4, factor=2, stop=1):
    """The default homotopy ladder: ``(4, 2, 1)``.

    :param start: The finest transcription's knob value. Defaults to **4** rather than the
        largest affordable number -- see the module docstring on why the fine end is the
        wrong end.
    :param factor: The coarsening factor between stages.
    :param stop: The coarsest stage, reached exactly. ``1`` is the ordinary unsegmented
        problem, so a ladder always ends by solving the fit the user actually asked about.

    Values are integers, strictly decreasing, ending at ``stop``.
    """
    start, factor, stop = int(start), int(factor), int(stop)
    if stop < 1:
        raise TranscriptionError('A coarsening ladder must end at 1 or above; got %r.' % stop)
    if start < stop:
        raise TranscriptionError('A coarsening ladder starts at or above its stop (%r < %r).'
                                 % (start, stop))
    if factor < 2:
        raise TranscriptionError('A coarsening ladder needs a factor of at least 2; got %r.'
                                 % factor)
    rungs = []
    value = start
    while value > stop:
        rungs.append(value)
        value = max(value // factor, stop)
    rungs.append(stop)
    return tuple(rungs)


class StageResult:
    """One rung of the ladder."""

    __slots__ = ('name', 'outer', 'final_point', 'layout')

    def __init__(self, name, outer, final_point, layout):
        self.name = str(name)
        self.outer = outer
        self.final_point = np.array(final_point, dtype=float, copy=True)
        self.layout = layout

    @property
    def best_score(self):
        """The best objective certified *within this stage* -- the number the trace prints."""
        return self.outer.best_score

    def __repr__(self):
        return 'StageResult(%s, best=%.6g, %s)' % (self.name, self.best_score,
                                                   self.outer.stop_reason)


class HomotopyResult:
    """What a whole ladder produced."""

    def __init__(self, stages, best, stop_reason, certified):
        self.stages = tuple(stages)
        self.best = best
        self.stop_reason = str(stop_reason)
        self.certified = bool(certified)

    @property
    def best_score(self):
        return self.best.score if self.best is not None else np.inf

    @property
    def reported(self):
        """The reported free parameters of the best certified iterate -- the fit result.
        ``None`` if nothing certified."""
        return None if self.best is None else self.best.reported.copy()

    @property
    def n_evaluations(self):
        return sum(stage.outer.n_evaluations for stage in self.stages)

    def trace(self):
        """The ladder in one line -- ``'m=4: -166.52   m=2: -221.91   m=1: -248.07'``.

        The single most informative artifact a homotopy produces: it shows whether the
        coarsening is converting the segmented stages, which is the mechanism the method
        rests on, and it shows it before the run finishes.
        """
        return '   '.join('%s: %.6g' % (s.name, s.best_score) for s in self.stages)

    def summary(self):
        best = ('%.6g' % self.best_score) if self.best is not None else 'none'
        won = '' if self.best is None else ' (from %s #%i)' % (self.best.stage,
                                                               self.best.iteration)
        return ('homotopy over %i stage(s), %s: best certified objective %s%s, %i evaluations%s'
                % (len(self.stages), self.stop_reason, best, won, self.n_evaluations,
                   '' if self.certified else ' [UNCERTIFIED]'))

    def __repr__(self):
        return 'HomotopyResult(stages=%i, best=%.6g)' % (len(self.stages), self.best_score)


def run_homotopy(stages, inner_solver, reported_start, schedule=None, max_outer=25,
                 stop_check=None, on_iterate=None, on_stage=None):
    """Run an augmented-Lagrangian solve on each stage in turn, warm-starting down the ladder.

    :param stages: The transcriptions, finest first. Each item is either a
        :class:`~pybnf.transcription.augmented.TranscriptionProblem` or a callable
        ``(reported) -> TranscriptionProblem`` -- the callable form exists because a stage's
        auxiliary variables usually have to be *seeded from the incoming parameters* (for
        multiple shooting, by reading a nominal trajectory at the current ``theta``), which
        is not knowable when the ladder is written.
    :param inner_solver: The inner-solver callable, passed through to
        :class:`~pybnf.transcription.outer.AugmentedLagrangian`.
    :param reported_start: The fit's start point, in sampling space, over the reported free
        parameters.
    :param schedule: The shared :class:`~pybnf.transcription.outer.PenaltySchedule`.
    :param max_outer: Outer-iteration cap **per stage**.
    :param stop_check: Zero-argument callable; ``True`` ends the ladder between or within
        stages (the wall-clock-budget seam).
    :param on_iterate: Per-iterate callback, passed through.
    :param on_stage: Optional callback given each :class:`StageResult` as it completes.

    Returns a :class:`HomotopyResult` whose :attr:`~HomotopyResult.best` is the best
    certified iterate over the **whole** ladder.
    """
    stages = list(stages)
    if not stages:
        raise TranscriptionError('A homotopy needs at least one stage.')
    reported = np.asarray(reported_start, dtype=float).reshape(-1)

    shared_best = CertifiedBest()
    results = []
    previous_layout = None
    point = None
    stop_reason = 'completed'
    certified = True

    for index, stage in enumerate(stages):
        if stop_check is not None and stop_check():
            stop_reason = 'stopped'
            break

        problem = stage if isinstance(stage, TranscriptionProblem) else stage(reported)
        if not isinstance(problem, TranscriptionProblem):
            raise TranscriptionError(
                'Homotopy stage %i produced %r, not a TranscriptionProblem.'
                % (index, type(problem).__name__))
        layout = problem.layout

        if previous_layout is None:
            start = layout.initial_point(reported)
        else:
            start = previous_layout.carry_over(point, layout)

        loop = AugmentedLagrangian(problem, inner_solver, schedule=schedule,
                                   max_outer=max_outer, shared_best=shared_best,
                                   stop_check=stop_check, on_iterate=on_iterate)
        outer = loop.run(start)
        certified = certified and outer.certified

        result = StageResult(problem.name, outer, outer.final_point, layout)
        results.append(result)
        if on_stage is not None:
            on_stage(result)

        point = outer.final_point
        previous_layout = layout
        reported = layout.reported_of(point)

        if outer.stop_reason in ('stopped', 'inner_failed'):
            # The ladder cannot continue from a point the inner solve never reached; the
            # best certified iterate so far is still the answer.
            stop_reason = outer.stop_reason
            break

    return HomotopyResult(results, shared_best.record, stop_reason, certified)
