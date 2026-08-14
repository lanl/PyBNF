"""Driving the ladder: one fit, transcribed at several segment counts in turn (#563).

The whole of multiple shooting's outer control flow is
:func:`~pybnf.transcription.homotopy.run_homotopy`; this module supplies the three things it
needs that are specific to segments -- the ladder of segment counts, a stage *factory* per
rung, and the defaults ADR-0109's findings fixed.

Why the stages are callables
----------------------------
``run_homotopy`` accepts a stage as a :class:`~pybnf.transcription.augmented.TranscriptionProblem`
or as a ``(reported) -> TranscriptionProblem`` callable, and multiple shooting needs the
second form: a stage's auxiliary variables are seeded by reading a *nominal trajectory at the
incoming parameters* at each knot, and the incoming parameters are whatever the previous rung
finished at. Seeding that way makes each stage feasible at its own iteration zero, so every
discontinuity a run holds is the optimizer's choice rather than an artifact of how the stage
was built.

The ladder starts in the middle
-------------------------------
``(4, 2, 1)``, from :func:`~pybnf.transcription.homotopy.coarsening_ladder`, and the middle
is the measured part. Starting at many short segments -- the easiest landscape, and what the
original formulation proposed -- is the wrong end on the motivating problem: with one
observed state of three and ~14 points per segment, ``m = 8`` is under-determined, the data
term is satisfiable without correct dynamics, and the stage routinely certifies *worse than
its own start*. Over eight paired starts, ``4-2-1`` had the best tail at moderate cost and
``8-4-2-1`` did not. The ladder always ends at ``m = 1``, which is the ordinary unsegmented
fit -- so a multiple-shooting run finishes by solving the problem the user actually asked
about, and its last rung is the one whose objective needs no certification because it *is*
the certificate.
"""

from ..transcription import PenaltySchedule, coarsening_ladder, run_homotopy
from .problem import AUX_DECADES, seed_stage
from .solver import GaussNewtonSolver


def feasible_ladder(specs, ladder=None):
    """The requested ladder, clamped to what the data can support.

    A segment count above what an experiment's data supports places knots between
    observations everywhere and leaves segments with nothing to fit -- a transcription whose
    auxiliary states are determined by continuity alone in *every* segment, which is not the
    method, it is an ODE solver with extra variables. Rungs above that are dropped and the
    caller is told which (a silent cap would read as "we ran the ladder you asked for").

    The ceiling is the experiment's own, because it depends on the knot placement: equal
    time needs one measurement per segment, equal observations needs two, and an explicit
    knot list *is* the finest rung (:func:`~pybnf.shooting.grid.max_segments`).

    Returns ``(rungs, dropped)``.
    """
    rungs = tuple(int(m) for m in (coarsening_ladder() if ladder is None else ladder))
    limit = min((spec.max_segments for spec in specs), default=1)
    kept = tuple(m for m in rungs if m <= max(1, limit))
    dropped = tuple(m for m in rungs if m not in kept)
    if 1 not in kept:
        # The ladder must end at the unsegmented problem: that rung is what makes the run's
        # answer an ordinary fit result rather than a segmented score.
        kept = kept + (1,)
    return kept, dropped


def coarsening_stages(specs, rungs, objective, variables, pset_from_u,
                      aux_decades=AUX_DECADES, pool=None):
    """One stage factory per rung, in the order ``run_homotopy`` will run them."""
    def factory(n_segments):
        def build(reported):
            return seed_stage(specs, n_segments, objective, variables, pset_from_u, reported,
                              aux_decades=aux_decades, pool=pool)
        return build
    return [factory(m) for m in rungs]


def run_multiple_shooting(specs, objective, variables, pset_from_u, reported_start,
                          ladder=None, schedule=None, inner_solver=None, max_outer=25,
                          aux_decades=AUX_DECADES, stop_check=None, on_iterate=None,
                          on_stage=None, pool=None):
    """Solve one fit by multiple shooting, down the coarsening ladder.

    :param specs: The :class:`~pybnf.shooting.problem.ShootingExperiment`\\ s -- one per
        scored ``(model, condition)`` pair, built once for the whole fit.
    :param objective: The fit's own objective function.
    :param variables: Its reported free parameters, in ``Configuration.variables`` order.
    :param pset_from_u: The algorithm's sampling-space to :class:`~pybnf.pset.PSet` bridge.
    :param reported_start: The fit's start point in sampling space.
    :param ladder: Segment counts, finest first. Defaults to ``(4, 2, 1)``.
    :param schedule: The :class:`~pybnf.transcription.outer.PenaltySchedule`. Its defaults
        carry ADR-0109 finding 5.1 -- the penalty starts **tight** (``rho_0 = 10``,
        ``gamma = 5``), because on the motivating problem a loose start was both worse and
        twice as expensive: the inner solve on a nearly-unconstrained subproblem never
        converges and burns its whole budget every outer iteration.
    :param inner_solver: Defaults to a :class:`~pybnf.shooting.solver.GaussNewtonSolver`.
    :param stop_check: The wall-clock-budget seam, passed to the outer loop *and* to the
        default inner solver, so a deadline lands inside a long inner solve rather than
        after it.
    :param pool: The :class:`~pybnf.shooting.parallel.SegmentPool` every rung's segment
        passes run through. Defaults to the serial one.

    Returns the :class:`~pybnf.transcription.homotopy.HomotopyResult`, whose ``best`` is the
    best **certified** iterate over the whole ladder -- not its last, which on one prototype
    start held ``-147.0`` while an earlier iterate certified at ``-196.3``.
    """
    rungs, _dropped = feasible_ladder(specs, ladder)
    if inner_solver is None:
        inner_solver = GaussNewtonSolver(stop_check=stop_check)
    stages = coarsening_stages(specs, rungs, objective, variables, pset_from_u,
                               aux_decades=aux_decades, pool=pool)
    return run_homotopy(stages, inner_solver, reported_start,
                        schedule=schedule or PenaltySchedule(), max_outer=max_outer,
                        stop_check=stop_check, on_iterate=on_iterate, on_stage=on_stage)
