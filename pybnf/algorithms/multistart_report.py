"""The per-start table a multi-start fit writes when it finishes (#658).

Several fit types run more than one search, each from a different starting point, and
report the best result. That single number cannot be checked. Twenty starts that all
reach about the same objective value say the fit has very likely found the best answer
available, and that running more starts would not help. Twenty starts that all land
somewhere different say the reported answer is only the least bad of twenty poor ones,
so more starts are needed or the model and the parameter bounds need another look.
Before this, both cases printed one number and nothing else, so a user had no basis for
telling them apart.

Every multi-start fit now writes ``Results/multistart_summary.txt``: one row per start,
sorted by final objective value from best to worst, plus a short version on the screen.
Sorting the values that way and looking at the shape of the resulting curve is the usual
way to answer the question in the parameter fitting literature, where the picture is
called a waterfall plot.

Three families of fit type run more than one start, and each keeps the numbers somewhere
different: the concurrent local optimizers (``trf`` / ``lbfgs`` / ``gntr`` / ``powell`` /
``sim``, and the polish phase of ``profile_likelihood``) hold them on their per-start
runner objects, multiple shooting (``ms``) on its per-start homotopy results, and the
metaheuristics (``de`` / ``ade`` / ``ss`` / ``pso``) in the mixin that drives their
restarts. So the row type and the formatting live here, apart from all three. Each family
hands rows up through :meth:`pybnf.algorithms.base.Algorithm.multistart_records`, and
:meth:`pybnf.algorithms.base.Algorithm._emit_multistart_summary` writes whatever it is
given. This module holds no state and touches no files, so it is testable on its own.
"""

import math
from collections import namedtuple
from statistics import median

#: One start's outcome, the row type of the summary.
#:
#: ``start`` numbers the starts from 1, matching how the run log names them.
#: ``objective`` is the best objective value that start reached; ``inf`` where the start
#: produced no usable fit (its start point failed to simulate, say) and ``None`` where it
#: never ran at all. ``iterations`` counts the steps the start took, in whatever unit the
#: method counts, and ``evaluations`` counts the simulations charged to it; either is
#: ``None`` for a method that does not keep that count. ``stop_reason`` is the method's
#: own explanation, or ``None`` for a start the fit ended before.
StartRecord = namedtuple('StartRecord',
                         'start objective iterations evaluations stop_reason',
                         defaults=(None, None, None, None))

#: What the reason column says for a start that had not stopped on its own when the fit
#: ended. A wall-time budget is the usual cause.
UNFINISHED = 'did not finish before the fit ended'

#: The reason for a start that never began, which happens when a fit whose starts run one
#: after another ends before it reaches them.
NOT_STARTED = 'did not start before the fit ended'

#: How close to the best objective a start has to land to count as agreeing with it, as a
#: fraction of the best value (so 0.001 is a tenth of a percent). Relative rather than
#: absolute because objective values here span many orders of magnitude, with a floor of
#: 1.0 on the scale so a best value at or near zero compares sensibly.
PLATEAU_REL_TOL = 1e-3


def _has_value(record):
    """Whether this start reported a real, usable objective value."""
    obj = record.objective
    return obj is not None and not math.isnan(obj)


def sorted_records(records):
    """The records ordered by objective value, best first.

    A start with no value at all sorts last, and ties keep start order, so the table is
    the same on every run of the same fit.
    """
    return sorted(records,
                  key=lambda r: (0, float(r.objective), r.start) if _has_value(r)
                  else (1, 0.0, r.start))


def _finite_values(records):
    return [float(r.objective) for r in records
            if _has_value(r) and math.isfinite(r.objective)]


def plateau_count(records, rel_tol=PLATEAU_REL_TOL):
    """How many starts finished within ``rel_tol`` of the best objective value.

    This is the number that answers the question the table exists for. A count close to
    the number of starts means the starts agree on an answer; a count of one means the
    best result stands alone and the search has not shown that it found anything.
    Returns 0 when no start produced a finite value.
    """
    values = _finite_values(records)
    if not values:
        return 0
    best = min(values)
    tolerance = rel_tol * max(abs(best), 1.0)
    return sum(1 for v in values if v - best <= tolerance)


def _objective_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'none'
    return '%.10g' % value


def _count_text(value):
    return 'n/a' if value is None else '%d' % value


def _reason_text(reason):
    if not reason:
        return UNFINISHED
    # The table is tab separated, so a reason carrying a tab or a newline of its own would
    # break the row it sits in.
    return ' '.join(str(reason).split())


def _percent_text(rel_tol):
    return '%g%%' % (rel_tol * 100.0)


def summary_lines(records, job_type=None, rel_tol=PLATEAU_REL_TOL):
    """The whole text of ``Results/multistart_summary.txt``, as a list of lines.

    ``job_type`` names the fit type in the header when it is known, so the file can be
    read without the configuration file beside it.
    """
    rows = sorted_records(records)
    values = _finite_values(rows)
    lines = [
        '# One row per start of this fit, sorted by final objective value, best first.',
        '# This fit ran several searches from different starting points and reported the',
        '#   best of them. This table says how the others did, which is what tells you',
        '#   whether to believe the reported best.',
        '# If many starts sit at about the same low objective value, the fit has very',
        '#   likely found the best answer available and more starts would not help. If the',
        '#   values keep climbing with no group of similar ones at the top of the table,',
        '#   the reported best is only the least bad of these starts, so run more starts or',
        '#   take another look at the model and the parameter bounds. Plotting this column',
        '#   in this order is often called a waterfall plot.',
        '#',
        '# objective: the best objective value that start reached. "inf" means the start',
        '#   produced no usable fit, and "none" means it never ran.',
        '# iterations: steps that start took, in whatever unit the method counts.',
        '# evaluations: simulations charged to that start.',
        '#   Either count is "n/a" for a method that does not keep it.',
        '#',
    ]
    if job_type:
        lines.append('job_type\t%s' % job_type)
    lines.append('starts\t%d' % len(rows))
    if values:
        lines.append('best_objective\t%.10g' % min(values))
        lines.append('worst_objective\t%.10g' % max(values))
        lines.append('median_objective\t%.10g' % median(values))
    lines.append('# starts_near_best counts the starts that came within %s of the best '
                 'objective value.' % _percent_text(rel_tol))
    lines.append('starts_near_best\t%d' % plateau_count(rows, rel_tol))
    lines.append('#')
    lines.append('# rank\tstart\tobjective\titerations\tevaluations\treason')
    for rank, r in enumerate(rows, start=1):
        lines.append('%d\t%d\t%s\t%s\t%s\t%s'
                     % (rank, r.start, _objective_text(r.objective),
                        _count_text(r.iterations), _count_text(r.evaluations),
                        _reason_text(r.stop_reason)))
    return lines


def console_lines(records, path, rel_tol=PLATEAU_REL_TOL):
    """The short version printed at the end of the run, as a list of lines."""
    rows = sorted_records(records)
    values = _finite_values(rows)
    if not values:
        return ['Multi-start summary: none of the %d starts produced a usable fit. '
                'Per-start table: %s' % (len(rows), path)]
    lines = ['Multi-start summary: %d starts. Best objective %.6g, median %.6g, '
             'worst %.6g.' % (len(rows), min(values), median(values), max(values))]
    unusable = len(rows) - len(values)
    if unusable:
        lines.append('  %d of the %d produced no usable fit.' % (unusable, len(rows)))
    lines.append('  %d of the %d came within %s of the best. The more starts that reach '
                 'the same low value, the more likely the fit has found the best answer '
                 'available.'
                 % (plateau_count(rows, rel_tol), len(rows), _percent_text(rel_tol)))
    lines.append('  Per-start table, best first: %s' % path)
    return lines
