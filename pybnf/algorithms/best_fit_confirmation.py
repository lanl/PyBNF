"""The end-of-fit stage that confirms the best fit of a stochastic model (#659).

When a model is stochastic, running it twice with the same parameter values gives two
different answers, so the objective value PyBNF computes is a noisy measurement rather
than a fixed number. A fit picks its answer by taking the best objective value it ever
saw, and every one of those values came from a single simulation. A long fit scores tens
of thousands of parameter sets, so the winner of that comparison is very often the
parameter set that happened to get a lucky simulation rather than the parameter set that
is genuinely best.

Two things come out wrong, and nothing in the output used to say so. The reported
objective value is the best of many noisy draws, so it is optimistic by a wide margin.
And the reported parameter values are not the best ones found, because a slightly worse
parameter set with a lucky draw beats a better one with an average draw.

The fix is one extra stage at the end of the fit. Take the top few parameter sets rather
than only the single best, run each of them again several times, rank them by their
average objective value, and report that winner. On a run where the good candidates
differ from each other by about as much as one simulation's noise, this removes about
ninety percent of the optimism.

A candidate has to survive its replicate runs to win that ranking. A run that fails, or
that scores something that is not a finite number, produces no value to average, and
averaging only the runs that did produce one would score a parameter set on whichever of
its simulations happened to work -- the same lucky draw in a different disguise. So a
candidate is confirmed, and eligible to win, only when more than half of its runs produced
a value; the rest are ranked below every confirmed one and shown in the table with their
failure counts (#720).

This module holds the row type, the arithmetic, and the report text. It keeps no state
and touches no files, so it is testable on its own. The orchestration -- choosing the
candidates, submitting the simulations, and recording the winner -- lives in
:meth:`pybnf.algorithms.base.Algorithm._confirm_best_fit`.
"""

import math
from collections import namedtuple
from statistics import fmean, stdev

#: One candidate parameter set and how it did when it was run again.
#:
#: ``name`` is the simulation name the search gave it, which is also the folder its
#: original simulation was written to. ``pset`` is the parameter set itself.
#: ``search_objective`` is the single noisy value the search recorded for it, kept so the
#: report can show how far off that value was. ``scores`` is the list of objective values
#: from the replicate runs, with failed and non-finite runs left out, and ``failures``
#: counts the runs that produced no usable value. The two counts together decide whether
#: this candidate is allowed to win; see :func:`confirmed`.
Candidate = namedtuple('Candidate', 'name pset search_objective scores failures',
                       defaults=((), 0))


def mean_objective(candidate):
    """The average objective value over this candidate's replicate runs that produced one.

    ``inf`` when every replicate failed, so a candidate that cannot be simulated at all
    sorts last. This is the number the report quotes and the winner is pinned with, and it
    is an average over the runs that worked -- which is the honest number to quote only for
    a candidate that :func:`confirmed` accepts.
    """
    if not candidate.scores:
        return math.inf
    return fmean(candidate.scores)


def attempts(candidate):
    """How many replicate runs of this candidate came back with a verdict, usable or not.

    Normally ``best_fit_replicates``. It is lower when a replicate was cancelled rather
    than run, which is the cluster's doing and not the parameter set's, so the rule below
    is written against this count rather than against the number of runs asked for.
    """
    return len(candidate.scores) + candidate.failures


def confirmed(candidate):
    """Whether this candidate was measured well enough to be allowed to win (#720).

    A replicate that fails, or that returns a non-finite objective value, is not a missing
    measurement. It is an outcome, and a bad one: the parameter set could not be simulated,
    or it was simulated and scored worse than any number. Averaging only the replicates that
    did produce a value estimates this candidate's objective value *given that the run
    worked*, which is worth quoting only when working is the normal case.

    So a candidate has to have produced a usable value in more than half of its runs. That
    is exactly the condition for the middle of all of its runs to be a real number: a
    failure is known to be worse than every value, so a candidate that fails half of its
    runs or more has no finite median, and its average over the survivors is describing the
    minority that happened to work. Letting that average win would put back the
    pick-the-luckiest-draw rule this whole stage exists to take out, through the failure
    channel rather than the noise channel -- a parameter set that fails nine runs of ten and
    survives one is scored on a single simulation again, with the unlucky draws deleted
    instead of averaged in.

    A candidate this rejects still appears in the table with its average and its failure
    count. It is barred from being the run's answer, not hidden.
    """
    n = attempts(candidate)
    return n > 0 and len(candidate.scores) * 2 > n


def standard_deviation(candidate):
    """How much this candidate's objective value varies from one run to the next, or
    ``None`` when fewer than two runs produced a value."""
    if len(candidate.scores) < 2:
        return None
    return stdev(candidate.scores)


def standard_error(candidate):
    """The uncertainty in this candidate's average objective value, or ``None`` when
    fewer than two runs produced a value.

    This is the number that says whether the ranking below it means anything. Two
    candidates whose averages differ by less than their standard errors have not been
    told apart by this stage.
    """
    sd = standard_deviation(candidate)
    if sd is None:
        return None
    return sd / math.sqrt(len(candidate.scores))


def ranked(candidates):
    """The candidates ordered best first: the ones this stage confirmed, by average
    objective value, and after all of them the ones it did not, also by average.

    The confirmed ones come first whatever the averages say, because an unconfirmed
    candidate's average is over a different and smaller thing than a confirmed one's and the
    two are not comparable (:func:`confirmed`). Ties keep the order they were given in,
    which is the order the search ranked them, so the table is the same on every run of the
    same fit.
    """
    return sorted(range(len(candidates)),
                  key=lambda i: (not confirmed(candidates[i]),
                                 mean_objective(candidates[i]), i))


def winner(candidates):
    """The best of the candidates this stage confirmed, or ``None`` when it confirmed none
    of them.

    ``None`` means this stage has no answer of its own and the one the search picked stands.
    It covers a candidate that could not be run again at all, and one that ran but failed
    half of its runs or more, which is not a measurement this stage is willing to hand the
    run's answer to (#720).
    """
    order = ranked(candidates)
    if not order:
        return None
    best = candidates[order[0]]
    if not confirmed(best):
        return None
    return best


def _number_text(value):
    if value is None:
        return 'n/a'
    if isinstance(value, float) and math.isnan(value):
        return 'n/a'
    return '%.10g' % value


def summary_lines(candidates, replicates):
    """The whole text of ``Results/best_fit_confirmation.txt``, as a list of lines."""
    order = ranked(candidates)
    best = winner(candidates)
    lines = [
        '# This fit used at least one stochastic model, so running the same parameter',
        '#   values twice gives two different objective values. The search picked its',
        '#   answer by taking the best objective value it ever saw, and every one of those',
        '#   came from a single simulation, so the winner of that comparison was often just',
        '#   the parameter set that got a lucky simulation.',
        '# To settle it, the top parameter sets of the search were each run again several',
        '#   times and ranked by their average objective value. The winner of that ranking',
        '#   is what the run reports as its best fit.',
        '#',
        '# search_objective: the single noisy value the search recorded for this parameter',
        '#   set. It is the value that made this parameter set look good enough to be worth',
        '#   checking, and it is usually better than the average below.',
        '# mean_objective: the average over the replicate runs that produced a value. It',
        '#   is what the ranking uses among the confirmed rows below, and it is the honest',
        '#   value to quote for one of them.',
        '# standard_error: the uncertainty in that average. Two rows whose averages differ',
        '#   by less than their standard errors have not really been told apart, so run',
        '#   more replicates if you need to separate them.',
        '# std_deviation: how much one run of this parameter set differs from the next.',
        '# runs: replicate runs that produced a usable value.',
        '# failed: replicate runs that produced none, because the simulation failed or',
        '#   because it scored something that is not a finite number.',
        '# confirmed: whether this parameter set produced a usable value in more than half',
        '#   of its runs. A failed run is a bad outcome and not a missing measurement, so a',
        '#   parameter set that fails half of them or more has not been measured -- the',
        '#   average beside it is over the minority of runs that worked, which is the lucky',
        '#   draw this stage exists to undo. Such a row is ranked below every confirmed one',
        '#   whatever its average says, and cannot be the winner.',
        '#',
    ]
    lines.append('candidates\t%d' % len(candidates))
    lines.append('replicates_requested\t%d' % replicates)
    if best is not None:
        lines.append('winner\t%s' % best.name)
        lines.append('winner_mean_objective\t%.10g' % mean_objective(best))
        lines.append('winner_standard_error\t%s' % _number_text(standard_error(best)))
        lines.append('winner_search_objective\t%s' % _number_text(best.search_objective))
        lines.append('winner_runs\t%d' % len(best.scores))
        lines.append('winner_failed\t%d' % best.failures)
        if best.failures:
            lines.append('# The winner failed %d of its %d runs, so the average above is over'
                         % (best.failures, attempts(best)))
            lines.append('#   the %d that worked. More than half of them did, which is why it'
                         % len(best.scores))
            lines.append('#   was allowed to win, but a parameter set that cannot be simulated')
            lines.append('#   every time is worth looking into before the answer is used.')
        if best.search_objective is not None and math.isfinite(best.search_objective):
            lines.append('# optimism is how much better the search made the winner look than')
            lines.append('#   running it again says it is.')
            lines.append('optimism\t%.10g' % (mean_objective(best) - best.search_objective))
        # A row other than the first says the search picked the wrong parameter set, which
        # is the half of this that changes the answer rather than only the number.
        if order and order[0] != 0:
            lines.append('# search_rank is where the winner sat in the search\'s own ranking.')
            lines.append('#   Anything but 1 means the search would have reported a different')
            lines.append('#   parameter set.')
            if confirmed(candidates[0]):
                lines.append('#   Here the one the search liked best does worse when it is run')
                lines.append('#   again.')
            else:
                lines.append('#   Here the one the search liked best failed half of its runs or')
                lines.append('#   more, so it was not confirmed.')
            lines.append('search_rank\t%d' % (order[0] + 1))
    else:
        if any(c.scores for c in candidates):
            lines.append('# Every candidate failed half of its runs or more, so none of them')
            lines.append('#   was confirmed and the best fit is the one the search picked. The')
            lines.append('#   averages below are each over the minority of runs that worked, so')
            lines.append('#   every one of them is optimistic and none of them is this run\'s')
            lines.append('#   answer. Something is making these simulations fail, and finding')
            lines.append('#   out what is the thing to do next.')
        else:
            lines.append('# No candidate produced a usable objective value when it was run again,')
            lines.append('#   so the best fit is the one the search picked.')
        lines.append('winner\tnone')
    unconfirmed = [candidates[i].name for i in order if not confirmed(candidates[i])]
    if unconfirmed:
        lines.append('# unconfirmed names the candidates that failed half of their runs or')
        lines.append('#   more. None of them can be the winner, whatever its average says.')
        lines.append('unconfirmed\t%s' % ', '.join(unconfirmed))
    lines.append('#')
    lines.append('# rank\tname\tmean_objective\tstandard_error\tstd_deviation\truns\tfailed'
                 '\tconfirmed\tsearch_objective')
    for rank, i in enumerate(order, start=1):
        c = candidates[i]
        lines.append('%d\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s'
                     % (rank, c.name, _number_text(mean_objective(c)),
                        _number_text(standard_error(c)),
                        _number_text(standard_deviation(c)),
                        len(c.scores), c.failures,
                        'yes' if confirmed(c) else 'no',
                        _number_text(c.search_objective)))
    return lines


def console_lines(candidates, replicates, path):
    """The short version printed at the end of the run, as a list of lines."""
    best = winner(candidates)
    if best is None:
        if any(c.scores for c in candidates):
            return ['Best-fit confirmation: every one of the %d candidate parameter sets '
                    'failed half of its runs or more, so none of them was confirmed and the '
                    'best fit is the one the search picked. Their averages are each over the '
                    'few runs that worked, so none of them is the answer; find out why the '
                    'simulations are failing. Details: %s' % (len(candidates), path)]
        return ['Best-fit confirmation: none of the %d candidate parameter sets could be '
                'run again, so the best fit is the one the search picked. Details: %s'
                % (len(candidates), path)]
    order = ranked(candidates)
    lines = ['Best-fit confirmation: ran the top %d parameter sets %d more times each and '
             'ranked them by average objective value.' % (len(candidates), replicates)]
    sem = standard_error(best)
    if sem is None:
        lines.append('  Best average objective %.6g (%s).'
                     % (mean_objective(best), best.name))
    else:
        lines.append('  Best average objective %.6g, give or take %.3g (%s).'
                     % (mean_objective(best), sem, best.name))
    if best.failures:
        lines.append('  %d of its %d runs produced no usable value, so that average is over '
                     'the %d that did.'
                     % (best.failures, attempts(best), len(best.scores)))
    if best.search_objective is not None and math.isfinite(best.search_objective):
        lines.append('  The search reported %.6g for it, which was %.3g too good because '
                     'that came from one simulation.'
                     % (best.search_objective, mean_objective(best) - best.search_objective))
    unconfirmed = [c for c in candidates if not confirmed(c)]
    if unconfirmed:
        lines.append('  This stage confirmed %d of the %d candidates; the rest failed half '
                     'of their runs or more, so they could not win whatever their averages '
                     'say.' % (len(candidates) - len(unconfirmed), len(candidates)))
    if order and order[0] != 0:
        if confirmed(candidates[0]):
            lines.append('  This is not the parameter set the search would have reported. It '
                         'was number %d in the search ranking, and the one the search liked '
                         'best does worse when it is run again.' % (order[0] + 1))
        else:
            lines.append('  This is not the parameter set the search would have reported. It '
                         'was number %d in the search ranking, and the one the search liked '
                         'best was not confirmed.' % (order[0] + 1))
    lines.append('  Details: %s' % path)
    return lines
