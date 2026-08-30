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
#: counts the runs that produced no usable value.
Candidate = namedtuple('Candidate', 'name pset search_objective scores failures',
                       defaults=((), 0))


def mean_objective(candidate):
    """The average objective value over this candidate's replicate runs.

    ``inf`` when every replicate failed, so a candidate that cannot be simulated sorts
    last instead of winning.
    """
    if not candidate.scores:
        return math.inf
    return fmean(candidate.scores)


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
    """The candidates ordered by average objective value, best first.

    Ties keep the order they were given in, which is the order the search ranked them,
    so the table is the same on every run of the same fit.
    """
    return sorted(range(len(candidates)), key=lambda i: (mean_objective(candidates[i]), i))


def winner(candidates):
    """The candidate with the best average objective value, or ``None`` when not one of
    them produced a usable value."""
    order = ranked(candidates)
    if not order:
        return None
    best = candidates[order[0]]
    if not best.scores:
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
        '# mean_objective: the average over the replicate runs. This is what the ranking',
        '#   uses and it is the honest value to quote.',
        '# standard_error: the uncertainty in that average. Two rows whose averages differ',
        '#   by less than their standard errors have not really been told apart, so run',
        '#   more replicates if you need to separate them.',
        '# std_deviation: how much one run of this parameter set differs from the next.',
        '# runs: replicate runs that produced a usable value.',
        '# failed: replicate runs that produced none.',
        '#',
    ]
    lines.append('candidates\t%d' % len(candidates))
    lines.append('replicates_requested\t%d' % replicates)
    if best is not None:
        lines.append('winner\t%s' % best.name)
        lines.append('winner_mean_objective\t%.10g' % mean_objective(best))
        lines.append('winner_standard_error\t%s' % _number_text(standard_error(best)))
        lines.append('winner_search_objective\t%s' % _number_text(best.search_objective))
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
            lines.append('search_rank\t%d' % (order[0] + 1))
    else:
        lines.append('# No candidate produced a usable objective value when it was run again,')
        lines.append('#   so the best fit is the one the search picked.')
        lines.append('winner\tnone')
    lines.append('#')
    lines.append('# rank\tname\tmean_objective\tstandard_error\tstd_deviation\truns\tfailed'
                 '\tsearch_objective')
    for rank, i in enumerate(order, start=1):
        c = candidates[i]
        lines.append('%d\t%s\t%s\t%s\t%s\t%d\t%d\t%s'
                     % (rank, c.name, _number_text(mean_objective(c)),
                        _number_text(standard_error(c)),
                        _number_text(standard_deviation(c)),
                        len(c.scores), c.failures,
                        _number_text(c.search_objective)))
    return lines


def console_lines(candidates, replicates, path):
    """The short version printed at the end of the run, as a list of lines."""
    best = winner(candidates)
    if best is None:
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
    if best.search_objective is not None and math.isfinite(best.search_objective):
        lines.append('  The search reported %.6g for it, which was %.3g too good because '
                     'that came from one simulation.'
                     % (best.search_objective, mean_objective(best) - best.search_objective))
    if order and order[0] != 0:
        lines.append('  This is not the parameter set the search would have reported. It '
                     'was number %d in the search ranking, and the one the search liked '
                     'best does worse when it is run again.' % (order[0] + 1))
    lines.append('  Details: %s' % path)
    return lines
