"""Writing a design down: what to measure, and what it is expected to buy (#574).

The report has two halves, because a recommendation nobody can check is not worth much.

The first half is the recommendation itself: the measurements to make, in the order they were
chosen, with a count when the same point is chosen more than once. The second half is the reason,
stated in the units the user has already seen from a profile-likelihood run -- each parameter's
confidence interval as it stands now, and as it would be once the recommended measurements are in
hand. Those predicted intervals come from the same information matrix the design was chosen with,
read through the quadratic approximation to the profile: the interval is
``theta* +- sqrt(threshold * variance)`` in the parameter's own fitted scale. For a linear model
that is exact, and for anything else it is the local approximation, which is also all the design
itself ever claimed to be.
"""

import numpy as np

from .criteria import interval_half_widths, lower_is_better
from .greedy import improvement


def _interval(variable, centre_u, half_width):
    """One parameter's predicted confidence interval in its own units, or ``None`` when the
    information leaves it undetermined and the interval is open."""
    if not np.isfinite(half_width):
        return None
    return (float(variable.from_sampling_space(centre_u - half_width)),
            float(variable.from_sampling_space(centre_u + half_width)))


def predicted_intervals(result, variables, u_star, threshold):
    """Every parameter's interval before and after the design, plus how much it shrinks.

    Each row is a dict with the parameter's name, its value at the best fit, the interval the
    existing data supports, the interval the design would support, and ``width_ratio`` -- the
    designed half-width over the current one, so 0.5 means the interval halves. The ratio is
    ``None`` when the current interval is open, which is the strongest result there is: the design
    replaces no answer with an answer."""
    current = interval_half_widths(result.baseline, threshold)
    designed = interval_half_widths(result.information, threshold)
    rows = []
    for index, variable in enumerate(variables):
        centre = float(u_star[index])
        ratio = None
        if np.isfinite(current[index]) and current[index] > 0.0:
            ratio = float(designed[index] / current[index])
        rows.append({
            'name': variable.name,
            'best': float(variable.from_sampling_space(centre)),
            'current': _interval(variable, centre, current[index]),
            'designed': _interval(variable, centre, designed[index]),
            'width_ratio': ratio,
        })
    return rows


def _format_interval(interval):
    return 'open' if interval is None else '[%.6g, %.6g]' % interval


def write_design_report(path, result, variables, u_star, threshold, confidence):
    """Write the design report to ``path`` as a tab-delimited file with commented headers, the
    same shape as the profile-likelihood summary beside it."""
    rows = predicted_intervals(result, variables, u_star, threshold)
    factor = improvement(result)
    with open(path, 'w') as handle:
        handle.write('# criterion=%s\t%s\n' % (result.criterion, result.criterion_name))
        handle.write('# targets=%s\n' % (', '.join(result.target_names) or 'all parameters'))
        handle.write('# confidence=%g\tdelta_chi2_threshold=%g\tdof=1\n'
                     % (confidence, threshold))
        handle.write('# criterion_before=%.10g\tcriterion_after=%.10g\t%s_is_better\n'
                     % (result.baseline_value, result.value,
                        'lower' if lower_is_better(result.criterion) else 'higher'))
        if factor is not None:
            handle.write('# improvement_factor=%.6g\n' % factor)
        if result.truncated:
            handle.write('# fewer measurements than asked for: no remaining candidate adds '
                         'anything the criterion can use\n')
        handle.write('#\n# recommended measurements, in the order they were chosen\n')
        handle.write('# rank\tmodel\texperiment\tobservable\tindependent_variable\tvalue\t'
                     'replicates\tcriterion_after\n')
        criterion_at = {rank: value for rank, value in enumerate(result.trace, start=1)}
        for measurement, replicates, rank in result.grouped():
            handle.write('%d\t%s\t%s\t%s\t%s\t%.10g\t%d\t%.10g\n' % (
                rank, measurement.model, measurement.experiment or '-', measurement.observable,
                measurement.independent_variable, measurement.time, replicates,
                criterion_at[rank]))
        handle.write('#\n# predicted confidence intervals, before and after\n')
        handle.write('# parameter\tbest\tcurrent_low\tcurrent_high\tdesigned_low\t'
                     'designed_high\twidth_ratio\n')
        for row in rows:
            current = row['current'] or (None, None)
            designed = row['designed'] or (None, None)
            handle.write('%s\t%.10g\t%s\t%s\t%s\t%s\t%s\n' % (
                row['name'], row['best'],
                'None' if current[0] is None else '%.10g' % current[0],
                'None' if current[1] is None else '%.10g' % current[1],
                'None' if designed[0] is None else '%.10g' % designed[0],
                'None' if designed[1] is None else '%.10g' % designed[1],
                'None' if row['width_ratio'] is None else '%.6g' % row['width_ratio']))


def format_design_summary(result, variables, u_star, threshold):
    """The same design as a short block of lines for the terminal: what to measure, and what it
    does to the intervals of the parameters the design was aimed at."""
    lines = ['Recommended next measurements (%s, aimed at %s):'
             % (result.criterion_name, ', '.join(result.target_names) or 'all parameters')]
    for measurement, replicates, rank in result.grouped():
        times = '' if replicates == 1 else ' (measure %d times)' % replicates
        lines.append('  %d. %s%s' % (rank, measurement, times))
    if result.truncated:
        lines.append('  (stopped early: no remaining candidate adds anything the criterion '
                     'can use)')
    rows = {row['name']: row for row in predicted_intervals(result, variables, u_star, threshold)}
    reported = result.target_names or [v.name for v in variables]
    lines.append('Predicted confidence intervals:')
    for name in reported:
        row = rows[name]
        shrink = ('' if row['width_ratio'] is None
                  else '  (%.3g times as wide)' % row['width_ratio'])
        lines.append('  %-16s now %s -> %s%s'
                     % (name, _format_interval(row['current']),
                        _format_interval(row['designed']), shrink))
    return lines
