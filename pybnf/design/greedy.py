"""Choosing the best few measurements out of the whole candidate space (#574).

Picking the best five of two hundred candidate measurements is a subset-selection problem, not a
continuous one: there are far too many subsets to try them all. The standard answer, and the one
here, is to take the measurements one at a time, each time adding whichever remaining candidate
improves the criterion most. It is not guaranteed to find the very best subset, but it is simple,
it always terminates, and for the D-criterion it is the classical exchange algorithm's forward
half.

The same candidate may be chosen more than once. That is not a bug to be suppressed: choosing a
point twice means measuring it twice, and it is exactly the right recommendation when the
precision of one measurement, rather than the shape of the trajectory, is what limits you.

One special case has to be handled explicitly. When the information matrix says nothing at all
about a direction the criterion cares about, the criterion sits at its worst possible value for
every candidate and cannot tell them apart -- an infinite variance is an infinite variance whatever
you add to it. The selection loop notices and asks a different question until that stops being
true: of the directions nothing yet sees, which candidate sees the most
(:func:`~pybnf.design.criteria.null_space_gain`)? Then it goes back to the requested criterion for
the rest of the picks. A design aimed at one parameter is not blocked by some *other* combination
of parameters being invisible, so this only fires when the target itself is the problem.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..printing import PybnfError
from .criteria import (
    CRITERIA,
    CRITERION_NAMES,
    criterion_score,
    criterion_value,
    lower_is_better,
    null_space_gain,
    unidentified_parameters,
)


@dataclass
class DesignResult:
    """A finished design: what to measure, and what it is expected to buy.

    ``measurements`` are in the order they were chosen, repeats included, so reading them in order
    shows how much each successive measurement is still worth. ``baseline`` is the information the
    existing data already carries and ``information`` is what it would become; ``trace`` is the
    criterion after each pick, so a design that stops paying off is visible rather than implied."""

    criterion: str
    targets: list
    target_names: list
    measurements: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    baseline: Any = None
    information: Any = None
    escaped_singular: int = 0
    truncated: bool = False

    @property
    def baseline_value(self):
        """The criterion before any of the recommended measurements are made."""
        return criterion_value(self.baseline, self.criterion, self.targets)

    @property
    def value(self):
        """The criterion once every recommended measurement has been made."""
        return criterion_value(self.information, self.criterion, self.targets)

    @property
    def criterion_name(self):
        return CRITERION_NAMES[self.criterion]

    def grouped(self):
        """The recommendation as ``(measurement, replicates, first_rank)`` in the order the
        measurements were first chosen -- the reading a person wants, where choosing one point
        three times is one row saying "measure it three times"."""
        order, counts = [], {}
        for rank, measurement in enumerate(self.measurements, start=1):
            if measurement not in counts:
                counts[measurement] = [0, rank]
                order.append(measurement)
            counts[measurement][0] += 1
        return [(m, counts[m][0], counts[m][1]) for m in order]


def resolve_targets(variables, names, criterion):
    """The parameter indices the A-criterion sums variances over, validated against the fit.

    ``names`` empty or absent means every free parameter. Naming exactly one makes this the
    classical c-criterion: minimize the variance of that one parameter, which is what a
    profile-likelihood verdict about a single parameter asks for.

    Targets are refused for the D and E criteria rather than quietly ignored. Both are properties
    of the whole information matrix -- a volume and a worst direction -- so restricting them to a
    subset of parameters means something different from what a reader would assume, and a request
    that cannot be honoured as written should say so."""
    if criterion not in CRITERIA:
        raise PybnfError(
            "design_criterion must be one of %s, not %r."
            % (', '.join(CRITERIA), criterion),
            hint='; '.join('%s = %s' % (code, CRITERION_NAMES[code]) for code in CRITERIA))
    names = list(names or [])
    if not names:
        return list(range(len(variables)))
    if criterion != 'a':
        raise PybnfError(
            "design_target names the parameters to measure the design against, which only the "
            "A-criterion (design_criterion = a) can use, but design_criterion = %s." % criterion,
            hint="Set design_criterion = a to target %s, or drop design_target."
                 % ', '.join(names))
    by_name = {v.name: i for i, v in enumerate(variables)}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise PybnfError(
            "design_target names %s, which %s not a free parameter of this fit."
            % (', '.join(unknown), 'is' if len(unknown) == 1 else 'are'),
            hint="List only free-parameter ids (the fit declares: %s)."
                 % ', '.join(v.name for v in variables))
    return [by_name[n] for n in names]


def require_identifiable(baseline, candidates, param_names, targets):
    """Refuse a design whose targets no experiment in the candidate space could ever pin down.

    Adding every candidate at once is the most a design over this space can know. A target still
    left with an infinite variance there is not short of data: no measurement of these observables,
    at any simulated time, tells the model apart along that direction. That is structural
    non-identifiability, and the fix is a different model or a different observable, not a
    different design."""
    n_param = len(param_names)
    everything = np.asarray(baseline, dtype=float) + candidates.total(n_param)
    lost = unidentified_parameters(everything, param_names)
    targeted = [param_names[i] for i in targets]
    blocking = [name for name in lost if name in targeted]
    if blocking:
        raise PybnfError(
            "No design over these observables and times can determine %s. Measuring every "
            "candidate point at once still leaves %s undetermined, so the model and the data it "
            "can produce do not distinguish %s from the other parameters at all."
            % (', '.join(blocking), 'it' if len(blocking) == 1 else 'them',
               'it' if len(blocking) == 1 else 'them'),
            hint="This is structural non-identifiability, not a shortage of data. Measure a "
                 "different observable, fix one of the parameters, or reparameterize the model. "
                 "A profile-likelihood run (job_type = profile_likelihood) reports which "
                 "parameters are structurally non-identifiable.")
    return lost


def _criterion_is_blind(information, criterion, targets):
    """Whether the criterion can still tell two candidates apart at this information.

    It cannot when the quantity it measures is already at its worst possible value for every
    candidate: an infinite variance for a target the data cannot see at all, a log determinant of
    ``-inf``, a smallest eigenvalue of zero. Note that a singular information matrix is not by
    itself blinding -- a design aimed at one parameter is perfectly well defined while some
    *other* combination of parameters remains unseen, and it should go on optimizing what it was
    asked to."""
    score = criterion_score(information, criterion, targets)
    return not np.isfinite(score) or (criterion == 'e' and score <= 0.0)


def select_design(baseline, candidates, n_points, criterion, targets, param_names):
    """Choose ``n_points`` measurements, one at a time, each the best next addition.

    ``baseline`` is the information the existing data carries, ``candidates`` the
    :class:`~pybnf.design.candidates.CandidateSet` to choose from. Returns a
    :class:`DesignResult`. Ties break toward the earliest candidate, which is the earliest time of
    the first observable of the first experiment, so the same inputs always give the same design.
    """
    if n_points <= 0:
        raise PybnfError("design_points must be at least 1, not %d." % n_points,
                         hint="design_points is how many measurements to recommend.")
    if not len(candidates):
        raise PybnfError(
            "There are no candidate measurements to choose from.",
            hint="A candidate is an observable this fit already measures, at a time its "
                 "simulation passes through. Check that the fit scores at least one observable.")

    information = np.array(baseline, dtype=float)
    result = DesignResult(criterion=criterion, targets=list(targets),
                          target_names=[param_names[i] for i in targets],
                          baseline=np.array(baseline, dtype=float))
    blocks = candidates.blocks
    for _pick in range(n_points):
        blind = _criterion_is_blind(information, criterion, targets)
        if blind:
            # The criterion cannot choose yet, so chase the directions nothing sees instead.
            scores = [null_space_gain(information, block, targets) for block in blocks]
            result.escaped_singular += 1
        else:
            scores = [criterion_score(information + block, criterion, targets)
                      for block in blocks]
        best = int(np.argmax(scores))
        if blind and scores[best] <= 0.0:
            # No candidate sees any of the missing directions. require_identifiable refuses that
            # before a design is ever selected, so this is unreachable from a job; stop and say so
            # rather than pick an arbitrary point that buys nothing.
            result.truncated = True
            break
        information = information + blocks[best]
        result.measurements.append(candidates.measurements[best])
        result.trace.append(criterion_value(information, criterion, targets))
    result.information = information
    return result


def improvement(result):
    """How much better the criterion got, as a plain ratio, or ``None`` when it cannot be stated.

    For the A-criterion this is the factor the summed variance shrank by, so 4.0 means the
    variance is a quarter of what it was and the confidence interval is half as wide. A baseline
    that was infinite (a parameter the existing data cannot determine at all) has no ratio, which
    is itself the headline: the design goes from no answer to an answer."""
    before, after = result.baseline_value, result.value
    if not np.isfinite(before) or not np.isfinite(after):
        return None
    if lower_is_better(result.criterion):
        return None if after <= 0.0 else before / after
    if result.criterion == 'd':
        return float(np.exp(after - before))     # a ratio of determinants, not of logs
    return None if before <= 0.0 else after / before
