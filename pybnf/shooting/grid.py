"""Knot placement: where a time course is cut, and which data point lands in which piece (#563).

Multiple shooting's first decision is where to break the horizon. :class:`SegmentGrid` owns
that decision and nothing else -- no simulator, no objective, no optimizer -- so the two
things that are easy to get subtly wrong are testable in isolation: which segment a data
point belongs to, and whether a coarser grid's knots are recognisably *the same knots* as a
finer grid's.

Three placements, and one of them is the default for a measured reason
----------------------------------------------------------------------
The issue asks for "a segment count or explicit knots; default to generic equal-time or
equal-observation segments", and all three are here:

``equal_time`` (the default)
    ``[start, horizon]`` cut into ``m`` equal spans. This is what the #563 prototype solved
    the motivating problem with, and it is the default because it places the knots using
    only the experiment's own time axis -- a fact rather than a guess. The obvious
    refinements that read a *trajectory* (knots at a burst, at a peak) are start-point
    dependent: they place the transcription's structure using dynamics the fit has not
    established, and on the motivating problem those dynamics are exactly what is in
    question (an oscillator whose period is wrong everywhere except a 3 % window).

``equal_observations``
    Cut so each segment owns the same number of measurements. It reads the data's own time
    axis and nothing else, so it is not start-point dependent either -- what it uses is the
    *sampling*, not the dynamics. It is the right placement when a time course is sampled
    unevenly, where equal spans leave some segments with nothing to fit and the auxiliary
    states of those segments are determined by continuity alone (ADR-0109 finding 5.2's
    under-determination, arrived at through the sampling rather than through ``m``). It
    needs at least two measurements per segment, so its own segment ceiling is half
    ``equal_time``'s.

explicit knots
    The caller supplies the times. Nothing here second-guesses them; they are validated for
    order and for lying strictly inside the horizon, and the run reports them.

Names, because the homotopy transfers by name
---------------------------------------------
:meth:`~pybnf.transcription.layout.AugmentedLayout.carry_over` moves an auxiliary block from
one stage to the next iff the next stage declares a block of the same **name and size** --
the layer never learns what a knot is, which is what keeps the rule generic. So the naming
here is load-bearing: a knot must get the same name at every segment count that has it, or
the ``4 -> 2 -> 1`` ladder reseeds instead of continuing and the coarsening (the mechanism,
ADR-0109 finding 5.2) buys nothing.

Naming a knot by its **exact fraction of the segment count** does that: knot ``i`` of ``m``
is ``Fraction(i, m)``. At ``m = 4`` the knots are ``1/4, 1/2, 3/4``; at ``m = 2``, ``1/2`` --
and ``Fraction(2, 4) == Fraction(1, 2)``, so the surviving knot carries its solved state
down the ladder while ``1/4`` and ``3/4`` are discarded, which is what coarsening *is*.
Exact rational arithmetic rather than a rounded float, so ``1/3`` and ``0.333333`` can never
be two names for one knot (or one name for two).

The fraction is an **ordinal**, not a claim about where the knot sits. Under ``equal_time``
the two coincide, which is why the naming reads so naturally there; under the other two
placements ``exp1@1/2`` is "the knot halfway along this grid's knot list", which is exactly
what carry-over needs to match on and is independent of what time it landed at. Every
placement therefore maps *the same fraction* to *the same knot*, at every rung -- which is
the whole property the ladder rests on.

Which segment owns a data point
-------------------------------
Half-open ``[start_j, start_{j+1})``, with the final segment owning the horizon endpoint --
the same convention ``tests/test_transcription.py::ShootingProblem`` pins offline. A point
lying exactly on a knot is therefore read at ``dt = 0`` from that knot's own auxiliary state,
so its prediction is the auxiliary variable itself: the one row where the data sees a
segment-start state directly rather than through an integration. The alternative (a knot
point belonging to the segment that ends there) reads it through the *previous* segment's
whole span, which is the same number only when continuity has already converged.
"""

from fractions import Fraction

import numpy as np

from ..printing import PybnfError

#: Separator between an experiment's label and a knot's fraction in a block name
#: (``'exp1@1/2'``). A block name may not contain the layout's ``'::'`` qualifier, and this
#: one does not.
KNOT = '@'

#: Knots at equal spans of the experiment's own time axis. The default, and what the #563
#: prototype solved the motivating problem with.
EQUAL_TIME = 'equal_time'

#: Knots placed so every segment owns the same number of measurements.
EQUAL_OBSERVATIONS = 'equal_observations'

#: Knots supplied by the caller (``ms_knots``), rather than derived from a rule.
EXPLICIT = 'explicit'

PLACEMENTS = (EQUAL_TIME, EQUAL_OBSERVATIONS, EXPLICIT)

#: Measurements a segment needs under :data:`EQUAL_OBSERVATIONS`. Two rather than one,
#: because a knot is placed *at* a measurement (that point then belongs to the later
#: segment, read at ``dt = 0``), so a one-observation-per-segment grid would put the last
#: knot on the horizon itself and leave a zero-length final segment.
OBSERVATIONS_PER_SEGMENT = 2


def max_segments(times, placement=EQUAL_TIME, knots=None):
    """The finest segment count ``times`` can support under ``placement``.

    Above it a rung is not the method: knots fall between observations everywhere and the
    auxiliary states of segments with no data are determined by continuity alone.
    :func:`~pybnf.shooting.driver.feasible_ladder` drops rungs above this and reports it.
    """
    if placement == EXPLICIT:
        return max(1, len(knots or ()) + 1)
    unique = len(np.unique(np.asarray(times, dtype=float).reshape(-1)))
    if placement == EQUAL_OBSERVATIONS:
        return max(1, unique // OBSERVATIONS_PER_SEGMENT)
    return max(1, unique)


class SegmentGrid:
    """The knots of one experiment's time course at one segment count.

    :param times: The experiment's measurement times, in the independent variable's units.
        Need not be sorted; duplicates are allowed (repeat measurements at one time).
    :param n_segments: ``m``, the number of segments. ``m = 1`` is the ordinary
        unsegmented problem: no knots, no auxiliary variables, no continuity constraints.
    :param label: The experiment's label, used to build block names. Two experiments in one
        fit get different labels so their knots never collide in one layout.
    :param start: The horizon's start. Defaults to ``min(times)``, but a time course whose
        first measurement is after the model's ``t = 0`` still integrates from ``0``, so a
        caller that knows the simulation's own start passes it.
    :param horizon: The horizon's end. Defaults to ``max(times)``.
    :param placement: One of :data:`PLACEMENTS`. See the module docstring.
    :param knots: Explicit knot times, which force ``placement = 'explicit'``. These are the
        **finest** rung's knots: a coarser rung of the same ladder keeps the sublist its own
        fractions select, so a ``4 -> 2 -> 1`` ladder over three explicit knots keeps the
        middle one at ``m = 2`` -- by the same fraction identity every other placement uses.
    """

    def __init__(self, times, n_segments, label='exp', start=None, horizon=None,
                 placement=EQUAL_TIME, knots=None):
        self.times = np.asarray(times, dtype=float).reshape(-1)
        if self.times.size == 0:
            raise PybnfError('A multiple-shooting segment grid needs at least one '
                             'measurement time.')
        self.n_segments = int(n_segments)
        if self.n_segments < 1:
            raise PybnfError('A multiple-shooting segment count must be at least 1; got %r.'
                             % n_segments)
        self.label = str(label)
        if KNOT in self.label:
            raise PybnfError("A multiple-shooting experiment label may not contain %r, the "
                             "knot-name separator; got %r." % (KNOT, self.label))
        self.placement = EXPLICIT if knots else str(placement)
        if self.placement not in PLACEMENTS:
            raise PybnfError('Unknown multiple-shooting knot placement %r; expected one of '
                             '%s.' % (placement, ', '.join(PLACEMENTS)))
        self.explicit_knots = tuple(float(t) for t in (knots or ()))
        self.start = float(np.min(self.times) if start is None else start)
        self.horizon = float(np.max(self.times) if horizon is None else horizon)
        if not self.horizon > self.start:
            raise PybnfError(
                'A multiple-shooting segment grid needs a positive horizon; experiment %r '
                'spans [%g, %g].' % (self.label, self.start, self.horizon))
        if np.any(self.times < self.start) or np.any(self.times > self.horizon):
            raise PybnfError(
                'Experiment %r has measurement times outside its own [%g, %g] horizon.'
                % (self.label, self.start, self.horizon))

        #: Each interior knot's exact fraction of the segment count -- the identity
        #: :meth:`~pybnf.transcription.layout.AugmentedLayout.carry_over` matches on. An
        #: ordinal, not a claim about where the knot sits (see the module docstring); under
        #: :data:`EQUAL_TIME` the two coincide.
        self.fractions = tuple(Fraction(i, self.n_segments)
                               for i in range(1, self.n_segments))
        #: Interior knot times (``m - 1`` of them; empty at ``m = 1``).
        self.knot_times = self._place()
        self._check_knots()
        #: Each segment's start time: the horizon's start, then every interior knot.
        self.starts = (self.start,) + self.knot_times
        #: Each segment's end time.
        self.ends = self.knot_times + (self.horizon,)
        #: Which segment each entry of :attr:`times` belongs to.
        self.segment_of = np.clip(
            np.searchsorted(np.asarray(self.starts, dtype=float), self.times, side='right') - 1,
            0, self.n_segments - 1)

    # -- placement --------------------------------------------------------------

    def _place(self):
        """Map each fraction to a knot time under this grid's placement rule."""
        if not self.fractions:
            return ()
        if self.placement == EQUAL_TIME:
            span = self.horizon - self.start
            return tuple(self.start + float(f) * span for f in self.fractions)
        if self.placement == EQUAL_OBSERVATIONS:
            unique = np.unique(self.times)
            n = len(unique)
            # Segment j owns unique[round(j n / m) : round((j+1) n / m)], so knot j sits at
            # unique[round(j n / m)] -- a *measurement* time, which by this module's
            # half-open convention belongs to the later segment and is read at dt = 0 from
            # that knot's own auxiliary state. Balanced by construction, and it is the same
            # index at every rung whose fraction coincides, which is what carries over.
            return tuple(float(unique[int(round(float(f) * n))]) for f in self.fractions)
        # Explicit: the fractions select from the finest rung's own list. At the finest
        # rung that is the whole list in order; at a rung whose fraction i/m equals a finest
        # fraction j/M it is knot j - 1, which is exactly the knot that carries over.
        finest = len(self.explicit_knots) + 1
        return tuple(self.explicit_knots[
            min(max(int(round(float(f) * finest)) - 1, 0), finest - 2)]
            for f in self.fractions)

    def _check_knots(self):
        """Knots must be strictly increasing and lie strictly inside the horizon.

        A repeated or out-of-order knot is a zero-length or negative-length segment, which
        no simulator can integrate and which would otherwise surface as an opaque failure
        several layers down. The message names the placement, because the fix differs: for a
        derived placement it is a segment count the data cannot support, and for an explicit
        one it is the supplied list.
        """
        bad = None
        previous = self.start
        for time in self.knot_times:
            if not previous < time:
                bad = time
                break
            previous = time
        if bad is None and self.knot_times and not self.knot_times[-1] < self.horizon:
            bad = self.knot_times[-1]
        if bad is None:
            return
        if self.placement == EXPLICIT:
            raise PybnfError(
                'The explicit multiple-shooting knots for experiment %r are not strictly '
                'increasing inside its own (%g, %g) horizon: %s.'
                % (self.label, self.start, self.horizon,
                   ', '.join('%g' % t for t in self.explicit_knots)),
                hint='Supply knot times in increasing order, each strictly between the '
                     'experiment\'s first and last measurement.')
        raise PybnfError(
            'Multiple shooting cannot cut experiment %r into %i segments by %s: the knots '
            'it places are not strictly increasing inside its (%g, %g) horizon.'
            % (self.label, self.n_segments, self.placement.replace('_', ' '),
               self.start, self.horizon),
            hint='Lower ms_segments, or use the default ms_knot_placement = equal_time, '
                 'which needs one measurement per segment rather than %i.'
                 % OBSERVATIONS_PER_SEGMENT)

    # -- identity ---------------------------------------------------------------

    @property
    def block_names(self):
        """One auxiliary-block name per interior knot, in segment order.

        ``'<label>@<fraction>'`` -- ``('exp1@1/4', 'exp1@1/2', 'exp1@3/4')`` at ``m = 4``.
        The ``m = 2`` grid of the same experiment declares ``('exp1@1/2',)``, which is a
        subset **by name**, so the ladder carries that knot's solved state over.
        """
        return tuple('%s%s%s' % (self.label, KNOT, f) for f in self.fractions)

    @property
    def n_knots(self):
        return len(self.fractions)

    # -- per-segment views ------------------------------------------------------

    def rows_in(self, segment):
        """Indices into :attr:`times` of the data points segment ``segment`` owns."""
        return np.flatnonzero(self.segment_of == int(segment))

    def sample_times(self, segment):
        """``(output times, data row indices)`` for one segment.

        The output grid is this segment's start knot, its own data times, and its end knot,
        de-duplicated and sorted. The start knot is present because integration begins
        there; the end knot because the continuity defect is read off the **same** run that
        produced the data rows, rather than from a second simulation of the same span.
        """
        rows = self.rows_in(segment)
        pts = np.unique(np.concatenate(
            ([self.starts[segment]], self.times[rows], [self.ends[segment]])))
        return pts, rows

    def row_positions(self, segment):
        """Where each of segment ``segment``'s data points lands in its output grid.

        Returned alongside the row indices as ``(output rows, data row indices)`` so a
        caller reads the simulated trajectory at exactly the measured times without
        re-searching the grid per point.
        """
        pts, rows = self.sample_times(segment)
        return np.searchsorted(pts, self.times[rows]), rows

    def seed_times(self):
        """One output grid covering the whole horizon: every knot and every data time.

        What a stage's auxiliary variables are seeded from -- one ordinary single-shoot
        simulation whose state is read off at each knot. Seeding this way makes the
        transcription **feasible at iteration zero**, so every discontinuity the run
        subsequently holds is the optimizer's own choice rather than an artifact of the
        start (the prototype's ``seed_aux``).
        """
        return np.unique(np.concatenate(
            ([self.start], self.times, np.asarray(self.knot_times, dtype=float),
             [self.horizon])))

    def describe(self):
        """One line for the run log."""
        if self.n_segments == 1:
            return ('%s: 1 segment over [%g, %g] (the ordinary unsegmented problem)'
                    % (self.label, self.start, self.horizon))
        counts = [len(self.rows_in(j)) for j in range(self.n_segments)]
        return ('%s: %i segments over [%g, %g] by %s, knots at %s, %s data point(s) per '
                'segment' % (self.label, self.n_segments, self.start, self.horizon,
                             self.placement.replace('_', ' '),
                             ', '.join('%g' % t for t in self.knot_times),
                             '/'.join(str(c) for c in counts)))

    def __repr__(self):
        return 'SegmentGrid(%r, m=%i, knots=%i, %s)' % (self.label, self.n_segments,
                                                        self.n_knots, self.placement)
