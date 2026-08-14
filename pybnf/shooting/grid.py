"""Knot placement: where a time course is cut, and which data point lands in which piece (#563).

Multiple shooting's first decision is where to break the horizon. :class:`SegmentGrid` owns
that decision and nothing else -- no simulator, no objective, no optimizer -- so the two
things that are easy to get subtly wrong are testable in isolation: which segment a data
point belongs to, and whether a coarser grid's knots are recognisably *the same knots* as a
finer grid's.

Equal time, because the alternative needs information a fit does not have
--------------------------------------------------------------------------
The #563 prototype cut ``[0, T]`` into ``m`` equal spans and solved the motivating problem
that way, so that is what ships. The obvious refinements -- knots at the data's own
quantiles, or at the features of a nominal trajectory (a burst, a peak) -- are *start-point
dependent*: they place the transcription's structure using a trajectory the fit has not
established yet, and on the motivating problem the nominal trajectory is exactly what is in
question (an oscillator whose period is wrong everywhere except a 3 % window). Equal spans
place the knots using only the experiment's own time axis, which is a fact rather than a
guess. ``knot_times`` is a plain attribute, so a future consumer can hand in its own.

Names, because the homotopy transfers by name
---------------------------------------------
:meth:`~pybnf.transcription.layout.AugmentedLayout.carry_over` moves an auxiliary block from
one stage to the next iff the next stage declares a block of the same **name and size** --
the layer never learns what a knot is, which is what keeps the rule generic. So the naming
here is load-bearing: a knot must get the same name at every segment count that has it, or
the ``4 -> 2 -> 1`` ladder reseeds instead of continuing and the coarsening (the mechanism,
ADR-0109 finding 5.2) buys nothing.

Naming a knot by its **exact fraction of the horizon** does that. At ``m = 4`` the knots are
``1/4, 1/2, 3/4``; at ``m = 2``, ``1/2`` -- and ``Fraction(2, 4) == Fraction(1, 2)``, so the
surviving knot carries its solved state down the ladder while ``1/4`` and ``3/4`` are
discarded, which is what coarsening *is*. Exact rational arithmetic rather than a rounded
float, so ``1/3`` and ``0.333333`` can never be two names for one knot (or one name for two).

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
    """

    def __init__(self, times, n_segments, label='exp', start=None, horizon=None):
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

        span = self.horizon - self.start
        #: Each interior knot's exact fraction of the horizon -- the identity
        #: :meth:`~pybnf.transcription.layout.AugmentedLayout.carry_over` matches on.
        self.fractions = tuple(Fraction(i, self.n_segments)
                               for i in range(1, self.n_segments))
        #: Interior knot times (``m - 1`` of them; empty at ``m = 1``).
        self.knot_times = tuple(self.start + float(f) * span for f in self.fractions)
        #: Each segment's start time: the horizon's start, then every interior knot.
        self.starts = (self.start,) + self.knot_times
        #: Each segment's end time.
        self.ends = self.knot_times + (self.horizon,)
        #: Which segment each entry of :attr:`times` belongs to.
        self.segment_of = np.clip(
            np.searchsorted(np.asarray(self.starts, dtype=float), self.times, side='right') - 1,
            0, self.n_segments - 1)

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
        return ('%s: %i segments over [%g, %g], knots at %s, %s data point(s) per segment'
                % (self.label, self.n_segments, self.start, self.horizon,
                   ', '.join('%g' % t for t in self.knot_times),
                   '/'.join(str(c) for c in counts)))

    def __repr__(self):
        return 'SegmentGrid(%r, m=%i, knots=%i)' % (self.label, self.n_segments, self.n_knots)
