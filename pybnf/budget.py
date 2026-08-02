"""The fit's total wall-clock budget (``wall_time_fit``, ADR-0093).

PyBNF's other time limits are per unit of work -- ``wall_time_sim`` bounds one
simulation, ``wall_time_gen`` bounds one network generation -- and neither bounds
a *fit*. A fit's only native budget is ``max_iterations`` x ``population_size``,
which is not convertible to wall time without knowing the per-iteration cost in
advance (it varies by orders of magnitude across problems, and within a problem
as the search moves through stiff regions). ``wall_time_fit`` is the missing
total: the number of seconds the whole fit may run, after which it stops cleanly
and **finalizes** -- writing the same end-of-fit artifacts a converged run writes,
against the best point found so far (issue #529).

This module is deliberately tiny and dependency-free: a stopwatch plus a
deadline. The policy that consumes it lives in :meth:`pybnf.algorithms.base.
Algorithm.run` (stop the run loop, cancel what is pending, finalize) and in
``pybnf.pybnf`` (do not start a refine or another bootstrap replicate once the
budget is spent).

The clock is :func:`time.monotonic`, so a system clock adjustment mid-fit cannot
lengthen or shorten a budget; the wall-clock ``time.time()`` origin the process
recorded at startup is folded in once, as an ``elapsed`` offset, so the budget
covers configuration loading and network generation too -- everything an external
``timeout`` around the ``pybnf`` process would have covered.
"""

import time


class FitBudget:
    """A fit's total wall-clock budget: ``limit`` seconds from the run's start.

    :param limit: The budget in seconds (``wall_time_fit``). Must be positive; a
        zero/absent ``wall_time_fit`` means "unbounded" and is represented by
        *no* FitBudget at all (:meth:`from_config` returns ``None``), not by a
        FitBudget with an infinite limit.
    :param elapsed: Seconds already spent before this object was built -- the gap
        between process start and the moment the configuration finished loading.
        Counted against the budget.
    :param clock: The monotonic clock to read; injectable for tests.
    """

    def __init__(self, limit, elapsed=0.0, clock=time.monotonic):
        self.limit = float(limit)
        self._clock = clock
        self._elapsed_before = float(elapsed)
        self._started_at = clock()

    @classmethod
    def from_config(cls, config, started_at=None, clock=time.monotonic):
        """Build the budget a configuration asks for, or ``None`` if it asks for none.

        ``started_at`` is a wall-clock (:func:`time.time`) stamp taken when the
        process began; the difference from *now* is charged to the budget before
        the deadline is set, so time spent loading the configuration and
        generating networks is inside the budget rather than free.
        """
        limit = config.config.get('wall_time_fit') or 0
        if limit <= 0:
            return None
        elapsed = 0.0 if started_at is None else max(0.0, time.time() - started_at)
        return cls(limit, elapsed=elapsed, clock=clock)

    def elapsed(self):
        """Seconds charged to the budget so far."""
        return self._elapsed_before + (self._clock() - self._started_at)

    def remaining(self):
        """Seconds left in the budget; never negative."""
        return max(0.0, self.limit - self.elapsed())

    def expired(self):
        """Whether the budget is spent."""
        return self.elapsed() >= self.limit

    def __repr__(self):
        return 'FitBudget(limit=%g, remaining=%g)' % (self.limit, self.remaining())


def format_duration(seconds):
    """``h:mm:ss`` for a duration in seconds -- the spelling ``main()`` already
    uses when it reports the total fitting time, so a budget message and the
    run's closing line read in the same units."""
    secs = max(0.0, float(seconds))
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return '%d:%02d:%02d' % (hrs, mins, secs)
