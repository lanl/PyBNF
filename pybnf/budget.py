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

One budget still bounds the whole run, but it is no longer first-come-first-served:
a budget may hold back a **reserve** -- a tail of seconds the search phase may not
spend, kept for the post-fit refine (``wall_time_refine_frac``, #564/ADR-0107).
While the reserve is held, :meth:`FitBudget.expired` is the *phase's* deadline, not
the run's; :func:`spend_reserve` releases it for the duration of the refine. Without
a refine to protect there is no reserve, and the budget is exactly the ADR-0093 one.
"""

import time
from contextlib import contextmanager


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
    :param reserve: Seconds of the limit held back for a later phase -- the refine
        (#564). The search sees a budget shorter by this much; :func:`spend_reserve`
        hands it to the phase it was kept for. ``0`` (the default) is the ADR-0093
        budget, spent first-come-first-served.
    """

    def __init__(self, limit, elapsed=0.0, clock=time.monotonic, reserve=0.0):
        self.limit = float(limit)
        self.reserve = max(0.0, min(float(reserve), float(limit)))
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

        The refine's reserve is sized here, from :func:`refine_reserve_seconds`, so
        the phase split is fixed before a single job is submitted rather than
        renegotiated as the clock runs down.
        """
        limit = config.config.get('wall_time_fit') or 0
        if limit <= 0:
            return None
        elapsed = 0.0 if started_at is None else max(0.0, time.time() - started_at)
        return cls(limit, elapsed=elapsed, clock=clock,
                   reserve=refine_reserve_seconds(config.config))

    def elapsed(self):
        """Seconds charged to the budget so far."""
        return self._elapsed_before + (self._clock() - self._started_at)

    def remaining(self):
        """Seconds left for the *current phase*; never negative.

        While a reserve is held back this is short of the run's own remaining time
        by exactly the reserve -- which is the number the search must be bounded by,
        since it is what the search is allowed to spend.
        """
        return max(0.0, self.limit - self.reserve - self.elapsed())

    def expired(self):
        """Whether the current phase's share of the budget is spent."""
        return self.elapsed() >= self.limit - self.reserve

    def __repr__(self):
        return ('FitBudget(limit=%g, reserve=%g, remaining=%g)'
                % (self.limit, self.reserve, self.remaining()))


@contextmanager
def spend_reserve(budget):
    """Make a budget's reserved tail spendable for the duration of the block.

    The refine is the phase the reserve was kept for, so inside this block the
    budget is the run's whole remaining time -- the reserve *plus* whatever the
    search left unspent by converging early. The reserve is restored on the way
    out, so a bootstrap run's next replicate searches under the same phase split
    its predecessor did.

    A no-op for ``None`` (an unbudgeted run), so callers need no guard.
    """
    if budget is None:
        yield None
        return
    held = budget.reserve
    budget.reserve = 0.0
    try:
        yield budget
    finally:
        budget.reserve = held


def refine_reserve_seconds(conf):
    """Seconds of ``wall_time_fit`` to hold back for the refine, from a config dict.

    ``refine = 1`` asks for a *method* -- "search globally, then polish locally" --
    and a wall-clock-budgeted search has no reason to leave anything behind, so
    without a reserve the polish essentially never runs (#564). The reserve is
    ``wall_time_refine_frac`` of the budget, and it is taken only when a refine will
    actually be attempted: no ``refine``, no budget, or a ``refine_method`` naming the
    algorithm the fit itself ran (which :func:`pybnf.pybnf._refine_best_fit` skips)
    all leave the search the whole budget, exactly as before.
    """
    limit = conf.get('wall_time_fit') or 0
    if limit <= 0 or conf.get('refine') != 1:
        return 0.0
    if conf.get('refine_method', 'sim') == conf.get('fit_type'):
        return 0.0
    frac = conf.get('wall_time_refine_frac')
    if frac is None:
        frac = 0.0
    return max(0.0, min(float(frac), 1.0)) * float(limit)


def format_duration(seconds):
    """``h:mm:ss`` for a duration in seconds -- the spelling ``main()`` already
    uses when it reports the total fitting time, so a budget message and the
    run's closing line read in the same units."""
    secs = max(0.0, float(seconds))
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return '%d:%02d:%02d' % (hrs, mins, secs)
