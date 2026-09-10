"""Rank-based uncertainty handling for a ranking-driven search on a stochastic model
(#661, ADR-0135): the method of Hansen, N., Niederberger, A. S. P., Guzzella, L. and
Koumoutsakos, P. (2009), "A method for handling uncertainty in evolutionary optimization
with an application to feedback control of combustion", IEEE Transactions on Evolutionary
Computation 13(1), 180-197, https://doi.org/10.1109/TEVC.2008.924423.

A stochastic simulation gives a different objective value every time it is run, so a
search that sorts its candidates on one value each is sorting partly on noise. CMA-ES reads
only that ordering, and scatter search's step sizes and acceptance tests are rank
comparisons too, so both can be pulled in arbitrary directions by it, and a step-size
adaptation that reads noise as stagnation shrinks a step that should not shrink.

The treatment measures how unreliable the ranking is and reacts only when it is. Each
generation a few candidates are simulated again at fresh seeds. Each re-evaluated
candidate's old and new values are ranked together, and how far a candidate moves in that
ranking is compared with how far it would move under pure noise (a quantile of the possible
rank changes at its position). The average excess, filtered over generations, is the
uncertainty level. Above zero, the search spends more simulations per candidate (their
average is ranked) and holds the step size up; below zero it spends fewer. Which
candidates are re-simulated is a random subset, so a lucky draw is not over-represented.

This module holds the measurement and the adaptation as a small picklable state object,
with no knowledge of any optimizer: :class:`RankChangeNoise` says how many candidates to
re-evaluate, turns a generation's old and new values into a level, and reports the number
of evaluations per candidate and the step-size factor to apply. CMA-ES drives it
(:mod:`pybnf.algorithms.optimizers.cmaes`); scatter search is its intended second consumer
(#660 step 3).
"""

import numpy as np


class RankChangeNoise:
    """The uncertainty level of a ranking, and the evaluations per candidate it calls for.

    ``n_dim`` sets the step-size factor ``1 + 2 / (n_dim + 10)`` (Hansen, Niederberger,
    Guzzella and Koumoutsakos 2009, cited in the module docstring);
    ``max_evals`` caps the evaluations per candidate; ``reevaluate_fraction`` is the share
    of a generation re-evaluated, floored at three candidates: with fewer, the pure-noise
    limit at every rank is zero, so the statistic can never come out negative and the
    evaluations per candidate could only ever grow. ``theta`` sets the quantile
    of the pure-noise rank change a real change is measured against, ``cum`` the filter
    on the level, and ``alpha_evals`` the factor the evaluations grow by.
    """

    def __init__(self, n_dim, max_evals=10, reevaluate_fraction=0.1, theta=0.5, cum=0.3,
                 alpha_evals=1.5):
        self.max_evals = max(1, int(max_evals))
        self.reevaluate_fraction = float(reevaluate_fraction)
        self.theta = float(theta)
        self.cum = float(cum)
        self.alpha_evals = float(alpha_evals)
        self.alpha_sigma = 1.0 + 2.0 / (int(n_dim) + 10.0)
        #: Evaluations per candidate, adapted; a float so the growth compounds smoothly.
        self.evals = 1.0
        #: The filtered uncertainty level: positive when the ranking is unreliable.
        self.level = 0.0
        #: The most recent generation's raw measurement, for the log.
        self.last_measurement = None

    def evaluations(self):
        """How many times each candidate of the next generation is simulated."""
        return max(1, min(self.max_evals, int(self.evals + 0.5)))

    def count_to_reevaluate(self, lam):
        """How many candidates of a generation of ``lam`` to simulate again: the fraction,
        floored at three, capped at the generation."""
        return max(1, min(int(lam), max(3, int(round(self.reevaluate_fraction * lam)))))

    def rank_change_limit(self, rank, count):
        """The rank change pure noise would produce at least ``theta * 50`` percent of the
        time for a value at ``rank`` among ``count`` others: that quantile of the possible
        rank changes ``|k - rank|`` for ``k = 1 .. count``."""
        if count < 1:
            return 0.0
        changes = np.sort(np.abs(np.arange(1, count + 1) - float(rank)))
        return float(np.percentile(changes, self.theta * 50.0, method='lower'))

    def measure(self, f_old, f_new):
        """One generation's rank-change statistic from the re-evaluated candidates' old
        and new values (``f_old`` and ``f_new``, parallel, both of length ``m``).

        The ``2 m`` values are ranked together, old before new on a tie. For each candidate
        the rank change ``|r_new - r_old| - 1`` (zero when the two are adjacent, as they are
        for a deterministic function) is doubled and reduced by the pure-noise limits at
        the two ranks, each taken among the ``2 m - 1`` other values. The statistic is the
        mean over candidates: positive when candidates move further than noise alone
        explains, negative when they move less.
        """
        f_old = np.asarray(f_old, dtype=float)
        f_new = np.asarray(f_new, dtype=float)
        m = len(f_old)
        if m == 0 or len(f_new) != m:
            return 0.0
        both = np.concatenate([f_old, f_new])
        # A non-finite value ranks last, and a tie ranks the old value first (stable sort).
        order = np.argsort(np.where(np.isfinite(both), both, np.inf), kind='stable')
        ranks = np.empty(2 * m, dtype=int)
        ranks[order] = np.arange(1, 2 * m + 1)
        total = 0.0
        for i in range(m):
            r_old, r_new = ranks[i], ranks[m + i]
            change = abs(r_new - r_old) - 1
            # Each value's rank among the other 2m - 1: drop the partner from below.
            lim_new = self.rank_change_limit(r_new - (1 if f_new[i] > f_old[i] else 0), 2 * m - 1)
            lim_old = self.rank_change_limit(r_old - (1 if f_old[i] > f_new[i] else 0), 2 * m - 1)
            total += 2.0 * change - lim_new - lim_old
        return total / m

    def update(self, measurement):
        """Fold one generation's measurement into the level and adapt.

        Returns the factor to multiply the step size by after the generation's update:
        ``alpha_sigma`` while the ranking is unreliable, else 1. Above zero the
        evaluations per candidate grow by ``alpha_evals`` up to the cap; at or below zero
        they shrink by ``alpha_evals ** 0.25`` down to one. Zero counts as reliable, as in
        the reference implementation.
        """
        self.last_measurement = float(measurement)
        self.level = (1.0 - self.cum) * self.level + self.cum * float(measurement)
        if self.level > 0.0:
            self.evals = min(float(self.max_evals), self.evals * self.alpha_evals)
            return self.alpha_sigma
        self.evals = max(1.0, self.evals * self.alpha_evals ** -0.25)
        return 1.0

    @staticmethod
    def combined(f_old, f_new):
        """The value a re-evaluated candidate is ranked on: the mean of its two."""
        return 0.5 * (np.asarray(f_old, dtype=float) + np.asarray(f_new, dtype=float))
