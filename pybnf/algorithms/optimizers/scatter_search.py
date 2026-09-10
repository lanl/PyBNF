"""ScatterSearch optimizer (the ``ss`` fit type).

Extracted byte-identical (M1 Step 4). Subclasses Algorithm and inherits its run
loop + execution seam; it makes no core.* call of its own.

A noise-aware reference set for a stochastic model (#660 step 3, ADR-0136). Scatter search
never needs its objective values to be precise, only its ordering to be right: whether a
child beat its parent, whether a member is stuck, and the rank gap that sets a
combination's step size are all ranking decisions. For a stochastic model each value is
one draw, so the reference set used to store a lucky draw as fact, no honest child could
beat it, and its stuck counter climbed until it was retired into the archive as a local
minimum it never was. When running a parameter set again would give a different answer
(``_replicates_would_differ``) and ``ss_noise_handling`` is on, every reference member
keeps its draws and is ranked on their mean, the draw-to-draw spread is pooled across the
whole fit (:func:`pybnf.algorithms.noise_handling.pooled_sd`), and a decision that the
spread leaves in doubt is not made: both sides are drawn again, up to
``ss_noise_max_draws`` draws each, and the decision waits for those draws. A member is
never counted stuck without being drawn again, so a lucky member's mean regresses to its
true value and an honest child can beat it. A deterministic fit makes every decision on
its single draws, as before, and is byte-identical.
"""
from ..base import Algorithm
from ..noise_handling import pooled_sd, separated
from .multistart import MultiStartConfig, MultiStartOptimizer
from ...pset import PSet
from ...printing import print1, print2
from ...registry import register_fit_type

import copy
import logging

import numpy as np
from pydantic import Field


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class ScatterSearchConfig(MultiStartConfig):
    """Scatter-search config fields, co-located with the method (ADR-0002,
    ADR-0006). ``local_min_limit`` is the only defaulted scatter key (it was the
    lone ``# --- scatter search ---`` entry in ``GlobalConfig``); ``init_size``
    and ``reserve_size`` are NOT here -- like PSO's ``particle_weight_final`` they
    are runtime-defaulted in ``__init__`` (``init_size`` → ``10*len(variables)``,
    ``reserve_size`` → ``max_iterations`` when absent), so they stay pass-through
    extras. Value byte-identical to the old global default. Inherits the shared
    ``n_starts`` multi-start field (``MultiStartConfig``, #498).
    """

    local_min_limit: int = 5
    # A noise-aware reference set for a stochastic model (#660 step 3, ADR-0136): the
    # switch, and the cap on draws per parameter set a decision in doubt may spend.
    ss_noise_handling: int = 1
    ss_noise_max_draws: int = Field(default=5, ge=1)

    # init_size (-> 10*len(variables)) and reserve_size (-> max_iterations) default at
    # runtime in __init__, so they are not schema fields but ARE valid ss keys (#401).
    RUNTIME_KEYS = frozenset({'init_size', 'reserve_size'})


@register_fit_type('ss', family='optimizer', display_name='Scatter Search',
                   schema=ScatterSearchConfig)
class ScatterSearch(MultiStartOptimizer, Algorithm):
    """
    Implements ScatterSearch as described in the introduction of Penas et al 2017 (but not the fancy parallelized
    version from that paper).
    Uses the individual combination method described in Egea et al 2009

    """

    # This fit runs a whole round of combinations, waits for all of it to finish, then
    # builds the next round, so some idle workers toward the end of each round are
    # expected (#621).
    waits_for_full_generation = True

    def __init__(self, config):  # variables, popsize, maxiters, saveevery):

        super().__init__(config)

        self.popsize = config.config['population_size']
        if self.popsize < 3:
            print1('Scatter search requires a population size of at least 3. '
                   'Increasing the population size to 3.')
            logger.warning('Increasing population_size to the minimum allowed value of 3')
            self.config.config['population_size'] = 3
            self.popsize = 3
        self.max_iterations = config.config['max_iterations']
        if 'reserve_size' in config.config:
            self.reserve_size = config.config['reserve_size']
        else:
            self.reserve_size = self.max_iterations
        if 'init_size' in config.config:
            self.init_size = config.config['init_size']
            if self.init_size < self.popsize:
                logger.warning('init_size less than population_size. Setting it equal to population_size.')
                print1("Scatter search parameter 'init_size' cannot be less than 'population_size'. "
                       "Automatically setting it equal to population_size.")
                self.init_size = self.popsize
        else:
            self.init_size = 10*len(self.variables)
            if self.init_size < self.popsize:
                logger.warning('init_size less than population_size. Setting it equal to population_size.')
                self.init_size = self.popsize

        self.local_min_limit = config.config['local_min_limit']
        # Noise-aware reference set (#660 step 3, ADR-0136): only where running a parameter
        # set again would give a different answer, and only when asked for (the default).
        self.max_draws = int(config.config.get('ss_noise_max_draws', 5))
        self.noise_handling = bool(config.config.get('ss_noise_handling', 1)) \
            and self._replicates_would_differ()
        if self.noise_handling:
            logger.info('Scatter search noise handling is on: a stochastic model gives a '
                        'different objective value every run, so reference members are ranked '
                        'on the mean of their draws and a decision the noise leaves in doubt '
                        'draws again (up to %d draws per parameter set)' % self.max_draws)

        self.pending = dict() # {pendingPSet: parentPSet}
        self.received = dict() # {parentPSet: [(donependingPSet, score)]
        self.refs = [] # (refPset, score)
        self.stuckcounter = dict()
        self.iteration = 0
        self.local_mins = [] # (Pset, score) pairs that were stuck for 5 gens, and so replaced.
        self.reserve = []
        # Noise-aware bookkeeping (#660 step 3): every draw of each reference member and
        # contender, the child whose contest with its parent was left in doubt, and the
        # re-draws in flight by the name they were queued under.
        self.draws = dict()        # {PSet: [score, ...]}
        self.contenders = dict()   # {parentPSet: childPSet}
        self.pending_draws = dict()  # {queued name: PSet the draw belongs to}
        # Every draw list that ever reached two draws, kept whether or not its parameter
        # set is still in play, since the pooled spread is a property of the fit and must
        # not forget a member the moment a child replaces it.
        self.repeat_draws = dict()   # {PSet: the same list as in draws}

    def expected_parallelism(self):
        """Scatter search runs up to ``population_size * (population_size - 1)``
        simulations at a time, one for every ordered pair in the reference set, and keeps
        doing that for the rest of the fit (#655).

        The first batch of jobs the run loop submits is a different number: it is the
        ``init_size`` random parameter sets the initialization round scores, which by
        default is ten per free parameter and has nothing to do with the population. On a
        cluster large enough to matter the two are far apart, so the parallelism report
        has to be told which one describes the fit. This is the same number
        ``_search_start_run`` already prints as "simulations per iteration".
        """
        return self.popsize * (self.popsize - 1)

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._reset_search_state()

    def _reset_search_state(self):
        """Clear all search-specific state (reference set, pending/received maps,
        stuck-counters, iteration count, local-minimum archive, reserve) WITHOUT
        touching the trajectory -- so multi-start's :meth:`_search_start_run` begins a
        fresh, independent scatter search each start while the trajectory keeps the
        global best across starts (#498). Shared by ``reset`` (which also clears the
        trajectory via ``super().reset``) and by each multi-start."""
        self.pending = dict()
        self.received = dict()
        self.refs = []
        self.stuckcounter = dict()
        self.iteration = 0
        self.local_mins = []
        self.reserve = []
        self.draws = dict()
        self.contenders = dict()
        self.pending_draws = dict()
        self.repeat_draws = dict()

    def _search_start_run(self):
        # Reset every search counter first (iteration / refs / archive that start_run
        # itself does not touch), so a multi-start restart begins a genuinely fresh
        # scatter search rather than resuming at the previous start's iteration.
        self._reset_search_state()
        print2('Running Scatter Search with population size %i (%i simulations per iteration) for %i iterations' %
               (self.popsize, self.popsize * (self.popsize - 1), self.max_iterations))
        # Generate big number = 10 * variable_count (or user's chosen init_size) initial individuals.
        if self.config.config['initialization'] == 'lh':
            psets = self.random_latin_hypercube_psets(self.init_size)
        else:
            psets = [self.random_pset() for i in range(self.init_size)]
        # ADR-0043 Phase 2: seed exactly one initial individual at the initial_value point
        # (a no-op unless a parameter: record declares one). Only the main psets are seeded
        # -- the latin-hypercube reserve below stays fully random.
        psets[0] = self._seed_start_point_pset(psets[0])
        for i in range(len(psets)):
            psets[i].name = 'init%i' % i

        # Generate a latin hypercube distributed "reserve". When we need a random new individual, pop one from here
        # so we aren't repeating ground. Size of this could be customizable.
        # Note that this is not part of the original algorithm description, Eshan made it up
        # because otherwise, the "choose a new random point" step of the algorithm can cause useless repetition.
        if self.reserve_size > 0:
            self.reserve = self.random_latin_hypercube_psets(self.reserve_size)
        else:
            self.reserve = []

        self.pending = {p: None for p in psets}
        self.received = {None: []}
        return psets

    def round_1_init(self):
        start_psets = sorted(self.received[None], key=lambda x: x[1])
        # Half is the top of the list, half is random.
        topcount = int(np.ceil(self.popsize / 2.))
        randcount = int(np.floor(self.popsize / 2.))
        self.refs = start_psets[:topcount]
        randindices = self.rng.choice(np.arange(topcount, len(start_psets)), randcount, replace=False)
        for i in randindices:
            self.refs.append(start_psets[i])
        self.stuckcounter = {r[0]: 0 for r in self.refs}

    def _search_got_result(self, res):
        """
        Called when a simulation run finishes

        :param res:
        :type res: Result
        :return:
        """

        ps = res.pset
        score = res.score

        if ps.name in self.pending_draws:
            # A fresh draw of a parameter set already in play (#660 step 3).
            target = self.pending_draws.pop(ps.name)
            draws = self.draws.setdefault(target, [])
            draws.append(score)
            self.repeat_draws[target] = draws
        else:
            parent = self.pending[ps]
            self.received[parent].append((ps, score))
            del self.pending[ps]

        if self.pending or self.pending_draws:
            return []

        # All of this generation done, make the next list of psets
        redraws = []
        if None in self.received:
            # This is the initialization round, special case
            self.round_1_init()
            for member, score_ in self.refs:
                self.draws[member] = [score_]
            if self.noise_handling:
                # Bootstrap the pooled noise estimate: one more draw of every member goes
                # out with the first round of children, so the first decisions have it.
                redraws += self._redraws(member for member, _ in self.refs)
        else:
            # 1) Replace parent with highest scoring child
            redraws += self._update_reference_set()

        # 2) Sort the refs list by quality.
        self.refs = sorted(self.refs, key=lambda x: x[1])
        if self.noise_handling:
            # The rank gap between neighbours sets a combination's step size, so a pair
            # the noise cannot order is worth another draw each (#660 step 3).
            redraws += self._redraws(self._unseparated_neighbours())
        logger.debug('Iteration %i' % self.iteration)
        if self.iteration % 10 == 0:
            print1('Completed iteration %i of %i' % (self.iteration, self.max_iterations))
        else:
            print2('Completed iteration %i of %i' % (self.iteration, self.max_iterations))
        print2('Current scores: ' + str([x[1] for x in self.refs]))
        print2('Best archived scores: ' + str([x[1] for x in self.local_mins]))
        if self.noise_handling:
            sd = self._noise_sd()
            print2('Draw-to-draw spread of the objective: %s; %d parameter set(s) drawn again'
                   % ('not yet measured' if sd is None else '%.4g' % sd, len(redraws)))

        if self.iteration % self.config.config['output_every'] == 0:
            self.output_results()

        self.iteration += 1
        if self.iteration == self.max_iterations:
            return 'STOP'

        # 3) Do the combination antics to generate new candidates
        query_psets = []
        for pi in range(self.popsize): # parent index
            for hi in range(self.popsize): # helper index
                if pi == hi:
                    continue
                new_vars = []
                for v in self.variables:
                    d = self.refs[hi][0].get_param(v.name).diff(self.refs[pi][0].get_param(v.name))
                    alpha = np.sign(hi-pi)
                    beta = (abs(hi-pi) - 1) / (self.popsize - 2)
                    new_vars.append(self.refs[pi][0].get_param(v.name).add_rand(-d*(1 + alpha*beta), d*(1 - alpha * beta), self.rng))
                newpset = PSet(new_vars)
                # Check to avoid duplicate PSets. If duplicate, don't have to try again because SS doesn't really
                # care about the number of PSets queried.
                if newpset not in self.pending:
                    newpset.name = 'iter%ip%ih%i' % (self.iteration, pi, hi)
                    query_psets.append(newpset)
                    self.pending[newpset] = self.refs[pi][0]
                else:
                    print(newpset)
        self.received = {r[0]: [] for r in self.refs}
        return query_psets + redraws

    # --- the noise-aware reference set (#660 step 3, ADR-0136) ------------ #
    def _update_reference_set(self):
        """Step 1 of a round: decide, for every reference member, whether its best child
        (or the contender left over from a round in doubt) replaces it, whether it is
        stuck, or whether the noise leaves the contest undecided. Returns the re-draws a
        deferred or stuck decision asks for.

        A deterministic fit (``noise_handling`` off) makes every decision on single
        draws, exactly as before: a child strictly better than its parent replaces it,
        anything else counts the parent stuck, and ``local_min_limit`` stuck rounds retire
        it into the archive. With noise handling on, both sides are estimates: a child
        replaces its parent only when it is better by more than the noise says two such
        estimates can differ by chance; a child clearly worse counts the parent stuck; and
        a contest in doubt is left open, with both sides drawn again, until it is settled
        or both have spent ``ss_noise_max_draws`` draws, at which point the means decide.
        A parent counted stuck is drawn again too, so a member that is only stuck because
        its recorded value was a lucky draw regresses to its true value and can be beaten.
        """
        redraws = []
        for i in range(len(self.refs)):
            parent = self.refs[i][0]
            p_mean, p_n = self._estimate(parent, fallback=self.refs[i][1])
            children = self.received[parent]
            contender = self.contenders.pop(parent, None)
            # A reference whose candidate children all collided with existing pending
            # psets receives no children -- this happens once the reference set collapses
            # on a smooth target (the combination step reproduces the parent point, which
            # is skipped as a duplicate). Treat the childless reference as non-improving so
            # the stuck-counter machinery eventually perturbs or retires it, rather than
            # crashing on min() of an empty list.
            candidates = [(child, [score]) for child, score in children]
            if contender is not None:
                candidates.append((contender, list(self.draws.get(contender, []))))
            best = None
            for child, child_draws in candidates:
                mean, n = self._estimate_draws(child_draws)
                if best is None or mean < best[1]:
                    best = (child, mean, n, child_draws)
            if best is None:
                self.refs[i] = (parent, p_mean)
                redraws += self._count_stuck(i)
                continue
            child, c_mean, c_n, child_draws = best
            better = c_mean < p_mean
            settled = (not self.noise_handling
                       or separated(c_mean, c_n, p_mean, p_n, self._noise_sd())
                       or (c_n >= self.max_draws and p_n >= self.max_draws))
            if better and settled:
                del self.stuckcounter[parent]
                self.stuckcounter[child] = 0
                self.refs[i] = (child, c_mean)
                self.draws[child] = child_draws
                self.draws.pop(parent, None)
                continue
            self.refs[i] = (parent, p_mean)
            if not settled:
                # In doubt: keep the contest open and draw both sides again.
                self.contenders[parent] = child
                self.draws[child] = child_draws
                redraws += self._redraws([child, parent])
                continue
            redraws += self._count_stuck(i)
        return redraws

    def _count_stuck(self, i):
        """Count reference ``i`` stuck for this round, retiring it into the archive at
        ``local_min_limit`` and refilling its slot from the reserve. With noise handling
        on, a stuck member is drawn again (up to the cap), since a member that keeps
        beating its children is exactly the one whose recorded value is being trusted."""
        parent = self.refs[i][0]
        self.stuckcounter[parent] += 1
        if self.stuckcounter[parent] < self.local_min_limit:
            return self._redraws([parent]) if self.noise_handling else []
        del self.stuckcounter[parent]
        self.contenders.pop(parent, None)
        self.draws.pop(parent, None)
        self.local_mins.append(self.refs[i])
        # For output. Not the most efficient, but not in a performance-critical section
        self.local_mins = sorted(self.local_mins, key=lambda x: x[1])
        self.local_mins = self.local_mins[:self.popsize] # So this doesn't get huge

        # Pick a new random pset
        if len(self.reserve) > 0:
            new_pset = self.reserve.pop()
        else:
            new_pset = self.random_pset()
        self.refs[i] = (new_pset, np.inf)  # For simplicity, assume its score is awful
        self.stuckcounter[new_pset] = 0
        self.draws[new_pset] = []
        return []

    def _estimate(self, pset, fallback):
        """``(mean, n)`` for a reference member from its recorded draws, or from the score
        the reference set carries when nothing was recorded (a member placed by hand)."""
        draws = self.draws.get(pset)
        if not draws:
            return (float(fallback), 1) if np.isfinite(fallback) else (np.inf, 0)
        return self._estimate_draws(draws)

    @staticmethod
    def _estimate_draws(draws):
        """``(mean, n)`` over a list of draws: the mean, or ``inf`` when any draw failed,
        since a parameter set that cannot be simulated reliably must not win on the draws
        that happened to succeed; ``n`` is the number of draws."""
        values = np.asarray(draws, dtype=float)
        if values.size == 0:
            return np.inf, 0
        if not np.all(np.isfinite(values)):
            return np.inf, int(values.size)
        return float(values.mean()), int(values.size)

    def _noise_sd(self):
        """The pooled draw-to-draw standard deviation over everything ever drawn more
        than once, or ``None`` before any parameter set has been."""
        return pooled_sd(self.repeat_draws.values())

    def _unseparated_neighbours(self):
        """The members of every adjacent pair in the sorted reference set that the noise
        cannot order, each once."""
        sd = self._noise_sd()
        wanted = []
        for j in range(len(self.refs) - 1):
            a, b = self.refs[j], self.refs[j + 1]
            a_mean, a_n = self._estimate(a[0], fallback=a[1])
            b_mean, b_n = self._estimate(b[0], fallback=b[1])
            if not separated(a_mean, a_n, b_mean, b_n, sd):
                for member in (a[0], b[0]):
                    if not any(member is w for w in wanted):
                        wanted.append(member)
        return wanted

    def _redraws(self, psets):
        """Queue one fresh draw of each of ``psets`` that has a name, has been drawn at
        least once, and has not reached ``ss_noise_max_draws``; returns the queued copies.
        Each copy is named for the draw it is and carries a replicate offset past every
        draw the parameter set has had, so it is a fresh trajectory under the default seed
        policy (ADR-0135). A draw already queued this round is not queued twice."""
        smoothing = max(1, int(self.config.config.get('smoothing') or 1))
        queued = []
        for pset in psets:
            draws = self.draws.get(pset, [])
            n = len(draws)
            if pset.name is None or n < 1 or n >= self.max_draws:
                continue
            name = '%s_d%i' % (pset.name, n)
            if name in self.pending_draws:
                continue
            again = copy.copy(pset)
            again.name = name
            again.replicate_offset = n * smoothing
            self.pending_draws[name] = pset
            queued.append(again)
        return queued

    def get_backup_every(self):
        """
        Overrides base method because Scatter Search runs n*(n-1) PSets per iteration.
        """
        return self.config.config['backup_every'] * self.config.config['population_size'] * \
            (self.config.config['population_size']-1) * self.config.config['smoothing']
