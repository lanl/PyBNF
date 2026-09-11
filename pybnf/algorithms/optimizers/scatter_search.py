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
from ... import edition
from .concurrent_multistart import DONE
from .multistart import MultiStartConfig, MultiStartOptimizer
from .simplex import SimplexRunner
from ...pset import PSet
from ...printing import print1, print2
from ...registry import register_fit_type

import copy
import logging
from typing import Optional

import numpy as np
from pydantic import Field


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')

#: The improvement method's Nelder-Mead constants (#660 step 1, ADR-0138): the simplex
#: fit type's own defaults, and a stop tolerance on the largest move in sampling space,
#: below which a refinement has converged as far as a search inside another search needs.
_LOCAL_REFLECTION, _LOCAL_EXPANSION, _LOCAL_CONTRACTION, _LOCAL_SHRINK = 1.0, 1.0, 0.5, 0.5
_LOCAL_STOP_TOL = 1e-4
#: The initial simplex's edge, as a fraction of the reference set's spread per coordinate.
_LOCAL_STEP_FRACTION = 0.1
#: The distance filter: a candidate closer than this, in root-mean-square per-coordinate
#: units of the initial population's spread, to a previous refinement's start or optimum
#: is in a basin already refined and is not refined again.
_LOCAL_MIN_DISTANCE = 0.05


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
    # The diverse half of the first reference set chosen by distance (#660 step 2,
    # ADR-0137): unset resolves to on under edition 2 and off under the legacy edition,
    # whose contract is that an unchanged conf keeps behaving as it always has.
    ss_diverse_by_distance: Optional[int] = None
    # The improvement method (#660 step 1, ADR-0138): a Nelder-Mead refinement of the best
    # child a round accepts, at most every ss_local_every rounds and ss_local_max_running
    # at a time, for ss_local_max_iterations simplex iterations. Unset follows the edition.
    ss_local_search: Optional[int] = None
    ss_local_every: int = Field(default=10, ge=1)
    ss_local_max_iterations: int = Field(default=50, ge=1)
    ss_local_max_running: int = Field(default=1, ge=1)

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
        # The diverse half of the first reference set (#660 step 2, ADR-0137): Glover's
        # template fills it with the members most distant from what is already in the set;
        # the pre-#660 code picked it at random. On by default under a modern edition, off
        # under the legacy one (ADR-0031); an explicit ss_diverse_by_distance wins.
        self.diverse_by_distance = self._resolve_diverse_by_distance()
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
        # The improvement method (#660 step 1, ADR-0138): Glover's template refines the
        # candidates combination produces with a local search. On by default under a modern
        # edition, off under the legacy one; an explicit ss_local_search wins; and off
        # whenever noise handling is on, since a simplex over single draws of a stochastic
        # model converges on the noise rather than the objective.
        self.local_search = self._resolve_local_search()
        self.local_every = int(config.config.get('ss_local_every', 10))
        self.local_max_iterations = int(config.config.get('ss_local_max_iterations', 50))
        self.local_max_running = int(config.config.get('ss_local_max_running', 1))
        if self.local_search:
            logger.info('Scatter search improvement method is on: the best child a round '
                        'accepts is refined by a Nelder-Mead simplex, at most every %d rounds '
                        'and %d at a time, for up to %d simplex iterations'
                        % (self.local_every, self.local_max_running, self.local_max_iterations))

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
        # The improvement method's bookkeeping (#660 step 1): the refinements in flight and
        # the ones finished but not yet folded into the reference set, each under a tag,
        # the member each started from, the tagged names of their jobs, every start and
        # optimum so far (the distance filter), and the round the last one started in.
        self.local_runners = dict()     # tag -> SimplexRunner in flight
        self.local_finished = dict()    # tag -> SimplexRunner that returned DONE
        self.local_origins = dict()     # tag -> the reference member it started from
        self.pending_local = dict()     # tagged pset name -> tag
        self.local_starts = []          # normalized u-vectors of every start
        self.local_optima = []          # (PSet, score) of every finished refinement
        self.local_count = 0
        self.last_local_iteration = None
        self._init_spread = None        # per-coordinate spread of the initial population in u
        self._round_accepted = []       # (child, score) accepted into the reference set this round

    def _resolve_local_search(self):
        """Whether the improvement method runs: never under noise handling; otherwise an
        explicit ``ss_local_search`` wins, and unset it is on under a modern edition and off
        under the legacy one (ADR-0031, ADR-0138)."""
        if self.noise_handling:
            if self.config.config.get('ss_local_search'):
                logger.info('ss_local_search is set, but this fit uses a stochastic model with '
                            'noise handling on, and a simplex over single draws converges on '
                            'the noise; the improvement method stays off')
            return False
        configured = self.config.config.get('ss_local_search')
        if configured is not None:
            return bool(int(configured))
        return edition.is_modern(edition.resolve_edition(self.config.config.get('edition')))

    def _resolve_diverse_by_distance(self):
        """Whether the first reference set's second half is chosen by distance: an explicit
        ``ss_diverse_by_distance`` wins; unset, it is on under a modern edition and off under
        the legacy one, whose contract is that an unchanged conf keeps behaving as it always
        has (ADR-0031, ADR-0137)."""
        configured = self.config.config.get('ss_diverse_by_distance')
        if configured is not None:
            return bool(int(configured))
        return edition.is_modern(edition.resolve_edition(self.config.config.get('edition')))

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
        self.local_runners = dict()
        self.local_finished = dict()
        self.local_origins = dict()
        self.pending_local = dict()
        self.local_starts = []
        self.local_optima = []
        self.local_count = 0
        self.last_local_iteration = None
        self._init_spread = None
        self._round_accepted = []

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
        # Half is the top of the list; the other half is the most diverse of the rest
        # (Glover's template, #660 step 2) or, under the legacy edition, random.
        topcount = int(np.ceil(self.popsize / 2.))
        randcount = int(np.floor(self.popsize / 2.))
        self.refs = start_psets[:topcount]
        if self.diverse_by_distance:
            self.refs += self._most_diverse(start_psets[topcount:], randcount)
        else:
            randindices = self.rng.choice(np.arange(topcount, len(start_psets)), randcount, replace=False)
            for i in randindices:
                self.refs.append(start_psets[i])
        self.stuckcounter = {r[0]: 0 for r in self.refs}

    def _most_diverse(self, candidates, count):
        """The ``count`` candidates that, added one at a time, each maximize the distance
        to the nearest member already in the reference set (#660 step 2, ADR-0137).

        Glover's template fills the second half of the first reference set with its most
        *diverse* members, not a random sample of the rest: a random half is diverse only
        on average, which in many dimensions is far weaker than choosing for it. The
        greedy max-min rule is the standard construction. Distance is Euclidean in sampling
        space, so a log-scaled parameter is measured on its log scale, with each coordinate
        divided by its spread over the whole initial population so no parameter dominates
        by its units. Ties go to the better score, since ``candidates`` arrive sorted by
        it. A candidate whose score is not finite is taken only when no finite one is left:
        a point the model could not simulate is not a useful reference, however far away.
        """
        if count <= 0 or not candidates:
            return []
        members = [self._param_vec(p) for p, _ in self.refs]
        pool = [(p, s, self._param_vec(p)) for p, s in candidates]
        everything = np.array(members + [u for _, _, u in pool], dtype=float)
        spread = everything.max(axis=0) - everything.min(axis=0)
        spread = np.where(np.isfinite(spread) & (spread > 0.0), spread, 1.0)
        chosen = []
        current = [u / spread for u in members]
        remaining = [(p, s, u / spread) for p, s, u in pool]
        while remaining and len(chosen) < count:
            finite = [c for c in remaining if np.isfinite(c[1])] or remaining
            best, best_distance = None, -1.0
            for candidate in finite:
                distance = min((float(np.linalg.norm(candidate[2] - m)) for m in current),
                               default=np.inf)
                if distance > best_distance:
                    best, best_distance = candidate, distance
            chosen.append((best[0], best[1]))
            current.append(best[2])
            remaining = [c for c in remaining if c is not best]
        return chosen

    def _search_got_result(self, res):
        """
        Called when a simulation run finishes

        :param res:
        :type res: Result
        :return:
        """

        ps = res.pset
        score = res.score

        if ps.name in self.pending_local:
            # A refinement's evaluation (#660 step 1): advance that search and queue its
            # next step at once; a round never waits for a refinement.
            return self._advance_local_search(ps, score)
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
            self._init_spread = self._spread([p for p, _ in self.received[None]])
            for member, score_ in self.refs:
                self.draws[member] = [score_]
            if self.noise_handling:
                # Bootstrap the pooled noise estimate: one more draw of every member goes
                # out with the first round of children, so the first decisions have it.
                redraws += self._redraws(member for member, _ in self.refs)
        else:
            # 1) Replace parent with highest scoring child
            redraws += self._update_reference_set()
        # 1b) Fold in every refinement that finished since the last round (#660 step 1).
        self._fold_finished_local_searches()

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

        # 2b) Refine the best child this round accepted, when the filters allow (#660 step 1).
        local_psets = self._maybe_start_local_search()

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
        return query_psets + redraws + local_psets

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
        self._round_accepted = []
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
                self._round_accepted.append((child, c_mean))
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

    # --- the improvement method (#660 step 1, ADR-0138) ---------------------- #
    def _spread(self, psets):
        """The per-coordinate spread of ``psets`` in sampling space, as an array over the
        variables; a coordinate with no spread (or none measurable) reads as 1."""
        if not psets:
            return np.ones(len(self.variables))
        u = np.array([self._param_vec(p) for p in psets], dtype=float)
        spread = u.max(axis=0) - u.min(axis=0)
        return np.where(np.isfinite(spread) & (spread > 0.0), spread, 1.0)

    def _normalized(self, pset):
        """``pset``'s sampling-space vector divided by the initial population's spread."""
        spread = self._init_spread if self._init_spread is not None else np.ones(len(self.variables))
        return np.asarray(self._param_vec(pset), dtype=float) / spread

    def _local_steps(self):
        """The initial simplex's per-variable edge in sampling space: a tenth of the
        reference set's spread, falling back to a tenth of the initial population's, and
        to one unit where neither has any."""
        current = self._spread([p for p, _ in self.refs])
        fallback = self._init_spread if self._init_spread is not None else np.ones(len(self.variables))
        steps = {}
        for v, cur, init in zip(self.variables, current, fallback):
            base = cur if cur > 0.0 and np.isfinite(cur) else init
            steps[v.name] = _LOCAL_STEP_FRACTION * (base if base > 0.0 and np.isfinite(base) else 1.0)
        return steps

    def _tag_local(self, tag, pset):
        pset.name = '%s_%s' % (tag, pset.name)
        self.pending_local[pset.name] = tag
        return pset

    def _maybe_start_local_search(self):
        """Start a refinement from the best child this round accepted, when the improvement
        method is on and Egea's filters allow: not more often than every ``ss_local_every``
        rounds, not more than ``ss_local_max_running`` at a time, and not from a point
        within ``_LOCAL_MIN_DISTANCE`` of a refinement already started or already found.
        Returns the refinement's first jobs, tagged, or nothing."""
        if not self.local_search or not self._round_accepted:
            return []
        if len(self.local_runners) >= self.local_max_running:
            return []
        if (self.last_local_iteration is not None
                and self.iteration - self.last_local_iteration < self.local_every):
            return []
        child, score = min(self._round_accepted, key=lambda x: x[1])
        if not np.isfinite(score):
            return []
        u = self._normalized(child)
        scale = np.sqrt(len(self.variables))
        seen = self.local_starts + [self._normalized(p) for p, _ in self.local_optima]
        if any(float(np.linalg.norm(u - v)) / scale < _LOCAL_MIN_DISTANCE for v in seen):
            logger.debug('Scatter search: not refining %s, within %g of a refinement already made'
                         % (child.name, _LOCAL_MIN_DISTANCE))
            return []
        tag = 'ls%i' % self.local_count
        self.local_count += 1
        start = copy.copy(child)
        runner = SimplexRunner(self.variables, np.random.default_rng(int(self.rng.integers(2 ** 32))),
                               start, self._local_steps(), self.local_max_iterations,
                               max(len(self.variables) - 1, 1),
                               _LOCAL_REFLECTION, _LOCAL_EXPANSION, _LOCAL_CONTRACTION,
                               _LOCAL_SHRINK, _LOCAL_STOP_TOL)
        self.local_runners[tag] = runner
        self.local_origins[tag] = child
        self.local_starts.append(u)
        self.last_local_iteration = self.iteration
        print2('Refining %s (objective %g) by a simplex search (%s)' % (child.name, score, tag))
        logger.info('Scatter search: starting refinement %s from %s at objective %g'
                    % (tag, child.name, score))
        return [self._tag_local(tag, p) for p in runner.start()]

    def _advance_local_search(self, pset, score):
        """Route one refinement result to its search and return the search's next jobs; a
        finished search is set aside to be folded in at the next round boundary."""
        tag = self.pending_local.pop(pset.name)
        runner = self.local_runners.get(tag)
        if runner is None:
            return []                       # a straggler of a search already set aside
        pset.name = pset.name[len(tag) + 1:]
        out = runner.got(pset, score)
        if out is DONE:
            del self.local_runners[tag]
            self.local_finished[tag] = runner
            logger.info('Scatter search: refinement %s finished at objective %g (%s)'
                        % (tag, runner.fval if runner.fval is not None else np.inf,
                           runner.stop_reason))
            return []
        return [self._tag_local(tag, p) for p in out]

    def _fold_finished_local_searches(self):
        """Put each finished refinement's best point into the reference set: in place of
        the member it started from when that member is still there and the point is
        better, else in place of the worst member when it beats that, else into the
        archive. Every optimum is recorded for the distance filter."""
        for tag in sorted(self.local_finished):
            runner = self.local_finished.pop(tag)
            origin = self.local_origins.pop(tag, None)
            if not runner.simplex:
                continue
            best_score, best_pset = min(runner.simplex, key=lambda x: x[0])
            best_pset = copy.copy(best_pset)
            best_pset.name = '%s_best' % tag
            self.local_optima.append((best_pset, best_score))
            slot = next((i for i, (m, _) in enumerate(self.refs) if m == origin), None)
            if slot is not None and best_score < self.refs[slot][1]:
                self._replace_member(slot, best_pset, best_score)
                print2('Refinement %s improved its start to %g' % (tag, best_score))
                continue
            worst = max(range(len(self.refs)), key=lambda i: self.refs[i][1])
            if best_score < self.refs[worst][1]:
                self._replace_member(worst, best_pset, best_score)
                print2('Refinement %s enters the reference set at %g' % (tag, best_score))
                continue
            self.local_mins.append((best_pset, best_score))
            self.local_mins = sorted(self.local_mins, key=lambda x: x[1])[:self.popsize]
            print2('Refinement %s archived at %g' % (tag, best_score))

    def _replace_member(self, slot, pset, score):
        old = self.refs[slot][0]
        self.stuckcounter.pop(old, None)
        self.contenders.pop(old, None)
        self.draws.pop(old, None)
        self.stuckcounter[pset] = 0
        self.draws[pset] = [score]
        self.refs[slot] = (pset, score)

    def get_backup_every(self):
        """
        Overrides base method because Scatter Search runs n*(n-1) PSets per iteration.
        """
        return self.config.config['backup_every'] * self.config.config['population_size'] * \
            (self.config.config['population_size']-1) * self.config.config['smoothing']
