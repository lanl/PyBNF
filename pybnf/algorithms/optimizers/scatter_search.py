"""ScatterSearch optimizer (the ``ss`` fit type).

Extracted byte-identical (M1 Step 4). Subclasses Algorithm and inherits its run
loop + execution seam; it makes no core.* call of its own.
"""


from ..base import Algorithm
from .multistart import MultiStartConfig, MultiStartOptimizer
from ...pset import PSet
from ...printing import print1, print2
from ...registry import register_fit_type

import logging
import numpy as np


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

        self.pending = dict() # {pendingPSet: parentPSet}
        self.received = dict() # {parentPSet: [(donependingPSet, score)]
        self.refs = [] # (refPset, score)
        self.stuckcounter = dict()
        self.iteration = 0
        self.local_mins = [] # (Pset, score) pairs that were stuck for 5 gens, and so replaced.
        self.reserve = []

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

        parent = self.pending[ps]
        self.received[parent].append((ps, score))
        del self.pending[ps]

        if len(self.pending) == 0:
            # All of this generation done, make the next list of psets

            if None in self.received:
                # This is the initialization round, special case
                self.round_1_init()
            else:
                # 1) Replace parent with highest scoring child
                for i in range(len(self.refs)):
                    children = self.received[self.refs[i][0]]
                    # A reference whose candidate children all collided with existing
                    # pending psets receives no children — this happens once the
                    # reference set collapses on a smooth target (the combination
                    # step reproduces the parent point, which is skipped as a
                    # duplicate). Treat the childless reference as non-improving so
                    # the stuck-counter machinery eventually perturbs or retires it,
                    # rather than crashing on min() of an empty list.
                    best_child = min(children, key=lambda x: x[1]) if children else None
                    if best_child is not None and best_child[1] < self.refs[i][1]:
                        del self.stuckcounter[self.refs[i][0]]
                        self.stuckcounter[best_child[0]] = 0
                        self.refs[i] = best_child
                    else:
                        self.stuckcounter[self.refs[i][0]] += 1
                        if self.stuckcounter[self.refs[i][0]] >= self.local_min_limit:
                            del self.stuckcounter[self.refs[i][0]]
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

            # 2) Sort the refs list by quality.
            self.refs = sorted(self.refs, key=lambda x: x[1])
            logger.debug('Iteration %i' % self.iteration)
            if self.iteration % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration, self.max_iterations))
            else:
                print2('Completed iteration %i of %i' % (self.iteration, self.max_iterations))
            print2('Current scores: ' + str([x[1] for x in self.refs]))
            print2('Best archived scores: ' + str([x[1] for x in self.local_mins]))

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
            return query_psets

        else:
            return []

    def get_backup_every(self):
        """
        Overrides base method because Scatter Search runs n*(n-1) PSets per iteration.
        """
        return self.config.config['backup_every'] * self.config.config['population_size'] * \
            (self.config.config['population_size']-1) * self.config.config['smoothing']
