"""The Differential Evolution optimizer family (``de`` and ``ade`` fit types).

DifferentialEvolutionBase is the shared base; DifferentialEvolution (``de``) and
AsynchronousDifferentialEvolution (``ade``) subclass it. Extracted byte-identical
(M1 Step 4). The family makes no core.* call of its own — the run loop and the
execution seam are inherited from Algorithm.
"""


from ..base import Algorithm
from .multistart import MultiStartConfig, MultiStartOptimizer
from ...config_schema import PyBNFConfigModel
from ...pset import PSet
from ...printing import print1, print2, PybnfError
from ...registry import register_fit_type

import logging
from typing import Optional

import numpy as np
from pydantic import Field
import re
import copy


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class DEFamilyConfig(PyBNFConfigModel):
    """Config fields shared by the whole DE family, co-located with
    ``DifferentialEvolutionBase`` (ADR-0002, ADR-0006) -- exactly the keys that
    base ``__init__`` reads, and nothing more. Neither ``de`` nor ``ade`` registers
    against this base directly: each extends it with its own subclass carrying the
    shared ``n_starts`` multi-start field (``MultiStartConfig``) -- ``de`` also adds
    the island/migration fields (:class:`DifferentialEvolutionConfig`), ``ade`` adds
    nothing else (:class:`AsyncDEConfig`). Keeping the shared base itself key-minimal
    preserves the ADR-0006 "``ade`` adds no keys to the family base" seam. Values are
    byte-identical to the old ``GlobalConfig`` defaults.
    """

    mutation_rate: float = 0.5
    mutation_factor: float = 0.5
    stop_tolerance: float = 0.002
    # The DE-family convergence tolerance as an absolute range in OBJECTIVE units
    # (#561, ADR-0114): the population is converged when the spread of its finite
    # fitnesses, ``max - min``, has collapsed to within this value. ``stop_tolerance``
    # was a *ratio* (``max/min``), which only reads as convergence on a positive
    # objective bounded below by 0; ``de_tolfun`` measures the range directly, so it
    # is well-defined for a likelihood objective too. It is a range in objective units
    # where ``stop_tolerance`` was dimensionless, so it gets its own key -- and falls
    # back to ``stop_tolerance`` when unset (mirroring ``cmaes_tolfun`` -> ``cmaes_stop_tol``,
    # ADR-0106), so an existing config keeps the threshold magnitude it had.
    de_tolfun: Optional[float] = Field(default=None, ge=0.0)
    de_strategy: str = 'rand1'


class DifferentialEvolutionConfig(MultiStartConfig, DEFamilyConfig):
    """``de``-specific config: the island/migration fields only synchronous DE
    reads (``ade`` is async and ignores them), plus the shared ``n_starts``
    multi-start field (``MultiStartConfig``, #498). The ``n_starts`` key rides each
    method's own subclass, not the shared ``DEFamilyConfig`` base, so the ADR-0006
    "``ade`` adds no keys to the family base" seam stays intact -- ``ade`` gets
    ``n_starts`` through its own :class:`AsyncDEConfig` (#501). Demonstrates the
    shared-base pattern the MCMC family reuses (ADR-0006)."""

    islands: int = 1
    migrate_every: int = 20
    num_to_migrate: int = 3


class AsyncDEConfig(MultiStartConfig, DEFamilyConfig):
    """``ade``-specific config: the shared DE family fields plus the ``n_starts``
    multi-start field (``MultiStartConfig``, #498/ADR-0071), and nothing else. ``ade``
    opts into multi-start through its own subclass -- mirroring how ``de`` extends the
    shared ``DEFamilyConfig`` base (:class:`DifferentialEvolutionConfig`) -- so the
    ADR-0006 "``ade`` adds no keys to the family base" seam stays intact: ``n_starts``
    rides this subclass, not the base ``de`` and ``ade`` share. Unlike ``de``, ``ade``
    has no islands or migrations, so this adds no fields of its own -- it is exactly
    ``DEFamilyConfig`` + ``n_starts`` (#501)."""


class DifferentialEvolutionBase(Algorithm):

    def __init__(self, config):
        super().__init__(config)

        self.mutation_rate = config.config['mutation_rate']
        self.mutation_factor = config.config['mutation_factor']
        self.max_iterations = config.config['max_iterations']
        self.stop_tolerance = config.config['stop_tolerance']
        # The convergence tolerance the run actually uses (#561, ADR-0114): an
        # absolute range in objective units. Unset ``de_tolfun`` falls back to
        # ``stop_tolerance`` -- the single knob the test used before -- so an existing
        # config keeps its threshold magnitude, now read as a range rather than a ratio.
        configured_tolfun = config.config['de_tolfun']
        self.de_tolfun = (self.stop_tolerance if configured_tolfun is None
                          else float(configured_tolfun))

        self.strategy = config.config['de_strategy']
        options = ('rand1', 'rand2', 'best1', 'best2', 'all1', 'all2')
        if self.strategy not in options:
            raise PybnfError('Invalid differential evolution strategy "{}". Options are: {}'.format(self.strategy, ','.join(options)))

    def new_individual(self, individuals, base_index=None):
        """
        Create a new individual for the specified island, according to the set strategy

        :param base_index: The index to use for the new individual, or None for a random index.
        :return:
        """

        # Choose a starting parameter set (either a random one or the base_index specified)
        # and others to cross over (always random)

        if '1' in self.strategy:
            pickn = 3
        else:
            pickn = 5

        # Choose pickn random unique indices, or if base_index was given, choose base_index followed by pickn-1 unique
        # indices
        picks = self.rng.choice(len(individuals), pickn, replace=False)
        if base_index is not None:
            if base_index in picks:
                # If we accidentally picked base_index, replace it with picks[0], preserving uniqueness in our list
                iswitch = list(picks).index(base_index)
                picks[iswitch] = picks[0]
            # Now overwrite picks[0] with base_index. If we have base_index, picks[0] was an "extra pick" we only needed
            # in case we sampled base_index and had to replace it.
            picks[0] = base_index
        base = individuals[picks[0]]
        others = [individuals[p] for p in picks[1:]]

        # Iterate through parameters; decide whether to mutate or leave the same.
        new_pset_vars = []
        for p in base:
            if self.rng.random() < self.mutation_rate:
                if '1' in self.strategy:
                    update_val = self.mutation_factor * others[0].get_param(p.name).diff(others[1].get_param(p.name))
                else:
                    update_val = self.mutation_factor * others[0].get_param(p.name).diff(others[1].get_param(p.name)) +\
                                 self.mutation_factor * others[2].get_param(p.name).diff(others[3].get_param(p.name))
                new_pset_vars.append(p.add(update_val))
            else:
                new_pset_vars.append(p)

        return PSet(new_pset_vars)

    def _population_converged(self):
        """The DE-family convergence test (#561, ADR-0114), shared by ``de`` and ``ade``.

        The population is converged when the spread of its objective values --
        measured as an **absolute range** ``max - min`` over the **finite** fitnesses
        only -- has collapsed to within ``de_tolfun``.

        This replaces the original **ratio** test ``max/min < 1 + stop_tolerance``,
        which is a convergence statement only when the objective is positive and
        bounded below by 0 (a chi-square, an SSE). On a likelihood objective -- a
        negative log-likelihood, unbounded below -- an all-negative population lands
        the ratio in ``(0, 1]`` and it fires at generation 0 regardless of the spread;
        and any single ``inf``-scored failed simulation, paired with a negative
        fitness, makes the ratio ``-inf`` and defeats *every* threshold. Both failure
        modes are the negative sign, not a real convergence -- so the family was
        unrunnable on any estimated-sigma likelihood fit.

        The range form is sign-agnostic. Ignoring the non-finite entries is what makes
        a failed simulation unable to either satisfy or defeat the test; it also
        subsumes the original ``!= 0`` guard against dividing by an all-zero
        population, which ``ade`` never had. An empty finite set (every member still
        ``inf``, e.g. before the first result) is never "converged".
        """
        finite = np.asarray(self.fitnesses, dtype=float)
        finite = finite[np.isfinite(finite)]
        return bool(finite.size and (finite.max() - finite.min()) <= self.de_tolfun)

    def start_run(self):
        return NotImplementedError("start_run() not implemented in DifferentialEvolutionBase class")

    def got_result(self, res):
        return NotImplementedError("got_result() not implemented in DifferentialEvolutionBase class")


@register_fit_type('de', family='optimizer', display_name='Differential Evolution',
                   schema=DifferentialEvolutionConfig)
class DifferentialEvolution(MultiStartOptimizer, DifferentialEvolutionBase):
    """
    Implements the parallelized, island-based differential evolution algorithm
    described in Penas et al 2015.

    In some cases, I had to make my own decisions for specifics I couldn't find in the original paper. Namely:
    At each migration, a user-defined number of individuals are migrated from each island. For each individual, a
    random index is chosen; the same index for all islands. A random permutation is used to redistribute individuals
    with that index to different islands.

    Each island performs its migration individually, on the first callback when all islands are ready for that
    migration. It receives individuals from the migration iteration, regardless of what the current iteration is.
    This can sometimes lead to wasted effort.
    For example, suppose migration is set to occur at iteration 40, but island 1 has reached iteration 42 by the time
    all islands reach 40. Individual j on island 1 after iteration 42 gets replaced with individual j on island X
    after iteration 40. Some other island Y receives individual j on island 1 after iteration 40.

    """

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        The following config keys specify algorithm parameters. For move information, see config_documentation.txt
        population_size
        num_islands
        max_iterations
        mutation_rate
        mutation_factor
        migrate_every
        num_to_migrate

        """
        super().__init__(config)

        self.num_islands = config.config['islands']
        self.num_per_island = int(config.config['population_size'] / self.num_islands)
        if self.num_per_island < 3:
            self.num_per_island = 3
            if self.num_islands == 1:
                print1('Differential evolution requires a population size of at least 3. Increased the population size '
                       'to 3.')
                logger.warning('Increased population size to minimum allowed value of 3')
            else:
                print1('Island-based differential evolution requires a population size of at least 3 times '
                       'the number of islands. Increased the population size to %i.' % (3*self.num_islands))
                logger.warning('Increased population size to minimum allowed value of 3 per island')
        if config.config['population_size'] % config.config['islands'] != 0:
            logger.warning('Reduced population_size to %i to evenly distribute it over %i islands' %
                            (self.num_islands * self.num_per_island, self.num_islands))
        self.migrate_every = config.config['migrate_every']
        if self.num_islands == 1:
            self.migrate_every = np.inf
        self.num_to_migrate = config.config['num_to_migrate']

        self.island_map = dict()  # Maps each proposed PSet to its location (island, individual_i)
        self.iter_num = [0] * self.num_islands  # Count the number of completed iterations on each island
        self.waiting_count = []  # Count of the number of PSets that are pending evaluation on the current iteration of each island.
        self.individuals = []  # Nested list; individuals[i][j] gives individual j on island i.
        self.proposed_individuals = []  # Nested list of the same shape, gives individuals proposed for replacement in next generation
        self.fitnesses = []  # Nested list of same shape, gives fitness of each individual
        self.migration_ready = [0] * self.num_islands  # What migration number is each island ready for
        self.migration_done = [0] * self.num_islands  # What migration number has each island completed

        # These variables store data related to individual migrations.
        # Each one has migration number as keys. When the first island starts migration, the required entries are
        # created. When the last island completes migration, they are deleted to keep these structures small.
        self.migration_transit = dict()  # Store (PSet, fitness) tuples here that are getting migrated - one list per island
        self.migration_indices = dict()  # Which individual numbers are migrating in migration i - a single tuple for
        # each migration, used for all islands
        self.migration_perms = dict()  # How do we rearrange between islands on migration i?
        # For each migration, a list of num_to_migrate permutations of range(num_islands)

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._reset_search_state()

    def _reset_search_state(self):
        """Clear all search-specific state (populations, per-island iteration and
        migration counters) WITHOUT touching the trajectory -- so multi-start's
        :meth:`_search_start_run` can begin a fresh, independent DE run each start while
        the trajectory keeps accumulating the global best across starts (#498). Shared by
        ``reset`` (which also resets the trajectory via ``super().reset``) and by each
        multi-start."""
        self.island_map = dict()
        self.iter_num = [0] * self.num_islands
        self.waiting_count = []
        self.individuals = []
        self.proposed_individuals = []
        self.fitnesses = []
        self.migration_ready = [0] * self.num_islands
        self.migration_done = [0] * self.num_islands

        self.migration_transit = dict()
        self.migration_indices = dict()
        self.migration_perms = dict()

    def _search_start_run(self):
        # Reset every search counter first (the per-island iteration and migration
        # state start_run itself does not touch), so a multi-start restart begins a
        # genuinely fresh DE run rather than resuming at the previous start's iteration.
        self._reset_search_state()
        if self.num_islands == 1:
            print2('Running Differential Evolution with population size %i for up to %i iterations' %
                   (self.num_per_island, self.max_iterations))
        else:
            print2('Running island-based Differential Evolution with %i islands of %i individuals each, '
                   'for up to %i iterations' % (self.num_islands, self.num_per_island, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            psets = self.random_latin_hypercube_psets(self.num_islands*self.num_per_island)
            self.proposed_individuals = [psets[i * self.num_per_island: (i + 1) * self.num_per_island]
                                         for i in range(self.num_islands)]
        else:
            self.proposed_individuals = [[self.random_pset() for i in range(self.num_per_island)]
                                         for j in range(self.num_islands)]

        # ADR-0043 Phase 2: seed exactly one member of the initial population at the
        # initial_value point (a no-op unless a parameter: record declares one). One
        # member only -- the rest stay random so the islands keep their diversity.
        self.proposed_individuals[0][0] = self._seed_initial_value_pset(self.proposed_individuals[0][0])

        # Initialize the individual list to empty, will be filled with the proposed_individuals once their fitnesses
        # are computed.
        self.individuals = [[None
                             for i in range(self.num_per_island)]
                            for j in range(self.num_islands)]

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual
        self.fitnesses = [[np.inf
                           for i in range(self.num_per_island)]
                          for j in range(self.num_islands)]

        for i in range(len(self.proposed_individuals)):
            for j in range(len(self.proposed_individuals[i])):
                self.island_map[self.proposed_individuals[i][j]] = (i, j)
                if self.num_islands == 1:
                    self.proposed_individuals[i][j].name = 'gen0ind%i' % j
                else:
                    self.proposed_individuals[i][j].name = 'gen0isl%iind%i' % (i, j)

        self.waiting_count = [self.num_per_island] * self.num_islands

        return [ind for island in self.proposed_individuals for ind in island]

    def _search_got_result(self, res):
        """
        Called when a simulation run finishes

        This is not thread safe - the Scheduler must ensure only one process at a time enters
        this function.
        (or, I should rewrite this function to make it thread safe)

        :param res: Result object
        :return:
        """

        pset = res.pset
        score = res.score

        # Calculate the fitness of this individual, and replace if it is better than the previous one.
        island, j = self.island_map.pop(pset)
        fitness = score
        if fitness <= self.fitnesses[island][j]:
            self.individuals[island][j] = pset
            self.fitnesses[island][j] = fitness

        self.waiting_count[island] -= 1

        # Determine if the current iteration is over for the current island
        if self.waiting_count[island] == 0:

            self.iter_num[island] += 1
            if min(self.iter_num) == self.iter_num[island]:
                # Last island to complete this iteration
                if self.iter_num[island] % self.config.config['output_every'] == 0:
                    self.output_results()
                if self.iter_num[island] % 10 == 0:
                    print1('Completed %i of %i iterations' % (self.iter_num[island], self.max_iterations))
                else:
                    print2('Completed %i of %i iterations' % (self.iter_num[island], self.max_iterations))
                print2('Current population fitnesses:')
                for l in self.fitnesses:
                    print2(sorted(l))

            if self.iter_num[island] == self.max_iterations:
                # Submit no more jobs for this island
                # Once all islands reach this, simulation is over.
                if min(self.iter_num) == self.max_iterations:
                    return 'STOP'
                else:
                    return []

            if self.iter_num[island] % self.migrate_every == 0:
                # This island prepares for migration
                migration_num = int(self.iter_num[island] / self.migrate_every)
                if max(self.migration_ready) < migration_num:
                    # This is the first island to reach this migration.
                    # Need to set global parameters for this migration.
                    self.migration_transit[migration_num] = [list() for i in range(self.num_islands)]
                    self.migration_indices[migration_num] = self.rng.choice(self.num_per_island,
                                                                            size=self.num_to_migrate, replace=False)
                    self.migration_perms[migration_num] = [self.rng.permutation(self.num_islands)
                                                           for i in range(self.num_to_migrate)]
                    logger.debug('Island %i just set up the migration.' % island)

                # Send the required PSets to migration_transit
                for j in self.migration_indices[migration_num]:
                    self.migration_transit[migration_num][island].append((self.individuals[island][j],
                                                                          self.fitnesses[island][j]))
                # Tell other islands that this one is ready for this migration.
                self.migration_ready[island] = migration_num

            if self.migration_done[island] < min(self.migration_ready):
                # This island performs a migration
                logger.debug('Island %i is migrating!' % island)
                migration_num = self.migration_done[island] + 1

                # Fetch the appropriate new individuals from migration_transit
                for migrater_index in range(self.num_to_migrate):
                    j = self.migration_indices[migration_num][migrater_index]  # Index of the individual
                    newisland = self.migration_perms[migration_num][migrater_index][island]
                    self.individuals[island][j], self.fitnesses[island][j] = \
                        self.migration_transit[migration_num][newisland][migrater_index]

                    logger.debug('Island %i gained new individual with fitness %f' % (island, self.fitnesses[island][j]))

                self.migration_done[island] = migration_num
                if min(self.migration_done) == migration_num:
                    # This is the last island to complete this migration
                    # Delete the migration data to free space.
                    del self.migration_transit[migration_num]
                    del self.migration_perms[migration_num]
                    del self.migration_indices[migration_num]

            # Set up the next generation
            best = np.argmin(self.fitnesses[island])
            for jj in range(self.num_per_island):
                if 'best' in self.strategy:
                    new_pset = self.new_individual(self.individuals[island], best)
                elif 'all' in self.strategy:
                    new_pset = self.new_individual(self.individuals[island], jj)
                else:
                    new_pset = self.new_individual(self.individuals[island])
                # If the new pset is a duplicate of one already in the island_map, it will cause problems.
                # As a workaround, perturb it slightly.
                while new_pset in self.island_map:
                    new_pset = PSet([v.add(self.rng.uniform(-1e-6, 1e-6)) for v in new_pset])
                self.proposed_individuals[island][jj] = new_pset
                self.island_map[new_pset] = (island, jj)
                if self.num_islands == 1:
                    new_pset.name = 'gen%iind%i' % (self.iter_num[island], jj)
                else:
                    new_pset.name = 'gen%iisl%iind%i' % (self.iter_num[island], island, jj)

            self.waiting_count[island] = self.num_per_island

            if self.iter_num[island] % 20 == 0:
                logger.debug('Island %i completed %i iterations' % (island, self.iter_num[island]))
                # print(sorted(self.fitnesses[island]))

            # Convergence check: absolute range of the finite population (#561, ADR-0114).
            # Only assessed once EVERY island has completed at least one iteration
            # (``min(self.iter_num) >= 1``): until then some islands are still at their
            # initial ``inf`` sentinel, and since the test (correctly) ignores non-finite
            # fitnesses, a single finished island's collapsed subpopulation could
            # otherwise stop the whole run before the others have searched at all. The
            # old ratio test got this for free -- an unevaluated island's ``inf`` made the
            # global ``max`` infinite -- but that same coupling is what let a failed sim
            # defeat the test; the gate restores the "wait for the whole population" half
            # without the sign/failed-sim half. (For a single island this holds from the
            # first iteration on, so single-island DE is unaffected.)
            if min(self.iter_num) >= 1 and self._population_converged():
                return 'STOP'

            # Return a copy, so our internal data structure is not tampered with.
            return copy.copy(self.proposed_individuals[island])

        else:
            # Add no new jobs, wait for this generation to complete.
            return []


@register_fit_type('ade', family='optimizer', display_name='Asynchronous Differential Evolution',
                   schema=AsyncDEConfig)
class AsynchronousDifferentialEvolution(MultiStartOptimizer, DifferentialEvolutionBase):
    """
    Implements a simple asynchronous differential evolution algorithm.

    Contains no islands or migrations. Instead, each time a PSet finishes, proposes a new PSet at the same index using
    the standard DE formula and whatever the current population happens to be at the time.

    Opts into ``n_starts`` sequential-restart multi-start (#498/#501): the
    :class:`~pybnf.algorithms.optimizers.multistart.MultiStartOptimizer` mixin (before
    the family base in the MRO) runs ``n_starts`` independent searches and keeps the
    global best. ``ade`` is the async one-in-one-out case the mixin's *draining* path
    was built for -- a full population stays in flight at each inner ``STOP``, so the
    mixin drains those stragglers (already scored into the trajectory) before seeding
    the next start. ``n_starts == 1`` (the default) is byte-identical to the single-run
    behavior.
    """

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        """
        super().__init__(config)

        self.population_size = config.config['population_size']
        if self.population_size < 3:
            self.population_size = 3
            self.config.config['population_size'] = 3
            print1('Asynchronous differential evolution requires a population size of at least 3. '
                   'Increasing the population size to 3.')
            logger.warning('Increased population_size to the minimum allowed value of 3')

        self.sims_completed = 0
        self.individuals = []  # List of individuals
        self.fitnesses = []  # List of same shape, gives fitness of each individual

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._reset_search_state()

    def _reset_search_state(self):
        """Clear the search-specific state (the population, its fitnesses, and the
        completed-sim counter) WITHOUT touching the trajectory -- so multi-start's
        :meth:`_search_start_run` can begin a fresh, independent ADE run each start
        while the trajectory keeps accumulating the global best across starts (#498).
        Shared by ``reset`` (which also resets the trajectory via ``super().reset``) and
        by each multi-start."""
        self.sims_completed = 0
        self.individuals = []
        self.fitnesses = []

    def _search_start_run(self):
        # Reset the search counter/population first (a no-op on the first start), so a
        # multi-start restart begins a genuinely fresh ADE run rather than resuming at
        # the previous start's population and sims_completed count.
        self._reset_search_state()
        print2('Running Asyncrhonous Differential Evolution with population size %i for up to %i iterations' %
               (self.population_size, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            self.individuals = self.random_latin_hypercube_psets(self.population_size)
        else:
            self.individuals = [self.random_pset() for i in range(self.population_size)]

        # ADR-0043 Phase 2: seed exactly one member at the initial_value point (a no-op
        # unless a parameter: record declares one); the rest stay random for diversity.
        self.individuals[0] = self._seed_initial_value_pset(self.individuals[0])

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual.
        # The first replacement will replace with a copy of the same PSet, with the correct objective calculated.
        self.fitnesses = [np.inf for i in range(self.population_size)]

        for i in range(len(self.individuals)):
            self.individuals[i].name = 'gen0ind%i' % i

        return copy.deepcopy(self.individuals)

    def _search_got_result(self, res):
        """
        Called when a simulation run finishes

        :param res: Result object
        :return:
        """

        pset = res.pset
        fitness = res.score

        gen = int(re.search(r'(?<=gen)\d+', pset.name).group(0))
        j = int(re.search(r'(?<=ind)\d+', pset.name).group(0))

        if fitness <= self.fitnesses[j]:
            self.individuals[j] = pset
            self.fitnesses[j] = fitness

        self.sims_completed += 1

        # Do various "per iteration" stuff
        if self.sims_completed % self.population_size == 0:
            iters_complete = self.sims_completed / self.population_size
            if iters_complete % self.config.config['output_every'] == 0:
                self.output_results()
            if iters_complete % 10 == 0:
                print1('Completed %i of %i simulations' % (self.sims_completed, self.max_iterations * self.population_size))
            else:
                print2('Completed %i of %i simulations' % (self.sims_completed, self.max_iterations * self.population_size))
            print2('Current population fitnesses:')
            print2(sorted(self.fitnesses))
            if iters_complete % 20 == 0:
                logger.debug('Completed %i simulations' % self.sims_completed)
            if iters_complete >= self.max_iterations:
                return 'STOP'
            # Convergence check: absolute range of the finite population (#561, ADR-0114).
            if self._population_converged():
                return 'STOP'

        if 'best' in self.strategy:
            best = np.argmin(self.fitnesses)
            new_pset = self.new_individual(self.individuals, best)
        elif 'all' in self.strategy:
            new_pset = self.new_individual(self.individuals, j)
        else:
            new_pset = self.new_individual(self.individuals)
        new_pset.name = 'gen%iind%i' % (gen+1, j)

        return [new_pset]
