
"""Contains the Algorithm class and subclasses as well as support classes and functions for running simulations"""


from . import core
# Re-export the core execution primitives so the package facade
# (algorithms.Result, algorithms.run_job, ...) keeps resolving for external
# callers and tests. The run loop itself resolves the patched names as
# core.run_job / core.as_completed / core.Job (ADR-0001), so it does not rely
# on these bare names. `X as X` marks them as intentional re-exports for ruff.
from .core import (
    Result as Result,
    FailedSimulation as FailedSimulation,
    Job as Job,
    JobGroup as JobGroup,
    MultimodelJobGroup as MultimodelJobGroup,
    HybridJobGroup as HybridJobGroup,
    run_job as run_job,
    result_from_completed as result_from_completed,
)
# The Algorithm base class and its module-level helpers live in base.py. Every
# leaf class below subclasses Algorithm, so it must be in this namespace before
# those definitions execute; exp10/latin_hypercube are re-exported so leaves
# (SimplexAlgorithm) and the facade (algorithms.exp10) keep resolving them.
from .base import (
    Algorithm as Algorithm,
    latin_hypercube as latin_hypercube,
    exp10 as exp10,
)
# BayesianAlgorithm (sampler base + R-hat/ESS diagnostics) lives in
# samplers/base.py. The Bayesian leaf classes below subclass it, so it must be
# in this namespace before their definitions execute; the facade re-export also
# keeps algorithms.BayesianAlgorithm resolving for tests.
from .samplers.base import BayesianAlgorithm as BayesianAlgorithm
from ..pset import PSet
from ..pset import OutOfBoundsException
from ..printing import print0, print1, print2, PybnfError
from ..objective import ConstraintCounter

import logging
import numpy as np
import os
import re
import shutil
import copy
import traceback
from scipy import stats
# Facade re-export: the run loop (now in base.py) catches CancelledError, and
# tests construct algorithms.CancelledError to drive that path.
from concurrent.futures import CancelledError as CancelledError



logger = logging.getLogger(__name__)


class ParticleSwarm(Algorithm):
    """
    Implements particle swarm optimization.

    The implementation roughly follows Moraes et al 2015, although is reorganized to better suit PyBNF's format.
    Note the global convergence criterion discussed in that paper is not used (would require too long a
    computation), and instead uses ????

    """

    def __init__(self, config):

        # Former params that are now part of the config
        # variable_list, num_particles, max_evals, cognitive=1.5, social=1.5, w0=1.,
        # wf=0.1, nmax=30, n_stop=np.inf, absolute_tol=0., relative_tol=0.)
        """
        Initial configuration of particle swarm optimizer
        :param conf_dict: The fitting configuration
        :type conf_dict: Configuration

        The config should contain the following definitions:

        population_size - Number of particles in the swarm
        max_iterations - Maximum number of iterations. More precisely, the max number of simulations run is this times
        the population size.
        cognitive - Acceleration toward the particle's own best
        social - Acceleration toward the global best
        particle_weight - Inertia weight of the particle (default 1)

        The following config parameters relate to the complicated method presented is Moraes et al for adjusting the
        inertia weight as you go. These are optional, and this feature will be disabled (by setting
        particle_weight_final = particle_weight) if these are not included.
        It remains to be seen whether this method is at all useful for our applications.

        particle_weight_final -  Inertia weight at the end of the simulation
        adaptive_n_max - Controls how quickly we approach wf - After nmax "unproductive" iterations, we are halfway from
        w0 to wf
        adaptive_n_stop - nd the entire run if we have had this many "unproductive" iterations (should be more than
        adaptive_n_max)
        adaptive_abs_tol - Tolerance for determining if an iteration was "unproductive". A run is unproductive if the
        change in global_best is less than absolute_tol + relative_tol * global_best
        adaptive_rel_tol - Tolerance 2 for determining if an iteration was "unproductive" (see above)

        """

        super(ParticleSwarm, self).__init__(config)

        # This default value gets special treatment because if missing, it should take the value of particle_weight,
        # disabling the adaptive weight change entirely.
        if 'particle_weight_final' not in self.config.config:
            self.config.config['particle_weight_final'] = self.config.config['particle_weight']

        # Save config parameters
        self.c1 = self.config.config['cognitive']
        self.c2 = self.config.config['social']
        self.max_evals = self.config.config['population_size'] * self.config.config['max_iterations']
        self.output_every = self.config.config['population_size'] * self.config.config['output_every']

        self.num_particles = self.config.config['population_size']
        # Todo: Nice error message if a required key is missing

        self.w0 = self.config.config['particle_weight']

        self.wf = self.config.config['particle_weight_final']
        self.nmax = self.config.config['adaptive_n_max']
        self.n_stop = self.config.config['adaptive_n_stop']
        self.absolute_tol = self.config.config['adaptive_abs_tol']
        self.relative_tol = self.config.config['adaptive_rel_tol']

        self.nv = 0  # Counter that controls the current weight. Counts number of "unproductive" iterations.
        self.num_evals = 0  # Counter for the total number of results received

        # Initialize storage for the swarm data
        self.swarm = []  # List of lists of the form [PSet, velocity]. Velocity is stored as a dict with the same keys
        # as PSet
        self.pset_map = dict()  # Maps each PSet to it s particle number, for easy lookup.
        # One independent [PSet, objective] slot per particle. A list-multiply
        # ([[None, inf]] * n) would alias one inner list across every particle --
        # harmless today (got_result reassigns the whole slot) but a foot-gun if
        # anyone ever mutates a slot in place.
        self.bests = [[None, np.inf] for _ in range(self.num_particles)]  # The best result for each particle
        self.global_best = [None, np.inf]  # The best result for the whole swarm
        self.last_best = np.inf

    def reset(self, bootstrap=None):
        super(ParticleSwarm, self).reset(bootstrap)
        self.nv = 0
        self.num_evals = 0
        self.swarm = []
        self.pset_map = dict()
        self.bests = [[None, np.inf] for _ in range(self.num_particles)]
        self.global_best = [None, np.inf]
        self.last_best = np.inf

    def start_run(self):
        """
        Start the run by initializing n particles at random positions and velocities
        :return:
        """
        print2('Running Particle Swarm Optimization with %i particles for %i total simulations' %
               (self.num_particles, self.max_evals))

        if self.config.config['initialization'] == 'lh':
            new_params_list = self.random_latin_hypercube_psets(self.num_particles)
        else:
            new_params_list = [self.random_pset() for i in range(self.num_particles)]

        for i in range(len(new_params_list)):
            p = new_params_list[i]
            p.name = 'iter0p%i' % i

            # As suggested by Engelbrecht 2012, set all initial velocities to 0
            new_velocity = dict({v.name: 0. for v in self.variables})

            self.swarm.append([p, new_velocity])
            self.pset_map[p] = len(self.swarm)-1  # Index of the newly added PSet.

        return [particle[0] for particle in self.swarm]

    def got_result(self, res):
        """
        Updates particle velocity and position after a simulation completes.

        :param res: Result object containing the run PSet and the resulting Data.
        :return:
        """

        paramset = res.pset
        score = res.score

        self.num_evals += 1

        if self.num_evals % self.num_particles == 0:
            if (self.num_evals / self.num_particles) % 10 == 0:
                print1('Completed %i of %i simulations' % (self.num_evals, self.max_evals))
            else:
                print2('Completed %i of %i simulations' % (self.num_evals, self.max_evals))
            print2('Current best score: %f' % self.global_best[1])
            # End of one "pseudoflight", check if it was productive.
            if (self.last_best != np.inf and
                    np.abs(self.last_best - self.global_best[1]) <
                    self.absolute_tol + self.relative_tol * self.last_best):
                self.nv += 1
            self.last_best = self.global_best[1]

            # Check stop criterion
            if self.config.config['v_stop'] > 0:
                max_speed = max([abs(v) for p in self.swarm for v in p[1].values()])
                if max_speed < self.config.config['v_stop']:
                    logger.info('Stopping particle swarm because the max speed is %s' % max_speed)
                    return 'STOP'

        if self.num_evals % self.output_every == 0:
            self.output_results()

        p = self.pset_map.pop(paramset)  # Particle number

        # Update best scores if needed.
        if score <= self.bests[p][1]:
            self.bests[p] = [paramset, score]
            if score <= self.global_best[1]:
                self.global_best = [paramset, score]

        # Update own position and velocity
        # The order matters - updating velocity first seems to make the best use of our current info.
        w = self.w0 + (self.wf - self.w0) * self.nv / (self.nv + self.nmax)
        self.swarm[p][1] = \
            {v.name:
                w * self.swarm[p][1][v.name] +
                self.c1 * np.random.random() * self.bests[p][0].get_param(v.name).diff(self.swarm[p][0].get_param(v.name)) +
                self.c2 * np.random.random() * self.global_best[0].get_param(v.name).diff(self.swarm[p][0].get_param(v.name))
            for v in self.variables}

        # Manually check to determine if reflection occurred (i.e. attempted assigning of variable outside its bounds)
        # If so, update based on reflection protocol and set velocity to 0
        new_vars = []
        for v in self.swarm[p][0]:
            new_vars.append(v.add(self.swarm[p][1][v.name]))
            if v.log_space:
                new_val = 10.**(np.log10(v.value) + self.swarm[p][1][v.name])
            else:
                new_val = v.value + self.swarm[p][1][v.name]
            if new_val < v.lower_bound or v.upper_bound < new_val:
                self.swarm[p][1][v.name] = 0.0

        new_pset = PSet(new_vars)
        self.swarm[p][0] = new_pset

        # This will cause a crash if new_pset happens to be the same as an already running pset in pset_map.
        # This could come up in practice if all parameters have hit a box constraint.
        # As a simple workaround, perturb the parameters slightly
        while new_pset in self.pset_map:
            new_pset = PSet([v.add_rand(-1e-6, 1e-6) for v in self.swarm[p][0]])

        self.pset_map[new_pset] = p

        # Set the new name: the old pset name is iter##p##
        # Extract the iter number
        iternum = int(re.search('iter([0-9]+)', paramset.name).groups()[0])
        new_pset.name = 'iter%ip%i' % (iternum+1, p)

        # Check for stopping criteria
        if self.num_evals >= self.max_evals or self.nv >= self.n_stop:
            return 'STOP'

        return [new_pset]

    def add_iterations(self, n):
        self.max_evals += n * self.config.config['population_size']


class DifferentialEvolutionBase(Algorithm):

    def __init__(self, config):
        super(DifferentialEvolutionBase, self).__init__(config)

        self.mutation_rate = config.config['mutation_rate']
        self.mutation_factor = config.config['mutation_factor']
        self.max_iterations = config.config['max_iterations']
        self.stop_tolerance = config.config['stop_tolerance']

        self.strategy = config.config['de_strategy']
        options = ('rand1', 'rand2', 'best1', 'best2', 'all1', 'all2')
        if self.strategy not in options:
            raise PybnfError('Invalid differential evolution strategy "%s". Options are: %s' %
                             (self.strategy, ','.join(options)))

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
        picks = np.random.choice(len(individuals), pickn, replace=False)
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
            if np.random.random() < self.mutation_rate:
                if '1' in self.strategy:
                    update_val = self.mutation_factor * others[0].get_param(p.name).diff(others[1].get_param(p.name))
                else:
                    update_val = self.mutation_factor * others[0].get_param(p.name).diff(others[1].get_param(p.name)) +\
                                 self.mutation_factor * others[2].get_param(p.name).diff(others[3].get_param(p.name))
                new_pset_vars.append(p.add(update_val))
            else:
                new_pset_vars.append(p)

        return PSet(new_pset_vars)

    def start_run(self):
        return NotImplementedError("start_run() not implemented in DifferentialEvolutionBase class")

    def got_result(self, res):
        return NotImplementedError("got_result() not implemented in DifferentialEvolutionBase class")


class DifferentialEvolution(DifferentialEvolutionBase):
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
        super(DifferentialEvolution, self).__init__(config)

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
        super(DifferentialEvolution, self).reset(bootstrap)
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

    def start_run(self):
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

    def got_result(self, res):
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
                    self.migration_indices[migration_num] = np.random.choice(range(self.num_per_island),
                                                                             size=self.num_to_migrate, replace=False)
                    self.migration_perms[migration_num] = [np.random.permutation(self.num_islands)
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
                    new_pset = PSet([v.add(np.random.uniform(-1e-6, 1e-6)) for v in new_pset])
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

            # Convergence check
            if (np.min(self.fitnesses) != 0) and (np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance):
                return 'STOP'

            # Return a copy, so our internal data structure is not tampered with.
            return copy.copy(self.proposed_individuals[island])

        else:
            # Add no new jobs, wait for this generation to complete.
            return []


class AsynchronousDifferentialEvolution(DifferentialEvolutionBase):
    """
    Implements a simple asynchronous differential evolution algorithm.

    Contains no islands or migrations. Instead, each time a PSet finishes, proposes a new PSet at the same index using
    the standard DE formula and whatever the current population happens to be at the time.

    """

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        """
        super(AsynchronousDifferentialEvolution, self).__init__(config)

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
        super(AsynchronousDifferentialEvolution, self).reset(bootstrap)
        self.sims_completed = 0
        self.individuals = []
        self.fitnesses = []

    def start_run(self):
        print2('Running Asyncrhonous Differential Evolution with population size %i for up to %i iterations' %
               (self.population_size, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            self.individuals = self.random_latin_hypercube_psets(self.population_size)
        else:
            self.individuals = [self.random_pset() for i in range(self.population_size)]

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual.
        # The first replacement will replace with a copy of the same PSet, with the correct objective calculated.
        self.fitnesses = [np.inf for i in range(self.population_size)]

        for i in range(len(self.individuals)):
            self.individuals[i].name = 'gen0ind%i' % i

        return copy.deepcopy(self.individuals)

    def got_result(self, res):
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
            # Convergence check
            if np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance:
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


class ScatterSearch(Algorithm):
    """
    Implements ScatterSearch as described in the introduction of Penas et al 2017 (but not the fancy parallelized
    version from that paper).
    Uses the individual combination method described in Egea et al 2009

    """

    def __init__(self, config):  # variables, popsize, maxiters, saveevery):

        super(ScatterSearch, self).__init__(config)

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
        super(ScatterSearch, self).reset(bootstrap)
        self.pending = dict()
        self.received = dict()
        self.refs = []
        self.stuckcounter = dict()
        self.iteration = 0
        self.local_mins = []
        self.reserve = []

    def start_run(self):
        print2('Running Scatter Search with population size %i (%i simulations per iteration) for %i iterations' %
               (self.popsize, self.popsize * (self.popsize - 1), self.max_iterations))
        # Generate big number = 10 * variable_count (or user's chosen init_size) initial individuals.
        if self.config.config['initialization'] == 'lh':
            psets = self.random_latin_hypercube_psets(self.init_size)
        else:
            psets = [self.random_pset() for i in range(self.init_size)]
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
        randindices = np.random.choice(range(topcount, len(start_psets)), randcount, replace=False)
        for i in randindices:
            self.refs.append(start_psets[i])
        self.stuckcounter = {r[0]: 0 for r in self.refs}

    def got_result(self, res):
        """
        Called when a simulation run finishes

        :param res:
        :type res Result
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
                        # d = (self.refs[hi][0][v] - self.refs[pi][0][v]) / 2.
                        d = self.refs[hi][0].get_param(v.name).diff(self.refs[pi][0].get_param(v.name))
                        alpha = np.sign(hi-pi)
                        beta = (abs(hi-pi) - 1) / (self.popsize - 2)
                        # c1 = self.refs[pi][0][v] - d*(1 + alpha*beta)
                        # c2 = self.refs[pi][0][v] + d*(1 - alpha*beta)
                        # newval = np.random.uniform(c1, c2)
                        # newdict[v] = max(min(newval, var[2]), var[1])
                        new_vars.append(self.refs[pi][0].get_param(v.name).add_rand(-d*(1 + alpha*beta), d*(1 - alpha * beta)))
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


class DreamAlgorithm(BayesianAlgorithm):
    """
    Implements a variant of the DREAM algorithm as described in Vrugt (2016) Environmental Modelling
    and Software.

    Adapts Bayesian MCMC to use methods from differential evolution for accelerated convergence and
    more efficient sampling of parameter space
    """

    def __init__(self, config):
        super(DreamAlgorithm, self).__init__(config)
        self.ncr = [(1+x)/self.config.config['crossover_number'] for x in range(self.config.config['crossover_number'])]
        self.ncr_count = len(self.ncr)
        self.g_prob = self.config.config['gamma_prob']
        self.adaptive_step_size = config.config['adaptive_step_size']
        self.acceptances = [0]*self.num_parallel
        self.acceptance_rates = [0.0]*self.num_parallel

        # CR adaptation state
        self.cr_probs = np.ones(self.ncr_count) / self.ncr_count
        self.cr_total_distance = np.zeros(self.ncr_count)
        self.cr_usage_count = np.zeros(self.ncr_count)
        self.cr_adapt_end = self.burn_in // 2
        self.cr_frozen = False

        # Per-generation tracking for CR adaptation
        self.gen_cr_indices = [None] * self.num_parallel
        self.gen_x_old = [None] * self.num_parallel
        self.gen_x_std = np.ones(self.n_dim)

        # ZS archive: external archive of past states for proposal generation
        m0 = config.config['archive_size']
        self.archive_m0 = m0 if m0 is not None else 10 * self.n_dim
        self.archive_thin_rate = config.config['archive_thin_rate']
        self.archive = []  # list of PSet objects

        # Snooker update
        self.snooker_prob = config.config['snooker_prob']
        self.gen_log_snooker_correction = [0.0] * self.num_parallel

        # Multiple chain pairs
        self.delta = config.config['delta']

        # Outlier detection method
        self.outlier_method = config.config['outlier_method']

    def start_run(self, setup_samples=True):
        first_psets = super().start_run(setup_samples)
        # Initialize the ZS archive with m0 random draws from the prior
        self.archive = [self.random_pset() for _ in range(self.archive_m0)]
        logger.info('Initialized ZS archive with %d entries (d=%d)' % (self.archive_m0, self.n_dim))
        return first_psets

    def calculate_snooker_pset(self, idx):
        """
        Snooker update proposal (ter Braak & Vrugt, 2008).
        Projects archive points onto the line through the current state and a reference archive point,
        then jumps along that axis.

        Returns (PSet or None, log_correction) where log_correction is the log of the Hastings
        correction factor (d-1)*log(||Xp - Zc|| / ||X - Zc||).
        """
        x0 = self.current_pset[idx]
        x0_vec = self._param_vec(x0)

        # Draw three distinct archive indices: c (reference), a, b (for projection difference)
        sel = np.random.choice(len(self.archive), 3, replace=False)
        zc_vec = self._param_vec(self.archive[sel[0]])
        za_vec = self._param_vec(self.archive[sel[1]])
        zb_vec = self._param_vec(self.archive[sel[2]])

        # Snooker axis: line through x0 and zc
        axis = x0_vec - zc_vec
        axis_norm_sq = np.dot(axis, axis)
        if axis_norm_sq < 1e-20:
            return None, 0.0

        # Project za and zb onto the snooker axis
        za_proj = zc_vec + axis * (np.dot(za_vec - zc_vec, axis) / axis_norm_sq)
        zb_proj = zc_vec + axis * (np.dot(zb_vec - zc_vec, axis) / axis_norm_sq)

        # Jump vector along the axis
        diff_proj = za_proj - zb_proj

        # Gamma for snooker: U(1.2, 2.2) per Vrugt (2016)
        gamma_s = np.random.uniform(1.2, 2.2)

        # Small perturbations
        zeta_vec = np.random.normal(0, self.config.config['zeta'], size=self.n_dim)
        lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

        xp_vec = x0_vec + zeta_vec + (1.0 + lamb) * gamma_s * diff_proj

        # Build the proposed PSet
        new_vars = []
        for i, v in enumerate(self.variables):
            try:
                if v.log_space:
                    new_var = v.set_value(10**xp_vec[i], reflect=False)
                else:
                    new_var = v.set_value(xp_vec[i], reflect=False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                return None, 0.0

        # Hastings correction: (||Xp - Zc|| / ||X - Zc||)^(d-1).
        # dist_x0_zc = sqrt(axis_norm_sq) is already guaranteed nonzero by the
        # axis_norm_sq < 1e-20 check above, so no divide-by-zero guard is needed here.
        dist_xp_zc = np.linalg.norm(xp_vec - zc_vec)
        dist_x0_zc = np.linalg.norm(x0_vec - zc_vec)
        log_correction = (self.n_dim - 1) * np.log(dist_xp_zc / dist_x0_zc)

        return PSet(new_vars), log_correction

    def _detect_outliers_iqr(self, mean_ln_p):
        """IQR outlier detection: chains below Q25 - 2*IQR are outliers."""
        Q75 = np.percentile(mean_ln_p, 75)
        Q25 = np.percentile(mean_ln_p, 25)
        iqr = Q75 - Q25
        return np.where(mean_ln_p < Q25 - 2.0 * iqr)[0]

    def _detect_outliers_grubbs(self, mean_ln_p):
        """Grubbs test for a single minimum outlier at significance alpha=0.01."""
        N = len(mean_ln_p)
        if N < 3:
            return np.array([], dtype=int)
        mu = np.mean(mean_ln_p)
        sd = np.std(mean_ln_p, ddof=1)
        if sd < 1e-20:
            return np.array([], dtype=int)
        G = (mu - np.min(mean_ln_p)) / sd
        alpha = 0.01
        t_crit_sq = stats.t.ppf(alpha / (2 * N), N - 2) ** 2
        T_c = (N - 1) / np.sqrt(N) * np.sqrt(t_crit_sq / (N - 2 + t_crit_sq))
        if G > T_c:
            return np.array([np.argmin(mean_ln_p)])
        return np.array([], dtype=int)

    def detect_and_reset_outliers(self):
        """
        Detect outlier chains using the configured method on mean log-posteriors
        (last 50% of history). Reset outlier chains to copies of randomly selected
        non-outlier chains.

        Methods: 'iqr' (interquartile range), 'grubbs' (Grubbs test at alpha=0.01).
        """
        min_len = min(len(h) for h in self.ln_posterior_history)
        start = min_len // 2
        if min_len - start < 5:
            return

        mean_ln_p = np.array([
            np.mean(self.ln_posterior_history[j][start:min_len]) for j in range(self.num_parallel)
        ])

        if self.outlier_method == 'grubbs':
            outlier_indices = self._detect_outliers_grubbs(mean_ln_p)
        else:
            outlier_indices = self._detect_outliers_iqr(mean_ln_p)

        if len(outlier_indices) == 0:
            return

        good_indices = [i for i in range(self.num_parallel) if i not in outlier_indices]
        if len(good_indices) == 0:
            return

        for out_idx in outlier_indices:
            donor_idx = np.random.choice(good_indices)
            logger.warning('Outlier chain %d reset to chain %d at iteration %d (method=%s)'
                           % (out_idx, donor_idx, self.iteration[out_idx], self.outlier_method))
            self.current_pset[out_idx] = copy.deepcopy(self.current_pset[donor_idx])
            self.ln_current_P[out_idx] = self.ln_current_P[donor_idx]
            self.ln_posterior_history[out_idx][start:min_len] = self.ln_posterior_history[donor_idx][start:min_len]
            self.chain_history[out_idx][start:min_len] = self.chain_history[donor_idx][start:min_len]

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: PSet that was run in this simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """

        pset = res.pset
        score = res.score
        self.total_evaluations += 1

        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        # Calculate posterior of finished job
        lnprior = self.ln_prior(pset)
        lnlikelihood = -score

        lnposterior = lnprior + lnlikelihood

        # Metropolis-Hastings criterion (includes snooker Hastings correction when applicable)
        ln_p_accept = min(0., lnposterior - self.ln_current_P[index]
                          + self.gen_log_snooker_correction[index])
        if np.log(np.random.uniform()) < ln_p_accept:  # accept update based on MH criterion
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.acceptances[index] += 1
            self.evaluate_constraints(res.simdata, index)

        # Store chain history (after accept/reject, so it reflects the kept state)
        self.chain_history[index].append(self._param_vec(self.current_pset[index]))
        self.ln_posterior_history[index].append(self.ln_current_P[index])

        # CR adaptation: compute standardized distance traveled
        if not self.cr_frozen and self.gen_cr_indices[index] is not None:
            x_new_vec = self._param_vec(self.current_pset[index])
            if self.gen_x_old[index] is not None:
                diff_vec = x_new_vec - self.gen_x_old[index]
                sd_dist = np.sum((diff_vec / np.maximum(self.gen_x_std, 1e-10)) ** 2)
                self.cr_total_distance[self.gen_cr_indices[index]] += sd_dist
                self.cr_usage_count[self.gen_cr_indices[index]] += 1

        # Record that this individual is complete
        self.wait_for_sync[index] = True
        self.iteration[index] += 1
        self.acceptance_rates[index] = self.acceptances[index] / self.iteration[index]

        # Update histograms and trajectories if necessary
        if self.iteration[index] % self.sample_every == 0 and self.iteration[index] > self.burn_in:
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        if (self.iteration[index] % (self.sample_every * self.output_hist_every) == 0
            and self.iteration[index] > self.burn_in):
            self.update_histograms('_%i' % self.iteration[index])

        # Wait for entire generation to finish
        # Loop handles the case where all proposals are out of bounds: advance
        # the generation counter and try again instead of returning an empty
        # list (which would exhaust the job pool and silently end the run).
        while np.all(self.wait_for_sync):

            self.wait_for_sync = [False] * self.num_parallel

            if min(self.iteration) >= self.max_iterations:
                self.update_histograms('_final')
                self.report_constraint_satisfaction('_final')
                return 'STOP'

            if self.iteration[index] % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                print2('Acceptance rates: %s\n' % str(self.acceptance_rates))
            else:
                print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
            # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
            if self.iteration[index] % self.diagnostics_every == 0:
                max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                if self.check_convergence(self.iteration[index], max_rhat):
                    return 'STOP'
            logger.debug('Completed %i iterations' % self.iteration[index])
            print2('Current -Ln Posteriors: %s' % str(self.ln_current_P))

            # Outlier detection (every 10 iterations, only during burn-in)
            if self.iteration[index] % 10 == 0 and self.iteration[index] <= self.burn_in:
                self.detect_and_reset_outliers()

            # CR adaptation: update probabilities
            if (self.iteration[index] % 10 == 0
                    and self.iteration[index] <= self.cr_adapt_end
                    and not self.cr_frozen):
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_dist = self.cr_total_distance / np.maximum(self.cr_usage_count, 1)
                if np.sum(mean_dist) > 0:
                    self.cr_probs = mean_dist / np.sum(mean_dist)
                    logger.debug('Updated CR probabilities: %s' % str(self.cr_probs))
            elif self.iteration[index] > self.cr_adapt_end and not self.cr_frozen:
                self.cr_frozen = True
                logger.debug('CR probabilities frozen at iteration %d: %s'
                            % (self.iteration[index], str(self.cr_probs)))

            # Grow the ZS archive: every K generations, append current chain states
            if self.iteration[index] % self.archive_thin_rate == 0:
                for i in range(self.num_parallel):
                    self.archive.append(copy.deepcopy(self.current_pset[i]))
                logger.debug('Archive grown to %d entries at iteration %d'
                            % (len(self.archive), self.iteration[index]))

            # Save old states and compute population std for CR adaptation
            for i in range(self.num_parallel):
                self.gen_x_old[i] = self._param_vec(self.current_pset[i])
            all_vecs = np.array(self.gen_x_old)
            self.gen_x_std = np.std(all_vecs, axis=0)

            next_gen = []
            for i, p in enumerate(self.current_pset):
                if np.random.uniform() < self.snooker_prob:
                    # Snooker update
                    new_pset, log_corr = self.calculate_snooker_pset(i)
                    self.gen_log_snooker_correction[i] = log_corr
                    self.gen_cr_indices[i] = None  # no CR for snooker
                else:
                    # Parallel direction update
                    new_pset, cr_idx = self.calculate_new_pset(i)
                    self.gen_log_snooker_correction[i] = 0.0
                    self.gen_cr_indices[i] = cr_idx
                if new_pset:
                    new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                    next_gen.append(new_pset)
                else:
                    # Out-of-bounds proposal: treat as a Metropolis rejection.
                    # Record the current state in chain history (chain stays in place)
                    # so that diagnostics (R-hat, ESS) correctly reflect the non-movement.
                    logger.debug('Proposed PSet for chain %d is out of bounds. Treating as rejection.' % i)
                    self.chain_history[i].append(self._param_vec(self.current_pset[i]))
                    self.ln_posterior_history[i].append(self.ln_current_P[i])
                    self.wait_for_sync[i] = True
                    self.iteration[i] += 1
                    self.acceptance_rates[i] = self.acceptances[i] / self.iteration[i]
                    if self.iteration[i] % self.sample_every == 0 and self.iteration[i] > self.burn_in:
                        self.sample_pset(self.current_pset[i], self.ln_current_P[i], i)

            if not next_gen:
                logger.warning('All %d proposals were out of bounds at iteration %d. '
                               'Advancing to next generation.'
                               % (self.num_parallel, min(self.iteration)))
                continue

            return next_gen

        return []

    def calculate_new_pset(self, idx):
        """
        Uses differential evolution-like update to calculate new PSet.
        Returns (PSet, cr_idx) or (None, cr_idx) if the proposal is out of bounds.

        :param idx: Index of PSet to update
        :return: tuple of (PSet or None, int)
        """

        x0 = self.current_pset[idx]

        # Draw 2*delta donor states from the ZS archive (without replacement)
        sel = np.random.choice(len(self.archive), 2 * self.delta, replace=False)

        # Sample crossover value and mask
        cr_idx = np.random.choice(self.ncr_count, p=self.cr_probs)
        cr = self.ncr[cr_idx]
        while True:
            ds = np.random.uniform(size=self.n_dim) <= cr  # sample parameter subspace
            if np.any(ds):
                break

        # Gamma selection: mode jump (gamma=1) or adaptive/fixed step size
        if np.random.uniform() < self.g_prob:
            gamma = 1
            ds[:] = True  # mode jump updates all dimensions
        else:
            d_prime = int(np.sum(ds))
            if self.adaptive_step_size:
                gamma = 2.38 / np.sqrt(2.0 * self.delta * d_prime)
            else:
                gamma = self.step_size

        new_vars = []
        for i, d in enumerate(ds):
            k = self.variables[i]
            if d:
                # Sum of delta difference vectors: sum_{j=1}^{delta} (Z_a_j - Z_b_j)
                total_diff = 0.0
                for j in range(self.delta):
                    total_diff += self.archive[sel[j]].get_param(k.name).diff(
                        self.archive[sel[self.delta + j]].get_param(k.name))
            else:
                total_diff = 0.0
            zeta = np.random.normal(0, self.config.config['zeta'])
            lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

            # Differential evolution calculation (while satisfying detailed balance)
            try:
                # Do not reflect the parameter (need to reject if outside bounds)
                new_var = x0.get_param(k.name).add(zeta + (1. + lamb) * gamma * total_diff, False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                logger.debug("Variable %s is outside of bounds")
                return None, cr_idx

        return PSet(new_vars), cr_idx


class PDreamAlgorithm(DreamAlgorithm):
    """
    P-DREAM: Preconditioned DREAM.

    Extends DREAM(ZS) by computing DE proposals in a covariance-whitened parameter space.
    An online covariance estimate C is learned from the chain history (as in Adaptive Metropolis).
    Donors are transformed to whitened coordinates z = L_inv @ x before computing DE differences,
    and crossover is applied in whitened space where dimensions are decorrelated.

    The proposal remains symmetric (DE differences from an external archive), so standard
    Metropolis-Hastings acceptance is valid without additional Hastings correction.

    After a configurable adaptation period, the covariance is updated every generation from
    the pooled chain history.  Before adaptation begins, the sampler behaves identically to
    DREAM(ZS).
    """

    def __init__(self, config):
        super(PDreamAlgorithm, self).__init__(config)
        pa = config.config['precondition_adapt']
        self.precondition_adapt = pa if pa is not None else self.burn_in // 2
        self._cov_L = None       # Cholesky factor of the covariance estimate
        self._cov_L_inv = None   # Inverse of Cholesky factor (whitening matrix)
        self._preconditioned = False

    def _update_covariance(self):
        """
        Estimate the covariance from pooled chain history and compute Cholesky factors.
        Discards the first 50% of each chain as warmup (matching the convention used by
        _get_split_chains for R-hat/ESS) so the early burn-in transient does not inflate
        the preconditioner. Pooling across chains is intentional: P-DREAM's global
        preconditioner wants a proposal scale large enough for archive-based mode hopping.
        """
        # Pool the post-warmup half of each chain history into one matrix
        all_samples = []
        for chain in self.chain_history:
            if len(chain) > 1:
                all_samples.extend(chain[len(chain) // 2:])
        if len(all_samples) < 2 * self.n_dim:
            return  # Not enough samples yet

        X = np.array(all_samples)
        n = X.shape[0]
        d = X.shape[1]

        # Sample covariance with Haario-style regularization: C = Cov(X) + eps*I
        cov = np.cov(X, rowvar=False)
        eps = 1e-6 * np.trace(cov) / d if np.trace(cov) > 0 else 1e-6
        cov += eps * np.eye(d)

        try:
            L = np.linalg.cholesky(cov)
            self._cov_L = L
            self._cov_L_inv = np.linalg.solve(L, np.eye(d))
            if not self._preconditioned:
                self._preconditioned = True
                logger.info('P-DREAM: preconditioning activated at iteration %d '
                            'with %d pooled samples (d=%d)'
                            % (min(self.iteration), n, d))
            else:
                logger.debug('P-DREAM: covariance updated with %d samples' % n)
        except np.linalg.LinAlgError:
            logger.warning('P-DREAM: Cholesky decomposition failed, '
                           'skipping covariance update')

    def _whiten(self, x_vec):
        """Transform a parameter vector to whitened space: z = L_inv @ x."""
        return self._cov_L_inv @ x_vec

    def _unwhiten_diff(self, dz_vec):
        """Transform a difference vector from whitened space back: dx = L @ dz."""
        return self._cov_L @ dz_vec

    def got_result(self, res):
        """Override to update covariance estimate after each generation sync."""
        result = super(PDreamAlgorithm, self).got_result(res)

        # After a full generation sync with new proposals, update the covariance
        if isinstance(result, list) and len(result) > 0:
            if min(self.iteration) >= self.precondition_adapt:
                self._update_covariance()

        return result

    def calculate_new_pset(self, idx):
        """
        DE proposal in whitened space.

        When preconditioning is active:
        1. Transform current state and archive donors to z = L_inv @ x
        2. Compute DE difference in z-space
        3. Apply crossover in z-space (dimensions are decorrelated)
        4. Scale and add perturbation in z-space
        5. Convert the total jump back: dx = L @ dz_total
        6. Propose x_new = x_current + dx

        Before preconditioning activates, falls back to standard DREAM proposals.
        """
        if not self._preconditioned:
            return super(PDreamAlgorithm, self).calculate_new_pset(idx)

        x0 = self.current_pset[idx]
        x0_vec = self._param_vec(x0)

        # Whiten the current state
        z0 = self._whiten(x0_vec)

        # Draw 2*delta donor states from the ZS archive (without replacement)
        sel = np.random.choice(len(self.archive), 2 * self.delta, replace=False)

        # Whiten the donor states
        z_donors = []
        for s in sel:
            z_donors.append(self._whiten(self._param_vec(self.archive[s])))

        # Sample crossover value and mask (in whitened space where dims are independent)
        cr_idx = np.random.choice(self.ncr_count, p=self.cr_probs)
        cr = self.ncr[cr_idx]
        while True:
            ds = np.random.uniform(size=self.n_dim) <= cr
            if np.any(ds):
                break

        # Gamma selection
        if np.random.uniform() < self.g_prob:
            gamma = 1
            ds[:] = True  # mode jump: update all dimensions
        else:
            d_prime = int(np.sum(ds))
            if self.adaptive_step_size:
                gamma = 2.38 / np.sqrt(2.0 * self.delta * d_prime)
            else:
                gamma = self.step_size

        # Compute DE difference in whitened space
        dz_total = np.zeros(self.n_dim)
        for j in range(self.delta):
            dz_total += z_donors[j] - z_donors[self.delta + j]

        # Apply crossover mask in whitened space
        dz_masked = np.where(ds, dz_total, 0.0)

        # Small perturbations in whitened space
        zeta_z = np.random.normal(0, self.config.config['zeta'], size=self.n_dim)
        lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

        # Total jump in whitened space, then transform back to original space
        dz_jump = zeta_z + (1.0 + lamb) * gamma * dz_masked
        dx_jump = self._unwhiten_diff(dz_jump)

        # Build proposed PSet in original space
        xp_vec = x0_vec + dx_jump
        new_vars = []
        for i, v in enumerate(self.variables):
            try:
                if v.log_space:
                    new_var = v.set_value(10**xp_vec[i], reflect=False)
                else:
                    new_var = v.set_value(xp_vec[i], reflect=False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                logger.debug("Variable %s is outside of bounds")
                return None, cr_idx

        return PSet(new_vars), cr_idx


class BasicBayesMCMCAlgorithm(BayesianAlgorithm):

    """
    Implements a Bayesian Markov chain Monte Carlo simulation.
    This is essentially a non-parallel algorithm, but here, we run n instances in parallel, and pool all results.
    This will give you a best fit (which is maybe not great), but more importantly, generates an extra result file
    that gives the probability distribution of each variable.
    This distribution depends on the prior, which is specified according to the variable initialization rules.
    With sa=True, this instead acts as a simulated annealing algorithm with n indepdendent chains.
    """

    def __init__(self, config, sa=False):  # expdata, objective, priorfile, gamma=0.1):
        super(BasicBayesMCMCAlgorithm, self).__init__(config)
        self.sa = sa

        if sa:
            self.cooling = config.config['cooling']
            self.beta_max = config.config['beta_max']

        self.exchange_every = config.config['exchange_every']
        self.pt = self.exchange_every != np.inf
        self.reps_per_beta = self.config.config['reps_per_beta']
        self.betas_per_group = self.num_parallel // self.reps_per_beta  # Number of unique betas considered (in PT)

        # The temperature of each replicate
        # For MCMC, probably n copies of the same number, unless the user set it up strangely
        # For SA, starts all the same (unless set up strangely), and independently decrease during the run
        # For PT, contains reps_per_beta copies of the same ascending sequence of betas, e.g.
        # [0.6, 0.8, 1., 0.6, 0.8, 1.]. Indices congruent to -1 mod (population_size/reps_per_beta) have the max beta
        # (probably 1), and only these replicas are sampled.
        self.betas = config.config['beta_list']

        self.wait_for_sync = [False] * self.num_parallel

        self.prior = None
        self.load_priors()

        self.attempts = 0
        self.accepted = 0
        self.exchange_attempts = 0
        self.exchange_accepted = 0

        self.staged = []  # Used only when resuming a run and adding iterations
        self.converged = False  # Set by try_to_choose_new_pset on R-hat convergence

    def reset(self, bootstrap=None):
        super(BasicBayesMCMCAlgorithm, self).reset(bootstrap)

        self.current_pset = None
        self.ln_current_P = None
        self.iteration = [0] * self.num_parallel

        self.wait_for_sync = [False] * self.num_parallel
        self.samples_file = None

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.
        :return: list of PSets
        """
        if self.sa:
            print2('Running simulated annealing on %i independent replicates in parallel, for %i iterations each or '
                   'until 1/T reaches %s' % (self.num_parallel, self.max_iterations, self.beta_max))
        else:
            if not self.pt:
                print2('Running Markov Chain Monte Carlo on %i independent replicates in parallel, for %i iterations each.'
                       % (self.num_parallel, self.max_iterations))
            else:
                print2('Running parallel tempering on %i replicates for %i iterations, with replica exchanges performed '
                       'every %i iterations' % (self.num_parallel, self.max_iterations, self.exchange_every))

            print2('Statistical samples will be recorded every %i iterations, after an initial %i-iteration burn-in period'
                   % (self.sample_every, self.burn_in))
            if self.max_iterations <= self.burn_in:
                raise PybnfError(
                    'max_iterations (%i) must be greater than burn_in (%i), '
                    'otherwise no samples will be collected.'
                    % (self.max_iterations, self.burn_in))

        setup_samples = not self.sa
        return super(BasicBayesMCMCAlgorithm, self).start_run(setup_samples=setup_samples)

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data
        :param res: PSet that was run in this simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """

        pset = res.pset
        score = res.score
        self.total_evaluations += 1

        # Figure out which parallel run this is from based on the .name field.
        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        # Calculate the acceptance probability
        lnprior = self.ln_prior(pset) # Need something clever for box constraints
        lnlikelihood = -score

        # Because the P's are so small to start, we express posterior, p_accept, and current_P in ln space
        lnposterior = lnprior + lnlikelihood

        ln_p_accept = min(0., lnposterior - self.ln_current_P[index])

        # Decide whether to accept move.
        self.attempts += 1
        if np.random.rand() < np.exp(ln_p_accept*self.betas[index]) or np.isnan(self.ln_current_P[index]):
            # Accept the move, so update our current PSet and P
            self.accepted += 1
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.evaluate_constraints(res.simdata, index)
            # For simulated annealing, reduce the temperature if this was an unfavorable move.
            if self.sa and ln_p_accept < 0.:
                self.betas[index] += self.cooling
                if self.betas[index] >= self.beta_max:
                    print2('Finished replicate %i because beta_max was reached.' % index)
                    logger.info('Finished replicate %i because beta_max was reached.' % index)
                    if min(self.betas) >= self.beta_max:
                        logger.info('All annealing replicates have reached the maximum beta value')
                        return 'STOP'
                    else:
                        return []

        # Store chain history (after accept/reject, so it reflects the kept state)
        if self.current_pset[index] is not None:
            self.chain_history[index].append(self._param_vec(self.current_pset[index]))
            self.ln_posterior_history[index].append(self.ln_current_P[index])

        # Record the current PSet (clarification: what if failed? Sample old again?)
        # Using either the newly accepted PSet or the old PSet, propose the next PSet.
        proposed_pset = self.try_to_choose_new_pset(index)

        if proposed_pset is None:
            if self.converged:
                print0('Overall move accept rate: %f' % (self.accepted/self.attempts))
                return 'STOP'
            elif np.all(self.wait_for_sync):
                # Do the replica exchange, then propose n new psets so all chains resume
                self.wait_for_sync = [False] * self.num_parallel
                return self.replica_exchange()
            elif min(self.iteration) >= self.max_iterations:
                print0('Overall move accept rate: %f' % (self.accepted/self.attempts))
                if not self.sa:
                    self.update_histograms('_final')
                    self.report_constraint_satisfaction('_final')
                return 'STOP'
            else:
                return []

        proposed_pset.name = 'iter%irun%i' % (self.iteration[index], index)
        # Note self.staged is empty unless we just resumed a run with added iterations and need to restart chains.
        if len(self.staged) != 0:
            toreturn = [proposed_pset] + self.staged
            self.staged = []
            return toreturn
        return [proposed_pset]

    def try_to_choose_new_pset(self, index):
        """
        Helper function
        Advances the iteration number, and tries to choose a new parameter set for chain index i
        If that fails (e.g. due to a box constraint), keeps advancing iteration number and trying again.
        If it hits an iteration where it has to stop and wait (a replica exchange iteration or the end), returns None
        Otherwise returns the new PSet.
        :param index:
        :return:
        """
        proposed_pset = None
        # This part is a loop in case a box constraint makes a move automatically rejected.
        loop_count = 0
        while proposed_pset is None:
            loop_count += 1
            if loop_count == 20:
                logger.warning('Instance %i spent 20 iterations at the same point' % index)
                print1('One of your samples is stuck at the same point for 20+ iterations because it keeps '
                       'hitting box constraints. Consider using looser box constraints or a smaller '
                       'step_size.')
            if loop_count == 1000:
                logger.warning('Instance %i terminated after 1000 iterations at the same point' % index)
                print1('Instance %i was terminated after it spent 1000 iterations stuck at the same point '
                       'because it kept hitting box constraints. Consider using looser box constraints or a '
                       'smaller step_size.' % index)
                self.iteration[index] = self.max_iterations

            self.iteration[index] += 1
            # Check if it's time to do various things
            if not self.sa:
                if self.iteration[index] > self.burn_in and self.iteration[index] % self.sample_every == 0 \
                        and self.should_sample(index):
                    self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
                if (self.iteration[index] > self.burn_in
                   and self.iteration[index] % (self.output_hist_every * self.sample_every) == 0
                   and self.iteration[index] == min(self.iteration)):
                    self.update_histograms('_%i' % self.iteration[index])

            if self.iteration[index] == min(self.iteration):
                if self.iteration[index] % self.config.config['output_every'] == 0:
                    self.output_results()
                if self.iteration[index] % 10 == 0:
                    print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                    print2('Current move accept rate: %f' % (self.accepted/self.attempts))
                    if self.exchange_attempts > 0:
                        print2('Current replica exchange rate: %f' % (self.exchange_accepted / self.exchange_attempts))
                else:
                    print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
                if not self.sa and self.iteration[index] % self.diagnostics_every == 0:
                    max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                    if self.check_convergence(self.iteration[index], max_rhat):
                        self.converged = True
                        return None
                logger.debug('Completed %i iterations' % self.iteration[index])
                logger.debug('Current move accept rate: %f' % (self.accepted/self.attempts))
                if self.exchange_attempts > 0:
                    logger.debug('Current replica exchange rate: %f' % (self.exchange_accepted / self.exchange_attempts))
                if self.sa:
                    logger.debug('Current betas: ' + str(self.betas))
                print2('Current -Ln Likelihoods: ' + str(self.ln_current_P))
            if self.iteration[index] >= self.max_iterations:
                logger.info('Finished replicate number %i' % index)
                print2('Finished replicate number %i' % index)
                return None
            if self.iteration[index] % self.exchange_every == 0:
                # Need to wait for the rest of the chains to catch up to do replica exchange
                self.wait_for_sync[index] = True
                return None
            proposed_pset = self.choose_new_pset(self.current_pset[index])
        return proposed_pset

    def should_sample(self, index):
        """
        Checks whether this replica index is one that gets sampled.
        For mcmc, always True. For pt, must be a replica at the max beta
        """
        return (index + 1) % self.betas_per_group == 0 if self.pt else True

    def choose_new_pset(self, oldpset):
        """
        Helper function to perturb the old PSet, generating a new proposed PSet.
        The step is a fixed-magnitude (step_size) random-walk move; any component
        that would leave the box is reflected back inside (see FreeParameter._reflect).
        :param oldpset: The PSet to be changed
        :type oldpset: PSet
        :return: the new PSet
        """

        delta_vector = {k: np.random.normal() for k in oldpset.keys()}
        delta_vector_magnitude = np.sqrt(sum([x ** 2 for x in delta_vector.values()]))
        delta_vector_normalized = {k: self.step_size * delta_vector[k] / delta_vector_magnitude for k in oldpset.keys()}
        new_vars = []
        for v in oldpset:
            # Box constraints are handled by reflection: FreeParameter.add defaults to
            # reflect=True, so a step that would leave the box is folded back inside
            # rather than rejected. The fold is symmetric, so the plain Metropolis
            # ratio in got_result still targets the correct bound-restricted posterior.
            new_var = v.add(delta_vector_normalized[v.name])
            new_vars.append(new_var)

        return PSet(new_vars)

    def replica_exchange(self):
        """
        Performs replica exchange for parallel tempering.
        Then proposes n new parameter sets to resume all chains after the exchange.
        :return: List of n PSets to run
        """
        logger.debug('Performing replica exchange on iteration %i' % self.iteration[0])
        # Who exchanges with whom is a little complicated. Each replica tries one exchange with a replica at the next
        # beta. But if we have multiple reps per beta, then the exchanges aren't necessarily within the same group of
        # reps. We use this random permutation to determine which groups exchange.
        for i in range(self.betas_per_group - 1):
            permutation = np.random.permutation(range(self.reps_per_beta))
            for group in range(self.reps_per_beta):
                # Determine the 2 indices we're exchanging, ind_hi and ind_lo
                ind_hi = self.betas_per_group * group + i
                other_group = permutation[group]
                ind_lo = self.betas_per_group * other_group + i + 1
                # Consider exchanging index ind_hi (higher T) with ind_lo (lower T)
                ln_p_exchange = min(0., -(self.betas[ind_lo]-self.betas[ind_hi]) * (self.ln_current_P[ind_lo]-self.ln_current_P[ind_hi]))
                # Scratch work: Should there be a - sign in front? You want to always accept if moving the better answer
                # to the lower temperature. ind_lo has lower T so higher beta, so the first term is positive. The second
                # term is positive if ind_lo is better. But you want a positive final answer when ind_hi, currently at
                # higher T, is better. So you need a - sign.
                self.exchange_attempts += 1
                if np.random.random() < np.exp(ln_p_exchange):
                    # Do the exchange
                    logger.debug('Exchanging individuals %i and %i' % (ind_hi, ind_lo))
                    self.exchange_accepted += 1
                    hold_pset = self.current_pset[ind_hi]
                    hold_p = self.ln_current_P[ind_hi]
                    self.current_pset[ind_hi] = self.current_pset[ind_lo]
                    self.ln_current_P[ind_hi] = self.ln_current_P[ind_lo]
                    self.current_pset[ind_lo] = hold_pset
                    self.ln_current_P[ind_lo] = hold_p
        # Propose new psets - it's more complicated because of going out of box, and other counters.
        proposed = []
        for j in range(self.num_parallel):
            proposed_pset = self.try_to_choose_new_pset(j)
            if proposed_pset is None:
                if np.all(self.wait_for_sync):
                    logger.error('Aborting because no changes were made between one replica exchange and the next.')
                    print0("I seem to have gone from one replica exchange to the next replica exchange without "
                           "proposing a single valid move. Something is probably wrong for this to happen, so I'm "
                           "going to stop.")
                    return 'STOP'
                elif min(self.iteration) >= self.max_iterations:
                    return 'STOP'
            else:
                # Iteration number got off by 1 because try_to_choose_new_pset() was called twice: once a while ago
                # when it reached the exchange point and returned None, and a second time just now.
                # Need to correct for that here.
                self.iteration[j] -= 1
                proposed_pset.name = 'iter%irun%i' % (self.iteration[j], j)
                proposed.append(proposed_pset)
        return proposed

    def cleanup(self):
        """Called when quitting due to error.
        Save the histograms in addition to the usual algorithm cleanup"""
        super().cleanup()
        self.update_histograms('_end')
        self.report_constraint_satisfaction('_end')

    def add_iterations(self, n):
        oldmax = self.max_iterations
        self.max_iterations += n
        # Any chains that already completed need to be restarted with a new proposed parameter set
        for index in range(self.num_parallel):
            if self.iteration[index] >= oldmax:
                ps = self.try_to_choose_new_pset(index)
                if ps:
                    # Add to a list of new psets to run that will be submitted when the first result comes back.
                    ps.name = 'iter%irun%i' % (self.iteration[index], index)
                    logger.debug('Added PSet %s to BayesAlgorithm.staged to resume a chain' % (ps.name))
                    self.staged.append(ps)

class Adaptive_MCMC(BayesianAlgorithm):
    def __init__(self, config):  # expdata, objective, priorfile, gamma=0.1):
        super(Adaptive_MCMC, self).__init__(config)
        # set the params decleared in the configuaration file
        if self.config.config['normalization']:
            self.norm = self.config.config['normalization']
        else:
            self.norm = None
           
        self.time = self.config.config['time_length'] 
       
        self.adaptive = self.config.config['adaptive']
        # The iteration number that the adaptive starts at
        self.valid_range = self.burn_in + self.adaptive
        # The length of the ouput arrays and the number of iterations before they are written out
        self.arr_length = 1
        # set recorders
        self.acceptances = 0
        self.acceptance_rates = 0
        self.attempts = 0
        self.factor = [0] * self.num_parallel
        self.staged = []
        self.alpha = [0] * self.num_parallel
        # start lists
        self.current_param_set = [0] * self.num_parallel
        self.current_param_set_diff = [0] * self.num_parallel
        self.scores = np.zeros((self.num_parallel, self.arr_length))
        # set arrays for features and graphs
        self.parameter_index = np.zeros((self.num_parallel, self.arr_length, len(self.variables)))
        self.mu = np.zeros((self.num_parallel, 1, len(self.variables))) 
        # warm start features
        
        os.makedirs(self.config.config['output_dir'] + '/adaptive_files', exist_ok=True)
        os.makedirs(self.config.config['output_dir'] + '/Results/A_MCMC/Runs', exist_ok=True)
        os.makedirs(self.config.config['output_dir'] + '/Results/Histograms/', exist_ok=True)
        
        if self.config.config['output_trajectory']:
            self.output_columns = []
            for i in self.config.config['output_trajectory']:
                new = i.replace(',', '')
                self.output_columns.append(new)
            self.output_run_current = {}
            self.output_run_all = {}
            for i in self.output_columns:
                for k in self.time.keys():
                    if '_Cum' in i:
                        self.output_run_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                    else:
                        self.output_run_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                     
        
        if self.config.config['output_noise_trajectory']:
            self.output_noise_columns = []
            for i in self.config.config['output_noise_trajectory']:
                new = i.replace(',', '')
                self.output_noise_columns.append(new)
            self.output_run_noise_current = {}
            self.output_run_noise_all = {}
            for i in self.output_noise_columns:
                for k in self.time.keys():
                    if '_Cum' in i:
                        self.output_run_noise_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_noise_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                    else:
                        self.output_run_noise_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_noise_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
        if self.config.config['continue_run'] == 1:
            adaptive_dir = self.config.config['output_dir'] + '/adaptive_files'
            required = ['diff.txt', 'MLE_params.txt', 'diffMatrix.txt']
            missing = [f for f in required if not os.path.exists(os.path.join(adaptive_dir, f))]
            if missing:
                raise PybnfError(
                    'continue_run = 1 requires adaptive files from a completed prior run, '
                    'but the following files are missing from %s: %s. '
                    'Run the model first without continue_run, or check that output_dir '
                    'points to a previous run\'s output.' % (adaptive_dir, ', '.join(missing)))
            self.diff = [self.step_size] * self.num_parallel
            self.diff_best = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diff.txt')
            self.diffMatrix = np.zeros((self.num_parallel, len(self.variables), len(self.variables))) 
            self.diffMatrix_log = np.zeros((self.num_parallel, len(self.variables), len(self.variables)))
            if self.adaptive != 1:
                self.mle_best = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/MLE_params.txt')
                self.diffMatrix_best = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diffMatrix.txt')
                for i in range(self.num_parallel):  
                    self.diffMatrix[i] = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diffMatrix.txt')
                    self.diff[i] = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diff.txt')
                    
        else:
            self.mle_best = np.zeros((self.arr_length, len(self.variables)))
            self.diff = [self.step_size] * self.num_parallel
            self.diff_best = self.step_size
            self.diffMatrix = np.zeros((self.num_parallel, len(self.variables), len(self.variables)))  
           

        # make sure that the adaptive and burn in iterations are less then the max iterations
        if self.adaptive + self.burn_in >= self.max_iterations - 1:
            raise PybnfError('The max iterations must be at least 2 more then the sum of the adaptive and burn-in iterations.')    
    ''' Used for resuming runs and adding iterations'''
    def reset(self, bootstrap=None):
        super(Adaptive_MCMC, self).reset(bootstrap)

        self.current_pset = None
        self.ln_current_P = None
        self.iteration = [0] * self.num_parallel

        self.wait_for_sync = [False] * self.num_parallel
        self.samples_file = None
    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.
        :return: list of PSets
        """

        print2(
                'Running Adaptive Markov Chain Monte Carlo on %i independent replicates in parallel, for %i iterations each.'
                % (self.num_parallel, self.max_iterations))


        return super(Adaptive_MCMC, self).start_run(setup_samples=True)

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data
        :param res: PSet that was run in this simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """
        pset = res.pset
        score = res.score
        self.total_evaluations += 1

        # Figure out which parallel run this is from based on the .name field.
        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        lnprior = self.ln_prior(pset)
        lnlikelihood = -score
        lnposterior = lnlikelihood + lnprior

        self.accept = False
        self.attempts += 1
        # Decide whether to accept move
        if lnposterior > self.ln_current_P[index] or np.isnan(self.ln_current_P[index]):
            self.accept = True
            self.alpha[index] = 1
        else:
            self.alpha[index] = np.exp((lnposterior-self.ln_current_P[index]))
            if np.random.random() < self.alpha[index]:
                self.accept = True
        # if accept then update the lists
        if self.accept == True:
            self.current_pset[index] = pset
            self.acceptances += 1
            self.evaluate_constraints(res.simdata, index)
            self.list_trajactory = []      
            self.cp = []
            for i in self.current_pset[index]:
                self.cp.append(i.value)
            self.current_param_set[index] = self.cp 
            # Keep track of the overall best chain and its adaptive features
            if lnposterior > max(self.ln_current_P):
                self.mle_best = self.current_param_set[index]
                self.diffMatrix_best = self.diffMatrix[index]
                self.diff_best = self.diff[index]
            if self.iteration[index] == 0:
                self.mle_best = self.current_param_set[index]
                self.diffMatrix_best = np.eye(len(self.variables))
                self.diff_best = self.diff[index]

            # The order of varible reassignment is very important here    
            self.ln_current_P[index] = lnposterior    
            if self.config.config['parallelize_models'] != 1:
                res.out = res.simdata
            if isinstance(res.out, FailedSimulation):
                pass
            else:
                if self.config.config['output_trajectory']:
                    for l in self.output_columns:     
                        for i in res.out:
                            for j in res.out[i]:
                                if l in res.out[i][j].cols:
                                    if self.norm:
                                        res.out[i][j].normalize(self.norm)
                                    column = res.out[i][j].cols[l]
                                    self.list_trajactory = []
                                    for z in res.out[i][j].data:
                                        self.list_trajactory.append(z.data[column])      
                                    if '_Cum' in l:
                                        getFirstValue = np.concatenate((self.list_trajactory[0],np.diff(self.list_trajactory)))
                                        self.output_run_current[j+l][index]= getFirstValue
                                    else:
                                        self.output_run_current[j+l][index]= self.list_trajactory
                                    self.list_trajactory = []
                if self.config.config['output_noise_trajectory']:
                    for la in self.output_noise_columns:     
                        for ib in res.out:
                            for js in res.out[ib]:
                                if la in res.out[ib][js].cols:
                                    if self.norm:
                                        res.out[ib][js].normalize(self.norm)
                                    column = res.out[ib][js].cols[la]
                                    self.list_trajactory = []
                                    for z in res.out[ib][js].data:
                                        self.list_trajactory.append(z.data[column])      
                                    if '_Cum' in la:
                                        getFirstValue = np.concatenate(([self.list_trajactory[0]],np.diff(self.list_trajactory)))
                                        self.output_run_noise_current[js+la][index]= getFirstValue
                                    else:
                                        self.output_run_noise_current[js+la][index]= self.list_trajactory
                                    self.list_trajactory = []
                                              
        # After the burn in period start to record the accepted params for the adaptive feature.
        if self.iteration[index] >= self.burn_in:
            self.parameter_index[index][self.factor[index]] = self.current_param_set[index]
        
        # record the trajactorys for the graphs
        if self.iteration[index] >= self.valid_range and self.iteration[index] % self.config.config['sample_every'] == 0:
            # if the objective function is negbin then add the negbin noise to the traj output else record accepted sim vals as is
            if (self.config.config['objfunc'] == 'neg_bin' and self.config.config['output_noise_trajectory']) or (self.config.config['objfunc'] == 'neg_bin_dynamic' and self.config.config['output_noise_trajectory']):
                for l in self.output_noise_columns:     
                    for i in self.output_run_noise_current.keys():
                        if l in i:
                            self.output_run_noise_all[i][index][self.factor[index]] =  self.generateBinomialNoise(self.output_run_noise_current[i][index][0], self.current_pset[index])
            if self.config.config['output_trajectory']:
                for l in self.output_columns:
                    for i in self.output_run_current.keys():
                        if l in i:
                            self.output_run_all[i][index][self.factor[index]] = self.output_run_current[i][index][0]

        # Record that this individual is complete
        self.scores[index][self.factor[index]] = self.ln_current_P[index]

        # Track chain history for convergence diagnostics (R-hat, ESS)
        if self.current_pset[index] is not None:
            self.chain_history[index].append(self._param_vec(self.current_pset[index]))
            self.ln_posterior_history[index].append(self.ln_current_P[index])

        self.iteration[index] += 1

        # Standard BayesianAlgorithm sampling
        if (self.iteration[index] > self.burn_in
                and self.iteration[index] % self.sample_every == 0):
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        if (self.iteration[index] > self.burn_in
                and self.iteration[index] % (self.sample_every * self.output_hist_every) == 0):
            self.update_histograms('_%i' % self.iteration[index])

        self.wait_for_sync[index] = True
        # Wait for entire generation to finish
        if np.all(self.wait_for_sync):
            self.acceptance_rates = self.acceptances / self.attempts
            #self.wait_for_sync = [False] * self.num_parallel
            # Increase or reset the factor number and see if it's time to write things out
            for i in range(self.num_parallel):
                # self.factor[i] +=1
                # if self.factor[i] == self.arr_length:
                #     self.factor[i] = 0
                if self.iteration[i] % self.arr_length == 0 :
                    self.write_out_scores(i)
                if self.iteration[i] >= (self.burn_in -1) and self.iteration[i] <= (self.burn_in + self.adaptive):
                    if self.iteration[i] % self.arr_length == 0:
                        self.write_out_params(i)
                if self.iteration[i] > (self.burn_in + self.adaptive) and self.iteration[i] % self.config.config['sample_every'] == 0:
                    if self.iteration[i] % self.arr_length == 0:
                        self.write_out_params(i)
                if self.config.config['output_trajectory']:
                    if self.iteration[i] >= self.valid_range and self.iteration[i] % self.config.config['sample_every'] == 0:
                        if self.iteration[i] % self.arr_length == 0:
                            self.write_out_trajactorys(i)
                if self.config.config['output_noise_trajectory']:
                    if self.iteration[i] >= self.valid_range and self.iteration[i] % self.config.config['sample_every'] == 0:
                        if self.iteration[i] % self.arr_length == 0:
                            self.write_out_trajactorys_noise(i)

            # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
            if self.iteration[index] % self.diagnostics_every == 0:
                max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                if self.check_convergence(self.iteration[index], max_rhat):
                    self.combine_chains_params()
                    self.combine_chains_traj()
                    self.samples_file = self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt'
                    return 'STOP'

            # Set here because I don't want these commands to exacute more then once.
            if min(self.iteration) >= self.max_iterations:
                # Save the current postion of the MCMC run
                self.diff_best = [self.diff_best]
                np.savetxt(self.config.config['output_dir'] + '/adaptive_files/MLE_params.txt', self.mle_best)
                np.savetxt(self.config.config['output_dir'] + '/adaptive_files/diffMatrix.txt', self.diffMatrix_best)
                np.savetxt(self.config.config['output_dir'] + '/adaptive_files/diff.txt', self.diff_best)
                self.combine_chains_params()
                self.combine_chains_traj()
                self.samples_file = self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt'
                self.report_constraint_satisfaction('_final')
                return 'STOP'
            # Check if it's time to report stuff
            if self.iteration[index] % 10 == 0:
                print2('Acceptance rates: %s\n' % str(self.acceptance_rates))
                print2('Current -Ln Posteriors: %s' % str(self.ln_current_P))
            print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))

            
            # Propose next Pset
            next_generation = []
            for i, p in enumerate(self.current_pset):
                new_pset = self.pick_new_pset(i)
                if new_pset:
                    new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                    next_generation.append(new_pset)
                self.wait_for_sync[i] = False        
            return next_generation
        return []

    def generateBinomialNoise(self, timeseries, pset):
        # Generate the binomial noise for the results
        self.output = np.copy(timeseries)
        self.pset = pset
        if self.config.config['objfunc'] == 'neg_bin_dynamic':
            for p in self.pset:
                if p.name == 'r__FREE':
                    self.r = p.value
        else:
            self.r = self.config.config['neg_bin_r']
        for i in range(len(timeseries)):
            self.prob = np.clip( self.r/(self.r+timeseries[i]), 1e-10, 1-1e-10)
            self.output[i] = stats.nbinom.rvs(n=self.r, p=self.prob, size=1)

        return self.output

    def write_out_scores(self, idx):
        # Write out the scores. Need more practical method
        self.write_out_score = self.scores[idx]
        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/scores_' + str(idx) + '.txt', 'a') as f:
            np.savetxt(f, self.write_out_score)

    def write_out_params(self, idx):
        # WRite out the param. Need more practical method
        if self.iteration[idx] == self.burn_in - 1:
            self.write_out_p = self.parameter_index[idx][~(self.parameter_index[idx]==0).all(1)]
            varibles = []
            for v in self.variables:
                varibles.append(v.name)
            varNames = '\t'.join(varibles)
            with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt', 'a') as f:
                f.write(varNames+'\n')
        else:
            self.write_out_p = self.parameter_index[idx][~(self.parameter_index[idx]==0).all(1)]
            with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt', 'a') as f:
                np.savetxt(f, self.write_out_p)

    def write_out_trajactorys(self, idx):
        # write out trajectories need more practical method
        for l in self.output_columns:     
            for i in self.output_run_current.keys():
                if l in i:
                    self.write_out_t = self.output_run_all[i][idx][~(self.output_run_all[i][idx]==0).all(1)]
                    if len(self.write_out_t) != 0:
                        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_' + i + '_chain_' + str(idx) + '.txt', 'a') as f:
                            np.savetxt(f, self.write_out_t)
    def write_out_trajactorys_noise(self, idx):
        # Basically this IO on every iter is to expensice timewise
        for l in self.output_noise_columns:     
            for i in self.output_run_noise_current.keys():
                if l in i: 
                    self.write_out_t = self.output_run_noise_all[i][idx][~(self.output_run_noise_all[i][idx]==0).all(1)]
                    if len(self.write_out_t) != 0:
                        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_noise_' + i + '_chain_' + str(idx) + '.txt', 'a') as f:
                            np.savetxt(f, self.write_out_t)
    def combine_chains_params(self):
        #combine the chains for the final output file
        # if self.num_parallel != 1:
        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt', 'w') as f:
            varsnNames = []
            for v in self.variables:
                varsnNames.append(v.name)
            varsNames = '\t'.join(varsnNames)    
            f.write(varsNames+'\n')
            for i in range(self.num_parallel):
                file_append = np.loadtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(i) + '.txt', skiprows=1)
                file_append = file_append[self.adaptive:]
                np.savetxt(f, file_append)   
        shutil.copyfile(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt', self.config.config['output_dir'] + '/adaptive_files/combined_params.txt')      
    def combine_chains_traj(self):
        # combine the trains for the file output file
        if self.num_parallel != 1:
            if self.config.config['output_trajectory']:
                for j in range(self.num_parallel):
                    for l in self.output_columns:     
                        for i in self.output_run_current.keys():
                            if l in i:
                                with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_traj_' + i + '.txt', 'a') as f:
                                    file_append = np.loadtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_' + i + '_chain_' + str(j) + '.txt')
                                    np.savetxt(f, file_append)
            if self.config.config['output_noise_trajectory']:
                for j in range(self.num_parallel):
                    for l in self.output_noise_columns:     
                        for i in self.output_run_noise_current.keys():
                            if l in i:
                                with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_traj_noise_' + i + '.txt', 'a') as f:
                                    file_append = np.loadtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_noise_' + i + '_chain_' + str(j) + '.txt')
                                    np.savetxt(f, file_append)                                     

    def pick_new_pset(self, idx):
        """
        :param idx: Index of PSet to update
        :return: A mew
        """
                   
        params = []
        for var in self.variables:
            if 'log' in var.type:
                # Work in base-10 log, consistent with how the proposal is applied
                # (FreeParameter.add -> 10**(log10(value)+summand)) and with the
                # rest of the codebase (loguniform_var dist, prior_logpdf,
                # _param_vec R-hat history, FreeParameter.diff all use log10).
                params.append(np.log10(self.current_pset[idx].get_param(var.name).value))
            else:
                params.append(self.current_pset[idx].get_param(var.name).value)
        len_params = len(params) 
        self.stablizingCov = self.config.config['stablizingCov']*np.eye(len_params)
        if self.iteration[idx] >= self.burn_in + self.adaptive:
            if self.iteration[idx] == self.burn_in + self.adaptive:
                self.parameter_index_file_input = np.genfromtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt', names = True)
                for v in self.variables:
                    if 'log' in v.type:
                        self.parameter_index_file_input[v.name] = np.log10(self.parameter_index_file_input[v.name])
                self.parameter_index_file = self.parameter_index_file_input.view((np.float64, len(self.parameter_index_file_input.dtype.names)))
                self.mu[idx] = np.reshape(np.mean(self.parameter_index_file,axis=0), [1, len_params])  # compute the mean parameters along the past chain 
                self.diffMatrix[idx] = np.matmul(self.parameter_index_file.T, self.parameter_index_file)/(self.iteration[idx] - self.burn_in)-np.matmul(self.mu[idx].T, self.mu[idx])+self.stablizingCov
                self.diff[idx] = 2.38**2/len_params
            # Weight each new sample by 1/(samples folded so far + 1). The seed
            # (above) is built from the `adaptive` post-burn-in history rows
            # (divisor iteration - burn_in == adaptive), so the running count is
            # (iteration - burn_in), NOT the global iteration. Using the global
            # counter under-weights new samples by ~(1+iteration)/(1+adaptive)
            # at the seeding step, freezing the proposal near the seed (AM-2).
            self.mu[idx] = self.mu[idx] + (1./(1+self.iteration[idx]-self.burn_in))*(params - self.mu[idx])
            self.diffVector = np.reshape(params - self.mu[idx], [1, len_params])
            self.diffMatrix[idx] = self.diffMatrix[idx] + (1./(1 + self.iteration[idx]-self.burn_in))*(np.matmul(self.diffVector.T, self.diffVector)+self.stablizingCov-self.diffMatrix[idx])
            self.diff[idx] = np.exp( np.log(self.diff[idx]) + (1./(1 + self.iteration[idx]- self.adaptive - self.burn_in))*(self.alpha[idx]-0.234))
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = np.random.multivariate_normal(mean=np.zeros((len_params,)), cov=self.diffMatrix[idx])
                delta_vector_add = {k: self.diff[idx]*delta_vector[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        if num < 10000:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                        else:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True) 
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass       
        elif self.config.config['continue_run'] == 1:
            if self.config.config['calculate_covari']:
                start_end = self.config.config['calculate_covari']
                start = int(start_end[0])
                end = int(start_end[1])
                if self.iteration[idx] == 1:
                    self.parameter_index_file_input = np.genfromtxt(self.config.config['output_dir'] + '/adaptive_files/combined_params.txt', names = True)
                    for v in self.variables:
                        if 'log' in v.type:
                            self.parameter_index_file_input[v.name] = np.log10(self.parameter_index_file_input[v.name])
                    self.parameter_index_file_range = self.parameter_index_file_input.view((np.float64, len(self.parameter_index_file_input.dtype.names)))
                    self.parameter_index_file = self.parameter_index_file_range[start:end]
                    self.mu[idx] = np.reshape(np.mean(self.parameter_index_file,axis=0), [1, len_params])  # compute the mean parameters along the past chain 
                    self.diffMatrix[idx] = (np.matmul(self.parameter_index_file.T, self.parameter_index_file)-np.matmul(self.mu[idx].T, self.mu[idx]))/(len(self.parameter_index_file_input)*0.75)
                    self.diff[idx] = self.config.config['step_size']
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = np.random.multivariate_normal(mean=np.zeros((len_params,)), cov=self.diffMatrix[idx])
                delta_vector_add = {k: self.diff[idx] * delta_vector[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        if num < 10000:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                        else:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True)      
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass       
        else:
            diffMatrix = np.eye(len_params)
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = np.random.multivariate_normal(mean=np.zeros((len_params,)), cov=diffMatrix)
                delta_vector_add = {k: self.step_size * delta_vector[i] for i,k in enumerate(oldpset.keys())}
                #delta_vector_multiply_log = {k: self.step_size*delta_vector_log[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        if num < 10000:
                            if 'log' in k.type:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                            else:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                        else:
                            if 'log' in k.type:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True) 
                            else:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True)
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass        
    
    def update_histograms(self, file_ext):
            pass   
class SimplexAlgorithm(Algorithm):
    """
    Implements a parallelized version of the Simplex local search algorithm, as described in Lee and Wiswall 2007,
    Computational Economics

    """

    _is_simplex = True

    def __init__(self, config, refine=False):
        super(SimplexAlgorithm, self).__init__(config)
        if 'simplex_start_point' not in self.config.config:
            # We need to set up the initial point ourselfs
            self._parse_start_point()
        if 'simplex_max_iterations' in self.config.config:
            self.max_iterations = self.config.config['simplex_max_iterations']
        else:
            self.max_iterations = self.config.config['max_iterations']
        self.start_point = self.config.config['simplex_start_point']
        # Set the start step for each variable to a variable-specific value, or else an algorithm-wide value
        self.start_steps = dict()
        for v in self.variables:
            if v.type in ('var', 'logvar') and v.p2 is not None:
                self.start_steps[v.name] = v.p2
            elif 'simplex_log_step' in self.config.config and v.log_space:
                self.start_steps[v.name] = self.config.config['simplex_log_step']
            else:
                self.start_steps[v.name] = self.config.config['simplex_step']

        self.parallel_count = min(self.config.config['population_size'], max(len(self.variables) - 1, 1))
        self.iteration = 0
        self.alpha = self.config.config['simplex_reflection']
        self.gamma = self.config.config['simplex_expansion']
        self.beta = self.config.config['simplex_contraction']
        self.tau = self.config.config['simplex_shrink']

        self.simplex = []  # (score, PSet) points making up the simplex. Sorted after each iteration.

        # Data structures to keep track of the progress of one iteration.
        # In these, index 0 corresponds to the process from the worst point on the simplex, simplex[-1], index 1 to
        # simplex[-2], etc.
        self.stages = []  # Which stage of the iteration am I on? -1 initialization; 1 running first point; 2 running
        # second point; 3 done
        self.first_points = []  # Store (score, PSet) after the first run of the iteration completes
        self.second_points = []  # Store (score, PSet) after the second run completes, if applicable
        self.cases = []  # Which case number triggered after I got the score for the first point? (1, 2 or 3)
        self.centroids = []  # Contains dicts containing the centroid of all simplex points except the one that I am
        # working with
        self.pending = dict()  # Maps PSet name (str) to the index of the point in the above 3 lists.
        self.refine = refine

    def reset(self, bootstrap=None):
        super(SimplexAlgorithm, self).reset(bootstrap)
        self.iteration = 0
        self.simplex = []

        self.stages = []
        self.first_points = []
        self.second_points = []
        self.cases = []
        self.centroids = []
        self.pending = dict()

    def _parse_start_point(self):
        """
        Called when the start point is not passed in the config (which is when we're doing a pure simplex run,
        as opposed to a refinement at the end of the run)
        Parses the info out of the variable specs, and sets the appropriate PSet into the config.
        """
        start_vars = []
        for v in self.variables:
            if v.type == 'var':
                start_vars.append(v.set_value(v.p1))
            elif v.type == 'logvar':
                start_vars.append(v.set_value(exp10(v.p1)))
        start_pset = PSet(start_vars)
        self.config.config['simplex_start_point'] = start_pset

    def start_run(self):
        print2('Running local optimization by the Simplex algorithm for %i iterations' % self.max_iterations)

        # Generate the initial  num_variables+1 points in the simplex by moving parameters, one at a time, by the
        # specified step size
        self.start_point.name = 'simplex_init0'
        init_psets = [self.start_point]
        self.pending[self.start_point.name] = 0
        i = 1
        for v in self.variables:
            new_vars = []
            for p in self.start_point:
                if p.name == v.name:
                    new_vars.append(p.add(self.start_steps[p.name]))
                else:
                    new_vars.append(p)
            new_pset = PSet(new_vars)
            new_pset.name = 'simplex_init%i' % i
            self.pending[new_pset.name] = i
            i += 1
            init_psets.append(new_pset)
        self.simplex = []
        self.stages = [-1]*len(init_psets)
        return init_psets

    def got_result(self, res):

        pset = res.pset
        score = res.score
        index = self.pending.pop(pset.name)

        if self.stages[index] == -1:
            # Point is part of initialization
            self.simplex.append((score, pset))
            self.stages[index] = 3
        elif self.stages[index] == 2:
            # Point is the 2nd point run within one iteration
            self.second_points[index] = (score, pset)
            self.stages[index] = 3
        elif self.stages[index] == 1:
            # Point is the 1st point run within one iteration
            # We do the case-wise breakdown to pick the 2nd point, if any.
            self.first_points[index] = (score, pset)
            if score < self.simplex[0][0]:
                # Case 1: The point is better than the current global min.
                # We calculate the expansion point
                self.cases[index] = 1
                new_vars = []
                for v in self.variables:
                    new_var = v.set_value(self.a_plus_b_times_c_minus_d(pset[v.name], self.gamma, pset[v.name], self.centroids[index][v.name],
                                                                v))
                    new_vars.append(new_var)
                new_pset = PSet(new_vars)
                new_pset.name = 'simplex_iter%i_pt%i-2' % (self.iteration, index)
                self.pending[new_pset.name] = index
                self.stages[index] = 2
                return [new_pset]
            elif score < self.simplex[-index-2][0]:
                # Case 2: The point is worse than the current min, but better than the next worst point
                # Note that simplex[-index-1] is the point that this one was built from, so we check [-index-2]
                # We don't run a second point in this case.
                self.cases[index] = 2
                self.stages[index] = 3
                if min(self.stages) < 3:
                    return []
                # Otherwise have to jump to next iteration, below.
            else:
                # Case 3: The point is not better than the next worst point.
                # We calculate the contraction point
                self.cases[index] = 3
                # Work off the original or the reflection, whichever is better
                if score < self.simplex[-index-1][0]:
                    a_hat = pset
                else:
                    a_hat = self.simplex[-index-1][1]
                new_vars = []
                for v in self.variables:
                    # I think the equation for this in Lee et al p. 178 is wrong; I am instead using the analog to the
                    # equation on p. 176
                    # new_dict[v] = self.centroids[index][v] + self.beta * (a_hat[v] - self.centroids[index][v])
                    new_var = v.set_value(self.a_plus_b_times_c_minus_d(self.centroids[index][v.name], self.beta, a_hat[v.name],
                                                                self.centroids[index][v.name], v))
                    new_vars.append(new_var)
                new_pset = PSet(new_vars)
                new_pset.name = 'simplex_iter%i_pt%i-2' % (self.iteration, index)
                self.pending[new_pset.name] = index
                self.stages[index] = 2
                return [new_pset]
        else:
            raise RuntimeError('Internal error in SimplexAlgorithm')

        if min(self.stages) == 3:
            # All points in current iteration completed
            self.iteration += 1
            if self.iteration % self.config.config['output_every'] == 0:
                self.output_results()
            if self.iteration % 10 == 0:
                print1('Completed %i of %i iterations' % (self.iteration, self.max_iterations))
            else:
                print2('Completed %i of %i iterations' % (self.iteration, self.max_iterations))
            print2('Current best score: %f' % sorted(self.simplex, key=lambda x: x[0])[0][0])

            # If not an initialization iteration, update the simplex based on all the results
            if len(self.first_points) > 0:
                productive = False
                for i in range(len(self.first_points)):
                    si = -i-1  # Index into the simplex
                    if self.cases[i] == 1:
                        productive = True
                        if self.first_points[i][0] < self.second_points[i][0]:
                            self.simplex[si] = self.first_points[i]
                        else:
                            self.simplex[si] = self.second_points[i]
                    elif self.cases[i] == 2:
                        productive = True
                        self.simplex[si] = self.first_points[i]
                    elif self.cases[i] == 3:
                        if (self.second_points[i][0] < self.first_points[i][0]
                           and self.second_points[i][0] < self.simplex[si][0]):
                            productive = True
                            self.simplex[si] = self.second_points[i]
                        elif self.first_points[i][0] < self.simplex[si][0]:
                            self.simplex[si] = self.first_points[i]
                        # else don't edit the simplex, neither is an improvement
                    else:
                        raise RuntimeError('Internal error in SimplexAlgorithm')

                if self.iteration == self.max_iterations:
                    return 'STOP'  # Quit after the final simplex update

                if not productive:
                    # None of the points in the last iteration improved the simplex.
                    # Now we have to contract the simplex
                    self.simplex = sorted(self.simplex, key=lambda x: x[0])
                    new_simplex = []
                    for i in range(1, len(self.simplex)):
                        new_vars = []
                        for v in self.variables:
                            # new_dict[v] = self.tau * self.simplex[i-1][1][v] + (1 - self.tau) * self.simplex[i][1][v]
                            new_var = v.set_value(self.ab_plus_cd(self.tau, self.simplex[0][1][v.name], 1 - self.tau,
                                                      self.simplex[i][1][v.name], v))
                            new_vars.append(new_var)
                        new_pset = PSet(new_vars)
                        new_pset.name = 'simplex_iter%i_pt%i' % (self.iteration, i)
                        self.pending[new_pset.name] = i - 1
                        new_simplex.append(new_pset)

                    # Prepare for new reinitialization run
                    # We don't need to rescore simplex[0], but the rest of the PSets are new and we do.
                    self.stages = [-1] * len(new_simplex)
                    self.first_points = []
                    self.second_points = []
                    self.simplex = [self.simplex[0]]
                    return new_simplex

            ###
            # Set up the next iteration
            # Re-sort the simplex based on the updated objectives
            self.simplex = sorted(self.simplex, key=lambda x: x[0])
            self._check_degeneracy()
            if self.iteration == self.max_iterations:
                return 'STOP' # Extra catch if finish on a rebuild the simplex iteration
            # Find the reflection point for the n worst points
            reflections = []
            self.centroids = []
            # Sum of each param value, to help take the reflections
            sums = self.get_sums() # Returns in log space for log variables
            max_diff = 0.
            for ai in range(self.parallel_count):
                a = self.simplex[-ai-1][1]
                new_vars = []
                this_centroid = dict()
                for v in self.variables:
                    if v.log_space:
                        # Calc centroid in regular space.
                        centroid = exp10((sums[v.name] - np.log10(a[v.name])) / (len(self.simplex) - 1))
                    else:
                        centroid = (sums[v.name] - a[v.name]) / (len(self.simplex) - 1)
                    this_centroid[v.name] = centroid
                    # new_dict[v] = centroid + self.alpha * (centroid - a[v])
                    new_var = v.set_value(self.a_plus_b_times_c_minus_d(centroid, self.alpha, centroid, a[v.name], v))
                    new_vars.append(new_var)
                    max_diff = max(max_diff, abs(new_var.diff(a.get_param(v.name))))
                self.centroids.append(this_centroid)
                new_pset = PSet(new_vars)
                new_pset.name = 'simplex_iter%i_pt%i' % (self.iteration, ai)
                reflections.append(new_pset)
                self.pending[new_pset.name] = ai
            # Check for stop criterion due to moves being too small
            if max_diff < self.config.config['simplex_stop_tol']:
                logger.info('Stopping simplex because the maximum move attempted this iteration was %s' % max_diff)
                return 'STOP'

            # Reset data structures to track this iteration
            self.stages = [1] * len(reflections)
            self.first_points = [None] * len(reflections)
            self.second_points = [None] * len(reflections)
            self.cases = [None] * len(reflections)

            return reflections
        else:
            # Wait for the rest of the parallel jobs to finish this iteration
            return []

    def get_sums(self):
        """
        Simplex helper function
        Returns a dict mapping parameter name p to the sum of the parameter value over the entire current simplex
        :return: dict
        """
        # return {p: sum(point[1][p] for point in self.simplex) for p in self.simplex[0][1].keys()}
        sums = dict()
        for p in self.simplex[0][1]:
            if not p.log_space:
                sums[p.name] = sum(point[1][p.name] for point in self.simplex)
            else:
                sums[p.name] = sum(np.log10(point[1][p.name]) for point in self.simplex)
        return sums

    def _check_degeneracy(self):
        """
        Check if the simplex has become degenerate (near-zero volume) and perturb vertices if so.
        Uses the determinant of the edge matrix to measure simplex volume.
        """
        if len(self.simplex) < 3:
            return
        n = len(self.variables)
        # Build edge matrix: rows are (vertex_i - vertex_0) for i=1..n, in the appropriate space
        v0 = self.simplex[0][1]
        edge_matrix = np.zeros((len(self.simplex) - 1, n))
        for i in range(1, len(self.simplex)):
            for j, v in enumerate(self.variables):
                if v.log_space:
                    edge_matrix[i - 1, j] = np.log10(self.simplex[i][1][v.name]) - np.log10(v0[v.name])
                else:
                    edge_matrix[i - 1, j] = self.simplex[i][1][v.name] - v0[v.name]

        # Compute a scale factor from the edge lengths to make the threshold relative
        edge_norms = np.linalg.norm(edge_matrix, axis=1)
        scale = np.mean(edge_norms) if np.mean(edge_norms) > 0 else 1.0

        # For a square matrix, volume ~ |det|. Check if it's near-zero relative to scale.
        if edge_matrix.shape[0] == edge_matrix.shape[1]:
            vol = abs(np.linalg.det(edge_matrix))
            threshold = (1e-10 * scale) ** n
        else:
            # Non-square: use product of singular values as a volume proxy
            sv = np.linalg.svd(edge_matrix, compute_uv=False)
            vol = np.prod(sv)
            threshold = (1e-10 * scale) ** min(edge_matrix.shape)

        if vol < threshold:
            logger.warning('Simplex is nearly degenerate (volume=%.2e). Perturbing vertices.' % vol)
            for i in range(1, len(self.simplex)):
                old_pset = self.simplex[i][1]
                new_vars = []
                for v in self.variables:
                    if v.log_space:
                        log_val = np.log10(old_pset[v.name])
                        perturbed = 10 ** (log_val + np.random.normal(0, 0.01 * scale))
                    else:
                        perturbed = old_pset[v.name] + np.random.normal(0, 0.01 * scale)
                    perturbed = max(v.lower_bound, min(v.upper_bound, perturbed))
                    new_vars.append(v.set_value(perturbed))
                new_pset = PSet(new_vars)
                new_pset.name = old_pset.name
                self.simplex[i] = (self.simplex[i][0], new_pset)

    def a_plus_b_times_c_minus_d(self, a, b, c, d, v):
        """
        Performs the calculation a + b*(c-d), where a, c, and d are assumed to be in log space if v is in log space,
        and the final result respects the box constraints on v.

        :param a:
        :param b:
        :param c:
        :param d:
        :param v:
        :type v: FreeParameter
        :return:
        """

        if v.log_space:
            result = 10 ** (np.log10(a) + b*(np.log10(c) - np.log10(d)))
        else:
            result = a + b*(c-d)
        return max(v.lower_bound, min(v.upper_bound, result))

    def ab_plus_cd(self, a, b, c, d, v):
        """
        Performs the calculation ab + cd where b and d are assumed to be in log space if v is in log space,
        and the final result respects the box constraints on v
        :param a:
        :param b:
        :param c:
        :param d:
        :param v:
        :type v: FreeParameter
        :return:
        """
        if v.log_space:
            result = 10 ** (a * np.log10(b) + c*np.log10(d))
        else:
            result = a * b + c * d
        return max(v.lower_bound, min(v.upper_bound, result))


class ModelCheck(object):
    """
    An algorithm that just checks the fit quality for a job with no free parameters.

    Does not subclass Algorithm. To run, instead call run_check() with no Cluster.
    """

    def __init__(self, config):
        """
        Instantiates ModelCheck with a Configuration object.
        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        self.bootstrap_number = None

        logger.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        if self.config.config['simulation_dir']:
            self.sim_dir = self.config.config['simulation_dir'] + '/Simulations'
        else:
            self.sim_dir = self.config.config['output_dir'] + '/Simulations'

        # Store a list of all Model objects.
        self.model_list = copy.deepcopy(list(self.config.models.values()))

    def run_check(self, debug=False):
        """Main loop for executing the algorithm"""

        print1('Running model checking on the given model(s)')

        empty = PSet([])
        empty.name = 'check'
        job = core.Job(self.model_list, empty, 'check', self.sim_dir, self.config.config['wall_time_sim'], None,
                  None, dict(), delete_folder=False,
                  stochastic_seed_policy=self.config.config['stochastic_seed'])
        result = core.run_job(job, debug, self.sim_dir)

        if isinstance(result, FailedSimulation):
            print0('Simulation failed.')
            return

        result.normalize(self.config.config['normalization'])
        try:
            result.postprocess_data(self.config.postprocessing)
        except Exception:
            logger.exception('User-defined post-processing script failed')
            traceback.print_exc()
            print0('User-defined post-processing script failed. Exiting')
            return

        result.score = self.objective.evaluate_multiple(result.simdata, self.exp_data, result.pset,
                                                         self.config.constraints)
        if result.score is None:
            print0('Simulation contained NaN or Inf values. Cannot calculate objective value.')
            return
        print0('Objective value is %s' % result.score)
        if len(self.config.constraints) > 0:
            counter = ConstraintCounter()
            fail_count = counter.evaluate_multiple(result.simdata, self.exp_data, self.config.constraints)
            total = sum([len(cset.constraints) for cset in self.config.constraints])
            print('Satisfied %i out of %i constraints' % (total-fail_count, total))
            for cset in self.config.constraints:
                cset.output_itemized_eval(result.simdata, self.sim_dir)
