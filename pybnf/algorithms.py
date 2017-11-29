"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


from distributed import as_completed
from distributed import Client
from subprocess import run
from subprocess import CalledProcessError
from subprocess import PIPE
from subprocess import STDOUT

from .data import Data
from .pset import PSet
from .pset import Trajectory

import logging
import numpy as np
import os
import re
import shutil
import copy


class Result(object):
    """
    Container for the results of a single evaluation in the fitting algorithm
    """

    def __init__(self, paramset, simdata, log, name):
        """
        Instantiates a Result

        :param paramset: The parameters corresponding to this evaluation
        :type paramset: PSet
        :param simdata: The simulation results corresponding to this evaluation
        :type simdata: list of Data instances
        :param log: The stdout + stderr of the simulations
        :type log: list of str
        """
        self.pset = paramset
        self.simdata = simdata
        self.log = log
        self.name = name
        self.score = None  # To be set later when the Result is scored.


class FailedSimulation(object):
    def __init__(self, i):
        self.id = i


class Job:
    """
    Container for information necessary to perform a single evaluation in the fitting algorithm
    """

    def __init__(self, models, params, job_id, bngcommand, output_dir):
        """
        Instantiates a Job

        :param models: The models to evaluate
        :type models: list of Model instances
        :param params: The parameter set with which to evaluate the model
        :type params: PSet
        :param job_id: Job identification; also the folder name that the job gets saved to
        :type job_id: str
        :param bngcommand: Command to run BioNetGen
        :type bngcommand: str
        :param output_dir path to the directory where I should create my simulation folder
        :type output_dir: str
        """
        self.models = models
        self.params = params
        self.job_id = job_id
        self.bng_program = bngcommand
        self.output_dir = output_dir
        self.home_dir = os.getcwd()

    def _name_with_id(self, model):
        return '%s_%s' % (model.name, self.job_id)

    def _write_models(self):
        """Writes models to file"""
        model_files = []
        for i, model in enumerate(self.models):
            model_file_name = self._name_with_id(model) + ".bngl"
            model_with_params = model.copy_with_param_set(self.params)
            model_with_params.save(model_file_name)
            model_files.append(model_file_name)
        return model_files

    def run_simulation(self):
        """Runs the simulation and reads in the result"""

        folder = '%s/%s' % (self.output_dir, self.job_id)
        os.mkdir(folder)
        try:
            os.chdir(folder)
            model_files = self._write_models()
            log = self.execute(model_files)
            simdata = self.load_simdata()
            os.chdir(self.home_dir)
            return Result(self.params, simdata, log, self.job_id)
        except CalledProcessError:
            return FailedSimulation(self.job_id)

    def execute(self, models):
        """Executes model simulations"""
        log = []
        for model in models:
            cmd = '%s %s' % (self.bng_program, model)
            cp = run(cmd, shell=True, check=True, stderr=STDOUT, stdout=PIPE)
            log.append(cp.stdout)
        return log

    def load_simdata(self):
        """
        Function to load simulation data after executing all simulations for an evaluation

        Returns a nested dictionary structure.  Top-level keys are model names and values are
        dictionaries whose keys are action suffixes and values are Data instances

        :return: dict of dict
        """
        ds = {}
        for model in self.models:
            ds[model.name] = {}
            for suff in model.suffixes:
                if suff[0] == 'simulate':
                    data_file = '%s_%s.gdat' % (self._name_with_id(model), suff[1])
                    data = Data(file_name=data_file)
                else:  # suff[0] == 'parameter_scan'
                    data_file = '%s_%s.scan' % (self._name_with_id(model), suff[1])
                    data = Data(file_name=data_file)
                ds[model.name][suff[1]] = data
        return ds


class Algorithm(object):
    def __init__(self, config):
        """
        Instantiates an Algorithm with a Configuration object.  Also initializes a
        Trajectory instance to track the fitting progress, and performs various additional
        configuration that is consistent for all algorithms

        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        self.trajectory = Trajectory(config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0

        # Store a list of all Model objects. Change this as needed for compatibility with other parts
        self.model_list = list(self.config.models.values())

        # Generate a list of variable names
        self.variables = self.config.variables

        # Set the space (log or regular) in which each variable moves, as well as the box constraints on the variable.
        # Currently, this is set based on what distribution the variable is initialized with, but these could be made
        # into a separate, custom options
        self.variable_space = dict()  # Contains tuples (space, min_value, max_value)
        for v in self.config.variables_specs:
            if v[1] == 'random_var':
                self.variable_space[v[0]] = ('regular', v[2], v[3])
            elif v[1] == 'lognormrandom_var':
                self.variable_space[v[0]] = ('log', 0., np.inf)  # Questionable if this is the behavior we want.
            elif v[1] == 'loguniform_var':
                self.variable_space[v[0]] = ('log', v[2], v[3])
            elif v[1] == 'static_list_var':
                self.variable_space[v[0]] = ('static', )  # Todo: what is the actual way to mutate this type of param?
            else:
                raise RuntimeError('Unrecognized variable type: %s' % v[1])

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        Algorithm subclasses optionally may set the .name field of the PSet objects to give a meaningful unique
        identifier such as 'gen0ind42'. If so, they MUST BE UNIQUE, as this determines the folder name.
        Uniqueness will not be checked elsewhere.

        :return: list of PSets
        """
        logging.info("Initializing algorithm")
        raise NotImplementedError("Subclasses must implement start_run()")

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: result from the completed simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """
        logging.info("Retrieved result")
        raise NotImplementedError("Subclasses must implement got_result()")

    def add_to_trajectory(self, res):
        """
        Evaluates the objective function for a Result, and adds the information from the Result to the Trajectory
        instance"""

        score = self.objective.evaluate_multiple(res.simdata, self.exp_data)
        res.score = score
        self.trajectory.add(res.pset, score, res.name)

    def random_pset(self):
        """
        Generates a random PSet based on the distributions and bounds for each parameter specified in the configuration

        :return:
        """
        param_dict = dict()
        for (name, type, val1, val2) in self.config.variables_specs:
            if type == 'random_var':
                param_dict[name] = np.random.uniform(val1, val2)
            elif type == 'loguniform_var':
                param_dict[name] = 10.**np.random.uniform(np.log10(val1), np.log10(val2))
            elif type == 'lognormrandom_var':
                param_dict[name] = 10.**np.random.normal(val1, val2)
            elif type == 'static_list_var':
                param_dict[name] = np.random.choice(val1)
            else:
                raise RuntimeError('Unrecognized variable type: %s' % type)
        return PSet(param_dict)

    def random_latin_hypercube_psets(self, n):
        """
        Generates n random PSets with a latin hypercube distribution
        More specifically, the random_var and loguniform_var variables follow the latin hypercube distribution,
        while lognorm and static_list variables are randomized normally.

        :param n: Number of psets to generate
        :return:
        """
        # Generate latin hypercube of dimension = number of uniformly distributed variables.
        num_uniform_vars = len([x for x in self.config.variables_specs
                               if x[1] == 'random_var' or x[1] == 'lognormrandom_var'])
        rands = latin_hypercube(n, num_uniform_vars)
        psets = []
        for row in rands:
            # Initialize the variables
            # Convert the 0 to 1 random numbers to the required variable range
            param_dict = dict()
            rowindex = 0
            for (name, type, val1, val2) in self.config.variables_specs:
                if type == 'random_var':
                    param_dict[name] = val1 + row[rowindex]*(val2-val1)
                    rowindex += 1
                elif type == 'loguniform_var':
                    param_dict[name] = 10. ** (val1 + row[rowindex]*(val2-val1))
                    rowindex += 1
                elif type == 'lognormrandom_var':
                    param_dict[name] = 10. ** np.random.normal(val1, val2)
                elif type == 'static_list_var':
                    param_dict[name] = np.random.choice(val1)
                else:
                    raise RuntimeError('Unrecognized variable type: %s' % type)
            psets.append(PSet(param_dict))
        return psets

    def add(self, paramset, param, value):
        """
        Helper function to add a value to a param in a parameter set,
        taking into account
        1) Whether this parameter is to be moved in regular or log space
        2) Box constraints on the parameter
        :param paramset:
        :type paramset: PSet
        :param param: name of the parameter
        :type param: str
        :param value: value to be added
        :type value: float
        :return: The result of the addition
        """
        if self.variable_space[param][0] == 'regular':
            return max(self.variable_space[param][1], min(self.variable_space[param][2], paramset[param] + value))
        elif self.variable_space[param][0] == 'log':
            return max(self.variable_space[param][1], min(self.variable_space[param][2],
                                                          10.**(np.log10(paramset[param]) + value)))
        elif self.variable_space[param][0] == 'static':
            return paramset[param]
        else:
            raise RuntimeError('Unrecognized variable space type: %s' % self.variable_space[param][0])

    def diff(self, paramset1, paramset2, param):
        """
        Helper function to calculate paramset1[param] - paramset2[param], taking into account whether
        param is in regular or log space
        """
        if self.variable_space[param][0] == 'regular':
            return paramset1[param] - paramset2[param]
        elif self.variable_space[param][0] == 'log':
            return np.log10(paramset1[param] / paramset2[param])
        elif self.variable_space[param][0] == 'static':
            return 0.  # Don't know what to do here...
        else:
            raise RuntimeError('Unrecognized variable space type: %s' % self.variable_space[param][0])

    def make_job(self, params):
        """
        Creates a new Job using the specified params, and additional specifications that are already saved in the
        Algorithm object

        :param params:
        :type params: PSet
        :return: Job
        """
        if params.name:
            job_id = params.name
        else:
            self.job_id_counter += 1
            job_id = 'sim_%i' % self.job_id_counter
        return Job(self.model_list, params, job_id, self.config.config['bng_command'],
                   self.config.config['output_dir']+'/Simulations/')

    def output_results(self, name=''):
        """
        Tells the Trajectory to output a log file now with the current best fits.

        This should be called periodically by each Algorithm subclass, and is called by the Algorithm class at the end
        of the simulation.
        :return:
        :param name: Custom string to add to the saved filename. If omitted, we just use a running counter of the
        number of times we've outputted.
        :type name: str
        """
        if name == '':
            name = str(self.output_counter)
        self.output_counter += 1
        filepath = '%s/Results/sorted_params_%s.txt' % (self.config.config['output_dir'], name)
        self.trajectory.write_to_file(filepath)

        # If the user has asked for fewer output files, each time we're here, move the new file to
        # Results/sorted_params.txt, overwriting the previous one.
        if self.config.config['delete_old_files'] == 1:
            noname_filepath = '%s/Results/sorted_params.txt' % self.config.config['output_dir']
            if os.path.isfile(noname_filepath):
                os.remove(noname_filepath)
            os.rename(filepath, noname_filepath)

    def run(self):
        """Main loop for executing the algorithm"""
        client = Client()
        psets = self.start_run()
        jobs = [self.make_job(p) for p in psets]
        futures = [client.submit(job.run_simulation) for job in jobs]
        pending = set(futures)
        pool = as_completed(futures, with_results=True)
        while True:
            f, res = next(pool)
            pending.remove(f)
            self.add_to_trajectory(res)
            response = self.got_result(res)
            if response == 'STOP':
                logging.info("Stop criterion satisfied")
                break
            else:
                new_jobs = [self.make_job(ps) for ps in response]
                new_futures = [client.submit(j.run_simulation) for j in new_jobs]
                pending.update(new_futures)
                pool.update(new_futures)
        client.cancel(list(pending))
        logging.debug("Pending jobs cancelled")
        client.close()
        self.output_results('final')

        # Copy the best simulations into the results folder
        best_name = self.trajectory.best_fit_name()
        for m in self.config.models:
            shutil.copy('%s/Simulations/%s/%s_%s.bngl' %
                        (self.config.config['output_dir'], best_name, m, best_name),
                        '%s/Results' % self.config.config['output_dir'])
            for suf in self.config.mapping[m]:
                shutil.copy('%s/Simulations/%s/%s_%s_%s.gdat' %
                            (self.config.config['output_dir'], best_name, m, best_name, suf),
                            '%s/Results' % self.config.config['output_dir'])

        logging.info("Fitting complete!")


class ParticleSwarm(Algorithm):
    """
    Implements particle swarm optimization.

    The implementation roughly follows Moraes et al 2015, although is reorganized to better suit PyBNF's format.
    Note the global convergence criterion discussed in that paper is not used (would require too long a
    computation), and instead uses ????

    """

    def __init__(self, config):

        # Former params that are now part of the config
        #variable_list, num_particles, max_evals, cognitive=1.5, social=1.5, w0=1.,
        #wf=0.1, nmax=30, n_stop=np.inf, absolute_tol=0., relative_tol=0.)
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

        # Set default values for non-essential parameters - no longer here; now done in Config.

        conf_dict = config.config  # Dictionary from the Configuration object

        # This default value gets special treatment because if missing, it should take the value of particle_weight,
        # disabling the adaptive weight change entirely.
        if 'particle_weight_final' not in conf_dict:
            conf_dict['particle_weight_final'] = conf_dict['particle_weight']

        # Save config parameters
        self.c1 = conf_dict['cognitive']
        self.c2 = conf_dict['social']
        self.max_evals = conf_dict['population_size'] * conf_dict['max_iterations']
        self.output_every = conf_dict['population_size'] * conf_dict['output_every']

        self.num_particles = conf_dict['population_size']
        # Todo: Nice error message if a required key is missing

        self.w0 = conf_dict['particle_weight']

        self.wf = conf_dict['particle_weight_final']
        self.nmax = conf_dict['adaptive_n_max']
        self.n_stop = conf_dict['adaptive_n_stop']
        self.absolute_tol = conf_dict['adaptive_abs_tol']
        self.relative_tol = conf_dict['adaptive_rel_tol']

        self.nv = 0  # Counter that controls the current weight. Counts number of "unproductive" iterations.
        self.num_evals = 0  # Counter for the total number of results received

        # Initialize storage for the swarm data
        self.swarm = []  # List of lists of the form [PSet, velocity]. Velocity is stored as a dict with the same keys
        # as PSet
        self.pset_map = dict()  # Maps each PSet to it s particle number, for easy lookup.
        self.bests = [[None, np.inf]] * self.num_particles  # The best result for each particle: list of the
        # form [PSet, objective]
        self.global_best = [None, np.inf]  # The best result for the whole swarm
        self.last_best = np.inf

    def start_run(self):
        """
        Start the run by initializing n particles at random positions and velocities
        :return:
        """

        if self.config.config['initialization'] == 'lh':
            new_params_list = self.random_latin_hypercube_psets(self.num_particles)
        else:
            new_params_list = [self.random_pset() for i in range(self.num_particles)]

        for i in range(len(new_params_list)):
            p = new_params_list[i]
            p.name = 'iter0p%i' % i
            # Todo: Smart way to initialize velocity?
            new_velocity = {xi: np.random.uniform(-1, 1) for xi in self.variables}
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
            # End of one "pseudoflight", check if it was productive.
            if (self.last_best != np.inf and
                    np.abs(self.last_best - self.global_best[1]) <
                    self.absolute_tol + self.relative_tol * self.last_best):
                self.nv += 1
            self.last_best = self.global_best[1]

        if self.num_evals % self.output_every == 0:
            self.output_results()

        p = self.pset_map.pop(paramset)  # Particle number

        # Update best scores if needed.
        if score < self.bests[p][1]:
            self.bests[p] = [paramset, score]
            if score < self.global_best[1]:
                self.global_best = [paramset, score]

        # Update own position and velocity
        # The order matters - updating velocity first seems to make the best use of our current info.
        w = self.w0 + (self.wf - self.w0) * self.nv / (self.nv + self.nmax)
        self.swarm[p][1] = {v:
                                w * self.swarm[p][1][v] + self.c1 * np.random.random() * (
                                self.diff(self.bests[p][0], self.swarm[p][0], v)) +
                                self.c2 * np.random.random() * self.diff(self.global_best[0], self.swarm[p][0], v)
                            for v in self.variables}
        new_pset = PSet({v: self.add(self.swarm[p][0], v, self.swarm[p][1][v]) for v in self.variables})
        self.swarm[p][0] = new_pset

        # This will cause a crash if new_pset happens to be the same as an already running pset in pset_map.
        # This could come up in practice if all parameters have hit a box constraint.
        # As a simple workaround, perturb the parameters slightly
        while new_pset in self.pset_map:
            retry_dict = {v: self.add(new_pset, v, np.random.uniform(-1e-6, 1e-6)) for v in self.variables}
            new_pset = PSet(retry_dict)

        self.pset_map[new_pset] = p

        # Set the new name: the old pset name is iter##p##
        # Extract the iter number
        iternum = int(re.search('iter([0-9]+)', paramset.name).groups()[0])
        new_pset.name = 'iter%ip%i' % (iternum+1, p)

        # Check for stopping criteria
        if self.num_evals >= self.max_evals or self.nv >= self.n_stop:
            return 'STOP'

        return [new_pset]


class DifferentialEvolution(Algorithm):
    """
    Implements the parallelized, island-based differential evolution algorithm
    described in Penas et al 2015.

    In some cases, I had to make my own decisions for specifics I couldn't find in the original paper. Namely:
    At each migration, a user-defined number of individuals are migrated from each island. For each individual, a
    random index is chosen; the same index for all islands. A random permutation is used to redistribute individuals
    with that index to different islands.

    Each island performs its migration individually, on the first callback when all islands are ready for that
    migration. It receives individuals from the migration iteration, regardless of what the current iteration is.
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
        if config.config['population_size'] % config.config['islands'] != 0:
            logging.warning('Reduced population_size to %i to evenly distribute it over %i islands' %
                            (self.num_islands * self.num_per_island, self.num_islands))
        self.migrate_every = config.config['migrate_every']
        if self.num_islands == 1:
            self.migrate_every = np.inf
        self.mutation_rate = config.config['mutation_rate']
        self.mutation_factor = config.config['mutation_factor']
        self.max_iterations = config.config['max_iterations']
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

        self.strategy = 'rand1'  # Customizable later

    def start_run(self):

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

        # Set all fitnesses to Inf, guaraneeting a replacement by the first proposed individual
        self.fitnesses = [[np.Inf
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
        if fitness < self.fitnesses[island][j]:
            print('Replaced at %i,%i; old score %f new score %f' % (island, j, self.fitnesses[island][j], fitness))
            self.individuals[island][j] = pset
            self.fitnesses[island][j] = fitness
        else:
            print(
                'Didn\'t replace at %i,%i; old score %f new score %f' % (island, j, self.fitnesses[island][j], fitness))

        self.waiting_count[island] -= 1

        # Determine if the current iteration is over for the current island
        if self.waiting_count[island] == 0:

            self.iter_num[island] += 1
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
                    logging.debug('Island %i just set up the migration.' % island)
                    # logging.debug(str(self.migration_transit))
                    # logging.debug(str(self.migration_indices))
                    # logging.debug(str(self.migration_perms))

                # Send the required PSets to migration_transit
                for j in self.migration_indices[migration_num]:
                    self.migration_transit[migration_num][island].append((self.individuals[island][j],
                                                                          self.fitnesses[island][j]))
                # Tell other islands that this one is ready for this migration.
                self.migration_ready[island] = migration_num

            if self.migration_done[island] < min(self.migration_ready):
                # This island performs a migration
                logging.debug('Island %i is migrating!' % island)
                # logging.debug(str(self.migration_transit))
                # logging.debug(str(self.migration_indices))
                # logging.debug(str(self.migration_perms))
                migration_num = self.migration_done[island] + 1

                # Fetch the appropriate new individuals from migration_transit
                for migrater_index in range(self.num_to_migrate):
                    j = self.migration_indices[migration_num][migrater_index]  # Index of the individual
                    newisland = self.migration_perms[migration_num][migrater_index][island]
                    self.individuals[island][j], self.fitnesses[island][j] = \
                        self.migration_transit[migration_num][newisland][migrater_index]

                    logging.debug('Island %i gained new individual with fitness %f' % (island, self.fitnesses[island][j]))

                self.migration_done[island] = migration_num
                if min(self.migration_done) == migration_num:
                    # This is the last island to complete this migration
                    # Delete the migration data to free space.
                    del self.migration_transit[migration_num]
                    del self.migration_perms[migration_num]
                    del self.migration_indices[migration_num]

            # Set up the next generation
            for jj in range(self.num_per_island):
                new_pset = self.new_individual(island)
                # If the new pset is a duplicate of one already in the island_map, it will cause problems.
                # As a workaround, perturb it slightly.
                while new_pset in self.island_map:
                    retry_dict = {v: self.add(new_pset, v, np.random.uniform(-1e-6, 1e-6)) for v in self.variables}
                    new_pset = PSet(retry_dict)
                self.proposed_individuals[island][jj] = new_pset
                self.island_map[new_pset] = (island, jj)
                if self.num_islands == 1:
                    new_pset.name = 'gen%iind%i' % (self.iter_num[island], jj)
                else:
                    new_pset.name = 'gen%iisl%iind%i' % (self.iter_num[island], island, jj)

            self.waiting_count[island] = self.num_per_island

            if self.iter_num[island] % 20 == 0:
                logging.info('Island %i completed %i iterations' % (island, self.iter_num[island]))
                # print(sorted(self.fitnesses[island]))

            # Convergence check
            if np.max(self.fitnesses) / np.min(self.fitnesses) < 1.002:
                return 'STOP'

            # Return a copy, so our internal data structure is not tampered with.
            return copy.copy(self.proposed_individuals[island])

        else:
            # Add no new jobs, wait for this generation to complete.
            return []

    def new_individual(self, island):
        """
        Create a new individual for the specified island, according to the set strategy

        :param island:
        :return:
        """

        # Choose a starting parameter set (either the best one, or a random one, or the one we want to replace)
        # and others to cross over
        if self.strategy in ['rand1']:
            picks = np.random.choice(self.individuals[island], 3, replace=False)
            base = picks[0]
            others = picks[1:]
        else:
            raise NotImplementedError('Please select one of the strategies from our extensive list of options: rand1')

        print('Base: ' + str(base))

        print('Others: ' + str(others))

        # Iterate through parameters; decide whether to mutate or leave the same.
        new_pset_dict = dict()
        for p in base.keys():
            if np.random.random() < self.mutation_rate:
                new_pset_dict[p] = self.add(base, p, self.mutation_rate * self.diff(others[0], others[1], p))
            else:
                new_pset_dict[p] = base[p]

        proposed = PSet(new_pset_dict)
        for isl in self.individuals:
            for ind in isl:
                if ind is None:
                    continue
                if ind == proposed:
                    print(str(proposed))
                    raise RuntimeError('Outrageous! You proposed a pset that we already have!')

        return PSet(new_pset_dict)


def latin_hypercube(nsamples, ndims):
    """
    Latin hypercube sampling.
    This code was dug out of the scipy.optimize.differentialevolution source code, and converted to work in the
    general case (which surprisingly, does not exist within scipy)

    Initializes the population with Latin Hypercube Sampling.
    Latin Hypercube Sampling ensures that each parameter is uniformly
    sampled over its range.

    Returns a nsamples by ndims array, with entries in the range [0,1]
    You'll have to rescale them to your actual param ranges.
    """
    rng = np.random

    # Each parameter range needs to be sampled uniformly. The scaled
    # parameter range ([0, 1)) needs to be split into
    # `self.num_population_members` segments, each of which has the following
    # size:
    segsize = 1.0 / nsamples

    # Within each segment we sample from a uniform random distribution.
    # We need to do this sampling for each parameter.
    samples = (segsize * rng.random_sample((nsamples, ndims))

    # Offset each segment to cover the entire parameter range [0, 1)
               + np.linspace(0., 1., nsamples,
                             endpoint=False)[:, np.newaxis])

    # Create an array for population of candidate solutions.
    population = np.zeros_like(samples)

    # Initialize population of candidate solutions by permutation of the
    # random samples.
    for j in range(ndims):
        order = rng.permutation(range(nsamples))
        population[:, j] = samples[order, j]

    return population
