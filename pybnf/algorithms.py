"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


from distributed import as_completed
from distributed import Client, LocalCluster
from subprocess import run
from subprocess import CalledProcessError
from subprocess import TimeoutExpired
from subprocess import STDOUT

from .data import Data
from .pset import PSet
from .pset import Trajectory
from .pset import NetModel
from .printing import print0, print1, print2

import logging
import numpy as np
import os
import re
import shutil
import copy
import sys
import traceback


class Result(object):
    """
    Container for the results of a single evaluation in the fitting algorithm
    """

    def __init__(self, paramset, simdata, name):
        """
        Instantiates a Result

        :param paramset: The parameters corresponding to this evaluation
        :type paramset: PSet
        :param simdata: The simulation results corresponding to this evaluation, as a nested dictionary structure.
        Top-level keys are model names and values are dictionaries whose keys are action suffixes and values are
        Data instances
        :type simdata: dict Returns a
        :param log: The stdout + stderr of the simulations
        :type log: list of str
        """
        self.pset = paramset
        self.simdata = simdata
        self.name = name
        self.score = None  # To be set later when the Result is scored.
        self.failed = False


class FailedSimulation(Result):
    def __init__(self, paramset, name, fail_type, einfo=tuple([None, None, None])):
        """
        Instantiates a FailedSimulation

        :param paramset:
        :param log:
        :param name:
        :param fail_type: 0 - Exceeded walltime, 1 - Other crash
        :type fail_type: int
        """
        super(FailedSimulation, self).__init__(paramset, None, name)
        self.fail_type = fail_type
        self.failed = True
        self.traceback = ''.join(traceback.format_exception(*einfo))


class Job:
    """
    Container for information necessary to perform a single evaluation in the fitting algorithm
    """

    def __init__(self, models, params, job_id, bngcommand, output_dir, timeout):
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
        self.home_dir = os.getcwd()  # This is safe because it is called from the scheduler, not the workers.
        # Force absolute paths for bngcommand and output_dir, because workers do not get the relative path info.
        if bngcommand[0] == '/':
            self.bng_program = bngcommand
        else:
            self.bng_program = self.home_dir + '/' + bngcommand
        if output_dir[0] == '/':
            self.output_dir = output_dir
        else:
            self.output_dir = self.home_dir + '/' + output_dir
        self.timeout = timeout

        # Folder where we save the model files and outputs.
        self.folder = '%s/%s' % (self.output_dir, self.job_id)

    def _name_with_id(self, model):
        return '%s_%s' % (model.name, self.job_id)

    def _write_models(self):
        """Writes models to file"""
        model_files = []
        for i, model in enumerate(self.models):
            model_file_prefix = self._name_with_id(model)
            model_with_params = model.copy_with_param_set(self.params)
            model_with_params.save('%s/%s' % (self.folder, model_file_prefix))
            model_files.append('%s/%s' % (self.folder, model_file_prefix))
        return model_files

    def run_simulation(self):
        """Runs the simulation and reads in the result"""

        # The check here is in case dask decides to run the same job twice, both of them can complete.
        made_folder = False
        failures = 0
        while not made_folder:
            try:
                os.mkdir(self.folder)
                made_folder = True
            except OSError:
                logging.info('Failed to create folder %s, trying again.' % self.folder)
                failures += 1
                self.folder = '%s/%s_rerun%i' % (self.output_dir, self.job_id, failures)
        try:
            model_files = self._write_models()
            self.execute(model_files)
            simdata = self.load_simdata()
            res = Result(self.params, simdata, self.job_id)
        except CalledProcessError:
            res = FailedSimulation(self.params, self.job_id, 1)
        except TimeoutExpired:
            res = FailedSimulation(self.params, self.job_id, 0)
        # This block is making bugs hard to diagnose
        # except Exception:
        #     res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())

        return res

    def execute(self, models):
        """Executes model simulations"""
        for model in models:
            cmd = '%s %s.bngl --outdir %s' % (self.bng_program, model, self.folder)
            log_file = '%s.log' % model
            with open(log_file, 'w') as lf:
                run(cmd, shell=True, check=True, stderr=STDOUT, stdout=lf, timeout=self.timeout)

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
                    data_file = '%s/%s_%s.gdat' % (self.folder, self._name_with_id(model), suff[1])
                    data = Data(file_name=data_file)
                else:  # suff[0] == 'parameter_scan'
                    data_file = '%s/%s_%s.scan' % (self.folder, self._name_with_id(model), suff[1])
                    data = Data(file_name=data_file)
                ds[model.name][suff[1]] = data
        return ds


class JobGroup:
    """
    Represents a group of jobs that are identical replicates to be averaged together for smoothing
    """
    def __init__(self, job_id, subjob_ids):
        """
        :param job_id: The name of the Job this group is representing
        :param subjob_ids: A list of the ids of the identical replicate Jobs.
        """
        self.job_id = job_id
        self.subjob_ids = subjob_ids
        self.result_list = []
        self.failed = None

    def job_finished(self, res):
        """
        Called when one job in this group has finished
        :param res: Result object for the completed job
        :return: Boolean, whether everything in this job group has finished
        """
        # Handle edge cases of failed simulations - if we get one FailedSimulation, we declare the group is done,
        # and return a FailedSimulation object as the average
        if self.failed:
            # JobGroup already finished when a previous failed simulation came in.
            return False
        if isinstance(res, FailedSimulation):
            self.failed = res
            return True

        if res.name not in self.subjob_ids:
            raise ValueError('Job group %s received unwanted result %s' % (self.job_id, res.name))
        self.result_list.append(res)
        return len(self.result_list) == len(self.subjob_ids)

    def average_results(self):
        """
        To be called after all results are in for this group.
        Averages the results and returns a new Result object containing the averages

        :return: New Result object with the job_id of this JobGroup and the averaged Data as the simdata
        """
        if self.failed:
            self.failed.name = self.job_id
            return self.failed

        # Iterate through the models and suffixes in the simdata strucutre, and calculate the average for each
        # Data object it contains
        avedata = dict()
        for m in self.result_list[0].simdata:
            avedata[m] = dict()
            for suf in self.result_list[0].simdata[m]:
                avedata[m][suf] = Data.average([r.simdata[m][suf] for r in self.result_list])
        return Result(self.result_list[0].pset, avedata, self.job_id)


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
        logging.debug('Instantiating Trajectory object')
        self.trajectory = Trajectory(config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0
        self.job_group_dir = dict()

        logging.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        # Store a list of all Model objects. Change this as needed for compatibility with other parts
        logging.debug('Initializing models')
        self.model_list = self._initialize_models()

        # Generate a list of variable names
        self.variables = self.config.variables

        # Set the space (log or regular) in which each variable moves, as well as the box constraints on the variable.
        # Currently, this is set based on what distribution the variable is initialized with, but these could be made
        # into a separate, custom options
        logging.debug('Evaluating variable space')
        self.variable_space = dict()  # Contains tuples (space, min_value, max_value)
        for v in self.config.variables_specs:
            if v[1] == 'random_var':
                self.variable_space[v[0]] = ('regular', v[2], v[3])
            elif v[1] == 'normrandom_var' or v[1]=='var':
                self.variable_space[v[0]] = ('regular', 0., np.inf)
            elif v[1] == 'lognormrandom_var' or v[1] == 'logvar':
                self.variable_space[v[0]] = ('log', 0., np.inf)  # Questionable if this is the behavior we want.
            elif v[1] == 'loguniform_var':
                self.variable_space[v[0]] = ('log', v[2], v[3])
            elif v[1] == 'static_list_var':
                self.variable_space[v[0]] = ('static', )  # Todo: what is the actual way to mutate this type of param?
            else:
                logging.info('Variable type not recognized... exiting')
                print0('Error: Unrecognized variable type: %s\nQuitting.' % v[1])
                exit()

    def _initialize_models(self):
        """
        Checks initial BNGLModel instances from the Configuration object for models that
        can be reinstantiated as NetModel instances

        :return: list of Model instances
        """
        home_dir = os.getcwd()
        os.chdir(self.config.config['output_dir'])  # requires creation of this directory prior to function call
        logging.debug('Copying list of models')
        init_model_list = copy.deepcopy(list(self.config.models.values()))  # keeps Configuration object unchanged
        final_model_list = []
        init_dir = os.getcwd() + '/Initialize'

        for m in init_model_list:
            if m.generates_network:
                logging.debug('Model %s requires network generation' % m.name)

                if not os.path.isdir(init_dir):
                    if not os.path.isdir(init_dir):
                        logging.debug('Creating initialization directory: %s' % init_dir)
                        os.mkdir(init_dir)
                    os.chdir(init_dir)

                gnm_name = '%s_gen_net' % m.name
                m.save(gnm_name, gen_only=True)
                gn_cmd = "%s %s.bngl" % (self.config.config['bng_command'], gnm_name)
                try:
                    with open('%s.log' % gnm_name, 'w') as lf:
                        run(gn_cmd, shell=True, check=True, stderr=STDOUT, stdout=lf, timeout=self.config.config['wall_time_gen'])
                except CalledProcessError as c:
                    logging.error("Command %s failed in directory %s" % (gn_cmd, os.getcwd()))
                    logging.error(c.stdout)
                    print0('Error: Initial network generation failed for model %s... see BioNetGen error log at '
                           '%s/%s.log' % (m.name, os.getcwd(), gnm_name))
                    exit(1)
                except TimeoutExpired:
                    logging.debug("Network generation exceeded %d seconds... exiting" % self.config.config['wall_time_gen'])
                    print0("Network generation took too long.  Increase 'wall_time_gen' configuration parameter")
                    exit(1)
                except Exception as e:
                    tb = ''.join(traceback.format_list(traceback.extract_tb(sys.exc_info())))
                    logging.debug("Other exception occurred:\n%s" % tb)
                    print0("Unknown error occurred during network generation, see log... exiting")
                    exit(1)
                finally:
                    os.chdir(home_dir)

                logging.info('Output for network generation of model %s logged in %s/%s.log' % (m.name, init_dir, gnm_name))
                final_model_list.append(NetModel(m.name, m.actions, m.suffixes, nf=init_dir + '/' + gnm_name + '.net'))
            else:
                logging.info('Model %s does not require network generation' % m.name)
                final_model_list.append(m)
        os.chdir(home_dir)
        return final_model_list

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        Algorithm subclasses optionally may set the .name field of the PSet objects to give a meaningful unique
        identifier such as 'gen0ind42'. If so, they MUST BE UNIQUE, as this determines the folder name.
        Uniqueness will not be checked elsewhere.

        :return: list of PSets
        """
        raise NotImplementedError("Subclasses must implement start_run()")

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: result from the completed simulation
        :type res: Result
        :return: List of PSet(s) to be run next or 'STOP' string.
        """
        raise NotImplementedError("Subclasses must implement got_result()")

    def add_to_trajectory(self, res):
        """
        Evaluates the objective function for a Result, and adds the information from the Result to the Trajectory
        instance"""
        score = self.objective.evaluate_multiple(res.simdata, self.exp_data)
        res.score = score
        logging.info('Adding Result %s to Trajectory with score %.4f' % (res.name, score))
        self.trajectory.add(res.pset, score, res.name)

    def random_pset(self):
        """
        Generates a random PSet based on the distributions and bounds for each parameter specified in the configuration

        :return:
        """
        # TODO CONFIRM THIS IS REDUNDANT WITH CODE IN __INIT__
        logging.debug("Generating a randomly distributed PSet")
        param_dict = dict()
        for (name, type, val1, val2) in self.config.variables_specs:
            if type == 'random_var':
                param_dict[name] = np.random.uniform(val1, val2)
            elif type == 'normrandom_var':
                param_dict[name] = max(np.random.normal(val1, val2), self.variable_space[name][1])
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
        logging.debug("Generating PSets using Latin hypercube sampling")
        # Generate latin hypercube of dimension = number of uniformly distributed variables.
        num_uniform_vars = len([x for x in self.config.variables_specs
                               if x[1] == 'random_var' or x[1] == 'loguniform_var'])
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
                    param_dict[name] = 10. ** (np.log10(val1) + row[rowindex]*(np.log10(val2)-np.log10(val1)))
                    rowindex += 1
                elif type == 'lognormrandom_var':
                    param_dict[name] = 10. ** np.random.normal(val1, val2)
                elif type == 'normrandom_var':
                    param_dict[name] = max(np.random.normal(val1, val2), self.variable_space[name][1])
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
        If smoothing is turned on, makes n identical Jobs and a JobGroup

        :param params:
        :type params: PSet
        :return: list of Jobs (of length equal to smoothing setting)
        """
        if params.name:
            job_id = params.name
        else:
            self.job_id_counter += 1
            job_id = 'sim_%i' % self.job_id_counter
        logging.debug('Creating Job %s' % job_id)
        if self.config.config['smoothing'] == 1:
            # Create a single job
            return [Job(self.model_list, params, job_id, self.config.config['bng_command'],
                       self.config.config['output_dir']+'/Simulations/', self.config.config['wall_time_sim'])]
        else:
            # Create multiple identical Jobs for use with smoothing
            newjobs = []
            newnames = []
            for i in range(self.config.config['smoothing']):
                thisname = '%s_rep%i' % (job_id, i)
                newnames.append(thisname)
                newjobs.append(Job(self.model_list, params, thisname, self.config.config['bng_command'],
                       self.config.config['output_dir']+'/Simulations/', self.config.config['wall_time_sim']))
            new_group = JobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs

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
        logging.info('Outputting results to file %s' % filepath)
        self.trajectory.write_to_file(filepath)

        # If the user has asked for fewer output files, each time we're here, move the new file to
        # Results/sorted_params.txt, overwriting the previous one.
        if self.config.config['delete_old_files'] == 1:
            logging.debug("Overwriting previous 'sorted_params.txt'")
            noname_filepath = '%s/Results/sorted_params.txt' % self.config.config['output_dir']
            if os.path.isfile(noname_filepath):
                os.remove(noname_filepath)
            os.rename(filepath, noname_filepath)

    def run(self):
        """Main loop for executing the algorithm"""

        logging.debug('Initializing dask Client object')
        if 'scheduler_address' in self.config.config:
            if 'parallel_count' in self.config.config:
                print1("Warning: 'parallel_count' option is not supported when you have also specified "
                       "'scheduler_address'. I will just use all of the workers in your scheduler.")
            client = Client(self.config.config['scheduler_address'])
        else:
            if 'parallel_count' in self.config.config:
                lc = LocalCluster(n_workers=self.config.config['parallel_count'], threads_per_worker=1)
                client = Client(lc)
            else:
                client = Client()
        logging.debug('Generating initial parameter sets')
        psets = self.start_run()
        jobs = []
        for p in psets:
            jobs += self.make_job(p)
        logging.info('Submitting initial set of %d Jobs' % len(jobs))
        futures = [client.submit(job.run_simulation) for job in jobs]
        pending = set(futures)
        pool = as_completed(futures, with_results=True)
        while True:
            f, res = next(pool)
            # Handle if this result is one of multiple instances for smoothing
            if self.config.config['smoothing'] > 1:
                group = self.job_group_dir.pop(res.name)
                done = group.job_finished(res)
                if not done:
                    continue
                res = group.average_results()
            if isinstance(res, FailedSimulation):
                tb = '\n'+res.traceback if res.fail_type == 2 else ''
                logging.debug('Job %s failed with code %d%s' % (res.name, res.fail_type, tb))
                print1('Job %s failed' % res.name)
            else:
                logging.debug('Job %s complete' % res.name)
            pending.remove(f)
            self.add_to_trajectory(res)
            response = self.got_result(res)
            if response == 'STOP':
                logging.info("Stop criterion satisfied")
                print1('Stop criterion satisfied')
                break
            else:
                new_jobs = []
                for ps in response:
                    new_jobs += self.make_job(ps)
                logging.debug('Submitting %d new Jobs' % len(new_jobs))
                new_futures = [client.submit(j.run_simulation) for j in new_jobs]
                pending.update(new_futures)
                pool.update(new_futures)
        logging.info("Cancelling %d pending jobs" % len(pending))
        client.cancel(list(pending))
        client.close()
        self.output_results('final')

        # Copy the best simulations into the results folder
        best_name = self.trajectory.best_fit_name()
        best_pset = self.trajectory.best_fit()
        logging.info('Copying simulation results from best fit parameter set to Results/ folder')
        for m in self.config.models:
            this_model = self.config.models[m]
            to_save = this_model.copy_with_param_set(best_pset)
            to_save.save('%s/Results/%s_%s' % (self.config.config['output_dir'], to_save.name, best_name), gen_only=False)
            for simtype, suf in this_model.suffixes:
                if simtype == 'simulate':
                    ext = 'gdat'
                else:  # parameter_scan
                    ext = 'scan'
                if self.config.config['smoothing'] > 1:
                    best_name = best_name + '_rep0'  # Look for one specific replicate of the data
                try:
                    shutil.copy('%s/Simulations/%s/%s_%s_%s.%s' %
                                (self.config.config['output_dir'], best_name, m, best_name, suf, ext),
                                '%s/Results' % self.config.config['output_dir'])
                except FileNotFoundError:
                    logging.error('Cannot find files corresponding to best fit parameter set... exiting')
                    print0('Could not find your best fit gdat file. This could happen if all of the simulations in your'
                          '\nrun failed, or if that gdat file was somehow deleted during the run.')
                    exit()

        logging.info("Fitting complete")


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
        print2('Running Particle Swarm Optimization with %i particles for %i total simulations' %
               (self.num_particles, self.max_evals))

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
            if (self.num_evals / self.num_particles) % 10 == 0:
                print1('Completed %i of %i simulations' % (self.num_evals, self.max_evals))
            else:
                print2('Completed %i of %i simulations' % (self.num_evals, self.max_evals))
            print2('Current best score: %d' % self.global_best[1])
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
        self.stop_tolerance = config.config['stop_tolerance']

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
        if self.num_islands == 1:
            print2('Running Differential Evolution with population size %i for up to %i iterations' %
                   (self.num_per_island, self.max_iterations))
        else:
            print2('Running asynchronous Differential Evolution with %i islands of %i individuals each, '
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
                    logging.debug('Island %i just set up the migration.' % island)

                # Send the required PSets to migration_transit
                for j in self.migration_indices[migration_num]:
                    self.migration_transit[migration_num][island].append((self.individuals[island][j],
                                                                          self.fitnesses[island][j]))
                # Tell other islands that this one is ready for this migration.
                self.migration_ready[island] = migration_num

            if self.migration_done[island] < min(self.migration_ready):
                # This island performs a migration
                logging.debug('Island %i is migrating!' % island)
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
            if np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance:
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

        # Iterate through parameters; decide whether to mutate or leave the same.
        new_pset_dict = dict()
        for p in base.keys():
            if np.random.random() < self.mutation_rate:
                new_pset_dict[p] = self.add(base, p, self.mutation_rate * self.diff(others[0], others[1], p))
            else:
                new_pset_dict[p] = base[p]

        return PSet(new_pset_dict)


class ScatterSearch(Algorithm):
    """
    Implements ScatterSearch as described in the introduction of Penas et al 2017 (but not the fancy parallelized
    version from that paper).
    Uses the individual combination method described in Egea et al 2009

    """

    def __init__(self, config): #variables, popsize, maxiters, saveevery):

        super(ScatterSearch, self).__init__(config)

        self.popsize = config.config['population_size']
        self.maxiters = config.config['max_iterations']
        if 'reserve_size' in config.config:
            self.reserve_size = config.config['reserve_size']
        else:
            self.reserve_size = self.maxiters
        if 'init_size' in config.config:
            self.init_size = config.config['init_size']
            if self.init_size < self.popsize:
                logging.warning('init_size less than population_size. Setting it equal to population_size.')
                print1("Scatter search parameter 'init_size' cannot be less than 'population_size'. "
                       "Automatically setting it equal to population_size.")
                self.init_size = self.popsize
        else:
            self.init_size = 10*len(self.variables)
            if self.init_size < self.popsize:
                logging.warning('init_size less than population_size. Setting it equal to population_size.')
                self.init_size = self.popsize

        self.local_min_limit = config.config['local_min_limit']

        self.pending = dict() # {pendingPSet: parentPSet}
        self.received = dict() # {parentPSet: [(donependingPSet, score)]
        self.refs = [] # (refPset, score)
        self.stuckcounter = dict()
        self.iteration = 0
        self.local_mins = [] # (Pset, score) pairs that were stuck for 5 gens, and so replaced.
        self.reserve = []

    def start_run(self):
        print2('Running Scatter Search with population size %i (%i simulations per iteration) for %i iterations' %
               (self.popsize, self.popsize * (self.popsize - 1), self.maxiters))
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
                    best_child = min(self.received[self.refs[i][0]], key=lambda x: x[1])
                    if best_child[1] < self.refs[i][1]:
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
            logging.info('Iteration %i' % self.iteration)
            if self.iteration % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration, self.maxiters))
            else:
                print2('Completed iteration %i of %i' % (self.iteration, self.maxiters))
            print2('Current scores: ' + str([x[1] for x in self.refs]))
            print2('Best archived scores: ' + str([x[1] for x in self.local_mins]))

            if self.iteration % self.config.config['output_every'] == 0:
                self.output_results()

            self.iteration += 1
            if self.iteration == self.maxiters:
                return 'STOP'

            # 3) Do the combination antics to generate new candidates
            query_psets = []
            for pi in range(self.popsize): # parent index
                for hi in range(self.popsize): # helper index
                    if pi == hi:
                        continue
                    newdict = dict()
                    for v in self.variables:
                        # d = (self.refs[hi][0][v] - self.refs[pi][0][v]) / 2.
                        d = self.diff(self.refs[hi][0], self.refs[pi][0], v)
                        alpha = np.sign(hi-pi)
                        beta = (abs(hi-pi) - 1) / (self.popsize - 2)
                        # c1 = self.refs[pi][0][v] - d*(1 + alpha*beta)
                        # c2 = self.refs[pi][0][v] + d*(1 - alpha*beta)
                        # newval = np.random.uniform(c1, c2)
                        # newdict[v] = max(min(newval, var[2]), var[1])
                        newdict[v] = self.rand_uniform_offset(
                            self.refs[pi][0], v, -d*(1 + alpha*beta), d*(1 - alpha * beta))
                    newpset = PSet(newdict, allow_negative=True)
                    # Check to avoid duplicate PSets. If duplicate, don't have to try again because SS doesn't really
                    # care about the number of PSets queried.
                    if newpset not in self.pending:
                        newpset.name = 'iter%ip%ih%i' % (self.iteration, pi, hi)
                        query_psets.append(newpset)
                        self.pending[newpset] = self.refs[pi][0]
            self.received = {r[0]: [] for r in self.refs}
            return query_psets

        else:
            return []

    def rand_uniform_offset(self, paramset, param, lower, upper):
        """
        Performs a particular random sampling required for scatter search,
        taking into account
        1) Whether this parameter is to be moved in regular or log space
        2) Box constraints on the parameter
        This could not be achieved with the Algorithm add and diff methods.

        :param paramset: PSet containing the initial value of the target param
        :type paramset: PSet
        :param param: name of the parameter
        :type param: str
        :param lower: The lower bound for the random pick is this + the current value of the param. You probably want to
        pass a negative value here. This is assumed to be in log space if the param is in log space
        :type lower: float
        :param upper: The upper bound for the random pick is this + the current value of the param.
        This is assumed to be in log space if the param is in log space
        :return: The chosen random value
        """
        if self.variable_space[param][0] == 'regular':
            lb = paramset[param] + lower
            ub = paramset[param] + upper
            pick = np.random.uniform(lb, ub)
            return max(self.variable_space[param][1], min(self.variable_space[param][2], pick))
        elif self.variable_space[param][0] == 'log':
            lb = np.log10(paramset[param]) + lower
            ub = np.log10(paramset[param]) + upper
            pick = np.random.uniform(lb, ub)
            return max(self.variable_space[param][1], min(self.variable_space[param][2], 10. ** pick))
        elif self.variable_space[param][0] == 'static':
            return paramset[param]
        else:
            raise RuntimeError('Unrecognized variable space type: %s' % self.variable_space[param][0])


class BayesAlgorithm(Algorithm):

    """
    Implements a Bayesian Markov chain Monte Carlo simulation.

    This is essentially a non-parallel algorithm, but here, we run n instances in parallel, and pool all results.
    This will give you a best fit (which is maybe not great), but more importantly, generates an extra result file
    that gives the probability distribution of each variable.
    This distribution depends on the prior, which is specified according to the variable initialization rules.

    """

    def __init__(self, config): #expdata, objective, priorfile, gamma=0.1):
        super(BayesAlgorithm, self).__init__(config)
        self.step_size = config.config['step_size']
        self.num_parallel = config.config['population_size']
        self.max_iterations = config.config['max_iterations']
        self.burn_in = config.config['burn_in'] # todo: 'auto' option
        self.sample_every = config.config['sample_every']
        self.output_hist_every = config.config['output_hist_every']
        # A list of the % credible intervals to save, eg [68. 95]
        self.credible_intervals = config.config['credible_intervals']
        self.num_bins = config.config['hist_bins']

        self.prior = None
        self.load_priors()

        self.current_pset = None # List of n PSets corresponding to the n independent runs
        self.ln_current_P = None # List of n probabilities of those n PSets.
        self.iteration = [0]*self.num_parallel # Iteration number that each PSet is on

        self.samples_file = None # Initialize later.

    def load_priors(self):
        """Builds the data structures for the priors, based on the variables specified in the config."""
        self.prior = dict()  # {variable: ('n', mean, sigma), variable2: ('b', min, max)}
        # n for normally distributed priors, b for box priors (which are weird, but should be allowed)
        for (name, type, val1, val2) in self.config.variables_specs:
            if type in ('lognormrandom_var', 'normrandom_var'):
                self.prior[name] = ('n', val1, val2)
            elif type in ('random_var', 'loguniform_var'):
                self.prior[name] = ('b', val1, val2)
            elif type in ('loguniform_var'):
                self.prior[name] = ('b', np.log10(val1), np.log10(val2))
            else:
                raise NotImplementedError('Bayesian MCMC cannot handle variable type %s' % type)


    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        :return: list of PSets
        """
        print2('Running Markov Chain Monte Carlo on %i independent replicates in parallel, for %i iterations each.' %
               (self.num_parallel, self.max_iterations))
        print2('Statistical samples will be recorded every %i iterations, after an initial %i-iteration burn-in period'
               % (self.sample_every, self.burn_in))

        # Set up the output files
        # Cant do this in the constructor because that happens before the output folder is potentially overwritten.
        self.samples_file = self.config.config['output_dir'] + '/Results/samples.txt'
        with open(self.samples_file, 'w') as f:
            f.write('# Name\tLn_probability\t')
            for v in self.variables:
                f.write(v + '\t')
            f.write('\n')
        os.makedirs(self.config.config['output_dir'] + '/Results/Histograms/', exist_ok=True)


        if self.config.config['initialization'] == 'lh':
            first_pset = self.random_latin_hypercube_psets(self.num_parallel)
        else:
            first_pset = [self.random_pset() for i in range(self.num_parallel)]

        self.ln_current_P = [np.nan]*self.num_parallel  # Forces accept on the first run
        self.current_pset = [None]*self.num_parallel
        for i in range(len(first_pset)):
            first_pset[i].name = 'iter0run%i' % i

        return first_pset

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

        # Figure out which parallel run this is from based on the .name field.
        m = re.search('(?<=run)\d+',pset.name)
        index = int(m.group(0))

        # Calculate the acceptance probability
        lnprior = self.ln_prior(pset) # Need something clever for box constraints
        lnlikelihood = -score
        # lnlikelihood = -self.objective.evaluate_multiple(simdata, self.exp_data)

        # Because the P's are so small to start, we express posterior, p_accept, and current_P in ln space
        lnposterior = lnprior + lnlikelihood

        ln_p_accept = min(0., lnposterior - self.ln_current_P[index])
        # print("lnprior:"+str(lnprior))
        # print("lnlikelihood:" + str(lnlikelihood))
        # print("lnposterior:" + str(lnposterior))
        # print("current_P" + str(self.current_P))
        # print("ln_p_accept:"+str(ln_p_accept))

        # Decide whether to accept move.

        if np.random.rand() < np.exp(ln_p_accept) or np.isnan(self.ln_current_P[index]):
            # Accept the move, so update our current PSet and P
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior

        # Record the current PSet (clarification: what if failed? Sample old again?)

        # Using either the newly accepted PSet or the old PSet, propose the next PSet.
        proposed_pset = None
        # This part is a loop in case a box constraint makes a move automatically rejected.
        loop_count = 0
        while proposed_pset is None:
            loop_count += 1
            if loop_count == 20:
                logging.warning('Instance %i spent 20 iterations at the same point' % index)
                print1('One of your samples is stuck at the same point for 20+ iterations because it keeps '
                                'hitting box constraints. Consider using looser box constraints or a smaller '
                                'step_size.')
            if loop_count == 1000:
                logging.warning('Instance %i terminated after 1000 iterations at the same point' % index)
                print1('Instance %i was terminated after it spent 1000 iterations stuck at the same point '
                              'because it kept hitting box constraints. Consider using looser box constraints or a '
                              'smaller step_size.' % index)
                self.iteration[index] = self.max_iterations

            proposed_pset = self.choose_new_pset(self.current_pset[index])
            self.iteration[index] += 1
            # Check if it's time to do various things
            if self.iteration[index] > self.burn_in and self.iteration[index] % self.sample_every == 0:
                self.sample_pset(self.current_pset[index], lnposterior)
            if (self.iteration[index] > self.burn_in and self.iteration[index] % self.output_hist_every == 0
               and self.iteration[index] == min(self.iteration)):
                self.update_histograms('_%i' % self.iteration[index])

            if self.iteration[index] == min(self.iteration):
                if self.iteration[index] % self.config.config['output_every'] == 0:
                    self.output_results()
                if self.iteration[index] % 10 == 0:
                    print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                    logging.info('Completed %i iterations' % self.iteration[index])
                else:
                    logging.info('Completed %i iterations' % self.iteration[index])
                print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                print2('Current objective values: ' + str(self.ln_current_P))
            if self.iteration[index] >= self.max_iterations:
                logging.info('Finished replicate number %i' % index)
                print2('Finished replicate number %i' % index)
                if self.iteration[index] == min(self.iteration):
                    self.update_histograms('_final')
                    return 'STOP'
                else:
                    # Others of the parallel runs are still going.
                    # Should *not* stop until they are all done, or we bias the distribution for fast-running simulations.
                    return []

        proposed_pset.name = 'iter%irun%i' % (self.iteration[index], index)

        return [proposed_pset]

    def choose_new_pset(self, oldpset):
        """
        Helper function to perturb the old PSet, generating a new proposed PSet
        If the new PSet fails automatically because it violates box constraints, returns None.

        :param oldpset: The PSet to be changed
        :type oldpset: PSet
        :return: the new PSet
        """

        keys = oldpset.keys()
        delta_vector = {k: np.random.normal() for k in keys}
        delta_vector_magnitude = np.sqrt(sum([x ** 2 for x in delta_vector.values()]))
        delta_vector_normalized = {k: self.step_size * delta_vector[k] / delta_vector_magnitude for k in keys}
        new_dict = dict()
        for k in keys:
            # For box constraints, need special treatment to keep correct statistics
            # If we tried to leave the box, the move automatically fails, we should increment the iteration counter
            # and retry.
            # The same could happen if normrandom_var's try to go below 0
            new_dict[k] = self.add(oldpset, k, delta_vector_normalized[k])
            if new_dict[k] == self.variable_space[k][1] or new_dict[k] == self.variable_space[k][2]:
                logging.debug('Rejected a move because %s moved outside the box constraint' % k)
                return None

        return PSet(new_dict)

    def ln_prior(self, pset):
        """
        Returns the value of the prior distribution for the given parameter set

        :param pset:
        :type pset: PSet
        :return: float value of ln times the prior distribution
        """
        total = 0.
        for v in self.prior:
            (typ, x1, x2) = self.prior[v]
            if typ == 'n':
                # Normal with mean x1 and value x2
                total += -1. / (2. * x2 ** 2.) * (x1 - pset[v])**2.
            else:
                # Uniform from x1 to x2
                if x1 <= pset[v] <= x2:
                    total += -np.log(x2-x1)
                else:
                    logging.warning('Box-constrained parameter %s reached a value outside the box.')
                    total += -np.inf
        return total

    def sample_pset(self, pset, ln_prob):
        """
        Adds this pset to the set of sampled psets for the final distribution.
        :param pset:
        :type pset: PSet
        :param ln_prob - The probability of this PSet to record in the samples file.
        :type ln_prob: float
        """
        with open(self.samples_file, 'a') as f:
            f.write(pset.name+'\t'+str(ln_prob)+'\t')
            for v in self.variables:
                f.write('%f\t' % pset[v])
            f.write('\n')

    def update_histograms(self, file_ext):
        """
        Updates the files that contain histogram points for each variable
        :param file_ext: String to append to the save file names
        :type file_ext: str
        :return:
        """
        # Read the samples file into an array, ignoring the first row (header)
        # and first 2 columns (pset names, probabilities)
        dat_array = np.genfromtxt(self.samples_file, delimiter='\t', dtype=float,
                                  usecols=range(2, len(self.variables)+2))

        # Open the file(s) to save the credible intervals
        cred_files = []
        for i in self.credible_intervals:
            f = open(self.config.config['output_dir']+'/Results/credible%i%s.txt' % (i, file_ext), 'w')
            f.write('# param\tlower_bound\tupper_bound\n')
            cred_files.append(f)

        for i in range(len(self.variables)):
            v = self.variables[i]
            fname = self.config.config['output_dir']+'/Results/Histograms/%s%s.txt' % (v, file_ext)
            # For log-space variables, we want the histogram in log space
            if self.variable_space[v][0] == 'log':
                histdata = np.log10(dat_array[:, i])
                header = 'log10_lower_bound\tlog10_upper_bound\tcount'
            else:
                histdata = dat_array[:, i]
                header = 'lower_bound\tupper_bound\tcount'
            hist, bin_edges = np.histogram(histdata, bins=self.num_bins)
            result_array = np.stack((bin_edges[:-1], bin_edges[1:], hist), axis=-1)
            np.savetxt(fname, result_array, delimiter='\t', header=header)

            sorted_data = sorted(dat_array[:, i])
            for interval, file in zip(self.credible_intervals, cred_files):
                min_index = int(np.round(len(sorted_data) * (1.-(interval/100)) / 2.))
                max_index = int(np.round(len(sorted_data) * (1. - ((1.-(interval/100)) / 2.))))
                file.write('%s\t%f\t%f\n' % (v, sorted_data[min_index], sorted_data[max_index]))


class SimplexAlgorithm(Algorithm):

    """
    Implements a parallelized version of the Simplex local search algorithm, as described in Lee and Wiswall 2007,
    Computational Economics

    """

    def __init__(self, config):
        super(SimplexAlgorithm, self).__init__(config)
        if 'simplex_start_point' not in config.config:
            # We need to set up the initial point ourselfs
            self._parse_start_point()
        if 'simplex_max_iterations' in config.config:
            self.max_iterations = config.config['simplex_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']
        self.start_point = config.config['simplex_start_point']
        self.start_steps = {v[0]: v[3] for v in config.variables_specs}
        self.parallel_count = min(config.config['population_size'], len(self.variables))
        self.iteration = 0
        self.alpha = config.config['simplex_reflection']
        self.gamma = config.config['simplex_expansion']
        self.beta = config.config['simplex_contraction']
        self.tau = config.config['simplex_shrink']

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

    def _parse_start_point(self):
        """
        Called when the start point is not passed in the config (which is when we're doing a pure simplex run,
        as opposed to a refinement at the end of the run)
        Parses the info out of the variable specs, and sets the appropriate PSet into the config.
        """
        start_dict = dict()
        for vinfo in self.config.variables_specs:
            if vinfo[1] == 'var':
                start_dict[vinfo[0]] = vinfo[2]
            elif vinfo[1] == 'logvar':
                start_dict[vinfo[0]] = 10.**vinfo[2]
            else:
                raise RuntimeError('Internal error in SimplexAlgorithm: Encountered variable type %s while trying'
                                   'to parse start point' % vinfo[1])
        start_pset = PSet(start_dict)
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
            new_dict = dict()
            for p in self.variables:
                if p == v:
                    new_dict[p] = self.add(self.start_point, p, self.start_steps[p])
                else:
                    new_dict[p] = self.start_point[p]
            new_pset = PSet(new_dict)
            new_pset.name = 'simplex_init%i' % i
            self.pending[new_pset.name] = i
            i += 1
            init_psets.append(new_pset)
        self.simplex = [None]*len(init_psets)
        self.stages = [-1]*len(init_psets)
        return init_psets

    def got_result(self, res):

        pset = res.pset
        score = res.score
        index = self.pending.pop(pset.name)

        if self.stages[index] == -1:
            # Point is part of initialization
            self.simplex[index] = (score, pset)
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
                new_dict = dict()
                for v in pset.keys():
                    # new_dict[v] = pset[v] + self.gamma * (pset[v] - self.centroids[index][v])
                    new_dict[v] = self.a_plus_b_times_c_minus_d(pset[v], self.gamma, pset[v], self.centroids[index][v],
                                                                v)
                new_pset = PSet(new_dict)
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
                new_dict = dict()
                for v in self.variables:
                    # I think the equation for this in Lee et al p. 178 is wrong; I am instead using the analog to the
                    # equation on p. 176
                    # new_dict[v] = self.centroids[index][v] + self.beta * (a_hat[v] - self.centroids[index][v])
                    new_dict[v] = self.a_plus_b_times_c_minus_d(self.centroids[index][v], self.beta, a_hat[v],
                                                                self.centroids[index][v], v)
                new_pset = PSet(new_dict)
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
                        new_dict = dict()
                        for v in self.simplex[i][1].keys():
                            # new_dict[v] = self.tau * self.simplex[i-1][1][v] + (1 - self.tau) * self.simplex[i][1][v]
                            new_dict[v] = self.ab_plus_cd(self.tau, self.simplex[i-1][1][v], 1 - self.tau,
                                                          self.simplex[i][1][v], v)
                        new_pset = PSet(new_dict)
                        new_pset.name = 'simplex_iter%i_pt%i' % (self.iteration, i)
                        self.pending[new_pset.name] = i - 1
                        new_simplex.append(new_pset)

                    # Prepare for new reinitialization run
                    # We don't need to rescore simplex[0], but the rest of the PSets are new and we do.
                    self.stages = [-1] * len(new_simplex)
                    self.first_points = []
                    self.second_points = []
                    self.simplex = [self.simplex[0]] + ([None] * len(new_simplex))
                    return new_simplex

            ###
            # Set up the next iteration
            # Re-sort the simplex based on the updated objectives
            self.simplex = sorted(self.simplex, key=lambda x: x[0])
            if self.iteration == self.max_iterations:
                return 'STOP' # Extra catch if finish on a rebuild the simplex iteration
            # Find the reflection point for the n worst points
            reflections = []
            self.centroids = []
            # Sum of each param value, to help take the reflections
            sums = self.get_sums() # Returns in log space for log variables
            for ai in range(self.parallel_count):
                a = self.simplex[-ai-1][1]
                new_dict = dict()
                this_centroid = dict()
                for v in a.keys():
                    if self.variable_space[v][0] == 'log':
                        # Calc centroid in regular space.
                        centroid = 10. ** ((sums[v] - np.log10(a[v])) / (len(self.simplex) - 1))
                    else:
                        centroid = (sums[v] - a[v]) / (len(self.simplex) - 1)
                    this_centroid[v] = centroid
                    # new_dict[v] = centroid + self.alpha * (centroid - a[v])
                    new_dict[v] = self.a_plus_b_times_c_minus_d(centroid, self.alpha, centroid, a[v], v)
                self.centroids.append(this_centroid)
                new_pset = PSet(new_dict)
                new_pset.name = 'simplex_iter%i_pt%i' % (self.iteration, ai)
                reflections.append(new_pset)
                self.pending[new_pset.name] = ai

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
        for p in self.simplex[0][1].keys():
            if self.variable_space[p][0] == 'regular':
                sums[p] = sum(point[1][p] for point in self.simplex)
            else:
                sums[p] = sum(np.log10(point[1][p]) for point in self.simplex)
        return sums

    def a_plus_b_times_c_minus_d(self, a, b, c, d, v):
        """
        Performs the calculation a + b*(c-d), where a, c, and d are assumed to be in log space if v is in log space,
        and the final result respects the box constraints on v.

        :param a:
        :param b:
        :param c:
        :param d:
        :param v:
        :return:
        """

        if self.variable_space[v][0] == 'log':
            result = 10 ** (np.log10(a) + b*(np.log10(c) - np.log10(d)))
        else:
            result = a + b*(c-d)
        return max(self.variable_space[v][1], min(self.variable_space[v][2], result))

    def ab_plus_cd(self, a, b, c, d, v):
        """
        Performs the calculation ab + cd where b and d are assumed to be in log space if v is in log space,
        and the final result respects the box constraints on v
        :param a:
        :param b:
        :param c:
        :param d:
        :param v:
        :return:
        """
        if self.variable_space[v][0] == 'log':
            result = 10 ** (a * np.log10(b) + c*np.log10(d))
        else:
            result = a * b + c * d
        return max(self.variable_space[v][1], min(self.variable_space[v][2], result))





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
