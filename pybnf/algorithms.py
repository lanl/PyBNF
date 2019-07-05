"""Contains the Algorithm class and subclasses as well as support classes and functions for running simulations"""


from distributed import as_completed
from subprocess import run
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import STDOUT

from .data import Data
from .pset import PSet
from .pset import Trajectory

from .pset import NetModel, BNGLModel, SbmlModelNoTimeout
from .pset import OutOfBoundsException
from .pset import FailedSimulationError
from .printing import print0, print1, print2, PybnfError
from .objective import ObjectiveCalculator, ConstraintCounter

import logging
import numpy as np
import os
import re
import shutil
import copy
import sys
import traceback
import pickle
from glob import glob
from tornado import gen
from distributed.client import _wait
from concurrent.futures._base import CancelledError


logger = logging.getLogger(__name__)


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

    def normalize(self, settings):
        """
        Normalizes the Data object in this result, according to settings
        :param settings: Config value for 'normalization': a string representing the normalization type, a dict mapping
        exp files to normalization type, or None
        :return:
        """
        if settings is None:
            return

        for m in self.simdata:
            for suff in self.simdata[m]:
                if type(settings) == str:
                    self.simdata[m][suff].normalize(settings)
                elif suff in settings:
                    self.simdata[m][suff].normalize(settings[suff])

    def postprocess_data(self, settings):
        """
        Postprocess the Data objects in this result with a user-defined Python script
        :param settings: A dict that maps a tuple (model, suffix) to a Python filename to load.
        That file is expected to contain the definition for the function postprocess(data),
        which takes a Data object and returns a processed data object
        :return: None
        """
        for m, suff in settings:
            rawdata = self.simdata[m][suff]
            # This could generate all kinds of errors if the user's script is bad. Whatever happens, it's caught
            # by the caller of postprocess_data()
            # exec(settings[m][suff])
            # noinspection PyUnresolvedReferences
            # self.simdata[m][suff] = postprocess(rawdata)

            # Cleaner attempt - follows good practice and is probably faster, but makes it hard for the user to create
            # a new Data object if they want to do that.
            # However, they can do that by `dataclass = data.__class__` `newdata = dataclass()`
            # Import the user-specified script as a module
            import importlib.util
            spec = importlib.util.spec_from_file_location("postprocessor", settings[m, suff])
            postproc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(postproc)
            # Now postproc is the user-defined Python module

            self.simdata[m][suff] = postproc.postprocess(rawdata)

    def add_result(self, other):
        """
        Add simulation data of other models from another Result object into this Result object
        :param other: The other Result object
        :return:
        """
        self.simdata.update(other.simdata)


class FailedSimulation(Result):
    def __init__(self, paramset, name, fail_type, einfo=tuple([None, None, None])):
        """
        Instantiates a FailedSimulation

        :param paramset:
        :param log:
        :param name:
        :param fail_type: 0 - Exceeded walltime, 1 - Other crash
        :type fail_type: int
        :param einfo:
        :type einfo: tuple
        """
        super(FailedSimulation, self).__init__(paramset, None, name)
        self.fail_type = fail_type
        self.failed = True
        self.traceback = ''.join(traceback.format_exception(*einfo))

    def normalize(self, settings):
        return

    def postprocess_data(self, settings):
        return


def run_job(j, debug=False, failed_logs_dir=''):
    """
    Runs the Job j.
    This function is passed to Dask instead of j.run_simulation because if you pass j.run_simulation, Dask leaks memory
    associated with j.
    """
    try:
        return j.run_simulation(debug, failed_logs_dir)
    except RuntimeError as e:
        # Catch the error for running out of threads here - it's the only place outside dask where we can catch it.
        if e.args[0] == "can't start new thread":
            logger.error("Reached thread limit - can't start new thread")
            print0('Too many threads! See "Troubleshooting" in the documentation for how to deal with this problem')
            return FailedSimulation(j.params, j.job_id, 1)
        else:
            raise


class Job:
    """
    Container for information necessary to perform a single evaluation in the fitting algorithm
    """

    # Seeing these logs for cluster-based fitting requires configuring dask to log to the
    # "pybnf.algorithms.job" logger
    jlogger = logging.getLogger('pybnf.algorithms.job')

    def __init__(self, models, params, job_id, output_dir, timeout, calc_future, norm_settings, postproc_settings,
                 delete_folder=False):
        """
        Instantiates a Job

        :param models: The models to evaluate
        :type models: list of Model instances
        :param params: The parameter set with which to evaluate the model
        :type params: PSet
        :param job_id: Job identification; also the folder name that the job gets saved to
        :type job_id: str
        :param output_dir path to the directory where I should create my simulation folder
        :type output_dir: str
        :param calc_future: Future for an ObjectiveCalculator containing the objective function and experimental data,
        which we can use to calculate the objective value.
        :type calc_future: Future
        :param norm_settings: Config value for 'normalization': a string representing the normalization type, a dict
        mapping exp files to normalization type, or None
        :type norm_settings: Union[str, dict, NoneType]
        :param postproc_settings: dict mapping (model, suffix) tuples to the path of a Python postprocessing file to
        run on the result.
        :param delete_folder: If True, delete the folder and files created after the simulation runs
        :type delete_folder: bool
        """
        self.models = models
        self.params = params
        self.job_id = job_id
        self.calc_future = calc_future
        self.norm_settings = norm_settings
        self.postproc_settings = postproc_settings
        # Whether to show warnings about missing data if the job includes an objective evaluation. Toggle this after
        # construction if needed.
        self.show_warnings = False
        self.home_dir = os.getcwd()  # This is safe because it is called from the scheduler, not the workers.
        # Force absolute paths for bngcommand and output_dir, because workers do not get the relative path info.
        if output_dir[0] == '/':
            self.output_dir = output_dir
        else:
            self.output_dir = self.home_dir + '/' + output_dir
        self.timeout = timeout

        # Folder where we save the model files and outputs.
        self.folder = '%s/%s' % (self.output_dir, self.job_id)
        self.delete_folder = delete_folder

    def _name_with_id(self, model):
        return '%s_%s' % (model.name, self.job_id)

    def _run_models(self):
        ds = {}
        for model in self.models:
            model_file_prefix = self._name_with_id(model)
            model_with_params = model.copy_with_param_set(self.params)
            ds[model.name] = model_with_params.execute(self.folder, model_file_prefix, self.timeout)
        return ds

    def _copy_log_files(self, failed_logs_dir):
        if failed_logs_dir == '':
            self.jlogger.error('Cannot save log files without specified directory')
            return
        for m in self.models:
            lf = '%s/%s.log' % (self.folder, self._name_with_id(m))
            if os.path.isfile(lf):
                self.jlogger.debug('Copying log file %s' % lf)
                shutil.copy(lf, failed_logs_dir)

    def run_simulation(self, debug=False, failed_logs_dir=''):
        """Runs the simulation and reads in the result"""

        # Force absolute path for failed_logs_dir
        if len(failed_logs_dir) > 0 and failed_logs_dir[0] != '/':
            failed_logs_dir = self.home_dir + '/' + failed_logs_dir

        # The check here is in case dask decides to run the same job twice, both of them can complete.
        made_folder = False
        failures = 0
        while not made_folder:
            try:
                os.mkdir(self.folder)
                self.jlogger.debug('Created folder %s for simulation' % self.folder)
                made_folder = True
            except OSError:
                self.jlogger.warning('Failed to create folder %s, trying again.' % self.folder)
                failures += 1
                self.folder = '%s/%s_rerun%i' % (self.output_dir, self.job_id, failures)
                if failures > 1000:
                    self.jlogger.error('Job %s failed because it was unable to write to the Simulations folder' %
                                       self.job_id)
                    return FailedSimulation(self.params, self.job_id, 1)
        try:
            simdata = self._run_models()
            res = Result(self.params, simdata, self.job_id)
        except (CalledProcessError, FailedSimulationError):
            if debug:
                self._copy_log_files(failed_logs_dir)
            res = FailedSimulation(self.params, self.job_id, 1)
        except TimeoutExpired:
            if debug:
                self._copy_log_files(failed_logs_dir)
            res = FailedSimulation(self.params, self.job_id, 0)
        except FileNotFoundError:
            self.jlogger.exception('File not found during job %s. This should only happen if the fitting '
                                   'is already done.' % self.job_id)
            res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())
        except Exception:
            if debug:
                self._copy_log_files(failed_logs_dir)
            print1('A simulation failed with an unknown error. See the log for details, and consider reporting this '
                   'as a bug.')
            self.jlogger.exception('Unknown error during job %s' % self.job_id)
            res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())
        else:
            if self.calc_future is not None:
                res.normalize(self.norm_settings)
                try:
                    res.postprocess_data(self.postproc_settings)
                except Exception:
                    self.jlogger.exception('User-defined post-processing script failed')
                    traceback.print_exc()
                    print0('User-defined post-processing script failed')
                    res.score = np.inf
                else:
                    res.score = self.calc_future.result().evaluate_objective(res.simdata, show_warnings=self.show_warnings)
                    if res.score is None:
                        res.score = np.inf
                        logger.warning('Simulation corresponding to Result %s contained NaNs or Infs' % res.name)
                        logger.warning('Discarding Result %s as having an infinite objective function value' % res.name)
                res.simdata = None
        if self.delete_folder:
            if os.name == 'nt':  # Windows
                try:
                    shutil.rmtree(self.folder)
                    self.jlogger.debug('Removed folder %s' % self.folder)
                except OSError:
                    self.jlogger.error('Failed to remove folder %s.' % self.folder)
            else:
                try:
                    run(['rm', '-rf', self.folder], check=True, timeout=1800)
                    self.jlogger.debug('Removed folder %s' % self.folder)
                except (CalledProcessError, TimeoutExpired):
                    self.jlogger.error('Failed to remove folder %s.' % self.folder)

        return res


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


class MultimodelJobGroup(JobGroup):
    """
    A JobGroup to handle model-level parallelism
    """

    def average_results(self):
        """
        To be called after all results are in for this group.
        Combines all results from the submodels into a single Result object
        :return:
        """
        if self.failed:
            self.failed.name = self.job_id
            return self.failed

        # Merge all models into a single Result object
        final_result = Result(self.result_list[0].pset, dict(), self.job_id)
        for res in self.result_list:
            final_result.add_result(res)
        return final_result


class custom_as_completed(as_completed):
    """
    Subclass created to modify a section of dask.distributed code
    By using this subclass instead of as_completed, if you get an exception in a job,
    that exception is returned as the result, instead of the job disappearing.
    """
    @gen.coroutine
    def track_future(self, future):
        try:
            yield _wait(future)
        except CancelledError:
            pass
        if self.with_results:
            try:
                result = yield future._result(raiseit=True)
            except Exception as e:
                result = DaskError(e, traceback.format_exc())
        with self.lock:
            self.futures[future] -= 1
            if not self.futures[future]:
                del self.futures[future]
            if self.with_results:
                self.queue.put_nowait((future, result))
            else:
                self.queue.put_nowait(future)
            self._notify()


class DaskError:
    """
    Class representing the result of a job that failed due to a raised exception
    """
    def __init__(self, error, tb):
        self.error = error
        self.traceback = tb


class Algorithm(object):
    """
    A superclass containing the structures common to all metaheuristic and MCMC-based algorithms
    defined in this software suite
    """

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
        logger.debug('Instantiating Trajectory object')
        self.trajectory = Trajectory(self.config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0
        self.job_group_dir = dict()
        self.fail_count = 0
        self.success_count = 0
        self.max_iterations = config.config['max_iterations']

        logger.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        if self.config.config['simulation_dir']:
            self.sim_dir = self.config.config['simulation_dir'] + '/Simulations'
        else:
            self.sim_dir = self.config.config['output_dir'] + '/Simulations'
        self.res_dir = self.config.config['output_dir'] + '/Results'
        self.failed_logs_dir = self.config.config['output_dir'] + '/FailedSimLogs'

        # Generate a list of variable names
        self.variables = self.config.variables

        # Store a list of all Model objects. Change this as needed for compatibility with other parts
        logger.debug('Initializing models')
        self.model_list = self._initialize_models()

        self.bootstrap_number = None
        self.best_fit_obj = None
        self.calc_future = None  # Created during Algorithm.run()
        self.refine = False

    def reset(self, bootstrap):
        """
        Resets the Algorithm, keeping loaded variables and models

        :param bootstrap: The bootstrap number (None if not bootstrapping)
        :type bootstrap: int or None
        :return:
        """
        logger.info('Resetting Algorithm for another run')
        self.trajectory = Trajectory(self.config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0
        self.job_group_dir = dict()
        self.fail_count = 0
        self.success_count = 0

        if bootstrap is not None:
            self.bootstrap_number = bootstrap

            self.sim_dir = self.config.config['output_dir'] + '/Simulations-boot%s' % bootstrap
            self.res_dir = self.config.config['output_dir'] + '/Results-boot%s' % bootstrap
            self.failed_logs_dir = self.config.config['output_dir'] + '/FailedSimLogs-boot%s' % bootstrap
            for boot_dir in (self.sim_dir, self.res_dir, self.failed_logs_dir):
                if os.path.exists(boot_dir):
                    try:
                        shutil.rmtree(boot_dir)
                    except OSError:
                        logger.error('Failed to remove bootstrap directory '+boot_dir)
                os.mkdir(boot_dir)

        self.best_fit_obj = None

    @staticmethod
    def should_pickle(k):
        """
        Checks to see if key 'k' should be included in pickling.  Currently allows all entries in instance dictionary
        except for 'trajectory'

        :param k:
        :return:
        """
        return k not in set(['trajectory', 'calc_future'])

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if self.should_pickle(k)}

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            backup_params = 'sorted_params_backup.txt' if not self.refine else 'sorted_params_refine_backup.txt'
            self.trajectory = Trajectory.load_trajectory('%s/%s' % (self.res_dir, backup_params),
                                                         self.config.variables, self.config.config['num_to_output'])
        except IOError:
            logger.exception('Failed to load trajectory from file')
            print1('Failed to load Results/sorted_params_backup.txt . Still resuming your run, but when I save the '
                   'best fits, it will only be the ones I\'ve seen since resuming.')
            self.trajectory = Trajectory(self.config.config['num_to_output'])

    def _initialize_models(self):
        """
        Checks initial BNGLModel instances from the Configuration object for models that
        can be reinstantiated as NetModel instances

        :return: list of Model instances
        """
        # Todo: Move to config or BNGL model class?
        home_dir = os.getcwd()
        os.chdir(self.config.config['output_dir'])  # requires creation of this directory prior to function call
        logger.debug('Copying list of models')
        init_model_list = copy.deepcopy(list(self.config.models.values()))  # keeps Configuration object unchanged
        final_model_list = []
        init_dir = os.getcwd() + '/Initialize'

        for m in init_model_list:
            if isinstance(m, BNGLModel) and m.generates_network:
                logger.debug('Model %s requires network generation' % m.name)

                if not os.path.isdir(init_dir):
                    logger.debug('Creating initialization directory: %s' % init_dir)
                    os.mkdir(init_dir)
                os.chdir(init_dir)

                gnm_name = '%s_gen_net' % m.name
                default_pset = PSet([var.set_value(var.default_value) for var in self.variables])
                m.save(gnm_name, gen_only=True, pset=default_pset)
                gn_cmd = [self.config.config['bng_command'], '%s.bngl' % gnm_name]
                if os.name == 'nt':  # Windows
                    # Explicitly call perl because the #! line in BNG2.pl is not supported.
                    gn_cmd = ['perl'] + gn_cmd
                try:
                    with open('%s.log' % gnm_name, 'w') as lf:
                        print2('Generating network for model %s.bngl' % gnm_name)
                        run(gn_cmd, check=True, stderr=STDOUT, stdout=lf, timeout=self.config.config['wall_time_gen'])
                except CalledProcessError as c:
                    logger.error("Command %s failed in directory %s" % (gn_cmd, os.getcwd()))
                    logger.error(c.stdout)
                    print0('Error: Initial network generation failed for model %s... see BioNetGen error log at '
                           '%s/%s.log' % (m.name, os.getcwd(), gnm_name))
                    exit(1)
                except TimeoutExpired:
                    logger.debug("Network generation exceeded %d seconds... exiting" %
                                  self.config.config['wall_time_gen'])
                    print0("Network generation took too long.  Increase 'wall_time_gen' configuration parameter")
                    exit(1)
                except:
                    tb = ''.join(traceback.format_list(traceback.extract_tb(sys.exc_info())))
                    logger.debug("Other exception occurred:\n%s" % tb)
                    print0("Unknown error occurred during network generation, see log... exiting")
                    exit(1)
                finally:
                    os.chdir(home_dir)

                logger.info('Output for network generation of model %s logged in %s/%s.log' %
                             (m.name, init_dir, gnm_name))
                final_model_list.append(NetModel(m.name, m.actions, m.suffixes, m.mutants, nf=init_dir + '/' + gnm_name + '.net'))
                final_model_list[-1].bng_command = m.bng_command
            else:
                logger.info('Model %s does not require network generation' % m.name)
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
        Adds the information from a Result to the Trajectory instance
        """
        # Evaluate objective if it wasn't done on workers.
        if res.score is None:  # Check if the objective wasn't evaluated on the workers
            res.normalize(self.config.config['normalization'])
            # Do custom postprocessing, if any
            try:
                res.postprocess_data(self.config.postprocessing)
            except Exception:
                logger.exception('User-defined post-processing script failed')
                traceback.print_exc()
                print0('User-defined post-processing script failed')
                res.score = np.inf
            else:
                res.score = self.objective.evaluate_multiple(res.simdata, self.exp_data, self.config.constraints)
            if res.score is None:  # Check if the above evaluation failed
                res.score = np.inf
                logger.warning('Simulation corresponding to Result %s contained NaNs or Infs' % res.name)
                logger.warning('Discarding Result %s as having an infinite objective function value' % res.name)
                print1('Simulation data in Result %s has NaN or Inf values.  Discarding this parameter set' % res.name)
        logger.info('Adding Result %s to Trajectory with score %.4f' % (res.name, res.score))
        self.trajectory.add(res.pset, res.score, res.name)

    def random_pset(self):
        """
        Generates a random PSet based on the distributions and bounds for each parameter specified in the configuration

        :return:
        """
        logger.debug("Generating a randomly distributed PSet")
        pset_vars = []
        for var in self.variables:
            pset_vars.append(var.sample_value())
        return PSet(pset_vars)

    def random_latin_hypercube_psets(self, n):
        """
        Generates n random PSets with a latin hypercube distribution
        More specifically, the uniform_var and loguniform_var variables follow the latin hypercube distribution,
        while lognorm are randomized normally.

        :param n: Number of psets to generate
        :return:
        """
        logger.debug("Generating PSets using Latin hypercube sampling")
        num_uniform_vars = 0
        for var in self.variables:
            if var.type == 'uniform_var' or var.type == 'loguniform_var':
                num_uniform_vars += 1

        # Generate latin hypercube of dimension = number of uniformly distributed variables.
        rands = latin_hypercube(n, num_uniform_vars)
        psets = []

        for row in rands:
            # Initialize the variables
            # Convert the 0 to 1 random numbers to the required variable range
            pset_vars = []
            rowindex = 0
            for var in self.variables:
                if var.type == 'uniform_var':
                    rescaled_val = var.p1 + row[rowindex]*(var.p2-var.p1)
                    pset_vars.append(var.set_value(rescaled_val))
                    rowindex += 1
                elif var.type == 'loguniform_var':
                    rescaled_val = exp10(np.log10(var.p1) + row[rowindex]*(np.log10(var.p2)-np.log10(var.p1)))
                    pset_vars.append(var.set_value(rescaled_val))
                    rowindex += 1
                else:
                    pset_vars.append(var.sample_value())
            psets.append(PSet(pset_vars))
        return psets

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
        logger.debug('Creating Job %s' % job_id)
        if self.config.config['smoothing'] > 1:
            # Create multiple identical Jobs for use with smoothing
            newjobs = []
            newnames = []
            for i in range(self.config.config['smoothing']):
                thisname = '%s_rep%i' % (job_id, i)
                newnames.append(thisname)
                # calc_future is supposed to be None here - the workers don't have enough info to calculate the
                # objective on their own
                newjobs.append(Job(self.model_list, params, thisname,
                                   self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                                   self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files'])))
            new_group = JobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs
        elif self.config.config['parallelize_models'] > 1:
            # Partition our model list into n different jobs
            newjobs = []
            newnames = []
            model_count = len(self.model_list)
            rep_count = self.config.config['parallelize_models']
            for i in range(rep_count):
                thisname = '%s_part%i' % (job_id, i)
                newnames.append(thisname)
                # calc_future is supposed to be None here - the workers don't have enough info to calculate the
                # objective on their own
                newjobs.append(Job(self.model_list[model_count*i//rep_count:model_count*(i+1)//rep_count],
                                   params, thisname, self.sim_dir, self.config.config['wall_time_sim'],
                                   self.calc_future, self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files'])))
            new_group = MultimodelJobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs
        else:
            # Create a single job
            return [Job(self.model_list, params, job_id,
                    self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                    self.config.config['normalization'], self.config.postprocessing,
                    bool(self.config.config['delete_old_files']))]


    def output_results(self, name='', no_move=False):
        """
        Tells the Trajectory to output a log file now with the current best fits.

        This should be called periodically by each Algorithm subclass, and is called by the Algorithm class at the end
        of the simulation.
        :return:
        :param name: Custom string to add to the saved filename. If omitted, we just use a running counter of the
        number of times we've outputted.
        :param no_move: If True, overrides the config setting delete_old_files=2, and does not move the result to
        overwrite sorted_params.txt
        :type name: str
        """
        if name == '':
            name = str(self.output_counter)
            self.output_counter += 1
        if self.refine:
            name = 'refine_%s' % name
        filepath = '%s/sorted_params_%s.txt' % (self.res_dir, name)
        logger.info('Outputting results to file %s' % filepath)
        self.trajectory.write_to_file(filepath)

        # If the user has asked for fewer output files, each time we're here, move the new file to
        # Results/sorted_params.txt, overwriting the previous one.
        if self.config.config['delete_old_files'] >= 2 and not no_move:
            logger.debug("Overwriting previous 'sorted_params.txt'")
            noname_filepath = '%s/sorted_params.txt' % self.res_dir
            if os.path.isfile(noname_filepath):
                os.remove(noname_filepath)
            os.replace(filepath, noname_filepath)

    def backup(self, pending_psets=()):
        """
        Create a backup of this algorithm object that can be reloaded later to resume the run

        :param pending_psets: Iterable of PSets that are currently submitted as jobs, and will need to get re-submitted
        when resuming the algorithm
        :return:
        """

        logger.info('Saving a backup of the algorithm')
        # Save a backup of the PSets
        self.output_results(name='backup', no_move=True)

        # Pickle the algorithm
        # Save to a temporary file first, so we can't get interrupted and left with no backup.
        picklepath = '%s/alg_backup.bp' % self.config.config['output_dir']
        temppicklepath = '%s/alg_backup_temp.bp' % self.config.config['output_dir']
        try:
            f = open(temppicklepath, 'wb')
            pickle.dump((self, pending_psets), f)
            f.close()
            os.replace(temppicklepath, picklepath)
        except IOError as e:
            logger.exception('Failed to save backup of algorithm')
            print1('Failed to save backup of the algorithm.\nSee log for more information')
            if e.strerror == 'Too many open files':
                print0('Too many open files! See "Troubleshooting" in the documentation for how to deal with this '
                       'problem.')

    def get_backup_every(self):
        """
        Returns a number telling after how many individual simulation returns should we back up the algorithm.
        Makes a good guess, but could be overridden in a subclass
        """
        return self.config.config['backup_every'] * self.config.config['population_size'] * \
            self.config.config['smoothing']

    def add_iterations(self, n):
        """
        Adds n additional iterations to the algorithm.
        May be overridden in subclasses that don't use self.max_iterations to track the iteration count
        """
        self.max_iterations += n

    def run(self, client, resume=None, debug=False):
        """Main loop for executing the algorithm"""

        if self.refine:
            logger.debug('Setting up Simplex refinement of previous algorithm')

        backup_every = self.get_backup_every()
        sim_count = 0

        logger.debug('Generating initial parameter sets')
        if resume:
            psets = resume
            logger.debug('Resume algorithm with the following PSets: %s' % [p.name for p in resume])
        else:
            psets = self.start_run()

        if not os.path.isdir(self.failed_logs_dir):
            os.mkdir(self.failed_logs_dir)

        if self.config.config['local_objective_eval'] == 0 and self.config.config['smoothing'] == 1 and \
                self.config.config['parallelize_models'] == 1:
            calculator = ObjectiveCalculator(self.objective, self.exp_data, self.config.constraints)
            [self.calc_future] = client.scatter([calculator], broadcast=True)
        else:
            self.calc_future = None

        jobs = []
        pending = dict()  # Maps pending futures to tuple (PSet, job_id).
        for p in psets:
            jobs += self.make_job(p)
        jobs[0].show_warnings = True  # For only the first job submitted, show warnings if exp data is unused.
        logger.info('Submitting initial set of %d Jobs' % len(jobs))
        futures = []
        for job in jobs:
            f = client.submit(run_job, job, True, self.failed_logs_dir)
            futures.append(f)
            pending[f] = (job.params, job.job_id)
        pool = custom_as_completed(futures, with_results=True, raise_errors=False)
        backed_up = True
        while True:
            if sim_count % backup_every == 0 and not backed_up:
                self.backup(set([pending[fut][0] for fut in pending]))
                backed_up = True
            f, res = next(pool)
            if isinstance(res, DaskError):
                if isinstance(res.error, PybnfError):
                    raise res.error  # User-targeted error should be raised instead of skipped
                logger.error('Job failed with an exception')
                logger.error(res.traceback)
                res = FailedSimulation(pending[f][0], pending[f][1], 3)
            # Handle if this result is one of multiple instances for smoothing
            del pending[f]
            if self.config.config['smoothing'] > 1 or self.config.config['parallelize_models'] > 1:
                group = self.job_group_dir.pop(res.name)
                done = group.job_finished(res)
                if not done:
                    continue
                res = group.average_results()
            sim_count += 1
            backed_up = False
            if isinstance(res, FailedSimulation):
                if res.fail_type >= 1:
                    self.fail_count += 1
                tb = '\n'+res.traceback if res.fail_type == 1 else ''

                logger.debug('Job %s failed with code %d%s' % (res.name, res.fail_type, tb))
                if res.fail_type >= 1:
                    print1('Job %s failed' % res.name)
                else:
                    print1('Job %s timed out' % res.name)
                if self.success_count == 0 and self.fail_count >= 100:
                    raise PybnfError('Aborted because all jobs are failing',
                                     'Your simulations are failing to run. Logs from failed simulations are saved in '
                                     'the FailedSimLogs directory. For help troubleshooting this error, refer to '
                                     'https://pybnf.readthedocs.io/en/latest/troubleshooting.html#failed-simulations')
            else:
                self.success_count += 1
                logger.debug('Job %s complete' % res.name)

            self.add_to_trajectory(res)
            if res.score < self.config.config['min_objective']:
                logger.info('Minimum objective value achieved')
                print1('Minimum objective value achieved')
                break
            response = self.got_result(res)
            if response == 'STOP':
                self.best_fit_obj = self.trajectory.best_score()
                logger.info("Stop criterion satisfied with objective function value of %s" % self.best_fit_obj)
                print1("Stop criterion satisfied with objective function value of %s" % self.best_fit_obj)
                break
            else:
                new_futures = []
                for ps in response:
                    new_js = self.make_job(ps)
                    for new_j in new_js:
                        new_f = client.submit(run_job, new_j, (debug or self.fail_count < 10), self.failed_logs_dir)
                        pending[new_f] = (ps, new_j.job_id)
                        new_futures.append(new_f)
                logger.debug('Submitting %d new Jobs' % len(new_futures))
                pool.update(new_futures)

        logger.info("Cancelling %d pending jobs" % len(pending))
        client.cancel(list(pending.keys()))
        self.output_results('final')

        # Copy the best simulations into the results folder
        best_name = self.trajectory.best_fit_name()
        best_pset = self.trajectory.best_fit()
        logger.info('Copying simulation results from best fit parameter set to Results/ folder')
        for m in self.config.models:
            this_model = self.config.models[m]
            to_save = this_model.copy_with_param_set(best_pset)
            to_save.save_all('%s/%s_%s' % (self.res_dir, to_save.name, best_name))
            if self.config.config['delete_old_files'] == 0:
                for simtype, suf in this_model.suffixes:
                    if simtype == 'simulate':
                        ext = 'gdat'
                    else:  # parameter_scan
                        ext = 'scan'
                    if self.config.config['smoothing'] > 1:
                        best_name = best_name + '_rep0'  # Look for one specific replicate of the data
                    try:
                        shutil.copy('%s/%s/%s_%s_%s.%s' % (self.sim_dir, best_name, m, best_name, suf, ext),
                                    '%s' % self.res_dir)
                    except FileNotFoundError:
                        logger.error('Cannot find files corresponding to best fit parameter set')
                        print0('Could not find your best fit gdat file. This could happen if all of the simulations\n'
                               ' in your run failed, or if that gdat file was somehow deleted during the run.')
        if self.config.config['delete_old_files'] > 0 and self.config.config['save_best_data']:
            # Rerun the best fit parameter set so the gdat file(s) are saved in the Results folder.
            logger.info('Rerunning best fit parameter set to save data files.')
            # Enable saving files for SBML models
            for m in self.model_list:
                if isinstance(m, SbmlModelNoTimeout):
                    m.save_files = True
            finaljob = Job(self.model_list, best_pset, 'bestfit',
                           self.sim_dir, self.config.config['wall_time_sim'], None,
                           self.config.config['normalization'], self.config.postprocessing,
                           False)
            try:
                run_job(finaljob)
            except Exception:
                logger.exception('Failed to rerun best fit parameter set')
                print1('Failed to rerun best fit parameter set. See log for details')
            else:
                # Copy all gdat and scan to Results
                for fname in glob(self.sim_dir+'/bestfit/*.gdat') + glob(self.sim_dir+'/bestfit/*.scan'):
                    shutil.copy(fname, self.res_dir)
            # Disable saving files for SBML models (in case there is future bootstrapping or refinement)
            for m in self.model_list:
                if isinstance(m, SbmlModelNoTimeout):
                    m.save_files = False

        if self.bootstrap_number is None or self.bootstrap_number == self.config.config['bootstrap']:
            try:
                os.replace('%s/alg_backup.bp' % self.config.config['output_dir'],
                          '%s/alg_%s.bp' % (self.config.config['output_dir'],
                                            ('finished' if not self.refine else 'refine_finished')))
                logger.info('Renamed pickled algorithm backup to alg_%s.bp' %
                            ('finished' if not self.refine else 'refine_finished'))
            except OSError:
                logger.warning('Tried to move pickled algorithm, but it was not found')

        if (isinstance(self, SimplexAlgorithm) or self.config.config['refine'] != 1) and self.bootstrap_number is None:
            # End of fitting; delete unneeded files
            if self.config.config['delete_old_files'] >= 1:
                if os.name == 'nt':  # Windows
                    try:
                        shutil.rmtree(self.sim_dir)
                    except OSError:
                        logger.error('Failed to remove simulations directory '+self.sim_dir)
                else:
                    run(['rm', '-rf', self.sim_dir])  # More likely to succeed than rmtree()

        logger.info("Fitting complete")

    def cleanup(self):
        """
        Called before the program exits due to an exception.
        :return:
        """
        self.output_results('end')


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
        self.bests = [[None, np.inf]] * self.num_particles  # The best result for each particle: list of the
        # form [PSet, objective]
        self.global_best = [None, np.inf]  # The best result for the whole swarm
        self.last_best = np.inf

    def reset(self, bootstrap=None):
        super(ParticleSwarm, self).reset(bootstrap)
        self.nv = 0
        self.num_evals = 0
        self.swarm = []
        self.pset_map = dict()
        self.bests = [[None, np.inf]] * self.num_particles
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
                logger.info('Island %i completed %i iterations' % (island, self.iter_num[island]))
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
        self.fitnesses = [np.Inf for i in range(self.population_size)]

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

        gen = int(re.search('(?<=gen)\d+', pset.name).group(0))
        j = int(re.search('(?<=ind)\d+', pset.name).group(0))

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
                logger.info('Completed %i simulations' % self.sims_completed)
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
            logger.info('Iteration %i' % self.iteration)
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


class BayesianAlgorithm(Algorithm):
    """Superclass for Bayesian MCMC algorithms"""

    def __init__(self, config):
        super(BayesianAlgorithm, self).__init__(config)
        self.num_parallel = config.config['population_size']
        self.max_iterations = config.config['max_iterations']
        self.step_size = config.config['step_size']

        self.iteration = [0] * self.num_parallel  # Iteration number that each PSet is on

        self.current_pset = None  # List of n PSets corresponding to the n independent runs
        self.ln_current_P = None  # List of n probabilities of those n PSets.

        self.burn_in = config.config['burn_in']  # todo: 'auto' option
        self.sample_every = config.config['sample_every']
        self.output_hist_every = config.config['output_hist_every']
        # A list of the % credible intervals to save, eg [68. 95]
        self.credible_intervals = config.config['credible_intervals']
        self.num_bins = config.config['hist_bins']

        self.wait_for_sync = [False] * self.num_parallel

        self.prior = None
        self.load_priors()

        self.samples_file = self.config.config['output_dir'] + '/Results/samples.txt'

    def load_priors(self):
        """Builds the data structures for the priors, based on the variables specified in the config."""
        self.prior = dict()  # Maps each variable to a 4-tuple (space, dist, val1, val2)
        # space is 'reg' for regular space, 'log' for log space. dist is 'n' for normal, 'b' for box.
        # For normal distribution, val1 = mean, val2 = sigma (in regular or log space as appropriate)
        # For box distribution, val1 = min, val2 = max (in regular or log space as appropriate)
        for var in self.variables:
            if var.type == 'normal_var':
                self.prior[var.name] = ('reg', 'n', var.p1, var.p2)
            elif var.type == 'lognormal_var':
                self.prior[var.name] = ('log', 'n', var.p1, var.p2)
            elif var.type == 'uniform_var':
                self.prior[var.name] = ('reg', 'b', var.p1, var.p2)
            elif var.type == 'loguniform_var':
                self.prior[var.name] = ('log', 'b', np.log10(var.p1), np.log10(var.p2))

    def start_run(self, setup_samples=True):
        if self.config.config['initialization'] == 'lh':
            first_psets = self.random_latin_hypercube_psets(self.num_parallel)
        else:
            first_psets = [self.random_pset() for i in range(self.num_parallel)]

        self.ln_current_P = [np.nan]*self.num_parallel  # Forces accept on the first run
        self.current_pset = [None]*self.num_parallel
        for i in range(len(first_psets)):
            first_psets[i].name = 'iter0run%i' % i

        # Set up the output files
        # Cant do this in the constructor because that happens before the output folder is potentially overwritten.
        if setup_samples:
            with open(self.samples_file, 'w') as f:
                f.write('# Name\tLn_probability\t'+first_psets[0].keys_to_string()+'\n')
            os.makedirs(self.config.config['output_dir'] + '/Results/Histograms/', exist_ok=True)

        return first_psets

    def got_result(self, res):
        NotImplementedError("got_result() must be implemented in BayesianAlgorithm subclass")

    def ln_prior(self, pset):
        """
        Returns the value of the prior distribution for the given parameter set

        :param pset:
        :type pset: PSet
        :return: float value of ln times the prior distribution
        """
        total = 0.
        for v in self.prior:
            (space, dist, x1, x2) = self.prior[v]
            if space == 'log':
                val = np.log10(pset[v])
            else:
                val = pset[v]

            if dist == 'n':
                # Normal with mean x1 and value x2
                total += -1. / (2. * x2 ** 2.) * (x1 - val)**2.
            else:
                # Uniform from x1 to x2
                if x1 <= val <= x2:
                    total += -np.log(x2-x1)
                else:
                    logger.warning('Box-constrained parameter %s reached a value outside the box.')
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
            f.write(pset.name+'\t'+str(ln_prob)+'\t'+pset.values_to_string()+'\n')

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
            fname = self.config.config['output_dir']+'/Results/Histograms/%s%s.txt' % (v.name, file_ext)
            # For log-space variables, we want the histogram in log space
            if v.log_space:
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
                n = len(sorted_data)
                want = n * (interval/100)
                min_index = int(np.round(n/2 - want/2))
                max_index = int(np.round(n/2 + want/2 - 1))
                file.write('%s\t%s\t%s\n' % (v.name, sorted_data[min_index], sorted_data[max_index]))

        for file in cred_files:
            file.close()

    def cleanup(self):
        """Called when quitting due to error.
        Save the histograms in addition to the usual algorithm cleanup"""
        super().cleanup()
        self.update_histograms('_end')


class DreamAlgorithm(BayesianAlgorithm):
    """
    **This algorithm is a work in progress, and does not currently work correctly. In our most recent testing, it
    generates incorrect probability distributions**

    Implements a variant of the DREAM algorithm as described in Vrugt (2016) Environmental Modelling
    and Software.

    Adapts Bayesian MCMC to use methods from differential evolution for accelerated convergence and
    more efficient sampling of parameter space
    """

    def __init__(self, config):
        super(DreamAlgorithm, self).__init__(config)
        print0('You are running the DREAM algorithm. This is a work in progress, and is not officially supported! In '
               'our most recent testing, it generates incorrect probability distributions.')
        self.n_dim = len(self.variables)
        self.all_idcs = np.arange(self.n_dim)
        self.ncr = [(1+x)/self.config.config['crossover_number'] for x in range(self.config.config['crossover_number'])]
        self.g_prob = self.config.config['gamma_prob']
        self.acceptances = [0]*self.num_parallel
        self.acceptance_rates = [0.0]*self.num_parallel

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

        m = re.search('(?<=run)\d+', pset.name)
        index = int(m.group(0))

        # Calculate posterior of finished job
        lnprior = self.ln_prior(pset)
        lnlikelihood = -score
        lnposterior = lnprior + lnlikelihood

        # Metropolis-Hastings criterion
        ln_p_accept = np.log10(np.random.uniform()) < min(0., lnposterior - self.ln_current_P[index])
        if ln_p_accept:  # accept update based on MH criterion
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.acceptances[index] += 1

        # Record that this individual is complete
        self.wait_for_sync[index] = True
        self.iteration[index] += 1
        self.acceptance_rates[index] = self.acceptances[index] / self.iteration[index]

        # Update histograms and trajectories if necessary
        if self.iteration[index] % self.sample_every == 0 and self.iteration[index] > self.burn_in:
            self.sample_pset(self.current_pset[index], self.ln_current_P[index])
        if (self.iteration[index] % (self.sample_every * self.output_hist_every) == 0
            and self.iteration[index] > self.burn_in):
            self.update_histograms('_%i' % self.iteration[index])

        # Wait for entire generation to finish
        if np.all(self.wait_for_sync):

            self.wait_for_sync = [False] * self.num_parallel

            if min(self.iteration) >= self.max_iterations:
                return 'STOP'

            if self.iteration[index] % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                print2('Acceptance rates: %s\n' % str(self.acceptance_rates))
            else:
                print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
            logger.info('Completed %i iterations' % self.iteration[index])
            print2('Current -Ln Posteriors: %s' % str(self.ln_current_P))

            next_gen = []
            for i, p in enumerate(self.current_pset):
                new_pset = self.calculate_new_pset(i)
                if new_pset:
                    new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                    next_gen.append(new_pset)
                else:
                    #  If new PSet is outside of variable bounds, keep current PSet and wait for next generation
                    logger.debug('Proposed PSet %s is invalid.  Rejecting and waiting until next iteration' % i)
                    self.wait_for_sync[i] = True
                    self.iteration[i] += 1

            return next_gen

        return []

    def calculate_new_pset(self, idx):
        """
        Uses differential evolution-like update to calculate new PSet

        :param idx: Index of PSet to update
        :return:
        """

        # Choose individuals (not individual to be updated) for mutation
        sel = np.random.choice(self.all_idcs[self.all_idcs != idx], 2, replace=False)
        x0 = self.current_pset[idx]
        x1 = self.current_pset[sel[0]]
        x2 = self.current_pset[sel[1]]

        # Sample the probability of modifying a parameter
        cr = np.random.choice(self.ncr)
        while True:
            ds = np.random.uniform(size=self.n_dim) <= cr  # sample parameter subspace
            if np.any(ds):
                break

        # Sample whether to jump to the mode (when gamma = 1)
        gamma = 1 if np.random.uniform() < self.g_prob else self.step_size

        new_vars = []
        for i, d in enumerate(np.random.permutation(ds)):
            k = self.variables[i]
            diff = x1.get_param(k.name).diff(x2.get_param(k.name)) if d else 0.0
            zeta = np.random.normal(0, self.config.config['zeta'])
            lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

            # Differential evolution calculation (while satisfying detailed balance)
            try:
                # Do not reflect the parameter (need to reject if outside bounds)
                new_var = x0.get_param(k.name).add(zeta + (1. + lamb) * gamma * diff, False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                logger.debug("Variable %s is outside of bounds")
                return None

        return PSet(new_vars)


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

        # Figure out which parallel run this is from based on the .name field.
        m = re.search('(?<=run)\d+', pset.name)
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

        # Record the current PSet (clarification: what if failed? Sample old again?)
        # Using either the newly accepted PSet or the old PSet, propose the next PSet.
        proposed_pset = self.try_to_choose_new_pset(index)

        if proposed_pset is None:
            if np.all(self.wait_for_sync):
                # Do the replica exchange, then propose n new psets so all chains resume
                self.wait_for_sync = [False] * self.num_parallel
                return self.replica_exchange()
            elif min(self.iteration) >= self.max_iterations:
                print0('Overall move accept rate: %f' % (self.accepted/self.attempts))
                self.update_histograms('_final')
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
                    self.sample_pset(self.current_pset[index], self.ln_current_P[index])
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
                logger.info('Completed %i iterations' % self.iteration[index])
                logger.info('Current move accept rate: %f' % (self.accepted/self.attempts))
                if self.exchange_attempts > 0:
                    logger.info('Current replica exchange rate: %f' % (self.exchange_accepted / self.exchange_attempts))
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
        Helper function to perturb the old PSet, generating a new proposed PSet
        If the new PSet fails automatically because it violates box constraints, returns None.

        :param oldpset: The PSet to be changed
        :type oldpset: PSet
        :return: the new PSet
        """

        delta_vector = {k: np.random.normal() for k in oldpset.keys()}
        delta_vector_magnitude = np.sqrt(sum([x ** 2 for x in delta_vector.values()]))
        delta_vector_normalized = {k: self.step_size * delta_vector[k] / delta_vector_magnitude for k in oldpset.keys()}
        new_vars = []
        for v in oldpset:
            # For box constraints, need special treatment to keep correct statistics
            # If we tried to leave the box, the move automatically fails, we should increment the iteration counter
            # and retry.
            # The same could happen if normal_var's try to go below 0
            try:
                new_var = v.add(delta_vector_normalized[v.name])
            except OutOfBoundsException:
                logger.debug('Rejected a move because %s=%.2E moved by %f, outside the box constraint [%.2E, %.2E]' %
                             (v.name, oldpset[v.name], delta_vector_normalized[v.name], v.lower_bound, v.upper_bound))
                return None
            new_vars.append(new_var)

        return PSet(new_vars)

    def replica_exchange(self):
        """
        Performs replica exchange for parallel tempering.
        Then proposes n new parameter sets to resume all chains after the exchange.
        :return: List of n PSets to run
        """
        logger.info('Performing replica exchange on iteration %i' % self.iteration[0])
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


class SimplexAlgorithm(Algorithm):
    """
    Implements a parallelized version of the Simplex local search algorithm, as described in Lee and Wiswall 2007,
    Computational Economics

    """

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

        self.parallel_count = min(self.config.config['population_size'], len(self.variables))
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


def latin_hypercube(nsamples, ndims):
    """
    Latin hypercube sampling.

    Returns a nsamples by ndims array, with entries in the range [0,1]
    You'll have to rescale them to your actual param ranges.
    """
    if ndims == 0:
        # Weird edge case - needed for other code counting on result having a number of rows
        return np.zeros((nsamples, 0))
    value_table = np.transpose(np.array([[i/nsamples + 1/nsamples * np.random.random() for i in range(nsamples)]
                                         for dim in range(ndims)]))
    for dim in range(ndims):
        np.random.shuffle(value_table[:, dim])
    return value_table


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
        job = Job(self.model_list, empty, 'check', self.sim_dir, self.config.config['wall_time_sim'], None,
                  None, dict(), delete_folder=False)
        result = run_job(job, debug, self.sim_dir)

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

        result.score = self.objective.evaluate_multiple(result.simdata, self.exp_data, self.config.constraints)
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

def exp10(n):
    """
    Raise 10 to the power of a possibly user-defined value, and raise a helpful error if it overflows
    :param n: A float
    :return: 10.** n
    """
    try:
        with np.errstate(over='raise'):
            ans = 10.**n
    except (OverflowError, FloatingPointError):
        logger.error('Overflow error in exp10()')
        logger.error(''.join(traceback.format_stack()))  # Log the entire traceback
        raise PybnfError('Overflow when calculating 10^%d\n'
                         'Logs are saved in bnf.log\n'
                         'This may be because you declared a lognormal_var or a logvar, and specified the '
                         'arguments in regular space instead of log10 space.' % n)
    return ans
