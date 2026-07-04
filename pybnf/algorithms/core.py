"""Core execution primitives shared by every algorithm.

Holds the ``Result`` / ``FailedSimulation`` containers, the ``Job`` runner,
the ``JobGroup`` smoothing/model-parallel folding family, ``run_job``,
``result_from_completed``, and the dask ``as_completed`` import.

This module is the patched seam (ADR-0001). Tests and the slow
``integration_harness`` patch ``run_job`` / ``as_completed`` / ``Job`` here,
and the run loop resolves them as ``core.<name>`` so that "patch where it is
defined" stays honest as the algorithm classes move out of the package
facade in later commits.

Import direction is one-way: this module imports only leaf modules
(``data`` / ``pset`` / ``printing``) and never anything from the
``algorithms`` package facade or any algorithm leaf.
"""

import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from concurrent.futures import CancelledError
from subprocess import run, CalledProcessError, TimeoutExpired
# Re-exported (not used inside this module): the run loop calls
# core.as_completed and the tests/integration_harness patch it here -- this is
# the dask seam (ADR-0001). `as as_completed` marks the intentional re-export.
from distributed import as_completed as as_completed

from ..data import Data
from ..pset import FailedSimulationError
from ..printing import print0, print1, PybnfError

# Preserve the original module logger name (was ``getLogger(__name__)`` when
# these functions lived in ``pybnf/algorithms.py``) so log routing stays
# byte-identical across the split. ``Job.jlogger`` keeps its own hardcoded
# ``'pybnf.algorithms.job'`` name below.
logger = logging.getLogger('pybnf.algorithms')


class Result:
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
        super().__init__(paramset, None, name)
        self.fail_type = fail_type
        self.failed = True
        self.traceback = ''.join(traceback.format_exception(*einfo))

    def normalize(self, settings):
        return

    def postprocess_data(self, settings):
        return


class _ResolvedFuture:
    """A minimal future-like handle wrapping an already-resolved value.

    dask substitutes a scattered ``Future`` with its concrete value only when the
    Future is a *direct* argument to ``client.submit`` (positional or keyword). A
    Future buried as an attribute of a submitted object is pickled verbatim and,
    since distributed 2026.6.0, deserializes "not properly initialized" on the
    worker, so calling ``.result()`` on it raises (lanl/PyBNF #476). ``Algorithm.run``
    therefore passes the once-scattered model_list / objective Futures to
    ``client.submit`` as direct ``models=`` / ``calc=`` kwargs, and ``run_job``
    rebinds their worker-resolved values onto the Job wrapped in this handle. That
    keeps the Job's duck-typed ``.result()`` consumers (``Job._get_models`` and the
    scoring in ``Job.run_simulation``) working unchanged while guaranteeing no raw
    Future is ever ``.result()``-ed off a Job attribute worker-side.
    """
    __slots__ = ('_value',)

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def run_job(j, debug=False, failed_logs_dir='', models=None, calc=None):
    """
    Runs the Job j.
    This function is passed to Dask instead of j.run_simulation because if you pass j.run_simulation, Dask leaks memory
    associated with j.

    ``models`` / ``calc`` are the worker-resolved model_list and ObjectiveCalculator
    that ``Algorithm.run`` scattered once per fit and hands to ``client.submit`` as
    direct kwargs, so dask substitutes the Futures with their concrete values here
    on the worker (see :class:`_ResolvedFuture` for why a Future stashed as a Job
    attribute cannot be used instead). When provided they are rebound onto the Job;
    when omitted -- the in-process paths that call ``run_job`` directly
    (``_rerun_best_fit_to_save_data``, ``ModelCheck``, the folder-free test
    harnesses) -- the Job's own ``models`` / ``calc_future`` attributes are used
    as-is.
    """
    if models is not None:
        j.models = _ResolvedFuture(models)
    if calc is not None:
        j.calc_future = _ResolvedFuture(calc)
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
                 delete_folder=False, replicate_index=0, stochastic_seed_policy='auto', model_slice=None):
        """
        Instantiates a Job

        :param models: The models to evaluate, as either a concrete list of Model
        instances or a dask Future wrapping the whole model_list scattered once
        per fit (see Algorithm.run / make_job). A Future is resolved worker-side
        and sliced by ``model_slice``; this avoids re-serializing the
        parameter-independent model graph on every evaluation (issue #416).
        :type models: list of Model instances, or distributed.Future
        :param model_slice: ``(start, stop)`` applied to the resolved model_list
        when ``models`` is a Future (parallelize_models hands each Job a slice);
        ``None`` means use the whole list. Ignored when ``models`` is a concrete
        list.
        :type model_slice: tuple or None
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
        :param replicate_index: Smoothing replicate index (0..smoothing-1); folded into stochastic seed
        derivation so replicates of the same param set yield distinct deterministic trajectories.
        :type replicate_index: int
        :param stochastic_seed_policy: One of 'auto', 'auto_honorbngl', 'random', 'random_honorbngl'
        from the `stochastic_seed` config option. Stamped onto each model copy at execute time.
        :type stochastic_seed_policy: str
        """
        self.models = models
        self.model_slice = model_slice
        self._resolved_models = None  # cache for _get_models() (worker-side)
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
            self.output_dir = str(Path(self.home_dir) / output_dir)
        self.timeout = timeout

        # Folder where we save the model files and outputs.
        self.folder = f'{self.output_dir}/{self.job_id}'
        self.delete_folder = delete_folder
        self.replicate_index = replicate_index
        self.stochastic_seed_policy = stochastic_seed_policy

    def _name_with_id(self, model):
        return f'{model.name}_{self.job_id}'

    def _get_models(self):
        """Resolve ``self.models`` to a concrete list of Model instances.

        ``self.models`` is either a plain list (tests / a Job built outside the
        run loop) or a dask Future for the model_list scattered once per fit
        (see Algorithm.make_job). A Future is resolved here, worker-side, and
        sliced by ``self.model_slice`` -- mirroring how ``calc_future`` is
        resolved via ``.result()``. The resolved list is cached so a job that
        dask happens to run twice does not re-fetch it.
        """
        if self._resolved_models is None:
            models = self.models
            if hasattr(models, 'result'):  # a scattered Future, not a list
                models = models.result()
                if self.model_slice is not None:
                    models = models[self.model_slice[0]:self.model_slice[1]]
            self._resolved_models = list(models)
        return self._resolved_models

    def _run_models(self):
        ds = {}

        for model in self._get_models():
            model_file_prefix = self._name_with_id(model)
            model_with_params = model.copy_with_param_set(self.params)
            # Stamp seed-policy context onto the per-evaluation copy so model
            # backends can derive deterministic seeds without plumbing config
            # through every execute() signature.
            model_with_params._pybnf_replicate_index = self.replicate_index
            model_with_params._pybnf_stochastic_seed_policy = self.stochastic_seed_policy
            ds[model.name] = model_with_params.execute(self.folder, model_file_prefix, self.timeout)

        return ds

    def _copy_log_files(self, failed_logs_dir):
        if failed_logs_dir == '':
            self.jlogger.error('Cannot save log files without specified directory')
            return
        for m in self._get_models():
            lf = f'{self.folder}/{self._name_with_id(m)}.log'
            if os.path.isfile(lf):
                self.jlogger.debug(f'Copying log file {lf}')
                shutil.copy(lf, failed_logs_dir)

    def run_simulation(self, debug=False, failed_logs_dir=''):
        """Runs the simulation and reads in the result"""
        # Force absolute path for failed_logs_dir
        if len(failed_logs_dir) > 0 and failed_logs_dir[0] != '/':
            failed_logs_dir = str(Path(self.home_dir) / failed_logs_dir)

        # The check here is in case dask decides to run the same job twice, both of them can complete.
        made_folder = False
        failures = 0
        while not made_folder:
            try:
                os.mkdir(self.folder)
                self.jlogger.debug(f'Created folder {self.folder} for simulation')
                made_folder = True
            except OSError:
                self.jlogger.warning(f'Failed to create folder {self.folder}, trying again.')
                failures += 1
                self.folder = '%s/%s_rerun%i' % (self.output_dir, self.job_id, failures)
                if failures > 1000:
                    self.jlogger.error(f'Job {self.job_id} failed because it was unable to write to the Simulations folder')
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
            self.jlogger.exception(f'File not found during job {self.job_id}. This should only happen if the fitting '
                                   'is already done.')
            res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())
        except Exception:
            if debug:
                self._copy_log_files(failed_logs_dir)
            print1('A simulation failed with an unknown error. See the log for details, and consider reporting this '
                   'as a bug.')
            self.jlogger.exception(f'Unknown error during job {self.job_id}')
            res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())
        else:
            if self.calc_future is not None:
                try:
                    res.normalize(self.norm_settings)
                    try:
                        res.postprocess_data(self.postproc_settings)
                    except Exception:
                        self.jlogger.exception('User-defined post-processing script failed')
                        traceback.print_exc()
                        print0('User-defined post-processing script failed')
                        res.score = np.inf
                    else:
                        res.score = self.calc_future.result().evaluate_objective(res.simdata, res.pset, show_warnings=self.show_warnings)
                        res.out = simdata
                        if res.score is None:
                            res.score = np.inf
                            logger.warning(f'Simulation corresponding to Result {res.name} contained NaNs or Infs')
                            logger.warning(f'Discarding Result {res.name} as having an infinite objective function value')
                except Exception:
                    # A failure while normalizing or scoring this one parameter set
                    # (e.g. a PybnfError from mismatched simulation/exp columns) should
                    # penalize this evaluation, not crash the whole run. See lanl/PyBNF#388.
                    if debug:
                        self._copy_log_files(failed_logs_dir)
                    self.jlogger.exception(f'Objective evaluation failed during job {self.job_id}')
                    res = FailedSimulation(self.params, self.job_id, 1, sys.exc_info())
                res.simdata = None
        if self.delete_folder:
            if os.name == 'nt':  # Windows
                try:
                    shutil.rmtree(self.folder)
                    self.jlogger.debug(f'Removed folder {self.folder}')
                except OSError:
                    self.jlogger.error(f'Failed to remove folder {self.folder}.')
            else:
                try:
                    run(['rm', '-rf', self.folder], check=True, timeout=1800)
                    self.jlogger.debug(f'Removed folder {self.folder}')
                except (CalledProcessError, TimeoutExpired):
                    self.jlogger.error(f'Failed to remove folder {self.folder}.')

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
            raise ValueError(f'Job group {self.job_id} received unwanted result {res.name}')
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


class HybridJobGroup(JobGroup):
    """
    A JobGroup to handle combined smoothing and model-level parallelism.
    """
    def __init__(self, job_id, replica_subjob_ids):
        """
        :param job_id: The name of the Job this group is representing
        :param replica_subjob_ids: List of (replica id, subjob ids) pairs for model partitions in each
        smoothing replica.
        """
        subjob_ids = [sid for _, ids in replica_subjob_ids for sid in ids]
        super().__init__(job_id, subjob_ids)
        self.replica_groups = [MultimodelJobGroup(replica_id, ids) for replica_id, ids in replica_subjob_ids]

    def job_finished(self, res):
        """
        Called when one job in this group has finished.
        :param res: Result object for the completed job
        :return: Boolean, whether everything in this job group has finished
        """
        if self.failed:
            return False
        if isinstance(res, FailedSimulation):
            self.failed = res
            return True

        if res.name not in self.subjob_ids:
            raise ValueError(f'Job group {self.job_id} received unwanted result {res.name}')

        for group in self.replica_groups:
            if res.name in group.subjob_ids:
                group.job_finished(res)
                break
        return all(len(group.result_list) == len(group.subjob_ids) for group in self.replica_groups)

    def average_results(self):
        """
        Merge model partitions within each smoothing replica, then average the complete replicas.
        """
        if self.failed:
            self.failed.name = self.job_id
            return self.failed

        replica_results = [group.average_results() for group in self.replica_groups]
        avedata = dict()
        for m in replica_results[0].simdata:
            avedata[m] = dict()
            for suf in replica_results[0].simdata[m]:
                avedata[m][suf] = Data.average([r.simdata[m][suf] for r in replica_results])
        return Result(replica_results[0].pset, avedata, self.job_id)


def result_from_completed(future, result, params, job_id):
    """
    Translate one completed item from
    ``as_completed(futures, with_results=True, raise_errors=False)`` into a Result.

    With those flags, dask hands back per future:
      * the job's return value (a Result / FailedSimulation) for a finished job,
      * the worker's ``(type, exception, traceback)`` tuple for an errored job, and
      * a ``CancelledError`` for a cancelled job.

    A failed job becomes a ``FailedSimulation`` so the run continues, except a
    user-targeted ``PybnfError`` is re-raised (it would fail every job, so failing
    fast is better than burning the whole fit). A ``CancelledError`` is returned
    unchanged for the caller to treat as fatal, and anything unrecognized is treated
    as a failed simulation rather than crashing the run.

    ``Future.status`` is part of dask's public Future API, so this relies on no
    dask internals. See lanl/PyBNF#388 for the history (the previous approach
    subclassed as_completed and overrode a private method that dask later renamed).
    """
    if getattr(future, 'status', None) == 'error':
        typ, exc, tb = result
        if isinstance(exc, PybnfError):
            raise exc  # User-targeted error should be raised instead of skipped
        logger.error(f'Job {job_id} failed with an exception')
        logger.error(''.join(traceback.format_exception(typ, exc, tb)))
        return FailedSimulation(params, job_id, 3)
    if isinstance(result, CancelledError):
        return result
    if not isinstance(result, Result):
        # e.g. a raw tuple leaking from as_completed for an errored future whose
        # status we somehow didn't catch; never crash the run over it.
        logger.error(f'Job {job_id} returned an unexpected result of type {type(result).__name__}; treating as a failed simulation')
        return FailedSimulation(params, job_id, 3)
    return result
