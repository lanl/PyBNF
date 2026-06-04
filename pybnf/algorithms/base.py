"""The Algorithm base class and the module-level helpers it relies on.

Extracted from ``algorithms.py`` (M1 Step 2). The run loop resolves the three
monkeypatched names as ``core.run_job`` / ``core.as_completed`` / ``core.Job``
(ADR-0001), so it imports the ``core`` module object rather than binding those
names locally; the never-patched data classes are imported by name.
"""


from . import core
from .core import (
    FailedSimulation,
    JobGroup,
    MultimodelJobGroup,
    HybridJobGroup,
    result_from_completed,
)
from subprocess import run, CalledProcessError, TimeoutExpired, STDOUT
from ..pset import PSet, Trajectory, BNGLModel, NetModel, run_subprocess
from ..bngsim_model import (
    BngsimModel,
    BngsimNfModel,
    BNGSIM_AVAILABLE,
    BNGSIM_ERROR,
    BNGSIM_BACKEND_NET,
    BNGSIM_BACKEND_NF,
    BNGSIM_BACKEND_HYBRID,
    classify_actions_for_bngsim,
    missing_bngsim_nf_action_support,
)
from ..printing import print0, print1, print2, PybnfError
from ..objective import ObjectiveCalculator

from abc import ABC, abstractmethod
import logging
import numpy as np
import os
import shutil
import copy
import traceback
import pickle
from glob import glob
from concurrent.futures import CancelledError


# Preserve the original module logger name (was getLogger(__name__) in
# algorithms.py) so log records keep the 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


BNGL_BACKEND_AUTO = 'auto'
BNGL_BACKEND_BIONETGEN = 'bionetgen'
BNGL_BACKEND_BNGSIM = 'bngsim'
BNGSIM_SUPPORTED_BNGL_BACKENDS = (BNGSIM_BACKEND_NET, BNGSIM_BACKEND_NF, BNGSIM_BACKEND_HYBRID)


def _bngsim_runtime_available():
    return BNGSIM_AVAILABLE and not os.environ.get('PYBNF_NO_BNGSIM')


def _bngsim_unavailable_reason():
    if os.environ.get('PYBNF_NO_BNGSIM'):
        return 'PYBNF_NO_BNGSIM is set'
    return BNGSIM_ERROR or 'bngsim is not available'


class Algorithm(ABC):
    """Base class for every PyBNF fit type ("method"); defines the run-loop contract.

    **The contract (ADR-0007).** A method plugs into the framework by subclassing
    ``Algorithm`` and implementing exactly two abstract methods:

    * :meth:`start_run` ``() -> list[PSet]`` — the initial parameter sets to evaluate.
    * :meth:`got_result` ``(Result) -> list[PSet] | 'STOP'`` — given one completed
      result, the next parameter sets to evaluate, or the string ``'STOP'`` to end
      the run.

    plus registering itself with a config schema via
    ``@register_fit_type(..., schema=...)`` (ADR-0002, ADR-0005). The ``.name`` of any
    returned PSet, if set, MUST be unique across the run (it determines the
    simulation folder name; uniqueness is not checked elsewhere).

    Everything else the framework needs has an overridable default (``reset``,
    ``add_iterations``, ``cleanup``, ``get_backup_every``, ``should_pickle``). The
    **run loop** itself — job submission, ``as_completed`` draining, backup cadence,
    best-fit save, sim-dir teardown — lives in :meth:`run` and is shared by every
    method, not replaceable: a method customizes *what to propose*, never the outer
    loop.
    """

    # Overridable flag, set True by SimplexAlgorithm. Replaces an
    # ``isinstance(self, SimplexAlgorithm)`` check in run()'s teardown so the
    # base class does not reference a leaf subclass.
    _is_simplex = False

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
        bngl_backend = self.config.config.get('bngl_backend', BNGL_BACKEND_AUTO)
        auto_bngsim = bngl_backend == BNGL_BACKEND_AUTO
        explicit_bngsim = bngl_backend == BNGL_BACKEND_BNGSIM
        allow_bngsim = auto_bngsim or explicit_bngsim
        bngsim_available = _bngsim_runtime_available()
        # Match the subprocess BNGLModel/NetModel behavior: when
        # delete_old_files=0 (keep every per-evaluation file), bngsim-backed
        # models must write .gdat/.scan during execute() so the final-results
        # copy at delete_old_files==0 finds something to copy. Best-fit reruns
        # flip this on explicitly below.
        bngsim_save_files = self.config.config.get('delete_old_files', 1) == 0

        for m in init_model_list:
            bridge_backend = None
            if isinstance(m, BNGLModel):
                bridge_backend = classify_actions_for_bngsim(m.actions)
            missing_nf_support = ()
            if isinstance(m, BNGLModel) and bridge_backend in (BNGSIM_BACKEND_NF, BNGSIM_BACKEND_HYBRID):
                missing_nf_support = missing_bngsim_nf_action_support(m.actions)

            if isinstance(m, BNGLModel) and explicit_bngsim:
                if bridge_backend not in BNGSIM_SUPPORTED_BNGL_BACKENDS:
                    raise PybnfError(
                        'bngl_backend = bngsim was requested for model %s, but its BNGL actions are not '
                        'supported by the bngsim bridge.' % m.name
                    )
                if not bngsim_available:
                    raise PybnfError(
                        'bngl_backend = bngsim was requested for model %s, but %s.' %
                        (m.name, _bngsim_unavailable_reason())
                    )
                if missing_nf_support:
                    raise PybnfError(
                        'bngl_backend = bngsim was requested for model %s, but the installed bngsim '
                        'does not provide %s support.' % (m.name, ', '.join(missing_nf_support))
                    )

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
                        run_subprocess(gn_cmd, timeout=self.config.config['wall_time_gen'], stdout=lf, stderr=STDOUT)
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
                    tb = traceback.format_exc()
                    logger.debug("Other exception occurred:\n%s" % tb)
                    print0("Unknown error occurred during network generation, see log... exiting")
                    exit(1)
                finally:
                    os.chdir(home_dir)

                logger.info('Output for network generation of model %s logged in %s/%s.log' %
                             (m.name, init_dir, gnm_name))
                net_path = init_dir + '/' + gnm_name + '.net'
                use_bngsim = allow_bngsim and bngsim_available and bridge_backend == BNGSIM_BACKEND_NET
                use_hybrid = (
                    allow_bngsim
                    and bngsim_available
                    and bridge_backend in (BNGSIM_BACKEND_NF, BNGSIM_BACKEND_HYBRID)
                    and not missing_nf_support
                )
                if auto_bngsim and bngsim_available and bridge_backend not in BNGSIM_SUPPORTED_BNGL_BACKENDS:
                    logger.info(
                        'Model %s uses actions not supported by the `.net` bngsim bridge; '
                        'falling back to BioNetGen subprocess simulation',
                        m.name,
                    )

                if use_hybrid:
                    # Hybrid path: generate_network already ran; now generate XML
                    # by running BNG2.pl again with generate_network + writeXML
                    logger.info(
                        'Model %s is hybrid (generate_network + NF simulate); '
                        'generating XML for bngsim network-free simulation',
                        m.name,
                    )
                    os.chdir(init_dir)
                    hybrid_name = '%s_gen_hybrid' % m.name
                    m_copy = copy.deepcopy(m)
                    m_copy.actions = ['writeXML()']
                    try:
                        m_copy.save(hybrid_name, pset=default_pset)
                    except Exception as exc:
                        if explicit_bngsim:
                            os.chdir(home_dir)
                            raise PybnfError(
                                'bngl_backend = bngsim was requested for model %s, but staging the '
                                'hybrid BNGL for XML generation failed: %s' % (m.name, exc)
                            )
                        logger.exception(
                            'Failed to stage the hybrid BNGL for model %s. '
                            'Falling back to subprocess simulation.',
                            m.name,
                        )
                        os.chdir(home_dir)
                        final_model_list.append(m)
                        final_model_list[-1].bng_command = m.bng_command
                        continue

                    hybrid_cmd = [self.config.config['bng_command'], '%s.bngl' % hybrid_name]
                    if os.name == 'nt':
                        hybrid_cmd = ['perl'] + hybrid_cmd
                    try:
                        with open('%s.log' % hybrid_name, 'w') as lf:
                            print2('Generating XML for hybrid model %s.bngl' % hybrid_name)
                            run_subprocess(
                                hybrid_cmd,
                                timeout=self.config.config['wall_time_gen'],
                                stdout=lf,
                                stderr=STDOUT,
                            )
                    except (CalledProcessError, TimeoutExpired, Exception) as exc:
                        if explicit_bngsim:
                            raise PybnfError(
                                'bngl_backend = bngsim was requested for model %s, but hybrid XML '
                                'generation failed: %s' % (m.name, exc)
                            )
                        logger.exception(
                            'Hybrid XML generation failed for model %s. '
                            'Falling back to subprocess simulation.',
                            m.name,
                        )
                        os.chdir(home_dir)
                        final_model_list.append(m)
                        final_model_list[-1].bng_command = m.bng_command
                        continue
                    finally:
                        os.chdir(home_dir)

                    xml_path = init_dir + '/' + hybrid_name + '.xml'
                    if not os.path.isfile(xml_path):
                        if explicit_bngsim:
                            raise PybnfError(
                                'bngl_backend = bngsim was requested for model %s, but hybrid XML '
                                'generation did not produce %s.' % (m.name, xml_path)
                            )
                        logger.warning(
                            'XML file not found at %s for model %s. '
                            'Falling back to subprocess simulation.',
                            xml_path,
                            m.name,
                        )
                        final_model_list.append(m)
                        final_model_list[-1].bng_command = m.bng_command
                        continue

                    try:
                        model = BngsimNfModel(
                            m.name,
                            m.actions,
                            m.suffixes,
                            m.mutants,
                            xml_path,
                            bngl_model_lines=m.model_lines,
                            split_line_index=m.split_line_index,
                            param_names=m.param_names,
                            source_dir=os.path.dirname(os.path.abspath(m.file_path)),
                            protocol=m.protocol,
                            save_files=bngsim_save_files,
                        )
                        model.bng_command = m.bng_command
                        final_model_list.append(model)
                    except Exception as exc:
                        if explicit_bngsim:
                            raise PybnfError(
                                'bngl_backend = bngsim was requested for model %s, but bngsim NF '
                                'bridge initialization failed: %s' % (m.name, exc)
                            )
                        logger.exception(
                            'Failed to initialize the bngsim NF bridge for hybrid model %s. '
                            'Falling back to BNGLModel subprocess simulation.',
                            m.name,
                        )
                        final_model_list.append(m)
                        final_model_list[-1].bng_command = m.bng_command
                elif use_bngsim:
                    try:
                        logger.info('Using bngsim for in-process simulation of model %s' % m.name)
                        model = BngsimModel(m.name, m.actions, m.suffixes, m.mutants, nf=net_path,
                                            protocol=m.protocol, save_files=bngsim_save_files)
                    except Exception as exc:
                        if explicit_bngsim:
                            raise PybnfError(
                                'bngl_backend = bngsim was requested for model %s, but bngsim bridge '
                                'initialization failed: %s' % (m.name, exc)
                            )
                        logger.exception(
                            'Failed to initialize bngsim bridge for model %s. Falling back to NetModel.',
                            m.name,
                        )
                        model = NetModel(m.name, m.actions, m.suffixes, m.mutants, nf=net_path)
                    final_model_list.append(model)
                    final_model_list[-1].bng_command = m.bng_command
                else:
                    model = NetModel(m.name, m.actions, m.suffixes, m.mutants, nf=net_path)
                    final_model_list.append(model)
                    final_model_list[-1].bng_command = m.bng_command
            elif isinstance(m, BNGLModel) and allow_bngsim and bridge_backend == BNGSIM_BACKEND_NF:
                if not bngsim_available:
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but %s.' %
                            (m.name, _bngsim_unavailable_reason())
                        )
                    logger.info(
                        'Model %s uses NF actions, but bngsim is not available; '
                        'falling back to BioNetGen subprocess simulation',
                        m.name,
                    )
                    final_model_list.append(m)
                    continue

                if missing_nf_support:
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but the installed bngsim '
                            'does not provide %s support.' % (m.name, ', '.join(missing_nf_support))
                        )
                    logger.info(
                        'Model %s uses NF actions, but the installed bngsim lacks %s support; '
                        'falling back to BioNetGen subprocess simulation',
                        m.name,
                        ', '.join(missing_nf_support),
                    )
                    final_model_list.append(m)
                    continue

                logger.info('Model %s is NF-only; generating XML for bngsim network-free simulation' % m.name)

                if not os.path.isdir(init_dir):
                    logger.debug('Creating initialization directory: %s' % init_dir)
                    os.mkdir(init_dir)
                os.chdir(init_dir)

                gnm_name = '%s_gen_xml' % m.name
                default_pset = PSet([var.set_value(var.default_value) for var in self.variables])
                m_copy = copy.deepcopy(m)
                m_copy.actions = ['writeXML()']
                try:
                    m_copy.save(gnm_name, pset=default_pset)
                except Exception as exc:
                    if explicit_bngsim:
                        os.chdir(home_dir)
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but staging the '
                            'XML-generation BNGL failed: %s' % (m.name, exc)
                        )
                    logger.exception(
                        'Failed to stage the XML-generation BNGL for model %s. '
                        'Falling back to subprocess simulation.',
                        m.name,
                    )
                    os.chdir(home_dir)
                    final_model_list.append(m)
                    continue

                gn_cmd = [self.config.config['bng_command'], '%s.bngl' % gnm_name]
                if os.name == 'nt':
                    gn_cmd = ['perl'] + gn_cmd
                try:
                    with open('%s.log' % gnm_name, 'w') as lf:
                        print2('Generating XML for network-free model %s.bngl' % gnm_name)
                        run_subprocess(
                            gn_cmd,
                            timeout=self.config.config['wall_time_gen'],
                            stdout=lf,
                            stderr=STDOUT,
                        )
                except CalledProcessError as c:
                    logger.error("Command %s failed in directory %s" % (gn_cmd, os.getcwd()))
                    logger.error(c.stdout)
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but XML generation '
                            'failed: %s' % (m.name, c)
                        )
                    logger.warning(
                        'XML generation failed for model %s. Falling back to subprocess simulation.',
                        m.name,
                    )
                    os.chdir(home_dir)
                    final_model_list.append(m)
                    continue
                except TimeoutExpired:
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but XML generation '
                            'timed out.' % m.name
                        )
                    logger.warning(
                        'XML generation timed out for model %s. Falling back to subprocess simulation.',
                        m.name,
                    )
                    os.chdir(home_dir)
                    final_model_list.append(m)
                    continue
                except Exception as exc:
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but XML generation '
                            'failed: %s' % (m.name, exc)
                        )
                    logger.exception(
                        'Unknown error during XML generation for model %s. '
                        'Falling back to subprocess simulation.',
                        m.name,
                    )
                    os.chdir(home_dir)
                    final_model_list.append(m)
                    continue
                finally:
                    os.chdir(home_dir)

                xml_path = init_dir + '/' + gnm_name + '.xml'
                if not os.path.isfile(xml_path):
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but XML generation did '
                            'not produce %s.' % (m.name, xml_path)
                        )
                    logger.warning(
                        'XML file not found at %s for model %s. Falling back to subprocess simulation.',
                        xml_path,
                        m.name,
                    )
                    final_model_list.append(m)
                    continue

                try:
                    model = BngsimNfModel(
                        m.name,
                        m.actions,
                        m.suffixes,
                        m.mutants,
                        xml_path,
                        bngl_model_lines=m.model_lines,
                        split_line_index=m.split_line_index,
                        param_names=m.param_names,
                        source_dir=os.path.dirname(os.path.abspath(m.file_path)),
                        protocol=m.protocol,
                        save_files=bngsim_save_files,
                    )
                    model.bng_command = m.bng_command
                    final_model_list.append(model)
                except Exception as exc:
                    if explicit_bngsim:
                        raise PybnfError(
                            'bngl_backend = bngsim was requested for model %s, but bngsim NF bridge '
                            'initialization failed: %s' % (m.name, exc)
                        )
                    logger.exception(
                        'Failed to initialize the bngsim NF bridge for model %s. '
                        'Falling back to BNGLModel subprocess simulation.',
                        m.name,
                    )
                    final_model_list.append(m)
            else:
                logger.info('Model %s does not require network generation' % m.name)
                final_model_list.append(m)
        os.chdir(home_dir)
        return final_model_list

    @abstractmethod
    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        Algorithm subclasses optionally may set the .name field of the PSet objects to give a meaningful unique
        identifier such as 'gen0ind42'. If so, they MUST BE UNIQUE, as this determines the folder name.
        Uniqueness will not be checked elsewhere.

        :return: list of PSets
        """

    @abstractmethod
    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: result from the completed simulation
        :type res: Result
        :return: List of PSet(s) to be run next or 'STOP' string.
        """

    def add_to_trajectory(self, res):
        """
        Adds the information from a Result to the Trajectory instance
        """
        # Evaluate objective if it wasn't done on workers.
        if res.score is None:  # Check if the objective wasn't evaluated on the workers
            try:
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
                    res.score = self.objective.evaluate_multiple(res.simdata, self.exp_data, res.pset, self.config.constraints)
            except Exception:
                # A failure while normalizing or scoring this one parameter set should
                # penalize this evaluation, not crash the whole run. See lanl/PyBNF#388.
                logger.exception('Objective evaluation failed for Result %s' % res.name)
                res.score = np.inf
                print1('Objective evaluation failed for Result %s; discarding this parameter set' % res.name)
            if res.score is None:  # Check if the above evaluation failed
                res.score = np.inf
                logger.warning('Simulation corresponding to Result %s contained NaNs or Infs' % res.name)
                logger.warning('Discarding Result %s as having an infinite objective function value' % res.name)
                print1('Simulation data in Result %s has NaN or Inf values.  Discarding this parameter set' % res.name)
        logger.debug('Adding Result %s to Trajectory with score %.4f' % (res.name, res.score))
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
        More specifically, the bounded-support (uniform/loguniform) variables follow the latin hypercube
        distribution, while the others are randomized from their prior.

        :param n: Number of psets to generate
        :return:
        """
        logger.debug("Generating PSets using Latin hypercube sampling")
        # Only the bounded-support families (Uniform) are stratified; the prior's
        # inverse CDF (value_from_quantile) handles the per-scale rescale.
        num_uniform_vars = sum(1 for var in self.variables if var.has_bounded_support)

        # Generate latin hypercube of dimension = number of uniformly distributed variables.
        rands = latin_hypercube(n, num_uniform_vars)
        psets = []

        for row in rands:
            # Initialize the variables
            # Convert the 0 to 1 random numbers to the required variable range
            pset_vars = []
            rowindex = 0
            for var in self.variables:
                if var.has_bounded_support:
                    pset_vars.append(var.value_from_quantile(row[rowindex]))
                    rowindex += 1
                else:
                    pset_vars.append(var.sample_value())
            psets.append(PSet(pset_vars))
        return psets

    def make_job(self, params):
        """
        Creates a new Job using the specified params, and additional specifications that are already saved in the
        Algorithm object
        If smoothing or model-level parallelism is turned on, makes grouped subjobs.

        :param params:
        :type params: PSet
        :return: list of Jobs
        """
        if params.name:
            job_id = params.name
        else:
            self.job_id_counter += 1
            job_id = 'sim_%i' % self.job_id_counter
        logger.debug('Creating Job %s' % job_id)
        if self.config.config['smoothing'] > 1 and self.config.config['parallelize_models'] > 1:
            # Create smoothing replicates, and partition each replicate's model list across jobs
            newjobs = []
            replica_subjob_ids = []
            model_count = len(self.model_list)
            rep_count = self.config.config['parallelize_models']
            for rep in range(self.config.config['smoothing']):
                replica_id = '%s_rep%i' % (job_id, rep)
                newnames = []
                for part in range(rep_count):
                    thisname = '%s_part%i' % (replica_id, part)
                    newnames.append(thisname)
                    model_slice = self.model_list[model_count*part//rep_count:model_count*(part+1)//rep_count]
                    newjobs.append(core.Job(model_slice,
                                       params, thisname, self.sim_dir, self.config.config['wall_time_sim'],
                                       self.calc_future, self.config.config['normalization'], dict(),
                                       bool(self.config.config['delete_old_files']),
                                       replicate_index=rep,
                                       stochastic_seed_policy=self.config.config['stochastic_seed']))
                replica_subjob_ids.append((replica_id, newnames))
            new_group = HybridJobGroup(job_id, replica_subjob_ids)
            for n in new_group.subjob_ids:
                self.job_group_dir[n] = new_group
            return newjobs
        elif self.config.config['smoothing'] > 1:
            # Create multiple identical Jobs for use with smoothing
            newjobs = []
            newnames = []
            for i in range(self.config.config['smoothing']):
                thisname = '%s_rep%i' % (job_id, i)
                newnames.append(thisname)
                # calc_future is supposed to be None here - the workers don't have enough info to calculate the
                # objective on their own
                newjobs.append(core.Job(self.model_list, params, thisname,
                                   self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                                   self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files']),
                                   replicate_index=i,
                                   stochastic_seed_policy=self.config.config['stochastic_seed']))
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
                newjobs.append(core.Job(self.model_list[model_count*i//rep_count:model_count*(i+1)//rep_count],
                                   params, thisname, self.sim_dir, self.config.config['wall_time_sim'],
                                   self.calc_future, self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files']),
                                   stochastic_seed_policy=self.config.config['stochastic_seed']))
            new_group = MultimodelJobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs
        else:
            # Create a single job
            return [core.Job(self.model_list, params, job_id,
                    self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                    self.config.config['normalization'], self.config.postprocessing,
                    bool(self.config.config['delete_old_files']),
                    stochastic_seed_policy=self.config.config['stochastic_seed'])]


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

    def _fold_group_result(self, res):
        """Accumulate one completed sub-result into its JobGroup (smoothing /
        model-level parallelism). Returns the combined Result once every sub-job
        in the group has finished, or None if more are still pending.

        Split out of run() so the folding decision can be unit-tested without a
        dask client. See tests/test_run_loop.py.
        """
        group = self.job_group_dir.pop(res.name)
        done = group.job_finished(res)
        if not done:
            return None
        return group.average_results()

    def _record_result_and_decide(self, res):
        """Classify a completed (already group-folded) result, record it in the
        trajectory, and decide what the run loop should do next.

        Returns ``'STOP'`` to end the run, or the list of PSets the algorithm
        wants evaluated next. Raises ``PybnfError`` on a fatal condition (all
        jobs failing with none succeeding, or a cancelled future).

        Split out of run() so these decisions can be unit-tested without a dask
        client. See tests/test_run_loop.py.
        """
        if isinstance(res, FailedSimulation):
            if res.fail_type >= 1:
                self.fail_count += 1
            tb = '\n'+res.traceback if res.fail_type == 1 else ''

            logger.debug('Job %s failed with code %d%s' % (res.name, res.fail_type, tb))
            if res.fail_type >= 1:
                print1('Job %s failed' % res.name)
            else:
                print1('Job %s timed out' % res.name)
            if self.success_count == 0 and self.fail_count >= self.config.config['max_failed_simulations']:
                raise PybnfError('Aborted because all jobs are failing',
                                 'Your simulations are failing to run. Logs from failed simulations are saved in '
                                 'the FailedSimLogs directory. For help troubleshooting this error, refer to '
                                 'https://pybnf.readthedocs.io/en/latest/troubleshooting.html#failed-simulations')
        elif isinstance(res, CancelledError):
            raise PybnfError('PyBNF has encounted a fatel error. If the error has occured on the inital run please varify your model '
                            'is funcational. To resume run please restart PyBNF using the -r flag')
        else:
            self.success_count += 1
            logger.debug('Job %s complete')

        self.add_to_trajectory(res)
        if res.score < self.config.config['min_objective']:
            logger.info('Minimum objective value achieved')
            print1('Minimum objective value achieved')
            return 'STOP'
        response = self.got_result(res)
        if response == 'STOP':
            self.best_fit_obj = self.trajectory.best_score()
            logger.info("Stop criterion satisfied with objective function value of %s" % self.best_fit_obj)
            print1("Stop criterion satisfied with objective function value of %s" % self.best_fit_obj)
            return 'STOP'
        return response

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
            f = client.submit(core.run_job, job, True, self.failed_logs_dir)
            futures.append(f)
            pending[f] = (job.params, job.job_id)
        pool = core.as_completed(futures, with_results=True, raise_errors=False)
        backed_up = True
        while True:
            if sim_count % backup_every == 0 and not backed_up:
                self.backup(set([pending[fut][0] for fut in pending]))
                backed_up = True
            try:
                f, res = next(pool)
            except StopIteration:
                logger.warning('Job pool exhausted unexpectedly — no pending futures remain. '
                               'This can happen when all proposed parameter sets in a generation '
                               'are out of bounds. Ending run.')
                print1('Warning: job pool exhausted (all proposals may have been out of bounds). Ending run.')
                break
            res = result_from_completed(f, res, pending[f][0], pending[f][1])
            del pending[f]
            # For smoothing / model-parallel runs, accumulate sub-results into
            # their group and skip ahead until the group is complete.
            if self.config.config['smoothing'] > 1 or self.config.config['parallelize_models'] > 1:
                res = self._fold_group_result(res)
                if res is None:
                    continue
            sim_count += 1
            backed_up = False
            decision = self._record_result_and_decide(res)
            if decision == 'STOP':
                break
            # Submit the next round of jobs the algorithm asked for.
            new_futures = []
            for ps in decision:
                new_js = self.make_job(ps)
                for new_j in new_js:
                    new_f = client.submit(core.run_job, new_j, (debug or self.fail_count < 10), self.failed_logs_dir)
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
        self._copy_best_fit_sims(best_pset, best_name)
        self._rerun_best_fit_to_save_data(best_pset)
        self._finalize_backup_pickle()
        self._teardown_sim_dir()

        logger.info("Fitting complete")

    def _copy_best_fit_sims(self, best_pset, best_name):
        """Copy the best-fit parameter set's simulation outputs into Results/.

        For each model, save a fresh copy parameterized by ``best_pset``; and when
        ``delete_old_files == 0`` (nothing was cleaned mid-run), also copy each
        suffix's already-written gdat/scan from ``Simulations/<best_name>/`` into
        Results/. A missing best-fit file is logged, not fatal.

        Split out of run()'s tail so the copy orchestration can be unit-tested
        without a dask client. See tests/test_run_loop.py.
        """
        logger.info('Copying simulation results from best fit parameter set to Results/ folder')
        for m in self.config.models:
            this_model = self.config.models[m]
            to_save = this_model.copy_with_param_set(best_pset)
            to_save.save_all('%s/%s_%s' % (self.res_dir, to_save.name, best_name))
            if self.config.config['delete_old_files'] == 0:
                for suffix_entry in this_model.suffixes:
                    # Most model types carry (sim_type, suffix) 2-tuples, but
                    # AnalyticalModel and SbmlModelNoTimeout use plain-string
                    # suffixes. Accept both rather than crashing on the unpack.
                    if isinstance(suffix_entry, (tuple, list)):
                        simtype, suf = suffix_entry
                    else:
                        simtype, suf = 'simulate', suffix_entry
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

    def _rerun_best_fit_to_save_data(self, best_pset):
        """Rerun the best-fit pset so its gdat/scan files land in Results/.

        Only when delete_old_files>0 (the per-suffix copy in _copy_best_fit_sims
        was skipped) and save_best_data is set. Toggles save_files on the in-process
        backends (SBML / Antimony / bngsim BNGL+NF; subprocess BNGLModels write via
        BNG2.pl regardless), submits a single bestfit Job through the core seam, and
        on success copies every gdat/scan into Results/. save_files is restored even
        if the rerun raises, so later bootstrapping/refinement is unaffected.

        Split out of run()'s tail so the rerun orchestration — and the core.Job /
        core.run_job seam (ADR-0001) — can be unit-tested without a dask client.
        See tests/test_run_loop.py.
        """
        if self.config.config['delete_old_files'] > 0 and self.config.config['save_best_data']:
            # Rerun the best fit parameter set so the gdat file(s) are saved in the Results folder.
            logger.info('Rerunning best fit parameter set to save data files.')
            # Enable saving files for in-process backends (SBML / Antimony / bngsim BNGL+NF).
            # Subprocess BNGLModels always write via BNG2.pl so they're skipped here.
            for m in self.model_list:
                if hasattr(m, 'save_files'):
                    m.save_files = True
            finaljob = core.Job(self.model_list, best_pset, 'bestfit',
                           self.sim_dir, self.config.config['wall_time_sim'], None,
                           self.config.config['normalization'], self.config.postprocessing,
                           False,
                           stochastic_seed_policy=self.config.config['stochastic_seed'])
            try:
                core.run_job(finaljob)
            except Exception:
                logger.exception('Failed to rerun best fit parameter set')
                print1('Failed to rerun best fit parameter set. See log for details')
            else:
                # Copy all gdat and scan to Results
                for fname in glob(self.sim_dir+'/bestfit/*.gdat') + glob(self.sim_dir+'/bestfit/*.scan'):
                    shutil.copy(fname, self.res_dir)
            # Restore save_files defaults (in case there is future bootstrapping or refinement)
            for m in self.model_list:
                if hasattr(m, 'save_files'):
                    m.save_files = False

    def _finalize_backup_pickle(self):
        """Rename the periodic backup pickle to its 'finished' name on completion.

        On the final (non-intermediate-bootstrap) pass, alg_backup.bp becomes
        alg_finished.bp — or alg_refine_finished.bp for a Simplex refinement — so a
        resumed run can tell a completed fit from one interrupted mid-flight. A
        missing backup (it was never written) is a warning, not a failure.

        Split out of run()'s tail so the rename can be unit-tested without a dask
        client. See tests/test_run_loop.py.
        """
        if self.bootstrap_number is None or self.bootstrap_number == self.config.config['bootstrap']:
            try:
                os.replace('%s/alg_backup.bp' % self.config.config['output_dir'],
                          '%s/alg_%s.bp' % (self.config.config['output_dir'],
                                            ('finished' if not self.refine else 'refine_finished')))
                logger.info('Renamed pickled algorithm backup to alg_%s.bp' %
                            ('finished' if not self.refine else 'refine_finished'))
            except OSError:
                logger.warning('Tried to move pickled algorithm, but it was not found')

    def _teardown_sim_dir(self):
        """Delete the Simulations/ working directory at the end of a fit.

        Skipped for an intermediate bootstrap replicate (bootstrap_number set) and
        for a non-final pass of a refinement chain (refine==1 on a non-Simplex
        algorithm), and only when delete_old_files>=1. Uses ``rm -rf`` on POSIX
        (more robust than rmtree against partially-written sim trees) and
        shutil.rmtree on Windows.

        Split out of run()'s tail so the teardown decision can be unit-tested
        without a dask client. See tests/test_run_loop.py.
        """
        if (self._is_simplex or self.config.config['refine'] != 1) and self.bootstrap_number is None:
            # End of fitting; delete unneeded files
            if self.config.config['delete_old_files'] >= 1:
                if os.name == 'nt':  # Windows
                    try:
                        shutil.rmtree(self.sim_dir)
                    except OSError:
                        logger.error('Failed to remove simulations directory '+self.sim_dir)
                else:
                    run(['rm', '-rf', self.sim_dir])  # More likely to succeed than rmtree()

    def cleanup(self):
        """
        Called before the program exits due to an exception.
        :return:
        """
        self.output_results('end')


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
