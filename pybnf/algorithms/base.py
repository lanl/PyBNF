"""The Algorithm base class and the module-level helpers it relies on.

Extracted from ``algorithms.py`` (M1 Step 2). The run loop resolves the three
monkeypatched names as ``core.run_job`` / ``core.as_completed`` / ``core.Job``
(ADR-0001), so it imports the ``core`` module object rather than binding those
names locally; the never-patched data classes are imported by name.
"""


from . import core
from . import multistart_report
from .core import (
    FailedSimulation,
    JobGroup,
    MultimodelJobGroup,
    HybridJobGroup,
    result_from_completed,
)
from subprocess import run, CalledProcessError, TimeoutExpired, STDOUT
from ..pset import PSet, Trajectory, BNGLModel, NetModel, run_subprocess, _stage_and_rewrite_tfun_files, _format_bngl_number
from .. import edition
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
from ..objective import ObjectiveCalculator, likelihood_information_criteria
from ..budget import format_duration

from abc import ABC, abstractmethod
import logging
import numpy as np
import os
import shutil
import copy
import traceback
import pickle
import re
from pathlib import Path
from glob import glob
from concurrent.futures import CancelledError


# Preserve the original module logger name (was getLogger(__name__) in
# algorithms.py) so log records keep the 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


BNGL_BACKEND_AUTO = 'auto'
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

    # Overridable flag, set True by the gradient optimizers (#386). When True,
    # run() keeps objective scoring on the master (calc_future = None) so each
    # Result returns with its full simdata -- the per-experiment forward
    # sensitivity tensors the gradient assembly consumes (#385). The worker
    # scoring path nulls res.simdata after scoring (core.Job.run_simulation), so
    # a method that needs simdata back must opt out of it. The same base-class
    # flag pattern as _is_simplex, so run() never references a leaf subclass.
    requires_master_scoring = False

    # Overridable flag, set True by the generational optimizers (de, cmaes, ss).
    # These propose a whole generation of parameter sets, wait for every simulation
    # in it to finish, then propose the next generation. Toward the end of each
    # generation only a few simulations are still running, so some workers sit idle by
    # design. _report_parallelism() says so, to keep that expected idle time from being
    # mistaken for a fit that is using fewer processors than it should (#621). The same
    # base-class flag pattern as _is_simplex, so run() never names a leaf subclass.
    waits_for_full_generation = False

    # Overridable: the name of the setting the parallelism report points at when it
    # advises the user (#655). Most fits size their concurrency from population_size, so
    # that is the default. A fit that follows a different setting names it instead (the
    # local multi-start optimizers name their own n_starts key), and a fit where no
    # single setting controls it sets this to None, which leaves the advice talking only
    # about how many processors to reserve.
    parallelism_setting = 'population_size'

    #: The fit's total wall-clock budget (``wall_time_fit``, #529/ADR-0093), or None for
    #: an unbounded fit. Set by ``pybnf.main()`` on the algorithm it is about to run --
    #: and passed on to a refiner / reused across bootstrap replicates -- so one deadline
    #: bounds the whole run rather than each phase separately. A **class** attribute so an
    #: algorithm unpickled by ``--resume`` (which deliberately does not carry the stale
    #: deadline; see :meth:`should_pickle`) still reads a well-defined None.
    budget = None

    #: Why the last :meth:`run` stopped, when that reason must not be mistaken for
    #: convergence (today: the wall-time budget). None for an ordinary run.
    stop_reason = None

    #: Simulations the last :meth:`run` consumed results from -- what the wall-time stop
    #: reason reports, kept so the run's phase record can report it too (#564). A class
    #: attribute for the same reason ``budget`` is: an unpickled algorithm reads 0.
    completed_simulations = 0

    #: The run's executed-method record (``Results/method_chain.json``, #564/ADR-0107),
    #: or None when nothing is keeping one. Like ``budget``, it is attached by
    #: ``pybnf.main()`` and shared with the refiner, so one file describes the whole run.
    method_chain = None

    #: The noise scales this fit profiled out of the search, at the best fit
    #: (``{name: sigma_hat}``, ``noise_profiling = 1``, ADR-0108) -- captured by the
    #: end-of-run tail when it scores the best point, and written to
    #: ``Results/profiled_noise.txt``. Empty for a fit that profiles nothing (the default).
    #: A **class** attribute, replaced rather than mutated, so an unpickled algorithm -- and
    #: the run loop's own tail -- always read a well-defined empty map.
    _profiled_noise = {}

    #: Name of the parameter set ``Results/information_criteria_backup.txt`` currently
    #: describes (#560), or None while no checkpoint has been written. The checkpoint costs
    #: one re-simulation, so it is skipped while this still names the best fit -- the file on
    #: disk already describes it. A **class** attribute for the same reason ``_profiled_noise``
    #: is: an algorithm unpickled from a backup written before this existed reads a
    #: well-defined None and simply writes its first checkpoint.
    _ic_checkpoint_name = None

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
            self.sim_dir = str(Path(self.config.config['simulation_dir']) / 'Simulations')
        else:
            self.sim_dir = str(Path(self.config.config['output_dir']) / 'Simulations')
        self.res_dir = str(Path(self.config.config['output_dir']) / 'Results')
        self.failed_logs_dir = str(Path(self.config.config['output_dir']) / 'FailedSimLogs')

        # Generate a list of variable names. Under noise_profiling (ADR-0108) this is the
        # SEARCHED subset: the estimated noise scales the objective solves for analytically
        # have been partitioned out into config.profiled_noise_params, so every algorithm
        # builds its box, population and PSets over the reduced dimension without knowing
        # profiling exists. They remain estimated parameters for AIC/BIC (see
        # _compute_information_criteria) and are reported by _emit_profiled_noise.
        self.variables = self.config.variables

        # Store a list of all Model objects. Change this as needed for compatibility with other parts
        logger.debug('Initializing models')
        self.model_list = self._initialize_models()

        self.bootstrap_number = None
        # Retry counter for the current bootstrap replicate (0 on the first attempt).
        # _run_bootstrapping bumps it before re-resetting a rejected replicate so reset()
        # advances the RNG sub-stream and the retry resamples afresh (see reset()).
        self.bootstrap_attempt = 0
        self.best_fit_obj = None
        self.calc_future = None  # Created during Algorithm.run()
        self.models_future = None  # Scattered model_list Future; created during Algorithm.run()
        self.refine = False

        # Random number generation. Every algorithm owns its own
        # np.random.Generator (PCG64) seeded from the run's resolved seed; the
        # legacy global np.random (MT19937) is no longer used. The seed was
        # resolved + logged by pybnf._initialize_random_seed before construction.
        self._init_rng()

    def _init_rng(self):
        """Build this algorithm's random streams from the resolved config seed.

        ``self.rng`` is the root Generator used for algorithm-level draws and for
        cross-chain coordination (replica exchange, outlier reset) that happens at
        deterministic synchronization barriers. ``self._seed_sequence`` spawns
        independent per-chain Generators on demand (:meth:`spawn_chain_rngs`).
        """
        self._base_seed = self.config.config['random_seed']
        self._reseed(np.random.SeedSequence(self._base_seed))

    def _reseed(self, seed_sequence):
        """Point this algorithm's RNG at a (possibly fresh) SeedSequence.

        Used both at construction and per bootstrap replicate, where each replicate
        gets an independent, deterministic sub-stream so its fit is reproducible
        from the run seed yet distinct across replicates.
        """
        self._seed_sequence = seed_sequence
        self.rng = np.random.default_rng(seed_sequence)

    def spawn_chain_rngs(self, n):
        """Return ``n`` independent Generators, one per parallel chain/replica.

        Spawning is deterministic in ``(seed, n)``, so chain ``i`` always draws
        from the same stream regardless of the order in which dask returns its
        results. This is what makes the parallel samplers reproducible under
        nondeterministic scheduling -- a single shared stream would interleave by
        completion order and so would not reproduce run to run.
        """
        return [np.random.default_rng(s) for s in self._seed_sequence.spawn(n)]

    def _rebuild_chain_rngs(self):
        """Hook: rebuild any per-chain Generators after a :meth:`_reseed`.

        A no-op for the single-root-Generator optimizers; :class:`BayesianAlgorithm`
        overrides it to re-spawn its ``chain_rngs`` list.
        """

    def reset(self, bootstrap):
        """
        Resets the Algorithm, keeping loaded variables and models

        :param bootstrap: The bootstrap number (None if not bootstrapping)
        :type bootstrap: int or None
        :return:

        A rejected bootstrap replicate (objective over ``bootstrap_max_obj``) is retried
        with :attr:`bootstrap_attempt` incremented by the caller; that advances the RNG
        sub-stream so the retry draws a *fresh* resample and fit rather than repeating the
        identical failing run. ``bootstrap_attempt == 0`` (the first try, and every
        non-bootstrap reset) reproduces the historical replicate-only seeding byte for byte.
        """
        logger.info('Resetting Algorithm for another run')
        self.trajectory = Trajectory(self.config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0
        # The next run writes its own Results directory, so nothing there describes a
        # best fit yet -- and PSet names restart from iter0run0, so a name carried over
        # from the previous run would suppress the first checkpoint (#560).
        self._ic_checkpoint_name = None
        self.job_group_dir = dict()
        self.fail_count = 0
        self.success_count = 0

        if bootstrap is not None:
            self.bootstrap_number = bootstrap

            self.sim_dir = self.config.config['output_dir'] + f'/Simulations-boot{bootstrap}'
            self.res_dir = self.config.config['output_dir'] + f'/Results-boot{bootstrap}'
            self.failed_logs_dir = self.config.config['output_dir'] + f'/FailedSimLogs-boot{bootstrap}'
            for boot_dir in (self.sim_dir, self.res_dir, self.failed_logs_dir):
                if os.path.exists(boot_dir):
                    try:
                        shutil.rmtree(boot_dir)
                    except OSError:
                        logger.error('Failed to remove bootstrap directory '+boot_dir)
                os.mkdir(boot_dir)

            # Give this bootstrap replicate an independent, deterministic RNG
            # sub-stream (keyed by the replicate number, plus the retry counter so a
            # retried replicate resamples afresh) so the replicate's fit -- and its
            # resampled data weights -- are reproducible from the run seed yet distinct
            # from the main fit and from every other replicate/retry. The first attempt
            # keeps the historical replicate-only spawn key for backward compatibility.
            attempt = self.bootstrap_attempt
            spawn_key = (bootstrap + 1,) if attempt == 0 else (bootstrap + 1, attempt)
            self._reseed(np.random.SeedSequence(self._base_seed, spawn_key=spawn_key))
            self._rebuild_chain_rngs()

        self.best_fit_obj = None

    def _param_vec(self, pset):
        """Project a PSet onto its parameter vector in sampling space ``u``,
        ordered by ``self.variables`` (``log10`` for log-scaled parameters,
        identity otherwise).

        The single PSet→u bridge: the Bayesian samplers use it for chain history
        and proposal arithmetic, the start-point optimizers for their search
        coordinate (where it is also exposed under the name ``_u_from_pset``).
        Hoisted here from ``BayesianAlgorithm`` once the start-point optimizers
        grew the identical transform (the ≥2-user event, ADR-0009). Asks each
        parameter for its θ→u transform rather than inlining ``log10`` (#412).
        """
        return np.array(
            [v.to_sampling_space(pset[v.name]) for v in self.variables], dtype=float)

    def _pset_from_u(self, u, name=None, reflect=True):
        """Materialize a sampling-space vector ``u`` (ordered by ``self.variables``)
        into a PSet -- the inverse peer of :meth:`_param_vec`.

        Each coordinate is mapped back to a stored value by its parameter
        (``FreeParameter.from_sampling_space``) and assigned with ``set_value``,
        which folds it into the box when ``reflect`` is True. ``reflect=False``
        lets a caller reject an out-of-bounds proposal instead of folding it: the
        offending ``set_value`` raises ``OutOfBoundsException`` (DREAM relies on
        this to discard a proposal rather than reflect it).

        Hoisted here next to the forward bridge so the u-vector↔PSet conversion
        lives in one place (#412); the start-point optimizers reach it through the
        ``_u_from_pset`` / ``_pset_from_u`` alias pair in ``local_base``.
        """
        fps = [v.set_value(v.from_sampling_space(u[i]), reflect)
               for i, v in enumerate(self.variables)]
        ps = PSet(fps)
        if name is not None:
            ps.name = name
        return ps

    @staticmethod
    def _chain_index_from_name(name):
        """Parse the parallel-run index from a PSet name of the form
        ``...run<N>...`` — the ``iter%irun%i`` naming the population samplers
        and Simulated Annealing assign their per-chain PSets."""
        return int(re.search(r'(?<=run)\d+', name).group(0))

    @staticmethod
    def should_pickle(k):
        """
        Checks to see if key 'k' should be included in pickling.  Currently allows all entries in instance dictionary
        except for 'trajectory'

        'budget' is excluded too: a wall-clock deadline is meaningless once restored into
        a later process, and a resumed run is a new run of its own -- ``main()`` builds it
        a fresh budget from the same ``wall_time_fit``, so ``--resume`` grants another full
        budget rather than inheriting an already-expired one (#529). 'method_chain' is
        excluded for the same reason: it records the *phases this process executed*, and
        ``main()`` builds the resumed run its own (#564).

        :param k:
        :return:
        """
        return k not in set(['trajectory', 'calc_future', 'models_future', 'budget',
                             'method_chain'])

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if self.should_pickle(k)}

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            backup_params = 'sorted_params_backup.txt' if not self.refine else 'sorted_params_refine_backup.txt'
            self.trajectory = Trajectory.load_trajectory(f'{self.res_dir}/{backup_params}',
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
        init_dir = str(Path(os.getcwd()) / 'Initialize')
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
                        f'bngl_backend = bngsim was requested for model {m.name}, but its BNGL actions are not '
                        'supported by the bngsim bridge.'
                    )
                if not bngsim_available:
                    raise PybnfError(
                        f'bngl_backend = bngsim was requested for model {m.name}, but {_bngsim_unavailable_reason()}.'
                    )
                if missing_nf_support:
                    raise PybnfError(
                        'bngl_backend = bngsim was requested for model {}, but the installed bngsim '
                        'does not provide {} support.'.format(m.name, ', '.join(missing_nf_support))
                    )

            if isinstance(m, BNGLModel) and m.generates_network:
                logger.debug(f'Model {m.name} requires network generation')

                if not os.path.isdir(init_dir):
                    logger.debug(f'Creating initialization directory: {init_dir}')
                    os.mkdir(init_dir)
                os.chdir(init_dir)

                gnm_name = f'{m.name}_gen_net'
                default_pset = PSet([var.set_value(var.default_value) for var in self.variables])
                m.save(gnm_name, gen_only=True, pset=default_pset)
                gn_cmd = [self.config.config['bng_command'], f'{gnm_name}.bngl']
                if os.name == 'nt':  # Windows
                    # Explicitly call perl because the #! line in BNG2.pl is not supported.
                    gn_cmd = ['perl'] + gn_cmd
                try:
                    with open(f'{gnm_name}.log', 'w') as lf:
                        print2(f'Generating network for model {gnm_name}.bngl')
                        run_subprocess(gn_cmd, timeout=self.config.config['wall_time_gen'], stdout=lf, stderr=STDOUT)
                except CalledProcessError as c:
                    logger.error(f"Command {gn_cmd} failed in directory {os.getcwd()}")
                    logger.error(c.stdout)
                    print0(f'Error: Initial network generation failed for model {m.name}... see BioNetGen error log at '
                           f'{os.getcwd()}/{gnm_name}.log')
                    exit(1)
                except TimeoutExpired:
                    logger.debug("Network generation exceeded %d seconds... exiting" %
                                  self.config.config['wall_time_gen'])
                    print0("Network generation took too long.  Increase 'wall_time_gen' configuration parameter")
                    exit(1)
                except:
                    tb = traceback.format_exc()
                    logger.debug(f"Other exception occurred:\n{tb}")
                    print0("Unknown error occurred during network generation, see log... exiting")
                    exit(1)
                finally:
                    os.chdir(home_dir)

                logger.info(f'Output for network generation of model {m.name} logged in {init_dir}/{gnm_name}.log')
                net_path = str(Path(init_dir) / f'{gnm_name}.net')
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
                    hybrid_name = f'{m.name}_gen_hybrid'
                    m_copy = copy.deepcopy(m)
                    m_copy.actions = ['writeXML()']
                    try:
                        m_copy.save(hybrid_name, pset=default_pset)
                    except Exception as exc:
                        if explicit_bngsim:
                            os.chdir(home_dir)
                            raise PybnfError(
                                f'bngl_backend = bngsim was requested for model {m.name}, but staging the '
                                f'hybrid BNGL for XML generation failed: {exc}'
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

                    hybrid_cmd = [self.config.config['bng_command'], f'{hybrid_name}.bngl']
                    if os.name == 'nt':
                        hybrid_cmd = ['perl'] + hybrid_cmd
                    try:
                        with open(f'{hybrid_name}.log', 'w') as lf:
                            print2(f'Generating XML for hybrid model {hybrid_name}.bngl')
                            run_subprocess(
                                hybrid_cmd,
                                timeout=self.config.config['wall_time_gen'],
                                stdout=lf,
                                stderr=STDOUT,
                            )
                    except (CalledProcessError, TimeoutExpired, Exception) as exc:
                        if explicit_bngsim:
                            raise PybnfError(
                                f'bngl_backend = bngsim was requested for model {m.name}, but hybrid XML '
                                f'generation failed: {exc}'
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

                    xml_path = str(Path(init_dir) / f'{hybrid_name}.xml')
                    if not os.path.isfile(xml_path):
                        if explicit_bngsim:
                            raise PybnfError(
                                f'bngl_backend = bngsim was requested for model {m.name}, but hybrid XML '
                                f'generation did not produce {xml_path}.'
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
                                f'bngl_backend = bngsim was requested for model {m.name}, but bngsim NF '
                                f'bridge initialization failed: {exc}'
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
                        logger.info(f'Using bngsim for in-process simulation of model {m.name}')
                        model = BngsimModel(m.name, m.actions, m.suffixes, m.mutants, nf=net_path,
                                            protocol=m.protocol, save_files=bngsim_save_files)
                    except Exception as exc:
                        if explicit_bngsim:
                            raise PybnfError(
                                f'bngl_backend = bngsim was requested for model {m.name}, but bngsim bridge '
                                f'initialization failed: {exc}'
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
                            f'bngl_backend = bngsim was requested for model {m.name}, but {_bngsim_unavailable_reason()}.'
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
                            'bngl_backend = bngsim was requested for model {}, but the installed bngsim '
                            'does not provide {} support.'.format(m.name, ', '.join(missing_nf_support))
                        )
                    logger.info(
                        'Model %s uses NF actions, but the installed bngsim lacks %s support; '
                        'falling back to BioNetGen subprocess simulation',
                        m.name,
                        ', '.join(missing_nf_support),
                    )
                    final_model_list.append(m)
                    continue

                logger.info(f'Model {m.name} is NF-only; generating XML for bngsim network-free simulation')

                if not os.path.isdir(init_dir):
                    logger.debug(f'Creating initialization directory: {init_dir}')
                    os.mkdir(init_dir)
                os.chdir(init_dir)

                gnm_name = f'{m.name}_gen_xml'
                default_pset = PSet([var.set_value(var.default_value) for var in self.variables])
                m_copy = copy.deepcopy(m)
                m_copy.actions = ['writeXML()']
                try:
                    m_copy.save(gnm_name, pset=default_pset)
                except Exception as exc:
                    if explicit_bngsim:
                        os.chdir(home_dir)
                        raise PybnfError(
                            f'bngl_backend = bngsim was requested for model {m.name}, but staging the '
                            f'XML-generation BNGL failed: {exc}'
                        )
                    logger.exception(
                        'Failed to stage the XML-generation BNGL for model %s. '
                        'Falling back to subprocess simulation.',
                        m.name,
                    )
                    os.chdir(home_dir)
                    final_model_list.append(m)
                    continue

                gn_cmd = [self.config.config['bng_command'], f'{gnm_name}.bngl']
                if os.name == 'nt':
                    gn_cmd = ['perl'] + gn_cmd
                try:
                    with open(f'{gnm_name}.log', 'w') as lf:
                        print2(f'Generating XML for network-free model {gnm_name}.bngl')
                        run_subprocess(
                            gn_cmd,
                            timeout=self.config.config['wall_time_gen'],
                            stdout=lf,
                            stderr=STDOUT,
                        )
                except CalledProcessError as c:
                    logger.error(f"Command {gn_cmd} failed in directory {os.getcwd()}")
                    logger.error(c.stdout)
                    if explicit_bngsim:
                        raise PybnfError(
                            f'bngl_backend = bngsim was requested for model {m.name}, but XML generation '
                            f'failed: {c}'
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
                            f'bngl_backend = bngsim was requested for model {m.name}, but XML generation '
                            'timed out.'
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
                            f'bngl_backend = bngsim was requested for model {m.name}, but XML generation '
                            f'failed: {exc}'
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

                xml_path = str(Path(init_dir) / f'{gnm_name}.xml')
                if not os.path.isfile(xml_path):
                    if explicit_bngsim:
                        raise PybnfError(
                            f'bngl_backend = bngsim was requested for model {m.name}, but XML generation did '
                            f'not produce {xml_path}.'
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
                            f'bngl_backend = bngsim was requested for model {m.name}, but bngsim NF bridge '
                            f'initialization failed: {exc}'
                        )
                    logger.exception(
                        'Failed to initialize the bngsim NF bridge for model %s. '
                        'Falling back to BNGLModel subprocess simulation.',
                        m.name,
                    )
                    final_model_list.append(m)
            else:
                logger.info(f'Model {m.name} does not require network generation')
                final_model_list.append(m)
        os.chdir(home_dir)
        # Off-diagonal cross-product pruning (#484, ADR-0069): attach each model's emit-set
        # (the full output suffixes any consumer reads) to the *runtime* model, so the
        # backend execute() simulates only the scored (action, condition) diagonal. Set here
        # rather than in config because this method rebuilds the models (a fresh
        # BngsimModel/NetModel per model drops config-set attributes -- the same reason the
        # gradient path sets set_scored_suffixes on the model_list, not in config). A model
        # config left out of emit_suffixes gets None -> pruning off -> byte-identical. Rides
        # copy.copy/scatter to the workers alongside the model.
        emit_suffixes = getattr(self.config, 'emit_suffixes', {}) or {}
        if emit_suffixes:  # only edition-2 Mechanism-A jobs; else models keep the None default
            for model in final_model_list:
                model.emit_suffixes = emit_suffixes.get(model.name)
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
                logger.exception(f'Objective evaluation failed for Result {res.name}')
                res.score = np.inf
                print1(f'Objective evaluation failed for Result {res.name}; discarding this parameter set')
            if res.score is None:  # Check if the above evaluation failed
                res.score = np.inf
                logger.warning(f'Simulation corresponding to Result {res.name} contained NaNs or Infs')
                logger.warning(f'Discarding Result {res.name} as having an infinite objective function value')
                print1(f'Simulation data in Result {res.name} has NaN or Inf values.  Discarding this parameter set')
        logger.debug(f'Adding Result {res.name} to Trajectory with score {res.score:.4f}')
        self.trajectory.add(res.pset, res.score, res.name)

    def random_pset(self):
        """
        Generates a random PSet based on the distributions and bounds for each parameter specified in the configuration

        :return:
        """
        logger.debug("Generating a randomly distributed PSet")
        pset_vars = []
        for var in self.variables:
            pset_vars.append(var.sample_initial_value(self.rng))
        return PSet(pset_vars)

    def _seed_start_point_pset(self, sampled_pset):
        """Pin the parameters that declare a start point to it, in ONE pset.

        ADR-0043 Phase 2, re-pointed at the unified carrier by ADR-0117. The declared start
        point -- ``start_point = <p> <v>`` lines and ``parameter:`` records'
        ``initial_value:`` fields alike, resolved and validated into
        ``Configuration.start_point`` -- is where the search/walk should begin (a PEtab
        ``nominalValue``, a published optimum). Exactly one member of a population
        algorithm's initial population is seeded there: this helper takes a member already
        drawn from the prior/bounds and overwrites each parameter that declares a start,
        leaving every other parameter at its sampled draw. So a partially-specified seed is
        still a complete pset -- pinned where a start was given, drawn otherwise.

        Returns ``sampled_pset`` unchanged when nothing declares a start point (the common
        case), so runs without one are byte-for-byte unaffected. Only this one member is
        ever touched, so the rest of the population keeps the full diversity global search
        (de/pso/ss) needs; this is distinct from the sampler-only ``starting_params``, which
        overrode *every* chain uniformly.

        Reading the config's resolved dict rather than ``FreeParameter.value`` is what makes
        the carrier universal: ``.value`` is unset for a no-prior record (whose start the
        loader folds into ``p1`` in sampling space instead) and unreachable from a legacy
        ``*_var`` conf, so it could never have carried the start point for every declaration
        style. Values in the dict are already validated against the declared box, so
        ``set_value`` here can never fold one.
        """
        declared = getattr(self.config, 'start_point', None) or {}
        if not declared:
            return sampled_pset
        # Only the FIRST start is pinned. ``MultiStartOptimizer`` re-enters the inner
        # search's start_run once per start, so without this gate a declared start point
        # would re-seed one member of every start's population -- turning an n_starts
        # multi-start into n partially degenerate searches, and disagreeing with the
        # start-point optimizers, where a declared start pins start 0 and leaves the rest
        # scattered. ``_start_index`` is absent on a single-start algorithm, hence the 0
        # default (#583).
        if getattr(self, '_start_index', 0) != 0:
            return sampled_pset
        fps = [v.set_value(declared[v.name], reflect=False) if v.name in declared
               else sampled_pset.get_param(v.name)
               for v in self.variables]
        seeded = PSet(fps)
        seeded.name = sampled_pset.name
        return seeded

    def random_latin_hypercube_psets(self, n):
        """
        Generates n random PSets with a latin hypercube distribution. Variables
        with bounded initialization distributions follow the Latin hypercube;
        the others are randomized independently from their initializer.

        :param n: Number of psets to generate
        :return:
        """
        logger.debug("Generating PSets using Latin hypercube sampling")
        num_uniform_vars = sum(1 for var in self.variables
                               if var.has_bounded_initialization)

        # Generate latin hypercube of dimension = number of uniformly distributed variables.
        rands = latin_hypercube(n, num_uniform_vars, self.rng)
        psets = []

        for row in rands:
            # Initialize the variables
            # Convert the 0 to 1 random numbers to the required variable range
            pset_vars = []
            rowindex = 0
            for var in self.variables:
                if var.has_bounded_initialization:
                    pset_vars.append(var.initial_value_from_quantile(row[rowindex]))
                    rowindex += 1
                else:
                    pset_vars.append(var.sample_initial_value(self.rng))
            psets.append(PSet(pset_vars))
        return psets

    def _job_models(self, model_slice=None):
        """Return the ``(models, model_slice)`` pair to hand a Job.

        Prefer the model_list Future scattered once per fit in :meth:`run` (so a
        submit re-pickles only a lightweight Future, not the whole model graph);
        the Job resolves and slices it worker-side. Fall back to a concrete
        (already-sliced) list when no scatter is available -- e.g. make_job
        called outside run() (some tests). ``model_slice`` is ``(start, stop)``
        or ``None`` for the whole list. See issue #416.
        """
        if self.models_future is not None:
            return self.models_future, model_slice
        if model_slice is None:
            return self.model_list, None
        return self.model_list[model_slice[0]:model_slice[1]], None

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
        logger.debug(f'Creating Job {job_id}')
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
                    models, mslice = self._job_models(
                        (model_count*part//rep_count, model_count*(part+1)//rep_count))
                    newjobs.append(core.Job(models,
                                       params, thisname, self.sim_dir, self.config.config['wall_time_sim'],
                                       self.calc_future, self.config.config['normalization'], dict(),
                                       bool(self.config.config['delete_old_files']),
                                       replicate_index=rep,
                                       stochastic_seed_policy=self.config.config['stochastic_seed'],
                                       model_slice=mslice))
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
                models, mslice = self._job_models()
                newjobs.append(core.Job(models, params, thisname,
                                   self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                                   self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files']),
                                   replicate_index=i,
                                   stochastic_seed_policy=self.config.config['stochastic_seed'],
                                   model_slice=mslice))
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
                models, mslice = self._job_models(
                    (model_count*i//rep_count, model_count*(i+1)//rep_count))
                newjobs.append(core.Job(models,
                                   params, thisname, self.sim_dir, self.config.config['wall_time_sim'],
                                   self.calc_future, self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files']),
                                   stochastic_seed_policy=self.config.config['stochastic_seed'],
                                   model_slice=mslice))
            new_group = MultimodelJobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs
        else:
            # Create a single job
            models, mslice = self._job_models()
            return [core.Job(models, params, job_id,
                    self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                    self.config.config['normalization'], self.config.postprocessing,
                    bool(self.config.config['delete_old_files']),
                    stochastic_seed_policy=self.config.config['stochastic_seed'],
                    model_slice=mslice)]


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
        alias = None
        if self.refine:
            # A refine's END-OF-RUN output is the RUN's end-of-run output, so it is
            # written under the conventional name too, not only the refine_-prefixed one.
            # Historically sorted_params_final.txt kept the PRE-refine point while
            # information_criteria.txt (which the refiner rewrites from the same tail)
            # described the refined one -- two files in Results/ disagreeing about which
            # parameter set they describe, with the conventional name carrying the point
            # the user's requested method chain did not end on (#564).
            if name == 'final':
                alias = 'final'
            name = f'refine_{name}'
        filepath = f'{self.res_dir}/sorted_params_{name}.txt'
        logger.info(f'Outputting results to file {filepath}')
        self.trajectory.write_to_file(filepath)

        # If the user has asked for fewer output files, each time we're here, move the new file to
        # Results/sorted_params.txt, overwriting the previous one.
        if self.config.config['delete_old_files'] >= 2 and not no_move:
            logger.debug("Overwriting previous 'sorted_params.txt'")
            noname_filepath = f'{self.res_dir}/sorted_params.txt'
            if os.path.isfile(noname_filepath):
                os.remove(noname_filepath)
            os.replace(filepath, noname_filepath)
            # That single file is already the alias: everything collapses into it.
        elif alias is not None:
            alias_path = f'{self.res_dir}/sorted_params_{alias}.txt'
            logger.info(f'Also outputting the refined results to {alias_path}')
            self.trajectory.write_to_file(alias_path)

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
        picklepath = '{}/alg_backup.bp'.format(self.config.config['output_dir'])
        temppicklepath = '{}/alg_backup_temp.bp'.format(self.config.config['output_dir'])
        try:
            with open(temppicklepath, 'wb') as f:
                pickle.dump((self, pending_psets), f)
            os.replace(temppicklepath, picklepath)
        except IOError as e:
            logger.exception('Failed to save backup of algorithm')
            print1('Failed to save backup of the algorithm.\nSee log for more information')
            if e.strerror == 'Too many open files':
                print0('Too many open files! See "Troubleshooting" in the documentation for how to deal with this '
                       'problem.')

        # Then the other half of a scoreable result: the absolute log-likelihood behind the
        # information criteria, which lives in no parameter table (#560). LAST, because it is
        # the only part of a checkpoint that can take a while -- it re-simulates the best fit
        # -- and a kill during it must not leave the resume state a whole interval behind the
        # parameter sets it belongs with.
        self._checkpoint_information_criteria()

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
                print1(f'Job {res.name} failed')
            else:
                print1(f'Job {res.name} timed out')
            if self.success_count == 0 and self.fail_count >= self.config.config['max_failed_simulations']:
                raise PybnfError('Aborted because all jobs are failing',
                                 'Your simulations are failing to run. Logs from failed simulations are saved in '
                                 'the FailedSimLogs directory. For help troubleshooting this error, refer to '
                                 'https://lanl.github.io/PyBNF/troubleshooting.html#failed-simulations')
        elif isinstance(res, CancelledError):
            raise PybnfError('PyBNF has encountered a fatal error. If the error has occurred on the initial run please verify your model '
                            'is functional. To resume the run please restart PyBNF using the -r flag')
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
            logger.info(f"Stop criterion satisfied with objective function value of {self.best_fit_obj}")
            print1(f"Stop criterion satisfied with objective function value of {self.best_fit_obj}")
            return 'STOP'
        return response

    def run(self, client, resume=None, debug=False):
        """Main loop for executing the algorithm.

        Submits work, drains completions through :meth:`_drain_job_pool`, and then runs
        the end-of-fit path -- final parameter sets, best-fit simulations, information
        criteria, the inference-data sidecar, the backup rename, the sim-dir teardown.
        That path is reached the same way however the loop ended: on the algorithm's own
        stop criterion, on an exhausted job pool, or on the wall-time budget
        (``wall_time_fit``, #529). A budgeted run is a *finished* run whose search was
        cut short, so it writes exactly what a converged one writes; only
        :attr:`stop_reason` differs (ADR-0093).
        """

        self.stop_reason = None
        self.completed_simulations = 0
        if self.refine:
            logger.debug('Setting up Simplex refinement of previous algorithm')

        backup_every = self.get_backup_every()

        logger.debug('Generating initial parameter sets')
        if resume:
            psets = resume
            logger.debug('Resume algorithm with the following PSets: %s' % [p.name for p in resume])
        else:
            psets = self.start_run()

        # Record where this fit actually began, before anything is scored (#583, ADR-0117).
        # Placed here because it is the one point every family passes through with its start
        # in hand: the start-point optimizers resolved theirs in __init__, the populations and
        # samplers in the start_run() just above, and a resumed run has none of either.
        self._emit_start_point(psets, resumed=bool(resume))

        if not os.path.isdir(self.failed_logs_dir):
            os.mkdir(self.failed_logs_dir)

        # Decide where the objective is scored. The default offloads scoring to
        # the workers (scatter an ObjectiveCalculator) for parallelism. A
        # gradient optimizer instead needs every Result's simdata back on the
        # master to assemble the residual Jacobian (#386), so it forces
        # master-side scoring via requires_master_scoring -- the worker path
        # would otherwise null res.simdata after scoring (#385).
        if self.config.config['local_objective_eval'] == 0 and self.config.config['smoothing'] == 1 and \
                self.config.config['parallelize_models'] == 1 and not self.requires_master_scoring:
            calculator = ObjectiveCalculator(self.objective, self.exp_data, self.config.constraints)
            [self.calc_future] = client.scatter([calculator], broadcast=True)
        else:
            self.calc_future = None

        # Scatter the parameter-independent model_list once and broadcast it to
        # every worker, so each job submission carries a lightweight Future
        # instead of re-serializing the whole model graph on every evaluation
        # (parameters live in the separate PSet). Jobs resolve and slice it
        # worker-side; see make_job and core.Job._get_models. Mirrors the
        # calc_future scatter above (issue #416).
        #
        # Both scattered Futures (models_future here, calc_future above) are
        # handed to client.submit below as *direct kwargs* -- NOT left buried as
        # Job attributes. dask only substitutes a Future for its concrete value
        # when the Future is a direct submit arg; a Future pickled inside the Job
        # deserializes broken on the worker under distributed >= 2026.6.0 and
        # raises on .result() (lanl/PyBNF #476). core.run_job rebinds the resolved
        # values onto the Job. See core._ResolvedFuture.
        [self.models_future] = client.scatter([self.model_list], broadcast=True)

        pending = dict()  # Maps pending futures to tuple (PSet, job_id).
        if self._budget_spent():
            # The budget was already gone before a single job was submitted (model
            # loading / network generation ate it). Launch nothing -- an expired budget
            # means "stop launching new work" -- and go straight to the finalize path,
            # which reports whatever this run has (nothing, on a fresh fit).
            self.stop_reason = self._wall_time_stop_reason(0)
        else:
            jobs = []
            for p in psets:
                jobs += self.make_job(p)
            jobs[0].show_warnings = True  # For only the first job submitted, show warnings if exp data is unused.
            logger.info('Submitting initial set of %d Jobs' % len(jobs))
            futures = []
            for job in jobs:
                f = client.submit(core.run_job, job, True, self.failed_logs_dir,
                                  models=self.models_future, calc=self.calc_future)
                futures.append(f)
                pending[f] = (job.params, job.job_id)
            # With a wall-time budget, bound the blocking wait for the next completion by
            # what is left of it, so the deadline lands on time even while every worker is
            # mid-simulation: as_completed then raises TimeoutError out of next() instead
            # of blocking until a simulation happens to finish (#529). Without a budget the
            # call is exactly the historical one.
            pool_kwargs = {'timeout': self.budget.remaining()} if self.budget is not None else {}
            pool = core.as_completed(futures, with_results=True, raise_errors=False, **pool_kwargs)
            self._report_parallelism(client, len(futures), len(psets))
            self.completed_simulations = self._drain_job_pool(client, pool, pending, backup_every, debug)

        logger.info("Cancelling %d pending jobs" % len(pending))
        client.cancel(list(pending.keys()))
        self._finalize_run()

    def expected_parallelism(self):
        """How many parameter sets this fit has out for evaluation at once once it is under
        way, or None when that is however many the run loop submits in its first batch
        (#655).

        The run loop knows only the size of the first batch of jobs it submits. For most
        fits that is also how many run at once for the rest of the fit, so the default
        here is None and :meth:`_report_parallelism` uses the first batch. A fit whose
        first batch is a one-time initialization round of a different size overrides this
        and returns the number it settles at, so the parallelism report describes the fit
        rather than its opening move. Scatter search is the case that prompted this: it
        starts with ``init_size`` parameter sets and then runs
        ``population_size * (population_size - 1)`` simulations per iteration forever
        after.

        Count parameter sets, not jobs. One parameter set is several jobs under
        ``smoothing`` or ``parallelize_models``, and :meth:`_report_parallelism` applies
        that multiplier itself.
        """
        return None

    def _report_parallelism(self, client, jobs_in_flight, psets_in_flight=None):
        """Log how many jobs the fit runs at once against how many workers connected, and
        warn when the two differ by a large margin (#621).

        How many simulations a fit runs at once follows its settings, usually
        population_size, not how many processors were reserved. When many more workers
        connect than there are jobs to run, the extra workers sit idle for the whole run
        and nothing else would say so, so a user can reserve several machines and quietly
        use a fraction of them. The opposite, many more jobs than workers, means work
        queues up and the larger population buys no extra speed. Either way both numbers go
        in the log so a finished run can be looked at afterwards.

        The number reported is the one the fit sustains, from :meth:`expected_parallelism`,
        which is not always the size of the first batch of jobs the run loop submits
        (#655). When the two differ, the first batch is a one-time initialization round, so
        it is reported as such and the warnings are decided on the sustained number.
        ``psets_in_flight`` is how many parameter sets that first batch of jobs came from,
        which is how a parameter-set count is converted into a job count; None means one
        job per parameter set.

        Only cluster runs are reported. A local run's worker count is exactly what the user
        asked for through parallel_count, so there is nothing to compare it against. The
        worker count comes from dask; anything that goes wrong reading it is logged and
        never stops a fit.
        """
        # A local run drives a LocalCluster that the Client owns, so client.cluster is set.
        # A cluster run connects to a scheduler by file or address and has no such object.
        if getattr(client, 'cluster', None) is not None:
            return
        try:
            n_workers = len(client.scheduler_info().get('workers', {}))
        except Exception:
            logger.exception('Could not read the number of connected workers from dask, '
                             'so the parallelism report is skipped')
            return
        if n_workers <= 0:
            return

        n_jobs = self.expected_parallelism()
        if n_jobs is None or n_jobs <= 0:
            n_jobs = jobs_in_flight
        elif psets_in_flight:
            # expected_parallelism() counts parameter sets, but one parameter set is more
            # than one job when smoothing runs replicates of it or parallelize_models
            # splits it across models. The first batch shows how many jobs a parameter set
            # becomes, and that ratio holds for the rest of the fit.
            n_jobs = int(round(n_jobs * jobs_in_flight / psets_in_flight))

        logger.info('Parallelism: the fit runs %d job(s) at a time and %d worker(s) are '
                    'connected.' % (n_jobs, n_workers))

        notes = []
        # A first batch of a different size is a one-time initialization round. Say so, so
        # that a user watching the start of the fit is not surprised by it and does not
        # read it as how busy the fit will be.
        if jobs_in_flight != n_jobs:
            notes.append('This fit begins with a one-time round of %d job(s) before it '
                         'settles at %d.' % (jobs_in_flight, n_jobs))
        # A generational fit drains each generation to almost nothing before starting the
        # next, so some idle time is expected with one and should not be read as a fault.
        if self.waits_for_full_generation:
            notes.append('This fit runs one generation at a time and waits for all of it '
                         'to finish before starting the next, so some idle time toward '
                         'the end of each generation is expected.')
        note = (' ' + ' '.join(notes)) if notes else ''

        # Name the setting that sizes the concurrency, when one setting does. A fit where
        # none does (profile likelihood, whose concurrency follows how many parameters it
        # profiles) leaves the advice to talk about processors only.
        setting = self.parallelism_setting
        by_setting = ('the fitting settings, mainly %s,' % setting) if setting \
            else 'the fitting settings,'
        raise_setting = ('raising %s or ' % setting) if setting else ''
        lower_setting = ('lowering %s or ' % setting) if setting else ''

        # A factor of two in either direction is the "large margin" that draws a warning.
        if n_workers >= 2 * n_jobs:
            msg = ('The fit runs only %d job(s) at a time but %d worker(s) are connected, '
                   'so about %d worker(s) will sit idle. How many jobs run at once is set '
                   'by %s not by how many processors were reserved. '
                   'Consider %sreserving fewer processors.%s'
                   % (n_jobs, n_workers, n_workers - n_jobs, by_setting,
                      raise_setting, note))
            logger.warning(msg)
            print1('Warning: ' + msg)
        elif n_jobs >= 2 * n_workers:
            msg = ('The fit runs %d job(s) at a time but only %d worker(s) are connected, '
                   'so jobs will queue and the extra jobs buy no extra speed. How many '
                   'jobs run at once is set by %s not by how many processors were '
                   'reserved. Consider %sreserving more processors.%s'
                   % (n_jobs, n_workers, by_setting, lower_setting, note))
            logger.warning(msg)
            print1('Warning: ' + msg)
        elif note:
            # The counts are close, but there is still something worth putting on the
            # record for this run.
            logger.info(note.strip())

    def _finalize_run(self):
        """The end-of-fit path: stop reason, final parameter sets, best-fit artifacts,
        teardown.

        Reached the same way however the search ended -- the algorithm's own stop
        criterion, an exhausted job pool, or the wall-time budget -- because a budgeted run
        is a *finished* run whose search was cut short and writes exactly what a converged
        one writes (ADR-0093). Split out of :meth:`run` so a method that does **not** drive
        the propose/score loop still writes the same artifacts every other fit type does,
        rather than forking the tail: today that is ``job_type = ms``, whose unit of work is
        a trajectory segment rather than a PSet evaluation and which therefore overrides
        :meth:`run` (as ``hmc`` does, for the same reason). Everything here reads
        :attr:`trajectory` and :attr:`stop_reason` and nothing about how they were filled.
        """
        self._announce_stop_reason()
        # How every start of a multi-start fit did, next to why the fit stopped -- both are
        # statements about the run rather than about its best parameter set, and a fit whose
        # starts all failed still has this to report even though it has no best fit below.
        self._emit_multistart_summary()

        # Write the final parameter sets, then copy the best simulations into the results
        # folder. A run that stopped before any result came back (an expired budget, an
        # immediately exhausted pool) has no parameter sets at all, so there is nothing to
        # write, copy, re-simulate, or score -- say so rather than raising out of an
        # otherwise-finished run.
        if len(self.trajectory) == 0:
            logger.warning('No parameter set completed, so there is no best fit to report')
            print1('No simulation completed, so there is no best fit to report.')
        else:
            self.output_results('final')
            best_name = self.trajectory.best_fit_name()
            best_pset = self.trajectory.best_fit()
            self._copy_best_fit_sims(best_pset, best_name)
            self._rerun_best_fit_to_save_data(best_pset)
            self._emit_best_fit_bngl(best_pset, best_name)
            self._emit_information_criteria(self._compute_information_criteria(best_pset))
            self._emit_profiled_noise()
            self._emit_inference_data()
        self._finalize_backup_pickle()
        self._teardown_sim_dir()

        logger.info("Fitting complete")

    def _drain_job_pool(self, client, pool, pending, backup_every, debug):
        """Drain completed jobs until the run stops, resubmitting what the algorithm asks
        for. Mutates ``pending`` (the in-flight future -> (PSet, job_id) map) and returns
        the number of results consumed.

        :meth:`run`'s inner loop, extracted so its three exits -- the algorithm's own
        ``'STOP'``, an exhausted pool, and the wall-time budget (#529) -- are one unit
        that can be exercised without the surrounding setup. A budget exit records
        :attr:`stop_reason` and leaves the pending futures for ``run`` to cancel, exactly
        as a ``'STOP'`` exit does: the difference between a budgeted stop and a converged
        one is the reason, not the artifacts.
        """
        sim_count = 0
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
            except TimeoutError:
                # The budget ran out while every worker was still busy; the in-flight
                # simulations are abandoned (run() cancels them next).
                self.stop_reason = self._wall_time_stop_reason(sim_count)
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
            # The completed result is recorded first (it is already paid for), then the
            # budget decides whether anything new may be launched.
            if self._budget_spent():
                self.stop_reason = self._wall_time_stop_reason(sim_count)
                break
            # Submit the next round of jobs the algorithm asked for.
            new_futures = []
            for ps in decision:
                new_js = self.make_job(ps)
                for new_j in new_js:
                    new_f = client.submit(core.run_job, new_j, (debug or self.fail_count < 10),
                                          self.failed_logs_dir,
                                          models=self.models_future, calc=self.calc_future)
                    pending[new_f] = (ps, new_j.job_id)
                    new_futures.append(new_f)
            logger.debug('Submitting %d new Jobs' % len(new_futures))
            pool.update(new_futures)
        return sim_count

    def _budget_spent(self):
        """Whether this fit's total wall-clock budget (``wall_time_fit``) is gone.
        Always False when no budget was configured (the default)."""
        return self.budget is not None and self.budget.expired()

    def _wall_time_stop_reason(self, sim_count):
        """The one-line reason a wall-time budget stop is reported under (#529).

        Deliberately in universal terms -- elapsed time, simulations completed, best
        objective so far -- because there is no iteration counter shared by every fit
        type (a generation, a start, and a chain step are all "one iteration" to
        different algorithms).
        """
        best = self.trajectory.best_score() if len(self.trajectory) else None
        # A held-back reserve means this phase stopped short of the run's own deadline,
        # on purpose, to leave the refine something to run on (#564). Say so, or the
        # elapsed time will not add up against wall_time_fit for anyone reading it.
        allowance = ''
        if self.budget.reserve > 0:
            allowance = (', less %d s reserved for the refine (wall_time_refine_frac)'
                         % int(round(self.budget.reserve)))
        return ('Wall-time budget reached: %sstopped after %s (wall_time_fit = %d s%s) and %d completed '
                'simulation(s), with a best objective of %s.'
                % ('the refine ' if self.refine else '',
                   format_duration(self.budget.elapsed()), int(self.budget.limit), allowance, sim_count,
                   ('%.10g' % best) if best is not None else 'n/a (no completed simulation)'))

    def _announce_stop_reason(self):
        """Log, print, and record why a run stopped, when it stopped for a reason the
        user must not mistake for convergence (today: the wall-time budget, #529).

        The durable half is ``Results/stop_reason.txt``: a budgeted run writes the same
        artifacts a converged one writes, so the *presence* of this file -- and nothing
        in the scoreable outputs -- is what tells a downstream consumer the fit hit its
        deadline instead of its stop criterion. A no-op for an ordinary run.

        A refine's reason is **appended** rather than written over the fit's: both phases
        share one Results directory, and a run where the search hit the deadline AND the
        polish did has two facts to report, not one that replaces the other (#564). The
        per-phase, machine-readable form of the same thing is ``method_chain.json``.
        """
        if not self.stop_reason:
            return
        logger.info(self.stop_reason)
        print1(self.stop_reason)
        path = str(Path(self.res_dir) / 'stop_reason.txt')
        try:
            with open(path, 'a' if self.refine else 'w') as f:
                f.write(self.stop_reason + '\n')
        except Exception:
            logger.exception('Failed to write stop_reason.txt')

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
            to_save.save_all(f'{self.res_dir}/{to_save.name}_{best_name}')
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
                        shutil.copy(f'{self.sim_dir}/{best_name}/{m}_{best_name}_{suf}.{ext}',
                                    f'{self.res_dir}')
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
                for fname in glob(str(Path(self.sim_dir) / 'bestfit' / '*.gdat')) + glob(str(Path(self.sim_dir) / 'bestfit' / '*.scan')):
                    shutil.copy(fname, self.res_dir)
            # Restore save_files defaults (in case there is future bootstrapping or refinement)
            for m in self.model_list:
                if hasattr(m, 'save_files'):
                    m.save_files = False

    def _emit_best_fit_bngl(self, best_pset, best_name):
        """Write a stable-named, family-labelled best-fit BNGL per model (ADR-0048).

        New-era (``edition >= 2``) only, and a no-op when there is no best fit (e.g.
        every evaluation failed). For each BNGL model, render ``best_pset`` into a
        runnable ``Results/<model>_bestfit.bngl`` prefaced by a header comment that
        records the objective value and labels the point honestly per algorithm
        family: an optimizer's *best fit (minimum objective)*; a Bayesian sampler's
        *maximum-likelihood point* -- the recorded objective excludes the prior, so
        it is NOT the MAP (see :meth:`_best_fit_header` and the ADR). When
        ``embed_best_fit_data`` is set, each time-indexed observable's experimental
        data is embedded inline (ADR-0054) as a ``tfun([t...],[y...], time)`` reference
        function (:meth:`_build_exp_data_tfuns`); when ``smooth_plot_points`` is set,
        the data-derived time-course actions render on a uniform fine grid so the
        artifact plots a smooth curve (:meth:`_smooth_action_lines`).

        Reuses the ADR-0034 rendering path (``copy_with_param_set`` -> ``model_text``)
        and the same ``_stage_and_rewrite_tfun_files`` staging ``save()`` uses, so a
        legacy run is untouched (it never reaches here) and a model carrying its own
        tfun file refs stays runnable. Split out of run()'s tail so it can be unit
        tested without a dask client; per-model failures are logged, never fatal.
        """
        if not edition.is_modern(edition.resolve_edition(self.config.config.get('edition'))):
            return
        if best_pset is None:
            return
        try:
            best_obj = self.trajectory.best_score()
        except (ValueError, IndexError):
            # Empty trajectory -> no best fit to emit.
            return
        header = self._best_fit_header(best_obj, best_name)
        embed = bool(self.config.config.get('embed_best_fit_data'))
        for m in self.config.models:
            model = self.config.models[m]
            if not isinstance(model, BNGLModel):
                # Only BNGL models render to a .bngl artifact; SBML/analytical models
                # carry non-BNGL text and are out of scope here.
                continue
            try:
                self._write_one_best_fit_bngl(model, best_pset, header, embed)
            except Exception:
                logger.exception('Failed to write best-fit BNGL for model %s' % m)
                print1('Could not write the best-fit BNGL artifact for model %s; see log.' % m)

    def _compute_information_criteria(self, best_pset):
        """AIC / BIC / AICc for the best-fit parameter set, or ``None``.

        A no-op (``None``) unless there is a best fit AND the objective is a proper
        likelihood (``supports_pointwise_log_likelihood`` -- the ADR-0011 noise
        families). A bare least-squares / distance / pass-through objective carries
        no normalized density, so no information criterion is defined for it -- the
        same gate LOO/WAIC use (ADR-0056).

        Otherwise the best pset is re-simulated once, in-process, to get its
        simulation data back (``core.run_job`` with no calculator future does not
        null ``res.simdata`` the way worker-side scoring does), then scored through
        the same normalize -> postprocess -> pointwise-``log_density`` path the fit
        used -- giving the FULL normalized log-likelihood behind an ABSOLUTE AIC.
        One extra simulation at the end of a run that already ran thousands is
        negligible, and it is the only place the best pset's simdata is in hand
        (an optimizer discards it after scoring on the workers).

        Every failure is logged and swallowed (returns ``None``): the run has
        otherwise completed, and a diagnostics field must never abort it. Split out
        of run()'s tail so it can be unit-tested without a dask client.
        """
        if best_pset is None:
            return None
        if not getattr(self.objective, 'supports_pointwise_log_likelihood', False):
            return None
        try:
            # calc_future=None keeps simdata on the Result (no worker-side scoring);
            # delete_folder=True cleans up the one-off rerun (its gdat/scan are already
            # saved by _copy_best_fit_sims / _rerun_best_fit_to_save_data above).
            job = core.Job(self.model_list, best_pset, 'bestfit_infocrit',
                           self.sim_dir, self.config.config['wall_time_sim'], None,
                           self.config.config['normalization'], self.config.postprocessing,
                           True,
                           stochastic_seed_policy=self.config.config['stochastic_seed'])
            res = core.run_job(job)
            if getattr(res, 'failed', False) or res.simdata is None:
                logger.warning('Could not re-simulate the best fit to compute information criteria')
                return None
            # Mirror add_to_trajectory's scoring setup so the log-likelihood is scored
            # against exactly the data the fit's objective saw.
            res.normalize(self.config.config['normalization'])
            res.postprocess_data(self.config.postprocessing)
            # An analytically profiled noise scale (ADR-0108) is still an ESTIMATED quantity,
            # so it keeps counting in k -- only the search dropped it. Counting the searched
            # variables alone would shift every AIC/BIC in a profiled fit relative to the same
            # fit run without profiling, which is exactly the comparison k exists to support.
            profiled = getattr(self.config, 'profiled_noise_params', ()) or ()
            k = len(self.variables) + len(profiled)
            ic = likelihood_information_criteria(
                self.objective, res.simdata, self.exp_data, best_pset, k)
            # The scoring call above put every profiled scale at its MLE for the best fit, so
            # the objective now holds the values this fit estimated for the removed dimensions.
            self._profiled_noise = dict(getattr(self.objective, '_profiled_noise', None) or {})
            return ic
        except Exception:
            logger.exception('Failed to compute information criteria for the best fit')
            return None

    def _checkpoint_information_criteria(self):
        """Write ``Results/information_criteria_backup.txt`` for the best fit so far -- the
        information-criteria half of the run's checkpoint (#560).

        ``sorted_params_backup.txt`` has been checkpointed since forever, but the information
        criteria were written only on the terminal path, so the two halves of a scoreable
        result had entirely different lifetimes: a run was un-scoreable at every moment
        except its last, even though the best parameter vector had been on disk the whole
        time. That matters because ``log_likelihood`` here is the only place PyBNF reports
        the FULL normalized log-likelihood -- the minimized ``Obj`` column of the parameter
        table is the *reduced* objective -- so an absolute AIC/BIC, or any benchmark score
        built on one, cannot be computed from the parameter checkpoint alone. A killed,
        crashed, or simply not-yet-finished run now carries both halves.

        Cheap by construction, and free unless it has something new to say:

        * a no-op unless the objective is a proper likelihood (the gate in
          :meth:`_compute_information_criteria`), so nothing is spent on the ``sos`` /
          ``sod`` / ``norm_sos`` / ``kl`` / ``wasserstein`` / ``direct_pass`` fits for which
          no information criterion is defined;
        * a no-op while the best fit is unchanged -- the file on disk already describes it.
          That is exactly the long converged tail of a search, where the reported number has
          stopped moving and only stragglers are still running;
        * otherwise one extra simulation per checkpoint, i.e. one per ``backup_every *
          population_size * smoothing`` simulation returns. ``backup_information_criteria =
          0`` turns it off for a model where even that is too expensive.

        Failures are logged and swallowed by the two helpers this calls: a diagnostics file
        must never abort a run, and a *periodic* one must not start aborting one mid-flight
        either.
        """
        if not self.config.config['backup_information_criteria']:
            return
        if len(self.trajectory) == 0:
            return
        best_name = self.trajectory.best_fit_name()
        if best_name == self._ic_checkpoint_name:
            logger.debug('Best fit is unchanged since the last information-criteria checkpoint; '
                         'not re-simulating it')
            return
        # _compute_information_criteria captures the profiled noise scales at whatever point
        # it scores, and profiled_noise.txt reports the run's FINAL best fit, so a checkpoint
        # must not leave a mid-run value behind for the end-of-run tail to report if its own
        # scoring pass fails.
        saved_profiled_noise = self._profiled_noise
        try:
            ic = self._compute_information_criteria(self.trajectory.best_fit())
        finally:
            self._profiled_noise = saved_profiled_noise
        if ic is None:
            return
        # Mirrors sorted_params_backup.txt / sorted_params_refine_backup.txt, so both halves
        # of one phase's checkpoint carry the same name, and the final artifact keeps its
        # exact current meaning.
        name = 'refine_backup' if self.refine else 'backup'
        wrote = self._emit_information_criteria(
            ic, name=name,
            preamble=['# CHECKPOINT of a run still in progress: the best fit SO FAR, not the',
                      "#   run's result. That is information_criteria.txt, written at the end.",
                      '# Parameter set: %s -- its row in sorted_params_%s.txt.' % (best_name, name)])
        if wrote:
            # Only a written file licenses skipping the next recompute; a transient I/O
            # failure must not leave the checkpoint permanently stale on a run whose best
            # fit never changes again.
            self._ic_checkpoint_name = best_name

    def _start_pset_for_record(self, psets):
        """The PSet that is genuinely this algorithm's start, for :meth:`_emit_start_point`.

        Not simply ``psets[0]``. A start-point optimizer resolved its start in ``__init__``
        and keeps it (``start_pset`` for CMA-ES, ``start_psets`` for the concurrent
        multi-start family), and for CMA-ES in particular the start is **never evaluated** --
        it only seeds the distribution mean, so the first generation's members are draws
        around it and none of them is the start. The population algorithms and samplers have
        no such attribute: their start really is the first member of the initial population,
        which ``_seed_start_point_pset`` has already pinned.
        """
        pset = getattr(self, 'start_pset', None)
        if pset is not None:
            return pset
        psets_attr = getattr(self, 'start_psets', None)
        if psets_attr:
            return psets_attr[0]
        return psets[0] if psets else None

    def _emit_start_point(self, psets, resumed=False):
        """Write ``Results/start_point.txt`` -- where this fit actually began (#583, ADR-0117).

        Provenance, not configuration: it answers "what point did the search start from,
        and was that the point I asked for?" for a run that has already happened. Every
        failure #583 catalogues is silent precisely because no artifact ever recorded the
        resolved start, so a fit displaced from its intended start is indistinguishable from
        a correct one -- the 1.97x displacement that motivated the issue was caught only
        because a known point failed to score what it provably scores.

        One row per free parameter, in **declaration** order (which is the order every
        u-space vector uses; note ``PSet``'s own string helpers sort alphabetically, so the
        two must not be mixed). Each row carries the start value, the box it was checked
        against, and its source, so the file is readable without the .conf beside it.

        ``psets`` is the initial population/start list. The first is the one the start point
        governs; the count is reported so a multi-start fit says outright how many of its
        starts were scattered rather than pinned. Written to ``self.res_dir``, so a bootstrap
        replicate records its own. Log-and-swallow on failure, like every artifact here: a
        provenance file must never be the reason a fit dies.
        """
        if resumed:
            # A resumed run has no start to resolve: `psets` is the pending in-flight list
            # unpickled from the backup, i.e. arbitrary mid-run individuals. Writing them
            # here would overwrite the original run's record -- destroying the one artifact
            # that says where the fit began -- and label mid-run values as a start point.
            # The record the original run left is the true one; leave it alone.
            return
        try:
            declared = getattr(self.config, 'start_point', None) or {}
            first = self._start_pset_for_record(psets)
            # Whether this algorithm RESOLVES a start point (a start-point optimizer, which
            # keeps `start_pset`/`start_psets`) or DRAWS an initial population. For the
            # latter an undeclared coordinate is a random / Latin-hypercube draw, not the box
            # centre -- labelling it 'box_center' would be a false statement in the one file
            # that exists so a reader can tell a displaced start from a correct one.
            resolves_start = (getattr(self, 'start_pset', None) is not None
                              or bool(getattr(self, 'start_psets', None)))
            rows = []
            for v in self.variables:
                spk = getattr(self, 'START_POINT_KEY', None)
                if spk is not None and spk in self.config.config:
                    source = 'refine'
                elif v.name in declared:
                    source = getattr(self.config, 'start_point_spelling', {}).get(v.name, 'declared')
                elif not resolves_start:
                    source = 'sampled'
                elif v.has_bounded_support:
                    source = 'box_center'
                elif not v.has_prior:
                    source = 'point'
                else:
                    source = 'prior'
                try:
                    value = '%.17g' % first.get_param(v.name).value if first is not None else 'n/a'
                except Exception:
                    value = 'n/a'
                lo_u, hi_u = v.prior_support()
                lo = '%.10g' % v.from_sampling_space(lo_u) if np.isfinite(lo_u) else '-inf'
                hi = '%.10g' % v.from_sampling_space(hi_u) if np.isfinite(hi_u) else 'inf'
                rows.append('%s\t%s\t%s\t%s\t%s' % (v.name, value, source, lo, hi))

            lines = [
                '# The point this fit started from, recorded before any parameter set was scored.',
                '# One row per free parameter, in declaration order (NOT the alphabetical order',
                '#   the parameter tables use).',
                '# source: start_point -- a "start_point = <p> <v>" line',
                '#         initial_value -- the initial_value: field of a parameter: record',
                '#         box_center -- the 0.5 quantile of the bounded prior. For a truncated',
                '#           non-uniform prior this is the MEDIAN, which is not the location',
                '#           parameter under asymmetric truncation -- declare a start point to pin it',
                '#         point      -- the single var / logvar / lnvar value',
                '#         prior      -- an unbounded prior with no declared start (p1)',
                '#         sampled    -- drawn from the prior / bounds, as a population',
                '#           algorithm or sampler does for every undeclared coordinate',
                '#         refine     -- the previous phase best fit, injected by a method chain',
                '# lower/upper are the DECLARED box, or +-inf where the parameter declares none',
                '#   (a parameter with no box can also finish outside any range you have in mind).',
                # A multi-start fit pins its FIRST start and scatters the rest, so say how
                # many there are: a reader who assumes their declared point governed the
                # whole fit is making the mistake this file exists to prevent. Only
                # MultiStartOptimizer has a start COUNT; for everyone else the initial pset
                # list is a population or a chain set, which is a different thing entirely.
                'starts\t%d' % int(getattr(self, 'n_starts', 1) or 1),
                'starts_pinned\t%d' % (1 if declared else 0),
                '#',
                '# parameter\tstart\tsource\tlower\tupper',
            ] + rows
            # A refine shares the fit's Results/ directory (pybnf.py points the refiner at
            # alg.res_dir), so it takes the _refine filename suffix every other artifact of a
            # second phase uses rather than overwriting the fit's own record.
            name = 'start_point_refine.txt' if self.refine else 'start_point.txt'
            path = str(Path(self.res_dir) / name)
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            logger.info('Wrote start point %s' % path)
        except Exception:
            logger.exception('Failed to write start_point.txt')

    def _emit_information_criteria(self, ic, name='', preamble=()):
        """Write ``Results/information_criteria.txt`` (and a console line) for a fit.

        ``ic`` is the :class:`~pybnf.objective.InformationCriteria` from
        :meth:`_compute_information_criteria`, or ``None`` -- a non-likelihood
        objective, or no best fit -- in which case nothing is written. AIC/BIC/AICc
        rank this fit against competing models (lower is better); this is the
        first-class form of the AIC lesson 45 (model selection) computes by hand.

        :param name: File suffix. ``''`` (the default) writes the run's result,
            ``information_criteria.txt``; anything else writes
            ``information_criteria_<name>.txt`` -- today the periodic checkpoint
            (:meth:`_checkpoint_information_criteria`, #560). A checkpoint differs from
            the result only in its ``#`` comments and in staying off the console, so one
            parser reads either file.
        :param preamble: Extra ``#`` comment lines placed above the standard header.
        :return: True if a file was written.
        """
        if ic is None:
            return False
        aicc_str = ('%.10g' % ic.aicc) if ic.aicc is not None else 'n/a (n <= k+1)'
        lines = list(preamble) + [
            '# Information criteria for the best-fit parameter set (lower is better).',
            '# Valid for a likelihood objective only (normal / lognormal / lnnormal / laplace /',
            '#   neg_bin / student_t). Computed from the full normalized log-likelihood',
            "#   at the best fit (the sum of the noise model's per-point log_density,",
            '#   ADR-0056), so AIC is an absolute value comparable across models.',
            '#   AIC  = 2k - 2*lnL',
            '#   BIC  = k*ln(n) - 2*lnL',
            '#   AICc = AIC + 2k(k+1)/(n-k-1)   (undefined when n <= k+1)',
            'k\t%d' % ic.k,
            'n\t%d' % ic.n,
            'log_likelihood\t%.10g' % ic.log_likelihood,
            'AIC\t%.10g' % ic.aic,
            'BIC\t%.10g' % ic.bic,
            'AICc\t%s' % aicc_str,
        ]
        filename = 'information_criteria.txt' if name == '' else 'information_criteria_%s.txt' % name
        path = str(Path(self.res_dir) / filename)
        try:
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            logger.exception('Failed to write %s' % filename)
            return False
        logger.info('Wrote information criteria %s' % path)
        if name == '':
            # The checkpoint says the same thing about a run still in progress, on a
            # cadence; like the parameter-set checkpoint beside it, it stays in the log.
            print1('Information criteria (best fit): AIC=%.6g  BIC=%.6g  AICc=%s  '
                   '(k=%d, n=%d, lnL=%.6g)'
                   % (ic.aic, ic.bic, aicc_str, ic.k, ic.n, ic.log_likelihood))
        return True

    def multistart_records(self):
        """One :class:`~pybnf.algorithms.multistart_report.StartRecord` per start of this
        fit, in any order -- :meth:`_emit_multistart_summary` sorts them (#658).

        Empty here, which is right for the great majority of fit types: they run one
        search and have nothing to compare it against. The three families that run several
        starts override this, each reading the numbers off wherever it happens to keep
        them (see :mod:`pybnf.algorithms.multistart_report`).
        """
        return ()

    def _emit_multistart_summary(self):
        """Write ``Results/multistart_summary.txt`` and print a short version of it (#658).

        A fit that runs several searches from different starting points reports the best
        of them. On its own that number says nothing about whether the search can be
        trusted, because a run whose starts all agreed and a run whose starts all
        disagreed print the same thing. This is the table that separates them: one row per
        start, best objective value first.

        Nothing is written for a fit with fewer than two starts, which includes every fit
        type that does not run a multi-start at all and every refine (a refine polishes
        the one point it was handed). A refine that does run several starts writes to the
        ``_refine`` name, as the other artifacts of a second phase do, rather than
        overwriting the searching fit's own table.

        Every failure is logged and swallowed. The fit has finished by this point, and a
        report must never be the reason a finished run dies.
        """
        try:
            records = list(self.multistart_records() or ())
        except Exception:
            logger.exception('Failed to collect the per-start results for the multi-start summary')
            return
        if len(records) < 2:
            return
        name = 'multistart_summary_refine.txt' if self.refine else 'multistart_summary.txt'
        path = str(Path(self.res_dir) / name)
        lines = multistart_report.summary_lines(
            records, job_type=self.config.config.get('fit_type'))
        try:
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            logger.exception('Failed to write %s' % name)
            return
        logger.info('Wrote the multi-start summary %s' % path)
        for line in multistart_report.console_lines(records, path):
            print1(line)

    def _emit_profiled_noise(self):
        """Write ``Results/profiled_noise.txt`` for a fit that profiles a noise scale out of
        the search (``noise_profiling = 1``, ADR-0108, #562).

        A profiled scale is fitted, not proposed: it is not a coordinate of the best PSet, so
        it appears in no ``sorted_params_*.txt`` row. This is where its estimate is reported --
        the maximum-likelihood value over the points that share it, at the best fit -- so a
        profiled run states every quantity it estimated, exactly as an unprofiled one does.

        A no-op when nothing was profiled (the default) or when the end-of-run scoring that
        produces the values did not run. Every failure is logged and swallowed: the run has
        completed, and a report must never abort it."""
        if not self._profiled_noise:
            return
        lines = [
            '# Noise scales profiled out of the search analytically (noise_profiling = 1).',
            '# Each value is the maximum-likelihood scale over the scored points that share',
            '#   it, evaluated at the best fit -- the estimate for a parameter the search',
            '#   never proposed, so it appears in no sorted_params_*.txt row. These ARE',
            '#   estimated parameters: they are counted in k in information_criteria.txt.',
            '# parameter\tvalue',
        ]
        lines += ['%s\t%.10g' % (name, value) for name, value in sorted(self._profiled_noise.items())]
        path = str(Path(self.res_dir) / 'profiled_noise.txt')
        try:
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            logger.exception('Failed to write profiled_noise.txt')
            return
        logger.info('Wrote profiled noise scales %s' % path)
        print1('Profiled noise scale(s) at the best fit: %s'
               % ', '.join('%s=%.6g' % (n, v) for n, v in sorted(self._profiled_noise.items())))

    def _emit_inference_data(self):
        """Write Results/inference_data.nc when ``output_inference_data`` is set (ADR-0055).

        Opt-in, and a no-op on a non-Bayesian fit (only the samplers write
        ``samples.txt``, the bridge's source). Builds the InferenceData from the
        saved samples via :func:`pybnf.inference_data.from_pybnf`, passing the live
        ``self.variables`` so log parameters land in sampling space without a config
        reload. The ``arviz`` extra is optional and imported lazily by the bridge; a
        missing extra (or any build/write failure) is logged, never fatal -- the run
        has already completed and written every other artifact.
        """
        if not self.config.config.get('output_inference_data'):
            return
        from .samplers.base import BayesianAlgorithm
        if not isinstance(self, BayesianAlgorithm):
            return
        try:
            from ..inference_data import from_pybnf
            idata = from_pybnf(self.res_dir, variables=self.variables)
            out_path = str(Path(self.res_dir) / 'inference_data.nc')
            idata.to_netcdf(out_path)
            logger.info('Wrote ArviZ InferenceData %s' % out_path)
        except ImportError:
            logger.warning('output_inference_data is set but the optional arviz extra is not '
                           'installed; skipping inference_data.nc. Install with: pip install pybnf[arviz]')
            print1('Skipped inference_data.nc: the optional arviz extra is not installed '
                   '(pip install pybnf[arviz]).')
        except Exception:
            logger.exception('Failed to write inference_data.nc')
            print1('Could not write the ArviZ InferenceData artifact; see log.')

    def _best_fit_header(self, best_obj, best_name):
        """The comment block prefacing a best-fit BNGL, labelled by family (ADR-0048).

        Optimizers get *best fit*; Bayesian samplers get *maximum-likelihood point*
        with an explicit note that the recorded objective omits the prior (so it is
        not the MAP) and a pointer to ``samples.txt`` for the posterior mode. The
        discriminator is the ``BayesianAlgorithm`` family base every sampler
        subclasses, imported lazily here to avoid the base<-sampler import cycle.
        """
        from .samplers.base import BayesianAlgorithm
        lines = [
            '# Best-fit model emitted by PyBNF (ADR-0048).',
            '# job_type: %s' % self.config.config.get('fit_type', '?'),
            '# Parameter set: %s' % best_name,
            '# Objective (minimum recorded): %.10g' % best_obj,
        ]
        if isinstance(self, BayesianAlgorithm):
            lines += [
                '# Point: MAXIMUM-LIKELIHOOD (minimum recorded objective).',
                '# NOTE: this is NOT the MAP. The recorded objective is the negative',
                '#   log-likelihood only -- the prior is not folded in. For the',
                '#   posterior mode (MAP), take the row with the largest Ln_probability',
                '#   in Results/samples.txt.',
            ]
        else:
            lines.append('# Point: BEST FIT (minimum objective).')
        return '\n'.join(lines) + '\n'

    def _write_one_best_fit_bngl(self, model, best_pset, header, embed):
        """Render one BNGL model's best-fit artifact (+ optional embedded data) to Results/."""
        to_save = model.copy_with_param_set(best_pset)
        # Smooth-curve opt-in (ADR-0054): re-render the data-derived time-course actions
        # onto a uniform fine grid so the artifact plots as a smooth curve. Acts only on
        # this deep-copied artifact model -- the fit already scored on the data grid, so
        # the objective is untouched. A no-op (0) leaves the ragged grid byte-identical.
        smooth = self.config.config.get('smooth_plot_points', 0) or 0
        if smooth > 0:
            to_save.actions = self._smooth_action_lines(to_save.actions, smooth)
        text = to_save.model_text()
        # Stage any tfun file refs the *source* model already carries (as save() does).
        # The embedded data below is now inline (ADR-0054), so it needs no staging.
        text = _stage_and_rewrite_tfun_files(
            text, os.path.dirname(model.file_path), self.res_dir)
        if embed:
            text = self._inject_function_lines(text, self._build_exp_data_tfuns(model))
        out_path = str(Path(self.res_dir) / ('%s_bestfit.bngl' % to_save.name))
        with open(out_path, 'w') as f:
            f.write(header + text)
        logger.info('Wrote best-fit BNGL artifact %s' % out_path)

    # The bracketed sample_times list of a synthesized time-course simulate action.
    _SAMPLE_TIMES_RE = re.compile(r'sample_times=>\[([^\]]*)\]')

    @classmethod
    def _smooth_action_lines(cls, action_lines, n_points):
        """Re-render data-derived time-course actions onto a uniform fine grid (ADR-0054).

        A new-era time course is emitted with the data's ragged ``sample_times=>[t0,...,tN]``
        grid (``pset._timecourse_line``); for a smooth, plottable best-fit artifact each such
        ``simulate(...)`` is re-rendered as ``t_end=>{max(tᵢ)},n_steps=>{n_points}`` -- exactly
        the uniform form ``_timecourse_line`` emits when ``explicit_points is None``, so
        ``method``/``t_start``/``suffix``/condition ``setParameter``s are preserved and the
        artifact stays a faithful, denser re-simulation of the same experiment.

        Only ``sample_times`` lists are rewritten: ``parameter_scan`` actions carry
        ``par_scan_vals`` (a swept axis, not time) and the steady-state pre-equilibration
        phase carries no ``sample_times`` -- both are left as-is. Returns a new list; the
        input is not mutated. Scoring is unaffected (this only touches the end-of-run
        artifact copy).
        """
        def repl(m):
            pts = [float(x) for x in m.group(1).split(',') if x.strip()]
            return 't_end=>%s,n_steps=>%d' % (_format_bngl_number(max(pts)), n_points)
        return [cls._SAMPLE_TIMES_RE.sub(repl, line) for line in action_lines]

    def _build_exp_data_tfuns(self, model):
        """Embed this model's experimental data as inline ``tfun`` reference functions.

        For each time-indexed experiment in ``self.exp_data[model.name]`` and each of
        its observable columns (excluding the independent variable and ``_SD`` noise
        columns), return an inline function line embedding the experimental (time, value)
        pairs directly in the BNGL (ADR-0054, was a sidecar ``.tfun`` file under ADR-0048)::

            <fn>() = tfun([t0,t1,...],[y0,y1,...], time)

        so ``<model>_bestfit.bngl`` self-contains its comparison curves in one file (no
        sidecar directory). The pairs are sorted and de-duplicated to the strictly-
        increasing index ``tfun`` requires; default linear interpolation. ``tfun`` is a
        bngsim feature (BNG2.pl parses no ``tfun`` form), so the embedded overlay is
        consumed through a bngsim path -- see ADR-0054.

        Non-time-indexed experiments (parameter scan / dose-response, whose indvar is
        a swept parameter) are skipped with a log note: a ``tfun(..., time)`` would
        misrepresent them. Columns with fewer than two finite points are skipped
        (``tfun`` needs at least two). Returns ``[]`` when nothing is embeddable.
        """
        func_lines = []
        model_data = self.exp_data.get(model.name, {})
        for suffix in sorted(model_data):
            data = model_data[suffix]
            if data.indvar != 'time':
                logger.info('best-fit data embed: skipping non-time experiment %r '
                            '(independent variable %r)' % (suffix, data.indvar))
                continue
            times = data[data.indvar]
            for obs in sorted(data.cols, key=data.cols.get):
                if obs == data.indvar or obs.endswith('_SD'):
                    continue
                pairs = self._clean_tfun_pairs(times, data[obs])
                if len(pairs) < 2:
                    continue
                fn = self._sanitize_id('expt_%s_%s' % (suffix, obs))
                ts = ','.join(_format_bngl_number(t) for t, _ in pairs)
                vs = ','.join(_format_bngl_number(v) for _, v in pairs)
                func_lines.append('%s() = tfun([%s],[%s], time)' % (fn, ts, vs))
        return func_lines

    @staticmethod
    def _clean_tfun_pairs(times, values):
        """(time, value) pairs ready for a ``.tfun``: finite, sorted, strictly increasing.

        Drops NaN/Inf in either column, sorts by time, and collapses repeated times to
        the first value seen (``tfun`` requires a strictly increasing index)."""
        pairs = []
        seen = set()
        for t, v in sorted(zip(times, values), key=lambda p: p[0]):
            if not (np.isfinite(t) and np.isfinite(v)) or t in seen:
                continue
            seen.add(t)
            pairs.append((float(t), float(v)))
        return pairs

    @staticmethod
    def _sanitize_id(name):
        """A safe BNGL identifier / filename stem: non-word chars -> ``_``, never
        leading-digit (the ``expt_``/``<exp>__`` prefixes already guarantee this)."""
        s = re.sub(r'\W', '_', str(name))
        return s if re.match(r'[A-Za-z_]', s) else '_' + s

    @staticmethod
    def _inject_function_lines(text, func_lines):
        """Insert ``func_lines`` into ``text``'s ``begin functions`` block (ADR-0048).

        Merges into an existing ``functions`` block (before its ``end functions``);
        otherwise opens a fresh ``begin functions ... end functions`` block before the
        model's ``end model`` (or, lacking one, ``begin actions``). A no-op for an
        empty ``func_lines``."""
        if not func_lines:
            return text
        body = '\n'.join(func_lines)
        end_fn = re.search(r'(?im)^[ \t]*end[ \t]+functions\b', text)
        if end_fn and re.search(r'(?im)^[ \t]*begin[ \t]+functions\b', text):
            i = end_fn.start()
            return text[:i] + body + '\n' + text[i:]
        block = 'begin functions\n' + body + '\nend functions\n'
        anchor = (re.search(r'(?im)^[ \t]*end[ \t]+model\b', text)
                  or re.search(r'(?im)^[ \t]*begin[ \t]+actions\b', text))
        if anchor:
            i = anchor.start()
            return text[:i] + block + text[i:]
        return text + '\n' + block

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
                os.replace('{}/alg_backup.bp'.format(self.config.config['output_dir']),
                          '{}/alg_{}.bp'.format(self.config.config['output_dir'],
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


def latin_hypercube(nsamples, ndims, rng):
    """
    Latin hypercube sampling.

    Returns a nsamples by ndims array, with entries in the range [0,1]
    You'll have to rescale them to your actual param ranges.

    ``rng`` is the caller's np.random.Generator (the algorithm's root rng).
    """
    if ndims == 0:
        # Weird edge case - needed for other code counting on result having a number of rows
        return np.zeros((nsamples, 0))
    value_table = np.transpose(np.array([[i/nsamples + 1/nsamples * rng.random() for i in range(nsamples)]
                                         for dim in range(ndims)]))
    for dim in range(ndims):
        rng.shuffle(value_table[:, dim])
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
