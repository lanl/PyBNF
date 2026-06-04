"""Classes and methods for configuring the fitting run"""


from .data import Data, DuplicateColumnError
from . import objective  # noqa: F401 -- imported for its side effect: running the module fires the @register_objfunc decorators, populating OBJFUNC_REGISTRY before _load_obj_func dispatches.
from . import algorithms  # noqa: F401 -- imported for its side effect: running the leaves fires the @register_fit_type decorators, populating FIT_TYPE_REGISTRY (incl. each method's config schema) before _build_config dispatches. No cycle: nothing in algorithms/ imports config.
from .registry import OBJFUNC_REGISTRY, FIT_TYPE_REGISTRY
from . import config_schema

from pydantic import ValidationError

from .pset import BNGLModel, ModelError, SbmlModel, SbmlModelNoTimeout, FreeParameter, TimeCourse, ParamScan, \
    Mutation, MutationSet
from .bngsim_sbml_model import (
    BNGSIM_HAS_SBML,
    BNGSIM_SBML_ERROR,
    BngsimSbmlModelNoTimeout,
)
from .bngsim_antimony_model import (
    BNGSIM_HAS_ANTIMONY,
    BNGSIM_ANTIMONY_ERROR,
    BngsimAntimonyModelNoTimeout,
)
from .printing import verbosity, print1, PybnfError
from .constraint import ConstraintSet

import numpy as np
import os
import re
import logging
import subprocess
import roadrunner


logger = logging.getLogger(__name__)


def init_logging(file_prefix, debug=False, log_level_name='info'):

    file_name = '%s.log' % file_prefix

    # Parse log level
    if log_level_name == 'debug' or log_level_name == 'd':
        log_level = logging.DEBUG
    elif log_level_name == 'info' or log_level_name == 'i':
        log_level = logging.INFO
    elif log_level_name == 'warning' or log_level_name == 'w':
        log_level = logging.WARNING
    elif log_level_name == 'error' or log_level_name == 'e':
        log_level = logging.ERROR
    elif log_level_name == 'critical' or log_level_name == 'c':
        log_level = logging.CRITICAL
    elif log_level_name == 'none' or log_level_name == 'n':
        log_level = logging.CRITICAL
        file_name = os.devnull
    else:
        # Should not get here because ArgumentParser catches invalid input
        raise ValueError('Invalid --log_level setting "%s"' % log_level_name)


    fmt = logging.Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(log_level)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(10)
    root.addHandler(fh)

    # Route Python warnings (numpy RuntimeWarning, YAMLLoadWarning, etc.) through
    # the logging system so they go to the log file instead of spamming the terminal.
    logging.captureWarnings(True)

    dlog = logging.getLogger('distributed')
    dlog.handlers[:] = []  # remove any existing handlers
    dlog.setLevel(max(logging.WARNING, log_level))
    dlog.addHandler(fh)

    tlog = logging.getLogger('tornado')
    tlog.handlers[:] = []  # remove any existing handlers
    tlog.setLevel(logging.CRITICAL)
    tlog.addHandler(fh)

    talog = logging.getLogger('tornado.application')
    talog.handlers[:] = []
    talog.setLevel(logging.CRITICAL)
    talog.addHandler(fh)

    asynclog = logging.getLogger('asyncio')
    asynclog.setLevel(999)  # Higher than critical -> silent

    if debug:
        dfh = logging.FileHandler('%s_debug.log' % file_prefix, mode='a')
        dfh.setLevel(logging.DEBUG)
        dfh.setFormatter(fmt)

        root.addHandler(dfh)
        dlog.addHandler(dfh)
        tlog.addHandler(dfh)
        talog.addHandler(dfh)

def reinit_logging(file_prefix, debug=False, log_level_name='info'):
    """
    Shut down logging, then restart it.
    Used when some module (e.g. distributed v1.22.0) breaks the logging.
    """
    if logging.root:
        del logging.root.handlers[:]
    init_logging(file_prefix, debug, log_level_name)

class Configuration(object):
    def __init__(self, d=None):
        """
        Instantiates a Configuration object using a dictionary generated
        by the configuration file parser.  Default key, value pairs are used
        when possible for pairs not present in the provided dictionary.
        :param d: The result from parsing a configuration file
        :type d: dict
        """
        if d is None:
            d = dict()
            
        if 'models' not in d or len(d['models']) == 0:
            raise UnspecifiedConfigurationKeyError("'model' must be specified in the configuration file.")
        if 'fit_type' not in d:
            d['fit_type'] = 'de'
            print1('Warning: fit_type was not specified. Defaulting to de (Differential Evolution).')
        if d['fit_type'] == 'bmc':
            d['fit_type'] = 'mh'  # 'bmc' option was renamed to 'mh'. Preserve backwards compatibility.
        if 'objfunc' not in d:
            print1('Warning: objfunc was not specified. Defaulting to chi_sq.')
        if not self._req_user_params() <= d.keys() and d['fit_type'] != 'check':
            unspecified_keys = []
            for k in self._req_user_params():
                if k not in d.keys():
                    unspecified_keys.append(k)
            raise UnspecifiedConfigurationKeyError(
                "The following configuration keys must be specified:\n\t"+",".join(unspecified_keys))

        if d['fit_type'] == 'check':
            d = self.check_unused_keys_model_checking(d)
        elif verbosity >= 1:
            self.check_unused_keys(d)
        # The MCMC-family beta-ladder preprocessing now runs inside _build_config
        # as the method schema's postprocess() hook (ADR-0006 #3), dispatched
        # uniformly there; non-MCMC methods inherit a no-op.
        self.config = self._build_config(d)
        if 'step_size' in d:
            self.config['adaptive_step_size'] = False
        self._check_random_seed()

        self._data_map = dict()  # Internal structure to help get both regular and mutant data to the right place
        self.models = self._load_models()
        logger.debug('Loaded models')
        self._load_actions()
        logger.debug('Loaded actions')
        self._load_simulators()
        logger.debug('Loaded simulators')
        self._load_mutants()
        logger.debug('Loaded mutants')
        self.mapping = self._check_actions()  # dict of model prefix -> set of experimental data prefixes
        logger.debug('Loaded model:exp mapping')
        self.exp_data, self.constraints = self._load_exp_data()
        logger.debug('Loaded data')
        self.obj = self._load_obj_func()
        logger.debug('Loaded objective function')
        self.variables = self._load_variables()
        if self.config['fit_type'] != 'check':
            self._check_variable_correspondence()
        logger.debug('Loaded variables')
        self._postprocess_normalization()
        self._load_postprocessing()
        self.config['time_length'] = self._load_t_length()
        logger.debug('Completed configuration')

    @staticmethod
    def default_config():
        """Default configuration values.

        The defaults now live in the typed Pydantic schema (``config_schema``,
        ADR-0002); this is a thin compat shim over it so existing callers keep
        working. Returns a fresh plain dict each call.
        """
        return config_schema.default_config_dict()

    @staticmethod
    def _build_config(d):
        """Build the effective config dict from a raw parsed config ``d``.

        The flow is ``raw dict -> Pydantic (validate / coerce / default) ->
        effective dict`` (ADR-0002). The effective dict is the **full union** of
        every method's defaults for every fit_type (ADR-0006), assembled in four
        overlays:

        1. a default-union baseline (the global defaults unioned with every
           registered method schema's defaults), so a ``de`` fit still carries
           ``cognitive`` / ``particle_weight`` / ... ;
        2. the validated *global* keys the user set (``GlobalConfig``);
        3. the selected fit_type's *method* schema -- its validated keys plus its
           own defaults -- for methods migrated to a co-located schema (Stage b);
        4. the *extras* -- required user keys (``population_size`` /
           ``max_iterations``), keys of not-yet-migrated methods, and the
           structural model-path / free-parameter (tuple) / ``models`` /
           ``exp_data`` keys -- carried through unchanged (already parse-coerced).

        The raw dict is partitioned by key ownership: a string key owned by
        ``GlobalConfig`` is a global key, one owned by the selected method's
        schema is a method key, everything else (including the keys of *other*
        methods) is an extra. The three buckets are disjoint, so an extra never
        clobbers a validated default. A Pydantic ``ValidationError`` becomes a
        ``PybnfError``.

        The result is a plain dict so the existing ``config.config['x']`` reaches
        and writes stay untouched (dict-compat per ADR-0002); typed access
        migrates opportunistically in Stage (c).
        """
        entry = FIT_TYPE_REGISTRY.get(d.get('fit_type'))
        method_schema = entry.schema if entry is not None else None
        if method_schema is not None:
            # Method-owned preprocessing on the RAW dict, before defaults merge
            # (raw-presence semantics like ``'beta' not in d`` mean "user did not
            # set it"): the MCMC family's beta-ladder mutates d in place; every
            # other model inherits the no-op postprocess (ADR-0006 #3).
            method_schema.postprocess(d, d.get('fit_type'))
        global_keys = config_schema.SCHEMA_KEYS
        method_keys = method_schema.owned_keys() if method_schema is not None else frozenset()

        global_input = {k: v for k, v in d.items()
                        if isinstance(k, str) and k in global_keys}
        method_input = {k: v for k, v in d.items()
                        if isinstance(k, str) and k in method_keys}
        extras = {k: v for k, v in d.items()
                  if not (isinstance(k, str) and (k in global_keys or k in method_keys))}
        try:
            effective = config_schema.default_union()
            effective.update(config_schema.build_effective_global(global_input))
            if method_schema is not None:
                effective.update(
                    config_schema.build_effective_method(method_schema, method_input))
        except ValidationError as e:
            raise PybnfError('Invalid configuration', 'Invalid configuration:\n%s' % e)
        effective.update(extras)
        return effective

    def _check_random_seed(self):
        """Validate the optional random seed before NumPy consumes it."""
        seed = self.config['random_seed']
        if seed is None:
            return
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0 or seed >= 2**32:
            raise PybnfError('Invalid random_seed %s' % seed,
                             "Config key 'random_seed' must be an integer from 0 to %i." % (2**32 - 1))
        self.config['random_seed'] = int(seed)

    @staticmethod
    def check_unused_keys(conf_dict):
        """
        Gives warnings if the user has specified parameters that will be ignored by the chosen algorithm.
        :param conf_dict: The config dictionary
        :return:
        """
        alg_specific = {'ade': {'mutation_rate', 'mutation_factor', 'stop_tolerance', 'de_strategy'},
                        'de': {'islands', 'migrate_every', 'num_to_migrate'},
                        'pso': {'cognitive', 'social', 'particle_weight', 'particle_weight_final', 'adaptive_n_max',
                                'adaptive_n_stop', 'adaptive_abs_tol', 'adaptive_rel_tol', 'v_stop'},
                        'ss': {'init_size', 'local_min_limit', 'reserve_size'},
                        'mh': {'step_size', 'burn_in', 'sample_every', 'output_hist_every', 'hist_bins',
                                'credible_intervals', 'beta', 'beta_range', 'exchange_every', 'beta_max', 'cooling',
                                'crossover_number', 'zeta', 'lambda', 'gamma_prob', 'adaptive', 'time_length',
                                'archive_size', 'archive_thin_rate', 'snooker_prob',
                                'delta', 'outlier_method', 'rhat_threshold', 'diagnostics_every',
                                'precondition_adapt'},
                        'sim': {'simplex_step', 'simplex_log_step', 'simplex_reflection', 'simplex_expansion',
                                'simplex_contraction', 'simplex_shrink', 'simplex_max_iterations',
                                'simplex_stop_tol'}
                        }
        ignored_params = set()
        thisalg = conf_dict['fit_type']
        if thisalg in ('pt', 'sa', 'dream', 'p_dream', 'am'):
            thisalg = 'mh'
        for alg in alg_specific:
            if (thisalg != alg
               and not(alg == 'sim' and 'refine' in conf_dict and conf_dict['refine'] == 1)
               and not (thisalg == 'de' and alg == 'ade')):
                ignored_params = ignored_params.union(alg_specific[alg])
        for k in ignored_params.intersection(set(conf_dict.keys())):
            print1('Warning: Configuration key %s is not used in fit_type %s, so I am ignoring it'
                            % (k, conf_dict['fit_type']))
            logger.warning('Ignoring unused key %s for fitting algorithm %s' % (k, conf_dict['fit_type']))

    @staticmethod
    def check_unused_keys_model_checking(conf_dict):
        """
        Gives warnings if the user has specified parameters that will be ignored by model checking. Uses different
        logic than the rest of the algorithms because so few keys are used for model checking
        :param conf_dict: The config dictionary
        :return: A modified config dictionary, after removing any extraneous keys that would crash PyBNF
        """
        used = {'model', 'output_dir', 'simulation_dir', 'fit_type', 'objfunc', 'normalization', 'postprocessing',
                'verbosity', 'wall_time_sim', 'bng_command', 'sbml_integrator', 'sbml_backend', 'bngl_backend',
                'sbml_ssa_strict', 'stochastic_seed',
                'time_course',
                'param_scan', 'mutant',
                'models', 'exp_data', 'random_seed'}
        would_crash = {'refine', 'bootstrap'}

        for k in conf_dict:
            # Guard the model-path regex with isinstance(k, str): free-parameter
            # keys are tuples (e.g. ('uniform_var', 'p1')), and re.search on a tuple
            # raises TypeError -- so fit_type=check + any *_var key used to crash
            # (CFG-CHECK-1). A non-string key is never a model path, so it falls
            # through to the unused-key warning, which is correct for check.
            if k not in used and not (isinstance(k, str) and re.search(r'\.(bngl|xml|ant)', k)):
                print1('Warning: Configuration key '+str(k)+' is not used in fit_type "check", so I am ignoring it')
                # % (k,) not % k: a tuple free-param key would otherwise be spread
                # across the format string as multiple args (TypeError) -- the same
                # CFG-CHECK-1 tuple-key hazard, one line down.
                logger.warning('Ignoring unused key %s for fitting algorithm "check"' % (k,))
        for k in would_crash:
            if k in conf_dict:
                del conf_dict[k]
        return conf_dict

    @staticmethod
    def _req_user_params():
        """Configuration keys that the user must specify"""
        return {'models', 'population_size', 'max_iterations'}

    @staticmethod
    def _absolute(directory):
        """
        Convert relative paths to absolute paths
        """
        home_dir = os.getcwd()
        if os.name == 'nt':  # Windows
            if directory == '':
                return ''
            # Check for both unix-like and windows-like paths starting from root
            if directory[0] == '/' or re.match(r'[A-Z]:', directory):
                return directory
            else:
                return os.path.join(home_dir, directory)
        return '' if directory == '' else directory if directory[0] == '/' else home_dir + '/' + directory

    def _load_t_length(self):
        timeDict = {}
        for mf in self.config['models']:
            if re.search(r'\.bngl$', mf):
                time = BNGLModel(mf, suppress_free_param_error=self.config['fit_type']=='check').find_t_length()
                for i,v in time.items():
                    timeDict[i] = v
            elif re.search(r'\.(xml|ant)$', mf):
                for tc in self.config['time_course']:
                    suffix = tc['suffix']
                    try:
                        step = float(tc['step'])
                    except KeyError:
                        step = 1.
                    end_time = float(tc['time'])
                    timeDict[suffix] = int(np.round(end_time / step))
        return timeDict    
        
                

    def _load_models(self):
        """
        Loads models specified in configuration file in a dictionary keyed on
        Model.name
        """

        allowed_sbml_backends = ('roadrunner', 'bngsim')
        if self.config['sbml_backend'] not in allowed_sbml_backends:
            raise PybnfError('Invalid sbml_backend %s. Options are: %s.' %
                             (self.config['sbml_backend'], ', '.join(allowed_sbml_backends)))
        allowed_bngl_backends = ('auto', 'bionetgen', 'bngsim')
        bngl_backend = self.config.get('bngl_backend', 'auto')
        self.config['bngl_backend'] = bngl_backend
        if bngl_backend not in allowed_bngl_backends:
            raise PybnfError('Invalid bngl_backend %s. Options are: %s.' %
                             (bngl_backend, ', '.join(allowed_bngl_backends)))

        allowed_stochastic_seed = ('auto', 'auto_honorbngl', 'random', 'random_honorbngl')
        stochastic_seed = self.config.get('stochastic_seed', 'auto')
        self.config['stochastic_seed'] = stochastic_seed
        if stochastic_seed not in allowed_stochastic_seed:
            raise PybnfError('Invalid stochastic_seed %s. Options are: %s.' %
                             (stochastic_seed, ', '.join(allowed_stochastic_seed)))

        # If needed, choose the default timeout, which depends on what simulators the models use.
        if self.config['wall_time_sim'] is None:
            self.config['wall_time_sim'] = 0
            for mf in self.config['models']:
                if re.search(r'\.bngl$', mf):
                    self.config['wall_time_sim'] = 3600
                    break

        md = {}
        for mf in self.config['models']:
            # Initialize model type based on extension
            try:
                if re.search(r'\.bngl$', mf):
                    model = BNGLModel(mf, suppress_free_param_error=self.config['fit_type']=='check')
                    model.bng_command = self._absolute(self.config['bng_command'])
                    logger.debug('Set model %s command to %s' % (mf, model.bng_command))
                elif re.search(r'\.xml$', mf):
                    save_flag = (self.config['delete_old_files'] == 0)
                    if self.config['sbml_backend'] == 'bngsim':
                        if not BNGSIM_HAS_SBML:
                            raise PybnfError(
                                'sbml_backend = bngsim was requested, but %s.' % BNGSIM_SBML_ERROR
                            )
                        strict_ssa = bool(self.config.get('sbml_ssa_strict', 1))
                        # bngsim now enforces wall_time_sim in-process via
                        # SimulationTimeout, so the subprocess wrapper is no
                        # longer needed for either zero or positive timeouts.
                        model = BngsimSbmlModelNoTimeout(
                            mf,
                            self._absolute(mf),
                            save_files=save_flag,
                            integrator=self.config['sbml_integrator'],
                            strict_ssa=strict_ssa,
                        )
                    elif self.config['wall_time_sim'] == 0:
                        model = SbmlModelNoTimeout(
                            mf,
                            self._absolute(mf),
                            save_files=save_flag,
                            integrator=self.config['sbml_integrator'],
                        )
                    else:
                        model = SbmlModel(
                            mf,
                            self._absolute(mf),
                                save_files=save_flag,
                                integrator=self.config['sbml_integrator'],
                            )
                elif re.search(r'\.ant$', mf):
                    save_flag = (self.config['delete_old_files'] == 0)
                    if not BNGSIM_HAS_ANTIMONY:
                        raise PybnfError(
                            'Antimony model support was requested, but %s.' % BNGSIM_ANTIMONY_ERROR
                        )
                    strict_ssa = bool(self.config.get('sbml_ssa_strict', 1))
                    # bngsim now enforces wall_time_sim in-process via
                    # SimulationTimeout, so the subprocess wrapper is no
                    # longer needed for either zero or positive timeouts.
                    model = BngsimAntimonyModelNoTimeout(
                        mf,
                        self._absolute(mf),
                        save_files=save_flag,
                        integrator=self.config['sbml_integrator'],
                        strict_ssa=strict_ssa,
                    )
                elif re.search(r'\.target$', mf):
                    from .analytical_model import AnalyticalModel
                    model = AnalyticalModel(mf)
                else:
                    # Should not get here - should be caught in parsing
                    raise ValueError('Unrecognized model suffix in %s' % mf)
            except FileNotFoundError:
                raise PybnfError('Model file %s was not found.' % mf)
            except ModelError as e:
                raise PybnfError('In model file %s: %s' % (mf, e.message))
            if model.name in md:
                raise PybnfError('Multiple models with the name "%s". Please give all your models different names. '
                                 % model.name)
            md[model.name] = model
            self._data_map[model.name] = self.config[mf]  # List of exp files associated with this model

        for model in md.values():
            if isinstance(model, BNGLModel) and not model.has_observables:
                print1('Warning: Model %s has no observables defined. Fitting will not work without observables.'
                       % model.file_path)
                logger.warning('Model %s has no observables defined' % model.file_path)

        if self.config['smoothing'] > 1:
            # Check for misuse of 'smoothing' feature
            stochastic = np.any([m.stochastic for m in md.values()])
            if not stochastic:
                print1('Warning: You specified smoothing=%i, but it looks like none of your models use a stochastic '
                       'method. All of your smoothing replicates will come out identical.' % self.config['smoothing'])
            seeded_models = [m for m in md.values() if isinstance(m, BNGLModel) and m.seeded]
            if seeded_models and self.config['stochastic_seed'].endswith('_honorbngl'):
                raise PybnfError(
                    'You specified smoothing=%i with stochastic_seed=%s, and one of your simulation '
                    'commands contains an explicit "seed" argument. Under the "_honorbngl" policies, '
                    'that seed is honored verbatim, which would cause all of your smoothing replicates '
                    'to come out the same. Switch to stochastic_seed=auto (default) or stochastic_seed=random '
                    'so PyBNF overrides the BNGL seed per replicate.'
                    % (self.config['smoothing'], self.config['stochastic_seed'])
                )

        # Warn once per model when explicit BNGL seeds will be overridden by the
        # current policy. Saves the user from "this fit gives different numbers
        # than I expected and I forgot model.bngl had seed=>42 in it" debugging.
        if self.config['stochastic_seed'] in ('auto', 'random'):
            for m in md.values():
                if isinstance(m, BNGLModel) and m.seeded:
                    print1('Warning: model %s contains an explicit "seed" argument; it will be '
                           'overridden by stochastic_seed=%s. Use stochastic_seed=%s_honorbngl to '
                           'honor the BNGL seed.' %
                           (m.name, self.config['stochastic_seed'], self.config['stochastic_seed']))

        if self.config['parallelize_models'] > len(md):
            raise PybnfError('Job contains %i models, so "parallelize_models" should be at most %i' % (len(md), len(md)))

        return md

    def _load_mutants(self):

        if 'mutant' not in self.config:
            return

        for base, name, mutations, exps in self.config['mutant']:
            base = self._file_prefix(base, '(bngl|xml|ant)')
            if base not in self.models:
                raise PybnfError('Mutant %s declared corresponding to model %s, but that model was not found' %
                                 (name, base))
            mut_objects = [Mutation(var, op, float(val)) for var, op, val in mutations]
            mut_set = MutationSet(mut_objects, name)
            self.models[base].add_mutant(mut_set)
            # Check that the exp files will have simulation outputs
            for ex in exps:
                ename = self._file_prefix(ex, '(exp|con|prop)')
                base_suffix = re.match('.*(?=%s)' % name, ename)
                suffix_choices = [x[1] for x in self.models[base].suffixes]
                if len(suffix_choices) == 0:
                    raise PybnfError("Model %s has no action suffixes, so I can't have mutant model %s with "
                                     "data file %s based on that model" % (base, name, ex))
                if not base_suffix or base_suffix.group(0) not in suffix_choices:
                    raise PybnfError('Experimental file name %s in mutant model %s. This file name should consist of '
                                     'the model suffix it corresponds to, followed by the mutant name (e.g. %s%s.exp)'
                                     % (ex, name, suffix_choices[0], name))
            # Stages these exp files to get loaded along with regular model ones
            self._data_map[base] += exps

    def _load_simulators(self):

        model_types = set([type(m) for m in self.models.values()])

        # For each model type that exists in the run, check that the simulator is available, and pass the simulator
        # path to the appropriate Model subclass
        if BNGLModel in model_types:
            if self.config['bng_command'] == '':
                raise PybnfError('The location of the BioNetGen simulator (BNG2.pl) is not specified. Please set the '
                                 '"bng_command" configuration key to the location of the file BNG2.pl, or set the '
                                 'BNGPATH environmental variable to the folder containing BNG2.pl.\n'
                                 'If BioNetGen is not yet installed, please refer to installation instructions at '
                                 'https://pybnf.readthedocs.io/en/latest/installation.html#bionetgen')
            elif re.search(r'BNG2.pl', self.config['bng_command']) is None:
                raise PybnfError('The specified "bng_command" parameter in the configuration file must include the script '
                                 'name at the end of the path (e.g. /path/to/BNG2.pl)')
            else:  # check to make sure BNG2.pl is available
                try:
                    logger.info('Checking to make sure bng_command is appropriately set')
                    cmd = [self.config['bng_command'], '-v']
                    if os.name == 'nt':  # Windows
                        cmd = ['perl'] + cmd
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError:
                    #  Occurs on Windows if BNG2.pl is nonexistent, or on Mac/Linux if BNG2.pl exists but crashed
                    raise PybnfError('BioNetGen failed to execute. Please check that "bng_command" parameter in the '
                                     'configuration file points to the BNG2.pl script, or that the BNGPATH environmental '
                                     'variable is set to the folder containing BNG2.pl.\n'
                                     'For help, refer to '
                                     'https://pybnf.readthedocs.io/en/latest/installation.html#bionetgen')
                except FileNotFoundError:
                    #  Occurs on Mac/Linux if BNG2.pl is nonexistent.
                    raise PybnfError('The BioNetGen simulator (BNG2.pl) was not found at the specified location. Please set the '
                                     '"bng_command" configuration key to the location of the file BNG2.pl, or set the '
                                     'BNGPATH environmental variable to the folder containing BNG2.pl.\n'
                                     'If BioNetGen is not yet installed, please refer to installation instructions at '
                                     'https://pybnf.readthedocs.io/en/latest/installation.html#bionetgen')
        # Check that the integrator is valid
        if self.config['sbml_backend'] == 'bngsim':
            bngsim_integrators = ('cvode', 'gillespie')
            if self.config['sbml_integrator'] not in bngsim_integrators:
                raise PybnfError(
                    'Config option "sbml_backend = bngsim" supports sbml_integrator in %s; got %s.' %
                    (', '.join(bngsim_integrators), self.config['sbml_integrator'])
                )
        else:
            integrators = ('cvode', 'euler', 'rk4', 'gillespie')
            if self.config['sbml_integrator'] not in integrators:
                raise PybnfError('Invalid sbml_integrator %s. Options are: %s.' % (self.config['sbml_integrator'],
                                                                                   ', '.join(integrators)))
            if self.config['sbml_integrator'] == 'euler':
                if roadrunner.__version__ < '1.5.1':
                    raise PybnfError('Config option "sbml_integrator = euler" requires Roadrunner version 1.5.1 or higher. You '
                                     'have version %s' % roadrunner.__version__)
                print1('Warning: "sbml_integrator = euler" can be numerically unstable. Confirm that your model is '
                       'producing reasonable output.')
            if self.config.get('sbml_ssa_strict', 1) != 1:
                raise PybnfError(
                    'Config option "sbml_ssa_strict" is only supported when sbml_backend = bngsim. '
                    'Current sbml_backend is "%s".' % self.config['sbml_backend']
                )

    def _load_actions(self):

        for (key, ActionType) in (('time_course', TimeCourse), ('param_scan', ParamScan)):
            # Iterate through all time courses and param scans included in the config dict, create the corresponding
            # Action objects, and add them to the appropriate model(s).
            for action_dict in self.config[key]:
                if 'subdivisions' in action_dict and self.config['sbml_integrator'] != 'euler':
                    print1('Warning: Ignoring "subdivisions" setting because that is only used with sbml_integrator = '
                           'euler')
                if 'model' in action_dict:
                    action = ActionType(action_dict)
                    try:
                        # Model lookup - should work if model name included the extension or not.
                        model_key = self._file_prefix(action_dict['model'], '(bngl|xml|ant)')
                        self.models[model_key].add_action(action)
                    except KeyError:
                        raise PybnfError('%s declared for model %s, but that model was not found.' %
                                         (key, action_dict['model']))
                else:
                    # Apply to all models (hopefully just 1)
                    if len(self.models) > 1:
                        print1('Warning: Applying the same %s action to all models in this fitting run.' % key)
                    for m in self.models:
                        self.models[m].add_action(ActionType(action_dict))

    @staticmethod
    def _file_prefix(ef, ext="exp"):
        return re.sub(r"\."+ext, "", re.split('/', ef)[-1])

    def _load_exp_data(self):
        """
        Loads experimental data files in a nested dictionary keyed on model name, then data prefix
        Also loads constraint files (which at this point are stored in the same structures as the exp files) and stores
        them in a set.
        """
        ed = {}
        csets = set()
        for m in self._data_map:
            ed[m] = {}
            for ef in self._data_map[m]:
                if re.search("exp$", ef):
                    try:
                        d = Data(file_name=ef)
                    except FileNotFoundError:
                        raise PybnfError('Experimental data file %s was not found.' % ef)
                    except DuplicateColumnError as err:
                        raise PybnfError('Parsing data file %s. %s' % (ef, err.args[0]))
                    ed[m][self._file_prefix(ef)] = d
                else:
                    cs = ConstraintSet(self._file_prefix(m, '(bngl|xml|ant)'), self._file_prefix(ef, '(con|prop)'))
                    try:
                        cs.load_constraint_file(ef, scale=self.config['constraint_scale'])
                    except FileNotFoundError:
                        raise PybnfError('Constraint file %s was not found' % ef)
                    csets.add(cs)
        return ed, csets

    def _check_actions(self):
        mapping = dict()
        for model in self.models.values():
            suffs = set(model.get_suffixes())
            efs_per_m = {self._file_prefix(ef) for ef in self.config[model.file_path] if re.search(r"\.exp$", ef)}
            if not efs_per_m <= suffs:
                for ef in efs_per_m:
                    if ef not in suffs:
                        raise UnmatchedExperimentalDataError("Action not specified for '%s.exp'" % ef,
                              "You specified that model %s corresponds to data file %s.exp, but I can't find the "
                              "corresponding action in the model file or config file. One of the actions in %s "
                              "needs to include the argument 'suffix=>\"%s\" ', or your config file needs to include "
                              "an action with the suffix %s." % (model.name, ef, model.file_path, ef, ef))
            logger.debug('Model %s was mapped to %s' % (model.name, efs_per_m))
            mapping[model.name] = efs_per_m
        return mapping

    def _load_obj_func(self):
        objfunc = self.config['objfunc']
        # Cross-config requirement check stays in config (not in the registry,
        # which holds only the construction recipe): neg_bin cannot be built
        # without its r parameter, so guard before pulling config['neg_bin_r'].
        if objfunc == 'neg_bin' and 'neg_bin_r' not in self.config:
            raise UnknownObjectiveFunctionError("Objective function neg_bin cannot be defined without "
                                                "configuration neg_bin_r defined")
        entry = OBJFUNC_REGISTRY.get(objfunc)
        if entry is None:
            raise UnknownObjectiveFunctionError("Objective function %s not defined" % objfunc,
                  "Objective function %s is not defined. Valid objective function choices are: "
                  "chi_sq, lognormal, sos, sod, norm_sos, ave_norm_sos, neg_bin, kl, direct_pass" % objfunc)
        # Uniform construction (ADR-0011): every objective builds itself from the
        # config via its from_config classmethod -- no per-objfunc recipe.
        return entry.cls.from_config(self.config)

    def _load_variables(self):
        """
        Loads the variable names from the config dict into FreeParameter instances.
        :return: a list of FreeParameter instances
        """
        #Compile a list of the varible names to determine if the reqired var is present
        if self.config['objfunc'] == 'neg_bin_dynamic':
            r_check = []
            for k in self.config.keys():
                r_check.append(k[1])
            if np.any('r__FREE' in r_check):
                pass
            else:
                raise PybnfError('Using the neg_bin_dynamic objective function requires the r__FREE parameter in the .conf file and the model file')
        if self.config['objfunc'] == 'chi_sq_dynamic':
            sigma_check = []
            for k in self.config.keys():
                sigma_check.append(k[1])
            if np.any('sigma__FREE' in sigma_check):
                pass
            else:
                raise PybnfError('Using the chi_sq_dynamic objective function requires the sigma__FREE parameter in the .conf file and the model file')        
        variables = []
        for k in self.config.keys():
            if isinstance(k, tuple):
                if re.search('var$', k[0]):
                    if self.config['fit_type'] == 'sim' and k[0] not in ('var', 'logvar'):
                        raise PybnfError('Invalid Simplex variable type %s' % k[0],
                               "You've specified the Simplex algorithm (fit_type = sim), "
                               "but defined variable %s with the %s keyword.\n"
                               "For Simplex, you must instead define a single initial value for each variable\n"
                               "using the var or logvar keyword (e.g. var=%s 42 )" % (k[1], k[0], k[1]))

                    if self.config['fit_type'] != 'sim' and k[0] in ('var', 'logvar'):
                        raise PybnfError('Tried to use Simplex variable type %s in another algorithm.' % k[0],
                               "You've specified variable %s with keyword %s, but that keyword "
                               "is only to be used with the Simplex algorithm (fit_type = sim)\n"
                               "Valid keywords for other algorithms are: uniform_var, normal_var, \n"
                               "lognormal_var, loguniform_var." % (k[1], k[0]))

                    if k[0] in ('var', 'logvar'):
                        # 2nd number (step size) may be absent, must fill in appropriately
                        if len(self.config[k]) >= 2:
                            stepsize = self.config[k][1] # easy, it was right there
                        else:
                            stepsize = None  # Will sort out within SimplexAlgorithm
                        free_param = FreeParameter(k[1], k[0], self.config[k][0], stepsize)
                    else:
                        if len(self.config[k]) == 3:
                            free_param = FreeParameter(k[1], k[0], self.config[k][0], self.config[k][1],
                                                           bounded=self.config[k][2])
                        else:
                            free_param = FreeParameter(k[1], k[0], self.config[k][0], self.config[k][1])

                    logger.debug('Adding parameter %s with bounds [%s, %s]' %
                                 (free_param.name, free_param.lower_bound, free_param.upper_bound))
                    variables.append(free_param)
        logger.info('Loaded variables')
        return variables

    def _check_variable_correspondence(self):
        """Verify the config's free parameters and the models' parameters line up.

        PyBNF's load-time guard against a mistyped or orphaned free parameter. It
        runs once at config load (every fit_type except 'check') and raises rather
        than letting a fit proceed with a silently-wrong parameterization. Two
        directions:

        * config -> model: every .conf variable must appear in *at least one*
          model; a variable in no model is almost always a typo. This is what makes
          the per-model appliers' silent skip safe -- BngsimModel.execute and
          SbmlModelNoTimeout._modify_params deliberately skip a parameter not in
          *that* model (it may belong to another model in a multi-model fit), and
          this check catches a genuine typo here, before any simulation runs.
        * model -> config: every ``__FREE`` declared in a model file must be in the
          .conf, so you can't forget to fit one.

        Parameter names are unioned across all models, so multi-model and
        mixed-type fits work (a variable valid in any one model passes). The models
        spell parameters differently -- BNGL uses ``k__FREE``, SBML/bngsim use the
        bare ``kcat`` -- and the union accommodates both in one PSet.

        Param-agnostic models are skipped: a model exposing no enumerable parameter
        set takes its parameters from the .conf, so nothing can be proven a typo
        against it. AnalyticalModel is the current example (empty ``param_names`` by
        design); the ``hasattr`` guard also covers any future model type that never
        sets ``param_names``. This is the single config-level correspondence guard;
        do not add a duplicate elsewhere, and keep its regression tests
        (test_config_class) in sync.
        """
        from .analytical_model import AnalyticalModel

        # Skip if any model is param-agnostic (no enumerable parameter set): its
        # parameters come from the .conf, so nothing here is provably a typo. The
        # hasattr clause future-proofs against a model type that never sets
        # param_names (which would otherwise AttributeError in the union below).
        for m in self.models.values():
            if isinstance(m, AnalyticalModel) or not hasattr(m, 'param_names'):
                return

        model_vars = set()
        for m in self.models.values():
            model_vars.update(getattr(m, 'param_names', set()))

        variables_names = {v.name for v in self.variables}
        extra_in_conf = variables_names.difference(model_vars)
        extra_in_model = set(model_vars).difference(variables_names)
        # Only __FREE-suffixed names are "must-fit" model parameters; ignore other
        # model params (e.g. SBML species/globals) that legitimately aren't fit.
        # (Was `p[-8:] == '__FREE'`, comparing an 8-char slice to a 6-char string --
        # always False, so this whole direction silently never fired.)
        extra_in_model = {p for p in extra_in_model if p.endswith('__FREE')}

        if len(extra_in_conf) > 0:
            raise PybnfError('The following variables are declared in the .conf file, but were not found in any model '
                             'file: %s' % extra_in_conf)
        if len(extra_in_model) > 0:
            raise PybnfError('The following free parameters are in your model files, but are not declared in your '
                             '.conf file: %s' % extra_in_model)

    def _postprocess_normalization(self):
        """
        Postprocessing on the 'normalization' key
        :return:
        """
        seedoc = "\nSee the documentation for the syntax options for the 'normalization' key"
        valid = ('init', 'peak', 'zero', 'unit')
        if type(self.config['normalization']) == dict:
            # Iterate through the keys, which should be .exp file names. Check that these are actual exp files that
            # are used in the fitting, then add to the dictionary just the suffix, for easier lookup later
            newdict = dict()
            for ef in self.config['normalization']:
                if ef not in self.config['exp_data']:
                    raise PybnfError("Invalid exp file %s under the normalization key" % ef,
                                     "The exp file %s given under the 'normalization' keyword is not associated with "
                                     "any model." % ef + seedoc)
                val = self.config['normalization'][ef]

                # Figure out how to get to the right data object (it's in a dict keyed on model name, then suffix)
                m = None
                for modelpath in self.config['models']:
                    if ef in self.config[modelpath]:
                        m = self._file_prefix(modelpath, '(bngl|xml|ant)')
                        break
                suff = self._file_prefix(ef)

                def checkval(v):
                    if v not in valid:
                        raise PybnfError("Invalid normalization type '%s'" % self.config['normalization'][ef],
                                         "Invalid normalization type '%s'. Options are: init, peak, zero, unit" %
                                         self.config['normalization'][ef] + seedoc)
                if type(val) == str:
                    # This exp file has a single normalization type for all columns.
                    # Convert to column-specific form using only the columns present in the .exp file,
                    # so that simulation columns used only by .prop constraints are not normalized.
                    checkval(val)
                    exp_cols = [c for c in self.exp_data[m][suff].cols
                                if self.exp_data[m][suff].cols[c] != 0 and not c.endswith('_SD')]
                    if not exp_cols:
                        continue
                    val = [(val, exp_cols)]
                else:
                    # This exp file has a list of one or more pairs specifying (normalization_type, [columns])
                    for (i, (ntype, cols)) in enumerate(val):
                        checkval(ntype)
                        new_cols = []
                        if type(cols[0]) == int:
                            # Need to convert to string labels, because the indices into the sim data will be different
                            to_convert = cols
                            for label in self.exp_data[m][suff].cols:
                                ci = self.exp_data[m][suff].cols[label]
                                if ci in to_convert:
                                    to_convert.remove(ci)
                                    new_cols.append(label)
                            if len(to_convert) > 0:
                                raise PybnfError("Invalid normalization column %s for file %s" % (to_convert[0], ef),
                                                 "Specified normalization for column %i in file %s, but that file "
                                                 "contains only %i columns." % (
                                                 to_convert[0], ef, self.exp_data[m][suff].data.shape[1]) + seedoc)
                        else:
                            new_cols = cols
                        # Iterate over a copy: the _SD branch below removes from
                        # new_cols, and aliasing the iterator to the same list
                        # would skip the element after each removal (so a second
                        # consecutive _SD column was silently kept).
                        new_cols_iter = list(new_cols)
                        for c in new_cols_iter:
                            if c not in self.exp_data[m][suff].cols:
                                raise PybnfError("Invalid normalization column %s for file %s" % (c, ef),
                                                 "Specified normalization for column %s in file %s, but that file does "
                                                 "not contain that column name." % (c, ef) + seedoc)
                            if c[-3:] == '_SD':
                                logger.info('Removing %s from the normalization list' % c)
                                print1("Warning: You specified a normalization for %s, but I can't normalize a "
                                       "standard deviation separately, because it's not an output of the simulation. "
                                       "I'm ignoring your %s setting and assuming it's on the same scale as its data "
                                       "column." % (c, c))
                                new_cols.remove(c)
                        # Update with the postprocessed normalization info
                        val[i] = (ntype, new_cols)

                newdict[suff] = val
            self.config['normalization'].update(newdict)
        elif type(self.config['normalization']) == str:
            if self.config['normalization'] not in valid:
                raise PybnfError("Invalid normalization type '%s'" % self.config['normalization'],
                                 "Invalid normalization type '%s'. Options are: init, peak, zero, unit" %
                                 self.config['normalization'] + seedoc)
            # Convert global normalization to column-specific form for each exp file,
            # so that simulation columns used only by .prop constraints are not normalized.
            ntype = self.config['normalization']
            newdict = dict()
            for m in self.exp_data:
                for suff in self.exp_data[m]:
                    exp_cols = [c for c in self.exp_data[m][suff].cols
                                if self.exp_data[m][suff].cols[c] != 0 and not c.endswith('_SD')]
                    if exp_cols:
                        newdict[suff] = [(ntype, exp_cols)]
            self.config['normalization'] = newdict

    def _load_postprocessing(self):
        """
        Loads config info for user-specified Python scripts for postprocessing data
        :return:
        """
        self.postprocessing = dict()
        if 'postprocess' not in self.config:
            return

        for spec in self.config['postprocess']:
            script = self._absolute(spec[0])
            suffixes = spec[1:]

            # Check for simple errors in the script here, before we start running anything.
            try:
                # This incantation loads the module as postproc
                import importlib.util
                logger.info('Prepare to load the script %s' % script)
                spec = importlib.util.spec_from_file_location("postprocessor", script)
                if not spec:
                    raise PybnfError('Could not load the postprocessing script %s. Make sure this is a Python '
                                     'file (.py)' % script)
                postproc = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(postproc)
                # Now postproc is the user-defined Python module
            except OSError:
                raise PybnfError('Could not load the postprocessing script %s' % script)
            try:
                func = postproc.postprocess
            except NameError:
                raise PybnfError('The postprocessing script %s should contain a definition of the function '
                                 'postprocess(data). This function was not found.' % script)

            for suff in suffixes:

                # Need to backsolve the model name based on the suffix.
                model_choices = []
                for modelname in self.models:
                    if suff in self.models[modelname].get_suffixes():
                        model_choices.append(modelname)
                if len(model_choices) == 0:
                    raise PybnfError('Suffix %s was specified for a postprocessing script, but that suffix was not '
                                     'found in any model' % suff)
                if len(model_choices) > 1:
                    raise PybnfError('Suffix %s was specified for a postprocessing script, but was found in multiple '
                                     'models. Please rename suffixes to avoid this ambiguity.' % suff)
                self.postprocessing[(model_choices[0], suff)] = script


class UnknownObjectiveFunctionError(PybnfError):
    pass


class UnspecifiedConfigurationKeyError(PybnfError):
    pass


class UnmatchedExperimentalDataError(PybnfError):
    pass
