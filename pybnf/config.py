"""pybnf.config: classes and methods for configuring the fit"""


from .data import Data
from .objective import ChiSquareObjective, SumOfSquaresObjective, NormSumOfSquaresObjective, \
    AveNormSumOfSquaresObjective

from .pset import BNGLModel, ModelError, SbmlModel, FreeParameter, TimeCourse, ParamScan
from .printing import verbosity, print1, PybnfError
from .constraint import ConstraintSet

import numpy as np
import os
import re
import logging
import subprocess


logger = logging.getLogger(__name__)


def init_logging(file_prefix, debug=False):
    fmt = logging.Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')

    fh = logging.FileHandler('%s.log' % file_prefix, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(10)
    root.addHandler(fh)

    dlog = logging.getLogger('distributed')
    dlog.handlers[:] = []  # remove any existing handlers
    dlog.setLevel(logging.WARNING)
    dlog.addHandler(fh)

    tlog = logging.getLogger('tornado')
    tlog.handlers[:] = []  # remove any existing handlers
    tlog.setLevel(logging.CRITICAL)
    tlog.addHandler(fh)

    talog = logging.getLogger('tornado.application')
    talog.handlers[:] = []
    talog.setLevel(logging.ERROR)
    talog.addHandler(fh)

    if debug:
        dfh = logging.FileHandler('%s_debug.log' % file_prefix, mode='a')
        dfh.setLevel(logging.DEBUG)
        dfh.setFormatter(fmt)

        root.addHandler(dfh)
        dlog.addHandler(dfh)
        tlog.addHandler(dfh)
        talog.addHandler(dfh)


class Configuration(object):
    def __init__(self, d=dict()):
        """
        Instantiates a Configuration object using a dictionary generated
        by the configuration file parser.  Default key, value pairs are used
        when possible for pairs not present in the provided dictionary.

        :param d: The result from parsing a configuration file
        :type d: dict
        """
        if 'models' not in d or len(d['models']) == 0:
            raise UnspecifiedConfigurationKeyError("'model' must be specified in the configuration file.")

        if not self._req_user_params() <= d.keys():
            unspecified_keys = []
            for k in self._req_user_params():
                if k not in d.keys():
                    unspecified_keys.append(k)
            raise UnspecifiedConfigurationKeyError(
                "The following configuration keys must be specified:\n\t"",".join(unspecified_keys))

        if 'fit_type' not in d:
            d['fit_type'] = 'de'
            print1('Warning: fit_type was not specified. Defaulting to de (Differential Evolution).')

        if verbosity >= 1:
            self.check_unused_keys(d)
        if d['fit_type'] in ('bmc', 'pt', 'sa'):
            self.postprocess_mcmc_keys(d)
        self.config = self.default_config()
        for k, v in d.items():
            self.config[k] = v

        self.models = self._load_models()
        self._load_simulators()
        self._load_actions()
        self.mapping = self._check_actions()  # dict of model prefix -> set of experimental data prefixes
        self.exp_data, self.constraints = self._load_exp_data()
        self.obj = self._load_obj_func()
        self.variables = self._load_variables()
        self._check_variable_correspondence()
        self._postprocess_normalization()

    @staticmethod
    def default_config():
        """Default configuration values"""
        try:
            bng_command = os.environ['BNGPATH'] + '/BNG2.pl'
        except KeyError:
            bng_command = ''

        default = {
            'objfunc': 'chi_sq', 'output_dir': 'bnf_out', 'delete_old_files': 0, 'num_to_output': 1000000,
            'output_every': 20, 'initialization': 'lh', 'refine': 0, 'bng_command': bng_command, 'smoothing': 1,
            'backup_every': 1, 'time_course': (), 'param_scan': (),

            'mutation_rate': 0.5, 'mutation_factor': 0.5, 'islands': 1, 'migrate_every': 20, 'num_to_migrate': 3,
            'stop_tolerance': 0.002,

            'particle_weight': 1.0, 'adaptive_n_max': 30, 'adaptive_n_stop': np.inf, 'adaptive_abs_tol': 0.0,
            'adaptive_rel_tol': 0.0, 'cognitive': 1.5, 'social': 1.5,

            'local_min_limit': 5,

            'step_size': 0.2, 'burn_in': 10000, 'sample_every': 100, 'output_hist_every': 100, 'hist_bins': 10,
            'credible_intervals': [68., 95.], 'beta': [1.0], 'exchange_every': 20, 'beta_max': np.inf, 'cooling': 0.01,

            'simplex_step': 1.0, 'simplex_reflection': 1.0, 'simplex_expansion':1.0, 'simplex_contraction': 0.5,
            'simplex_shrink': 0.5,

            'wall_time_gen': 3600,
            'wall_time_sim': 3600,
            'normalization': None,

            'cluster_type': None,
            'scheduler_node': None,
            'worker_nodes': None
        }
        return default

    @staticmethod
    def check_unused_keys(conf_dict):
        """
        Gives warnings if the user has specified parameters that will be ignored by the chosen algorithm.
        :param conf_dict: The config dictionary
        :return:
        """
        alg_specific = {'de': {'mutation_rate', 'mutation_factor', 'stop_tolerance', 'islands', 'migrate_every',
                               'num_to_migrate'},
                        'pso': {'cognitive', 'social', 'particle_weight', 'particle_weight_final', 'adaptive_n_max',
                                'adaptive_n_stop', 'adaptive_abs_tol', 'adaptive_rel_tol'},
                        'ss': {'init_size', 'local_min_limit', 'reserve_size'},
                        'bmc': {'step_size', 'burn_in', 'sample_every', 'output_hist_every', 'hist_bins',
                                'credible_intervals', 'beta', 'beta_range', 'exchange_every', 'beta_max', 'cooling'},
                        'sim': {'simplex_step', 'simplex_log_step', 'simplex_reflection', 'simplex_expansion',
                                'simplex_contraction', 'simplex_shrink', 'simplex_max_iterations'}}
        ignored_params = set()
        thisalg = conf_dict['fit_type']
        if thisalg in ('pt', 'sa'):
            thisalg = 'bmc'
        for alg in alg_specific:
            if (thisalg != alg
               and not(alg == 'sim' and 'refine' in conf_dict and conf_dict['refine'] == 1)):
                ignored_params = ignored_params.union(alg_specific[alg])
        for k in ignored_params.intersection(set(conf_dict.keys())):
            print1('Warning: Configuration key %s is not used in fit_type %s, so I am ignoring it'
                            % (k, conf_dict['fit_type']))

    @staticmethod
    def postprocess_mcmc_keys(conf_dict):
        """
        Algorithms 'bmc', 'pt', and 'sa' have similar but non-identical valid config keys. This helper method
        does post-processing of config keys for these 3 algorithms

        :param conf_dict:
        :return:
        """
        # Check keys that only work for a subset of the 3 algorithms
        if conf_dict['fit_type'] != 'pt' and 'exchange_every' in conf_dict:
            if 'exchange_every' in conf_dict:
                print1('Warning: Configuration key exchange_every is not used in fit_type %s, so I am ignoring it'
                       % conf_dict['fit_type'])
            conf_dict['exchange_every'] = np.inf
        if conf_dict['fit_type'] != 'sa':
            for k in ['cooling', 'beta_max']:
                if k in conf_dict:
                    print1('Warning: Configuration key %s is not used in fit_type %s, so I am ignoring it'
                           % (k, conf_dict['fit_type']))
        if conf_dict['fit_type'] == 'sa':
            for k in ['burn_in', 'sample_every', 'output_hist_every', 'hist_bins', 'credible_intervals']:
                if k in conf_dict:
                    print1('Warning: Configuration key %s is not used in fit_type %s, so I am ignoring it'
                           % (k, conf_dict['fit_type']))

        # Create the starting list of betas based on the various available options. Warn if tried to do something weird
        if 'beta' not in conf_dict and 'beta_range' not in conf_dict:
            conf_dict['beta'] = [1.]
        if 'beta_range' in conf_dict:
            if len(conf_dict['beta_range']) != 2:
                raise PybnfError("Wrong number of entries in beta_range",
                                 "Config key 'beta_range' must have exactly 2 numbers: the min and the max.")
            if 'beta' in conf_dict:
                print1("Warning: Ignoring config key 'beta' because it is overridden by config key 'beta_range'")
            if conf_dict['fit_type'] != 'pt':
                print1("Warning: You used 'beta_range' with the method %s. This is an odd thing to do. Usually, you "
                       "would want all your replicates starting at the same beta value." % conf_dict['fit_type'])
            betalist = np.linspace(conf_dict['beta_range'][0], conf_dict['beta_range'][1], conf_dict['population_size'])
        elif len(conf_dict['beta']) > 1:
            betalist = conf_dict['beta']
            if conf_dict['fit_type'] != 'pt':
                print1("Warning: You specified multiple beta values with the method %s. This is an odd thing to do. "
                       "Usually, you would specify one beta value to use with all your replicates. " % conf_dict['fit_type'])
            if len(betalist) != conf_dict['population_size']:
                print1("Warning: You specified %i beta values, so I will run %i replicates instead of using your "
                       "population_size setting" % (len(betalist), len(betalist)))
                conf_dict['population_size'] = len(betalist)
        else:
            betalist = conf_dict['beta'] * conf_dict['population_size']  # n copies of the single beta value
            if conf_dict['fit_type'] == 'pt':
                print1("Warning: You specified a single beta value with the method pt. This makes the algorithm's "
                       "replica exchanges accomplish nothing. To make good use of this algorithm, set the key "
                       "'beta_range' or specify multiple values with the 'beta' key.")
        betalist.sort()
        conf_dict['beta_list'] = betalist

    @staticmethod
    def _req_user_params():
        """Configuration keys that the user must specify"""
        return {'models', 'population_size', 'max_iterations'}

    def _load_models(self):
        """
        Loads models specified in configuration file in a dictionary keyed on
        Model.name
        """
        # Force absolute paths for all simulator paths. Safe to do here because this is the main thread.
        home_dir = os.getcwd()

        def absolute(dir):
            # Convert relative path to absolute path
            return dir if dir[0] == '/' else home_dir + '/' + dir

        md = {}
        for mf in self.config['models']:
            # Initialize model type based on extension
            try:
                if re.search('\.bngl$', mf):
                    model = BNGLModel(mf)
                    model.bng_command = absolute(self.config['bng_command'])
                    logger.debug('Set model %s command to %s' % (mf, model.bng_command))
                elif re.search('\.xml$', mf):
                    model = SbmlModel(mf)
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

        if self.config['smoothing'] > 1:
            # Check for misuse of 'smoothing' feature
            stochastic = np.any([m.stochastic for m in md.values()])
            if not stochastic:
                print1('Warning: You specified smoothing=%i, but it looks like none of your models use a stochastic '
                       'method. All of your smoothing replicates will come out identical.' % self.config['smoothing'])

        return md


    def _load_simulators(self):

        model_types = set([type(m) for m in self.models.values()])
        logger.debug('Here are all your sweet model types: %s' % model_types)

        # For each model type that exists in the run, check that the simulator is available, and pass the simulator
        # path to the appropriate Model subclass
        if BNGLModel in model_types:
            if self.config['bng_command'] == '':
                raise PybnfError('Path to BNG2.pl not defined.  Please specify using the "bng_command" parameter '
                                 'in the configuration file or set the BNGPATH environmental variable')
            elif re.search(r'BNG2.pl', self.config['bng_command']) is None:
                raise PybnfError('The specified "bng_command" parameter in the configuration file must include the script '
                                 'name at the end of the path (e.g. /path/to/BNG2.pl)')
            else:  # check to make sure BNG2.pl is available
                try:
                    logger.info('Checking to make sure bng_command is appropriately set')
                    subprocess.run(self.config['bng_command'] + ' -v', shell=True, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError:
                    raise PybnfError('BioNetGen failed to execute.  Please check that "bng_command" parameter in the '
                                     'configuration file points to the BNG2.pl script or that the BNGPATH environmental '
                                     'variable is correctly set')

    def _load_actions(self):

        for (key, ActionType) in (('time_course', TimeCourse), ('param_scan', ParamScan)):
            # Iterate through all time courses and param scans included in the config dict, create the corresponding
            # Action objects, and add them to the appropriate model(s).
            for action_dict in self.config[key]:
                if 'model' in action_dict:
                    action = ActionType(action_dict)
                    try:
                        # Model lookup - should work if model name included the extension or not.
                        model_key = self._file_prefix(action_dict['model'], '(bngl|xml)')
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
        return re.sub("\."+ext, "", re.split('/', ef)[-1])

    def _load_exp_data(self):
        """
        Loads experimental data files in a dictionary keyed on data file prefix
        Also loads constraint files (which at this point are stored in the same structures as the exp files) and stores
        them in a set.
        """
        ed = {}
        csets = set()
        for m in self.config['models']:
            for ef in self.config[m]:
                if re.search("exp$", ef):
                    try:
                        d = Data(file_name=ef)
                    except FileNotFoundError:
                        raise PybnfError('Experimental data file %s was not found.' % ef)
                    ed[self._file_prefix(ef)] = d
                else:
                    cs = ConstraintSet(self._file_prefix(m, 'bngl'), self._file_prefix(ef, 'con'))
                    try:
                        cs.load_constraint_file(ef)
                    except FileNotFoundError:
                        raise PybnfError('Constraint file %s was not found' % ef)
                    csets.add(cs)
        return ed, csets

    def _check_actions(self):
        mapping = dict()
        for model in self.models.values():
            suffs = {s[1] for s in model.suffixes}
            efs_per_m = {self._file_prefix(ef) for ef in self.config[model.file_path] if re.search("\.exp$", ef)}
            if not efs_per_m <= suffs:
                for ef in efs_per_m:
                    if ef not in suffs:
                        raise UnmatchedExperimentalDataError("Action not specified for '%s.exp'" % ef,
                              "You specified that model %s corresponds to data file %s.exp, but I can't find the "
                              "corresponding action in the model file or config file. One of the actions in %s.bngl "
                              "needs to include the argument 'suffix=>\"%s\" ', or your config file needs to include "
                              "an action with the suffix %s." % (model.name, ef, model.name, ef, ef))
            logger.debug('Model %s was mapped to %s' % (model.name, efs_per_m))
            mapping[model.name] = efs_per_m
        return mapping

    def _load_obj_func(self):
        if self.config['objfunc'] == 'chi_sq':
            return ChiSquareObjective()
        elif self.config['objfunc'] == 'sos':
            return SumOfSquaresObjective()
        elif self.config['objfunc'] == 'norm_sos':
            return NormSumOfSquaresObjective()
        elif self.config['objfunc'] == 'ave_norm_sos':
            return AveNormSumOfSquaresObjective()
        raise UnknownObjectiveFunctionError("Objective function %s not defined" % self.config['objfunc'],
              "Objective function %s is not defined. Valid objective function choices are: "
              "chi_sq, sos, norm_sos, ave_norm_sos" % self.config['objfunc'])

    def _load_variables(self):
        """
        Loads the variable names from the config dict into FreeParameter instances.

        :return: a list of FreeParameter instances
        """
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
        """
        Verifies that each variable specified in the configuration appears in at least one model file, and each
        FREE parameter specified in a model file appears in the config file
        :return:
        """
        model_vars = set()
        for m in self.models.values():
            model_vars.update(m.get_param_names())

        variables_names = {v.name for v in self.variables}
        extra_in_conf = variables_names.difference(model_vars)
        extra_in_model = set(model_vars).difference(variables_names)
        extra_in_model = {p for p in extra_in_model if p[-8:] == '__FREE__'}

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
                suff = self._file_prefix(ef)
                def checkval(v):
                    if v not in valid:
                        raise PybnfError("Invalid normalization type '%s'" % self.config['normalization'][ef],
                                         "Invalid normalization type '%s'. Options are: init, peak, zero, unit" %
                                         self.config['normalization'][ef] + seedoc)
                if type(val) == str:
                    # This exp file has a single normalization type for all columns
                    checkval(val)
                else:
                    # This exp file has a list of one or more pairs specifying (normalization_type, [columns])
                    for (i, (ntype, cols)) in enumerate(val):
                        checkval(ntype)
                        new_cols = []
                        if type(cols[0]) == int:
                            # Need to convert to string labels, because the indices into the sim data will be different
                            to_convert = cols
                            for label in self.exp_data[suff].cols:
                                ci = self.exp_data[suff].cols[label]
                                if ci in to_convert:
                                    to_convert.remove(ci)
                                    new_cols.append(label)
                            if len(to_convert) > 0:
                                raise PybnfError("Invalid normalization column %s for file %s" % (to_convert[0], ef),
                                                 "Specified normalization for column %i in file %s, but that file "
                                                 "contains only %i columns." % (
                                                 to_convert[0], ef, self.exp_data[suff].data.shape[1]) + seedoc)
                        else:
                            new_cols = cols
                        new_cols_iter = new_cols
                        for c in new_cols_iter:
                            if c not in self.exp_data[suff].cols:
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


class UnknownObjectiveFunctionError(PybnfError):
    pass


class UnspecifiedConfigurationKeyError(PybnfError):
    pass


class UnmatchedExperimentalDataError(PybnfError):
    pass
