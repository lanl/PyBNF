"""pybnf.config: classes and methods for configuring the fit"""


from .data import Data
from .objective import ChiSquareObjective, SumOfSquaresObjective, NormSumOfSquaresObjective, \
    AveNormSumOfSquaresObjective
from .pset import BNGLModel, ModelError
from .printing import verbosity, print0, print1, print2, PybnfError

import numpy as np
import os
import re
import logging


def init_logging():
    fmt = logging.Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')

    dfh = logging.FileHandler('bnf_debug.log')
    dfh.setLevel(logging.DEBUG)
    efh = logging.FileHandler('bnf_errors.log')
    efh.setLevel(logging.ERROR)
    dfh.setFormatter(fmt)
    efh.setFormatter(fmt)

    dlog = logging.getLogger('distributed')
    dlog.setLevel(logging.DEBUG)
    dlog.addHandler(dfh)
    dlog.addHandler(efh)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(dfh)
    root.addHandler(efh)


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

        self.config = self.default_config()
        for k, v in d.items():
            self.config[k] = v

        self.models = self._load_models()
        self.mapping = self._check_actions()  # dict of model prefix -> set of experimental data prefixes
        self.exp_data = self._load_exp_data()
        self.obj = self._load_obj_func()
        self.variables, self.variables_specs = self._load_variables()
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

            'mutation_rate': 0.5, 'mutation_factor': 1.0, 'islands': 1, 'migrate_every': 20, 'num_to_migrate': 3,
            'stop_tolerance': 0.002,

            'particle_weight': 1.0, 'adaptive_n_max': 30, 'adaptive_n_stop': np.inf, 'adaptive_abs_tol': 0.0,
            'adaptive_rel_tol': 0.0, 'cognitive': 1.5, 'social': 1.5,

            'local_min_limit': 5,

            'step_size': 0.2, 'burn_in': 10000, 'sample_every': 100, 'output_hist_every': 10000, 'hist_bins': 10,
            'credible_intervals': [68., 95.],

            'simplex_step': 1.0, 'simplex_reflection': 1.0, 'simplex_expansion':1.0, 'simplex_contraction': 0.5,
            'simplex_shrink': 0.5,

            'wall_time_gen': 3600,
            'wall_time_sim': 3600,
            'normalization': None
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
                                'credible_intervals'},
                        'sim': {'simplex_step', 'simplex_log_step', 'simplex_reflection', 'simplex_expansion',
                                'simplex_contraction', 'simplex_shrink', 'simplex_max_iterations'}}
        ignored_params = set()
        for alg in alg_specific:
            if (conf_dict['fit_type'] != alg
               and not(alg == 'sim' and 'refine' in conf_dict and conf_dict['refine'] == 1)):
                ignored_params = ignored_params.union(alg_specific[alg])
        for k in ignored_params.intersection(set(conf_dict.keys())):
            print1('Warning: Configuration key %s is not used in fit_type %s, so I am ignoring it'
                            % (k, conf_dict['fit_type']))

    @staticmethod
    def _req_user_params():
        """Configuration keys that the user must specify"""
        return {'models', 'population_size', 'max_iterations'}

    def _load_models(self):
        """
        Loads models specified in configuration file in a dictionary keyed on
        Model.name
        """
        md = {}
        for mf in self.config['models']:
            try:
                model = BNGLModel(mf)
            except FileNotFoundError:
                raise PybnfError('Model file %s was not found.' % mf)
            except ModelError as e:
                raise PybnfError('In model file %s: %s' % (mf, e.message))
            md[model.name] = model

        if self.config['smoothing'] > 1:
            # Check for misuse of 'smoothing' feature
            stochastic = np.any([m.stochastic for m in md.values()])
            if not stochastic:
                print1('Warning: You specified smoothing=%i, but it looks like none of your models use a stochastic '
                       'method. All of your smoothing replicates will come out identical.' % self.config['smoothing'])

        return md

    @staticmethod
    def _exp_file_prefix(ef):
        return re.sub(".exp", "", re.split('/', ef)[-1])

    def _load_exp_data(self):
        """
        Loads experimental data files in a dictionary keyed on data file prefix
        """
        ed = {}
        for ef in self.config['exp_data']:
            try:
                d = Data(file_name=ef)
            except FileNotFoundError:
                raise PybnfError('Experimental data file %s was not found.' % ef)
            ed[self._exp_file_prefix(ef)] = d
        return ed

    def _check_actions(self):
        mapping = dict()
        for model in self.models.values():
            suffs = {s[1] for s in model.suffixes}
            efs_per_m = {self._exp_file_prefix(ef) for ef in self.config[model.file_path]}
            if not efs_per_m <= suffs:
                for ef in efs_per_m:
                    if ef not in suffs:
                        raise UnmatchedExperimentalDataError("Action not specified for '%s.exp'" % ef,
                              "You specified that model %s.bngl corresponds to data file %s.exp, but I can't find the "
                              "corresponding action in the model file. One of the actions in %s.bngl needs to include "
                              "the argument 'suffix=>\"%s\" '." % (model.name, ef, model.name, ef))
            logging.debug('Model %s was mapped to %s' % (model.name, efs_per_m))
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
        Loads the variable names from the config dict, and stores them in more easily accessible data structures.
        :return: 2-tuple (variables, variables_specs), where variables in a list of the variable names, and
         variables_specs is a list of 4-tuples (variable_name, variable_type, min_value, max_value).
         For static_list_var variables, variables_specs instead takes the form (variable_name, static_list_var,
         [list of possible values], None)
         For var and logvar variables (for Simplex algorithm), variables_specs takes the form (variable_name,
         variable_type, init_value, init_step) where init_step may be read from the global setting.
        """
        variables = []
        variables_specs = []
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
                               "Valid keywords for other algorithms are: random_var, normrandom_var, \n"
                               "lognormrandom_var, loguniform_var." % (k[1], k[0]))
                    variables.append(k[1])
                    if k[0] == 'static_list_var':
                        variables_specs.append((k[1], k[0], self.config[k], None))
                    elif k[0] in ('var', 'logvar'):
                        # 2nd number (step size) may be absent, must fill in appropriately
                        if len(self.config[k]) >= 2:
                            stepsize = self.config[k][1] # easy, it was right there
                        elif k[0] == 'logvar' and 'simplex_log_step' in k:
                            stepsize = self.config['simplex_log_step']  # This one is preferred if it's there
                        else:
                            stepsize = self.config['simplex_step']  # This is always there because it had a default set
                        variables_specs.append((k[1], k[0], self.config[k][0], stepsize))
                    else:
                        variables_specs.append((k[1], k[0], self.config[k][0], self.config[k][1]))
        return variables, variables_specs


    def _check_variable_correspondence(self):
        """
        Verifies that each variable specified in the configuration appears in at least one model file, and each
        FREE parameter specified in a model file appears in the config file
        :return:
        """
        model_vars = set()
        for m in self.models.values():
            model_vars.update(m.param_names)

        extra_in_conf = set(self.variables).difference(model_vars)
        extra_in_model = set(model_vars).difference(self.variables)
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
        valid = ('init', 'peak', 'zero')
        if type(self.config['normalization']) == dict:
            # Iterate through the keys, which should be .exp file names. Check that these are actual exp files that
            # are used in the fitting, then add to the dictionary just the suffix, for easier lookup later
            newdict = dict()
            for ef in self.config['normalization']:
                if ef not in self.config['exp_data']:
                    # Could be just a warning?
                    raise PybnfError("Invalid exp file %s under the normalization key" % ef,
                                     "The exp file %s given under the 'normalization' keyword is not associated with "
                                     "any model." % ef)
                val = self.config['normalization'][ef]
                if val not in valid:
                    raise PybnfError("Invalid normalization type '%s'" % self.config['normalization'][ef],
                                     "Invalid normalization type '%s'. Options are: init, peak, zero" %
                                     self.config['normalization'][ef])
                newdict[self._exp_file_prefix(ef)] = val
            self.config['normalization'].update(newdict)
        elif type(self.config['normalization']) == str:
            if self.config['normalization'] not in valid:
                raise PybnfError("Invalid normalization type '%s'" % self.config['normalization'],
                                 "Invalid normalization type '%s'. Options are: init, peak, zero" %
                                 self.config['normalization'])


class UnknownObjectiveFunctionError(PybnfError):
    pass


class UnspecifiedConfigurationKeyError(PybnfError):
    pass


class UnmatchedExperimentalDataError(PybnfError):
    pass