"""pybnf.config: classes and methods for configuring the fit"""


from .data import Data
from .objective import ChiSquareObjective
from .pset import Model
import numpy as np
import re


class Configuration(object):
    def __init__(self, d=dict()):
        """
        Instantiates a Configuration object using a dictionary generated
        by the configuration file parser.  Default key, value pairs are used
        when possible for pairs not present in the provided dictionary.

        :param d: The result from parsing a configuration file
        :type d: dict
        """
        if not self._req_user_params() <= d.keys():
            unspecified_keys = []
            for k in self._req_user_params():
                if k not in d.keys():
                    unspecified_keys.append(k)
            raise UnspecifiedConfigurationKeyError(
                "The following configuration keys must be specified:\n\t"",".join(unspecified_keys))

        self.config = self.default_config()
        for k, v in d.items():
            self.config[k] = v

        self.models = self._load_models()
        self.mapping = self._check_actions()  # dict of model prefix -> set of experimental data prefixes
        self.exp_data = self._load_exp_data()
        self.obj = self._load_obj_func()
        self.variables, self.variables_specs = self._load_variables()

    def default_config(self):
        """Default configuration values"""
        default = {
            'objfunc': 'chi_sq',

            'particle_weight': 1.0, 'adaptive_n_max': 30, 'adaptive_n_stop': np.inf, 'adaptive_abs_tol': 0.0,
            'adaptive_rel_tol': 0.0
        }
        return default

    @staticmethod
    def _req_user_params():
        """Configuration keys that the user must specify"""
        return {'models'}

    def _load_models(self):
        """
        Loads models specified in configuration file in a dictionary keyed on
        Model.name
        """
        md = {}
        for mf in self.config['models']:
            model = Model(mf)
            md[model.name] = model
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
            d = Data(file_name=ef)
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
                        raise UnmatchedExperimentalDataError("Action not specified for '%s.exp'" % ef)
            mapping[model.name] = efs_per_m
        return mapping

    def _load_obj_func(self):
        if self.config['objfunc'] == 'chi_sq':
            return ChiSquareObjective()
        raise UnknownObjectiveFunctionError("Objective function %s not defined" % self.config['objfunc'])

    def _load_variables(self):
        """
        Loads the variable names from the config dict, and stores them in more easily accessible data structures.
        :return: 2-tuple (variables, variables_specs), where variables in a list of the variable names, and
         variables_specs is a list of 4-tuples (variable_name, variable_type, min_value, max_value).
         For static_list_var variables, variables_specs instead takes the form (variable_name, static_list_var,
         [list of possible values], None)
        """
        variables = []
        variables_specs = []
        for k in self.config.keys():
            if isinstance(k, tuple):
                if re.search('var$', k[0]):
                    variables.append(k[1])
                    if k[0] == 'static_list_var':
                        variables_specs.append((k[1], k[0], self.config[k], None))
                    else:
                        variables_specs.append((k[1], k[0], self.config[k][0], self.config[k][1]))
        return variables, variables_specs


class UnknownObjectiveFunctionError(Exception):
    pass


class UnspecifiedConfigurationKeyError(Exception):
    pass


class UnmatchedExperimentalDataError(Exception):
    pass