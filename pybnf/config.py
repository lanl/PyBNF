"""pybnf.config: classes and methods for configuring the fit"""


from .data import Data
from .objective import ChiSquareObjective
from .pset import Model

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
        self.variables = self._load_variables()

    def default_config(self):
        """Default configuration values"""
        default = {
            'objfunc': 'chi_sq'
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
        variables = []
        for k in self.config.keys():
            if isinstance(k, tuple):
                if re.search('var$', k[0]):
                    variables.append(k[1])
        return variables


class UnknownObjectiveFunctionError(Exception):
    pass


class UnspecifiedConfigurationKeyError(Exception):
    pass


class UnmatchedExperimentalDataError(Exception):
    pass