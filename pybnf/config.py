"""pybnf.config: classes and methods for configuring the fit"""


from .data import Data
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

    def default_config(self):
        """Default configuration values"""
        return dict()

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
                raise UnmatchedExperimentalDataError
            mapping[model.name] = efs_per_m
        return mapping

    @staticmethod
    def _exp_file_prefix(ef):
        return re.sub(".exp", "", re.split('/', ef)[-1])


class UnspecifiedConfigurationKeyError(Exception):
    pass


class UnmatchedExperimentalDataError(Exception):
    pass