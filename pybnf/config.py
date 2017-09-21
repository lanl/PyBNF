"""pybnf.config: classes and methods for configuring the fit"""


from .pset import Model


class Configuration(object):
    def __init__(self, d=dict()):
        """
        Instantiates a Configuration object using a dictionary generated
        by the configuration file parser.  Default key, value pairs are used
        when possible for pairs not present in the provided dictionary.

        :param d: The result from parsing a configuration file
        :type d: dict
        """
        if not self._req_user_params() < d.keys():
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

    def default_config(self):
        """Default configuration values"""
        return dict()

    @staticmethod
    def _req_user_params():
        """Configuration keys that the user must specify"""
        return {'models'}

    def _load_models(self):
        """Loads models specified in configuration file"""
        md = {}
        print(self.config['models'])
        for mf in self.config['models']:
            model = Model(mf)
            md[model.name] = model
        return md


class UnspecifiedConfigurationKeyError(Exception):
    pass