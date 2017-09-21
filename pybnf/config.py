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

    def default_config(self):
        pass

    @staticmethod
    def _req_user_params():
        return set('models' 'exp_data')


class UnspecifiedConfigurationKeyError(Exception):
    pass