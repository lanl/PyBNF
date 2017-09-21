from .context import config
from .context import data
from .context import pset

from nose.tools import raises


class TestConfig(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.cf0 = {'models': ['bngl_files/Tricky.bngl'],
                   'bngl_files/Tricky.bngl': ['bngl_files/Tricky_p1_5.exp', 'bngl_files/Tricky_thing.exp']}

    @classmethod
    def teardown_class(cls):
        pass

    def test_config_init(self):
        c = config.Configuration(self.cf0)
        assert isinstance(c.models['Tricky'], pset.Model)
        assert isinstance(c.exp_data['Tricky'][0], data.Data)

    @raises(config.UnspecifiedConfigurationKeyError)
    def test_bad_config_init(self):
        config.Configuration(dict())
