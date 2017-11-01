from .context import config
from .context import data
from .context import objective
from .context import pset

from nose.tools import raises


class TestConfig(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.cf0 = {'models': {'bngl_files/Tricky.bngl'},
                   'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                   ('random_var', 'avar__FREE__'): [4., 5.],
                   ('loguniform_var', 'bvar__FREE__'): [0.01, 1e5],
                   ('static_list_var', 'cvar__FREE__'): [17., 23., 37., 42.]}
        cls.cf1 = {'models': {'bngl_files/TrickyUS.bngl'},
                   'bngl_files/TrickyUS.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'}}

    @classmethod
    def teardown_class(cls):
        pass

    def test_config_init(self):
        c = config.Configuration(self.cf0)
        assert isinstance(c.models['Tricky'], pset.Model)
        assert isinstance(c.exp_data['p1_5'], data.Data)
        assert 'p1_5' in c.mapping['Tricky']
        assert 'thing' in c.mapping['Tricky']
        assert isinstance(c.obj, objective.ChiSquareObjective)
        assert sorted(c.variables) == ['avar__FREE__', 'bvar__FREE__', 'cvar__FREE__']
        assert sorted(c.variables_specs) == [('avar__FREE__', 'random_var', 4., 5.),
                                             ('bvar__FREE__', 'loguniform_var', 0.01, 1e5),
                                             ('cvar__FREE__', 'static_list_var', [17., 23., 37., 42.], None)]

    @raises(config.UnspecifiedConfigurationKeyError)
    def test_bad_config_init(self):
        config.Configuration(dict())

    @raises(config.UnmatchedExperimentalDataError)
    def test_unmatched_data(self):
        config.Configuration(self.cf1)
