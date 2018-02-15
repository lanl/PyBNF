from .context import config
from .context import data
from .context import objective
from .context import pset
from .context import printing

from nose.tools import raises


class TestConfig(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.cf0 = {'models': {'bngl_files/Tricky.bngl'},
                   'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                   ('random_var', 'koff__FREE__'): [4., 5.],
                   ('loguniform_var', '__koff2__FREE__'): [0.01, 1e5],
                   ('static_list_var', 'kase__FREE__'): [17., 23., 37., 42.],
                   ('random_var', 'pase__FREE__'): [6., 7.],
                   'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                   'normalization': {'bngl_files/p1_5.exp': 'init'}}
        cls.cf1 = {'models': {'bngl_files/TrickyUS.bngl'},
                   'bngl_files/TrickyUS.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'}, 'fit_type': 'de',
                   'population_size': 10, 'max_iterations': 10}

    @classmethod
    def teardown_class(cls):
        pass

    def test_config_init(self):
        c = config.Configuration(self.cf0)
        assert isinstance(c.models['Tricky'], pset.BNGLModel)
        assert isinstance(c.exp_data['p1_5'], data.Data)
        assert 'p1_5' in c.mapping['Tricky']
        assert 'thing' in c.mapping['Tricky']
        assert isinstance(c.obj, objective.ChiSquareObjective)
        assert sorted(c.variables) == ['__koff2__FREE__', 'kase__FREE__', 'koff__FREE__', 'pase__FREE__']
        assert sorted(c.variables_specs) == [('__koff2__FREE__', 'loguniform_var', 0.01, 1e5),
                                             ('kase__FREE__', 'static_list_var', [17., 23., 37., 42.], None),
                                             ('koff__FREE__', 'random_var', 4., 5.),
                                             ('pase__FREE__', 'random_var', 6., 7.),
                                             ]
        assert c.config['normalization']['p1_5'] == 'init'

    def test_config_normalization(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                   'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                   ('random_var', 'koff__FREE__'): [4., 5.],
                   ('loguniform_var', '__koff2__FREE__'): [0.01, 1e5],
                   ('static_list_var', 'kase__FREE__'): [17., 23., 37., 42.],
                   ('random_var', 'pase__FREE__'): [6., 7.],
                   'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                   'normalization': {'bngl_files/p1_5.exp': [('init', [1])], 'bngl_files/thing.exp': [('peak', ['Ag_total'])]}})
        assert c.config['normalization']['p1_5'] == [('init', ['R_free'])]
        assert c.config['normalization']['thing'] == [('peak', ['Ag_total'])]

    @raises(printing.PybnfError)
    def test_normalization_err(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                   'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                   ('random_var', 'koff__FREE__'): [4., 5.],
                   ('loguniform_var', '__koff2__FREE__'): [0.01, 1e5],
                   ('static_list_var', 'kase__FREE__'): [17., 23., 37., 42.],
                   ('random_var', 'pase__FREE__'): [6., 7.],
                   'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                   'normalization': {'bngl_files/p1_5.exp': [('init', [1])], 'bngl_files/thing.exp': [('peak', ['R_free'])]}})

    @raises(printing.PybnfError)
    def test_normalization_err2(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                   'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                   ('random_var', 'koff__FREE__'): [4., 5.],
                   ('loguniform_var', '__koff2__FREE__'): [0.01, 1e5],
                   ('static_list_var', 'kase__FREE__'): [17., 23., 37., 42.],
                   ('random_var', 'pase__FREE__'): [6., 7.],
                   'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                   'normalization': {'bngl_files/p1_5.exp': [('init', [2])], 'bngl_files/thing.exp': [('peak', ['Ag_total'])]}})

    @raises(config.UnspecifiedConfigurationKeyError)
    def test_bad_config_init(self):
        config.Configuration(dict())

    @raises(config.UnmatchedExperimentalDataError)
    def test_unmatched_data(self):
        config.Configuration(self.cf1)
