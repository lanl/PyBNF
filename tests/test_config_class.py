from .context import config
from .context import data
from .context import objective
from .context import pset
from .context import printing
from .context import raises

import numpy as np
import operator


class TestConfig(object):
    @classmethod
    def setup_class(cls):
        cls.cf0 = {'models': {'bngl_files/Tricky.bngl'},
                   'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                   'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                   ('uniform_var', 'koff__FREE'): [4., 5., 'u'],
                   ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
                   ('normal_var', 'kase__FREE'): [28., 5.],
                   ('uniform_var', 'pase__FREE'): [6., 7.],
                   'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                   'normalization': {'bngl_files/p1_5.exp': 'init'},
                   'param_scan': [{'model': 'Tricky.bngl', 'param': 'koff__FREE', 'min': '1', 'max': '10', 'step': '1', 'time': '3600'}]}
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
        assert isinstance(c.exp_data['Tricky']['p1_5'], data.Data)
        assert 'p1_5' in c.mapping['Tricky']
        assert 'thing' in c.mapping['Tricky']
        assert isinstance(c.obj, objective.ChiSquareObjective)
        sorted_vars = sorted(c.variables, key=operator.attrgetter('name'))
        assert sorted_vars[0].name == '__koff2__FREE'
        assert sorted_vars[0].type == 'loguniform_var'
        assert sorted_vars[0].bounded
        assert sorted_vars[0].log_space
        assert not sorted_vars[1].log_space
        assert not sorted_vars[1].bounded
        assert [v.name for v in sorted_vars] == ['__koff2__FREE', 'kase__FREE', 'koff__FREE', 'pase__FREE']
        assert c.config['normalization']['p1_5'] == [('init', ['R_free'])]
        assert c.config['cluster_type'] is None

    def test_random_seed_default(self):
        assert config.Configuration.default_config()['random_seed'] is None

    def test_bngl_backend_default(self):
        assert config.Configuration.default_config()['bngl_backend'] == 'auto'

    @raises(printing.PybnfError)
    def test_invalid_bngl_backend(self):
        c = object.__new__(config.Configuration)
        c.config = {
            'models': set(),
            'sbml_backend': 'roadrunner',
            'bngl_backend': 'unknown',
            'wall_time_sim': 0,
        }
        c._load_models()

    @raises(printing.PybnfError)
    def test_random_seed_must_be_nonnegative(self):
        c = dict(self.cf1)
        c['random_seed'] = -1
        config.Configuration(c)

    @raises(printing.PybnfError)
    def test_random_seed_must_fit_numpy_seed_range(self):
        c = dict(self.cf1)
        c['random_seed'] = 2**32
        config.Configuration(c)

    def test_config_normalization(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                                  'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                                  'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                                  ('uniform_var', 'koff__FREE'): [4., 5.],
                                  ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
                                  ('normal_var', 'kase__FREE'): [28., 5.],
                                  ('uniform_var', 'pase__FREE'): [6., 7.],
                                  'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                                  'normalization': {'bngl_files/p1_5.exp': [('init', [1])],
                                                    'bngl_files/thing.exp': [('peak', ['Ag_total'])]}})
        assert c.config['normalization']['p1_5'] == [('init', ['R_free'])]
        assert c.config['normalization']['thing'] == [('peak', ['Ag_total'])]

    def test_normalization_consecutive_sd_columns_all_removed(self):
        # _SD columns can't be normalized separately, so they're dropped from the
        # normalization list. With two *consecutive* _SD columns the old loop
        # iterated and removed from the same list object, skipping (and wrongly
        # keeping) the second. par1.exp columns: time, x, y, x_SD, y_SD.
        c = object.__new__(config.Configuration)
        d = data.Data(file_name='bngl_files/par1.exp')
        c.exp_data = {'parabola': {'par1': d}}
        c.config = {'normalization': {'bngl_files/par1.exp': [('init', ['x_SD', 'y_SD'])]},
                    'exp_data': {'bngl_files/par1.exp'},
                    'models': {'bngl_files/parabola.bngl'},
                    'bngl_files/parabola.bngl': ['bngl_files/par1.exp']}
        c._postprocess_normalization()
        # Both _SD columns gone -> empty list (pre-fix left ['y_SD']).
        assert c.config['normalization']['par1'] == [('init', [])]

    def test_crossover_number_warns_when_ignored(self, monkeypatch):
        # 'crossover_number' is a DREAM-family key; for am it's unused and should
        # warn. The warning moved out of MCMCFamilyConfig.postprocess into the
        # unified, schema-derived Configuration.check_unused_keys (#401, ADR-0014);
        # postprocess now only transforms config. Assert the warning still names the
        # key, from its new home.
        import pybnf.config as cfgmod
        warnings = []
        monkeypatch.setattr(cfgmod, 'print1', lambda msg, *a, **k: warnings.append(msg))
        conf = {'fit_type': 'am', 'population_size': 10, 'crossover_number': 3}
        config.Configuration.check_unused_keys(conf)
        assert any('crossover_number' in w for w in warnings)

    def test_postprocess_no_longer_warns(self, monkeypatch):
        # The warn-only branches are gone from postprocess (#401); driving it must
        # emit no print1 warning, only the beta-ladder transformation.
        from pybnf.algorithms.samplers import base as samplers_base
        from pybnf.algorithms.samplers.base import MCMCFamilyConfig
        warnings = []
        monkeypatch.setattr(samplers_base, 'print1', lambda msg, *a, **k: warnings.append(msg))
        conf = {'fit_type': 'am', 'population_size': 10, 'crossover_number': 3,
                'exchange_every': 5, 'cooling': 0.1}
        MCMCFamilyConfig.postprocess(conf, 'am')
        assert warnings == []
        assert conf['exchange_every'] == np.inf  # transformation still happens

    def test_check_tolerates_tuple_keys(self):
        # CFG-CHECK-1: a free-parameter tuple key (e.g. ('uniform_var','p1')) used to
        # crash the model-checking unused-key scan -- first on re.search(regex, tuple),
        # then on '...%s...' % tuple. The isinstance(k, str) guard in _is_unused_key
        # keeps re.search/% off tuples, so check_unused_keys must not raise; and
        # _strip_uncheckable_keys removes refine/bootstrap while leaving model-path
        # and tuple keys untouched.
        conf = {'fit_type': 'check', 'model.bngl': ['a.exp'],
                ('uniform_var', 'p1'): [-10.0, 10.0], 'refine': 1, 'bootstrap': 1}
        out = config.Configuration._strip_uncheckable_keys(conf)
        assert ('uniform_var', 'p1') in out                    # tuple key survives
        assert 'refine' not in out and 'bootstrap' not in out  # uncheckable stripped
        assert 'model.bngl' in out                             # model path untouched
        config.Configuration.check_unused_keys(conf)           # must not raise on the tuple key

    # --- _check_variable_keyword_combination (var/logvar-vs-prior rule, #404) ---
    # Driven directly on a bare instance (no full build): it only reads self.config
    # keys + the fit_type. The start-point optimizers are sim/powell (point-only)
    # and cmaes (also box-capable, start_from_box).

    @staticmethod
    def _kw_checker(var_tuples):
        c = object.__new__(config.Configuration)
        c.config = dict(var_tuples)
        return c

    def test_kw_combo_box_optimizer_accepts_bounded_priors(self):
        """CMA-ES (start_from_box) accepts a bounded-prior box -> no raise."""
        c = self._kw_checker({('uniform_var', 'p1'): [-10., 10.],
                              ('loguniform_var', 'p2'): [0.1, 100.]})
        c._check_variable_keyword_combination('cmaes')  # must not raise

    def test_kw_combo_box_optimizer_accepts_point_start(self):
        """CMA-ES still accepts a single var/logvar start point -> no raise."""
        c = self._kw_checker({('var', 'p1'): [1., 0.5], ('logvar', 'p2'): [3.]})
        c._check_variable_keyword_combination('cmaes')  # must not raise

    @raises(printing.PybnfError)
    def test_kw_combo_box_optimizer_rejects_unbounded_prior(self):
        """A box search needs a bounded box: normal_var (unbounded) is rejected."""
        c = self._kw_checker({('normal_var', 'p1'): [0., 1.]})
        c._check_variable_keyword_combination('cmaes')

    @raises(printing.PybnfError)
    def test_kw_combo_box_optimizer_rejects_mixed_point_and_box(self):
        """A var point mixed with a uniform_var box is ambiguous -> rejected."""
        c = self._kw_checker({('var', 'p1'): [1.], ('uniform_var', 'p2'): [-10., 10.]})
        c._check_variable_keyword_combination('cmaes')

    @raises(printing.PybnfError)
    def test_kw_combo_point_only_optimizer_rejects_bounded_prior(self):
        """Powell (point-only, not start_from_box) still rejects uniform_var."""
        c = self._kw_checker({('uniform_var', 'p1'): [-10., 10.]})
        c._check_variable_keyword_combination('powell')

    def test_kw_combo_point_only_optimizer_accepts_point_start(self):
        """Negative control: Simplex with var/logvar -> no raise."""
        c = self._kw_checker({('var', 'p1'): [1., 0.5], ('logvar', 'p2'): [3.]})
        c._check_variable_keyword_combination('sim')  # must not raise

    @raises(printing.PybnfError)
    def test_kw_combo_sampler_rejects_var_keyword(self):
        """A non-start-point method (de) rejects the var/logvar start-point keyword."""
        c = self._kw_checker({('var', 'p1'): [1.]})
        c._check_variable_keyword_combination('de')

    def test_kw_combo_sampler_accepts_priors(self):
        """Negative control: de with uniform_var priors -> no raise."""
        c = self._kw_checker({('uniform_var', 'p1'): [-10., 10.]})
        c._check_variable_keyword_combination('de')  # must not raise

    # --- _check_variable_correspondence (the config-level free-parameter guard) ---
    # Tricky.bngl declares __FREE params: koff__FREE, __koff2__FREE, kase__FREE, pase__FREE.

    @staticmethod
    def _corr_conf(free_params):
        """A minimal loadable config over Tricky.bngl with the given free params."""
        conf = {'models': {'bngl_files/Tricky.bngl'},
                'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                'fit_type': 'de', 'population_size': 10, 'max_iterations': 10}
        conf.update(free_params)
        return conf

    def test_all_free_params_valid_loads(self):
        """Negative control: all four model __FREE params declared -> loads, no raise."""
        c = config.Configuration(self._corr_conf({
            ('uniform_var', 'koff__FREE'): [4., 5.],
            ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
            ('normal_var', 'kase__FREE'): [28., 5.],
            ('uniform_var', 'pase__FREE'): [6., 7.],
        }))
        assert {v.name for v in c.variables} == {
            'koff__FREE', '__koff2__FREE', 'kase__FREE', 'pase__FREE'}

    @raises(printing.PybnfError)
    def test_free_param_not_in_any_model_raises(self):
        """config -> model: a free parameter present in no model is rejected at load
        (catches a typo before any simulation runs; makes the per-model silent skip safe)."""
        config.Configuration(self._corr_conf({
            ('uniform_var', 'koff__FREE'): [4., 5.],
            ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
            ('normal_var', 'kase__FREE'): [28., 5.],
            ('uniform_var', 'pase__FREE'): [6., 7.],
            ('uniform_var', 'bogus_typo__FREE'): [1., 2.],  # not in Tricky.bngl
        }))

    @raises(printing.PybnfError)
    def test_model_free_param_missing_from_config_raises(self):
        """model -> config: a __FREE declared in the model but not in the .conf is rejected
        (pase__FREE omitted below)."""
        config.Configuration(self._corr_conf({
            ('uniform_var', 'koff__FREE'): [4., 5.],
            ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
            ('normal_var', 'kase__FREE'): [28., 5.],
        }))

    @raises(printing.PybnfError)
    def test_normalization_err(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                                  'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                                  'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                                  ('uniform_var', 'koff__FREE'): [4., 5.],
                                  ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
                                  ('normal_var', 'kase__FREE'): [28., 5.],
                                  ('uniform_var', 'pase__FREE'): [6., 7.],
                                  'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                                  'normalization': {'bngl_files/p1_5.exp': [('init', [1])],
                                                    'bngl_files/thing.exp': [('peak', ['R_free'])]}})

    @raises(printing.PybnfError)
    def test_normalization_err2(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                                  'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                                  'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                                  ('uniform_var', 'koff__FREE'): [4., 5.],
                                  ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
                                  ('normal_var', 'kase__FREE'): [28., 5.],
                                  ('uniform_var', 'pase__FREE'): [6., 7.],
                                  'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                                  'normalization': {'bngl_files/p1_5.exp': [('init', [2])],
                                                    'bngl_files/thing.exp': [('peak', ['Ag_total'])]}})

    @raises(printing.PybnfError)
    def test_incorrect_bng_command(self):
        c = config.Configuration({'models': {'bngl_files/Tricky.bngl'},
                                  'bng_command': "/incorrect/path/to/BNG2.pl",
                                  'bngl_files/Tricky.bngl': ['bngl_files/p1_5.exp', 'bngl_files/thing.exp'],
                                  'exp_data': {'bngl_files/p1_5.exp', 'bngl_files/thing.exp'},
                                  ('uniform_var', 'koff__FREE'): [4., 5.],
                                  ('loguniform_var', '__koff2__FREE'): [0.01, 1e5],
                                  ('normal_var', 'kase__FREE'): [28., 5.],
                                  ('uniform_var', 'pase__FREE'): [6., 7.],
                                  'fit_type': 'de', 'population_size': 10, 'max_iterations': 10,
                                  'normalization': {'bngl_files/p1_5.exp': [('init', [2])],
                                                    'bngl_files/thing.exp': [('peak', ['Ag_total'])]}})

    def test_load_t_length_xml_integer_step(self):
        """_load_t_length should compute step count from time and step for XML models"""
        c = object.__new__(config.Configuration)
        c.config = {
            'models': {'model.xml'},
            'time_course': [{'suffix': 'tc', 'step': '10', 'time': '100'}],
            'fit_type': 'de',
        }
        result = c._load_t_length()
        assert result == {'tc': 10}

    def test_load_t_length_xml_float_step(self):
        """_load_t_length should compute correct step count with float step (#354)"""
        c = object.__new__(config.Configuration)
        c.config = {
            'models': {'model.xml'},
            'time_course': [{'suffix': 'tc', 'step': '0.5', 'time': '100'}],
            'fit_type': 'de',
        }
        result = c._load_t_length()
        assert result == {'tc': 200}

    def test_load_t_length_xml_default_step(self):
        """_load_t_length should default to step=1 when step is not specified"""
        c = object.__new__(config.Configuration)
        c.config = {
            'models': {'model.xml'},
            'time_course': [{'suffix': 'tc', 'time': '50'}],
            'fit_type': 'de',
        }
        result = c._load_t_length()
        assert result == {'tc': 50}

    def test_load_t_length_xml_multiple_time_courses(self):
        """_load_t_length should handle multiple time_course entries for XML models (#354)"""
        c = object.__new__(config.Configuration)
        c.config = {
            'models': {'model.xml'},
            'time_course': [
                {'suffix': 'tc1', 'step': '10', 'time': '100'},
                {'suffix': 'tc2', 'step': '0.5', 'time': '50'},
            ],
            'fit_type': 'de',
        }
        result = c._load_t_length()
        assert result == {'tc1': 10, 'tc2': 100}

    @raises(config.UnspecifiedConfigurationKeyError)
    def test_bad_config_init(self):
        config.Configuration(dict())

    @raises(config.UnmatchedExperimentalDataError)
    def test_unmatched_data(self):
        config.Configuration(self.cf1)
