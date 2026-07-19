from .context import config
from .context import data
from .context import objective
from .context import parse
from .context import pset
from .context import printing
from .context import raises

import json
import numpy as np
import operator
import os
import pytest


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
    # keys + the fit_type. All three start-point optimizers are now box-capable
    # (start_from_box): cmaes (#404) and -- via concurrent multi-start, #498/ADR-0072
    # -- sim and powell. So the box rules below hold for the whole set.
    _BOX_OPTIMIZERS = ['cmaes', 'sim', 'powell']

    @staticmethod
    def _kw_checker(var_tuples):
        c = object.__new__(config.Configuration)
        c.config = dict(var_tuples)
        return c

    @pytest.mark.parametrize('fit_type', _BOX_OPTIMIZERS)
    def test_kw_combo_box_optimizer_accepts_bounded_priors(self, fit_type):
        """A box optimizer (start_from_box) accepts a bounded-prior box -> no raise."""
        c = self._kw_checker({('uniform_var', 'p1'): [-10., 10.],
                              ('loguniform_var', 'p2'): [0.1, 100.]})
        c._check_variable_keyword_combination(fit_type)  # must not raise

    @pytest.mark.parametrize('fit_type', _BOX_OPTIMIZERS)
    def test_kw_combo_box_optimizer_accepts_point_start(self, fit_type):
        """A box optimizer still accepts a single var/logvar start point -> no raise."""
        c = self._kw_checker({('var', 'p1'): [1., 0.5], ('logvar', 'p2'): [3.]})
        c._check_variable_keyword_combination(fit_type)  # must not raise

    @pytest.mark.parametrize('fit_type', _BOX_OPTIMIZERS)
    def test_kw_combo_box_optimizer_rejects_unbounded_prior(self, fit_type):
        """A box search needs a bounded box: normal_var (unbounded) is rejected."""
        c = self._kw_checker({('normal_var', 'p1'): [0., 1.]})
        with pytest.raises(printing.PybnfError):
            c._check_variable_keyword_combination(fit_type)

    @pytest.mark.parametrize('fit_type', _BOX_OPTIMIZERS)
    def test_kw_combo_box_optimizer_rejects_mixed_point_and_box(self, fit_type):
        """A var point mixed with a uniform_var box is ambiguous -> rejected."""
        c = self._kw_checker({('var', 'p1'): [1.], ('uniform_var', 'p2'): [-10., 10.]})
        with pytest.raises(printing.PybnfError):
            c._check_variable_keyword_combination(fit_type)

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

    # --- new-era (edition >= 2) bind-by-id typo check (ADR-0034) ---
    # Tricky.bngl's *parameter ids* are f, NA, ..., koff, kase, pase, ... (its
    # model_param_names); the __FREE tokens live only in RHS expressions. Under
    # edition >= 2 a free parameter binds to a parameter id directly -- no marker.

    @staticmethod
    def _modern_corr(var_names, obj=None, model_file='bngl_files/Tricky.bngl'):
        """A Configuration stub exercising _check_variable_correspondence under a
        modern edition: a real BNGL model (so model_param_names is populated), the
        given free parameters as bare ids, and an objective supplying the nuisance
        set (default: none)."""
        c = object.__new__(config.Configuration)
        c.config = {'edition': 2, 'fit_type': 'de'}
        c.models = {'m': pset.BNGLModel(model_file, suppress_free_param_error=True)}
        c.variables = [pset.FreeParameter(n, 'uniform_var', 0., 1.) for n in var_names]
        c.obj = obj if obj is not None else objective.SumOfSquaresObjective()
        return c

    def test_modern_free_param_binds_to_model_id(self):
        """A free parameter matching a parameter id binds by id -> no raise (the bare
        ids koff/kase, never koff__FREE)."""
        self._modern_corr(['koff', 'kase'])._check_variable_correspondence()  # no raise

    @raises(printing.PybnfError)
    def test_modern_orphan_free_param_raises(self):
        """A free parameter matching no parameter id and referenced by no objective /
        noise surface is almost certainly a typo -> error."""
        self._modern_corr(['bogus_typo'])._check_variable_correspondence()

    @raises(printing.PybnfError)
    def test_modern_legacy_free_marker_name_is_now_a_typo(self):
        """The legacy ``koff__FREE`` spelling matches no parameter id under the new era
        (the id is ``koff``), and is not a noise nuisance -> it is reported as a typo.
        This is the bind-by-id contract: declare the parameter id, not the marker."""
        self._modern_corr(['koff__FREE'])._check_variable_correspondence()

    def test_modern_nuisance_noise_param_is_allowed(self):
        """A free parameter the model never sees but the objective estimates (a free
        sigma) is an intended nuisance, bound to no id -> no raise."""
        obj = objective.ChiSquareObjective_Dynamic()  # requires sigma__FREE
        assert 'sigma__FREE' in obj.required_free_noise_params()
        self._modern_corr(['koff', 'sigma__FREE'], obj)._check_variable_correspondence()

    def test_modern_skips_when_a_model_is_param_agnostic(self, tmp_path):
        """A param-agnostic model (AnalyticalModel) takes its parameters from the conf,
        so nothing is provably a typo: the whole check is skipped, mirroring legacy --
        even a bogus free parameter does not raise."""
        from pybnf.analytical_model import AnalyticalModel
        target = tmp_path / 'gauss.target'
        target.write_text('{"type": "gaussian", "mean": [0.0], "variance": [1.0]}')
        c = self._modern_corr(['bogus_typo'])
        c.models['analytic'] = AnalyticalModel(str(target))
        c._check_variable_correspondence()  # no raise

    def test_modern_no_free_marker_model_loads_and_binds_end_to_end(self, tmp_path):
        """End-to-end: a full edition=2 Configuration over a marker-free BNGL model
        loads (the no-__FREE guard is suppressed) and binds its bare-id free
        parameters (k, S0) by id through the modern correspondence check."""
        import os
        import shutil
        from pybnf.parse import load_config
        shutil.copy('bngl_files/e2e_ode_decay.bngl', tmp_path / 'decay.bngl')
        (tmp_path / 'tc.exp').write_text('#\ttime\tStot\n0\t100\n5\t22\n10\t5\n')
        (tmp_path / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = sos\n'
            'model = decay.bngl : tc.exp\n'
            'uniform_var = k 0 10\nuniform_var = S0 0 200\n'
            'population_size = 10\nmax_iterations = 10\nwall_time_sim = 0\n')
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            c = load_config('job.conf')
        finally:
            os.chdir(cwd)
        assert c.config['edition'] == 2
        assert c.models['decay'].param_names == ()                  # no __FREE markers
        assert c.models['decay'].model_param_names == ('S0', 'k')   # bound by id
        assert {v.name for v in c.variables} == {'k', 'S0'}

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


class TestParameterRecordConfig:
    """The new-era ``parameter:`` record (ADR-0043) -> FreeParameter, end to end.

    Every part of the line is named: prior family, space (lin/log10), the family's own
    distribution fields, lower/upper bounds (which truncate the prior, ADR-0020/#417), and
    initial_value. Built simulator-free over an AnalyticalModel ``.target`` (the golden-config
    tier) under edition 2, plus direct unit checks of the field->FreeParameter mapping."""

    _GAUSS_TARGET = json.dumps({'type': 'gaussian', 'mean': [0.0], 'variance': [1.0]})
    _TARGET_EXP = '# index\tscore\n0\t0\n'

    def _build_vars(self, tmp_path, monkeypatch, param_lines):
        (tmp_path / 'gaussian.target').write_text(self._GAUSS_TARGET)
        (tmp_path / 'target.exp').write_text(self._TARGET_EXP)
        monkeypatch.chdir(tmp_path)
        conf = ('edition = 2\njob_type = de\nobjective = sos\n'
                'model = gaussian.target : target.exp\n'
                'population_size = 5\nmax_iterations = 5\nwall_time_sim = 0\n'
                + ''.join(line + '\n' for line in param_lines))
        c = config.Configuration(parse.ploop(conf.splitlines(keepends=True)))
        return {v.name: v for v in c.variables}

    def test_truncated_normal_record_matches_direct(self, tmp_path, monkeypatch):
        v = self._build_vars(tmp_path, monkeypatch,
                             ['parameter: t, prior: normal, mean: 0, sd: 1, lower: -5, upper: 5'])['t']
        ref = pset.FreeParameter('t', 'normal_var', 0.0, 1.0, lb=-5.0, ub=5.0)
        assert v.type == 'normal_var' and v.bounded and v.has_bounded_support
        assert (v.lower_bound, v.upper_bound) == (ref.lower_bound, ref.upper_bound)
        for theta in (-5.0, 0.0, 4.9, 10.0):
            assert v.prior_logpdf(theta) == ref.prior_logpdf(theta)

    def test_half_bounded_record_parses_and_builds(self, tmp_path, monkeypatch):
        # The full parse path (ploop grammar -> record builder): an open side is an
        # explicit +-inf token (ADR-0047). Covers the parse.py num-token inf support
        # and the graded floor canonicalization, end to end.
        got = self._build_vars(tmp_path, monkeypatch, [
            'parameter: a, prior: normal, mean: 0, sd: 1, lower: -inf, upper: 5',   # open below
            'parameter: b, prior: normal, mean: 0, sd: 1, lower: -2, upper: inf',   # open above
            'parameter: c, prior: gamma, shape: 2, scale: 3, lower: -inf, upper: 9',  # -inf canon -> 0
            'parameter: d, prior: normal, parameter_scale: log10, mean: 1, sd: 0.5, lower: 0, upper: 100',
        ])
        assert got['a'].bounded and (got['a'].lower_bound, got['a'].upper_bound) == (-np.inf, 5.0)
        assert (got['b'].lower_bound, got['b'].upper_bound) == (-2.0, np.inf)
        # gamma floor is 0: 'lower: -inf' canonicalizes to the floor, a wall at 0.
        assert (got['c'].lower_bound, got['c'].upper_bound) == (0.0, 9.0)
        # log family: lower 0 is open below (log10(0) = -inf in u), wall at 100 above.
        assert got['d'].type == 'lognormal_var' and got['d'].has_bounded_support
        assert (got['d'].lower_bound, got['d'].upper_bound) == (0.0, 100.0)

    def test_record_family_variety(self, tmp_path, monkeypatch):
        got = self._build_vars(tmp_path, monkeypatch, [
            'parameter: a, prior: normal, mean: 0, sd: 1',                       # unbounded
            'parameter: b, prior: uniform, lower: 0, upper: 10',                 # uniform
            'parameter: c, prior: normal, parameter_scale: log10, mean: 1, sd: 0.5, lower: 0.1, upper: 100',
            'parameter: e, prior: exponential, scale: 0.5',                      # one-param family
        ])
        a = got['a']
        assert a.type == 'normal_var' and not a.bounded
        assert a.lower_bound == -np.inf and a.upper_bound == np.inf
        b = got['b']
        assert b.type == 'uniform_var' and (b.lower_bound, b.upper_bound) == (0.0, 10.0)
        c = got['c']
        assert c.type == 'lognormal_var' and c.log_space
        assert (c.lower_bound, c.upper_bound) == (0.1, 100.0) and c.has_bounded_support
        e = got['e']
        assert e.type == 'exponential_var' and e.p1 == 0.5

    def test_no_prior_bounds_defaults_to_uniform(self, tmp_path, monkeypatch):
        # prior omitted but bounds given -> uniform over the bounds (PEtab's default).
        v = self._build_vars(tmp_path, monkeypatch,
                             ['parameter: u, lower: 0, upper: 10'])['u']
        assert v.type == 'uniform_var' and (v.lower_bound, v.upper_bound) == (0.0, 10.0)

    def test_batch_univariate_families(self, tmp_path, monkeypatch):
        # The #438 item-1 batch families build through the full record path (ADR-0057): the
        # one-parameter half-* scale priors, the bounded beta, and the two-parameter
        # positive/location-scale families. Each is oracled against its scipy density at a
        # probe point in its own support.
        from scipy import stats
        got = self._build_vars(tmp_path, monkeypatch, [
            'parameter: hn, prior: half_normal, scale: 2',
            'parameter: hc, prior: half_cauchy, scale: 1.5',
            'parameter: bb, prior: beta, alpha: 2, beta: 5',
            'parameter: ig, prior: inv_gamma, shape: 3, scale: 2',
            'parameter: wb, prior: weibull, shape: 1.5, scale: 2',
            'parameter: gu, prior: gumbel, location: 0, scale: 1',
            'parameter: lo, prior: logistic, location: 0, scale: 1',
        ])
        cases = {
            'hn': (stats.halfnorm(scale=2), 1.0, 'half_normal_var'),
            'hc': (stats.halfcauchy(scale=1.5), 1.0, 'half_cauchy_var'),
            'bb': (stats.beta(a=2, b=5), 0.3, 'beta_var'),
            'ig': (stats.invgamma(a=3, scale=2), 1.0, 'inv_gamma_var'),
            'wb': (stats.weibull_min(c=1.5, scale=2), 1.0, 'weibull_var'),
            'gu': (stats.gumbel_r(loc=0, scale=1), 0.5, 'gumbel_var'),
            'lo': (stats.logistic(loc=0, scale=1), 0.5, 'logistic_var'),
        }
        for name, (ref, probe, keyword) in cases.items():
            v = got[name]
            assert v.type == keyword, name
            assert v.prior_logpdf(probe) == pytest.approx(float(ref.logpdf(probe))), name

    def test_student_t_record_three_params(self, tmp_path, monkeypatch):
        # The three-parameter family is reachable ONLY through the record (ADR-0057); df,
        # location, and scale land in p1/p2/p3, oracled against scipy.stats.t. A log-scaled,
        # truncated variant exercises the third value travelling the whole carrier.
        from scipy import stats
        got = self._build_vars(tmp_path, monkeypatch, [
            'parameter: st, prior: student_t, df: 4, location: 1, scale: 2',
            'parameter: lt, prior: student_t, parameter_scale: log10, df: 3, location: 1, '
            'scale: 0.5, lower: 0.1, upper: 100',
        ])
        st = got['st']
        assert st.type == 'student_t_var'
        assert (st.p1, st.p2, st.p3) == (4.0, 1.0, 2.0)
        ref = stats.t(df=4, loc=1, scale=2)
        for theta in (-3.0, 1.0, 5.0):
            assert st.prior_logpdf(theta) == pytest.approx(float(ref.logpdf(theta)))
        lt = got['lt']
        assert lt.type == 'logstudent_t_var' and lt.p3 == 0.5
        assert (lt.lower_bound, lt.upper_bound) == (0.1, 100.0) and lt.has_bounded_support

    def test_student_t_missing_field_errors(self, tmp_path, monkeypatch):
        # A three-field family needs all three; omitting one is a clear error, not a silent
        # default (the record names every part).
        with pytest.raises(printing.PybnfError, match="needs field 'scale'"):
            self._build_vars(tmp_path, monkeypatch,
                             ['parameter: st, prior: student_t, df: 4, location: 1'])

    def test_initial_value_routing(self, tmp_path, monkeypatch):
        # A prior parameter carries initial_value as its theta-space .value (read by the
        # population seed); a no-prior start point carries it as the first slot (Simplex reads p1).
        got = self._build_vars(tmp_path, monkeypatch, [
            'parameter: p, prior: normal, mean: 0, sd: 1, lower: -5, upper: 5, initial_value: 2',
            'parameter: s, initial_value: 3',
        ])
        assert got['p'].value == 2.0
        assert got['s'].type == 'var' and got['s'].p1 == 3.0

    def test_parameter_scale_linear_log10_ln(self, tmp_path, monkeypatch):
        # All three sampling scales build; each prior is over the parameter in its OWN base
        # (ADR-0043/0022): a natural-log normal is Normal over ln(theta), a log10 normal over
        # log10(theta), and they differ -- proof the base is honored, not silently coerced.
        from scipy import stats
        got = self._build_vars(tmp_path, monkeypatch, [
            'parameter: a, prior: normal, mean: 1, sd: 0.5',                       # linear (default)
            'parameter: b, prior: normal, parameter_scale: log10, mean: 1, sd: 0.5',
            'parameter: c, prior: normal, parameter_scale: ln, mean: 1, sd: 0.5',
            'parameter: u, prior: uniform, parameter_scale: ln, lower: 0.1, upper: 100',
            'parameter: s, parameter_scale: ln, initial_value: 2',                 # no-prior ln start point
        ])
        ref = stats.norm(1, 0.5)
        assert got['a'].scale_name == 'linear' and not got['a'].log_space
        assert got['b'].type == 'lognormal_var' and got['b'].scale_name == 'log10'
        assert got['c'].type == 'lnnormal_var' and got['c'].scale_name == 'ln'
        theta = 3.0
        assert got['b'].prior_logpdf(theta) == pytest.approx(ref.logpdf(np.log10(theta)))
        assert got['c'].prior_logpdf(theta) == pytest.approx(ref.logpdf(np.log(theta)))
        assert got['c'].prior_logpdf(theta) != got['b'].prior_logpdf(theta)   # base is honored
        assert got['u'].type == 'lnuniform_var' and (got['u'].lower_bound, got['u'].upper_bound) == (0.1, 100.0)
        # the ln start point's initial_value is the REAL value (theta): it round-trips out of
        # ln sampling space to exactly 2, not exp(2) (initial_value is theta on every scale).
        s = got['s']
        assert s.type == 'lnvar' and s.from_sampling_space(s.p1) == pytest.approx(2.0)

    def test_record_is_edition_gated(self, tmp_path, monkeypatch):
        # A parameter: record in an otherwise-legacy job is rejected -- it needs edition 2.
        (tmp_path / 'gaussian.target').write_text(self._GAUSS_TARGET)
        (tmp_path / 'target.exp').write_text(self._TARGET_EXP)
        monkeypatch.chdir(tmp_path)
        conf = ('objfunc = direct_pass\nfit_type = de\n'
                'model = gaussian.target : target.exp\n'
                'parameter: a, prior: normal, mean: 0, sd: 1\n'
                'population_size = 5\nmax_iterations = 5\nwall_time_sim = 0\n')
        with pytest.raises(printing.PybnfError, match='edition 2'):
            config.Configuration(parse.ploop(conf.splitlines(keepends=True)))

    @pytest.mark.parametrize('fields,match', [
        ({'prior': 'gaussian', 'mean': '0', 'sd': '1'}, 'unknown prior family'),
        ({'prior': 'normal', 'mean': '0'}, "needs field 'sd'"),
        ({'prior': 'normal', 'mean': '0', 'sd': '1', 'oops': '2'}, 'unknown field'),
        ({'prior': 'normal', 'mean': '0', 'sd': '1', 'lower': '-5'}, "come as a pair"),
        ({'parameter_scale': 'log10'}, 'nothing to fit'),
        ({'prior': 'normal', 'mean': 'abc', 'sd': '1'}, 'must be a number'),
        ({'prior': 'normal', 'parameter_scale': 'bogus', 'mean': '0', 'sd': '1'}, "'linear', 'log10', or"),
        ({'prior': 'normal', 'parameter_scale': 'log', 'mean': '0', 'sd': '1'}, 'ambiguous'),
        # A finite lower wall below a log family's support floor (0) is a wall in the
        # zero-density region -> error (ADR-0047). lower: 0 / -inf is now *allowed*
        # (open below); only a finite sub-floor value raises.
        ({'prior': 'normal', 'parameter_scale': 'ln', 'mean': '1', 'sd': '0.5',
          'lower': '-5', 'upper': '100'}, "support floor"),
        ({'parameter_scale': 'log10', 'initial_value': '-5'}, 'initial_value > 0'),
    ])
    def test_record_field_errors(self, fields, match):
        c = object.__new__(config.Configuration)
        with pytest.raises(printing.PybnfError, match=match):
            c._free_parameter_from_record('k', fields, 'prior')


class TestNewEraNormalization:
    """Per-observable normalization on the new-era surface (#444, ADR-0053).

    Normalization is a per-observable *prediction* transform -- a sibling of the
    per-observable ``noise_model`` / ``cumulative`` surface -- so the new era keys it by
    observable (``normalization <obs> = <type>``) or by experiment+observable
    (``normalization <exp>.<obs> = <type>``), never by filename. The rules form a total
    specificity order, most-specific-wins: ``<exp>.<obs>`` > ``<obs>`` > the whole-fit
    default ``normalization = <type>``. ``par1.exp`` columns are: time, x, y, x_SD, y_SD.

    These exercise ``_postprocess_normalization`` / ``_resolve_normalization_grid`` directly
    through the lightweight ``object.__new__`` harness (no model build, no backend), the same
    pattern the legacy ``_postprocess_normalization`` tests use.
    """

    @staticmethod
    def _resolve(norm, edition_val=2, experiments=('egf_high', 'egf_low')):
        c = object.__new__(config.Configuration)
        c.exp_data = {'parabola': {e: data.Data(file_name='bngl_files/par1.exp')
                                   for e in experiments}}
        c._experiment_data_keys = {e: ('parabola', e) for e in experiments}
        cfg = {'models': {'bngl_files/parabola.bngl'}, 'bngl_files/parabola.bngl': [],
               'exp_data': set(), 'edition': edition_val, 'normalization': None}
        cfg.update(norm)
        c.config = cfg
        c._postprocess_normalization()
        return c.config['normalization']

    def test_per_observable_applies_to_every_experiment(self):
        out = self._resolve({('normalization', 'x'): 'peak'})
        assert out == {'egf_high': [('peak', ['x'])], 'egf_low': [('peak', ['x'])]}

    def test_observable_without_a_rule_is_not_normalized(self):
        # No whole-fit default: y has no rule, so it is left un-normalized (only declared
        # observables are transformed).
        out = self._resolve({('normalization', 'x'): 'peak'})
        for dk in out:
            assert 'y' not in [c for _t, cols in out[dk] for c in cols]

    def test_whole_fit_default_plus_per_observable_override(self):
        out = self._resolve({'normalization': 'init', ('normalization', 'x'): 'peak'})
        assert out == {'egf_high': [('peak', ['x']), ('init', ['y'])],
                       'egf_low': [('peak', ['x']), ('init', ['y'])]}

    def test_qualified_override_is_most_specific(self):
        # Total order: <exp>.<obs>  beats  <obs>  beats  whole-fit default.
        out = self._resolve({'normalization': 'init',
                             ('normalization', 'x'): 'peak',
                             ('normalization', 'egf_high.x'): 'zero'})
        assert out['egf_high'] == [('zero', ['x']), ('init', ['y'])]   # qualified wins for egf_high.x
        assert out['egf_low'] == [('peak', ['x']), ('init', ['y'])]    # per-observable elsewhere

    def test_qualified_matches_conditioned_experiment_by_name(self):
        # A conditioned experiment's data_key is name+condition, but the user authors by
        # experiment NAME; the override resolves through the experiment-name map (ADR-0053).
        c = object.__new__(config.Configuration)
        c.exp_data = {'parabola': {'egf_highdimer': data.Data(file_name='bngl_files/par1.exp')}}
        c._experiment_data_keys = {'egf_high': ('parabola', 'egf_highdimer')}
        c.config = {'models': {'bngl_files/parabola.bngl'}, 'bngl_files/parabola.bngl': [],
                    'exp_data': set(), 'edition': 2, 'normalization': None,
                    ('normalization', 'egf_high.x'): 'zero'}
        c._postprocess_normalization()
        assert c.config['normalization'] == {'egf_highdimer': [('zero', ['x'])]}

    def test_unknown_observable_raises(self):
        with pytest.raises(printing.PybnfError, match='unknown observable'):
            self._resolve({('normalization', 'zzz'): 'peak'})

    def test_qualified_unknown_experiment_raises(self):
        with pytest.raises(printing.PybnfError, match='unknown experiment'):
            self._resolve({('normalization', 'nope.x'): 'peak'})

    def test_qualified_unknown_observable_in_experiment_raises(self):
        with pytest.raises(printing.PybnfError, match='unknown observable'):
            self._resolve({('normalization', 'egf_high.zzz'): 'peak'})

    def test_invalid_type_raises(self):
        with pytest.raises(printing.PybnfError, match='Invalid normalization type'):
            self._resolve({('normalization', 'x'): 'bogus'})

    def test_modern_form_requires_edition_2(self):
        with pytest.raises(printing.PybnfError, match='edition'):
            self._resolve({('normalization', 'x'): 'peak'}, edition_val=None)

    def test_legacy_per_file_form_rejected_under_edition_2(self):
        # The pre-#444 failure mode: a legacy filename-keyed dict on the new-era surface
        # crashed with ``KeyError: None`` (the filename stem is not the new-era data key).
        # It is now a clear redirect to the per-observable form, not a crash.
        with pytest.raises(printing.PybnfError, match='legacy form'):
            self._resolve({'normalization': {'bngl_files/par1.exp': 'peak'}})

    # --- ADR-0066 (#479): floor + analytic scale chains ---

    @staticmethod
    def _build_grid(norm, experiments=('egf_high', 'egf_low')):
        """Like ``_resolve`` but returns the whole Configuration harness, so a test can read
        the compiled ``normalization`` dict, the ``analytic_scale`` key, and the (floored)
        experimental data together."""
        c = object.__new__(config.Configuration)
        c.exp_data = {'parabola': {e: data.Data(file_name='bngl_files/par1.exp')
                                   for e in experiments}}
        c._experiment_data_keys = {e: ('parabola', e) for e in experiments}
        c.config = {'models': {'bngl_files/parabola.bngl'}, 'bngl_files/parabola.bngl': [],
                    'exp_data': set(), 'edition': 2, 'normalization': None, **norm}
        c._postprocess_normalization()
        return c

    def test_floor_scale_chain_splits_into_the_two_seams(self):
        # x: floor+scale, y: floor. floor lands in the sim normalization dict (both cols); scale
        # is routed to analytic_scale (x only); the data column is floored in place.
        c = self._build_grid({('normalization', 'x'): [('floor', 0.03), 'scale'],
                              ('normalization', 'y'): [('floor', 0.03)]})
        assert c.config['normalization'] == {
            'egf_high': [(('floor', 0.03), ['x']), (('floor', 0.03), ['y'])],
            'egf_low': [(('floor', 0.03), ['x']), (('floor', 0.03), ['y'])]}
        assert c.config['analytic_scale'] == {'egf_high': frozenset({'x'}),
                                              'egf_low': frozenset({'x'})}

    def test_floor_is_applied_symmetrically_to_the_experimental_data(self):
        d = data.Data(file_name='bngl_files/par1.exp')
        raw_x = d.data[:, d.cols['x']].copy()
        c = self._build_grid({('normalization', 'x'): [('floor', 0.03)]})
        floored = c.exp_data['parabola']['egf_high']
        # x' = x + 0.03*max(x); the data column is floored from its OWN max, once.
        np.testing.assert_allclose(floored.data[:, floored.cols['x']],
                            raw_x + 0.03 * np.nanmax(raw_x))
        # y (no rule) is untouched.
        np.testing.assert_allclose(floored.data[:, floored.cols['y']], d.data[:, d.cols['y']])

    def test_scale_only_leaves_normalization_none(self):
        # A bare ``scale`` is not a Data transform: no sim normalization, only analytic_scale.
        c = self._build_grid({('normalization', 'x'): 'scale'})
        assert c.config['normalization'] is None
        assert c.config['analytic_scale'] == {'egf_high': frozenset({'x'}),
                                              'egf_low': frozenset({'x'})}

    def test_whole_fit_chain_applies_to_every_observable(self):
        c = self._build_grid({'normalization': [('floor', 0.03), 'scale']})
        assert c.config['normalization'] == {
            'egf_high': [(('floor', 0.03), ['x', 'y'])],
            'egf_low': [(('floor', 0.03), ['x', 'y'])]}
        assert c.config['analytic_scale'] == {'egf_high': frozenset({'x', 'y'}),
                                              'egf_low': frozenset({'x', 'y'})}

    def test_invalid_type_in_chain_lists_floor_and_scale(self):
        with pytest.raises(printing.PybnfError, match='floor, scale'):
            self._resolve({('normalization', 'x'): [('bogus', 1.0)]})

    def test_scale_refused_for_a_column_joint_objective(self):
        # `scale` rides the per-point prediction seam, which a column-joint kl / wasserstein
        # lacks -- refused at attach time (mirroring `cumulative`).
        c = object.__new__(config.Configuration)
        c.obj = objective.KLLikelihood()
        c.config = {'analytic_scale': {'e': frozenset({'x'})}}
        with pytest.raises(printing.PybnfError, match='column-joint'):
            c._attach_analytic_scale()


class TestNewEraPreprocessingKeysRideThrough:
    """#444 item 1: the other three preprocessing keys -- ``smoothing`` /
    ``ind_var_rounding`` / ``constraint_scale`` -- are global scalars, NOT filename-coupled,
    so (unlike the per-file normalization dict) they ride the new-era surface unchanged.
    Verified by building a real edition-2 ``Configuration`` (the per_observable_noise_v2
    example: two observables x, y) and reading the keys back, including the one that reaches
    a consumer (``ind_var_rounding`` -> the objective). Backend-free (no bngsim / BNG2)."""

    @staticmethod
    def _build(extra):
        home = os.getcwd()
        os.chdir('examples/per_observable_noise')
        try:
            base = open('per_observable_noise_v2.conf').read()
            return config.Configuration(
                parse.ploop((base + extra).splitlines(keepends=True)))
        finally:
            os.chdir(home)

    def test_ind_var_rounding_rides_through_and_reaches_the_objective(self):
        c = self._build('\nind_var_rounding = 1\n')
        assert c.config['ind_var_rounding'] == 1
        assert c.obj.rounding == 1

    def test_constraint_scale_rides_through(self):
        c = self._build('\nconstraint_scale = 2.5\n')
        assert c.config['constraint_scale'] == 2.5

    def test_smoothing_rides_through(self):
        c = self._build('\nsmoothing = 3\n')
        assert c.config['smoothing'] == 3

    def test_per_observable_normalization_resolves_on_a_real_build(self):
        # End-to-end: the new-era per-observable + qualified forms resolve against the real
        # experiment's columns during a full Configuration build.
        c = self._build('\nnormalization x = peak\nnormalization par1.y = init\n')
        assert c.config['normalization'] == {'par1': [('peak', ['x']), ('init', ['y'])]}

    def test_floor_scale_chain_wires_the_objective_and_floors_the_data(self):
        # End-to-end (ADR-0066, #479): a floor+scale chain on a real edition-2 build routes the
        # floor to the sim normalization dict + the experimental data (symmetric) and the scale to
        # the objective, keyed by data_key.
        c = self._build('\nnormalization x = floor 0.03, scale\n')
        assert c.config['normalization'] == {'par1': [(('floor', 0.03), ['x'])]}
        assert c.obj._analytic_scale == {'par1': frozenset({'x'})}
        # x is scored with normal (linear) noise, so its scale profiles in linear space.
        assert c.obj._scale_mode('x') == 'linear'
        # The experimental x column was floored in place (a 'floor' record proves it).
        floored = next(iter(c.exp_data.values()))['par1']
        assert floored.normalization['x'].method == 'floor'
