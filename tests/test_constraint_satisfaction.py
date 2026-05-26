"""Tests for the constraint satisfaction tracking feature (#324)."""
from .context import constraint, algorithms, config, data
import os
import shutil
import tempfile
from unittest.mock import patch


# Patch targets: skip BNG simulator detection and network generation
_no_bng = patch.object(config.Configuration, '_load_simulators')
_no_init = patch.object(algorithms.Algorithm, '_initialize_models', return_value=[])

_BASE_CFG = {
    'population_size': 2, 'max_iterations': 10, 'step_size': 0.2,
    'output_hist_every': 5, 'sample_every': 1, 'burn_in': 1,
    'credible_intervals': [95], 'num_bins': 10,
    ('uniform_var', 'v1__FREE'): [0, 10],
    ('uniform_var', 'v2__FREE'): [0, 10],
    ('uniform_var', 'v3__FREE'): [0, 10],
    'models': {'tests/bngl_files/parabola.bngl'},
    'exp_data': {'tests/bngl_files/par1.exp'},
    'tests/bngl_files/parabola.bngl': ['tests/bngl_files/par1.exp'],
    'fit_type': 'mh',
}


class TestSourceLine:
    """Constraint.source_line is populated during parsing."""

    def test_source_line_stored(self):
        cs = constraint.ConstraintSet('model', 'suffix')
        cs.load_constraint_file('tests/bngl_files/con_test.prop')
        for c in cs.constraints:
            assert c.source_line != '', 'source_line should be populated'

    def test_source_line_default_empty(self):
        c = constraint.AtConstraint('A', '<', 'B', 'model', 'suffix', 1.0,
                                    atvar=None, atval=5.0)
        assert c.source_line == ''


class TestEvaluateConstraints:
    """BayesianAlgorithm.evaluate_constraints caches pass/fail per chain."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.tmpdir, 'Results'))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)

    def test_no_constraints(self):
        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            cfg.constraints = set()
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)
        assert ba.all_constraints == []
        ba.evaluate_constraints({}, 0)
        assert ba.current_constraint_satisfied[0] is None

    def test_constraint_satisfied(self):
        c = constraint.AlwaysConstraint('A', '<', 10.0, 'parabola', 'par1', 1.0)
        c.source_line = 'A < 10 always weight 1'
        cs = constraint.ConstraintSet('parabola', 'par1')
        cs.constraints = [c]

        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            cfg.constraints = {cs}
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)

        d = data.Data()
        d.data = d._read_file_lines(['# time A\n', '1 5\n', '2 7\n'], r'\s+')
        ba.evaluate_constraints({'parabola': {'par1': d}}, 0)
        assert ba.current_constraint_satisfied[0] == [1]

    def test_constraint_violated(self):
        c = constraint.AlwaysConstraint('A', '<', 5.0, 'parabola', 'par1', 1.0)
        c.source_line = 'A < 5 always weight 1'
        cs = constraint.ConstraintSet('parabola', 'par1')
        cs.constraints = [c]

        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            cfg.constraints = {cs}
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)

        d = data.Data()
        d.data = d._read_file_lines(['# time A\n', '1 3\n', '2 8\n'], r'\s+')
        ba.evaluate_constraints({'parabola': {'par1': d}}, 0)
        assert ba.current_constraint_satisfied[0] == [0]

    def test_multiple_constraints(self):
        c1 = constraint.AlwaysConstraint('A', '<', 10.0, 'parabola', 'par1', 1.0)
        c1.source_line = 'A < 10 always'
        c2 = constraint.AlwaysConstraint('A', '>', 6.0, 'parabola', 'par1', 1.0)
        c2.source_line = 'A > 6 always'
        cs = constraint.ConstraintSet('parabola', 'par1')
        cs.constraints = [c1, c2]

        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            cfg.constraints = {cs}
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)

        d = data.Data()
        d.data = d._read_file_lines(['# time A\n', '1 3\n', '2 7\n'], r'\s+')
        ba.evaluate_constraints({'parabola': {'par1': d}}, 0)
        # c1 satisfied (always < 10), c2 violated (A=3 < 6)
        assert ba.current_constraint_satisfied[0] == [1, 0]


class TestSamplePsetConstraints:
    """sample_pset writes constraint results to file."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.tmpdir, 'Results', 'Histograms'))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)
        nocstr = cls.tmpdir + '_nocstr'
        if os.path.exists(nocstr):
            shutil.rmtree(nocstr)

    def test_sample_pset_writes_constraints(self):
        c = constraint.AlwaysConstraint('A', '<', 10.0, 'parabola', 'par1', 1.0)
        c.source_line = 'A < 10 always'
        cs = constraint.ConstraintSet('parabola', 'par1')
        cs.constraints = [c]

        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            cfg.constraints = {cs}
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)
            start_psets = ba.start_run()

        ba.current_constraint_satisfied[0] = [1]
        ba.current_constraint_satisfied[1] = [0]

        ba.sample_pset(start_psets[0], -10.0, 0)
        ba.sample_pset(start_psets[1], -12.0, 1)

        with open(ba.constraint_samples_file) as f:
            lines = f.readlines()
        assert lines[0].startswith('#')  # header
        assert lines[1].strip() == '1'  # chain 0: satisfied
        assert lines[2].strip() == '0'  # chain 1: violated

    def test_sample_pset_no_constraints(self):
        nocstr_dir = self.tmpdir + '_nocstr'
        os.makedirs(nocstr_dir + '/Results/Histograms', exist_ok=True)
        cfg_dict = dict(_BASE_CFG, output_dir=nocstr_dir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)
            start_psets = ba.start_run()

        ba.sample_pset(start_psets[0], -10.0)  # No chain_index
        assert not os.path.exists(ba.constraint_samples_file)


class TestReportConstraintSatisfaction:
    """report_constraint_satisfaction writes a summary file."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.tmpdir, 'Results', 'Histograms'))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)
        nocstr = cls.tmpdir + '_nocstr'
        if os.path.exists(nocstr):
            shutil.rmtree(nocstr)

    def test_report_output(self):
        c1 = constraint.AlwaysConstraint('A', '<', 10.0, 'parabola', 'par1', 1.0)
        c1.source_line = 'A < 10 always'
        c2 = constraint.AlwaysConstraint('A', '>', 5.0, 'parabola', 'par1', 1.0)
        c2.source_line = 'A > 5 always'
        cs = constraint.ConstraintSet('parabola', 'par1')
        cs.constraints = [c1, c2]

        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            cfg.constraints = {cs}
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)
            ba.start_run()

        # Write synthetic constraint samples: 4 samples, 2 constraints
        with open(ba.constraint_samples_file, 'w') as f:
            f.write('# A < 10 always\tA > 5 always\n')
            f.write('1\t0\n')
            f.write('1\t1\n')
            f.write('1\t0\n')
            f.write('0\t0\n')

        ba.report_constraint_satisfaction('_test')

        report_path = os.path.join(self.tmpdir, 'Results', 'constraint_satisfaction_test.txt')
        assert os.path.exists(report_path)

        with open(report_path) as f:
            lines = f.readlines()
        assert lines[0].startswith('#')  # header
        # c1: 3 satisfied out of 4 = 75.0%
        assert '75.0%' in lines[1]
        assert '3' in lines[1]
        assert '4' in lines[1]
        # c2: 1 satisfied out of 4 = 25.0%
        assert '25.0%' in lines[2]
        assert '1' in lines[2]

    def test_report_no_constraints(self):
        cfg_dict = dict(_BASE_CFG, output_dir=self.tmpdir + '_nocstr')
        os.makedirs(cfg_dict['output_dir'] + '/Results/Histograms', exist_ok=True)
        with _no_bng, _no_init:
            cfg = config.Configuration(cfg_dict)
            ba = algorithms.BasicBayesMCMCAlgorithm(cfg)
            ba.start_run()

        ba.report_constraint_satisfaction('_test')
        report_path = os.path.join(cfg_dict['output_dir'], 'Results', 'constraint_satisfaction_test.txt')
        assert not os.path.exists(report_path)
