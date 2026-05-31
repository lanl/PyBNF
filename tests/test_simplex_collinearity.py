"""Tests for simplex collinearity fix (#207)."""
from .context import algorithms, config, pset
import os
import shutil
import tempfile
from unittest.mock import patch


_no_bng = patch.object(config.Configuration, '_load_simulators')
_no_init = patch.object(algorithms.Algorithm, '_initialize_models', return_value=[])


class TestParallelCountCap:
    """parallel_count is capped at n_vars - 1 to prevent collinearity."""

    def test_2d_cap(self):
        """In 2D, parallel_count should be 1 even with population_size > 1."""
        with _no_bng, _no_init:
            cfg = config.Configuration({
                'population_size': 5, 'max_iterations': 10, 'fit_type': 'sim',
                ('var', 'v1__FREE'): [2.], ('var', 'v2__FREE'): [3.],
                'models': {'tests/bngl_files/parabola_2d.bngl'},
                'exp_data': {'tests/bngl_files/par1.exp'},
                'tests/bngl_files/parabola_2d.bngl': ['tests/bngl_files/par1.exp'],
            })
            sim = algorithms.SimplexAlgorithm(cfg)
        # 2 variables, so parallel_count = min(5, 2-1) = 1
        assert sim.parallel_count == 1

    def test_3d_cap(self):
        """In 3D, parallel_count should be at most 2."""
        with _no_bng, _no_init:
            cfg = config.Configuration({
                'population_size': 10, 'max_iterations': 10, 'fit_type': 'sim',
                ('var', 'v1__FREE'): [2.], ('var', 'v2__FREE'): [3.], ('var', 'v3__FREE'): [4.],
                'models': {'tests/bngl_files/parabola.bngl'},
                'exp_data': {'tests/bngl_files/par1.exp'},
                'tests/bngl_files/parabola.bngl': ['tests/bngl_files/par1.exp'],
            })
            sim = algorithms.SimplexAlgorithm(cfg)
        # 3 variables, so parallel_count = min(10, 3-1) = 2
        assert sim.parallel_count == 2

    def test_1d_minimum(self):
        """In 1D, parallel_count should be 1 (the minimum)."""
        with _no_bng, _no_init:
            cfg = config.Configuration({
                'population_size': 5, 'max_iterations': 10, 'fit_type': 'sim',
                ('var', 'v1__FREE'): [2.],
                'models': {'tests/bngl_files/parabola_1d.bngl'},
                'exp_data': {'tests/bngl_files/par1.exp'},
                'tests/bngl_files/parabola_1d.bngl': ['tests/bngl_files/par1.exp'],
            })
            sim = algorithms.SimplexAlgorithm(cfg)
        # 1 variable, so parallel_count = min(5, max(0, 1)) = 1
        assert sim.parallel_count == 1

    def test_pop_size_1_unchanged(self):
        """population_size=1 should still give parallel_count=1."""
        with _no_bng, _no_init:
            cfg = config.Configuration({
                'population_size': 1, 'max_iterations': 10, 'fit_type': 'sim',
                ('var', 'v1__FREE'): [2.], ('var', 'v2__FREE'): [3.], ('var', 'v3__FREE'): [4.],
                'models': {'tests/bngl_files/parabola.bngl'},
                'exp_data': {'tests/bngl_files/par1.exp'},
                'tests/bngl_files/parabola.bngl': ['tests/bngl_files/par1.exp'],
            })
            sim = algorithms.SimplexAlgorithm(cfg)
        assert sim.parallel_count == 1


class TestDegeneracyDetection:
    """_check_degeneracy detects and perturbs degenerate simplices."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.tmpdir, 'Results'))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)

    def _make_simplex(self):
        """Create a SimplexAlgorithm with 2D config."""
        with _no_bng, _no_init:
            cfg = config.Configuration({
                'population_size': 1, 'max_iterations': 10, 'fit_type': 'sim',
                'output_dir': self.tmpdir,
                ('var', 'v1__FREE'): [50.],
                ('var', 'v2__FREE'): [50.],
                'models': {'tests/bngl_files/parabola_2d.bngl'},
                'exp_data': {'tests/bngl_files/par1.exp'},
                'tests/bngl_files/parabola_2d.bngl': ['tests/bngl_files/par1.exp'],
            })
            sim = algorithms.SimplexAlgorithm(cfg)
        return sim

    def test_healthy_simplex_unchanged(self):
        """A non-degenerate simplex should not be perturbed."""
        sim = self._make_simplex()
        v1 = sim.variables[0]
        v2 = sim.variables[1]
        # Triangle with good volume
        sim.simplex = [
            (1.0, pset.PSet([v1.set_value(10), v2.set_value(10)])),
            (2.0, pset.PSet([v1.set_value(20), v2.set_value(10)])),
            (3.0, pset.PSet([v1.set_value(10), v2.set_value(20)])),
        ]
        original_values = [(s[1][v1.name], s[1][v2.name]) for s in sim.simplex]
        sim._check_degeneracy()
        new_values = [(s[1][v1.name], s[1][v2.name]) for s in sim.simplex]
        assert original_values == new_values

    def test_collinear_simplex_perturbed(self):
        """A collinear simplex should be detected and perturbed."""
        sim = self._make_simplex()
        v1 = sim.variables[0]
        v2 = sim.variables[1]
        # Three collinear points along y=x
        sim.simplex = [
            (1.0, pset.PSet([v1.set_value(10), v2.set_value(10)])),
            (2.0, pset.PSet([v1.set_value(20), v2.set_value(20)])),
            (3.0, pset.PSet([v1.set_value(30), v2.set_value(30)])),
        ]
        sim._check_degeneracy()
        # After perturbation, points should no longer be collinear
        # Vertex 0 (best) should be unchanged
        assert sim.simplex[0][1][v1.name] == 10
        assert sim.simplex[0][1][v2.name] == 10
        # At least one other vertex should have been perturbed
        v1_1 = sim.simplex[1][1][v1.name]
        v2_1 = sim.simplex[1][1][v2.name]
        v1_2 = sim.simplex[2][1][v1.name]
        v2_2 = sim.simplex[2][1][v2.name]
        # Check they're no longer perfectly on the line y=x
        not_collinear = (v1_1 != v2_1) or (v1_2 != v2_2)
        assert not_collinear, "Perturbed simplex should not be collinear"

    def test_nearly_degenerate_simplex_perturbed(self):
        """A nearly degenerate simplex (tiny volume) should be perturbed."""
        sim = self._make_simplex()
        v1 = sim.variables[0]
        v2 = sim.variables[1]
        # Almost collinear: tiny offset from the line
        sim.simplex = [
            (1.0, pset.PSet([v1.set_value(10), v2.set_value(10)])),
            (2.0, pset.PSet([v1.set_value(20), v2.set_value(20)])),
            (3.0, pset.PSet([v1.set_value(30), v2.set_value(30.0 + 1e-15)])),
        ]
        sim._check_degeneracy()
        # Best vertex unchanged
        assert sim.simplex[0][1][v1.name] == 10
        # Other vertices should be perturbed away from near-collinearity
        v2_2 = sim.simplex[2][1][v2.name]
        assert abs(v2_2 - 30.0) > 1e-15, "Near-degenerate simplex should be perturbed"
