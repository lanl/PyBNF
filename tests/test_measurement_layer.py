"""Unit tests for the measurement-model observation layer (#407, ADR-0036).

The weakest-but-foundational oracle: the layer materializes an ``observableFormula`` column
post-simulation, over a **crafted trajectory** (hand-built columns), and the result must equal
a ``numpy`` hand-computation. The layer is backend- and language-agnostic by construction --
it only sees a :class:`~pybnf.data.Data` and a ``{name: value}`` PSet map -- so the same
formula over a *BNGL-style* trace (observable columns) and an *SBML-style* trace (species
columns) yields the same numbers, with no model, no simulator, and no model-file edit.

``petab``/``sympy`` is the optional ``pybnf[petab]`` extra; the compile-path tests
``importorskip('petab')``. The empty-layer no-op and the column-collision guard need no
compile, so they run dependency-free (the layer's structural contract).
"""

import pickle
from collections import namedtuple

import numpy as np
import pytest

from pybnf.data import Data
from pybnf.measurement import MeasurementLayer, MeasurementModel
from pybnf.printing import PybnfError

# A minimal stand-in for a FreeParameter as evaluate_multiple iterates the PSet (it reads
# only ``.name`` / ``.value``); keeps these tests off the pset/config machinery.
_FP = namedtuple('FP', 'name value')


def _trace(headers, rows):
    """A crafted trajectory Data from an ordered header list + row tuples."""
    return Data.from_columns(np.array(rows, dtype=float), headers)


# ---------------------------------------------------------------------------
# MeasurementModel.materialize -- the numpy hand-computation oracle
# ---------------------------------------------------------------------------

class TestMaterialize:

    def test_sbml_style_quotient_of_sums_over_species(self):
        # SBML-style trace: species columns only (what RoadRunner/bngsim emit).
        pytest.importorskip('petab')
        data = _trace(['time', 'S1', 'S2'],
                      [(0., 10., 4.), (1., 8., 5.), (2., 6., 6.)])
        mm = MeasurementModel('obs', '(100*S1 + 200*S2) / (S1 + S2)', {'S1', 'S2'})
        got = mm.materialize(data, pset_values={})
        s1, s2 = data['S1'], data['S2']
        np.testing.assert_allclose(got, (100 * s1 + 200 * s2) / (s1 + s2))

    def test_bngl_style_over_observables_and_a_free_parameter(self):
        # BNGL-style trace: observable columns + a free parameter from the PSet.
        pytest.importorskip('petab')
        data = _trace(['time', 'obsA', 'obsB'],
                      [(0., 2., 1.), (1., 3., 2.), (2., 5., 1.)])
        mm = MeasurementModel('pRel', 'kA*obsA + obsB', {'kA', 'obsA', 'obsB'})
        got = mm.materialize(data, pset_values={'kA': 2.5})
        np.testing.assert_allclose(got, 2.5 * data['obsA'] + data['obsB'])

    def test_same_formula_same_numbers_across_trace_shapes(self):
        # Language-agnosticism: identical formula+inputs over a "BNGL" and an "SBML" trace
        # whose shared columns carry the same numbers yields identical results.
        pytest.importorskip('petab')
        formula, syms = 'x^2 + 3*x', {'x'}
        a = _trace(['time', 'x'], [(0., 1.), (1., 2.), (2., 3.)])
        b = _trace(['time', 'x', 'extra'], [(0., 1., 9.), (1., 2., 9.), (2., 3., 9.)])
        ma = MeasurementModel('o', formula, syms).materialize(a, {})
        mb = MeasurementModel('o', formula, syms).materialize(b, {})
        np.testing.assert_allclose(ma, mb)
        np.testing.assert_allclose(ma, data_x_squared(a['x']))

    def test_resolution_priority_column_then_pset_then_constant(self):
        # A symbol present as a column wins over a same-named pset value; a symbol absent
        # from both falls to the model-constant snapshot.
        pytest.importorskip('petab')
        data = _trace(['time', 'k'], [(0., 1.), (1., 2.)])
        mm = MeasurementModel('o', 'k + c', {'k', 'c'}, constants={'c': 100.})
        # 'k' is a column (1,2), not the pset's 999; 'c' is the constant 100.
        got = mm.materialize(data, pset_values={'k': 999.})
        np.testing.assert_allclose(got, data['k'] + 100.)

    def test_all_scalar_formula_broadcasts_to_trace_length(self):
        pytest.importorskip('petab')
        data = _trace(['time', 'S1'], [(0., 5.), (1., 6.), (2., 7.)])
        mm = MeasurementModel('o', 'kA + kB', {'kA', 'kB'})
        got = mm.materialize(data, pset_values={'kA': 2., 'kB': 3.})
        np.testing.assert_allclose(got, np.full(3, 5.))

    def test_time_is_resolvable_as_a_column(self):
        pytest.importorskip('petab')
        data = _trace(['time', 'S1'], [(0., 5.), (2., 6.), (4., 7.)])
        mm = MeasurementModel('o', 'S1 + time', {'S1', 'time'})
        np.testing.assert_allclose(mm.materialize(data, {}), data['S1'] + data['time'])

    def test_unknown_symbol_rejected_at_compile(self):
        pytest.importorskip('petab')
        mm = MeasurementModel('o', 'S1 + nope', {'S1'})
        with pytest.raises(PybnfError, match='not a known model entity'):
            mm.materialize(_trace(['time', 'S1'], [(0., 1.)]), {})

    def test_placeholder_symbol_is_the_deferred_frontier(self):
        pytest.importorskip('petab')
        mm = MeasurementModel('o', 'S1 * observableParameter1_o', {'S1'})
        with pytest.raises(NotImplementedError, match='placeholder'):
            mm.materialize(_trace(['time', 'S1'], [(0., 1.)]), {})


def data_x_squared(x):
    return x ** 2 + 3 * x


# ---------------------------------------------------------------------------
# MeasurementLayer.apply -- the (sim_data_dict, pset) transform
# ---------------------------------------------------------------------------

class TestLayerApply:

    def _sim_dict(self):
        return {'model': {'time_course':
                          _trace(['time', 'S1', 'S2'],
                                 [(0., 10., 4.), (1., 8., 5.), (2., 6., 6.)])}}

    def test_apply_materializes_columns_in_place(self):
        pytest.importorskip('petab')
        sim = self._sim_dict()
        layer = MeasurementLayer([
            MeasurementModel('ratio', '(100*S1 + 200*S2)/(S1 + S2)', {'S1', 'S2'}),
            MeasurementModel('sum', 'S1 + S2', {'S1', 'S2'}),
        ])
        layer.apply(sim, pset_values={})
        data = sim['model']['time_course']
        assert 'ratio' in data.cols and 'sum' in data.cols
        s1, s2 = data['S1'], data['S2']
        np.testing.assert_allclose(data['ratio'], (100 * s1 + 200 * s2) / (s1 + s2))
        np.testing.assert_allclose(data['sum'], s1 + s2)

    def test_empty_layer_is_an_exact_no_op(self):
        # No compile, no petab needed: the structural no-op contract.
        sim = self._sim_dict()
        before = sim['model']['time_course'].data.copy()
        layer = MeasurementLayer()
        assert not layer
        out = layer.apply(sim, pset_values={})
        assert out is sim
        np.testing.assert_array_equal(sim['model']['time_course'].data, before)

    def test_column_collision_raises_not_silently_overwrites(self):
        pytest.importorskip('petab')
        sim = self._sim_dict()
        layer = MeasurementLayer([MeasurementModel('S1', 'S2 + 1', {'S2'})])  # S1 already a col
        with pytest.raises(PybnfError, match='shadow an existing'):
            layer.apply(sim, pset_values={})

    def test_materialized_layer_survives_pickling(self):
        # The objective carrying the layer is scattered to dask workers; a lambdify callable
        # is not picklable, so the model must recompile worker-side (ADR-0036 §5).
        pytest.importorskip('petab')
        mm = MeasurementModel('o', 'S1*2 + S2', {'S1', 'S2'})
        data = _trace(['time', 'S1', 'S2'], [(0., 3., 1.), (1., 4., 2.)])
        expected = mm.materialize(data, {})           # compiles, caches a callable
        restored = pickle.loads(pickle.dumps(mm))     # _compiled must be dropped + rebuilt
        np.testing.assert_allclose(restored.materialize(data, {}), expected)


# ---------------------------------------------------------------------------
# The objective seam -- evaluate_multiple applies the layer before scoring
# ---------------------------------------------------------------------------

class TestObjectiveSeam:

    def test_evaluate_multiple_materializes_then_scores_by_name(self):
        # End to end at the seam: a sos objective with a measurement layer scores the
        # *materialized* observable column against the exp data, using the PSet.
        pytest.importorskip('petab')
        from pybnf.objective import SumOfSquaresObjective

        sim = {'m': {'tc': _trace(['time', 'S1', 'S2'],
                                  [(0., 10., 4.), (1., 8., 5.), (2., 6., 6.)])}}
        # exp data has only the measurement-model column 'obs' (no S1/S2) -> the objective
        # can only match it because the layer materialized it.
        exp = {'m': {'tc': _trace(['time', 'obs'], [(0., 70.), (1., 60.), (2., 50.)])}}
        obj = SumOfSquaresObjective.from_config({'ind_var_rounding': 0})
        obj.measurement = MeasurementLayer(
            [MeasurementModel('obs', '5*S1 + S2', {'S1', 'S2'})])
        pset = [_FP('dummy', 1.0)]

        score = obj.evaluate_multiple(sim, exp, pset)
        materialized = 5 * np.array([10., 8., 6.]) + np.array([4., 5., 6.])
        expected = float(np.sum((materialized - np.array([70., 60., 50.])) ** 2))
        assert score == pytest.approx(expected)

    def test_no_layer_is_byte_identical_to_plain_objective(self):
        # The default (measurement is None) leaves the objective exactly as before.
        from pybnf.objective import SumOfSquaresObjective
        sim = {'m': {'tc': _trace(['time', 'obs'], [(0., 1.), (1., 2.)])}}
        exp = {'m': {'tc': _trace(['time', 'obs'], [(0., 1.5), (1., 2.5)])}}
        obj = SumOfSquaresObjective.from_config({'ind_var_rounding': 0})
        assert obj.measurement is None
        score = obj.evaluate_multiple(sim, exp, [_FP('x', 1.0)])
        assert score == pytest.approx(0.25 + 0.25)


# ---------------------------------------------------------------------------
# Config wiring -- an `observable: <id>, formula:` line builds the layer (ADR-0036)
# ---------------------------------------------------------------------------

class TestConfigBuildsMeasurementLayer:
    """The config wiring (no BNG / no simulation needed): a new-era ``observable: <id>,
    formula: <expr>`` line compiles into the measurement layer on the objective, with the
    model's expression namespace and a fixed-constant snapshot (free parameters excluded)."""

    _MODEL = """\
begin model
begin parameters
  kA 2
  kB 3
  scale 100
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 10
  B() 4
end seed species
begin observables
  Molecules obsA A()
  Molecules obsB B()
end observables
begin reaction rules
  A() -> B() kA
end reaction rules
end model
"""
    _EXP = '# time\tratio\n0\t5\n1\t6\n2\t7\n'

    def _build(self, tmp_path, *, formula=None):
        import os

        from pybnf import config as config_mod
        from pybnf.parse import ploop
        (tmp_path / 'm.bngl').write_text(self._MODEL)
        (tmp_path / 'meas.exp').write_text(self._EXP)
        lines = ['edition = 2', 'job_type = de', 'objective = sos', 'model: m.bngl']
        if formula is not None:
            lines.append(f'observable: ratio, formula: {formula}')
        lines += ['experiment: meas, data: meas.exp',
                  'uniform_var = kA 0 10', 'uniform_var = kB 0 10',
                  'population_size = 4', 'max_iterations = 1', 'verbosity = 0']
        conf_text = '\n'.join(lines) + '\n'
        home = os.getcwd()
        os.chdir(tmp_path)
        try:
            return config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
        finally:
            os.chdir(home)

    def test_layer_attached_with_model_namespace(self, tmp_path):
        pytest.importorskip('petab')
        conf = self._build(tmp_path, formula='(100*obsA + 200*obsB)/(obsB + kB*obsA)')
        assert conf.obj.measurement and len(conf.obj.measurement) == 1
        mm = conf.obj.measurement.models[0]
        assert mm.observable_id == 'ratio'
        assert mm.allowed_symbols == {'kA', 'kB', 'scale', 'obsA', 'obsB'}

    def test_fixed_param_is_a_constant_free_param_is_not(self, tmp_path):
        # A fixed model parameter (scale) is snapshotted as a constant; a free parameter
        # (kA) is excluded -- it resolves from the PSet at eval time (ADR-0036 §4).
        pytest.importorskip('petab')
        conf = self._build(tmp_path, formula='scale*obsA + kA')
        mm = conf.obj.measurement.models[0]
        assert mm.constants == {'scale': 100.0}
        assert 'kA' not in mm.constants and 'kB' not in mm.constants

    def test_unknown_symbol_in_formula_fails_fast_at_load(self, tmp_path):
        pytest.importorskip('petab')
        with pytest.raises(PybnfError, match='not a known model entity'):
            self._build(tmp_path, formula='obsA + nope')

    def test_no_formula_line_leaves_no_op_default(self, tmp_path):
        conf = self._build(tmp_path, formula=None)
        assert conf.obj.measurement is None
