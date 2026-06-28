"""Data-bound inline objective expressions (ADR-0050 data follow-up) -- the last Tier-1 piece.

``objective = expression`` + a ``data = curve.exp`` key turns the inline NLL into a
**per-observation** contribution over the free parameters *and* the bound data columns:

    objective   = expression
    expression  = 0.5*(y - (m*x + b))^2     # references params m, b and data columns x, y
    data        = line.exp
    job_type    = de

The model (``ExpressionModel``) evaluates the expression once per data row -- parameters bound to
scalars, data columns to the row's arrays -- and **sums** over every row and every bound experiment
(the ``Σ per-point NLL`` taxonomy, #424). Data columns are not coordinates the sampler varies, so
``coordinate_order`` returns only the parameter symbols. With no ``data`` key the expression is the
original pure-parameter form, scored once.

These tests exercise the real parser + Configuration + algorithm end to end against analytic truth
(a line's least-squares mode), using the in-process fakes from ``integration_harness``.
"""
import logging
import pickle

import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.parse import ploop
from pybnf.config import Configuration
from pybnf.analytical_model import ExpressionModel
from pybnf.printing import PybnfError

pytest.importorskip('petab', reason='objective = expression needs the optional pybnf[petab] extra')


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _build(tmp_path, body):
    text = body + f'\noutput_dir = {tmp_path}/out\nwall_time_sim = 0\n'
    return Configuration(ploop(text.splitlines(keepends=True)))


def _write_exp(tmp_path, name, rows, headers=('x', 'y')):
    """Write a minimal ``.exp`` file (a '# h1 h2' header + whitespace rows) and return its path."""
    path = tmp_path / f'{name}.exp'
    lines = ['# ' + ' '.join(headers)]
    lines += [' '.join(repr(float(v)) for v in row) for row in rows]
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


# Points exactly on y = 2x + 1, so the least-squares mode is (m, b) = (2, 1) with SSE 0.
_LINE_ROWS = [(0, 1), (1, 3), (2, 5), (3, 7)]
_LINE_EXPR = 'expression = 0.5*(y - (m*x + b))^2'
_DE_TAIL = ('job_type = de\nuniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')


def _line_body(tmp_path, names=('line',)):
    paths = [_write_exp(tmp_path, n, _LINE_ROWS) for n in names]
    return (f'edition = 2\nobjective = expression\n{_LINE_EXPR}\n'
            f'data = {", ".join(paths)}\n' + _DE_TAIL)


# --------------------------------------------------------------------------- #
# Synthesis: a data-bound expression compiles over params + data columns
# --------------------------------------------------------------------------- #
def test_expression_binds_data_columns(tmp_path):
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['expression']
    assert isinstance(m, ExpressionModel)
    assert m._data_columns == {'x', 'y'}               # the referenced data columns
    assert m._param_names == ['b', 'm']                # params only, sorted (NOT the columns)
    assert set(m._data) == {'line'}                    # bound data keyed by file stem


def test_expression_coordinate_order_is_param_only(tmp_path):
    # The HMC permutation binds on coordinate_order; a data column must never appear there (the
    # sampler does not vary data). For a pure-param expression it is unchanged (all of ordered_names).
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['expression']
    assert m.coordinate_order(['m', 'b']) == ['b', 'm']   # params only


def test_expression_data_pickles_with_model(tmp_path):
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['expression']
    m._compiled()
    restored = pickle.loads(pickle.dumps(m))
    assert restored._func is None                         # callable dropped...
    assert set(restored._data) == {'line'}                # ...but the bound data travels
    assert restored._data_columns == {'x', 'y'}
    assert list(restored._data['line']['y']) == [1.0, 3.0, 5.0, 7.0]


# --------------------------------------------------------------------------- #
# Evaluation: per-observation sum over rows (and experiments)
# --------------------------------------------------------------------------- #
def test_expression_data_sums_per_observation(tmp_path):
    from pybnf.pset import PSet, FreeParameter
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['expression']
    # SSE is 0 at the true (m, b) = (2, 1), and the sum of squared residuals off it.
    true = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=2.0),
                 FreeParameter('b', 'uniform_var', -5, 5, value=1.0)])
    out = m.copy_with_param_set(true).execute('', '', 0)
    assert out['expression'].data[0, out['expression'].cols['score']] == pytest.approx(0.0)
    # At (m, b) = (0, 0) the residuals are y = [1,3,5,7]; 0.5*sum(y^2) = 0.5*84 = 42.
    origin = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=0.0),
                   FreeParameter('b', 'uniform_var', -5, 5, value=0.0)])
    out0 = m.copy_with_param_set(origin).execute('', '', 0)
    assert out0['expression'].data[0, out0['expression'].cols['score']] == pytest.approx(42.0)


def test_expression_data_pools_over_experiments(tmp_path):
    # Two .exp files -> the per-observation sum runs over BOTH; split y = 2x + 1 across them, so the
    # pooled mode is still (2, 1) with SSE 0.
    from pybnf.pset import PSet, FreeParameter
    a = _write_exp(tmp_path, 'expA', [(0, 1), (1, 3)])
    b = _write_exp(tmp_path, 'expB', [(2, 5), (3, 7)])
    body = (f'edition = 2\nobjective = expression\n{_LINE_EXPR}\n'
            f'data = {a}, {b}\n' + _DE_TAIL)
    c = _build(tmp_path, body)
    m = c.models['expression']
    assert set(m._data) == {'expA', 'expB'}
    true = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=2.0),
                 FreeParameter('b', 'uniform_var', -5, 5, value=1.0)])
    out = m.copy_with_param_set(true).execute('', '', 0)
    assert out['expression'].data[0, out['expression'].cols['score']] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Pointed errors at config load
# --------------------------------------------------------------------------- #
def test_expression_data_column_param_collision_errors(tmp_path):
    # A data column whose name equals a declared free parameter is ambiguous -> rejected.
    body = ('edition = 2\nobjective = expression\nexpression = (y - x)^2\n'
            f'data = {_write_exp(tmp_path, "d", _LINE_ROWS)}\n'
            'job_type = de\nuniform_var = x -5 5\nuniform_var = y -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match='collide'):
        _build(tmp_path, body)


def test_expression_data_missing_column_in_experiment_errors(tmp_path):
    # The expression references 'y', but one bound experiment carries only x/z -> fail fast.
    a = _write_exp(tmp_path, 'good', _LINE_ROWS, headers=('x', 'y'))
    b = _write_exp(tmp_path, 'bad', _LINE_ROWS, headers=('x', 'z'))
    body = (f'edition = 2\nobjective = expression\n{_LINE_EXPR}\n'
            f'data = {a}, {b}\n' + _DE_TAIL)
    with pytest.raises(PybnfError, match="missing data column"):
        _build(tmp_path, body)


def test_expression_data_unknown_symbol_still_errors(tmp_path):
    # A symbol that is neither a declared parameter nor a bound data column is a typo -> error.
    body = ('edition = 2\nobjective = expression\nexpression = (y - m*x - zzz)^2\n'
            f'data = {_write_exp(tmp_path, "d", _LINE_ROWS)}\n'
            'job_type = de\nuniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match='zzz'):
        _build(tmp_path, body)


def test_expression_data_bound_but_unused_warns(tmp_path, caplog):
    # Data bound but the expression references no column -> the data is unused; warn, don't error.
    body = ('edition = 2\nobjective = expression\nexpression = (m - 3)^2\n'
            f'data = {_write_exp(tmp_path, "d", _LINE_ROWS)}\n' + _DE_TAIL)
    with caplog.at_level(logging.WARNING):
        c = _build(tmp_path, body)
    # No data columns referenced -> pure-parameter model, the original path.
    assert c.models['expression']._data_columns == set()


def test_expression_without_data_is_unchanged(tmp_path):
    # The original pure-parameter contract: no data key -> _data is None, no data columns.
    body = ('edition = 2\nobjective = expression\nexpression = (m - 3)^2 + (b - 1)^2\n' + _DE_TAIL)
    c = _build(tmp_path, body)
    m = c.models['expression']
    assert m._data is None and m._data_columns == set()
    assert m._param_names == ['b', 'm']


# --------------------------------------------------------------------------- #
# End to end: a fit on a data-bound expression recovers the curve
# --------------------------------------------------------------------------- #
def test_expression_data_de_recovers_line_fit(tmp_path):
    body = (f'edition = 2\nobjective = expression\n{_LINE_EXPR}\n'
            f'data = {_write_exp(tmp_path, "line", _LINE_ROWS)}\njob_type = de\n'
            'uniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 20\nmax_iterations = 300\nrandom_seed = 42')
    c = _build(tmp_path, body)
    alg = algorithms.DifferentialEvolution(c)
    H.drive(alg)
    bf = alg.trajectory.best_fit()
    assert bf['m'] == pytest.approx(2.0, abs=0.1)
    assert bf['b'] == pytest.approx(1.0, abs=0.1)
    assert alg.trajectory.best_score() == pytest.approx(0.0, abs=0.05)
