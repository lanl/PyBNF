"""Bind-by-name on the analytical menu (ADR-0034, the edition=2 binding contract).

The menu analytical targets (``AnalyticalModel``: banana / gaussian / rotated_gaussian /
rotated_quartic / multimodal, from a ``.target`` file or the inline ``objective = banana``
line) used to bind their anonymous coordinates to free parameters by **sorted name** -- a
silent, surprising positional convention (the #425 footgun: ``z``/``a`` bind to coordinate
2/1 only because ``a`` sorts first). This retrofits the edition=2 bind-by-name contract
(ADR-0034, already used by the ``objective = expression`` path): coordinate ``i`` is the
declared parameter whose name ends in index ``i+1`` (``x1``/``p1`` -> coordinate 1), so the
binding is independent of declaration order and lexical sort, and an unindexed / wrong-count
name set errors clearly instead of binding the wrong coordinate.

These tests exercise the real parser + ``Configuration`` + the model's eval, plus an
end-to-end fit, against the analytical truth (an asymmetric target whose coordinates are
distinguishable).
"""
import json
import os

import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.parse import ploop
from pybnf.config import Configuration
from pybnf.analytical_model import AnalyticalModel
from pybnf.pset import PSet, FreeParameter
from pybnf.printing import PybnfError


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _build(tmp_path, body):
    text = body + f'\noutput_dir = {tmp_path}/out\nwall_time_sim = 0\n'
    return Configuration(ploop(text.splitlines(keepends=True)))


def _banana(a=1.0, b=100.0):
    return AnalyticalModel(target_def={'type': 'banana', 'a': a, 'b': b}, name='banana')


def _gaussian(mean, variance):
    return AnalyticalModel(
        target_def={'type': 'gaussian', 'mean': list(mean), 'variance': list(variance)},
        name='g')


# --------------------------------------------------------------------------- #
# coordinate_order: the by-name ordering rule (the heart of the fix)
# --------------------------------------------------------------------------- #
def test_coordinate_order_is_by_index_not_declaration_order():
    # Declaration order reversed; the integer index still decides the coordinate.
    assert _banana().coordinate_order(['p2', 'p1']) == ['p1', 'p2']
    assert _banana().coordinate_order(['x2', 'x1']) == ['x1', 'x2']


def test_coordinate_order_any_prefix_works():
    # The contract is the integer suffix, not a fixed 'x'/'p' prefix.
    assert _banana().coordinate_order(['theta1', 'theta2']) == ['theta1', 'theta2']


def test_coordinate_order_is_natural_not_lexical_sort():
    # The old sorted() convention put x10/x11 before x2 (lexical); the index order is natural,
    # so an 11-D target binds x1..x11 correctly. (banana is any-dimension: D = #declared.)
    names = [f'x{i}' for i in range(1, 12)]
    scrambled = ['x11', 'x2', 'x1', 'x10', 'x3', 'x7', 'x5', 'x9', 'x4', 'x8', 'x6']
    assert _banana().coordinate_order(scrambled) == names
    # A lexical sort would have produced this wrong order -- guard against a regression to it.
    assert sorted(scrambled) != names


def test_coordinate_order_unindexed_name_errors_naming_it():
    with pytest.raises(PybnfError, match='alpha'):
        _banana().coordinate_order(['alpha', 'beta'])


def test_coordinate_order_wrong_count_for_fixed_dim_errors():
    # A 2-D gaussian has exactly two coordinates; three indexed params is an error (previously
    # the extra silently rode along and broke a numpy broadcast deep in _compute_nll).
    with pytest.raises(PybnfError, match=r'2 coordinate'):
        _gaussian([0.0, 0.0], [1.0, 1.0]).coordinate_order(['p1', 'p2', 'p3'])


def test_coordinate_order_gap_errors():
    with pytest.raises(PybnfError, match=r'\[1, 3\]'):
        _gaussian([0.0, 0.0], [1.0, 1.0]).coordinate_order(['x1', 'x3'])


def test_coordinate_order_duplicate_index_errors():
    with pytest.raises(PybnfError):
        _gaussian([0.0, 0.0], [1.0, 1.0]).coordinate_order(['x1', 'y1'])


# --------------------------------------------------------------------------- #
# execute(): the score path binds coordinates to parameter VALUES by name
# --------------------------------------------------------------------------- #
def test_execute_binds_values_by_name_on_asymmetric_gaussian():
    # mean = [10, -10] -> coordinate 1 peaks at +10, coordinate 2 at -10. With p1 = coord 1,
    # p2 = coord 2, the values at the mean give NLL 0; swapping the values gives a large NLL.
    m = _gaussian([10.0, -10.0], [1.0, 1.0])
    at_mode = PSet([FreeParameter('p1', 'uniform_var', -50, 50, value=10.0),
                    FreeParameter('p2', 'uniform_var', -50, 50, value=-10.0)])
    out = m.copy_with_param_set(at_mode).execute('', '', 0)['target']
    assert out.data[0, out.cols['score']] == pytest.approx(0.0)

    swapped = PSet([FreeParameter('p1', 'uniform_var', -50, 50, value=-10.0),
                    FreeParameter('p2', 'uniform_var', -50, 50, value=10.0)])
    out2 = m.copy_with_param_set(swapped).execute('', '', 0)['target']
    assert out2.data[0, out2.cols['score']] == pytest.approx(400.0)


# --------------------------------------------------------------------------- #
# Eager (config-load) validation: the footgun errors at build, not at run
# --------------------------------------------------------------------------- #
def test_inline_banana_unindexed_params_error_at_config_load(tmp_path):
    body = ('edition = 2\nobjective = banana, a = 1, b = 100\njob_type = de\n'
            'uniform_var = alpha -5 5\nuniform_var = beta -5 5\n'
            'population_size = 5\nmax_iterations = 3')
    with pytest.raises(PybnfError, match='alpha'):
        _build(tmp_path, body)


def test_file_target_wrong_param_count_rejected_when_bound(tmp_path):
    # The file ``.target`` path validates coordinates lazily (the model may be a param-agnostic
    # throwaway vehicle in a config-only test), so the config BUILDS; the by-name check fires
    # when a fit binds the coordinates (``coordinate_order`` / ``_get_param_values``). Three
    # params for a 2-D gaussian is then a pointed error, not a silent broadcast.
    (tmp_path / 'gaussian.target').write_text(
        json.dumps({'type': 'gaussian', 'mean': [1.0, -1.0], 'variance': [1.0, 1.0]}))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        body = ('edition = 2\nmodel: gaussian.target\nobjective = score\njob_type = de\n'
                'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\nuniform_var = x3 -5 5\n'
                'population_size = 5\nmax_iterations = 3')
        c = _build(tmp_path, body)            # builds: file .target is lazily validated
        with pytest.raises(PybnfError, match=r'2 coordinate'):
            c.models['gaussian'].coordinate_order(['x1', 'x2', 'x3'])
    finally:
        os.chdir(cwd)


def test_inline_banana_indexed_params_build_fine(tmp_path):
    body = ('edition = 2\nobjective = banana, a = 1, b = 100\njob_type = de\n'
            'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3')
    c = _build(tmp_path, body)
    assert isinstance(c.models['banana'], AnalyticalModel)


# --------------------------------------------------------------------------- #
# End to end: a fit binds by name regardless of declaration order
# --------------------------------------------------------------------------- #
def test_de_recovers_asymmetric_banana_with_reversed_declaration(tmp_path):
    # banana mode is (x1, x2) = (a, a^2) = (2, 4) -- distinguishable coordinates. Declaring x2
    # BEFORE x1 must not change the binding (bind-by-name): DE still recovers x1=2, x2=4.
    body = ('edition = 2\nobjective = banana, a = 2, b = 20\njob_type = de\n'
            'uniform_var = x2 -5 8\nuniform_var = x1 -5 8\n'      # reversed declaration
            'population_size = 30\nmax_iterations = 400\nrandom_seed = 17')
    c = _build(tmp_path, body)
    alg = algorithms.DifferentialEvolution(c)
    H.drive(alg)
    bf = alg.trajectory.best_fit()
    assert bf['x1'] == pytest.approx(2.0, abs=0.15)
    assert bf['x2'] == pytest.approx(4.0, abs=0.3)
    assert alg.trajectory.best_score() == pytest.approx(0.0, abs=0.05)
