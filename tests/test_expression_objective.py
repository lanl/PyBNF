"""Bring-your-own inline objective expression (ADR-0050, #425 "Tier 1").

``objective = expression`` + ``expression = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)`` declares a
user-written negative log-likelihood as PEtab math on the config line -- no ``.bngl`` /
``.target`` model file, no ``.exp`` data file, fit by the gradient-free samplers (de / am /
dream / p_dream). The expression compiles to a numpy callable
(``pybnf.petab.formula.compile_objective_expression``) whose free symbols bind to the declared
free parameters **by name** (``x1`` -> ``x1``), the bind-by-name fix the named-target slice
deferred (ADR-0059 item 6). It desugars to the existing ``score``-column seam, so there is no
new objective, sampler, or run-loop code -- a synthesized :class:`ExpressionModel` produces the
``score`` cell that ``DirectPassObjective`` already reads.

These tests exercise the real parser (``ploop``) + ``Configuration`` + the fitting algorithm
end to end against the analytical truth (the Rosenbrock mode), using the in-process fakes from
``integration_harness``. NB PEtab math uses ``^`` for exponentiation, not ``**``.
"""
import logging
import pickle

import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.parse import ploop
from pybnf.config import Configuration
from pybnf.objective import DirectPassObjective
from pybnf.analytical_model import ExpressionModel
from pybnf.printing import PybnfError

# The expression path needs the optional petab/sympy extra to compile the math; skip the whole
# module (not error) when it is absent, matching the petab-suite convention.
pytest.importorskip('petab', reason='objective = expression needs the optional pybnf[petab] extra')


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _build(tmp_path, body):
    """Build a real Configuration from a config-file ``body`` (output_dir appended)."""
    text = body + f'\noutput_dir = {tmp_path}/out\nwall_time_sim = 0\n'
    return Configuration(ploop(text.splitlines(keepends=True)))


_ROSENBROCK = 'expression = 0.5*((1 - x1)^2 + 20*(x2 - x1^2)^2)\n'
_DE_TAIL = """job_type = de
uniform_var = x1 -5 5
uniform_var = x2 -5 5
population_size = 5
max_iterations = 3
"""


# --------------------------------------------------------------------------- #
# Parsing: the new ``expression`` key + the ``objective = expression`` selector
# --------------------------------------------------------------------------- #
def test_parse_expression_key():
    d = ploop(['objective = expression\n', _ROSENBROCK])
    assert d['objective'] == 'expression'
    assert d['expression'] == '0.5*((1 - x1)^2 + 20*(x2 - x1^2)^2)'
    # An expression is NOT a named target -- no structural ('objective_target', None) key.
    assert ('objective_target', None) not in d


def test_parse_expression_strips_trailing_comment():
    d = ploop(['expression = x1 + x2   # the cost\n'])
    assert d['expression'] == 'x1 + x2'


def test_parse_expression_keeps_internal_punctuation():
    # The value grammar is permissive: operators, parens, spaces, ^ powers all survive verbatim.
    expr = 'log(1 + exp(-k)) + 0.5*(x - mu)^2 / s^2'
    d = ploop([f'expression = {expr}\n'])
    assert d['expression'] == expr


# --------------------------------------------------------------------------- #
# Configuration: synthesizes an ExpressionModel + DirectPassObjective, no files
# --------------------------------------------------------------------------- #
def test_expression_synthesizes_model_and_direct_pass(tmp_path):
    c = _build(tmp_path, 'edition = 2\nobjective = expression\n' + _ROSENBROCK + _DE_TAIL)
    assert list(c.models) == ['expression']
    m = c.models['expression']
    assert isinstance(m, ExpressionModel)
    assert isinstance(c.obj, DirectPassObjective)
    assert m.formula == '0.5*((1 - x1)^2 + 20*(x2 - x1^2)^2)'
    assert c._data_map['expression'] == []        # no experimental data
    assert c.mapping['expression'] == set()
    assert c.exp_data['expression'] == {}


def test_expression_binds_free_symbols_by_name(tmp_path):
    # ordered_names is the expression's free symbols (sorted), bound BY NAME to the PSet -- not
    # by the sorted-positional convention the menu (.target) path uses.
    c = _build(tmp_path, 'edition = 2\nobjective = expression\n' + _ROSENBROCK + _DE_TAIL)
    assert c.models['expression']._ordered_names == ['x1', 'x2']


def test_expression_subset_of_declared_params_is_allowed(tmp_path):
    # A declared parameter the expression does not reference is fine (the likelihood is flat in
    # that direction); only the referenced symbols bind.
    body = ('edition = 2\nobjective = expression\nexpression = (x1 - 3)^2\n'
            'job_type = de\nuniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    c = _build(tmp_path, body)
    assert c.models['expression']._ordered_names == ['x1']


def test_expression_echoes_at_run_start(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _build(tmp_path, 'edition = 2\nobjective = expression\n' + _ROSENBROCK + _DE_TAIL)
    assert any('x1' in r.message and 'x2' in r.message and 'expression' in r.message.lower()
               for r in caplog.records), 'the expression must be echoed at run start'


# --------------------------------------------------------------------------- #
# Bind-by-name validation: an undeclared symbol errors clearly, naming it
# --------------------------------------------------------------------------- #
def test_expression_unknown_symbol_errors_naming_it(tmp_path):
    body = ('edition = 2\nobjective = expression\nexpression = (x1 - zzz)^2\n'
            'job_type = de\nuniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match='zzz'):
        _build(tmp_path, body)


def test_expression_unparseable_errors(tmp_path):
    # ** is not PEtab math (it uses ^); a malformed expression errors at config load, not mid-run.
    body = ('edition = 2\nobjective = expression\nexpression = x1 ** ** 2\n'
            'job_type = de\nuniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError):
        _build(tmp_path, body)


def test_expression_missing_expression_key_errors(tmp_path):
    body = ('edition = 2\nobjective = expression\n'
            'job_type = de\nuniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match="requires an 'expression' key"):
        _build(tmp_path, body)


def test_expression_requires_edition_2(tmp_path):
    # The bring-your-own surface is modern syntax; without edition = 2 it errors clearly.
    with pytest.raises(PybnfError, match='edition'):
        _build(tmp_path, 'objective = expression\n' + _ROSENBROCK +
               'fit_type = de\nuniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
               'population_size = 5\nmax_iterations = 3')


def test_expression_needs_no_model_or_exp(tmp_path):
    # The whole point: a fileless objective. No model file and no .exp on disk -- the parser's
    # model-file set and exp-data set stay empty -- yet the synthesized model exists and runs.
    c = _build(tmp_path, 'edition = 2\nobjective = expression\n' + _ROSENBROCK + _DE_TAIL)
    assert c.config['models'] == set()       # no model *file* declared
    assert c.config['exp_data'] == set()     # no experimental data file declared
    assert set(c.models) == {'expression'}   # but the synthesized model is present


# --------------------------------------------------------------------------- #
# The synthesized model: bind-by-name eval, score column, picklable across dask
# --------------------------------------------------------------------------- #
def test_expression_model_pickles_dropping_callable(tmp_path):
    # The lambdified callable is not picklable; the model carries the formula string and
    # recompiles lazily on the worker. Force a compile, then round-trip.
    c = _build(tmp_path, 'edition = 2\nobjective = expression\n' + _ROSENBROCK + _DE_TAIL)
    m = c.models['expression']
    m._compiled()
    assert m._func is not None
    restored = pickle.loads(pickle.dumps(m))
    assert restored._func is None                       # dropped from pickle state
    assert restored.formula == m.formula
    assert restored._ordered_names == m._ordered_names
    assert restored._compiled() is not None             # recompiles on demand


def test_expression_model_evaluates_nll_by_name(tmp_path):
    from pybnf.pset import PSet, FreeParameter
    c = _build(tmp_path, 'edition = 2\nobjective = expression\n' + _ROSENBROCK + _DE_TAIL)
    m = c.models['expression']
    # The Rosenbrock NLL is 0 at the mode (x1, x2) = (1, 1) and 0.5 at the origin.
    pset_mode = PSet([FreeParameter('x1', 'uniform_var', -5, 5, value=1.0),
                      FreeParameter('x2', 'uniform_var', -5, 5, value=1.0)])
    out = m.copy_with_param_set(pset_mode).execute('', '', 0)
    data = out['expression']
    assert data.data[0, data.cols['score']] == pytest.approx(0.0)
    pset_origin = PSet([FreeParameter('x1', 'uniform_var', -5, 5, value=0.0),
                        FreeParameter('x2', 'uniform_var', -5, 5, value=0.0)])
    out0 = m.copy_with_param_set(pset_origin).execute('', '', 0)
    assert out0['expression'].data[0, out0['expression'].cols['score']] == pytest.approx(0.5)


def test_expression_binding_is_by_name_not_position(tmp_path):
    """Bind-by-name, the ADR-0050 §4 win: a parameter whose name sorts *after* the others but
    appears *first* in the cost must still bind to its own value. ``z`` sorts last yet is the
    minimized coordinate, so the mode is at z=4 regardless of sort order."""
    from pybnf.pset import PSet, FreeParameter
    body = ('edition = 2\nobjective = expression\nexpression = (z - 4)^2 + (a - 1)^2\n'
            'job_type = de\nuniform_var = a -10 10\nuniform_var = z -10 10\n'
            'population_size = 5\nmax_iterations = 3\n')
    c = _build(tmp_path, body)
    m = c.models['expression']
    assert m._ordered_names == ['a', 'z']
    pset = PSet([FreeParameter('a', 'uniform_var', -10, 10, value=1.0),
                 FreeParameter('z', 'uniform_var', -10, 10, value=4.0)])
    out = m.copy_with_param_set(pset).execute('', '', 0)
    assert out['expression'].data[0, out['expression'].cols['score']] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# End to end: a fit on the inline expression recovers the analytical truth
# --------------------------------------------------------------------------- #
def test_expression_de_recovers_rosenbrock_mode(tmp_path):
    # Rosenbrock mode is at (x1, x2) = (1, 1); a gentle curvature keeps DE tractable.
    body = ('edition = 2\nobjective = expression\n'
            'expression = 0.5*((1 - x1)^2 + 20*(x2 - x1^2)^2)\njob_type = de\n'
            'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 20\nmax_iterations = 300\nrandom_seed = 42')
    c = _build(tmp_path, body)
    alg = algorithms.DifferentialEvolution(c)
    H.drive(alg)
    bf = alg.trajectory.best_fit()
    assert bf['x1'] == pytest.approx(1.0, abs=0.1)
    assert bf['x2'] == pytest.approx(1.0, abs=0.1)
    assert alg.trajectory.best_score() == pytest.approx(0.0, abs=0.05)
