"""Bring-your-own callable objective (ADR-0050, #425 "Tier 1") -- the expression form's sibling.

``objective = callable`` + ``callable = mymodule:negative_log_likelihood`` points a fit/sample at a
user-written negative log-likelihood supplied as a **Python callable** -- the escape hatch for
densities the inline ``expression`` grammar cannot express (logsumexp mixtures, loops over groups,
``scipy.stats``). The entry point resolves to ``f(params, data=None) -> float``; the parameters are
passed **by name** (``dict(pset)``). Like the expression form it desugars to the existing
``score``-column seam -- a synthesized :class:`CallableModel` produces the ``score`` cell that
``DirectPassObjective`` already reads -- with no ``.bngl`` / ``.target`` model file and no ``.exp``,
fit by the gradient-free samplers (de / am / dream / p_dream). It is gradient-free (a general Python
callable is not JAX-traceable), so ``job_type = hmc`` rejects it with a pointed error.

These tests exercise the real parser (``ploop``) + ``Configuration`` + the fitting algorithm end to
end against the analytical truth (an isotropic Gaussian's mode), using the in-process fakes from
``integration_harness``. The callable form needs no optional extra (it imports user Python, no
sympy/jax), so this module is never skipped.
"""
import logging
import pickle

import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.parse import ploop
from pybnf.config import Configuration
from pybnf.objective import DirectPassObjective
from pybnf.analytical_model import CallableModel
from pybnf.printing import PybnfError


# A throwaway callable-objective target module written to tmp_path. ``gaussian_nll`` is an isotropic
# Gaussian NLL with its mode (NLL 0) at (x1, x2) = (3, -2); ``offset_nll`` lets a late-sorting name
# (``z``) carry the minimized coordinate to prove bind-by-name; ``not_a_function`` is a non-callable
# attribute for the pointed-error test.
_TARGET_SRC = '''
def gaussian_nll(params, data=None):
    return 0.5 * ((params["x1"] - 3.0) ** 2 + (params["x2"] + 2.0) ** 2)


def offset_nll(params, data=None):
    return (params["z"] - 4.0) ** 2 + (params["a"] - 1.0) ** 2


def line_sse(params, data):
    # Sum of squares of a line y = m*x + b against the single bound "line" experiment.
    d = data["line"]
    pred = params["m"] * d["x"] + params["b"]
    return 0.5 * float(((d["y"] - pred) ** 2).sum())


def pooled_sse(params, data):
    # SSE POOLED across ALL bound experiments (multi-experiment): one shared slope/intercept.
    total = 0.0
    for d in data.values():
        pred = params["m"] * d["x"] + params["b"]
        total += float(((d["y"] - pred) ** 2).sum())
    return 0.5 * total


def names_of_data(params, data):
    # Returns 0 always but ASSERTS the data arg is the name->Data mapping we expect; lets a test
    # observe what the callable actually receives without a side channel.
    assert sorted(data) == ["expA", "expB"], sorted(data)
    return 0.0


not_a_function = 42
'''


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _write_target(tmp_path, name='callable_target'):
    """Write the target module to ``tmp_path/<name>.py`` and return its path (str)."""
    path = tmp_path / f'{name}.py'
    path.write_text(_TARGET_SRC)
    return str(path)


def _write_exp(tmp_path, name, rows, headers=('x', 'y')):
    """Write a minimal ``.exp`` file (a '# h1 h2' header + whitespace rows) and return its path."""
    path = tmp_path / f'{name}.exp'
    lines = ['# ' + ' '.join(headers)]
    lines += [' '.join(repr(float(v)) for v in row) for row in rows]
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


# Points exactly on y = 2x + 1, so the least-squares mode is (m, b) = (2, 1) with SSE 0.
_LINE_ROWS = [(0, 1), (1, 3), (2, 5), (3, 7)]


def _build(tmp_path, body):
    """Build a real Configuration from a config-file ``body`` (output_dir appended)."""
    text = body + f'\noutput_dir = {tmp_path}/out\nwall_time_sim = 0\n'
    return Configuration(ploop(text.splitlines(keepends=True)))


_DE_TAIL = """job_type = de
uniform_var = x1 -5 5
uniform_var = x2 -5 5
population_size = 5
max_iterations = 3
"""


def _gaussian_body(tmp_path, *, entry=None):
    """An ``objective = callable`` config body whose entry point is the file-path form by default."""
    if entry is None:
        entry = f'{_write_target(tmp_path)}:gaussian_nll'
    return f'edition = 2\nobjective = callable\ncallable = {entry}\n' + _DE_TAIL


# --------------------------------------------------------------------------- #
# Parsing: the new ``callable`` key + the ``objective = callable`` selector
# --------------------------------------------------------------------------- #
def test_parse_callable_key():
    d = ploop(['objective = callable\n', 'callable = mymodule:negative_log_likelihood\n'])
    assert d['objective'] == 'callable'
    assert d['callable'] == 'mymodule:negative_log_likelihood'
    # A callable is NOT a named target -- no structural ('objective_target', None) key.
    assert ('objective_target', None) not in d


def test_parse_callable_strips_trailing_comment():
    d = ploop(['callable = mymodule:func   # the entry point\n'])
    assert d['callable'] == 'mymodule:func'


def test_parse_callable_keeps_path_punctuation():
    # The value grammar is permissive: a dotted module, a file path, and the ':' separator survive.
    d = ploop(['callable = path/to/file.py:negative_log_likelihood\n'])
    assert d['callable'] == 'path/to/file.py:negative_log_likelihood'


# --------------------------------------------------------------------------- #
# Configuration: synthesizes a CallableModel + DirectPassObjective, no files
# --------------------------------------------------------------------------- #
def test_callable_synthesizes_model_and_direct_pass(tmp_path):
    entry = f'{_write_target(tmp_path)}:gaussian_nll'
    c = _build(tmp_path, _gaussian_body(tmp_path, entry=entry))
    assert list(c.models) == ['callable']
    m = c.models['callable']
    assert isinstance(m, CallableModel)
    assert isinstance(c.obj, DirectPassObjective)
    assert m.entry_point == entry
    assert c._data_map['callable'] == []        # no experimental data
    assert c.mapping['callable'] == set()
    assert c.exp_data['callable'] == {}


def test_callable_resolves_dotted_module(tmp_path, monkeypatch):
    # The other entry-point form: an importable dotted module (vs. the file path above). Put the
    # module dir on sys.path and reference it by bare module name.
    _write_target(tmp_path, name='callable_dotted_target')
    monkeypatch.syspath_prepend(str(tmp_path))
    c = _build(tmp_path, _gaussian_body(tmp_path, entry='callable_dotted_target:gaussian_nll'))
    assert isinstance(c.models['callable'], CallableModel)


def test_callable_echoes_at_run_start(tmp_path, caplog):
    entry = f'{_write_target(tmp_path)}:gaussian_nll'
    with caplog.at_level(logging.INFO):
        _build(tmp_path, _gaussian_body(tmp_path, entry=entry))
    assert any('callable' in r.message.lower() and 'gaussian_nll' in r.message
               for r in caplog.records), 'the callable entry point must be echoed at run start'


# --------------------------------------------------------------------------- #
# Fail-fast validation at config load: pointed errors, not mid-run failures
# --------------------------------------------------------------------------- #
def test_callable_missing_callable_key_errors(tmp_path):
    body = 'edition = 2\nobjective = callable\n' + _DE_TAIL
    with pytest.raises(PybnfError, match="requires a 'callable' key"):
        _build(tmp_path, body)


def test_callable_requires_edition_2(tmp_path):
    # The bring-your-own surface is modern syntax; without edition = 2 it errors clearly.
    entry = f'{_write_target(tmp_path)}:gaussian_nll'
    body = (f'objective = callable\ncallable = {entry}\nfit_type = de\n'
            'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match='edition'):
        _build(tmp_path, body)


def test_callable_unknown_module_errors(tmp_path):
    body = ('edition = 2\nobjective = callable\ncallable = no_such_module_xyz:func\n' + _DE_TAIL)
    with pytest.raises(PybnfError, match='no_such_module_xyz'):
        _build(tmp_path, body)


def test_callable_missing_attribute_errors(tmp_path):
    entry = f'{_write_target(tmp_path)}:does_not_exist'
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n' + _DE_TAIL)
    with pytest.raises(PybnfError, match='does_not_exist'):
        _build(tmp_path, body)


def test_callable_malformed_no_colon_errors(tmp_path):
    body = ('edition = 2\nobjective = callable\ncallable = justamodulename\n' + _DE_TAIL)
    with pytest.raises(PybnfError, match='[Mm]alformed'):
        _build(tmp_path, body)


def test_callable_non_callable_attribute_errors(tmp_path):
    entry = f'{_write_target(tmp_path)}:not_a_function'
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n' + _DE_TAIL)
    with pytest.raises(PybnfError, match='non-callable'):
        _build(tmp_path, body)


def test_callable_needs_no_model_or_exp(tmp_path):
    # The whole point: a fileless objective. No model file and no .exp on disk -- yet the
    # synthesized model exists and runs.
    c = _build(tmp_path, _gaussian_body(tmp_path))
    assert c.config['models'] == set()       # no model *file* declared
    assert c.config['exp_data'] == set()     # no experimental data file declared
    assert set(c.models) == {'callable'}     # but the synthesized model is present


# --------------------------------------------------------------------------- #
# The synthesized model: bind-by-name eval, score column, picklable across dask
# --------------------------------------------------------------------------- #
def test_callable_model_pickles_dropping_func(tmp_path):
    # The resolved function is not assumed picklable; the model carries the entry-point string and
    # re-imports lazily on the worker. Force a resolve, then round-trip.
    c = _build(tmp_path, _gaussian_body(tmp_path))
    m = c.models['callable']
    m._resolved()
    assert m._func is not None
    restored = pickle.loads(pickle.dumps(m))
    assert restored._func is None                       # dropped from pickle state
    assert restored.entry_point == m.entry_point
    assert restored._resolved() is not None             # re-imports on demand


def test_callable_model_evaluates_nll_by_name(tmp_path):
    from pybnf.pset import PSet, FreeParameter
    c = _build(tmp_path, _gaussian_body(tmp_path))
    m = c.models['callable']
    # gaussian_nll is 0 at the mode (x1, x2) = (3, -2) and 0.5 at (4, -2).
    pset_mode = PSet([FreeParameter('x1', 'uniform_var', -5, 5, value=3.0),
                      FreeParameter('x2', 'uniform_var', -5, 5, value=-2.0)])
    out = m.copy_with_param_set(pset_mode).execute('', '', 0)
    data = out['callable']
    assert data.data[0, data.cols['score']] == pytest.approx(0.0)
    pset_off = PSet([FreeParameter('x1', 'uniform_var', -5, 5, value=4.0),
                     FreeParameter('x2', 'uniform_var', -5, 5, value=-2.0)])
    out2 = m.copy_with_param_set(pset_off).execute('', '', 0)
    assert out2['callable'].data[0, out2['callable'].cols['score']] == pytest.approx(0.5)


def test_callable_binding_is_by_name(tmp_path):
    """The callable receives the whole {name: value} map, so binding is by name regardless of how
    the parameters sort: ``z`` sorts last yet is a minimized coordinate, mode at z=4."""
    from pybnf.pset import PSet, FreeParameter
    entry = f'{_write_target(tmp_path)}:offset_nll'
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n'
            'job_type = de\nuniform_var = a -10 10\nuniform_var = z -10 10\n'
            'population_size = 5\nmax_iterations = 3\n')
    c = _build(tmp_path, body)
    m = c.models['callable']
    pset = PSet([FreeParameter('a', 'uniform_var', -10, 10, value=1.0),
                 FreeParameter('z', 'uniform_var', -10, 10, value=4.0)])
    out = m.copy_with_param_set(pset).execute('', '', 0)
    assert out['callable'].data[0, out['callable'].cols['score']] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# End to end: a fit on the inline callable recovers the analytical truth
# --------------------------------------------------------------------------- #
def test_callable_de_recovers_gaussian_mode(tmp_path):
    entry = f'{_write_target(tmp_path)}:gaussian_nll'
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\njob_type = de\n'
            'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 20\nmax_iterations = 300\nrandom_seed = 42')
    c = _build(tmp_path, body)
    alg = algorithms.DifferentialEvolution(c)
    H.drive(alg)
    bf = alg.trajectory.best_fit()
    assert bf['x1'] == pytest.approx(3.0, abs=0.1)
    assert bf['x2'] == pytest.approx(-2.0, abs=0.1)
    assert alg.trajectory.best_score() == pytest.approx(0.0, abs=0.05)


# --------------------------------------------------------------------------- #
# Gradient-free: job_type = hmc rejects a callable target with a pointed error
# --------------------------------------------------------------------------- #
def test_callable_rejected_by_hmc(tmp_path):
    # A general Python callable is not JAX-traceable; HMC must refuse it (no log-density gradient),
    # pointing at objective = expression / a menu target instead. Resolved without importing jax --
    # the rejection is in _resolve_analytical_model, before any gradient machinery.
    entry = f'{_write_target(tmp_path)}:gaussian_nll'
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\njob_type = hmc\n'
            'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 4\nmax_iterations = 10\n')
    c = _build(tmp_path, body)
    alg = algorithms.HMCSampler(c)
    with pytest.raises(PybnfError, match='gradient-free'):
        alg._resolve_analytical_model()


# --------------------------------------------------------------------------- #
# Data binding (ADR-0050 follow-up): the ``data = f.exp, ...`` key feeds the
# callable a name->Data mapping keyed by file stem (mirrors the params map)
# --------------------------------------------------------------------------- #
def _line_body(tmp_path, entry_func='line_sse', data_files=('line',)):
    entry = f'{_write_target(tmp_path)}:{entry_func}'
    data_paths = [_write_exp(tmp_path, name, _LINE_ROWS) for name in data_files]
    return (f'edition = 2\nobjective = callable\ncallable = {entry}\n'
            f'data = {", ".join(data_paths)}\njob_type = de\n'
            'uniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')


def test_callable_no_data_key_passes_none(tmp_path):
    # The pure-analytical case is unchanged: no data key -> the model carries None, callable
    # invoked f(params, data=None).
    c = _build(tmp_path, _gaussian_body(tmp_path))
    assert c.models['callable']._data is None


def test_callable_binds_single_experiment_keyed_by_stem(tmp_path):
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['callable']
    from pybnf.data import Data
    assert set(m._data) == {'line'}            # keyed by file stem, one entry per experiment
    assert isinstance(m._data['line'], Data)
    # The loaded Data exposes the file's columns to the callable.
    assert list(m._data['line']['y']) == [1.0, 3.0, 5.0, 7.0]


def test_callable_data_evaluates_against_measurements(tmp_path):
    from pybnf.pset import PSet, FreeParameter
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['callable']
    # SSE is 0 at the true (m, b) = (2, 1) and positive off it.
    pset_true = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=2.0),
                      FreeParameter('b', 'uniform_var', -5, 5, value=1.0)])
    out = m.copy_with_param_set(pset_true).execute('', '', 0)
    assert out['callable'].data[0, out['callable'].cols['score']] == pytest.approx(0.0)
    pset_off = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=0.0),
                     FreeParameter('b', 'uniform_var', -5, 5, value=0.0)])
    out_off = m.copy_with_param_set(pset_off).execute('', '', 0)
    assert out_off['callable'].data[0, out_off['callable'].cols['score']] > 0.0


def test_callable_multi_experiment_pooled(tmp_path):
    # Two .exp files -> two named entries; the callable pools its NLL across both. The pieces of
    # y = 2x + 1 are split across the files, so the pooled mode is still (2, 1) with SSE 0.
    entry = f'{_write_target(tmp_path)}:pooled_sse'
    a = _write_exp(tmp_path, 'expA', [(0, 1), (1, 3)])
    b = _write_exp(tmp_path, 'expB', [(2, 5), (3, 7)])
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n'
            f'data = {a}, {b}\njob_type = de\n'
            'uniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    c = _build(tmp_path, body)
    m = c.models['callable']
    assert set(m._data) == {'expA', 'expB'}
    from pybnf.pset import PSet, FreeParameter
    pset_true = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=2.0),
                      FreeParameter('b', 'uniform_var', -5, 5, value=1.0)])
    out = m.copy_with_param_set(pset_true).execute('', '', 0)
    assert out['callable'].data[0, out['callable'].cols['score']] == pytest.approx(0.0)


def test_callable_receives_name_keyed_mapping(tmp_path):
    # The callable asserts internally that it receives {'expA': Data, 'expB': Data}; a clean
    # execute proves the multi-experiment presentation is by name.
    from pybnf.pset import PSet, FreeParameter
    entry = f'{_write_target(tmp_path)}:names_of_data'
    a = _write_exp(tmp_path, 'expA', _LINE_ROWS)
    b = _write_exp(tmp_path, 'expB', _LINE_ROWS)
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n'
            f'data = {a}, {b}\njob_type = de\n'
            'uniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    c = _build(tmp_path, body)
    m = c.models['callable']
    pset = PSet([FreeParameter('m', 'uniform_var', -5, 5, value=0.0),
                 FreeParameter('b', 'uniform_var', -5, 5, value=0.0)])
    out = m.copy_with_param_set(pset).execute('', '', 0)   # the internal assert must not raise
    assert out['callable'].data[0, out['callable'].cols['score']] == pytest.approx(0.0)


def test_callable_data_pickles_with_model(tmp_path):
    import pickle
    c = _build(tmp_path, _line_body(tmp_path))
    m = c.models['callable']
    restored = pickle.loads(pickle.dumps(m))
    # The bound data travels with the model (unlike _func, which is dropped + re-imported).
    assert set(restored._data) == {'line'}
    assert list(restored._data['line']['x']) == [0.0, 1.0, 2.0, 3.0]


def test_callable_data_de_recovers_line_fit(tmp_path):
    body = (f'edition = 2\nobjective = callable\n'
            f'callable = {_write_target(tmp_path)}:line_sse\n'
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


def test_callable_data_missing_file_errors(tmp_path):
    entry = f'{_write_target(tmp_path)}:line_sse'
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n'
            f'data = {tmp_path}/nope.exp\njob_type = de\n'
            'uniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match='not found'):
        _build(tmp_path, body)


def test_callable_data_stem_collision_errors(tmp_path):
    # Two files with the same stem in different dirs would map to the same experiment name.
    entry = f'{_write_target(tmp_path)}:pooled_sse'
    sub = tmp_path / 'sub'
    sub.mkdir()
    a = _write_exp(tmp_path, 'curve', _LINE_ROWS)
    b = _write_exp(sub, 'curve', _LINE_ROWS)
    body = (f'edition = 2\nobjective = callable\ncallable = {entry}\n'
            f'data = {a}, {b}\njob_type = de\n'
            'uniform_var = m -5 5\nuniform_var = b -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match='same experiment name'):
        _build(tmp_path, body)


def test_data_key_with_non_analytical_objective_errors(tmp_path):
    # The data key is for bring-your-own analytical objectives (expression/callable) only; a stray
    # data key under a menu target (or any other objective) is a pointed error, not a silent no-op.
    body = ('edition = 2\nobjective = banana, a = 1, b = 100\n'
            f'data = {_write_exp(tmp_path, "line", _LINE_ROWS)}\njob_type = de\n'
            'uniform_var = x1 -5 5\nuniform_var = x2 -5 5\n'
            'population_size = 5\nmax_iterations = 3\n')
    with pytest.raises(PybnfError, match="only valid with 'objective = expression'"):
        _build(tmp_path, body)


def test_callable_data_echoes_at_run_start(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _build(tmp_path, _line_body(tmp_path))
    assert any('line' in r.message and 'data' in r.message.lower()
               for r in caplog.records), 'the bound data must be echoed at run start'
