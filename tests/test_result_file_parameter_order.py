"""Which order each result file lists the free parameters in (lanl/PyBNF#932).

The ``starting_params`` entry in docs/config_keys.rst warns that ``starting_params`` is
positional against declaration order, while ``samples.txt``, ``sorted_params*.txt`` and
``bootstrapped_parameter_sets.txt`` sort their parameter columns by name, case-sensitively, so a row
copied out of one of those files into ``starting_params`` lands its values on the wrong
parameters. docs/troubleshooting.rst says ``start_point.txt`` is in declaration order. The
same entry says ``continue_run = 1`` overrides ``starting_params``. These tests pin each of
those statements against what PyBNF really writes, so the documentation cannot go stale
silently when a writer changes.

The parameters are declared ``kcat``, ``Km``, ``alpha``. That order is neither sorted nor
case-insensitive alphabetical (``alpha``, ``kcat``, ``Km``), and the files hold
``Km``, ``alpha``, ``kcat``: the sort is Python's case-sensitive string order, which puts
every uppercase letter before every lowercase one. The expected orders below are written
out by hand rather than computed with ``sorted()``, so they are an oracle independent of
the writers.

Each parameter's box is disjoint from the other two, so a value written under the wrong
header lands outside the box its header names. That is how the row checks below tell a
correctly labelled column from a permuted one without knowing what the fit converged to.
"""
import os
import types

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms, config

DECLARED = ['kcat', 'Km', 'alpha']
SORTED_IN_FILES = ['Km', 'alpha', 'kcat']   # case-sensitive: uppercase first
BOXES = {'kcat': (1.0, 9.0), 'Km': (50.0, 90.0), 'alpha': (0.1, 0.9)}
TRUE = {'kcat': 7.0, 'Km': 70.0, 'alpha': 0.3}


def test_the_fixture_distinguishes_the_three_orders():
    """Declaration order, case-insensitive alphabetical order and the order the files
    use must all differ, or the tests below could not tell them apart."""
    assert sorted(DECLARED, key=str.casefold) == ['alpha', 'kcat', 'Km']
    assert len({tuple(DECLARED), tuple(SORTED_IN_FILES), ('alpha', 'kcat', 'Km')}) == 3
    for name, (lo, hi) in BOXES.items():
        for other, (lo2, hi2) in BOXES.items():
            if other != name:
                assert hi < lo2 or hi2 < lo, (name, other)
        assert lo < TRUE[name] < hi


def _vars_conf(declare):
    return ''.join('uniform_var = %s %g %g\n' % (n, *BOXES[n]) for n in declare)


def _read_table(path, n_lead):
    """Header names after the ``n_lead`` leading non-parameter columns, and the rows as
    ``{name: value}`` dicts keyed by those header names."""
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    header = lines[0].lstrip('#').split()
    names = header[n_lead:]
    rows = [dict(zip(names, map(float, ln.split()[n_lead:]))) for ln in lines[1:]]
    return names, rows


def _assert_each_value_in_the_box_its_header_names(rows, where):
    assert rows, where
    for row in rows:
        for name, value in row.items():
            lo, hi = BOXES[name]
            assert lo <= value <= hi, (where, name, value)


def _declared_rows(path):
    """``(name, first value)`` for each row of a one-row-per-parameter file
    (start_point.txt, credible*.txt), skipping comments and the ``starts`` lines."""
    out = []
    with open(path) as f:
        for ln in f.read().splitlines():
            parts = ln.split('\t')
            if ln.startswith('#') or not ln.strip() or parts[0] in ('starts', 'starts_pinned'):
                continue
            out.append((parts[0], float(parts[1])))
    return out


# --------------------------------------------------------------------------- #
# A sampler: samples.txt and sorted_params are sorted, the per-parameter files are
# in declaration order, and starting_params is applied by declaration order.
# --------------------------------------------------------------------------- #

# One negative log-likelihood, written twice: as ``objective = expression`` and as an
# ``objective = callable`` Python function. The expression goes through the PEtab math
# translator, so its cases need the optional petab extra (#1023). The callable needs no
# extra, so the sampler's file orders are still checked where petab is not installed.
_NLL_EXPRESSION = ('0.5*((kcat - 7)/0.1)^2 + 0.5*((Km - 70)/1)^2 '
                   '+ 0.5*((alpha - 0.3)/0.01)^2')
_NLL_CALLABLE_SRC = (
    'def nll(params, data=None):\n'
    '    return (0.5 * ((params["kcat"] - 7) / 0.1) ** 2 + 0.5 * ((params["Km"] - 70) / 1) ** 2\n'
    '            + 0.5 * ((params["alpha"] - 0.3) / 0.01) ** 2)\n')
OBJECTIVES = [pytest.param('expression', marks=pytest.mark.petab), 'callable']

_SAMPLER_TAIL = (
    _vars_conf(DECLARED) +
    'starting_params = 7 70 0.3\n'
    'job_type = am\n'
    'max_iterations = 40\n'
    'burn_in = 10\n'
    'adaptive = 10\n'
    'sample_every = 1\n'
    'population_size = 2\n'
    'output_hist_every = 1000000000\n'
    'credible_intervals = 68\n'
    'random_seed = 1\n'
    'output_dir = out\n'
    'verbosity = 0\n'
    'backup_every = 1000000000\n'
    'output_every = 1000000000\n'
    'diagnostics_every = 1000000000\n')


def _nll_by_hand(row):
    """The objective at a ``{name: value}`` row, computed here rather than by PyBNF."""
    return (0.5 * ((row['kcat'] - 7) / 0.1) ** 2 + 0.5 * ((row['Km'] - 70) / 1) ** 2
            + 0.5 * ((row['alpha'] - 0.3) / 0.01) ** 2)


def _column(path, k):
    """Column ``k`` of every data row of a result table, as floats."""
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return [float(ln.split()[k]) for ln in lines[1:]]


def _sampler_conf(tmp_path, objective):
    if objective == 'expression':
        head = 'objective = expression\nexpression = %s\n' % _NLL_EXPRESSION
    else:
        target = tmp_path / 'order_nll.py'
        target.write_text(_NLL_CALLABLE_SRC)
        head = 'objective = callable\ncallable = %s:nll\n' % target
    return 'edition = 2\n' + head + _SAMPLER_TAIL


def _run_conf(tmp_path, monkeypatch, conf_text):
    from pybnf.parse import load_config
    from pybnf.pybnf import _create_algorithm
    H.install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'order.conf').write_text(conf_text)
    conf = load_config(str(tmp_path / 'order.conf'))
    alg = _create_algorithm(conf)
    H.drive(alg)
    return conf, alg, tmp_path / 'out' / 'Results'


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_sampler_files_and_starting_params_order(tmp_path, monkeypatch, objective):
    _, alg, results = _run_conf(tmp_path, monkeypatch, _sampler_conf(tmp_path, objective))
    assert [v.name for v in alg.variables] == DECLARED

    # starting_params = 7 70 0.3 is read by declaration order: kcat 7, Km 70, alpha 0.3.
    # start_point.txt lists the rows in declaration order.
    assert _declared_rows(results / 'start_point.txt') == [
        ('kcat', 7.0), ('Km', 70.0), ('alpha', 0.3)]

    # samples.txt: Name, Ln_probability, then the parameters in sorted order.
    names, rows = _read_table(results / 'samples.txt', 2)
    assert names == SORTED_IN_FILES
    _assert_each_value_in_the_box_its_header_names(rows, 'samples.txt')
    # Each row's Ln_probability is the log posterior of the values under their headers:
    # minus the objective, plus the log density of the three uniform priors.
    log_prior = -sum(np.log(hi - lo) for lo, hi in BOXES.values())
    np.testing.assert_allclose(_column(results / 'samples.txt', 1),
                               [log_prior - _nll_by_hand(r) for r in rows], rtol=1e-9)
    assert len({tuple(r.values()) for r in rows}) > 1      # the chains moved

    # sorted_params_final.txt: Simulation, Obj, then the parameters in sorted order.
    names, rows = _read_table(results / 'sorted_params_final.txt', 2)
    assert names == SORTED_IN_FILES
    _assert_each_value_in_the_box_its_header_names(rows, 'sorted_params_final.txt')
    np.testing.assert_allclose(_column(results / 'sorted_params_final.txt', 1),
                               [_nll_by_hand(r) for r in rows], rtol=1e-9, atol=1e-12)

    # The credible-interval rows are in declaration order (#932).
    credible = sorted(results.glob('credible68*_final.txt'))
    assert len(credible) == 1
    assert [n for n, _ in _declared_rows(credible[0])] == DECLARED

    # am's own posterior files are in declaration order too.
    with open(results / 'A_MCMC' / 'Runs' / 'params_0.txt') as f:
        assert f.readline().split() == DECLARED


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_continue_run_overrides_starting_params(tmp_path, monkeypatch, objective):
    """With ``continue_run = 1`` every chain starts from the previous run's
    ``adaptive_files/MLE_params.txt`` (one value per parameter, declaration order), and
    ``starting_params`` is ignored."""
    from pybnf.parse import load_config
    from pybnf.pybnf import _create_algorithm
    H.install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    adaptive = tmp_path / 'out' / 'adaptive_files'
    adaptive.mkdir(parents=True)
    previous_mle = [3.0, 60.0, 0.5]                       # kcat, Km, alpha
    np.savetxt(adaptive / 'MLE_params.txt', previous_mle)
    np.savetxt(adaptive / 'diff.txt', [0.1])
    np.savetxt(adaptive / 'diffMatrix.txt', np.eye(3))
    (tmp_path / 'order.conf').write_text(
        _sampler_conf(tmp_path, objective) + 'continue_run = 1\n')
    alg = _create_algorithm(load_config(str(tmp_path / 'order.conf')))
    os.makedirs(alg.res_dir, exist_ok=True)

    first = alg.start_run()

    assert len(first) == 2
    for pset in first:
        assert [(p.name, p.value) for p in pset] == list(zip(DECLARED, previous_mle))


# --------------------------------------------------------------------------- #
# An optimizer with bootstrapping: sorted_params.txt (delete_old_files = 2), each
# replicate's sorted_params.txt, and bootstrapped_parameter_sets.txt are sorted.
# --------------------------------------------------------------------------- #

class _NamedLinearModel(H.LinearGaussianModel):
    """``f(x) = A x`` with ``x = (kcat, Km, alpha)`` read by name, scored by chi_sq
    against real data, so the bootstrap has data rows to resample."""

    def execute(self, folder, filename, timeout):
        from pybnf.data import Data
        x = np.array([self._pset[n] for n in DECLARED])
        y = self.A @ x
        d = Data(arr=np.column_stack([np.arange(len(y), dtype=float), y]))
        d.cols = {'index': 0, 'y': 1}
        d.headers = {0: 'index', 1: 'y'}
        return {'target': d}


def _bootstrap_config(tmp_path):
    A = np.vstack([np.eye(3), np.eye(3)])                  # each parameter measured twice
    truth = np.array([TRUE[n] for n in DECLARED])
    sigma = np.tile([0.1, 1.0, 0.01], 2)
    tgt, _ = H.write_target(tmp_path, H.gaussian_spec([0.0] * 3, [1.0] * 3))
    exp = tmp_path / 'target.exp'
    exp.write_text('# index\ty\ty_SD\n' + ''.join(
        '%d\t%.17g\t%.17g\n' % (i, v, s) for i, (v, s) in enumerate(zip(A @ truth, sigma))))
    base = {
        'output_dir': str(tmp_path / 'out'),
        'models': {tgt}, tgt: [str(exp)], 'exp_data': {str(exp)},
        'objfunc': 'chi_sq', 'fit_type': 'de', 'initialization': 'lh',
        'population_size': 6, 'max_iterations': 3, 'bootstrap': 2,
        'delete_old_files': 2, 'verbosity': 0, 'wall_time_sim': 0, 'random_seed': 7,
    }
    base.update({('uniform_var', n): list(BOXES[n]) for n in DECLARED})
    conf = config.Configuration(base)
    conf.models['g'] = _NamedLinearModel(A, name='g')
    return conf


def test_optimizer_and_bootstrap_files_are_sorted(tmp_path, monkeypatch):
    from pybnf.pybnf import _run_bootstrapping
    H.install(monkeypatch)
    conf = _bootstrap_config(tmp_path)
    alg = algorithms.DifferentialEvolution(conf)
    assert [v.name for v in alg.variables] == DECLARED
    H.drive(alg)
    out = tmp_path / 'out'
    assert [n for n, _ in _declared_rows(out / 'Results' / 'start_point.txt')] == DECLARED

    # delete_old_files = 2 collapses every sorted_params_<name>.txt into sorted_params.txt.
    names, rows = _read_table(out / 'Results' / 'sorted_params.txt', 2)
    assert names == SORTED_IN_FILES
    _assert_each_value_in_the_box_its_header_names(rows, 'sorted_params.txt')

    _run_bootstrapping(conf, alg, types.SimpleNamespace(client=H.FakeClient()), False)

    names, rows = _read_table(out / 'Results' / 'bootstrapped_parameter_sets.txt', 2)
    assert names == SORTED_IN_FILES
    assert len(rows) == 2
    _assert_each_value_in_the_box_its_header_names(rows, 'bootstrapped_parameter_sets.txt')
    for n in range(2):
        names, rows = _read_table(out / ('Results-boot%d' % n) / 'sorted_params.txt', 2)
        assert names == SORTED_IN_FILES
        _assert_each_value_in_the_box_its_header_names(rows, 'Results-boot%d' % n)
        assert [r for r, _ in _declared_rows(out / ('Results-boot%d' % n) / 'start_point.txt')] \
            == DECLARED
