"""Tests for ``BayesianAlgorithm.update_histograms`` — the shared output path that
turns ``Results/samples.txt`` into per-parameter histograms and credible-interval
files for ``mh``, ``pt``, ``dream`` and ``p_dream``.

The method reads the samples file back with ``numpy.genfromtxt``, which drops any
axis of length one. A fit with a single free parameter therefore read back as a 1-D
array of samples, a lone sample as a 1-D array of parameters, and one of each as a
0-d scalar — and the old ``ndim < 2`` emptiness guard treated all three as "no
samples collected", skipping every histogram and credible interval for the whole run
while the samples file itself filled up as normal (lanl/PyBNF#769). These tests pin
the shape contract at each of those corners against the *content* of the files, so
they fail if the array comes back transposed as well as if it comes back flat.

The method also has to find each parameter's column by name. samples.txt stores the
parameters in alphabetical order, while ``variables`` is in the order the configuration
declares them, and the method used to pair the two by position: whenever the orders
differed, one parameter's credible intervals and histogram were built from another
parameter's samples, binned in the wrong scale (lanl/PyBNF#856). ``_write_samples``
below writes its header in declaration order, which is why the tests above could not
see that; the tests at the end write the file with PyBNF's own writer.

The method reads only ``samples_file``, ``variables``, ``credible_intervals``,
``num_bins`` and the output directory, so it is exercised on bare ``object.__new__``
instances (following test_bayesian_diagnostics) rather than paying for the full
model-parsing constructor — these need no BNG2.pl.
"""
import glob
import importlib.util
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms, pset
from pybnf.printing import PybnfError

BA = algorithms.BayesianAlgorithm


def _bare_ba(tmp_path, n_vars, credible_intervals=(68, 95), num_bins=10):
    """A BayesianAlgorithm with only what ``update_histograms`` reads, and the
    ``Results/Histograms`` scaffolding ``start_run`` would normally have made."""
    out = str(tmp_path)
    os.makedirs(os.path.join(out, 'Results', 'Histograms'), exist_ok=True)
    ba = object.__new__(BA)
    ba.samples_file = os.path.join(out, 'Results', 'samples.txt')
    ba.variables = [pset.FreeParameter('p%d' % (i + 1), 'uniform_var', 0.0, 1.0, 0.5)
                    for i in range(n_vars)]
    ba.credible_intervals = list(credible_intervals)
    ba.num_bins = num_bins
    ba.config = SimpleNamespace(config={'output_dir': out})
    return ba


def _write_samples(ba, rows):
    """Write a samples file in the real format: a commented header, then one row of
    ``name``, ``ln_probability``, and a value per variable."""
    names = [v.name for v in ba.variables]
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\t' + '\t'.join(names) + '\n')
        for i, row in enumerate(rows):
            f.write('s%d\t-1.0\t' % i + '\t'.join('%.17g' % x for x in row) + '\n')


def _read_credible(tmp_path, interval, ext):
    """Parse ``credible{interval}{ext}.txt`` into ``{param: (lower, upper)}``."""
    path = os.path.join(str(tmp_path), 'Results', 'credible%d%s.txt' % (interval, ext))
    with open(path) as f:
        lines = f.readlines()
    assert lines[0] == '# param\tlower_bound\tupper_bound\n'
    out = {}
    for line in lines[1:]:
        name, lo, hi = line.split('\t')
        out[name] = (float(lo), float(hi))
    return out


@pytest.mark.parametrize('n_vars', [1, 2, 3])
def test_histograms_written_for_every_parameter_count(tmp_path, n_vars):
    """One file per parameter, and the counts add up to the number of samples.

    n_vars = 1 is the regression: ``genfromtxt`` over a single column returns a 1-D
    array, which the old guard read as an empty sample set, so a one-parameter fit
    wrote no histogram at all. n_vars = 2 and 3 are the controls that always worked
    and must keep working.
    """
    n_samples = 40
    rng = np.random.default_rng(5)
    rows = rng.uniform(0.0, 1.0, size=(n_samples, n_vars))
    ba = _bare_ba(tmp_path, n_vars)
    _write_samples(ba, rows)

    ba.update_histograms('_10')

    hist_dir = os.path.join(str(tmp_path), 'Results', 'Histograms')
    assert sorted(os.listdir(hist_dir)) == ['p%d_10.txt' % (i + 1) for i in range(n_vars)]
    for i in range(n_vars):
        a = np.genfromtxt(os.path.join(hist_dir, 'p%d_10.txt' % (i + 1)))
        assert a.shape == (ba.num_bins, 3)          # lower edge, upper edge, count
        assert a[:, 2].sum() == n_samples           # every sample landed in a bin
        # The binning is this parameter's column, not another's and not the transpose.
        assert a[0, 0] == pytest.approx(rows[:, i].min())
        assert a[-1, 1] == pytest.approx(rows[:, i].max())


def test_credible_intervals_written_with_one_parameter(tmp_path):
    """The credible intervals are the deliverable of a Bayesian fit, and a
    one-parameter fit silently produced none. Oracle: the method reports the order
    statistics of the sorted column at the documented indices, so with a known sample
    the bounds are exact values rather than a tolerance.
    """
    n_samples = 100
    rows = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)   # sorted, known quantiles
    ba = _bare_ba(tmp_path, 1)
    _write_samples(ba, rows)

    ba.update_histograms('_final')

    col = sorted(rows[:, 0])
    for interval in (68, 95):
        want = n_samples * (interval / 100)
        lo_i = max(0, int(np.round(n_samples / 2 - want / 2)))
        hi_i = min(n_samples - 1, int(np.round(n_samples / 2 + want / 2 - 1)))
        bounds = _read_credible(tmp_path, interval, '_final')
        assert set(bounds) == {'p1'}
        assert bounds['p1'] == (pytest.approx(col[lo_i]), pytest.approx(col[hi_i]))


def test_single_sample_still_reported(tmp_path):
    """The other axis ``genfromtxt`` collapses: a file holding exactly one sample
    comes back 1-D over the *parameters* (and 0-d when there is also only one
    parameter). Both used to read as "no samples". One sample is a degenerate
    posterior, but it is a real state a short or early-stopped run reaches, and the
    files must describe it rather than go missing: the whole count sits in one bin and
    every credible interval collapses to that value.
    """
    for n_vars in (1, 2):
        sub = tmp_path / ('nv%d' % n_vars)
        sub.mkdir()
        ba = _bare_ba(sub, n_vars)
        _write_samples(ba, [[0.25 + 0.1 * i for i in range(n_vars)]])

        ba.update_histograms('_1')

        for i in range(n_vars):
            a = np.genfromtxt(os.path.join(str(sub), 'Results', 'Histograms',
                                           'p%d_1.txt' % (i + 1)))
            assert a[:, 2].sum() == 1
        bounds = _read_credible(sub, 68, '_1')
        assert len(bounds) == n_vars
        for i in range(n_vars):
            lo, hi = bounds['p%d' % (i + 1)]
            assert lo == pytest.approx(0.25 + 0.1 * i)
            assert hi == pytest.approx(0.25 + 0.1 * i)


@pytest.mark.parametrize('n_vars', [1, 2])
def test_no_samples_still_skips(tmp_path, n_vars):
    """The guard's real job is preserved: a samples file holding only its header is
    genuinely empty, so nothing is written and nothing raises. This is what keeps the
    fix from being "delete the check" -- ``update_histograms`` runs on a stride from
    the run loop and fires before the first sample exists.
    """
    ba = _bare_ba(tmp_path, n_vars)
    _write_samples(ba, [])

    ba.update_histograms('_0')

    assert os.listdir(os.path.join(str(tmp_path), 'Results', 'Histograms')) == []
    assert [f for f in os.listdir(os.path.join(str(tmp_path), 'Results'))
            if f.startswith('credible')] == []


# --------------------------------------------------------------------------- #
# Columns are found by name, not by position (lanl/PyBNF#856)
# --------------------------------------------------------------------------- #

# Declared out of alphabetical order, with a log parameter among linear ones, and with
# samples in ranges that do not overlap, so a column read for the wrong parameter shows
# up both in the values and in the histogram's scale.
_OUT_OF_ORDER = [('x2', 'uniform_var', 40.0, 60.0, (45.0, 55.0)),
                 ('x1', 'uniform_var', -10.0, 10.0, (-2.0, 2.0)),
                 ('a3', 'loguniform_var', 0.01, 1000.0, (1.0, 100.0))]


def _order_statistics(column, interval):
    """The bounds ``update_histograms`` documents: order statistics of the sorted
    column at the rounded central indices."""
    col = sorted(column)
    n = len(col)
    want = n * (interval / 100)
    lo_i = max(0, int(np.round(n / 2 - want / 2)))
    hi_i = min(n - 1, int(np.round(n / 2 + want / 2 - 1)))
    return col[lo_i], col[hi_i]


def _write_samples_as_pybnf_does(ba, columns, n_samples):
    """Write samples.txt with the same PSet methods ``sample_pset`` and ``start_run``
    use, so the column order is whatever PyBNF really writes (alphabetical)."""
    with open(ba.samples_file, 'w') as f:
        header_pset = pset.PSet([v.set_value(columns[v.name][0]) for v in ba.variables])
        f.write('# Name\tLn_probability\t' + header_pset.keys_to_string() + '\n')
        for i in range(n_samples):
            ps = pset.PSet([v.set_value(columns[v.name][i]) for v in ba.variables])
            f.write('s%d\t-1.0\t' % i + ps.values_to_string() + '\n')


def test_columns_are_read_by_name_when_declared_out_of_order(tmp_path):
    """Oracle: each parameter's own samples, generated here and never read back from
    the file. Every credible bound must be the order statistic of that parameter's
    column, and every histogram must span that parameter's samples in that
    parameter's scale."""
    out = str(tmp_path)
    os.makedirs(os.path.join(out, 'Results', 'Histograms'))
    ba = object.__new__(BA)
    ba.samples_file = os.path.join(out, 'Results', 'samples.txt')
    ba.variables = [pset.FreeParameter(name, kind, lo, hi, (lo + hi) / 2 if kind == 'uniform_var' else 1.0)
                    for name, kind, lo, hi, _ in _OUT_OF_ORDER]
    ba.credible_intervals = [68, 95]
    ba.num_bins = 10
    ba.config = SimpleNamespace(config={'output_dir': out})

    n_samples = 200
    rng = np.random.default_rng(3)
    columns = {name: rng.uniform(a, b, n_samples) for name, _, _, _, (a, b) in _OUT_OF_ORDER}
    _write_samples_as_pybnf_does(ba, columns, n_samples)
    with open(ba.samples_file) as f:
        assert f.readline().split()[3:] == ['a3', 'x1', 'x2']   # really not declaration order

    ba.update_histograms('_final')

    for interval in (68, 95):
        bounds = _read_credible(tmp_path, interval, '_final')
        assert set(bounds) == {'x2', 'x1', 'a3'}
        for name in bounds:
            assert bounds[name] == tuple(pytest.approx(b) for b in
                                         _order_statistics(columns[name], interval)), name
    for v in ba.variables:
        path = os.path.join(out, 'Results', 'Histograms', v.name + '_final.txt')
        with open(path) as f:
            header = f.readline()
        assert header.startswith('# log10_lower_bound' if v.name == 'a3' else '# lower_bound'), v.name
        a = np.genfromtxt(path)
        own = columns[v.name] if v.name != 'a3' else np.log10(columns[v.name])
        assert a[0, 0] == pytest.approx(own.min()), v.name
        assert a[-1, 1] == pytest.approx(own.max()), v.name
        assert a[:, 2].sum() == n_samples


def test_samples_file_without_a_parameter_column_is_refused(tmp_path):
    """A variable with no column must stop the run with its name, not be paired with
    whatever column happens to sit at its index."""
    ba = _bare_ba(tmp_path, 2)
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tp1\tq9\n')
        f.write('s0\t-1.0\t0.1\t0.2\n')
    with pytest.raises(PybnfError, match='p2'):
        ba.update_histograms('_final')


def test_samples_file_without_the_header_is_refused(tmp_path):
    ba = _bare_ba(tmp_path, 1)
    with open(ba.samples_file, 'w') as f:
        f.write('0.1 0.2\n')
    with pytest.raises(PybnfError, match='header'):
        ba.update_histograms('_final')


def _nothing_written(tmp_path):
    results = os.path.join(str(tmp_path), 'Results')
    return (os.listdir(os.path.join(results, 'Histograms')) == []
            and not [f for f in os.listdir(results) if f.startswith('credible')])


def test_samples_file_with_a_column_that_is_not_a_parameter_is_refused(tmp_path):
    """Every parameter has its column here, plus one that names no parameter of this
    fit. Such a file belongs to another configuration, so the run must stop naming the
    stray column, not read the columns that happen to overlap and ignore the rest."""
    ba = _bare_ba(tmp_path, 2)
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tp1\tp2\tq9\n')
        f.write('s0\t-1.0\t0.1\t0.2\t0.3\n')
    with pytest.raises(PybnfError, match='a column for q9, which is not a free parameter of this fit'):
        ba.update_histograms('_final')
    assert _nothing_written(tmp_path)


def test_samples_file_with_a_repeated_column_is_refused(tmp_path):
    """Two columns headed p2 leave no way to say which holds p2's samples, so the run
    must stop naming it rather than take either one."""
    ba = _bare_ba(tmp_path, 2)
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tp1\tp2\tp2\n')
        f.write('s0\t-1.0\t0.1\t0.2\t0.3\n')
    with pytest.raises(PybnfError, match='more than one column for p2'):
        ba.update_histograms('_final')
    assert _nothing_written(tmp_path)


def test_samples_file_refusal_names_every_mismatch(tmp_path):
    """A missing parameter and a stray column are both named in one message, so the
    user sees the whole mismatch at once."""
    ba = _bare_ba(tmp_path, 3)
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tp1\tp3\tq9\tq8\n')
        f.write('s0\t-1.0\t0.1\t0.2\t0.3\t0.4\n')
    with pytest.raises(PybnfError) as err:
        ba.update_histograms('_final')
    assert 'no column for p2' in str(err.value)
    assert 'columns for q9, q8, which are not free parameters' in str(err.value)


# --------------------------------------------------------------------------- #
# Columns are found by name: the corners of the name lookup (lanl/PyBNF#856)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('declared', [
    # Python's str sort is case-sensitive (upper before lower) and puts '_' before
    # 'a', so the file holds Kd, k_on, ka: exactly the reverse of the declaration.
    ['ka', 'k_on', 'Kd'],
    # Names equal to the two header tokens the method strips off the front.
    ['zeta', 'Name', 'Ln_probability', 'alpha'],
    # More than nine parameters: str sort puts p10..p12 between p1 and p2, the shape
    # of the shipped W1..W9, W0 example.
    ['p%d' % i for i in range(1, 13)],
], ids=['case', 'header_tokens', 'twelve'])
def test_columns_read_by_name_not_by_permutation(tmp_path, declared):
    """Each variable is read from the column carrying its name, including when a name
    sorts by case, when it equals a header token, and when numeric suffixes sort as
    strings.

    Oracle: each parameter's own samples, drawn here in ranges that do not overlap, so
    a column read for the wrong parameter changes the bounds."""
    out = str(tmp_path)
    os.makedirs(os.path.join(out, 'Results', 'Histograms'))
    ba = object.__new__(BA)
    ba.samples_file = os.path.join(out, 'Results', 'samples.txt')
    ba.variables = [pset.FreeParameter(n, 'uniform_var', -1e6, 1e6, 0.0) for n in declared]
    ba.credible_intervals = [68, 95]
    ba.num_bins = 10
    ba.config = SimpleNamespace(config={'output_dir': out})

    n_samples = 60
    rng = np.random.default_rng(11)
    columns = {n: rng.uniform(100.0 * k, 100.0 * k + 50.0, n_samples)
               for k, n in enumerate(declared)}
    in_file = sorted(columns)                     # the order PSet.keys_to_string writes
    assert in_file != declared                    # the declaration really is out of order
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\t' + '\t'.join(in_file) + '\n')
        for i in range(n_samples):
            f.write('iter%drun0\t-1.0\t' % i
                    + '\t'.join(repr(float(columns[n][i])) for n in in_file) + '\n')

    ba.update_histograms('_final')

    for interval in (68, 95):
        path = os.path.join(out, 'Results', 'credible%d_final.txt' % interval)
        with open(path) as f:
            lines = f.read().splitlines()[1:]
        assert [line.split('\t')[0] for line in lines] == declared
        for line in lines:
            name, lo, hi = line.split('\t')
            assert (float(lo), float(hi)) == tuple(
                pytest.approx(b) for b in _order_statistics(columns[name], interval)), name
    for name in declared:
        h = np.genfromtxt(os.path.join(out, 'Results', 'Histograms', name + '_final.txt'))
        assert h[0, 0] == pytest.approx(columns[name].min()), name
        assert h[-1, 1] == pytest.approx(columns[name].max()), name


# --------------------------------------------------------------------------- #
# Every sampler, every write: strided _<iter> and _final (lanl/PyBNF#856)
# --------------------------------------------------------------------------- #

_ORDER_CONF = (
    'edition = 2\n'
    'objective = expression\n'
    'expression = 0.5*((x2 - 50)/1)^2 + 0.5*((x1 - 0)/1)^2 + 0.5*((a3 - 10)/2)^2\n'
    'uniform_var = x2 40 60\n'
    'uniform_var = x1 -10 10\n'
    'loguniform_var = a3 0.01 1000\n'
    'credible_intervals = 68 95\n'
    'random_seed = 1\n'
    'output_dir = out\n'
    'verbosity = 0\n'
    # Keep the per-iteration bookkeeping out of the fast tier's time.
    'backup_every = 1000000000\n'
    'output_every = 1000000000\n'
    'diagnostics_every = 1000000000\n')

_HAS_JAX = all(importlib.util.find_spec(m) is not None for m in ('jax', 'blackjax'))

_PER_SAMPLER = {
    'mh': ('max_iterations = 150\nburn_in = 50\nsample_every = 1\npopulation_size = 2\n'
           'output_hist_every = 50\n'),
    'pt': ('max_iterations = 150\nburn_in = 50\nsample_every = 1\npopulation_size = 4\n'
           'reps_per_beta = 2\nbeta = 0.5 1.0\nexchange_every = 10\noutput_hist_every = 50\n'),
    'am': ('max_iterations = 200\nburn_in = 50\nadaptive = 50\nsample_every = 1\n'
           'population_size = 2\noutput_hist_every = 50\n'),
    'dream': ('max_iterations = 120\nburn_in = 60\nsample_every = 1\npopulation_size = 4\n'
              'output_hist_every = 30\n'),
    'p_dream': ('max_iterations = 120\nburn_in = 60\nsample_every = 1\npopulation_size = 4\n'
                'output_hist_every = 30\n'),
    # hmc writes only _final.
    'hmc': 'num_warmup = 50\nnum_samples = 75\npopulation_size = 2\nmax_iterations = 75\n',
}


def _build_order_alg(tmp_path, monkeypatch, conf):
    """Build the algorithm for ``conf`` under the in-process harness without running it."""
    from pybnf.parse import load_config
    from pybnf.pybnf import _create_algorithm
    H.install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'order.conf').write_text(conf)
    return _create_algorithm(load_config(str(tmp_path / 'order.conf')))


def _run_order_conf(tmp_path, monkeypatch, conf):
    """Run ``conf`` to its stop through the in-process harness; return the algorithm
    and its ``Results`` directory."""
    alg = _build_order_alg(tmp_path, monkeypatch, conf)
    H.drive(alg)
    return alg, tmp_path / 'out' / 'Results'


def _assert_end_equals_final(results):
    """Every ``_final`` credible file and histogram has an ``_end`` twin, byte for byte."""
    finals = glob.glob(str(results / 'credible*_final.txt'))
    finals += glob.glob(str(results / 'Histograms' / '*_final.txt'))
    assert len(finals) == 2 + 3
    for final in finals:
        end = final.replace('_final.txt', '_end.txt')
        assert os.path.exists(end), end
        with open(final) as f, open(end) as g:
            assert f.read() == g.read(), final


def _check_every_credible_file(results):
    """Check every credible file under ``results`` against samples.txt, and return the
    set of extensions (``_final``, ``_<iter>``, ``_end``) found.

    Each file must hold, on each parameter's row, the order statistics of that
    parameter's own samples.txt column. samples.txt is append-only, so a strided write
    saw exactly its first N rows, N being the count total of that write's histogram.
    Each histogram must also span that parameter's samples in that parameter's scale.
    """
    with open(results / 'samples.txt') as f:
        header = f.readline().lstrip('#').split()
    assert header[2:] == ['a3', 'x1', 'x2']
    data = np.genfromtxt(results / 'samples.txt', skip_header=1, ndmin=2)
    by_name = {n: data[:, i] for i, n in enumerate(header) if i >= 2}
    space = {'x2': (np.asarray, ''), 'x1': (np.asarray, ''), 'a3': (np.log10, 'log10_')}

    exts = set()
    for path in glob.glob(str(results / 'credible*.txt')):
        m = re.fullmatch(r'credible(\d+(?:\.\d+)?)(_\w+)\.txt', os.path.basename(path))
        interval, ext = float(m.group(1)), m.group(2)
        exts.add(ext)
        n_used = int(np.genfromtxt(results / 'Histograms' / ('x1%s.txt' % ext),
                                   ndmin=2)[:, 2].sum())
        assert 1 < n_used <= data.shape[0]
        with open(path) as f:
            rows = {line.split('\t')[0]: tuple(float(x) for x in line.split('\t')[1:])
                    for line in f.read().splitlines()[1:]}
        assert set(rows) == set(by_name)
        for name, col in by_name.items():
            assert rows[name] == tuple(pytest.approx(b) for b in
                                       _order_statistics(col[:n_used], interval)), (path, name)
            hist_path = results / 'Histograms' / ('%s%s.txt' % (name, ext))
            with open(hist_path) as f:
                assert f.readline().startswith('# %slower_bound' % space[name][1]), hist_path
            h = np.genfromtxt(hist_path, ndmin=2)
            own = space[name][0](col[:n_used])
            assert h[0, 0] == pytest.approx(own.min()), hist_path
            assert h[-1, 1] == pytest.approx(own.max()), hist_path
    return exts


@pytest.mark.parametrize('job_type', [
    'mh', 'pt', 'am', 'dream', 'p_dream',
    pytest.param('hmc', marks=pytest.mark.skipif(
        not _HAS_JAX, reason='needs the optional jax extra (pip install pybnf[jax])')),
])
def test_every_sampler_writes_every_interval_from_the_named_column(tmp_path, monkeypatch, job_type):
    """#856 names six job types and two in-run call sites; the reproduction test above
    runs mh and reads only ``_final``. Here every sampler runs with the parameters
    declared x2, x1, a3, and EVERY credible file it leaves -- the strided ``_<iter>``
    writes as well as ``_final`` -- is checked against samples.txt by
    :func:`_check_every_credible_file`.
    """
    _, results = _run_order_conf(
        tmp_path, monkeypatch, _ORDER_CONF + 'job_type = %s\n' % job_type + _PER_SAMPLER[job_type])
    exts = _check_every_credible_file(results)
    assert '_final' in exts
    if job_type != 'hmc':
        assert len(exts) > 1, 'no strided write was checked: %s' % sorted(exts)


# --------------------------------------------------------------------------- #
# am after its stop, through the paths main() actually takes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('stop', ['max_iterations', 'rhat'])
def test_am_failure_in_the_end_of_run_tail_leaves_end_files_through_finalize(
        tmp_path, monkeypatch, stop):
    """The error-exit path main() takes, not a direct cleanup() call: am stops, a step
    of run()'s end-of-run tail raises (here the best-fit BNGL write), and
    ``pybnf._finalize(success=False, ...)`` runs the cleanup. _finalize logs and
    swallows any exception from the cleanup, so the only visible symptom of a broken
    cleanup is a missing ``_end`` file -- which is what this checks.

    am used to point samples_file at combined_params.txt on both of its stop paths.
    The histogram step cannot read that file, so the cleanup failed and wrote nothing.

    Oracle: each parameter's own samples.txt column (_check_every_credible_file), and
    the ``_final`` files written at the stop, byte for byte."""
    from pybnf import pybnf as P
    conf = _ORDER_CONF + 'job_type = am\n' + _PER_SAMPLER['am']
    if stop == 'rhat':
        # A threshold every R-hat meets, checked every 50 iterations: the run stops on
        # convergence at the first check after burn_in, well before max_iterations.
        conf = conf.replace('diagnostics_every = 1000000000\n', 'diagnostics_every = 50\n')
        conf += 'rhat_threshold = 100\n'
    alg = _build_order_alg(tmp_path, monkeypatch, conf)
    results = tmp_path / 'out' / 'Results'

    def fail(self, *args, **kwargs):
        raise RuntimeError('injected failure after the stop')
    monkeypatch.setattr(algorithms.Algorithm, '_emit_best_fit_bngl', fail)
    monkeypatch.setattr(P, 'clear_sim_registry', lambda: None)
    with pytest.raises(RuntimeError, match='injected failure after the stop'):
        H.drive(alg)
    # The failure came after the stop: the stop's _final files are already there.
    assert (results / 'credible68.0_final.txt').exists()
    assert not glob.glob(str(results / 'credible*_end.txt'))
    assert os.path.samefile(alg.samples_file, results / 'samples.txt')
    n_rows = np.genfromtxt(results / 'samples.txt', skip_header=1, ndmin=2).shape[0]
    full_run = 2 * (200 - 50)        # population_size * (max_iterations - burn_in)
    assert n_rows < full_run if stop == 'rhat' else n_rows == full_run

    with pytest.raises(SystemExit) as exited:
        P._finalize(False, alg, 0.0)
    assert exited.value.code == 1

    assert {'_final', '_end'} <= _check_every_credible_file(results)
    _assert_end_equals_final(results)


def test_am_finished_run_resumed_with_more_iterations_keeps_reading_samples_txt(
        tmp_path, monkeypatch):
    """``pybnf -r N`` on a finished am run: the algorithm comes back from
    alg_finished.bp, runs N more iterations, and stops again. The pickle was taken
    before the stop, so it points at samples.txt; after the second stop the file must
    still be samples.txt, every credible file must match it, and an error exit after
    the second stop must still write ``_end`` files equal to the new ``_final`` ones."""
    from pybnf import pybnf as P
    conf = (_ORDER_CONF + 'job_type = am\n' + _PER_SAMPLER['am']).replace(
        'backup_every = 1000000000\n', 'backup_every = 5\n')
    alg = _build_order_alg(tmp_path, monkeypatch, conf)
    H.drive(alg)
    results = tmp_path / 'out' / 'Results'
    finished = tmp_path / 'out' / 'alg_finished.bp'
    assert finished.exists()
    rows_first = np.genfromtxt(results / 'samples.txt', skip_header=1, ndmin=2).shape[0]

    resumed, pending, _ = P._load_resumed_algorithm(finished, SimpleNamespace(resume=40))
    assert os.path.samefile(resumed.samples_file, results / 'samples.txt')
    assert resumed.max_iterations == 240
    resumed.run(H.FakeClient(), resume=pending)

    assert os.path.samefile(resumed.samples_file, results / 'samples.txt')
    rows_second = np.genfromtxt(results / 'samples.txt', skip_header=1, ndmin=2).shape[0]
    assert rows_second >= rows_first + 2 * 40      # both chains ran the 40 extra iterations
    assert '_final' in _check_every_credible_file(results)
    n_final = int(np.genfromtxt(results / 'Histograms' / 'x1_final.txt', ndmin=2)[:, 2].sum())
    assert n_final == rows_second                  # the new _final read the whole file

    monkeypatch.setattr(P, 'clear_sim_registry', lambda: None)
    with pytest.raises(SystemExit):
        P._finalize(False, resumed, 0.0)
    _assert_end_equals_final(results)
