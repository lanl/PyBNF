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

The method reads only ``samples_file``, ``variables``, ``credible_intervals``,
``num_bins`` and the output directory, so it is exercised on bare ``object.__new__``
instances (following test_bayesian_diagnostics) rather than paying for the full
model-parsing constructor — these need no BNG2.pl.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from .context import algorithms, pset

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
