"""A transcendental function of a strided column must not reach PyBNF's output files.

numpy 1.26 on an AVX-512 CPU computes ``10.0 ** x``, ``exp``, ``log10`` and the like for a
strided view through Intel's SVML, or through the C library, depending on where malloc puts the
result. Its overlap check takes a strided input to run ``stride * length`` bytes, past the end
of the array it is a column of, so a result placed right after that array looks like an overlap.
The two paths disagree in the last bit for about a quarter of the values. Measured on GitHub's
AVX-512 runners (Xeon 8370C and 8573C, EPYC 9V45 and 9V74), a (30, 4) array's columns:
``10.0 ** u[:, j]`` differed in 16618 of 60000 values between a result placed right after ``u``
and one placed elsewhere, ``np.exp(u[:, j])`` in 14033, and a contiguous copy of the column in
none. numpy 2.5.4 on the same CPUs differed in none, and so did an EPYC 7763, which has no
AVX-512.

So, on the dependency-floors CI leg, the Liu–West filter's ``_theta`` made a run's files depend
on memory addresses. Two runs of the same job in one process wrote different ``weight_ess.txt``,
and a continuation did not write what the uninterrupted run writes, though the documentation
promises both (``docs/algorithms.rst``, "Random streams, the state file and continuation").

This machine may not have such a CPU. The tests therefore stand in for the log scales, and for
the filter module's numpy, with versions whose result for a strided input depends on how many
calls came before it, the way the real one depends on the allocator's history. A contiguous
input is computed exactly. Each test checks the promise itself, and also lists every strided
call it saw, so a failure names the call site.
"""

import os
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.priors import scale as scale_module

from . import lwf_harness as C
from . import recovery_harness as H
from .context import algorithms, pset
from .test_lwf_state import _continued, _files, _run

#: numpy functions the filter calls that SVML can compute.
TRANSCENDENTAL = ('exp', 'log', 'log10', 'log1p', 'expm1', 'power', 'sqrt')


class _HistoryDependent:
    """Stand in for the log scales and the filter's numpy: a strided array input gives a result
    whose last bit depends on the number of strided calls before it."""

    def __init__(self, monkeypatch):
        self.strided = []
        for cls in (scale_module.Log10, scale_module.Ln):
            for name in ('forward', 'inverse'):
                monkeypatch.setattr(cls, name, self._method(getattr(cls, name),
                                                            '%s.%s' % (cls.__name__, name)))
        monkeypatch.setattr(lw, 'np', _NumpyProxy(self))

    def nudge(self, label, arrays, out):
        """``out``, with alternate elements moved by one ulp when an input is strided."""
        if not any(isinstance(a, np.ndarray) and a.ndim and not a.flags.c_contiguous
                   for a in arrays):
            return out
        caller = next(frame for frame in reversed(traceback.extract_stack()[:-2])
                      if not frame.filename.endswith(('test_strided_transcendentals.py', 'pset.py',
                                                      'scale.py')))
        self.strided.append('%s from %s:%d' % (label, os.path.basename(caller.filename),
                                               caller.lineno))
        out = np.array(out, dtype=float)
        moved = (np.arange(out.size) + len(self.strided)) % 2 == 0
        out.reshape(-1)[moved] = np.nextafter(out.reshape(-1)[moved], np.inf)
        return out

    def _method(self, original, label):
        stand_in = self

        def method(scale, x):
            return stand_in.nudge(label, [x], original(scale, x))
        return method


class _NumpyProxy:
    def __init__(self, stand_in):
        self._stand_in = stand_in

    def __getattr__(self, name):
        attr = getattr(np, name)
        if name not in TRANSCENDENTAL:
            return attr
        stand_in = self._stand_in

        def call(*args, **kwargs):
            return stand_in.nudge('np.' + name, args, attr(*args, **kwargs))
        return call


def test_the_stand_in_moves_a_strided_result_and_leaves_a_contiguous_one(monkeypatch):
    """The stand-in itself: without it the tests below would prove nothing."""
    stand_in = _HistoryDependent(monkeypatch)
    u = np.linspace(-1.0, 1.0, 120).reshape(30, 4)
    exact = 10.0 ** np.ascontiguousarray(u[:, 2])
    assert np.array_equal(scale_module.LOG10.inverse(np.ascontiguousarray(u[:, 2])), exact)
    assert stand_in.strided == []
    moved = scale_module.LOG10.inverse(u[:, 2])
    assert not np.array_equal(moved, exact)
    assert np.all(np.abs(moved - exact) <= np.spacing(exact))
    assert len(stand_in.strided) == 1


@pytest.mark.bngsim
@pytest.mark.parametrize('runs, j', [(1, 2), (1, 12), (2, 3)],
                         ids=['one_run_J2', 'one_run_J12', 'two_runs_J3'])
def test_lwf_writes_the_same_bytes_whatever_a_strided_transcendental_returns(
        tmp_path, monkeypatch, runs, j):
    """The uninterrupted run is the oracle (as in ``test_lwf_state``): the parts run and its
    continuation must write its files byte for byte. ``one_run_J12`` continues past the last
    row, so it compares two whole runs of the same job, the other promise."""
    H.require_bng2pl()
    stand_in = _HistoryDependent(monkeypatch)
    whole = _run(tmp_path / 'whole', runs=runs)
    _run(tmp_path / 'parts', runs=runs, counts=C.COUNTS[:j])
    continued, _ = _continued(tmp_path / 'parts', runs=runs)
    assert stand_in.strided == [], (
        'the filter computed a transcendental of a strided array: %s'
        % sorted(set(stand_in.strided)))
    assert _files(continued) == _files(whole)


def test_sampler_histograms_do_not_depend_on_what_came_before(tmp_path, monkeypatch):
    """``update_histograms`` bins each log parameter in log10. Two calls on the same samples,
    with different call histories, must write the same bin edges."""
    out = str(tmp_path)
    os.makedirs(os.path.join(out, 'Results', 'Histograms'))
    ba = object.__new__(algorithms.BayesianAlgorithm)
    ba.samples_file = os.path.join(out, 'Results', 'samples.txt')
    ba.variables = [pset.FreeParameter('k%d' % i, 'loguniform_var', 0.01, 100.0, 1.0)
                    for i in range(3)]
    ba.credible_intervals = [68]
    ba.num_bins = 10
    ba.config = SimpleNamespace(config={'output_dir': out})
    rows = 10.0 ** np.random.default_rng(11).uniform(-2.0, 2.0, (40, 3))
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tk0\tk1\tk2\n')
        for i, row in enumerate(rows):
            f.write('s%d\t-1.0\t%s\n' % (i, '\t'.join('%.17g' % x for x in row)))
    stand_in = _HistoryDependent(monkeypatch)

    def histograms():
        ba.update_histograms('_a')
        folder = os.path.join(out, 'Results', 'Histograms')
        files = {}
        for name in sorted(os.listdir(folder)):
            with open(os.path.join(folder, name), 'rb') as f:
                files[name] = f.read()
        return files

    first = histograms()      # without the fix its strided calls change the next call's result
    second = histograms()
    assert stand_in.strided == [], (
        'update_histograms computed a transcendental of a strided array: %s'
        % sorted(set(stand_in.strided)))
    assert first == second
    # The bin edges are those of log10 of the parameter's own samples, binned by hand here.
    edges = np.histogram(np.log10(np.ascontiguousarray(rows[:, 1])), bins=10)[1]
    lines = first['k1_a.txt'].decode().splitlines()[1:]
    assert np.array_equal([float(line.split()[0]) for line in lines], edges[:-1])
