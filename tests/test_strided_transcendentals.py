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

import glob
import math
import os
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.priors import scale as scale_module

from . import integration_harness as IH
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
        # Which elements move depends on the whole call count, not only its parity, so two runs
        # that made different numbers of strided calls almost surely disagree somewhere.
        moved = np.random.default_rng(len(self.strided)).random(out.size) < 0.5
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
    # The nudge is one ulp; on an AVX-512 CPU with numpy 1.26 the real strided result may
    # itself be an ulp or so off the contiguous one, which is the defect being modelled.
    assert np.all(np.abs(moved - exact) <= 4 * np.spacing(exact))
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


def _am_run(tmp_path, monkeypatch, sub):
    """A short ``am`` run over two log-uniform parameters, past the step at which ``am``
    seeds its adaptive covariance from its own ``params_<i>.txt``."""
    IH.install(monkeypatch)
    target, data = IH.write_target(tmp_path, IH.gaussian_spec([1.0, 2.0], [1.0, 1.0]))
    conf = IH.make_config(tmp_path, 'am', target, data, 2, bounds=(0.01, 100.0),
                          var_type='loguniform_var', burn_in=80, sample_every=2,
                          rhat_threshold=0, output_hist_every=10 ** 9, hist_bins=10,
                          output_dir=str(tmp_path / sub), random_seed=4321,
                          population_size=3, max_iterations=240, adaptive=60, num_bins=10,
                          credible_intervals=[68], step_size=0.6)
    alg = algorithms.Adaptive_MCMC(conf)
    IH.drive(alg)
    return alg, conf.config['output_dir']


def test_am_draws_the_same_chain_from_the_same_seed(tmp_path, monkeypatch):
    """``random_seed`` promises the same draws. ``am`` seeds its adaptive covariance from the
    log of a column of its own parameter file; the second run reaches that step after a
    different number of strided calls, so a strided transform would draw another chain."""
    stand_in = _HistoryDependent(monkeypatch)
    _, first = _am_run(tmp_path, monkeypatch, 'a')
    _, second = _am_run(tmp_path, monkeypatch, 'b')
    assert stand_in.strided == [], (
        'am computed a transcendental of a strided array: %s' % sorted(set(stand_in.strided)))
    assert np.array_equal(IH.read_samples(first, 2), IH.read_samples(second, 2))


@pytest.mark.arviz
def test_inference_data_puts_the_same_draws_in_sampling_space_every_time(tmp_path, monkeypatch):
    """``from_pybnf`` takes log10 of each log parameter's draws, a slice of a (chain, draw,
    parameter) array. Two conversions of one run must give the same posterior."""
    from pybnf.inference_data import from_pybnf
    alg, out = _am_run(tmp_path, monkeypatch, 'a')
    stand_in = _HistoryDependent(monkeypatch)
    first = from_pybnf(os.path.join(out, 'Results'), variables=alg.variables)
    second = from_pybnf(os.path.join(out, 'Results'), variables=alg.variables)
    assert stand_in.strided == [], (
        'from_pybnf computed a transcendental of a strided array: %s'
        % sorted(set(stand_in.strided)))
    for name in first.posterior.data_vars:
        assert np.array_equal(first.posterior[name].values, second.posterior[name].values)
    # Oracle: the posterior holds log10 of the parameter values am wrote, taken here by hand
    # one value at a time.
    from pybnf.inference_data import _posterior_var_name
    runs = sorted(glob.glob(os.path.join(out, 'Results', 'A_MCMC', 'Runs', 'params_*.txt')))
    assert runs
    for v in alg.variables:
        written = [x for path in runs
                   for x in np.atleast_1d(np.genfromtxt(path, names=True)[v.name])]
        drawn = np.sort(first.posterior[_posterior_var_name(v.name, v)].values.ravel())
        by_hand = np.sort([math.log10(float(x)) for x in written])
        np.testing.assert_allclose(drawn, by_hand, rtol=4 * np.finfo(float).eps, atol=0)
