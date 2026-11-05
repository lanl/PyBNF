"""Review tests for the strided-transcendental fix, written by a reviewer who did not write it.

They differ from ``test_strided_transcendentals`` in three ways.

* **The stand-in.** ``_AddressLike`` wraps numpy's own module attributes, so it sees ``np.exp``
  and the like called from any module (the noise model, the objective, scipy's Python layer),
  not only the filter module's ``np``, and also checks ``out=``. Each strided call moves a
  pseudo-random half of its elements by one ulp, keyed on a process-wide call count, so two
  strided calls agree only by chance. The author's stand-in keys on the parity of the count, a
  one-bit history: two runs whose call counts differ by an even number get the same nudges, and
  their files compare equal even when a strided call reaches them. With the fix reverted, its
  ``two_runs_J3`` case fails only through its list of strided calls, not through the files.
* **The oracle.** Values are checked element by element against Python's own ``10.0 ** x`` and
  ``math.exp`` / ``math.log10`` / ``math.log``, within a few ulps, in C, Fortran and
  non-contiguous layouts, and the layouts must agree with each other to the bit. On an AVX-512
  CPU with numpy 1.26 that agreement is itself a check on the real hardware.
* **The configurations.** A natural-log parameter, unbounded log10 and natural-log priors, a
  truncated normal, ``lwf_bounds = reflect`` with ``initialization = lh``, and a configuration of
  ``uniform_var`` lines only, whose one transcendental is the logit of the kernel's working
  space.

What the stand-in cannot see: the ``**`` operator on an array outside the scale classes (a C
slot), a ufunc's methods (``np.log.reduce``), compiled code (``scipy.special``, the
Generator's draws, bngsim), and a strided array inside a list or other container.
"""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.priors import scale as scale_module
from pybnf.pset import FreeParameter

from . import lwf_harness as C
from . import recovery_harness as H
from .context import algorithms
from .test_lwf_state import _continued, _files, _run

#: numpy's float64 transcendentals: the ones numpy 1.26 sends to SVML or its own AVX-512 code.
TRANSCENDENTAL = ('exp', 'log', 'log10', 'log2', 'log1p', 'expm1', 'exp2', 'power',
                  'float_power', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
                  'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'cbrt', 'logaddexp',
                  'logaddexp2')


def _strided(a):
    """An array whose ufunc inner loop can be strided: neither C- nor F-contiguous."""
    return (isinstance(a, np.ndarray) and a.ndim > 0 and a.size > 1
            and not (a.flags.c_contiguous or a.flags.f_contiguous))


class _AddressLike:
    """numpy 1.26 on an AVX-512 CPU, as far as a test can tell: a strided input gives a result
    whose last bits differ from call to call; a contiguous one is computed exactly."""

    def __init__(self, monkeypatch):
        self.strided = []
        for name in TRANSCENDENTAL:
            monkeypatch.setattr(np, name, self._wrap('np.' + name, getattr(np, name)))
        for cls in (scale_module.Log10, scale_module.Ln):
            for name in ('forward', 'inverse'):
                monkeypatch.setattr(cls, name, self._method(getattr(cls, name),
                                                            '%s.%s' % (cls.__name__, name)))

    def nudge(self, label, arrays, out):
        if not any(_strided(a) for a in arrays):
            return out
        self.strided.append(label)
        out = np.array(out, dtype=float)
        flat = out.reshape(-1)
        rng = np.random.default_rng(len(self.strided))
        up = rng.random(flat.size) < 0.5
        flat[up] = np.nextafter(flat[up], np.inf)
        return out

    def _wrap(self, label, ufunc):
        stand_in = self

        class Wrapped:
            def __call__(self, *args, **kwargs):
                o = kwargs.get('out')
                arrays = list(args) + (list(o) if isinstance(o, tuple) else [o])
                result = ufunc(*args, **kwargs)
                if o is not None:   # a strided out= is as address-dependent as an input
                    if any(_strided(a) for a in arrays):
                        stand_in.strided.append(label + ' (out=)')
                    return result
                return stand_in.nudge(label, arrays, result)

            def __getattr__(self, name):
                return getattr(ufunc, name)

        return Wrapped()

    def _method(self, original, label):
        stand_in = self

        def method(scale, x):
            return stand_in.nudge(label, [x], original(scale, x))
        return method


def test_the_stand_in_sees_numpy_called_from_another_module(monkeypatch):
    """Without this the module-level wrapping would prove nothing: a strided np.log10 from
    outside the filter is seen and moved, a contiguous one is exact, and two strided calls on
    the same column differ."""
    stand_in = _AddressLike(monkeypatch)
    u = np.linspace(0.5, 3.0, 120).reshape(30, 4)
    # numpy's own log10 may differ from the C library's by an ulp or so (SVML on an AVX-512
    # CPU), so "exact" here means within a few ulps of math.log10.
    exact = np.array([math.log10(x) for x in u[:, 3]])
    contiguous = np.log10(np.ascontiguousarray(u[:, 3]))
    assert np.all(np.abs(contiguous - exact) <= 4 * np.spacing(np.abs(exact)))
    assert stand_in.strided == []
    first, second = np.log10(u[:, 3]), np.log10(u[:, 3])
    assert stand_in.strided == ['np.log10', 'np.log10']
    assert not np.array_equal(first, second)
    assert np.all(np.abs(first - exact) <= 5 * np.spacing(np.abs(exact)))


def _variables():
    return [FreeParameter('a', 'loguniform_var', 0.01, 100.0, 1.0),
            FreeParameter('b', 'lnuniform_var', 0.05, 20.0, 1.0),
            FreeParameter('c', 'uniform_var', 0.1, 0.5, 0.2),
            FreeParameter('d', 'lognormal_var', 1.0, 0.5),
            FreeParameter('e', 'lnnormal_var', 0.0, 2.0)]


def _shell(variables):
    """Enough of a filter for _theta: its variables and each parameter's box in theta."""
    alg = object.__new__(lw.LiuWestFilter)
    alg.variables = variables
    alg._theta_lo = np.array([v.lower_bound for v in variables], dtype=float)
    alg._theta_hi = np.array([v.upper_bound for v in variables], dtype=float)
    return alg


def _by_hand(v, x):
    """theta for one sampling-space value, in Python floats, clipped into the box."""
    x = float(x)
    theta = {'log10': lambda: 10.0 ** x, 'ln': lambda: math.exp(x), 'linear': lambda: x}[
        v.scale_name]()
    return min(max(theta, v.lower_bound), v.upper_bound)


def test_theta_is_pythons_own_pow_and_exp_in_every_layout(monkeypatch):
    """The mapping from sampling space is unchanged: within a few ulps of ``10.0 ** x`` and
    ``math.exp(x)`` element by element, clipped into the box, and the same bits whether the
    particle array is C-ordered (strided columns), Fortran-ordered (contiguous columns, as
    the prior draw's transpose is) or a non-contiguous slice of a larger array."""
    variables = _variables()
    alg = _shell(variables)
    rng = np.random.default_rng(3)
    # Inside and outside each box, so the clip is exercised too.
    centre = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    width = np.array([2.6, 3.4, 0.25, 1.5, 4.0])
    u = centre + width * rng.uniform(-1.0, 1.0, (41, 5))
    layouts = {'C': np.ascontiguousarray(u), 'F': np.asfortranarray(u)}
    bigger = np.zeros((82, 7))
    bigger[::2, 1:6] = u
    layouts['slice'] = bigger[::2, 1:6]
    assert not (layouts['slice'].flags.c_contiguous or layouts['slice'].flags.f_contiguous)

    expected = np.array([[_by_hand(v, x) for v, x in zip(variables, row)] for row in u])
    first = None
    for name, layout in layouts.items():
        theta = alg._theta(layout)
        # Within a few ulps: numpy's pow and exp may differ from the C library's by an ulp or
        # so on an AVX-512 CPU (SVML), which is not what this test is about.
        assert np.all(np.abs(theta - expected) <= 4 * np.spacing(np.abs(expected))), name
        if first is None:
            first = theta
        assert theta.tobytes(order='C') == first.tobytes(order='C'), name
    # The same, through the stand-in: nothing strided reaches a transcendental.
    stand_in = _AddressLike(monkeypatch)
    for name, layout in layouts.items():
        assert alg._theta(layout).tobytes(order='C') == first.tobytes(order='C'), name
    assert stand_in.strided == []


def test_histograms_bin_by_hand_in_each_parameters_own_space(tmp_path, monkeypatch):
    """``update_histograms`` with a log10, a natural-log and a linear parameter: the edges are
    those of ``math.log10`` / ``math.log`` / the value itself, binned by hand, the counts are
    the number of samples in each bin, each header names its space, and a second call with a
    different history writes the same bytes."""
    out = str(tmp_path)
    folder = os.path.join(out, 'Results', 'Histograms')
    os.makedirs(folder)
    ba = object.__new__(algorithms.BayesianAlgorithm)
    ba.samples_file = os.path.join(out, 'Results', 'samples.txt')
    ba.variables = [FreeParameter('k0', 'loguniform_var', 0.01, 100.0, 1.0),
                    FreeParameter('k1', 'lnuniform_var', 0.05, 20.0, 1.0),
                    FreeParameter('k2', 'uniform_var', 0.1, 0.5, 0.2)]
    ba.credible_intervals = [68]
    ba.num_bins = 7
    ba.config = SimpleNamespace(config={'output_dir': out})
    rng = np.random.default_rng(5)
    rows = np.column_stack([10.0 ** rng.uniform(-2.0, 2.0, 53),
                            np.exp(rng.uniform(math.log(0.05), math.log(20.0), 53)),
                            rng.uniform(0.1, 0.5, 53)])
    with open(ba.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tk0\tk1\tk2\n')
        for i, row in enumerate(rows):
            f.write('s%d\t-1.0\t%s\n' % (i, '\t'.join('%.17g' % x for x in row)))
    stand_in = _AddressLike(monkeypatch)

    def histograms():
        ba.update_histograms('_r')
        files = {}
        for name in sorted(os.listdir(folder)):
            with open(os.path.join(folder, name), 'rb') as f:
                files[name] = f.read()
        return files

    first = histograms()
    np.log10(rows[:, 0])        # a strided call between the two: a different history
    second = histograms()
    assert stand_in.strided == ['np.log10']
    assert first == second
    space = {'k0': (math.log10, 'log10_'), 'k1': (math.log, 'ln_'), 'k2': (float, '')}
    for j, (name, (forward, prefix)) in enumerate(space.items()):
        values = [forward(float(x)) for x in rows[:, j]]
        lo, hi = min(values), max(values)
        lines = first['%s_r.txt' % name].decode().splitlines()
        assert lines[0] == '# %slower_bound\t%supper_bound\tcount' % (prefix, prefix)
        table = np.array([[float(x) for x in line.split()] for line in lines[1:]])
        edges = [lo + i * (hi - lo) / ba.num_bins for i in range(ba.num_bins + 1)]
        tol = 4 * np.spacing(max(abs(lo), abs(hi)))
        assert np.all(np.abs(table[:, 0] - edges[:-1]) <= tol), name
        assert np.all(np.abs(table[:, 1] - edges[1:]) <= tol), name
        # Counts against the file's own edges (the last bin is closed on the right).
        counts = [sum(1 for x in values if a <= x < b or (i == ba.num_bins - 1 and x == b))
                  for i, (a, b) in enumerate(zip(table[:, 0], table[:, 1]))]
        assert table[:, 2].tolist() == counts, name
        assert sum(counts) == len(values)


PRIORS = (
    'uniform_var = gamma 0.1 0.5',
    'loguniform_var = beta 0.2 2',
    'parameter: rho, prior: normal, mean: 0.5, sd: 0.1, lower: 0.3, upper: 0.9',
    'lognormal_var = N 4 0.1',
    'parameter: I0, prior: normal, parameter_scale: ln, mean: 2.3, sd: 0.3',
    'loguniform_var = r 1 100',
)
_BASE_PRIORS = {'loguniform_var = beta': None, 'uniform_var = gamma': None,
                'loguniform_var = r': None, 'loguniform_var = I0': None}
CONFIGURATIONS = {
    'ln_lognormal_truncated_reflect_lh': dict(
        replace=_BASE_PRIORS,
        extra=list(PRIORS) + ['lwf_bounds = reflect', 'initialization = lh']),
    'uniform_var_only': dict(
        replace=dict(_BASE_PRIORS),
        extra=['uniform_var = beta 0.2 2', 'uniform_var = gamma 0.1 0.5',
               'uniform_var = r 1 100', 'uniform_var = I0 1 100']),
}


@pytest.mark.bngsim
@pytest.mark.parametrize('configuration', sorted(CONFIGURATIONS))
def test_lwf_files_do_not_depend_on_the_call_history_in_other_configurations(
        tmp_path, monkeypatch, configuration):
    """The uninterrupted run is the oracle: a second run of the same job in the same process,
    and a run of 4 rows continued over the rest, write its files byte for byte."""
    H.require_bng2pl()
    kw = CONFIGURATIONS[configuration]
    stand_in = _AddressLike(monkeypatch)
    whole = _files(_run(tmp_path / 'whole', **kw))
    again = _files(_run(tmp_path / 'again', **kw))
    _run(tmp_path / 'parts', counts=C.COUNTS[:4], **kw)
    continued, _ = _continued(tmp_path / 'parts', **kw)
    assert stand_in.strided == [], sorted(set(stand_in.strided))
    assert again == whole
    assert _files(continued) == whole


def _theta_before_the_fix(self, u):
    theta = np.empty_like(u)
    for j, v in enumerate(self.variables):
        theta[:, j] = v.from_sampling_space(u[:, j])
    return np.clip(theta, self._theta_lo, self._theta_hi)


@pytest.mark.bngsim
def test_the_files_alone_catch_the_old_theta_where_a_parity_history_does_not(
        tmp_path, monkeypatch):
    """The stand-in's own check, at the level of the files: with the old ``_theta`` and two
    independent runs continued after 3 rows, the continuation's files differ from the
    uninterrupted run's. The author's parity stand-in passes this comparison."""
    H.require_bng2pl()
    monkeypatch.setattr(lw.LiuWestFilter, '_theta', _theta_before_the_fix)
    stand_in = _AddressLike(monkeypatch)
    whole = _files(_run(tmp_path / 'whole', runs=2))
    _run(tmp_path / 'parts', runs=2, counts=C.COUNTS[:3])
    continued, _ = _continued(tmp_path / 'parts', runs=2)
    assert stand_in.strided    # the old _theta's strided calls were seen
    assert _files(continued) != whole
