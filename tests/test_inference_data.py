"""Tests for the ArviZ InferenceData bridge (pybnf.inference_data, ADR-0055, #438).

The bridge is a pure format mapping from a finished MCMC run's saved samples onto
an arviz.InferenceData, so the oracles here are exact: the posterior values are the
samples.txt columns (log parameters mapped to sampling space), the chain/draw shape
is recovered from the `iter<draw>run<chain>` names, and sample_stats.lp is the
Ln_probability column. arviz is an optional extra, so the whole module skips when
it is absent.
"""

import sys

import numpy as np
import pytest

az = pytest.importorskip('arviz')

from pybnf.inference_data import from_pybnf
from pybnf.pset import FreeParameter

pytestmark = pytest.mark.arviz


def _write_samples(path, param_names, rows):
    """Write a synthetic samples.txt. `rows` is a list of (name, lp, values)."""
    with open(path, 'w') as f:
        f.write('# Name\tLn_probability\t' + '\t'.join(param_names) + '\n')
        for name, lp, vals in rows:
            f.write('%s\t%r\t%s\n' % (name, lp, '\t'.join(repr(v) for v in vals)))


@pytest.fixture
def two_chain_samples(tmp_path):
    """Two chains x two draws, params (a linear, k log-scaled), known values."""
    results = tmp_path / 'Results'
    results.mkdir()
    rows = [
        ('iter1000run0', -5.0, [3.0, 0.1]),
        ('iter1100run0', -4.5, [3.2, 0.12]),
        ('iter1000run1', -6.0, [2.8, 0.09]),
        ('iter1100run1', -5.5, [2.9, 0.11]),
    ]
    _write_samples(results / 'samples.txt', ['a', 'k'], rows)
    return results, rows


def test_basic_shape_and_lp_no_config(two_chain_samples, caplog):
    """With no config/variables: natural-space posterior, correct chain/draw shape,
    lp wired from the Ln_probability column."""
    results, rows = two_chain_samples
    idata = from_pybnf(results)

    assert set(idata.posterior.data_vars) == {'a', 'k'}  # natural space -> bare names
    assert idata.posterior.sizes['chain'] == 2
    assert idata.posterior.sizes['draw'] == 2

    # Posterior values are the raw columns (no scale recovery -> natural space).
    np.testing.assert_allclose(idata.posterior['a'].values, [[3.0, 3.2], [2.8, 2.9]])
    np.testing.assert_allclose(idata.posterior['k'].values, [[0.1, 0.12], [0.09, 0.11]])
    # lp is the Ln_probability column, chain x draw.
    np.testing.assert_allclose(idata.sample_stats['lp'].values, [[-5.0, -4.5], [-6.0, -5.5]])


def test_sampling_space_with_variables(two_chain_samples):
    """A log parameter passed via `variables` is renamed `log10_<name>` and emitted
    in sampling space (log10 of the natural column); a linear one is untouched."""
    results, rows = two_chain_samples
    variables = [
        FreeParameter('a', 'uniform_var', 0, 10, value=3.0),
        FreeParameter('k', 'loguniform_var', 1e-3, 1e3, value=0.1),
    ]
    idata = from_pybnf(results, variables=variables)

    assert set(idata.posterior.data_vars) == {'a', 'log10_k'}
    np.testing.assert_allclose(idata.posterior['a'].values, [[3.0, 3.2], [2.8, 2.9]])
    np.testing.assert_allclose(idata.posterior['log10_k'].values,
                               np.log10([[0.1, 0.12], [0.09, 0.11]]))


def test_chain_count_from_run_indices(tmp_path):
    """The number of chains is the number of distinct run<c> indices."""
    results = tmp_path / 'Results'
    results.mkdir()
    rows = []
    for c in range(3):
        for d in range(4):
            rows.append(('iter%drun%d' % (1000 + 100 * d, c), -1.0 * c, [float(c) + d]))
    _write_samples(results / 'samples.txt', ['x'], rows)

    idata = from_pybnf(results)
    assert idata.posterior.sizes['chain'] == 3
    assert idata.posterior.sizes['draw'] == 4


def test_ragged_chains_truncated_to_min(tmp_path, caplog):
    """Unequal-length chains are truncated to the shortest for a rectangular array,
    keeping each chain's earliest draws (ordered by iteration)."""
    results = tmp_path / 'Results'
    results.mkdir()
    rows = [
        ('iter1000run0', -1.0, [1.0]),
        ('iter1100run0', -1.0, [2.0]),
        ('iter1200run0', -1.0, [3.0]),  # chain 0 has 3 draws
        ('iter1000run1', -1.0, [10.0]),
        ('iter1100run1', -1.0, [20.0]),  # chain 1 has 2 draws
    ]
    _write_samples(results / 'samples.txt', ['x'], rows)

    idata = from_pybnf(results)
    assert idata.posterior.sizes['draw'] == 2  # truncated to the shorter chain
    # Earliest two draws of chain 0 kept (iteration order), chain 1 intact.
    np.testing.assert_allclose(idata.posterior['x'].values, [[1.0, 2.0], [10.0, 20.0]])


def test_draws_ordered_by_iteration(tmp_path):
    """Rows are ordered into draws by their iteration number, not file order."""
    results = tmp_path / 'Results'
    results.mkdir()
    rows = [  # deliberately out of order in the file
        ('iter1200run0', -1.0, [3.0]),
        ('iter1000run0', -1.0, [1.0]),
        ('iter1100run0', -1.0, [2.0]),
    ]
    _write_samples(results / 'samples.txt', ['x'], rows)

    idata = from_pybnf(results)
    np.testing.assert_allclose(idata.posterior['x'].values, [[1.0, 2.0, 3.0]])


def test_source_path_variants(two_chain_samples, tmp_path):
    """source accepts a Results/ dir, the samples.txt file, and an output dir
    containing Results/."""
    results, _ = two_chain_samples
    from_pybnf(results)                       # Results/ dir
    from_pybnf(results / 'samples.txt')       # the file itself
    from_pybnf(tmp_path)                       # output dir -> Results/samples.txt


def test_diagnostics_copied_into_attrs(two_chain_samples):
    """PyBNF's own final R-hat/ESS (diagnostics.txt) ride along in attrs, so the
    object keeps the authoritative dense-chain numbers next to ArviZ's thinned ones."""
    results, _ = two_chain_samples
    with open(results / 'diagnostics.txt', 'w') as f:
        f.write('# iteration\ttotal_evaluations\trhat_a\tbulk_ess_a\ttail_ess_a\trhat_k\tbulk_ess_k\ttail_ess_k\n')
        f.write('1000\t40000\t1.01\t250.0\t300.0\t1.05\t120.0\t150.0\n')
        f.write('2000\t80000\t1.00\t500.0\t600.0\t1.02\t240.0\t280.0\n')

    idata = from_pybnf(results)
    attrs = idata.posterior.attrs
    assert attrs['inference_library'] == 'pybnf'
    assert attrs['pybnf_max_rhat'] == pytest.approx(1.02)       # max over params, last row
    assert attrs['pybnf_min_bulk_ess'] == pytest.approx(240.0)  # min over params, last row
    assert attrs['pybnf_min_tail_ess'] == pytest.approx(280.0)


def test_conf_autodiscovery_recovers_scale(two_chain_samples, monkeypatch):
    """The standalone path auto-discovers a .conf in Results/ and uses its variables
    to put log parameters in sampling space (here the load is stubbed; the wiring is
    what's under test)."""
    results, _ = two_chain_samples
    (results / 'run.conf').write_text('# placeholder conf\n')

    class _FakeConfig:
        variables = [
            FreeParameter('a', 'uniform_var', 0, 10, value=3.0),
            FreeParameter('k', 'loguniform_var', 1e-3, 1e3, value=0.1),
        ]

    monkeypatch.setattr('pybnf.parse.load_config', lambda path: _FakeConfig(), raising=True)
    idata = from_pybnf(results)
    assert set(idata.posterior.data_vars) == {'a', 'log10_k'}


def test_conf_load_failure_falls_back_to_natural(two_chain_samples, monkeypatch, caplog):
    """If a discovered conf cannot be loaded, the bridge degrades to natural space
    with a warning rather than raising -- an archived run still produces an object."""
    results, _ = two_chain_samples
    (results / 'run.conf').write_text('# unloadable conf\n')

    def _boom(path):
        raise RuntimeError('model file missing')

    monkeypatch.setattr('pybnf.parse.load_config', _boom, raising=True)
    idata = from_pybnf(results)
    assert set(idata.posterior.data_vars) == {'a', 'k'}  # natural-space fallback
    assert any('NATURAL space' in r.message for r in caplog.records)


def test_empty_samples_raises(tmp_path):
    """A header-only samples.txt (a run that recorded no post-burn-in draws) raises
    a clear error rather than producing an empty object."""
    results = tmp_path / 'Results'
    results.mkdir()
    _write_samples(results / 'samples.txt', ['x'], [])
    with pytest.raises(ValueError, match='No samples'):
        from_pybnf(results)


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_pybnf(tmp_path / 'does_not_exist')


def test_missing_arviz_raises_with_hint(two_chain_samples, monkeypatch):
    """When the optional arviz extra is absent, the bridge raises ImportError with an
    install hint -- before touching the filesystem."""
    results, _ = two_chain_samples
    monkeypatch.setitem(sys.modules, 'arviz', None)  # make `import arviz` fail
    with pytest.raises(ImportError, match=r'pybnf\[arviz\]'):
        from_pybnf(results)


class _StubConfig:
    def __init__(self, d):
        self.config = d


def _bayesian_stub(res_dir, output_flag, variables=None):
    """A BayesianAlgorithm instance with __init__ bypassed, carrying just the three
    attributes _emit_inference_data reads -- enough to exercise the run() hook's
    guard + delegation without standing up a full fit."""
    from pybnf.algorithms.samplers.adaptive_mcmc import Adaptive_MCMC
    algo = object.__new__(Adaptive_MCMC)
    algo.res_dir = str(res_dir)
    algo.variables = variables or []
    algo.config = _StubConfig({'output_inference_data': output_flag})
    return algo


def test_run_hook_emits_netcdf_when_enabled(two_chain_samples):
    """`output_inference_data = 1` makes the run-end hook write inference_data.nc."""
    pytest.importorskip('h5netcdf')
    results, _ = two_chain_samples
    variables = [
        FreeParameter('a', 'uniform_var', 0, 10, value=3.0),
        FreeParameter('k', 'loguniform_var', 1e-3, 1e3, value=0.1),
    ]
    algo = _bayesian_stub(results, 1, variables)
    algo._emit_inference_data()

    out = results / 'inference_data.nc'
    assert out.is_file()
    reloaded = az.from_netcdf(str(out))
    assert set(reloaded.posterior.data_vars) == {'a', 'log10_k'}  # live variables used


def test_run_hook_noop_when_disabled(two_chain_samples):
    """The hook is a no-op (no file) when the key is unset."""
    results, _ = two_chain_samples
    _bayesian_stub(results, 0)._emit_inference_data()
    assert not (results / 'inference_data.nc').exists()


def test_netcdf_roundtrip(two_chain_samples, tmp_path):
    """The object writes to netCDF and reloads with posterior + sample_stats intact
    (the auto-emit path's on-disk artifact)."""
    pytest.importorskip('h5netcdf')
    results, _ = two_chain_samples
    idata = from_pybnf(results)
    out = tmp_path / 'inference_data.nc'
    idata.to_netcdf(str(out))
    reloaded = az.from_netcdf(str(out))
    np.testing.assert_allclose(reloaded.posterior['a'].values, idata.posterior['a'].values)
    np.testing.assert_allclose(reloaded.sample_stats['lp'].values, idata.sample_stats['lp'].values)


# --------------------------------------------------------------------------- #
# The log_likelihood group: LOO/WAIC (ADR-0056, #438 item 4)
# --------------------------------------------------------------------------- #

def _group_names(idata):
    """Group names across both arviz lines (0.x InferenceData.groups() method;
    1.x DataTree.groups tuple of '/path' strings)."""
    g = idata.groups
    g = g() if callable(g) else g
    return {str(x).strip('/').split('/')[-1] for x in g}


def _write_loglik(path, obs_ids, rows):
    with open(path, 'w') as f:
        f.write('# ' + '\t'.join(obs_ids) + '\n')
        for r in rows:
            f.write('\t'.join('%.17g' % v for v in r) + '\n')


@pytest.fixture
def samples_with_loglik(tmp_path):
    """Two chains x two draws + a 3-observation sidecar whose rows are in the file order
    sample_pset writes (interleaved by draw then chain) -- so row i of the sidecar aligns
    with samples row i."""
    results = tmp_path / 'Results'
    results.mkdir()
    rows = [
        ('iter1000run0', -5.0, [3.0, 0.1]),
        ('iter1000run1', -6.0, [2.8, 0.09]),
        ('iter1100run0', -4.5, [3.2, 0.12]),
        ('iter1100run1', -5.5, [2.9, 0.11]),
    ]
    _write_samples(results / 'samples.txt', ['a', 'k'], rows)
    obs_ids = ['m/s/o@x=0', 'm/s/o@x=1', 'm/s/o@x=2']
    ll_rows = [[-1.0, -2.0, -3.0], [-1.1, -2.1, -3.1], [-1.2, -2.2, -3.2], [-1.3, -2.3, -3.3]]
    _write_loglik(results / 'log_likelihood.txt', obs_ids, ll_rows)
    return results, obs_ids, ll_rows


def test_loglik_group_present_with_labels(samples_with_loglik):
    """A sidecar yields a log_likelihood group: var 'y' over the labelled obs_id axis,
    dims chain x draw x obs (the shape az.loo / az.waic / az.compare consume)."""
    results, obs_ids, _ = samples_with_loglik
    idata = from_pybnf(results)
    assert 'log_likelihood' in _group_names(idata)
    ll = idata.log_likelihood['y']
    assert dict(ll.sizes) == {'chain': 2, 'draw': 2, 'obs_id': 3}
    assert [str(x) for x in idata.log_likelihood.coords['obs_id'].values] == obs_ids


def test_loo_and_compare_run_on_bridged_group(tmp_path):
    """az.loo / az.compare compute on the bridged group -- the actual #438 item 4
    payoff. Needs enough draws for PSIS tail diagnostics, so this builds a larger
    synthetic run than the structural fixtures (whose tiny shape is for alignment)."""
    rng = np.random.default_rng(0)

    def build(sub, loc):
        results = tmp_path / sub / 'Results'
        results.mkdir(parents=True)
        rows, ll_rows = [], []
        for it in range(60):
            for c in range(4):
                rows.append(('iter%drun%d' % (it, c), -float(rng.normal(50, 2)),
                             [float(rng.normal(1, 0.1)), float(rng.normal(2, 0.1))]))
                ll_rows.append(list(rng.normal(loc, 0.5, 5)))
        _write_samples(results / 'samples.txt', ['a', 'k'], rows)
        _write_loglik(results / 'log_likelihood.txt',
                      ['o%d' % i for i in range(5)], ll_rows)
        return from_pybnf(results)

    m1, m2 = build('m1', -2.0), build('m2', -2.4)
    az.loo(m1)                       # PSIS-LOO computes
    assert len(az.compare({'m1': m1, 'm2': m2})) == 2  # model comparison ranks both


def test_loglik_row_alignment(samples_with_loglik):
    """The sidecar's i-th row maps to samples' i-th row, then reshapes by the recovered
    (chain, draw). chain0/draw@iter1100 is sidecar row 2 -> [-1.2, -2.2, -3.2]."""
    results, _, ll_rows = samples_with_loglik
    y = from_pybnf(results).log_likelihood['y']
    # chain order = sorted run indices [0,1]; draw order = sorted iters [1000,1100].
    np.testing.assert_allclose(y.values[0, 0, :], ll_rows[0])  # run0 @ iter1000
    np.testing.assert_allclose(y.values[1, 0, :], ll_rows[1])  # run1 @ iter1000
    np.testing.assert_allclose(y.values[0, 1, :], ll_rows[2])  # run0 @ iter1100
    np.testing.assert_allclose(y.values[1, 1, :], ll_rows[3])  # run1 @ iter1100


def test_loglik_absent_when_no_sidecar(two_chain_samples):
    """No sidecar (no output_inference_data, or a non-likelihood objfunc) -> no group,
    so the bridge still works for every other run -- the group is purely additive."""
    results, _ = two_chain_samples
    assert 'log_likelihood' not in _group_names(from_pybnf(results))


def test_loglik_width_mismatch_omits_group(samples_with_loglik, caplog):
    """A corrupt sidecar row (wrong width) drops the whole group with a warning rather
    than building a misaligned array -- and the posterior is unaffected."""
    results, obs_ids, ll_rows = samples_with_loglik
    ll_rows[2] = [-1.2, -2.2]  # 2 values where 3 are expected
    _write_loglik(results / 'log_likelihood.txt', obs_ids, ll_rows)
    idata = from_pybnf(results)
    assert 'log_likelihood' not in _group_names(idata)
    assert 'omitting the log_likelihood group' in caplog.text
    assert dict(idata.posterior.sizes) == {'chain': 2, 'draw': 2}  # posterior still whole


def test_loglik_netcdf_roundtrip(samples_with_loglik, tmp_path):
    """The log_likelihood group survives the netCDF round-trip the auto-emit path uses."""
    pytest.importorskip('h5netcdf')
    results, _, _ = samples_with_loglik
    idata = from_pybnf(results)
    out = tmp_path / 'idata.nc'
    idata.to_netcdf(str(out))
    reloaded = az.from_netcdf(str(out))
    assert 'log_likelihood' in _group_names(reloaded)
    np.testing.assert_allclose(reloaded.log_likelihood['y'].values,
                               idata.log_likelihood['y'].values)


# --------------------------------------------------------------------------- #
# The sampler writes the sidecar (the record path that feeds the bridge above)
# --------------------------------------------------------------------------- #

def _mk_data(lines):
    from pybnf import data
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


class _LPParam:
    def __init__(self, name, value):
        self.name, self.value = name, value


class _StubPSet:
    def __init__(self, name, vals):
        self.name, self._vals = name, vals

    def values_to_string(self):
        return '\t'.join('%g' % v for v in self._vals)


class _StubRes:
    def __init__(self, simdata, pset):
        self.simdata, self.pset = simdata, pset


def test_sampler_writes_aligned_sidecar_that_bridges(tmp_path):
    """End-to-end of the record path: driving a real BayesianAlgorithm's accept-cache
    (record_pointwise_loglik) + sample write (sample_pset) over two chains writes
    log_likelihood.txt row-aligned with samples.txt, and from_pybnf turns it into a
    working log_likelihood group. Exercises the actual chi_sq objective + sampler I/O."""
    from pybnf.algorithms.samplers.adaptive_mcmc import Adaptive_MCMC
    from pybnf import objective

    results = tmp_path / 'Results'
    results.mkdir()
    exp = _mk_data(['# x  o  o_SD\n', ' 0 3 0.5\n', ' 1 5 0.5\n'])  # 2 observations

    algo = object.__new__(Adaptive_MCMC)
    algo.samples_file = str(results / 'samples.txt')
    algo.log_likelihood_file = str(results / 'log_likelihood.txt')
    algo.objective = objective.ChiSquareObjective()
    algo.exp_data = {'m': {'s': exp}}
    algo.all_constraints = []
    algo._record_loglik = True
    algo._loglik_ids = None
    algo._loglik_header_written = False
    algo.current_pointwise_loglik = [None, None]
    with open(algo.samples_file, 'w') as f:
        f.write('# Name\tLn_probability\tk\n')

    for it in (1000, 1100):
        for c in (0, 1):
            sim = _mk_data(['# x  o\n', ' 0 %g\n' % (3 + 0.1 * c), ' 1 %g\n' % (5 + 0.1 * it / 1000)])
            algo.record_pointwise_loglik(_StubRes({'m': {'s': sim}}, [_LPParam('k', 1.0)]), c)
            algo.sample_pset(_StubPSet('iter%drun%d' % (it, c), [1.0]), -5.0, c)

    # Both files carry one data row per sample, so they are row-aligned (4 rows each).
    n_samples = sum(1 for ln in open(algo.samples_file) if not ln.startswith('#'))
    n_loglik = sum(1 for ln in open(algo.log_likelihood_file) if not ln.startswith('#'))
    assert n_samples == n_loglik == 4

    idata = from_pybnf(results)
    assert 'log_likelihood' in _group_names(idata)
    assert dict(idata.log_likelihood['y'].sizes) == {'chain': 2, 'draw': 2, 'obs_id': 2}
    assert [str(x) for x in idata.log_likelihood.coords['obs_id'].values] == ['m/s/o@x=0', 'm/s/o@x=1']


def test_sampler_no_sidecar_when_objfunc_not_a_likelihood(tmp_path):
    """The no-op gate: with a non-likelihood objfunc (here sum-of-squares) the recorder
    never fires, so no sidecar is written and the bridge omits the group."""
    from pybnf.algorithms.samplers.adaptive_mcmc import Adaptive_MCMC
    from pybnf import objective

    results = tmp_path / 'Results'
    results.mkdir()
    algo = object.__new__(Adaptive_MCMC)
    algo.log_likelihood_file = str(results / 'log_likelihood.txt')
    algo.objective = objective.SumOfSquaresObjective()
    algo.exp_data = {'m': {'s': _mk_data(['# x  o\n', ' 0 3\n'])}}
    # _record_loglik is what __init__ computes; here a non-likelihood objfunc => False.
    algo._record_loglik = bool(0) and algo.objective.supports_pointwise_log_likelihood
    res = _StubRes({'m': {'s': _mk_data(['# x  o\n', ' 0 3\n'])}}, [_LPParam('k', 1.0)])
    algo.current_pointwise_loglik = [None]
    algo._loglik_ids = None
    algo.record_pointwise_loglik(res, 0)  # a no-op
    assert algo.current_pointwise_loglik[0] is None
    assert not (results / 'log_likelihood.txt').exists()
