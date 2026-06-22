"""End-of-run best-fit BNGL artifact (ADR-0048).

``Algorithm._emit_best_fit_bngl`` writes a stable-named, family-labelled
``Results/<model>_bestfit.bngl`` for new-era (``edition >= 2``) runs, optionally
embedding each time-indexed observable's experimental data as a sidecar ``.tfun``
reference function. The hook was split out of ``run()``'s tail so it is unit
testable with no dask client (mirrors ``tests/test_run_loop.py``): build a bare
``Algorithm`` instance carrying only the attributes the method reads
(``config.models`` / ``config.config`` / ``res_dir`` / ``trajectory`` /
``exp_data``), call the method, and assert on the real files under ``tmp_path``.

The model fixture is ``bngl_files/e2e_ode_decay.bngl`` -- a new-era *bind-by-id*
model (params ``S0``/``k``, observable ``Stot``; ADR-0034), so the artifact also
exercises the bind-by-id rendering path. One ``@pytest.mark.bngsim`` test feeds
the generated ``.tfun`` to the real engine (auto-skipped when bngsim is absent).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from .context import algorithms, pset, data
from pybnf.pset import Trajectory, BNGLModel
from pybnf.algorithms.samplers.base import BayesianAlgorithm


DECAY = 'bngl_files/e2e_ode_decay.bngl'
ARTIFACT = 'e2e_ode_decay_bestfit.bngl'
TFUN_DIR = 'e2e_ode_decay_bestfit_tfun'


class _Opt(algorithms.Algorithm):
    """Concrete non-Bayesian Algorithm (start_run/got_result are abstract)."""
    def start_run(self):
        return []

    def got_result(self, res):
        return []


class _Bayes(BayesianAlgorithm):
    """Concrete Bayesian Algorithm -- only isinstance(...) matters for the label."""
    def start_run(self):
        return []

    def got_result(self, res):
        return []


def _decay_model():
    return BNGLModel(DECAY, suppress_free_param_error=True)


def _decay_pset(k=0.5, S0=42.0):
    return pset.PSet([pset.FreeParameter('k', 'uniform_var', 0, 10, value=k),
                      pset.FreeParameter('S0', 'uniform_var', 0, 200, value=S0)])


def _exp(headers, rows, indvar='time'):
    return data.Data.from_columns(np.array(rows, dtype=float), headers, indvar=indvar)


def _decay_exp(rows=((0, 100), (2, 55), (5, 22.3), (10, 5))):
    return {'e2e_ode_decay': {'tc': _exp(['time', 'Stot'], list(rows))}}


def _algo(tmp_path, *, edition=2, fit_type='de', embed=0, exp_data=None,
          bayesian=False, obj=12.5, models=None):
    """A bare Algorithm carrying only what _emit_best_fit_bngl reads."""
    res_dir = str(tmp_path / 'Results')
    os.makedirs(res_dir, exist_ok=True)
    algo = object.__new__(_Bayes if bayesian else _Opt)
    cfg = type('Cfg', (), {})()
    cfg.config = {'edition': edition, 'fit_type': fit_type, 'embed_best_fit_data': embed}
    cfg.models = {'e2e_ode_decay': _decay_model()} if models is None else models
    algo.config = cfg
    algo.res_dir = res_dir
    algo.exp_data = exp_data or {}
    best = _decay_pset()
    traj = Trajectory(100)
    traj.add(best, obj, 'iter5run0')
    algo.trajectory = traj
    return algo, best


def _artifact_text(algo):
    return (Path(algo.res_dir) / ARTIFACT).read_text()


# --------------------------------------------------------------------------- #
# Phase 1: the bare, labelled artifact
# --------------------------------------------------------------------------- #
class TestBareArtifact:

    def test_emits_stable_named_runnable_bngl(self, tmp_path):
        """A new-era run writes Results/<model>_bestfit.bngl with the fit values wired
        in (ADR-0034 bind-by-id) and re-parses back into the same parameter namespace."""
        algo, best = _algo(tmp_path)
        algo._emit_best_fit_bngl(best, 'iter5run0')

        out = Path(algo.res_dir) / ARTIFACT
        assert out.is_file()
        text = out.read_text()
        assert '\nk 0.5\n' in text and '\nS0 42.0\n' in text   # fit values applied
        assert '0.3' not in text and ' 100\n' not in text       # nominal values gone

        reparsed = BNGLModel(str(out), suppress_free_param_error=True)
        assert set(reparsed.model_param_names) == {'S0', 'k'}

    def test_optimizer_header_labels_best_fit(self, tmp_path):
        algo, best = _algo(tmp_path, fit_type='de', obj=3.25)
        algo._emit_best_fit_bngl(best, 'iter5run0')

        text = _artifact_text(algo)
        assert '# Point: BEST FIT (minimum objective).' in text
        assert '# fit_type: de' in text
        assert 'Objective (minimum recorded): 3.25' in text
        assert 'MAXIMUM-LIKELIHOOD' not in text and 'samples.txt' not in text

    def test_sampler_header_labels_max_likelihood_not_map(self, tmp_path):
        """A sampler's best_fit() is the MLE, not the MAP -- the header must say so and
        point at samples.txt for the posterior mode (the core ADR-0048 decision)."""
        algo, best = _algo(tmp_path, fit_type='am', bayesian=True)
        algo._emit_best_fit_bngl(best, 'iter5run0')

        text = _artifact_text(algo)
        assert 'MAXIMUM-LIKELIHOOD' in text
        assert 'NOT the MAP' in text
        assert 'Results/samples.txt' in text
        assert '# Point: BEST FIT' not in text

    @pytest.mark.parametrize('ed', [None, 1])
    def test_legacy_edition_emits_nothing(self, tmp_path, ed):
        """Edition-2-only: a legacy run is untouched (it keeps only its existing
        <model>_<best_name>.bngl, written elsewhere)."""
        algo, best = _algo(tmp_path, edition=ed)
        algo._emit_best_fit_bngl(best, 'iter5run0')
        assert not list(Path(algo.res_dir).glob('*_bestfit.bngl'))

    def test_no_best_pset_is_noop(self, tmp_path):
        algo, _ = _algo(tmp_path)
        algo._emit_best_fit_bngl(None, 'iter5run0')
        assert not list(Path(algo.res_dir).glob('*_bestfit.bngl'))

    def test_empty_trajectory_is_noop(self, tmp_path):
        """No successful evaluation -> empty trajectory -> nothing emitted, no crash."""
        algo, best = _algo(tmp_path)
        algo.trajectory = Trajectory(100)
        algo._emit_best_fit_bngl(best, 'iter5run0')
        assert not list(Path(algo.res_dir).glob('*_bestfit.bngl'))

    def test_non_bngl_model_skipped(self, tmp_path):
        """SBML/analytical models carry non-BNGL text and are out of scope."""
        not_bngl = type('NotBngl', (), {'name': 'sbmlmod'})()
        algo, best = _algo(tmp_path, models={'sbmlmod': not_bngl})
        algo._emit_best_fit_bngl(best, 'iter5run0')
        assert not list(Path(algo.res_dir).glob('*_bestfit.bngl'))


# --------------------------------------------------------------------------- #
# Phase 2: embedded experimental data as sidecar .tfun reference functions
# --------------------------------------------------------------------------- #
class TestDataEmbedding:

    def test_writes_sidecar_and_injects_function(self, tmp_path):
        algo, best = _algo(tmp_path, embed=1, exp_data=_decay_exp())
        algo._emit_best_fit_bngl(best, 'iter5run0')

        text = _artifact_text(algo)
        assert "expt_tc_Stot() = tfun('%s/tc__Stot.tfun', time)" % TFUN_DIR in text
        assert 'begin functions' in text and 'end functions' in text

        sidecar = Path(algo.res_dir) / TFUN_DIR / 'tc__Stot.tfun'
        assert sidecar.is_file()
        assert sidecar.read_text().splitlines()[0] == '# time expt_tc_Stot'

    def test_sidecar_data_roundtrips_strictly_increasing(self, tmp_path):
        algo, best = _algo(tmp_path, embed=1, exp_data=_decay_exp())
        algo._emit_best_fit_bngl(best, 'iter5run0')

        arr = np.loadtxt(Path(algo.res_dir) / TFUN_DIR / 'tc__Stot.tfun')
        np.testing.assert_array_equal(arr[:, 0], [0, 2, 5, 10])
        np.testing.assert_allclose(arr[:, 1], [100, 55, 22.3, 5])
        assert np.all(np.diff(arr[:, 0]) > 0)

    def test_sorts_and_dedupes_repeated_times(self, tmp_path):
        """Unsorted rows with a repeated time become a strictly-increasing index,
        keeping the first value seen at the repeated time (tfun's requirement)."""
        exp = {'e2e_ode_decay': {'tc': _exp(
            ['time', 'Stot'], [[5, 22], [0, 100], [2, 55], [2, 999], [10, 5]])}}
        algo, best = _algo(tmp_path, embed=1, exp_data=exp)
        algo._emit_best_fit_bngl(best, 'iter5run0')

        arr = np.loadtxt(Path(algo.res_dir) / TFUN_DIR / 'tc__Stot.tfun')
        np.testing.assert_array_equal(arr[:, 0], [0, 2, 5, 10])
        np.testing.assert_array_equal(arr[:, 1], [100, 55, 22, 5])  # first value at t=2

    def test_skips_indvar_and_sd_columns(self, tmp_path):
        exp = {'e2e_ode_decay': {'tc': _exp(
            ['time', 'Stot', 'Stot_SD'], [[0, 100, 1], [2, 55, 2], [5, 22, 3]])}}
        algo, best = _algo(tmp_path, embed=1, exp_data=exp)
        algo._emit_best_fit_bngl(best, 'iter5run0')

        names = sorted(p.name for p in (Path(algo.res_dir) / TFUN_DIR).iterdir())
        assert names == ['tc__Stot.tfun']  # no time, no Stot_SD

    def test_skips_non_time_experiment(self, tmp_path, caplog):
        """A parameter-scan experiment's indvar is a swept parameter, not time, so a
        tfun(file, time) would misrepresent it -- skip with a log note."""
        exp = {'e2e_ode_decay': {'dose': _exp(
            ['dose', 'Stot'], [[0, 1], [1, 2], [2, 3]], indvar='dose')}}
        algo, best = _algo(tmp_path, embed=1, exp_data=exp)
        with caplog.at_level(logging.INFO, logger='pybnf.algorithms'):
            algo._emit_best_fit_bngl(best, 'iter5run0')

        assert 'tfun(' not in _artifact_text(algo)
        assert not (Path(algo.res_dir) / TFUN_DIR).exists()
        assert 'non-time' in caplog.text.lower()

    def test_off_by_default(self, tmp_path):
        algo, best = _algo(tmp_path, embed=0, exp_data=_decay_exp())
        algo._emit_best_fit_bngl(best, 'iter5run0')

        assert 'tfun(' not in _artifact_text(algo)
        assert not (Path(algo.res_dir) / TFUN_DIR).exists()


# --------------------------------------------------------------------------- #
# Pure helpers (no filesystem, no engine)
# --------------------------------------------------------------------------- #
class TestPureHelpers:

    def test_inject_opens_standalone_block_before_end_model(self):
        t = 'begin model\nbegin parameters\nend parameters\nend model\nbegin actions\nend actions\n'
        out = algorithms.Algorithm._inject_function_lines(t, ["f() = tfun('a', time)"])
        assert "begin functions\nf() = tfun('a', time)\nend functions\n" in out
        assert out.index('begin functions') < out.index('end model')

    def test_inject_merges_into_existing_block(self):
        t = 'begin functions\n  g()=1\nend functions\n'
        out = algorithms.Algorithm._inject_function_lines(t, ["f() = tfun('a', time)"])
        assert out.count('begin functions') == 1 and out.count('end functions') == 1
        assert out.index('g()=1') < out.index('f() = tfun') < out.index('end functions')

    def test_inject_empty_is_noop(self):
        assert algorithms.Algorithm._inject_function_lines('x', []) == 'x'

    def test_clean_pairs_drops_nonfinite_sorts_dedupes(self):
        pairs = algorithms.Algorithm._clean_tfun_pairs(
            [0, 2, 2, 1, np.nan, 3], [10, 20, 99, 15, 5, np.inf])
        assert pairs == [(0., 10.), (1., 15.), (2., 20.)]

    def test_sanitize_id(self):
        assert algorithms.Algorithm._sanitize_id('a b/c') == 'a_b_c'
        assert algorithms.Algorithm._sanitize_id('1x') == '_1x'


# --------------------------------------------------------------------------- #
# Real engine: the generated .tfun loads into bngsim (auto-skipped if absent)
# --------------------------------------------------------------------------- #
@pytest.mark.bngsim
def test_generated_tfun_loads_into_real_bngsim_engine(tmp_path):
    """The exact sidecar this feature writes is accepted by the real bngsim table-
    function reader -- the embedded data round-trips into the engine (ADR-0048)."""
    from pybnf.bngsim_model import _runtime

    algo, best = _algo(tmp_path, embed=1, exp_data=_decay_exp())
    algo._emit_best_fit_bngl(best, 'iter5run0')
    sidecar = Path(algo.res_dir) / TFUN_DIR / 'tc__Stot.tfun'

    engine = _runtime.bngsim.Model.from_net('tests/bngl_files/e2e_ode_decay.net')
    engine.add_table_function('expt_tc_Stot', file=str(sidecar), index='time', method='linear')
