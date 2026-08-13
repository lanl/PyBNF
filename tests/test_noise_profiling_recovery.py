"""Analytic noise profiling against a real fit (``noise_profiling = 1``, ADR-0108, #562).

The unit tier (``test_noise_profiling.py``) pins the closed form, the gate, and the gradient
against numeric oracles. This tier asks the question those cannot: does a *real* fit --
bngsim simulating a real BNGL model, a real optimizer proposing points -- land in the same
place with the scale profiled out as with it searched, and does it report the scale it
estimated?

Both runs use lesson 36's noisy decay (``examples/tutorial/36_estimate_noise/``), whose data
carries a true constant noise of 4 and a true rate of 0.5. The searched run is the committed
``estimated_sigma.conf`` verbatim; the profiled run is that same conf plus one key. The
envelope theorem says profiling does not move the joint optimum, so the two must agree on
the rate, on the noise level, and on the best objective — which is exactly what makes the
comparison a test rather than two independent recoveries.

Driven inline through the faked-dask recovery harness (bngsim real, dask faked, seed pinned)::

    pytest tests/test_noise_profiling_recovery.py -m recovery
"""
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '36_estimate_noise'

pytestmark = [pytest.mark.bngsim, pytest.mark.recovery]

_K_TRUE = 0.5
_SIGMA_TRUE = 4.0


def _fit(tmp_path, label, extra=None):
    """Fit lesson 36's ``estimated_sigma.conf`` (paths relative to the lesson folder) with
    ``extra`` config keys merged in, and return the finished algorithm."""
    text = (_LESSON / 'estimated_sigma.conf').read_text()
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / label)
        raw['random_seed'] = 1234
        raw['verbosity'] = 0
        raw.update(extra or {})
        conf = config.Configuration(raw)
    finally:
        os.chdir(home)
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    alg = FIT_TYPE_REGISTRY['de'].cls(conf)
    H.drive(alg)
    return alg


def _profiled_noise_file(alg):
    """``{name: value}`` parsed back out of ``Results/profiled_noise.txt`` -- the artifact a
    user reads, not the in-memory map, so the report itself is under test."""
    rows = (Path(alg.res_dir) / 'profiled_noise.txt').read_text().splitlines()
    return {line.split('\t')[0]: float(line.split('\t')[1])
            for line in rows if line and not line.startswith('#')}


def test_profiling_recovers_the_rate_and_reports_the_noise_it_estimated(tmp_path, monkeypatch):
    """The whole feature in one run: the scale leaves the search (no ``noise_level`` column in
    the best PSet), the rate still recovers, and the estimated scale is reported in its own
    file, close to the true noise of 4."""
    H.require_bng2pl()
    H.install(monkeypatch)

    alg = _fit(tmp_path, 'profiled', {'noise_profiling': 1})
    best = alg.trajectory.best_fit()

    assert 'noise_level' not in best.keys(), (
        'a profiled scale must not be a coordinate of the best PSet: %s' % (best.keys(),))
    assert best['k'] == pytest.approx(_K_TRUE, rel=0.05), (
        'recovered k=%s, expected %s' % (best['k'], _K_TRUE))

    reported = _profiled_noise_file(alg)
    assert set(reported) == {'noise_level'}
    # A generous window that brackets the truth: the ML estimate of a standard deviation is
    # slightly biased low, and the fitted rate absorbs a sliver of the scatter (the same
    # window the searched lesson-36 recovery uses).
    assert 2.5 < reported['noise_level'] < 6.0, reported


def test_profiling_lands_where_the_searched_fit_lands(tmp_path, monkeypatch):
    """The envelope theorem's empirical form: profiling removes a search dimension, it does
    not move the joint optimum. Same conf, one key apart -- so the rate and the noise level
    must agree, and the profiled run must not score *worse* (its scale is optimal at every
    point, where the searched run could only sample one).

    These are two independent stochastic searches, each stopped on its own criterion, so the
    agreement asserted is to a fraction of a percent -- the optimizers' own convergence noise,
    not the feature's. The *ordering* of the two scores is the exact claim, and it is asserted
    exactly."""
    H.require_bng2pl()
    H.install(monkeypatch)

    searched = _fit(tmp_path, 'searched')
    profiled = _fit(tmp_path, 'profiled', {'noise_profiling': 1})

    searched_best = searched.trajectory.best_fit()
    profiled_best = profiled.trajectory.best_fit()

    assert profiled_best['k'] == pytest.approx(searched_best['k'], rel=1e-2)
    assert _profiled_noise_file(profiled)['noise_level'] == pytest.approx(
        searched_best['noise_level'], rel=1e-2)
    # Never worse: every point the profiled run scored was scale-optimal, and the searched
    # run's own best sigma is one the profiled surface already accounts for.
    assert profiled.trajectory.best_score() <= searched.trajectory.best_score()


def test_the_searched_run_writes_no_profiled_noise_file(tmp_path, monkeypatch):
    """The default is an exact no-op down to the artifacts: a fit that profiles nothing must
    not gain a report about it."""
    H.require_bng2pl()
    H.install(monkeypatch)
    alg = _fit(tmp_path, 'searched_only')
    assert not (Path(alg.res_dir) / 'profiled_noise.txt').exists()


def test_k_is_unchanged_by_profiling(tmp_path, monkeypatch):
    """A profiled scale is still an estimated parameter, so ``information_criteria.txt``
    reports the same ``k`` (and, both runs having found the same optimum, the same AIC to
    within their convergence noise) either way -- otherwise the two runs would not be
    comparable, which is the one thing an information criterion is for. ``k`` and ``n`` are
    asserted exactly; the AIC to a tenth of a unit, against a value near 229."""
    H.require_bng2pl()
    H.install(monkeypatch)

    searched = _fit(tmp_path, 'searched')
    profiled = _fit(tmp_path, 'profiled', {'noise_profiling': 1})

    def report(alg):
        text = (Path(alg.res_dir) / 'information_criteria.txt').read_text()
        return {line.split('\t')[0]: line.split('\t')[1]
                for line in text.splitlines() if line and not line.startswith('#')}

    searched_ic, profiled_ic = report(searched), report(profiled)
    assert profiled_ic['k'] == searched_ic['k'] == '2'
    assert profiled_ic['n'] == searched_ic['n']
    assert float(profiled_ic['AIC']) == pytest.approx(float(searched_ic['AIC']), abs=0.1)
