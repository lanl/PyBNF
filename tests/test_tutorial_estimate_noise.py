"""Estimating the measurement noise lesson (``examples/tutorial/36_estimate_noise/``).

The new-era noise-model surface lets a Gaussian's ``sigma`` come from three sources:
``fix_at 1`` (unweighted least squares), ``read_exp_file _SD`` (supplied error bars),
or ``fit <name>`` (a named nuisance the fit ESTIMATES). This lesson fits one noisy
decay through all three, on data with a true constant noise of 4, so the fit reports
both the rate and -- with ``fit`` -- the noise level it inferred.

The distinguishing check needs a k-tight, sigma-loose split that a single-tolerance
manifest ``ConfCheck`` can't express, so it lives here: all three confs recover
``k = 0.5`` tightly, and the ``fit`` conf additionally recovers its ``noise_level``
close to the true 4 (the ML estimate of a standard deviation is slightly biased
low, so a generous window that brackets the truth is the honest assertion).

Driven inline through the faked-dask recovery harness (bngsim real, dask faked,
seed pinned)::

    pytest tests/test_tutorial_estimate_noise.py -m recovery
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
_CONFS = ('fixed_sigma.conf', 'supplied_sigma.conf', 'estimated_sigma.conf')


def _fit(conf_name, tmp_path):
    """Load a committed conf (paths relative to the lesson folder), fit it inline, and
    return its best-fit PSet."""
    text = (_LESSON / conf_name).read_text()
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / conf_name.replace('.conf', ''))
        raw['random_seed'] = 1234
        raw['verbosity'] = 0
        conf = config.Configuration(raw)
    finally:
        os.chdir(home)
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    alg = FIT_TYPE_REGISTRY['de'].cls(conf)
    H.drive(alg)
    return alg.trajectory.best_fit()


def test_all_three_sigma_sources_recover_the_rate(tmp_path, monkeypatch):
    """A Gaussian noise model with sigma from fix_at (unweighted), read_exp_file
    (supplied) or fit (estimated) all recover the true rate on constant-noise data --
    the difference is what they say about the NOISE, not the rate."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real
    for conf_name in _CONFS:
        best = _fit(conf_name, tmp_path)
        assert best['k'] == pytest.approx(_K_TRUE, rel=0.05), (
            f'{conf_name}: recovered k={best["k"]}, expected {_K_TRUE}')


def test_fit_source_estimates_the_noise_level(tmp_path, monkeypatch):
    """`sigma = fit noise_level` recovers the rate AND estimates the constant noise
    level close to the true 4 (a generous window that brackets the truth -- the ML
    sigma estimate is slightly biased low)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    best = _fit('estimated_sigma.conf', tmp_path)
    assert best['k'] == pytest.approx(_K_TRUE, rel=0.05), (
        f'recovered k={best["k"]}, expected {_K_TRUE}')
    assert 'noise_level' in best.keys(), f'noise_level not among {best.keys()}'
    noise = best['noise_level']
    assert 0.75 * _SIGMA_TRUE <= noise <= 1.25 * _SIGMA_TRUE, (
        f'estimated noise_level={noise:g} does not bracket the true {_SIGMA_TRUE} '
        f'within 25%')
