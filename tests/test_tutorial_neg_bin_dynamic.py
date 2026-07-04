"""Estimate-the-dispersion lesson (``examples/tutorial/41_estimate_dispersion/``).

Lessons 18/28 fit COUNT data with the negative binomial at a PINNED dispersion;
this one ESTIMATES the over-dispersion jointly with the dynamics via the dynamic
count likelihood::

    noise_model = neg_bin, dispersion = fit r_disp, location = mean

The over-dispersion is made identifiable by design (the lesson-18 hazard: fitting
dispersion + count SCALE is weakly identified): the population scale and the
recovery rate ``gamma`` are known/fixed, only the transmission rate ``beta`` is
fitted, and the counts are REPLICATED (6 observations per time point) so the
scatter -- and hence the dispersion -- is pinned.

The check needs a beta-tight, dispersion-loose split a single-tolerance manifest
``ConfCheck`` can't express, so it lives here: ``beta`` comes back tight, and
``r_disp`` within a generous window that brackets the true 25 (a dispersion is a
variance-of-variance estimate -- inherently noisier than a rate even with
replicates).

Driven inline through the faked-dask recovery harness (bngsim real, dask faked,
seed pinned)::

    pytest tests/test_tutorial_neg_bin_dynamic.py -m recovery
"""
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '41_estimate_dispersion'

pytestmark = [pytest.mark.bngsim, pytest.mark.recovery]

_BETA_TRUE = 1.2
_DISPERSION_TRUE = 25.0


def _fit(conf_name, tmp_path):
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
    if conf.config.get('refine'):
        H.refine(alg, conf)
    return alg.trajectory.best_fit()


def test_estimates_transmission_rate_and_dispersion(tmp_path, monkeypatch):
    """The dynamic neg_bin recovers beta tightly and estimates the over-dispersion
    r_disp within a generous window bracketing the true 25 from replicate counts."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    best = _fit('estimate_dispersion.conf', tmp_path)

    assert best['beta'] == pytest.approx(_BETA_TRUE, rel=0.05), (
        f'recovered beta={best["beta"]}, expected {_BETA_TRUE}')
    assert 'r_disp' in best.keys(), f'r_disp not among {best.keys()}'
    r = best['r_disp']
    assert 0.6 * _DISPERSION_TRUE <= r <= 1.5 * _DISPERSION_TRUE, (
        f'estimated dispersion r_disp={r:g} does not bracket the true '
        f'{_DISPERSION_TRUE} within the [0.6x, 1.5x] window')
