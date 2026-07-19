"""State-dependent (prediction-dependent) noise lesson
(``examples/tutorial/48_state_dependent_noise/``).

A combined additive+proportional error model: the measurement scatter is
``sigma = sd_abs + sd_rel * A``, an additive floor PLUS a term proportional to the
predicted signal. The new-era noise-model surface expresses this with the
``prediction_formula`` sigma source, whose expression references a model observable
and is evaluated against the *current simulation* at each scored point (ADR-0075) --
the noise-side peer of a measurement model. The fit estimates the rate ``k`` and both
noise coefficients (``sd_abs``, ``sd_rel``) jointly, with NO ``_SD`` column (sigma is a
function of the prediction, not a data value).

The distinguishing check needs a k-tight, coefficient-loose split a single-tolerance
manifest ``ConfCheck`` can't express (a combined error model's floor and slope are
weakly identified -- the high-signal early points pin ``sd_rel``, the low-signal tail
pins ``sd_abs``), so it lives here: the fit recovers ``k = 0.4`` tightly and brackets
each true coefficient within a generous window.

Driven inline through the faked-dask recovery harness (bngsim real, dask faked,
seed pinned)::

    pytest tests/test_tutorial_state_dependent_noise.py -m recovery
"""
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '48_state_dependent_noise'

pytestmark = [pytest.mark.bngsim, pytest.mark.recovery]

_K_TRUE = 0.4
_SD_ABS_TRUE = 5.0
_SD_REL_TRUE = 0.1
_CONF = 'state_dependent_noise.conf'


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


def test_prediction_formula_sigma_recovers_the_rate(tmp_path, monkeypatch):
    """A Gaussian noise model whose sigma is a ``prediction_formula`` over the simulated
    observable recovers the true rate from data with combined additive+proportional
    noise -- the state-dependent scale does not bias the rate."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real
    best = _fit(_CONF, tmp_path)
    assert best['k'] == pytest.approx(_K_TRUE, rel=0.10), (
        f'recovered k={best["k"]}, expected {_K_TRUE}')


def test_prediction_formula_sigma_estimates_both_noise_coefficients(tmp_path, monkeypatch):
    """`sigma = prediction_formula sd_abs + sd_rel*Obs_A` recovers the rate AND brackets
    both the additive floor and the proportional coefficient near their true values
    (generous windows -- a combined error model's two coefficients are weakly identified,
    so the honest assertion is that each brackets the truth, not a tight tolerance)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    best = _fit(_CONF, tmp_path)
    assert best['k'] == pytest.approx(_K_TRUE, rel=0.10), (
        f'recovered k={best["k"]}, expected {_K_TRUE}')
    for name in ('sd_abs', 'sd_rel'):
        assert name in best.keys(), f'{name} not among {best.keys()}'
    sd_abs, sd_rel = best['sd_abs'], best['sd_rel']
    # A factor-of-~2.5 window brackets each true coefficient (the floor and the slope are
    # each pinned by only one end of the signal range, so the estimates are noisy).
    assert 0.4 * _SD_ABS_TRUE <= sd_abs <= 2.5 * _SD_ABS_TRUE, (
        f'estimated sd_abs={sd_abs:g} does not bracket the true {_SD_ABS_TRUE}')
    assert 0.4 * _SD_REL_TRUE <= sd_rel <= 2.5 * _SD_REL_TRUE, (
        f'estimated sd_rel={sd_rel:g} does not bracket the true {_SD_REL_TRUE}')
