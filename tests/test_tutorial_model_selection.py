"""Model-selection lesson (``examples/tutorial/45_model_selection/``).

When you don't know which growth LAW your data follows, you fit several candidate
models to the same data and rank them. Four growth laws (logistic / gompertz /
richards / von_bertalanffy) are fit to one noisy Richards curve, each scored with
the same weighted least-squares objective (``noise_model = normal, sigma =
read_exp_file _SD``) so the best chi-square is comparable across models. Ranking by
the Akaike Information Criterion (``AIC = chi_square + 2*k``) picks the true model:
Richards fits the asymmetric curve so much better that its extra shape parameter is
more than paid for.

This cross-model comparison can't be expressed as a per-conf ConfCheck, so it lives
here: Richards has the lowest AIC (and recovers its true ``(r, K, b)``), and every
other candidate is a worse fit.

Driven inline through the faked-dask recovery harness (bngsim real, dask faked,
seed pinned)::

    pytest tests/test_tutorial_model_selection.py -m recovery
"""
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '45_model_selection'

pytestmark = [pytest.mark.bngsim, pytest.mark.recovery]

# candidate conf -> number of fitted parameters (k, for AIC)
_CANDIDATES = {
    'fit_logistic.conf': 2,
    'fit_gompertz.conf': 2,
    'fit_richards.conf': 3,
    'fit_von_bertalanffy.conf': 2,
}
_TRUTH = 'fit_richards.conf'


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
    return alg


def test_aic_selects_the_true_growth_law(tmp_path, monkeypatch):
    """Fitting four growth laws to the Richards data, the AIC ranking picks the
    Richards LAW (the model structure the data came from), decisively."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    aic = {}
    chi2s = {}
    for conf_name, k in _CANDIDATES.items():
        alg = _fit(conf_name, tmp_path)
        chi2 = alg.trajectory.best_score()
        chi2s[conf_name] = chi2
        aic[conf_name] = chi2 + 2 * k

    winner = min(aic, key=aic.get)
    assert winner == _TRUTH, (
        f'AIC selected {winner}, not the true model {_TRUTH}. AIC ranking: '
        + ', '.join(f'{c}={aic[c]:.1f}' for c in sorted(aic, key=aic.get)))

    # Richards is not just the lowest -- it is clearly better, so the *fit* (not the
    # parameter-count tie-break) is what decides. It fits the asymmetric curve well
    # (chi-square on the order of the point count); the others are the wrong shape and
    # cannot match it, so their extra parsimony cannot save them.
    runner_up = min((c for c in aic if c != _TRUTH), key=aic.get)
    assert aic[_TRUTH] < aic[runner_up] - 10, (
        f'Richards AIC {aic[_TRUTH]:.1f} is not decisively below the runner-up '
        f'{runner_up} {aic[runner_up]:.1f}')
    assert chi2s[_TRUTH] < 0.5 * chi2s[runner_up], (
        f'Richards chi-square {chi2s[_TRUTH]:.1f} is not clearly better than the '
        f'runner-up {runner_up} {chi2s[runner_up]:.1f}')
