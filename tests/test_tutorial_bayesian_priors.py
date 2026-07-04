"""Priors-as-a-fitting-feature lesson (``examples/tutorial/27_priors/``).

Priors are applied ONLY inside a Bayesian sampler (``dream``/``mh``/``pt``); a point
optimizer maximizes the likelihood and ignores the prior family entirely. So the
lesson is Bayesian: the same Bateman posterior is sampled twice, changing nothing
but the prior on the weakly-identified rate ``k2``.

The data has two channels of different quality -- a precise ``Obs_A`` (``_SD`` 3,
which pins ``k1``) and an imprecise ``Obs_C`` (``_SD`` 25, the only, noisy, handle
on ``k2``). With a flat ``uniform_var`` prior ``k2``'s posterior is wide; swapping
in an informative ``gamma_var`` (mean 0.25, sd 0.05, from "independent knowledge")
collapses it onto a tight interval that still brackets the truth -- while ``k1``,
which the strong ``Obs_A`` already pins, is unaffected in both runs.

The assertions are the robust ones (bracketing + a width comparison, not a precise
interval): the informative 95% credible interval for ``k2`` is clearly NARROWER
than the flat one, both bracket the true ``k2``, and both runs bracket the true
``k1`` (strong data overrides any prior).

Driven through the faked-dask recovery harness (bngsim real, dask faked, seed
pinned), inline and deterministic, but ``slow``-marked -- two DREAM populations::

    pytest tests/test_tutorial_bayesian_priors.py -m slow
"""
import glob
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '27_priors'

pytestmark = [pytest.mark.bngsim, pytest.mark.slow]


def _load_conf(conf_name, tmp_path):
    """Load a committed conf with its output dir redirected under the test's tmp dir.
    Paths in the conf are relative to the lesson folder, so parse from inside it."""
    text = (_LESSON / conf_name).read_text()
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / conf_name.replace('.conf', ''))
        raw['random_seed'] = 1234
        raw['verbosity'] = 0
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _read_credible(path):
    """Parse a ``credibleNN_final.txt`` file into ``{param: (lower, upper)}``."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name, lo, hi = line.split('\t')
        out[name] = (float(lo), float(hi))
    return out


def _run(conf_name, tmp_path):
    """Sample one conf's posterior inline; return its 95% credible intervals."""
    conf = _load_conf(conf_name, tmp_path)
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        alg = FIT_TYPE_REGISTRY['dream'].cls(conf)
    finally:
        os.chdir(home)
    H.drive(alg)
    results = Path(conf.config['output_dir']) / 'Results'
    matches = sorted(glob.glob(str(results / 'credible95*_final.txt')))
    assert matches, f'{conf_name}: no 95% credible-interval file under {results}'
    return _read_credible(matches[0])


def test_informative_prior_narrows_weak_posterior(tmp_path, monkeypatch):
    """An informative gamma prior on the weakly-identified k2 yields a 95% credible
    interval clearly narrower than the flat prior's, both bracketing the truth, while
    the well-identified k1 brackets its truth in both runs (data overrides the prior)."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    flat = _run('flat_prior.conf', tmp_path)
    info = _run('informative_prior.conf', tmp_path)

    # k1 is pinned by the precise Obs_A in BOTH runs -- the prior on k2 doesn't move it.
    for tag, cred in (('flat', flat), ('informative', info)):
        assert 'k1' in cred, f'{tag}: k1 missing from {cred}'
        lo, hi = cred['k1']
        assert lo < 0.8 < hi, f'{tag}: k1 95% CI [{lo:g}, {hi:g}] does not bracket 0.8'

    # k2 is weakly identified: both intervals bracket the truth, but the informative
    # prior's is much narrower than the flat prior's.
    flat_lo, flat_hi = flat['k2']
    info_lo, info_hi = info['k2']
    assert flat_lo < 0.25 < flat_hi, (
        f'flat k2 95% CI [{flat_lo:g}, {flat_hi:g}] does not bracket 0.25')
    assert info_lo < 0.25 < info_hi, (
        f'informative k2 95% CI [{info_lo:g}, {info_hi:g}] does not bracket 0.25')

    flat_w = flat_hi - flat_lo
    info_w = info_hi - info_lo
    assert info_w < 0.75 * flat_w, (
        f'informative prior did not clearly narrow k2: flat width {flat_w:g} '
        f'([{flat_lo:g}, {flat_hi:g}]) vs informative width {info_w:g} '
        f'([{info_lo:g}, {info_hi:g}])')
