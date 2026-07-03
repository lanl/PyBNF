"""MCMC-sampler lesson (``examples/tutorial/26_mcmc_samplers/``).

The two Markov-chain Monte Carlo samplers PyBNF ships besides DREAM (lesson 17):
Metropolis-Hastings (``job_type = mh``) and Parallel Tempering (``job_type = pt``).
Both sample the same well-identified Bateman posterior and -- unlike Adaptive_MCMC,
whose histogram step is a no-op -- write real credible intervals. As in lesson 17
the assertion is the robust one: the 95% credible interval brackets the known truth.

Driven through the faked-dask recovery harness (bngsim real, dask faked, seed
pinned), inline and deterministic, but ``slow``-marked -- each sampler is ~1000
generations of a chain population::

    pytest tests/test_tutorial_mcmc.py -m slow
"""
import glob
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '26_mcmc_samplers'

pytestmark = [pytest.mark.bngsim, pytest.mark.slow]


def _load_conf(conf_name, tmp_path):
    text = (_LESSON / conf_name).read_text()
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        raw['random_seed'] = 1234
        raw['verbosity'] = 0
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _read_credible(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name, lo, hi = line.split('\t')
        out[name] = (float(lo), float(hi))
    return out


@pytest.mark.parametrize('conf_name, job_type', [
    ('mh_posterior.conf', 'mh'),
    ('pt_posterior.conf', 'pt'),
])
def test_sampler_credible_interval_brackets_truth(conf_name, job_type, tmp_path, monkeypatch):
    """Each MCMC sampler writes a 95% credible interval that brackets the known-true
    (k1, k2)."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    conf = _load_conf(conf_name, tmp_path)
    assert conf.config['fit_type'] == job_type
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        alg = FIT_TYPE_REGISTRY[job_type].cls(conf)
    finally:
        os.chdir(home)
    H.drive(alg)

    results = Path(conf.config['output_dir']) / 'Results'
    matches = sorted(glob.glob(str(results / 'credible95*_final.txt')))
    assert matches, f'{conf_name}: no 95% credible-interval file under {results}'

    cred = _read_credible(matches[0])
    for p, truth in (('k1', 0.8), ('k2', 0.25)):
        assert p in cred, f'{conf_name}: {p} missing from credible intervals: {cred}'
        lo, hi = cred[p]
        assert lo < truth < hi, (
            f'{conf_name}: 95% credible interval for {p} is [{lo:g}, {hi:g}], '
            f'which does not bracket the true {truth}')
