"""Bayesian uncertainty lesson (``examples/tutorial/17_bayesian_uncertainty/``).

A posterior sampler (``job_type = dream``, DiffeRential Evolution Adaptive
Metropolis) is run on the Bateman chain and asked for credible intervals. Unlike
the point-optimizer lessons, the payoff is a *distribution*: the 95% credible
interval each parameter's posterior draws fall in.

The sampler is driven through the same faked-dask recovery harness as the other
tutorial fits (bngsim simulation real, dask faked, ``random_seed`` pinned), so the
run is deterministic and inline -- no CLI subprocess. It is still ``slow``-marked:
a DREAM population over ~1500 generations is many more ODE solves than a quick
optimizer fit.

The assertion is deliberately robust: the model is well-identified and the data
sits exactly at the truth (the ``_SD`` column only sets the likelihood width), so
the posterior is centred on the truth and its 95% credible interval brackets it.
Bracketing is a weak, stable property -- it does not require the sampler to nail a
precise interval width, only that the chains found and explored the right region
(which the run's R-hat -> 1 confirms)::

    pytest tests/test_tutorial_bayesian.py -m slow
"""
import glob
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '17_bayesian_uncertainty'

pytestmark = [pytest.mark.bngsim, pytest.mark.slow]


def _load_conf(lesson, conf_name, tmp_path):
    """Load a committed conf with its output dir redirected to the test's tmp dir.
    Paths in the conf are relative to the lesson folder, so parse from inside it."""
    text = (lesson / conf_name).read_text()
    home = os.getcwd()
    os.chdir(lesson)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
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


def test_posterior_credible_interval_brackets_truth(tmp_path, monkeypatch):
    """Running the Bayesian conf yields a 95% credible interval that brackets the
    known-true (k1, k2) for both rates."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    conf = _load_conf(_LESSON, 'bateman_posterior.conf', tmp_path)
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        alg = FIT_TYPE_REGISTRY['dream'].cls(conf)
    finally:
        os.chdir(home)
    H.drive(alg)

    results = Path(conf.config['output_dir']) / 'Results'
    matches = sorted(glob.glob(str(results / 'credible95*_final.txt')))
    assert matches, f'no 95% credible-interval file written under {results}'

    cred = _read_credible(matches[0])
    for p, truth in (('k1', 0.8), ('k2', 0.25)):
        assert p in cred, f'{p} missing from credible intervals: {cred}'
        lo, hi = cred[p]
        assert lo < truth < hi, (
            f'95% credible interval for {p} is [{lo:g}, {hi:g}], '
            f'which does not bracket the true {truth}')
