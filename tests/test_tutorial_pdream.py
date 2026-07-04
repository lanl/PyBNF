"""Preconditioned-DREAM lesson (``examples/tutorial/40_preconditioned_dream/``).

Preconditioned DREAM (``job_type = p_dream``) is DREAM(ZS) with its proposals
computed in a covariance-whitened parameter space -- the adaptive sampler for a
strongly TILTED posterior. Here the linearized Lotka-Volterra oscillator gives a
long, thin, anti-correlated ``(alpha, gamma)`` ridge (the frequency
``sqrt(alpha*gamma)`` is pinned, the rates trade off along it), which is exactly
the geometry preconditioning is built for.

Like ``dream``/``mh``/``pt`` -- and unlike ``am`` -- ``p_dream`` inherits the base
sampler's histogram step, so it writes real credible intervals; the assertion is
the robust one from lesson 26: the 95% credible interval brackets the known truth.
The test also pins the parser wiring for ``precondition_adapt`` (the one config
knob ``p_dream`` adds) and confirms the preconditioner actually activated.

Driven through the faked-dask recovery harness (bngsim real, dask faked, seed
pinned), inline and deterministic, ``slow``-marked (a ~1500-generation chain
population)::

    pytest tests/test_tutorial_pdream.py -m slow
"""
import glob
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '40_preconditioned_dream'

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


def test_pdream_credible_interval_brackets_truth(tmp_path, monkeypatch):
    """p_dream writes a 95% credible interval that brackets the known-true
    (alpha, gamma), its ``precondition_adapt`` knob parses, and the covariance
    preconditioner actually activates during the run."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    conf = _load_conf('preconditioned_dream.conf', tmp_path)
    assert conf.config['fit_type'] == 'p_dream'
    # The p_dream-only knob is wired through the parser + schema (it was long a
    # PDreamConfig field with no grammar entry -- a conf line for it used to fail
    # to parse).
    assert conf.config['precondition_adapt'] == 250

    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        alg = FIT_TYPE_REGISTRY['p_dream'].cls(conf)
    finally:
        os.chdir(home)
    H.drive(alg)

    # The whitened-proposal path was exercised (not a hollow dream alias).
    assert alg._preconditioned, 'p_dream never activated its covariance preconditioner'

    results = Path(conf.config['output_dir']) / 'Results'
    matches = sorted(glob.glob(str(results / 'credible95*_final.txt')))
    assert matches, f'no 95% credible-interval file under {results}'

    cred = _read_credible(matches[0])
    for p, truth in (('alpha', 1.2), ('gamma', 0.8)):
        assert p in cred, f'{p} missing from credible intervals: {cred}'
        lo, hi = cred[p]
        assert lo < truth < hi, (
            f'95% credible interval for {p} is [{lo:g}, {hi:g}], '
            f'which does not bracket the true {truth}')
