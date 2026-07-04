"""Adaptive-Metropolis + formal MCMC diagnostics lesson
(``examples/tutorial/39_adaptive_mcmc/``).

Lesson 26 showed the two *non-adaptive* samplers (mh, pt). This lesson runs the
*adaptive* one -- Adaptive Metropolis (``job_type = am``) -- on the two-species
harmonic oscillator's correlated (k4, k6) posterior, and reads formal convergence
diagnostics (R-hat / ESS) out of it via **ArviZ**.

The mechanic being verified is the one the lesson teaches: ``am`` records its draws
to per-chain ``Results/A_MCMC/Runs/params_*.txt`` files (not a single
``samples.txt``), and PyBNF's ArviZ bridge (``pybnf.inference_data.from_pybnf``)
reads those per-chain files directly -- reshaping them to ``(chain, draw, param)``
-- so ``from_pybnf`` returns a usable ``InferenceData`` for an ``am`` run, which is
exactly what this test loads its diagnostics from.

Driven through the same faked-dask recovery harness as lesson 26 (bngsim real,
dask faked, seed pinned): inline and deterministic, but ``slow``-marked. The data
is zero-noise and the seed is fixed, so R-hat / recovery are stable run to run::

    BNGPATH=... pytest tests/test_tutorial_am_diagnostics.py -m slow
"""
import os
from pathlib import Path

import numpy as np
import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.inference_data import from_pybnf
from pybnf.registry import FIT_TYPE_REGISTRY

az = pytest.importorskip('arviz')   # the diagnostics beat needs the arviz extra

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '39_adaptive_mcmc'

pytestmark = [pytest.mark.bngsim, pytest.mark.slow]

TRUTH = {'k4': 41.77, 'k6': 92.2}
NAMES = ['k4', 'k6']


def _load_conf(conf_name, tmp_path):
    text = (_LESSON / conf_name).read_text()
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        raw['verbosity'] = 0
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _posterior_array(idata, names):
    """The InferenceData posterior stacked back into a ``(chain, draw, param)``
    array for the recovery / correlation checks. ``from_pybnf`` already reshaped
    ``am``'s per-chain ``params_*.txt`` files into this ``chain`` x ``draw``
    posterior (truncated to the shortest chain), so this just orders the columns."""
    return np.stack([idata.posterior[n].values for n in names], axis=-1)


def test_am_posterior_diagnostics(tmp_path, monkeypatch):
    """am recovers the correlated (k4, k6) posterior, and the InferenceData
    from_pybnf builds from its per-chain draws reports healthy R-hat and ESS."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    conf = _load_conf('adaptive_covariance.conf', tmp_path)
    assert conf.config['fit_type'] == 'am'
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        alg = FIT_TYPE_REGISTRY['am'].cls(conf)
    finally:
        os.chdir(home)
    H.drive(alg)

    # --- the §E mechanic: the ArviZ bridge reads am's per-chain params_*.txt files
    #     directly (am writes those, not samples.txt) and returns a usable InferenceData.
    idata = from_pybnf(conf.config['output_dir'])   # finds A_MCMC/Runs/params_*.txt
    arr = _posterior_array(idata, NAMES)             # (chain, draw, param)
    assert arr.shape[0] >= 4, f'expected >=4 chains, got shape {arr.shape}'
    # am records only draws (no per-draw log-posterior), so there is no sample_stats.
    groups = idata.groups() if callable(idata.groups) else idata.groups
    assert not any(str(g).rstrip('/').endswith('sample_stats') for g in groups)
    # az.summary is the human-readable diagnostics table the README shows. In
    # arviz 1.2 its r_hat/mean/sd columns are STRING-formatted, so we assert on the
    # numeric az.rhat / az.ess functions instead of parsing the display table.
    az.summary(idata, var_names=NAMES)
    rhat = az.rhat(idata)
    ess = az.ess(idata)                       # bulk ESS by default
    rhat_vals = {n: float(rhat[n].values) for n in NAMES}
    ess_vals = {n: float(ess[n].values) for n in NAMES}

    # R-hat / ESS: convergence must be healthy (chains mixed and agree). The run is
    # deterministic (fixed seed, zero-noise data); these bounds sit well clear of the
    # observed R-hat ~1.05 / bulk ESS ~110 so minor numerical drift can't flake them.
    assert max(rhat_vals.values()) < 1.1, f'R-hat did not converge: {rhat_vals}'
    assert min(ess_vals.values()) > 80, f'bulk ESS too low: {ess_vals}'

    # Recovery: the pooled posterior mean lands on the known truth (zero-noise data).
    pooled = arr.reshape(-1, len(NAMES))
    for i, name in enumerate(NAMES):
        mean = float(pooled[:, i].mean())
        assert abs(mean - TRUTH[name]) / TRUTH[name] < 0.05, (
            f'{name}: posterior mean {mean:g} is not within 5% of truth {TRUTH[name]}')

    # The posterior is genuinely CORRELATED -- the tilt am's covariance adapts to.
    corr = float(np.corrcoef(pooled.T)[0, 1])
    assert corr > 0.4, f'expected a correlated (k4,k6) posterior, got corr={corr:g}'
