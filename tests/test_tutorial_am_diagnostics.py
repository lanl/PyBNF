"""Adaptive-Metropolis + formal MCMC diagnostics lesson
(``examples/tutorial/39_adaptive_mcmc/``).

Lesson 26 showed the two *non-adaptive* samplers (mh, pt). This lesson runs the
*adaptive* one -- Adaptive Metropolis (``job_type = am``) -- on the two-species
harmonic oscillator's correlated (k4, k6) posterior, and reads formal convergence
diagnostics (R-hat / ESS) out of it via **ArviZ**.

The mechanic being verified is the one the lesson teaches: ``am`` does NOT write
``Results/samples.txt`` (its draws land in ``Results/A_MCMC/Runs/params_*.txt``),
so the automatic ArviZ bridge (``pybnf.inference_data.from_pybnf``) never finds
them. The diagnostics therefore have to build the ``InferenceData`` BY HAND from
the per-chain sample files -- reshaped to ``(chain, draw, param)`` and handed to
``az.from_dict`` -- which is exactly what this test does.

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


def _read_chains(output_dir, names):
    """am's per-chain post-burn-in draws as a ``(chain, draw, param)`` array.

    Reads ``Results/A_MCMC/Runs/params_*.txt`` (one file per chain; each is a
    header row of parameter names followed by whitespace-separated draws) and
    stacks the requested columns, truncating to the shortest chain so ArviZ gets a
    rectangular ``(chain, draw, param)`` block."""
    runs = Path(output_dir) / 'Results' / 'A_MCMC' / 'Runs'
    chains = []
    for fn in sorted(runs.glob('params_*.txt')):
        d = np.genfromtxt(fn, names=True)
        if d.size == 0:
            continue
        d = np.atleast_1d(d)
        chains.append(np.column_stack([d[n] for n in names]))
    assert chains, f'no per-chain params_*.txt found under {runs}'
    m = min(c.shape[0] for c in chains)
    return np.stack([c[:m] for c in chains], axis=0)


def test_am_posterior_diagnostics(tmp_path, monkeypatch):
    """am recovers the correlated (k4, k6) posterior, and the by-hand ArviZ
    InferenceData reports healthy R-hat and ESS."""
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

    # --- the §E mechanic: build an ArviZ InferenceData BY HAND from am's per-chain
    #     draws (am writes params_*.txt, NOT samples.txt, so the auto-bridge misses it).
    arr = _read_chains(conf.config['output_dir'], NAMES)   # (chain, draw, param)
    assert arr.shape[0] >= 4, f'expected >=4 chains, got shape {arr.shape}'
    idata = az.from_dict({'posterior': {NAMES[i]: arr[:, :, i]
                                        for i in range(len(NAMES))}})
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
