"""Verify the HMC tutorial lessons (37 benchmark geometries, 38 analytical ODE).

These lessons are architecturally distinct from the rest of the tutorial suite:
they run ``job_type = hmc`` (blackjax NUTS) on an **analytical** target -- a
built-in menu distribution (37) or a closed-form ODE likelihood written as
``objective = expression`` (38). There is no BNGL/SBML model, no simulator, and
no BNG2.pl network generation, so they do NOT go through the bngsim recovery
harness (``tests/test_tutorial_examples.py``) or the model-driven manifest.
Instead each committed conf is parsed, the sampler is built from PyBNF's real
dispatch, and the posterior is checked against the truth the data was generated
from (lesson 38) or the known target geometry (lesson 37).

The truths below mirror ``examples/tutorial/38_hmc_analytical_ode/regenerate_fixtures.py``.

Opt-in: needs the ``jax`` extra (``pip install pybnf[jax]``) and is ``slow`` (NUTS
warmup + draws in-process). Run with::

    pytest tests/test_tutorial_hmc.py -m slow
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from . import integration_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_HAS_JAX = all(importlib.util.find_spec(m) is not None for m in ('jax', 'blackjax'))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_JAX,
                       reason='needs the optional jax extra (pip install pybnf[jax])'),
]

_TUT = Path(__file__).resolve().parents[1] / 'examples' / 'tutorial'


def _sample(folder, conf_name, tmp_path):
    """Parse a committed HMC conf (paths are relative to its folder), build the
    sampler via PyBNF's real dispatch, run it in-process, and return the pooled
    posterior draws as a {param_name: 1-D array} plus the total sample count."""
    path = _TUT / folder
    text = (path / conf_name).read_text()
    home = os.getcwd()
    os.chdir(path)                                  # `data = *.exp` resolves here at config load
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        raw['verbosity'] = 0
        conf = config.Configuration(raw)
    finally:
        os.chdir(home)

    alg = FIT_TYPE_REGISTRY['hmc'].cls(conf)
    H.drive(alg)                                    # makedirs + alg.run(FakeClient()), in-process

    res = Path(conf.config['output_dir']) / 'Results' / 'samples.txt'
    header = res.read_text().splitlines()[0].lstrip('#').split()
    col = {name: i for i, name in enumerate(header)}
    data = np.genfromtxt(res, skip_header=1)
    params = [p for p in header if p not in ('Name', 'Ln_probability')]
    return {p: data[:, col[p]] for p in params}, data.shape[0]


# --------------------------------------------------------------------------- #
# Lesson 38 -- closed-form ODE solution as the HMC likelihood
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('conf, recover, tol', [
    ('viral_decay_hmc.conf', {'V0': 100.0, 'c': 0.60}, 0.06),
    ('damped_oscillator_hmc.conf', {'C': 5.0, 'a': 0.35, 'w': 3.0}, 0.08),
], ids=['viral_decay', 'damped_oscillator'])
def test_analytical_ode_posterior_recovers_truth(conf, recover, tol, tmp_path):
    """The NUTS posterior mean of an analytical-ODE likelihood recovers the
    parameters the synthetic data was generated from."""
    post, n = _sample('38_hmc_analytical_ode', conf, tmp_path)
    assert n > 0
    for p, true in recover.items():
        rel = abs(post[p].mean() - true) / abs(true)
        assert rel < tol, (f'{conf}: {p} posterior mean {post[p].mean():g}, '
                           f'expected ~{true:g} ({rel * 100:.1f}% off > {tol * 100:.0f}%)')


# --------------------------------------------------------------------------- #
# Lesson 37 -- built-in benchmark geometries
# --------------------------------------------------------------------------- #
def test_gaussian_baseline_recovers_moments(tmp_path):
    """The easy round target: NUTS recovers N(0, 1) marginals in each coordinate."""
    post, _ = _sample('37_hmc_benchmark_geometry', 'gaussian.conf', tmp_path)
    for p in ('g1', 'g2'):
        assert abs(post[p].mean()) < 0.15, f'{p} mean {post[p].mean():g} not ~0'
        assert abs(post[p].std() - 1.0) < 0.2, f'{p} sd {post[p].std():g} not ~1'


def test_banana_samples_the_curved_valley(tmp_path):
    """The hard curved target samples without error and concentrates on the
    Rosenbrock valley near (x1, x2) ~ (1, 1.5)."""
    post, n = _sample('37_hmc_benchmark_geometry', 'banana.conf', tmp_path)
    assert n > 0
    assert 0.6 < post['x1'].mean() < 1.6, f"x1 mean {post['x1'].mean():g} off the valley"
    assert post['x2'].mean() > 0.5, f"x2 mean {post['x2'].mean():g} not up the valley"
