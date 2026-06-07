"""Recovery tier (opt-in ``-m recovery``): synthetic-data parameter recovery for
``m01_exp_decay`` through the **real bngsim backend**.

The model is first-order decay ``S' = -k*S`` with ``S(0)=10``, so
``S(t) = 10*exp(-k*t)``. We simulate it at a known-true ``k`` to generate a
zero-noise ``.exp`` (the oracle), then a real DE fit must recover ``k``.

Per the orchestration-testing skill, the one end-to-end fit is decomposed into
separately-named decisions so a failure points at the right layer:

  * oracle well-posedness  -- the generated data matches the analytic solution
    (no optimizer involved);
  * data reproduced        -- the fit drives the objective to ~0 (hard gate);
  * parameter recovered    -- the identifiable rate constant comes back (soft
    gate), checked across two seeds so it can't pass by a lucky one;
  * reproducibility        -- a fixed seed gives a bit-identical best fit.

Needs bngsim (auto-skipped via the ``bngsim`` marker) and BNG2.pl for the
one-time network generation (``recovery_harness.require_bng2pl`` skips otherwise).
"""
import numpy as np
import pytest

from . import recovery_harness as H


pytestmark = [pytest.mark.recovery, pytest.mark.bngsim]

M01 = H.RECOVERY_MODELS_DIR / 'm01_exp_decay.bngl'
K_TRUE = 0.3
S_INIT = 10.0
FREE = {'k__FREE': (1e-3, 5.0)}     # uniform prior bracketing the truth
OBS = ['Obs_Tot_S']
SUFFIX = 'ode'                       # the model's simulate-action suffix
DE_BUDGET = dict(population_size=16, max_iterations=50)


@pytest.fixture(scope='module')
def m01_exp(tmp_path_factory):
    """Generate the zero-noise synthetic ``.exp`` once for the module."""
    H.require_bng2pl()
    gen_dir = tmp_path_factory.mktemp('m01_gen')
    return H.simulate_truth(gen_dir, M01, {'k__FREE': K_TRUE}, FREE, OBS, SUFFIX)


@pytest.fixture
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _read_exp(path):
    """Return ``(cols, arr)`` for a ``.exp`` file: a name->index map and the
    numeric data (the ``#`` header line is skipped by genfromtxt as a comment)."""
    with open(path) as f:
        header = f.readline().lstrip('#').split()
    arr = np.genfromtxt(path)
    return {name: i for i, name in enumerate(header)}, arr


def _fit_de(tmp_path, exp_path, seed):
    conf = H.make_config(tmp_path, M01, exp_path, FREE, 'de',
                         random_seed=seed, **DE_BUDGET)
    alg = H.build(conf, 'de')
    H.drive(alg)
    return alg


# --------------------------------------------------------------------------- #
# Oracle well-posedness (no optimizer)
# --------------------------------------------------------------------------- #
def test_m01_synthetic_data_matches_analytic(m01_exp):
    """The generated data is the known analytic decay ``10*exp(-0.3 t)`` -- an
    independent oracle that the fit has a reachable global optimum at the truth.
    Validates the bngsim simulation + the data-generation/.exp-writing path with
    no optimizer in the loop."""
    cols, arr = _read_exp(m01_exp)
    t = arr[:, cols['time']]
    s = arr[:, cols['Obs_Tot_S']]
    expected = S_INIT * np.exp(-K_TRUE * t)
    # bngsim ODE matches the closed form to ~1e-4 (see test_bngsim_bngl_e2e);
    # 1e-3 leaves margin without masking a real regression.
    np.testing.assert_allclose(s, expected, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Recovery (hard gate + soft gate), across two seeds
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('seed', [1234, 7])
def test_m01_de_recovers(tmp_path, m01_exp, seed):
    """A real DE fit through bngsim reproduces the data and recovers ``k``."""
    alg = _fit_de(tmp_path, m01_exp, seed)

    # Hard gate: the loop drove the objective to ~0 (zero-noise data is exactly
    # reproducible; total data magnitude ~220, so <0.05 is a <5e-4 relative miss).
    best_score = alg.trajectory.best_score()
    assert best_score < 0.05, 'best objective %g did not floor near 0' % best_score

    # Soft gate: the identifiable rate constant comes back within tolerance.
    k = H.best_params(alg, ['k__FREE'])['k__FREE']
    assert abs(k - K_TRUE) / K_TRUE < 0.15, 'recovered k=%g, expected ~%g' % (k, K_TRUE)


# --------------------------------------------------------------------------- #
# Determinism (guards the RNG-migration contract on the real-sim path)
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures('_fakes')
def test_m01_de_reproducible(tmp_path, m01_exp):
    """A fixed seed yields a bit-identical best fit (synchronous fake dask +
    deterministic bngsim ODE + per-algorithm seeded RNG)."""
    k1 = H.best_params(_fit_de(tmp_path / 'a', m01_exp, 99), ['k__FREE'])['k__FREE']
    k2 = H.best_params(_fit_de(tmp_path / 'b', m01_exp, 99), ['k__FREE'])['k__FREE']
    assert k1 == k2, 'fixed seed gave different best k: %r vs %r' % (k1, k2)
