"""End-to-end BNGL execution through the real bngsim engine (issue #379, Phase 1).

These tests construct ``BngsimModel`` against committed ``.net`` fixtures and
call ``.execute()`` for real, then check numerical output. They complement the
fake-bngsim coverage in ``test_bngsim_bridge.py``, which only exercises
routing and the ``_execute_actions`` plumbing without running a simulator.

Fixtures live under ``tests/bngl_files/`` and were pre-generated once with
BNG2.pl (BioNetGen 2.9.3). Re-run BNG2.pl if a fixture's ``.bngl`` source
changes.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf import pset


pytestmark = pytest.mark.bngsim


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'


def _bngsim_bngl_model(net_path, actions):
    """Construct a BngsimModel against a committed .net fixture."""
    suffixes = []
    for a in actions:
        m = a
        idx = m.find('suffix=>"')
        if idx == -1:
            continue
        end = m.find('"', idx + len('suffix=>"'))
        suffixes.append(('simulate', m[idx + len('suffix=>"'):end]))
    return bngsim_model.BngsimModel(
        Path(net_path).stem,
        list(actions),
        suffixes,
        [],
        nf=str(net_path),
    )


# ---------------------------------------------------------------- ODE -----

def test_bngsim_bngl_ode_matches_analytic_decay(tmp_path):
    """Gap 1.1: BNGL ``method=>"ode"`` through real bngsim.

    ``S() -> 0`` at rate k with S0=100 has S(t) = S0*exp(-k*t). Compare every
    sampled point against the analytic value.
    """
    net_path = FIXTURES / 'e2e_ode_decay.net'
    actions = [
        'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>20,suffix=>"tc"})',
    ]
    model = _bngsim_bngl_model(net_path, actions)
    model.param_set = pset.PSet([])

    result = model.execute(str(tmp_path), 'ode_decay', 60)
    data = result['tc']
    times = data.data[:, data.cols['time']]
    stot = data.data[:, data.cols['Stot']]

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(10.0)
    expected = 100.0 * np.exp(-0.3 * times)
    np.testing.assert_allclose(stot, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- SSA -----

# Birth-death process: stationary distribution is Poisson(k_prod/k_deg). With
# k_prod=10, k_deg=1, the mean and variance at long time are both 10. We use
# the second half of the trajectory (after several time constants of burn-in)
# to estimate the stationary mean and variance from each replicate, and then
# pool across replicates so the statistical test scales with N_REPLICATES
# rather than with the (heavily autocorrelated) sample count per replicate.

SSA_REPLICATES = 200
SSA_BURN_IN_T = 25.0  # first half of the t=50 simulation is burn-in
SSA_STATIONARY_MEAN = 10.0


def _ssa_birthdeath_actions():
    return [
        'simulate({method=>"ssa",t_start=>0,t_end=>50,n_steps=>50,suffix=>"tc"})',
    ]


def _collect_ssa_replicates(model, tmp_path, prefix, n_replicates):
    """Run N replicates of a stochastic BngsimModel, returning the time grid
    and an (N, T) array of the ``Stot`` observable."""
    traj = []
    times = None
    for i in range(n_replicates):
        # Mimic Job._run_models so each replicate derives a distinct seed
        # under the default ``auto`` stochastic_seed policy.
        model._pybnf_replicate_index = i
        result = model.execute(str(tmp_path), '%s_%d' % (prefix, i), 60)
        data = result['tc']
        if times is None:
            times = data.data[:, data.cols['time']]
        traj.append(data.data[:, data.cols['Stot']])
    return times, np.asarray(traj)


def test_bngsim_bngl_ssa_steady_state_distribution(tmp_path):
    """Gap 1.2: BNGL ``method=>"ssa"`` through real bngsim.

    Birth-death process with k_prod=10, k_deg=1 reaches Poisson(10) at
    steady state. Verify pooled mean and variance over the stationary
    portion of N=200 replicates against the analytic moments.
    """
    net_path = FIXTURES / 'e2e_ssa_birthdeath.net'
    model = _bngsim_bngl_model(net_path, _ssa_birthdeath_actions())
    model.param_set = pset.PSet([])

    times, traj = _collect_ssa_replicates(
        model, tmp_path, 'ssa', SSA_REPLICATES,
    )
    # Sanity: every observation is a non-negative integer count.
    assert np.all(traj >= 0)
    np.testing.assert_array_equal(traj, np.round(traj))

    stationary_mask = times >= SSA_BURN_IN_T
    assert stationary_mask.sum() >= 10, (
        'expected several stationary time points after burn-in'
    )
    stationary = traj[:, stationary_mask].ravel()

    # The "effective" sample count is N_REPLICATES * (number of stationary
    # time points), but adjacent samples within one replicate are correlated.
    # Conservatively use N_REPLICATES as the effective independent count.
    n_eff = SSA_REPLICATES
    sample_mean = stationary.mean()
    sample_var = stationary.var(ddof=1)
    mean_se = np.sqrt(SSA_STATIONARY_MEAN / n_eff)
    # 5-sigma tolerance on the mean.
    assert abs(sample_mean - SSA_STATIONARY_MEAN) < 5.0 * mean_se, (
        'SSA mean %.3f deviates from Poisson(%.1f) by more than 5 SE (%.3f)'
        % (sample_mean, SSA_STATIONARY_MEAN, mean_se)
    )
    # Variance should also be near the Poisson value; use a looser ratio
    # bound to absorb autocorrelation in the variance estimate.
    assert 0.5 * SSA_STATIONARY_MEAN < sample_var < 2.0 * SSA_STATIONARY_MEAN, (
        'SSA variance %.3f is not within [5, 20] (Poisson reference %.1f)'
        % (sample_var, SSA_STATIONARY_MEAN)
    )


def test_bngsim_ssa_same_seed_reproduces_trajectory(tmp_path):
    """End-to-end stochastic reproducibility under ``stochastic_seed=auto``.

    The per-run simulator seed is derived as a SHA-256 content hash of the
    parameter set + replicate index (pybnf/_seed.py), NOT drawn from NumPy's RNG.
    So the SAME params at the SAME replicate index reproduce the SAME stochastic
    trajectory, while a different replicate index derives a different seed and a
    different trajectory. The default_rng migration leaves ``_seed.py`` untouched
    and makes the algorithm RNG deterministic (test_seed_determinism), so a full
    stochastic fit re-run with the same ``random_seed`` reproduces its saved data
    end to end: deterministic proposals -> deterministic derived sim seeds ->
    deterministic trajectories."""
    net_path = FIXTURES / 'e2e_ssa_birthdeath.net'

    def run(prefix, replicate):
        model = _bngsim_bngl_model(net_path, _ssa_birthdeath_actions())
        model.param_set = pset.PSet([])
        model._pybnf_replicate_index = replicate   # absent policy attr defaults to 'auto'
        return model.execute(str(tmp_path), prefix, 60)['tc'].data

    rep0_a = run('rep0_a', 0)
    rep0_b = run('rep0_b', 0)
    rep1 = run('rep1', 1)

    # Same params + same replicate index -> identical derived seed -> identical run.
    np.testing.assert_array_equal(rep0_a, rep0_b)
    # A different replicate index derives a different seed -> a different trajectory.
    assert not np.array_equal(rep0_a, rep1)


# ---------------------------------------------------------------- PSA -----

# Pure first-order production: N(t) is a Poisson process with mean and
# variance both k_prod*t. We sample at t_end and check the final-time
# distribution.

PSA_REPLICATES = 150
PSA_K_PROD = 5.0
PSA_T_END = 10.0
PSA_EXPECTED_MEAN = PSA_K_PROD * PSA_T_END  # 50.0


@pytest.mark.parametrize('psa_action', [
    pytest.param(
        'simulate({method=>"psa",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})',
        id='direct-psa-token',
    ),
    pytest.param(
        'simulate({method=>"ssa",population=>5,t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})',
        id='ssa-population-mapping',
    ),
])
def test_bngsim_bngl_psa_pure_birth_matches_poisson(tmp_path, psa_action):
    """Gap 1.3: BNGL population-SSA through real bngsim.

    Pure first-order production gives N(t_end) ~ Poisson(k_prod*t_end). Verify
    integer-valued trajectories, monotone-non-decreasing counts, and a mean
    consistent with the Poisson reference within 5 SE.

    Parameterized over both PyBNF spellings: ``method=>"psa"`` directly, and
    ``method=>"ssa", population=>N`` which ``_normalize_action_method`` maps to
    ``psa``.
    """
    net_path = FIXTURES / 'e2e_psa_production.net'
    model = _bngsim_bngl_model(net_path, [psa_action])
    model.param_set = pset.PSet([])

    times, traj = _collect_ssa_replicates(
        model, tmp_path, 'psa', PSA_REPLICATES,
    )
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(PSA_T_END)

    # Population SSA must produce integer counts and never decrease (pure birth).
    np.testing.assert_array_equal(traj, np.round(traj))
    assert np.all(traj >= 0)
    diffs = np.diff(traj, axis=1)
    assert np.all(diffs >= 0), 'pure-birth trajectories must be monotone'

    final = traj[:, -1]
    sample_mean = final.mean()
    mean_se = np.sqrt(PSA_EXPECTED_MEAN / PSA_REPLICATES)
    assert abs(sample_mean - PSA_EXPECTED_MEAN) < 5.0 * mean_se, (
        'PSA final mean %.3f deviates from Poisson(%.1f) by more than 5 SE (%.3f)'
        % (sample_mean, PSA_EXPECTED_MEAN, mean_se)
    )


# ------------------------------------------------------- ROB-2 surfacing ----

def test_unmatched_free_param_warns_instead_of_silent(tmp_path, caplog):
    """ROB-2: a free parameter that doesn't map to a model parameter used to be
    silently dropped in ``execute()`` -- the optimizer believes it is varying
    the parameter while the model never sees it (a confidently-wrong fit). It
    must now surface as a warning. (Dropping is legitimate only in multi-model
    fits, where the shared PSet spans other models' parameters, hence a warning
    rather than a hard error.)"""
    import logging

    net_path = FIXTURES / 'e2e_ode_decay.net'
    actions = [
        'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>20,suffix=>"tc"})',
    ]
    model = _bngsim_bngl_model(net_path, actions)
    model.param_set = pset.PSet([
        pset.FreeParameter('bogus__FREE', 'normal_var', 0, 1, value=5.0),
    ])

    with caplog.at_level(logging.WARNING, logger='pybnf.bngsim_model'):
        model.execute(str(tmp_path), 'ode_decay_badparam', 60)

    assert any(
        'bogus__FREE' in r.getMessage() and 'not found in this model' in r.getMessage()
        for r in caplog.records
    ), 'expected a warning that the unmatched free parameter was dropped'
