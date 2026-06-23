"""Recovery tier: synthetic-data parameter recovery for a small set of tiny ODE
models, fit through the **real bngsim backend**.

For each model we simulate at known-true parameters to generate a zero-noise
``.exp`` (the oracle), then a real fit must recover those parameters. This
exercises the simulate -> score -> propose loop end to end with a genuine
simulation engine -- the integration surface the analytical tiers
(``test_optimizer_integration`` / ``test_sampler_integration``) deliberately fake.

**Two sub-tiers split by marker (#436):**

  * the **fast new-era sub-tier** (``newera`` marker, NOT ``recovery``) -- the two
    tiny edition-2 recovery fits (a synthesized time course, m01; a synthesized
    steady-state dose-response, m08). These run **by default** wherever bngsim is
    present (a plain ``pytest`` includes them; they auto-skip via the ``bngsim``
    marker where it is absent, e.g. the public-CI leg), so the new-era
    ``experiment:`` surface gets default real-backend coverage without ``-m recovery``.
  * the **heavy opt-in tier** (``recovery`` marker) -- the broader multi-model /
    multi-seed fits, the sampler recovery, and the real-``run_job`` smoke. These stay
    deselected by default (``addopts = -m 'not slow and not recovery'``) and run with
    ``-m recovery``.

Both need bngsim (auto-skipped via the ``bngsim`` marker) and BNG2.pl for the
one-time network generation (``recovery_harness.require_bng2pl`` skips otherwise).

See ``tests/recovery_harness.py`` for the faithfulness boundary (bngsim
simulation is real; dask + per-evaluation folders are faked, the latter covered
by ``test_run_loop`` / ``test_job_execution`` -- with one ``real_run_job`` smoke
below that exercises the genuine path).

Per the orchestration-testing skill, the work is decomposed into separately-named
decisions so a failure points at the right layer:

  * ``test_synthetic_data_matches_analytic`` -- oracle well-posedness (no
    optimizer): the generated data matches a closed-form solution where one
    exists (m01 decay, m02 logistic).
  * ``test_de_recovers``    -- the fit reproduces the data (hard gate, relative to
    data magnitude) AND recovers the identifiable parameters (soft gate), across
    two seeds so it can't pass by a lucky one.
  * ``test_de_recovers_via_experiment_surface`` (``newera``) -- the same m01 recovery
    driven by a new-era ``experiment:`` / ``data:`` conf (ADR-0028) on a model with NO
    ``begin actions`` block, proving PyBNF's synthesized ``sample_times`` simulation
    scores correctly end to end through the real backend.
  * ``test_de_recovers_dose_response_steady_state`` (``newera``) -- a new-era
    ``parameter_scan`` recovering a rate from steady-state dose-response data (ADR-0046).
  * ``test_de_recovers_preequilibration`` (``newera``) -- a new-era ``preequilibrate:``
    experiment recovering a rate from two-phase (equilibrate-to-steady-state -> perturb ->
    measure) data whose measurement t=0 value is the equilibration steady state, proving
    state carry-over across the synthesized phases end to end (ADR-0052, #440).
  * ``test_de_reproducible`` -- a fixed seed gives a bit-identical fit (RNG
    determinism on the real-sim path).
  * ``test_m01_real_run_job_smoke`` -- one fit through the genuine run_job/folders.

The module carries the ``bngsim`` marker for every test; the heavy fits add
``recovery`` (opt-in), while the three ``newera`` fits carry neither ``recovery`` nor
``slow`` so they run by default wherever bngsim is present (#436).
"""
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from . import recovery_harness as H


# Every test in this module needs the real bngsim backend (auto-skipped via the
# ``bngsim`` marker where bngsim is absent). The ``recovery`` (opt-in) vs ``newera``
# (default-run) split is applied per test below (#436).
pytestmark = [pytest.mark.bngsim]


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str
    free: dict          # {param: (var_type, low, high)}
    true: dict          # {param: true_value}
    obs: tuple          # observable column names written into the .exp
    identifiable: tuple # params asserted to recover (subset of free)
    de_budget: dict
    soft_tol: float = 0.15      # fractional param-recovery tolerance
    hard_rel_tol: float = 1e-3  # best_score < hard_rel_tol * (sum of data^2)
    suffix: str = 'ode'
    analytic: tuple = None      # (obs_col, fn(t)->values) closed form, or None

    @property
    def path(self):
        return H.RECOVERY_MODELS_DIR / (self.name + '.bngl')


def _logistic(t):
    X0, K, r = 100.0, 500.0, 0.5
    return K * X0 / (X0 + (K - X0) * np.exp(-r * t))


MODELS = {
    'm01_exp_decay': ModelSpec(
        name='m01_exp_decay',
        free={'k__FREE': ('uniform_var', 1e-3, 5.0)},
        true={'k__FREE': 0.3},
        obs=('Obs_Tot_S',),
        identifiable=('k__FREE',),
        de_budget=dict(population_size=10, max_iterations=30),
        analytic=('Obs_Tot_S', lambda t: 10.0 * np.exp(-0.3 * t)),
    ),
    'm02_logistic': ModelSpec(
        name='m02_logistic',
        free={'r__FREE': ('uniform_var', 0.05, 5.0),
              'K__FREE': ('uniform_var', 50.0, 5000.0)},
        true={'r__FREE': 0.5, 'K__FREE': 500.0},
        obs=('Obs_Tot_X',),
        identifiable=('r__FREE', 'K__FREE'),
        de_budget=dict(population_size=16, max_iterations=50),
        analytic=('Obs_Tot_X', _logistic),
    ),
    'm03_Lotka_Volterra': ModelSpec(
        name='m03_Lotka_Volterra',
        # Oscillatory over ~2 periods -> a multimodal landscape (phase/period
        # traps) where wide bounds let DE stall in a wrong basin. The recovery
        # tier is an integration test, not a global-optimization benchmark, so we
        # bracket the truth ~3-5x (an informed modeler's scale prior); the soft
        # gate still validates genuine recovery within that region.
        free={'a__FREE': ('uniform_var', 0.3, 3.0),
              'd__FREE': ('uniform_var', 0.1, 1.5),
              'b__FREE': ('loguniform_var', 1e-3, 2e-2),
              'c__FREE': ('loguniform_var', 2e-4, 5e-3)},
        true={'a__FREE': 1.1, 'b__FREE': 0.004, 'c__FREE': 0.001, 'd__FREE': 0.4},
        obs=('Obs_Tot_X', 'Obs_Tot_Y'),
        identifiable=('a__FREE', 'b__FREE', 'c__FREE', 'd__FREE'),
        de_budget=dict(population_size=30, max_iterations=70),
        soft_tol=0.25, hard_rel_tol=1e-2,   # oscillatory -> harder; looser gates
    ),
    'm07_SIR': ModelSpec(
        name='m07_SIR',
        free={'beta_rate__FREE': ('loguniform_var', 1e-9, 1e-5),
              'gamma_rate__FREE': ('uniform_var', 0.01, 1.0)},
        true={'beta_rate__FREE': 1e-7, 'gamma_rate__FREE': 0.142857142857143},
        obs=('Obs_Tot_S', 'Obs_Tot_I', 'Obs_Tot_R'),
        identifiable=('beta_rate__FREE', 'gamma_rate__FREE'),
        de_budget=dict(population_size=16, max_iterations=60),
    ),
}


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def _fakes(monkeypatch):
    H.install(monkeypatch)


@pytest.fixture(scope='module')
def exp_for(tmp_path_factory):
    """Lazily generate (and cache) each model's zero-noise synthetic ``.exp``.

    Module-scoped so each model's data is generated once even across the
    seed-parametrized fits; lazy so a ``-k`` selection only builds what it needs.
    """
    cache = {}

    def get(name):
        if name not in cache:
            H.require_bng2pl()
            spec = MODELS[name]
            d = tmp_path_factory.mktemp(name + '_gen')
            cache[name] = H.simulate_truth(
                d, spec.path, spec.true, spec.free, spec.obs, spec.suffix)
        return cache[name]

    return get


def _read_exp(path):
    """Return ``(cols, arr)``: a name->index map and the numeric data (the ``#``
    header line is skipped by genfromtxt as a comment)."""
    with open(path) as f:
        header = f.readline().lstrip('#').split()
    arr = np.genfromtxt(path)
    return {name: i for i, name in enumerate(header)}, arr


def _data_ss(cols, arr, obs):
    """Sum of squares of the observable columns -- the scale the hard gate is
    relative to (so one threshold works across magnitudes ~10 to ~1e7)."""
    return float(sum((arr[:, cols[o]] ** 2).sum() for o in obs))


def _fit_de(tmp_path, spec, exp_path, seed):
    # DE then a Simplex refine (how PyBNF fits are actually finished): the polish
    # drives zero-noise data to the true optimum, so a shallow valley (e.g.
    # logistic r) recovers tightly and the hard gate is met reliably. Exercises
    # the refine->Simplex path with the bngsim backend as a bonus.
    # refine=1 makes the Configuration pull in the Simplex schema defaults
    # (simplex_step, ...) via the refine->Simplex overlay; H.refine() then runs it
    # (drive()/alg.run() alone does not -- pybnf.main orchestrates refine).
    conf = H.make_config(tmp_path, spec.path, exp_path, spec.free, 'de',
                         random_seed=seed, refine=1, **spec.de_budget)
    alg = H.build(conf, 'de')
    H.drive(alg)
    H.refine(alg, conf)
    return alg


def _assert_recovered(spec, alg):
    rec = H.best_params(alg, spec.identifiable)
    for p in spec.identifiable:
        rel = abs(rec[p] - spec.true[p]) / abs(spec.true[p])
        assert rel < spec.soft_tol, \
            '%s: %s recovered %g, expected ~%g (%.0f%% off > %.0f%%)' % (
                spec.name, p, rec[p], spec.true[p], rel * 100, spec.soft_tol * 100)


# --------------------------------------------------------------------------- #
# Oracle well-posedness (no optimizer)
# --------------------------------------------------------------------------- #
@pytest.mark.recovery
@pytest.mark.parametrize('name', [n for n, s in MODELS.items() if s.analytic])
def test_synthetic_data_matches_analytic(name, exp_for):
    """Where a closed form exists, the generated data matches it -- an
    independent oracle that the fit has a reachable global optimum at the truth,
    validating the bngsim simulation + the data-generation path with no optimizer."""
    spec = MODELS[name]
    obs_col, fn = spec.analytic
    cols, arr = _read_exp(exp_for(name))
    t = arr[:, cols['time']]
    np.testing.assert_allclose(arr[:, cols[obs_col]], fn(t), rtol=1e-3, atol=1e-2)


# --------------------------------------------------------------------------- #
# Recovery: hard gate (data reproduced) + soft gate (params), across two seeds
# --------------------------------------------------------------------------- #
@pytest.mark.recovery
@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('name', list(MODELS))
@pytest.mark.parametrize('seed', [1234, 7])
def test_de_recovers(name, seed, tmp_path, exp_for):
    """A real DE fit through bngsim reproduces the data and recovers the params."""
    spec = MODELS[name]
    exp_path = exp_for(name)
    alg = _fit_de(tmp_path, spec, exp_path, seed)

    # Hard gate (relative to data magnitude): the loop drove the objective to ~0.
    cols, arr = _read_exp(exp_path)
    bound = spec.hard_rel_tol * _data_ss(cols, arr, spec.obs)
    best = alg.trajectory.best_score()
    assert best < bound, '%s: best objective %g not < %g' % (spec.name, best, bound)

    # Soft gate: the identifiable parameters come back within tolerance.
    _assert_recovered(spec, alg)


# --------------------------------------------------------------------------- #
# New-era experiment:/data: surface (ADR-0028 Chunk 3d): same recovery, modern conf
# --------------------------------------------------------------------------- #
@pytest.mark.newera
@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('seed', [1234, 7])
def test_de_recovers_via_experiment_surface(seed, tmp_path, exp_for):
    """A new-era ``experiment:`` / ``data:`` conf recovers the parameter through the real
    bngsim backend, proving the synthesized ``sample_times`` path scores end to end
    (ADR-0028). The model file carries NO ``begin actions`` block -- PyBNF synthesizes the
    whole simulation from the experiment's data (the data's ``time`` column becomes the
    output grid). This is the migrated equivalent of ``test_de_recovers`` for m01.
    """
    spec = MODELS['m01_exp_decay']
    exp_path = exp_for(spec.name)   # zero-noise oracle (generated via the original model)
    # New-era binds free parameters BY ID (ADR-0034): drop the legacy `k k__FREE` alias for
    # a bare `k <nominal>` and declare the free parameter by its bare id `k` (the
    # make_config tests above still exercise the legacy __FREE alias). The nominal is
    # irrelevant to the fit (lh initialization samples from the bounds).
    newera_model = H.strip_actions_block(spec.path, tmp_path / 'm01_newera.bngl')
    text = re.sub(r'(?m)^(\s*k\s+)k__FREE\b', r'\g<1>1.0',
                  Path(newera_model).read_text())
    Path(newera_model).write_text(text)
    newera_spec = replace(spec, free={'k': ('uniform_var', 1e-3, 5.0)},
                          true={'k': spec.true['k__FREE']}, identifiable=('k',))
    conf = H.make_newera_config(tmp_path, newera_model, exp_path, newera_spec.free,
                                'decay', 'de', random_seed=seed, refine=1,
                                **spec.de_budget)

    # The simulation is synthesized from the data: the BNGL action carries sample_times
    # (the data's time points), and the data binds under the experiment name.
    model = conf.models['m01_newera']
    assert any('sample_times=>[' in a for a in model.actions), \
        'expected a synthesized sample_times action, got %r' % model.actions
    assert conf.exp_data['m01_newera']['decay'].data.shape[0] > 0

    alg = H.build(conf, 'de')
    H.drive(alg)
    H.refine(alg, conf)

    # Hard gate (data reproduced) + soft gate (param recovered), same as test_de_recovers.
    cols, arr = _read_exp(exp_path)
    bound = spec.hard_rel_tol * _data_ss(cols, arr, spec.obs)
    assert alg.trajectory.best_score() < bound, \
        '%s (new-era): best objective %g not < %g' % (spec.name, alg.trajectory.best_score(), bound)
    _assert_recovered(newera_spec, alg)


# --------------------------------------------------------------------------- #
# New-era parameter_scan (dose-response) at steady state (ADR-0046)
# --------------------------------------------------------------------------- #
@pytest.mark.newera
@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('seed', [1234, 7])
def test_de_recovers_dose_response_steady_state(seed, tmp_path):
    """A new-era ``parameter_scan`` experiment recovers a parameter from STEADY-STATE
    dose-response data through the real bngsim backend (ADR-0046).

    Birth-death: ``0 -> A`` (rate ``k_prod``), ``A -> 0`` (rate ``k_deg``); the steady
    state ``A_ss = k_prod / k_deg`` is an exact closed form, so the synthetic ``.exp``
    (column 0 = the swept dose ``k_prod``, column 1 = ``A_tot = dose / k_deg_true``) is a
    zero-noise oracle with a reachable optimum at the truth. The conf carries NO ``t_end:``,
    so PyBNF synthesizes a ``steady_state=>1`` scan (bngsim's KINSOL solve per dose); a
    correct steady-state simulate -> score -> propose loop recovers ``k_deg``. This is the
    dose-response counterpart of ``test_de_recovers_via_experiment_surface`` (a time course).
    """
    H.require_bng2pl()
    model_path = H.RECOVERY_MODELS_DIR / 'm08_birth_death.bngl'

    k_deg_true = 2.0
    doses = [1.0, 2.0, 4.0, 8.0, 16.0]            # k_prod values (the .exp swept axis)
    exp_path = tmp_path / 'dose.exp'
    lines = ['#\tk_prod\tA_tot']
    lines += ['%.12g\t%.12g' % (d, d / k_deg_true) for d in doses]
    exp_path.write_text('\n'.join(lines) + '\n')

    # k_deg is the fitted free parameter (bound by id, ADR-0034); k_prod is the swept dose.
    # The non-`time` independent variable ('k_prod') infers type=parameter_scan -- no type:
    # field needed. No t_end: => steady state (PEtab time=inf).
    conf = H.make_newera_config(tmp_path, str(model_path), str(exp_path),
                                {'k_deg': ('uniform_var', 0.1, 10.0)}, 'dose', 'de',
                                random_seed=seed, refine=1,
                                population_size=10, max_iterations=20)

    # The synthesized action is a steady-state scan over exactly the data's doses (a BNGL
    # model carries its actions as emitted BNGL strings).
    scan_line = next(a for a in conf.models['m08_birth_death'].actions if 'parameter_scan' in a)
    assert 'steady_state=>1' in scan_line and 'ss_method=>"newton"' in scan_line
    assert 'par_scan_vals=>[1.0,2.0,4.0,8.0,16.0]' in scan_line
    assert 'parameter=>"k_prod"' in scan_line
    assert conf.exp_data['m08_birth_death']['dose'].indvar == 'k_prod'

    alg = H.build(conf, 'de')
    H.drive(alg)
    H.refine(alg, conf)

    # Hard gate (data reproduced): zero-noise steady-state data -> objective floors near 0.
    data_ss = sum((d / k_deg_true) ** 2 for d in doses)
    assert alg.trajectory.best_score() < 1e-3 * data_ss, \
        'dose-response: best objective %g not < %g' % (
            alg.trajectory.best_score(), 1e-3 * data_ss)
    # Soft gate: the degradation rate comes back at the truth.
    rec = H.best_params(alg, ('k_deg',))['k_deg']
    rel = abs(rec - k_deg_true) / k_deg_true
    assert rel < 0.15, 'k_deg recovered %g, expected ~%g (%.0f%% off)' % (rec, k_deg_true, rel * 100)


# --------------------------------------------------------------------------- #
# New-era pre-equilibration surface (ADR-0052 #440): two-phase protocol, real bngsim
# --------------------------------------------------------------------------- #
@pytest.mark.newera
@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('seed', [1234, 7])
def test_de_recovers_preequilibration(seed, tmp_path):
    """A new-era ``preequilibrate:`` experiment recovers a rate from a two-phase protocol
    through the real bngsim backend (ADR-0052, #440).

    Birth-death with switchable production (``m09_preequilibration``): equilibrating with
    ``Production_isOn=1`` settles ``A`` to ``A_ss = k_prod/k_deg``; switching production OFF
    then makes ``A`` decay as ``A(t) = (k_prod/k_deg)*exp(-k_deg*t)`` -- an exact closed form,
    so the synthetic ``.exp`` is a zero-noise oracle with a reachable optimum at the truth.
    The conf carries ``preequilibrate: prod_on, condition: prod_off``, so PyBNF synthesizes the
    two-phase action (steady-state equilibration -> setParameter -> measurement) and the fit
    recovers ``k_deg``.

    The measurement's t=0 value (``A_ss``) is *entirely* the equilibration steady state, so this
    is a sharp **carry-over** gate: if state did not carry from the equilibration phase into the
    measurement, the measurement would start at the seed ``A=0`` and stay flat, matching nothing.
    This is the pre-equilibration counterpart of ``test_de_recovers_dose_response_steady_state``.
    """
    H.require_bng2pl()
    model_path = H.RECOVERY_MODELS_DIR / 'm09_preequilibration.bngl'

    k_prod, k_deg_true = 3.0, 2.0
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    a_ss = k_prod / k_deg_true                                   # the equilibration steady state
    exp_path = tmp_path / 'relax.exp'
    lines = ['#\ttime\tA_tot']
    lines += ['%.12g\t%.12g' % (t, a_ss * np.exp(-k_deg_true * t)) for t in times]
    exp_path.write_text('\n'.join(lines) + '\n')

    # k_deg is the fitted free parameter (bound by id). prod_on (Production_isOn=1) is the
    # pre-equilibration condition; prod_off (Production_isOn=0) the measurement condition. No
    # t_end: anywhere -> the equilibration runs to steady state (ADR-0052/0046).
    conf = H.make_newera_config(tmp_path, str(model_path), str(exp_path),
                                {'k_deg': ('uniform_var', 0.1, 10.0)}, 'relax', 'de',
                                condition=('prod_off', 'Production_isOn = 0'),
                                preequilibrate=('prod_on', 'Production_isOn = 1'),
                                random_seed=seed, refine=1,
                                population_size=10, max_iterations=20)

    # The synthesized action is the two-phase block: a steady-state equilibration simulate
    # (unmeasured, suffix *_preequil), the two setParameter switches, and the measurement
    # simulate over the data grid -- and ONLY 'relax' is a scored suffix.
    model = conf.models['m09_preequilibration']
    acts = model.actions
    assert any('steady_state=>1' in a and 'relax_preequil' in a for a in acts), acts
    assert 'setParameter("Production_isOn",1)' in acts        # pre-equilibration: production ON
    assert 'setParameter("Production_isOn",0)' in acts        # measurement: production OFF
    # the equilibration setParameter precedes the equilibration simulate precedes the
    # measurement setParameter precedes the measurement simulate (phase order)
    i_on = acts.index('setParameter("Production_isOn",1)')
    i_equil = next(i for i, a in enumerate(acts) if 'relax_preequil' in a)
    i_off = acts.index('setParameter("Production_isOn",0)')
    i_meas = next(i for i, a in enumerate(acts) if 'sample_times' in a and 'suffix=>"relax"' in a)
    assert i_on < i_equil < i_off < i_meas, acts
    # carry-over invariant: NO resetConcentrations between equilibration and measurement
    assert 'resetConcentrations()' not in acts[i_equil:i_meas + 1], acts[i_equil:i_meas + 1]
    assert [s[1] for s in model.suffixes] == ['relax']        # equilibration is unmeasured
    assert not model.mutants                                  # both conditions consumed inline
    assert conf.exp_data['m09_preequilibration']['relax'].indvar == 'time'

    alg = H.build(conf, 'de')
    H.drive(alg)
    H.refine(alg, conf)

    # Hard gate (data reproduced): zero-noise two-phase data -> objective floors near 0.
    data_ss = sum((a_ss * np.exp(-k_deg_true * t)) ** 2 for t in times)
    assert alg.trajectory.best_score() < 1e-3 * data_ss, \
        'pre-equilibration: best objective %g not < %g' % (
            alg.trajectory.best_score(), 1e-3 * data_ss)
    # Soft gate: the degradation rate comes back at the truth.
    rec = H.best_params(alg, ('k_deg',))['k_deg']
    rel = abs(rec - k_deg_true) / k_deg_true
    assert rel < 0.15, 'k_deg recovered %g, expected ~%g (%.0f%% off)' % (rec, k_deg_true, rel * 100)


@pytest.mark.newera
@pytest.mark.usefixtures('_fakes')
def test_receptor_v2_example_builds_and_fits(tmp_path):
    """The shipped ``examples/receptor/receptor_v2`` (the edition-2 pre-equilibration form of
    the BioNetFit ex.5 receptor fit) builds the two-phase action and fits through real bngsim
    (ADR-0052, #440 acceptance).

    This is the real-model counterpart of ``test_de_recovers_preequilibration`` (a synthetic
    2-param oracle): receptor is a 6-parameter rule-based model, so this is a **qualitative
    smoke** -- it proves the full receptor model network-generates, synthesizes the two-phase
    pre-equilibration action, and runs the simulate -> score -> propose loop end to end, not
    that 6 parameters are recovered from one short fit. The motivating example for the whole
    feature; previously receptor was legacy-only (#436, ``NEW_ERA_NOTE.md``)."""
    H.require_bng2pl()
    from pybnf import config
    from pybnf.parse import ploop

    example_dir = Path(__file__).resolve().parents[1] / 'examples' / 'receptor'
    text = (example_dir / 'receptor_v2.conf').read_text()
    d = ploop(text.splitlines(keepends=True))
    # A short, deterministic DE smoke (the conf ships scatter search over 50 iterations); keep
    # the loop tiny -- the gate is "it runs + improves", not convergence. output_dir -> tmp.
    d.update({'job_type': 'de', 'population_size': 6, 'max_iterations': 3,
              'random_seed': 1234, 'output_dir': str(tmp_path / 'out'),
              'bngl_backend': 'bngsim', 'verbosity': 0, 'wall_time_sim': 0,
              'delete_old_files': 1})
    home = os.getcwd()
    os.chdir(example_dir)            # relative model:/data: paths resolve at the conf's dir
    try:
        conf = config.Configuration(d)
    finally:
        os.chdir(home)

    # Build assertion: the synthesized action is the two-phase pre-equilibration block
    # (steady-state equilibration -> the two Ligand_isPresent switches -> measurement), with
    # only 'receptor' a scored suffix and both conditions consumed inline.
    model = conf.models['receptor_v2']
    acts = model.actions
    assert any('steady_state=>1' in a and 'receptor_preequil' in a for a in acts), acts
    assert 'setParameter("Ligand_isPresent",0)' in acts          # pre-equilibrate: no ligand
    assert 'setParameter("Ligand_isPresent",1)' in acts          # measure: ligand added
    assert [s[1] for s in model.suffixes] == ['receptor']
    assert not model.mutants
    assert {v.name for v in conf.variables} == {
        'KD1', 'km1', 'K2RT', 'km2', 'kphos', 'kdephos'}         # 6 bare-id free params

    alg = H.build(conf, 'de')
    H.drive(alg)

    # Smoke gate: the fit ran end to end through the two-phase bngsim simulation and found a
    # finite best score (the loop scored real receptor trajectories -- e.g. nonzero pR at t=0
    # carried over from the unmeasured equilibration phase).
    best = alg.trajectory.best_score()
    assert np.isfinite(best), 'receptor_v2 fit produced no finite objective (got %r)' % best


# --------------------------------------------------------------------------- #
# Determinism (guards the RNG-migration contract on the real-sim path)
# --------------------------------------------------------------------------- #
@pytest.mark.recovery
@pytest.mark.usefixtures('_fakes')
def test_de_reproducible(tmp_path, exp_for):
    """A fixed seed yields a bit-identical best fit. Determinism is a property of
    the RNG/framework, not the model, so one model suffices."""
    spec = MODELS['m01_exp_decay']
    exp_path = exp_for(spec.name)
    r1 = H.best_params(_fit_de(tmp_path / 'a', spec, exp_path, 99), spec.identifiable)
    r2 = H.best_params(_fit_de(tmp_path / 'b', spec, exp_path, 99), spec.identifiable)
    assert r1 == r2, 'fixed seed gave different best fit: %r vs %r' % (r1, r2)


# --------------------------------------------------------------------------- #
# Real run_job smoke (genuine production path with the bngsim backend)
# --------------------------------------------------------------------------- #
@pytest.mark.recovery
def test_m01_real_run_job_smoke(tmp_path, exp_for, monkeypatch):
    """One fit through the GENUINE ``run_job`` + per-evaluation folders with the
    bngsim backend, so the production path (not just ``slim_run_job``) is covered."""
    H.install(monkeypatch, real_run_job=True)
    spec = MODELS['m01_exp_decay']
    alg = _fit_de(tmp_path, spec, exp_for(spec.name), seed=1234)
    _assert_recovered(spec, alg)


# --------------------------------------------------------------------------- #
# Sampler recovery: the Adaptive_MCMC posterior concentrates at the truth
# --------------------------------------------------------------------------- #
AM_BUDGET = dict(population_size=3, max_iterations=600, adaptive=100, burn_in=200,
                 sample_every=2, step_size=0.2, num_bins=10, hist_bins=10,
                 credible_intervals=[68, 95], output_hist_every=10 ** 9,
                 rhat_threshold=0)


@pytest.mark.recovery
@pytest.mark.usefixtures('_fakes')
def test_am_recovers_m01(tmp_path, exp_for):
    """The ``am`` sampler run through bngsim concentrates its posterior at the
    true parameter. With zero-noise data the implied posterior is narrow but
    proper (finite objective curvature), so the pooled posterior mean recovers
    the truth -- proving the sampler's simulate->score loop works with a real
    backend (the optimizer counterpart of test_de_recovers)."""
    spec = MODELS['m01_exp_decay']
    conf = H.make_config(tmp_path, spec.path, exp_for(spec.name), spec.free, 'am',
                         random_seed=1234, **AM_BUDGET)
    alg = H.build(conf, 'am')
    H.drive(alg)

    samples = H.read_am_samples(conf.config['output_dir'], spec.identifiable)
    k = samples['k__FREE']
    assert k.size > 0, 'am produced no samples'
    mean_k = float(k.mean())
    assert abs(mean_k - spec.true['k__FREE']) / spec.true['k__FREE'] < 0.2, \
        'am posterior mean k=%g, expected ~%g' % (mean_k, spec.true['k__FREE'])
