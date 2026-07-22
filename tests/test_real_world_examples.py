"""Validate the ``examples/real-world/`` corpus through the bngsim backend (#380/#472).

The 2019 PyBNF-paper case studies (Mitra et al., iScience 2019), upgraded to the edition-2
surface, and the compact Rijal/Mehta 2025 promoter-noise collection are the representative
deterministic / stochastic / network-free examples validated against the BNGsim-backed default
path. This is the integration surface the tiny tutorial fits and fixture-level unit tests (#379)
deliberately do not exercise. Their committed ``.conf`` files ARE the test inputs;
``examples/real-world/_manifest.py`` records what each should produce.

Two tiers:

* ``test_real_world_conf_is_wellformed`` -- **default CI, backend-free.** Every conf
  parses, is edition 2, selects the simulator (ode/ssa/nf) the manifest expects, and
  binds its data. Catches conf bit-rot without needing bngsim or BNG2.pl.

* ``test_real_world_runs_through_bngsim`` -- **opt-in (``recovery``; ``heavy`` ones add
  ``slow``).** Each example is built through the real bngsim backend (BNG2.pl network
  generation for ode/ssa, BNGXML for nf) and driven through a short bounded fit inline
  (faked Dask, real simulation). Asserts the whole simulate -> score -> propose loop
  runs and yields a finite, improving objective -- i.e. observables map, the objective
  scores, and the optimizer advances, on a real paper model through bngsim. This is the
  end-to-end integration check; per-observable numerics are covered at the fixture level
  by test_bngsim_bngl_e2e / test_bngsim_nf_e2e / test_bngsim_ssa_replaces_rr (#379).

Run the heavy tier with::

    BNGPATH=... pytest tests/test_real_world_examples.py -m recovery
    BNGPATH=... pytest tests/test_real_world_examples.py -m 'recovery and slow'   # incl. heavy
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf import _bngsim_caps
from pybnf.printing import PybnfError
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO_ROOT / 'examples' / 'real-world' / '_manifest.py'

_spec = importlib.util.spec_from_file_location('_real_world_manifest', _MANIFEST)
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)
EXAMPLES = _manifest.EXAMPLES


def _load_conf(example, tmp_path, *, max_iterations=None, population_size=None):
    """Parse a committed real-world conf into a Configuration, output redirected to
    tmp and the RNG pinned. Paths in the conf are relative to the example folder, so
    we parse from inside it (as tests/test_tutorial_examples.py does)."""
    text = (example.path / example.conf).read_text()
    home = os.getcwd()
    os.chdir(example.path)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        raw['random_seed'] = 1234
        raw['verbosity'] = 0
        if max_iterations is not None:
            raw['max_iterations'] = max_iterations
        if population_size is not None:
            raw['population_size'] = population_size
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _build(conf, fit_type):
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        return FIT_TYPE_REGISTRY[fit_type].cls(conf)
    finally:
        os.chdir(home)


# --------------------------------------------------------------------------- #
# Tier 1: backend-free -- runs in default CI.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('example', EXAMPLES, ids=lambda e: e.folder)
def test_real_world_conf_is_wellformed(example, tmp_path):
    """Every committed real-world conf parses, is edition 2, selects its documented
    simulator, and binds its data -- backend-free (no bngsim / BNG2.pl needed)."""
    conf = _load_conf(example, tmp_path)
    assert conf.config.get('edition') == 2, f'{example.folder}: not edition 2'
    # job_type -> fit_type on the resolved config; must be a real algorithm.
    fit_type = conf.config['fit_type']
    assert fit_type in FIT_TYPE_REGISTRY, f'{example.folder}: unknown fit_type {fit_type!r}'
    # The experiment introduced its data (exp_data is keyed by the model name).
    assert conf.exp_data, f'{example.folder}: no experiment data bound'
    # The free parameters bind by id (edition-2, ADR-0034): none carry a __FREE marker.
    names = [v.name for v in conf.variables]
    assert names, f'{example.folder}: no free parameters'
    for name in names:
        assert '__FREE' not in name, f'{example.folder}: {name} still uses a __FREE alias'
    # #471: the synthesized simulate/parameter_scan sets the model `stochastic` flag from its
    # method (ssa/nf), so the smoothing misuse check no longer false-alarms on an edition-2
    # stochastic fit. The manifest records which examples are stochastic; assert the resolved
    # model agrees -- this is the corpus that surfaced the bug.
    model_stochastic = any(getattr(m, 'stochastic', False) for m in conf.models.values())
    assert model_stochastic == example.stochastic, (
        f'{example.folder}: model.stochastic={model_stochastic}, manifest '
        f'stochastic={example.stochastic}')


def test_real_world_has_workstation_ssa_recovery_coverage():
    """#472's regression guard: SSA must have a non-heavy real-world example in the recovery
    tier, rather than only the cluster-scale FcERI reference and fixture-level execution tests."""
    ssa = [example for example in EXAMPLES if example.simulator == 'ssa' and not example.heavy]
    assert ssa, 'real-world recovery tier has no workstation-scale SSA example'
    assert any(example.folder.startswith('Rijal-2025/') for example in ssa)


@pytest.mark.parametrize('example', [e for e in EXAMPLES if e.simulator == 'nf'],
                         ids=lambda e: e.folder)
def test_real_world_nf_synthesis_is_network_free(example, tmp_path):
    """Regression for the edition-2 NF synthesis fix: a ``method: nf`` experiment must
    synthesise a *network-free* action set -- no ``resetConcentrations()`` (the bngsim
    NF bridge rejects it; NFsim re-seeds each run) and ``generates_network`` left False
    (NFsim models have unbounded networks) -- so it classifies as the NF bridge and
    routes to writeXML -> BngsimNfModel rather than (impossible) network generation.

    The network-free *shape* checks are backend-free (run everywhere). The NF-bridge
    routing check needs bngsim: ``classify_actions_for_bngsim`` normalizes the method
    through the vendored NFsim engine, so on a bngsim-less install (PYBNF_NO_BNGSIM /
    public CI) it returns None, not 'nf' -- it is only asserted where bngsim can perform
    the classification."""
    from pybnf.bngsim_model.classification import classify_actions_for_bngsim
    conf = _load_conf(example, tmp_path)
    m = list(conf.models.values())[0]
    assert not m.generates_network, f'{example.folder}: NF experiment forces network generation'
    assert not any('resetConcentrations' in a for a in m.actions), (
        f'{example.folder}: synthesized NF actions contain resetConcentrations (rejected by '
        f'the bngsim NF bridge)')
    if _bngsim_caps.BNGSIM_AVAILABLE:
        assert classify_actions_for_bngsim(m.actions) == 'nf', (
            f'{example.folder}: actions do not classify as the bngsim NF bridge')


def test_receptor_nf_equilibrates_for_a_fixed_time(tmp_path):
    """NF pre-equilibration uses a FIXED-time equilibration (``equil_t_end:``), not a
    steady-state solve (NFsim has none): the unmeasured equilibration simulate integrates to
    t_end without ``steady_state=>1``. Backend-free."""
    ex = _manifest.example_by_folder('receptor_nf')
    m = list(_load_conf(ex, tmp_path).models.values())[0]
    equil = [a for a in m.actions if 'preequil' in a]
    assert len(equil) == 1, m.actions
    assert 'steady_state' not in equil[0], f'NF equilibration must not request steady_state: {equil[0]}'
    assert 't_end=>600' in equil[0], f'NF equilibration should run to the fixed equil_t_end: {equil[0]}'


def test_egfr_nf_carries_nfsim_options(tmp_path):
    """The ``gml:``/``complex:`` NFsim options ride into every synthesized NF action.
    Backend-free."""
    ex = _manifest.example_by_folder('egfr_nf')
    m = list(_load_conf(ex, tmp_path).models.values())[0]
    sims = [a for a in m.actions if a.startswith(('simulate', 'parameter_scan'))]
    assert sims, m.actions
    for a in sims:
        assert 'gml=>1000000' in a and 'complex=>1' in a, f'NF action missing gml/complex: {a}'


def test_egfr_ode_carries_generate_network_options(tmp_path):
    """#473: the model-scoped ``generate_network`` conf option rides into the synthesized
    ``generate_network`` line. The Kozer EGFR ectodomain model crosslinks (EGF/EGFR are
    multivalent), so its reaction network is finite ONLY under a ``max_stoich`` cap; with the
    edition-2 actions block stripped, the cap lives in the job config. Without this the
    synthesized network is unbounded and generation never terminates. Backend-free (the ODE
    ``egfr_nf`` NFsim guard's deterministic sibling)."""
    ex = _manifest.example_by_folder('egfr_ode')
    m = list(_load_conf(ex, tmp_path).models.values())[0]
    assert m.generates_network, 'egfr_ode should generate a reaction network'
    assert m.generate_network_line == 'generate_network({overwrite=>1,max_stoich=>{EGF=>4,EGFR=>4}})', (
        f'egfr_ode generate_network line missing the max_stoich cap: {m.generate_network_line}')


def test_nf_preequilibration_without_equil_t_end_errors(tmp_path):
    """A network-free pre-equilibration MUST give a fixed equilibration time (``equil_t_end:``)
    because NFsim has no steady-state solve -- omitting it is a clear error at config time, not a
    hang at run time. Backend-free (the error is raised during action synthesis)."""
    ex = _manifest.example_by_folder('receptor_nf')
    model, data = ex.path / 'receptor_nf.bngl', ex.path / 'receptor_nf.exp'
    conf = '\n'.join([
        'edition = 2', 'bngl_backend = bngsim', f'output_dir = {tmp_path}/out',
        f'model: {model}', 'job_type = ss', 'objective = sos',
        'population_size = 4', 'max_iterations = 1',
        'condition: noligand,   perturbations: Ligand_isPresent = 0',
        'condition: withligand, perturbations: Ligand_isPresent = 1',
    ] + [f'loguniform_var = {p} 0.01 100'
         for p in ('KD1', 'km1', 'K2RT', 'km2', 'kphos', 'kdephos')]
        + [f'experiment: receptor_nf, preequilibrate: noligand, condition: withligand, '
           f'method: nf, data: {data}']) + '\n'
    with pytest.raises(PybnfError, match='(?i)equil_t_end'):
        config.Configuration(parse.ploop(conf.splitlines(keepends=True)))


# --------------------------------------------------------------------------- #
# Tier 2: real bngsim end-to-end -- opt-in (recovery); cluster-scale examples excluded.
# --------------------------------------------------------------------------- #
def _e2e_params():
    # Cluster-scale examples (``heavy``) are excluded from the executable tier: the Kozer
    # EGFR crosslinking network takes >10 min to generate, the FcERI network is ~58k reactions,
    # and the network-free receptor/EGFR jobs require many event-by-event simulations. They are
    # covered by the backend-free tier above and validated by construction; running them here is
    # impractical. The compact Rijal promoter networks supply the workstation-scale SSA coverage.
    out = []
    for ex in EXAMPLES:
        if ex.heavy:
            continue
        out.append(pytest.param(ex, marks=[pytest.mark.bngsim, pytest.mark.recovery], id=ex.folder))
    return out


@pytest.mark.parametrize('example', _e2e_params())
def test_real_world_runs_through_bngsim(example, tmp_path, monkeypatch):
    """A representative paper example builds and fits through the real bngsim backend:
    a short bounded fit yields a finite, improving objective -- observables mapped, the
    objective scored, and the optimizer advanced, end to end through bngsim."""
    H.require_bng2pl()
    # Tolerate the odd failed simulation (stochastic NF/SSA can occasionally diverge);
    # a failed sim is scored +inf and the optimizer routes around it, as in production.
    H.install(monkeypatch, catch_sim_failures=True)

    # Keep the bounded run small: enough to take at least one optimizer step so a best
    # score exists, but not a full production fit.
    conf = _load_conf(example, tmp_path, max_iterations=2, population_size=6)
    fit_type = conf.config['fit_type']

    alg = _build(conf, fit_type)
    assert alg.model_list, f'{example.folder}: no models built'

    H.drive(alg)

    best = alg.trajectory.best_score()
    assert best is not None and np.isfinite(best), (
        f'{example.folder}: best objective is {best!r} -- no parameter set produced a '
        f'finite score through bngsim (fail_count={getattr(alg, "fail_count", "?")})')

    # Optional recovery signal for the synthetic-data examples (see manifest.recover).
    if example.recover:
        rec = H.best_params(alg, tuple(example.recover))
        for p, true in example.recover.items():
            rel = abs(rec[p] - true) / abs(true)
            assert rel < example.tol, (
                f'{example.folder}: {p} recovered {rec[p]:g}, expected ~{true:g} '
                f'({rel * 100:.0f}% off > {example.tol * 100:.0f}%)')
