"""Fast new-era (edition-2) validation tier (#436): the closing gate for #423.

The new-era ``experiment:`` / ``data:`` config surface (ADR-0028) was built (Chunks 0-6)
but **under-validated end to end**: the fast analytical tiers
(``test_optimizer_integration`` / ``test_sampler_integration``) drive the *algorithms*
against ``direct_pass`` with no simulation backend and never touch the config surface,
and the real-backend coverage of the surface lived only in the opt-in recovery tier
(``-m recovery``, needs bngsim + BNG2.pl). This module fills the gap with a **fast,
default-CI** tier that proves the surface end to end **without a simulator**:

  * **config build + action synthesis** -- a tiny edition-2 ``experiment:``/``data:`` conf
    builds a real ``Configuration`` whose model carries no ``begin actions`` block, so the
    time-course / dose-response (``parameter_scan``, #426) actions must be *synthesized* from
    the data (ADR-0028/0046).
  * **import -> export -> import round-trips** (the gap #423 flagged: *zero* round-trips on
    real problems). A conf is exported to PEtab v2, re-imported, and re-exported; the emitted
    problem is **petablint-clean** (the external oracle: ``Problem.from_yaml`` -> ``BnglModel``
    -> ``BNG2.pl --check`` + every ``default_validation_task``) and the round trip is
    **fit-preserving** (a synthetic trajectory scores identically through the original and the
    re-imported objective -- the round-trip-oracle pattern of ``test_petab_export.py``).
  * **the rewritten Tier-0/1 examples** -- ``demo/parabola`` (toy ODE time course),
    ``per_observable_noise`` (two observables, two noise families), and ``egfr_ode`` (the
    highest-coverage case: a multi-observable time course **and** a dose-response in one job)
    each have an edition-2 ``_v2`` form that builds and round-trips.

Backend-free means **no bngsim** (the simulation engine the recovery tier needs); BNG2.pl
(``Configuration`` validation execs ``BNG2.pl -v``; the petab oracle execs ``--check``) and
petab (the v2 typed-table API + validator) ARE used, exactly as in ``test_petab_export.py``,
and are present in the default-CI leg. ``pytest.importorskip('petab.v2')`` guards the petab
oracle so the suite still collects where petab is absent.

The ``receptor`` example (a multi-phase pre-equilibration protocol, ADR-0052) is now fully on
the new-era surface: it builds + fits (Phase 1, #440), EXPORTS to a petablint-clean two-period
problem (Phase 2, #441), and IMPORTS / round-trips (Phase 3, #442) -- the two-period ``-inf``/
``0`` Experiment reads back as a ``preequilibrate:`` experiment and re-exports byte-identically.
All three legs are covered below; ``examples/receptor/NEW_ERA_NOTE.md`` tracks the arc.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from pybnf import config as config_mod
from pybnf.data import Data
from pybnf.parse import ploop
from pybnf.petab.export import export_job
from pybnf.petab.import_ import import_job

pytestmark = pytest.mark.newera

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tsv_rows(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _petab_validation_errors(problem_yaml):
    """ERROR-level petablint issues for ``problem_yaml`` (empty == clean).

    The model-level external oracle, identical to ``test_petab_export``'s: load the whole
    problem via ``Problem.from_yaml`` (the real petablint path -- ``model_factory`` ->
    ``BnglModel.from_file`` -> ``BNG2.pl --check``, after ``register_bngl()``), then run
    **every** ``default_validation_task`` (the model-cross checks included).
    """
    pytest.importorskip('petab.v2')
    from petab.v2 import Problem
    from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

    from pybnf.petab.bngl_model import register_bngl
    register_bngl()
    problem = Problem.from_yaml(str(problem_yaml))
    assert type(problem.model).__name__ == 'BnglModel'   # the BNGL loader ran
    errors = []
    for task in default_validation_tasks:
        issue = task.run(problem)
        if issue is not None and getattr(issue, 'level', None) == \
                ValidationIssueSeverity.ERROR:
            errors.append((type(task).__name__, issue.message))
    return errors


def _build_cfg(conf_path):
    """Build a real ``Configuration`` from ``conf_path`` (cwd at the conf's dir so its
    relative ``model:`` / ``data:`` paths resolve). Execs ``BNG2.pl -v`` to validate the
    BNGL model, but never generates the network or simulates -- the backend-free surface."""
    conf_path = Path(conf_path)
    home = os.getcwd()
    os.chdir(conf_path.parent)
    try:
        return config_mod.Configuration(
            ploop(conf_path.read_text().splitlines(keepends=True)))
    finally:
        os.chdir(home)


def _synthetic_sim(cfg, model, offset=0.7):
    """A synthetic simulation trajectory for every experiment of ``cfg``'s ``model``: each
    observable column is the *observed* value shifted by a constant ``offset``, on the data's
    own independent-variable grid (time points or dose axis). Because the round trip preserves
    the data (values + ``_SD``), feeding ``observed + offset`` to two configs that differ only
    in PEtab naming yields the **same** per-point residual, hence the same objective -- so an
    equal score is the fit-preservation oracle, independent of how experiments got renamed.
    Gaussian/chi_sq predictions admit any offset (no positivity constraint)."""
    sim = {model: {}}
    for expname, data in cfg.exp_data[model].items():
        cols, indvar = data.cols, data.indvar
        obs = [c for c in cols if c != indvar and not c.endswith('_SD')]
        arr = np.asarray(data.data)
        mat = np.column_stack(
            [arr[:, cols[indvar]]] + [arr[:, cols[c]] + offset for c in obs])
        sim[model][expname] = Data.from_columns(mat, [indvar] + obs, indvar=indvar)
    return sim


def _score(cfg, offset=0.7, pset=()):
    """Score ``_synthetic_sim`` through ``cfg``'s objective (no simulator runs)."""
    model = next(iter(cfg.models))
    return cfg.obj.evaluate_multiple(
        _synthetic_sim(cfg, model, offset), cfg.exp_data, list(pset), show_warnings=False)


def _round_trip(conf_path, out_root):
    """``conf -> petab1 -> imported.conf -> petab2``; returns ``(petab1, imp, petab2)`` dirs.

    The import->export->import the issue calls out: export to PEtab v2, re-import to a new-era
    conf, re-export. ``petab1`` is the well-formedness target; the re-imported conf is the
    fit-preservation target; ``petab2`` proves a second export round-trips."""
    out_root = Path(out_root)
    petab1, imp, petab2 = out_root / 'petab1', out_root / 'imp', out_root / 'petab2'
    export_job(Path(conf_path), petab1)
    import_job(petab1 / 'problem.yaml', imp)
    export_job(imp / 'imported.conf', petab2)
    return petab1, imp, petab2


# --------------------------------------------------------------------------- #
# Tiny synthetic edition-2 problems (self-contained; no examples/ dependency)
# --------------------------------------------------------------------------- #
_TC_MODEL = """\
begin model
begin parameters
  k1  1.0
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A()  0
end seed species
begin observables
  Molecules  a  A()
end observables
begin reaction rules
  birth: 0 -> A()  k1
end reaction rules
end model
"""

# A birth-death model whose steady state A_ss = k_prod / k_deg is an exact closed form; the
# swept dose k_prod and the fitted k_deg are both model parameters, the observable a is a model
# observable. (The export only reads the entity surface, so the steady-state physics is inert.)
_DR_MODEL = """\
begin model
begin parameters
  k_prod  1.0
  k_deg   2.0
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A()  0
end seed species
begin observables
  Molecules  a  A()
end observables
begin reaction rules
  birth: 0 -> A()    k_prod
  death: A() -> 0    k_deg
end reaction rules
end model
"""


def _write_time_course_job(d):
    (d / 'tc.bngl').write_text(_TC_MODEL)
    (d / 'tc.exp').write_text('# time\ta\ta_SD\n0\t0.0\t0.5\n1\t1.2\t0.5\n2\t2.1\t0.5\n3\t2.8\t0.5\n')
    conf = d / 'job.conf'
    conf.write_text(
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        'model: tc.bngl\n'
        'experiment: tc, data: tc.exp\n'
        'uniform_var = k1 0 10\n'
        'population_size = 4\nmax_iterations = 1\nverbosity = 0\n')
    return conf


def _write_dose_response_job(d):
    (d / 'dr.bngl').write_text(_DR_MODEL)
    (d / 'dose.exp').write_text('# k_prod\ta\ta_SD\n1\t0.5\t0.1\n2\t1.0\t0.1\n4\t2.0\t0.1\n8\t4.0\t0.1\n')
    conf = d / 'job.conf'
    conf.write_text(
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        'model: dr.bngl\n'
        'experiment: dose, data: dose.exp\n'   # non-time indvar (k_prod) => parameter_scan
        'uniform_var = k_deg 0.1 10\n'
        'population_size = 4\nmax_iterations = 1\nverbosity = 0\n')
    return conf


class TestTinySyntheticTimeCourse:
    """A minimal edition-2 time course: build (action synthesized from the data grid),
    export (petablint-clean), and import->export->import fit-preserving."""

    @pytest.fixture(scope='class')
    def job(self, tmp_path_factory):
        d = tmp_path_factory.mktemp('tc_job')
        conf = _write_time_course_job(d)
        return conf, _round_trip(conf, tmp_path_factory.mktemp('tc_rt'))

    def test_action_is_synthesized_from_the_data_grid(self, job):
        conf, _ = job
        cfg = _build_cfg(conf)
        (action,) = [a for a in cfg.models['tc'].actions if 'simulate(' in a]
        assert 'method=>"ode"' in action
        # the data's time points (0,1,2,3) became the explicit output grid -- ADR-0028
        assert 'sample_times=>[0.0,1.0,2.0,3.0]' in action

    def test_export_is_petab_clean(self, job):
        _, (petab1, _imp, _petab2) = job
        assert _petab_validation_errors(petab1 / 'problem.yaml') == []

    def test_round_trip_is_fit_preserving(self, job):
        conf, (_petab1, imp, _petab2) = job
        original = _score(_build_cfg(conf))
        reimported = _score(_build_cfg(imp / 'imported.conf'))
        assert reimported == pytest.approx(original)

    def test_second_export_is_also_clean(self, job):
        _, (_petab1, _imp, petab2) = job
        assert _petab_validation_errors(petab2 / 'problem.yaml') == []


class TestTinySyntheticDoseResponse:
    """A minimal edition-2 dose-response (parameter_scan, #426/ADR-0046): the non-``time``
    independent variable infers the scan; no ``t_end:`` => steady state (PEtab time=inf)."""

    @pytest.fixture(scope='class')
    def job(self, tmp_path_factory):
        d = tmp_path_factory.mktemp('dr_job')
        conf = _write_dose_response_job(d)
        return conf, _round_trip(conf, tmp_path_factory.mktemp('dr_rt'))

    def test_scan_action_is_synthesized_over_the_dose_axis(self, job):
        conf, _ = job
        cfg = _build_cfg(conf)
        (scan,) = [a for a in cfg.models['dr'].actions if 'parameter_scan' in a]
        assert 'parameter=>"k_prod"' in scan        # the dose axis is the swept parameter
        assert 'par_scan_vals=>[1.0,2.0,4.0,8.0]' in scan
        assert 'steady_state=>1' in scan            # no t_end: => steady state (ADR-0046)
        assert cfg.exp_data['dr']['dose'].indvar == 'k_prod'

    def test_export_emits_one_condition_per_dose_at_time_inf(self, job):
        _, (petab1, _imp, _petab2) = job
        conds = _tsv_rows(petab1 / 'conditions.tsv')
        assert [(c['targetId'], c['targetValue']) for c in conds] == [
            ('k_prod', '1'), ('k_prod', '2'), ('k_prod', '4'), ('k_prod', '8')]
        assert all(m['time'] == 'inf' for m in _tsv_rows(petab1 / 'measurements.tsv'))

    def test_export_is_petab_clean(self, job):
        _, (petab1, _imp, _petab2) = job
        assert _petab_validation_errors(petab1 / 'problem.yaml') == []

    def test_round_trip_is_fit_preserving(self, job):
        conf, (_petab1, imp, _petab2) = job
        original = _score(_build_cfg(conf))
        reimported = _score(_build_cfg(imp / 'imported.conf'))
        assert reimported == pytest.approx(original)


# --------------------------------------------------------------------------- #
# The rewritten Tier-0/1 examples
# --------------------------------------------------------------------------- #
# (conf, model id, expected action kinds present): the round-trippable examples (an
# exportable -- non-``fit`` -- noise surface). ``per_observable_noise`` is covered separately
# (its fitted Laplace scale is a documented export deferral).
EXAMPLE_CASES = {
    'demo_parabola': (EXAMPLES / 'demo' / 'demo_bng_v2.conf', 'parabola_v2', {'time_course'}),
    'egfr_ode': (EXAMPLES / 'egfr_ode' / 'egfr_ode_v2.conf', 'egfr_ode_v2',
                 {'time_course', 'parameter_scan'}),
}


@pytest.fixture(scope='module')
def example_round_trip(tmp_path_factory):
    """Export -> import -> export each example once (the expensive BNG2.pl path), cached."""
    cache = {}

    def get(case):
        if case not in cache:
            conf = EXAMPLE_CASES[case][0]
            cache[case] = _round_trip(conf, tmp_path_factory.mktemp(case + '_rt'))
        return cache[case]

    return get


class TestExampleRoundTrip:

    @pytest.mark.parametrize('case', list(EXAMPLE_CASES))
    def test_actions_synthesized(self, case):
        conf, model, kinds = EXAMPLE_CASES[case]
        cfg = _build_cfg(conf)
        actions = cfg.models[model].actions
        if 'time_course' in kinds:
            assert any('simulate(' in a and 'parameter_scan' not in a for a in actions)
        if 'parameter_scan' in kinds:
            assert any('parameter_scan' in a for a in actions)

    @pytest.mark.parametrize('case', list(EXAMPLE_CASES))
    def test_export_is_petab_clean(self, case, example_round_trip):
        petab1, _imp, _petab2 = example_round_trip(case)
        assert _petab_validation_errors(petab1 / 'problem.yaml') == []

    @pytest.mark.parametrize('case', list(EXAMPLE_CASES))
    def test_round_trip_is_fit_preserving(self, case, example_round_trip):
        conf = EXAMPLE_CASES[case][0]
        _petab1, imp, _petab2 = example_round_trip(case)
        original = _score(_build_cfg(conf))
        reimported = _score(_build_cfg(imp / 'imported.conf'))
        assert reimported == pytest.approx(original)

    @pytest.mark.parametrize('case', list(EXAMPLE_CASES))
    def test_second_export_is_also_clean(self, case, example_round_trip):
        _petab1, _imp, petab2 = example_round_trip(case)
        assert _petab_validation_errors(petab2 / 'problem.yaml') == []

    def test_egfr_dose_response_conditions_cover_the_dose_axis(self, example_round_trip):
        # The dose-response half of the highest-coverage example: each EGF dose (LT) becomes a
        # Condition; the all-NaN 0.01 dose contributes a condition but no measurement.
        petab1, _imp, _petab2 = example_round_trip('egfr_ode')
        conds = {(c['targetId'], c['targetValue']) for c in _tsv_rows(petab1 / 'conditions.tsv')}
        assert ('LT', '0.001') in conds and ('LT', '100') in conds
        # the dose endpoint is finite (t_end: 1200), not steady state
        dose_meas = [m for m in _tsv_rows(petab1 / 'measurements.tsv')
                     if m['observableId'].endswith('_dose')]
        assert dose_meas and all(m['time'] == '1200' for m in dose_meas)


# --------------------------------------------------------------------------- #
# per_observable_noise (Tier 0): per-observable noise families on the new-era surface --
# two observables scored by two different families, one with an estimated (fitted) noise
# scale. b_y is a pure observation-layer nuisance (not a model parameter); config build +
# scoring accept it, and since #439 the exporter emits the `fit` sigma as a bare-id
# noiseFormula naming an estimated parameter, so this case round-trips fully.
# --------------------------------------------------------------------------- #
PON_CONF = EXAMPLES / 'per_observable_noise' / 'per_observable_noise_v2.conf'


class TestPerObservableNoiseExample:

    @pytest.fixture(scope='class')
    def cfg(self):
        return _build_cfg(PON_CONF)

    @pytest.fixture(scope='class')
    def round_trip(self, tmp_path_factory):
        return _round_trip(PON_CONF, tmp_path_factory.mktemp('pon_rt'))

    def test_two_distinct_per_observable_specs_loaded(self, cfg):
        from pybnf import noise
        x_family, x_sources = cfg.obj._spec_for('x')
        y_family, y_sources = cfg.obj._spec_for('y')
        (x_source,) = x_sources.values()
        (y_source,) = y_sources.values()
        # x: Gaussian with sigma read per point from x_SD (fixed source, no normalizer)
        assert isinstance(x_family, noise.Gaussian) and isinstance(x_source, noise.DataColumnSigma)
        assert x_source.estimated is False
        # y: Laplace with the scale b_y estimated as a free parameter (normalizer retained)
        assert isinstance(y_family, noise.Laplace) and isinstance(y_source, noise.FreeParameterSigma)
        assert y_source.estimated is True

    def test_time_course_action_synthesized_from_par1(self, cfg):
        (action,) = [a for a in cfg.models['parabola_v2'].actions if 'simulate(' in a]
        assert 'sample_times=>[0.0,1.0,2.0' in action     # par1.exp's 0..20 grid

    def test_laplace_scale_drives_the_y_contribution(self, cfg):
        # Guard against a false wiring: y's estimated Laplace scale b_y must actually enter the
        # likelihood. Its contribution is |pred-obs|/b_y + log(2 b_y) (an estimated source keeps
        # the normalizer -- ADR-0011), so scoring the same synthetic trajectory at two different
        # b_y values must give two different scores. x reads x_SD and is b_y-independent, so the
        # difference isolates the live Laplace term on y. (The family/source types are pinned by
        # test_two_distinct_per_observable_specs_loaded; this proves the scale is evaluated.)
        import types
        s2 = _score(cfg, pset=[types.SimpleNamespace(name='b_y', value=2.0)])
        s4 = _score(cfg, pset=[types.SimpleNamespace(name='b_y', value=4.0)])
        assert np.isfinite(s2) and np.isfinite(s4)
        assert s2 != pytest.approx(s4)

    def test_export_emits_fit_sigma_as_a_bare_estimated_noise_param(self, round_trip):
        # #439: the estimated Laplace scale on y exports as a bare-id noiseFormula naming the
        # noise parameter (no per-measurement placeholder), with b_y declared estimated in the
        # parameter table as a pure observation-layer nuisance (NOT a model entity). x keeps its
        # per-point _SD placeholder, so the two families coexist in one observables table.
        petab1, _imp, _petab2 = round_trip
        obs = {r['observableId']: r for r in _tsv_rows(petab1 / 'observables.tsv')}
        assert obs['func_y']['noiseFormula'] == 'b_y'
        assert obs['func_y']['noisePlaceholders'] == ''
        assert obs['func_y']['noiseDistribution'] == 'laplace'
        assert obs['obs_x']['noiseFormula'] == 'noiseParameter1_obs_x'   # x: per-point _SD
        params = {r['parameterId']: r for r in _tsv_rows(petab1 / 'parameters.tsv')}
        assert params['b_y']['estimate'] == 'true'    # the noise scale is an estimated parameter

    def test_export_is_petab_clean(self, round_trip):
        petab1, _imp, _petab2 = round_trip
        assert _petab_validation_errors(petab1 / 'problem.yaml') == []

    def test_round_trip_is_fit_preserving(self, round_trip):
        # The estimated noise scale (b_y) survives export -> re-import: scoring the same
        # synthetic trajectory through the original and the re-imported objective (with b_y
        # supplied in the pset -- the FreeParameterSigma reads it) gives the same total, so the
        # `fit` sigma source round-trips faithfully (#439).
        import types
        _petab1, imp, _petab2 = round_trip
        b_y = [types.SimpleNamespace(name='b_y', value=2.0)]
        original = _score(_build_cfg(PON_CONF), pset=b_y)
        reimported = _score(_build_cfg(imp / 'imported.conf'), pset=b_y)
        assert reimported == pytest.approx(original)


# --------------------------------------------------------------------------- #
# receptor pre-equilibration (ADR-0052): the full arc on the edition-2 receptor_v2,
# backend-free -- the FITTER builds its two-phase action (#440 Phase 1), the job EXPORTS
# to a petablint-clean two-period problem (#441 Phase 2), and it IMPORTS / round-trips
# (#442 Phase 3, the multi-period inversion that recovers `preequilibrate:`).
# --------------------------------------------------------------------------- #
RECEPTOR_V2_CONF = EXAMPLES / 'receptor' / 'receptor_v2.conf'


def test_receptor_v2_builds_the_two_phase_preequilibration_action():
    """``examples/receptor/receptor_v2`` builds the synthesized two-phase pre-equilibration
    action (ADR-0052, #440) -- backend-free (``BNG2.pl -v`` validates the model; no bngsim,
    no simulation). The fit itself runs through bngsim in
    ``test_recovery.py::test_receptor_v2_example_builds_and_fits``; this covers the build +
    action synthesis in the default leg where bngsim is absent."""
    cfg = _build_cfg(RECEPTOR_V2_CONF)
    model = cfg.models['receptor_v2']
    acts = model.actions
    # equilibrate (unmeasured, steady state) -> setParameter switches -> measurement, in order
    i_off = acts.index('setParameter("Ligand_isPresent",0)')        # pre-equilibrate: no ligand
    i_equil = next(i for i, a in enumerate(acts) if 'steady_state=>1' in a and 'receptor_preequil' in a)
    i_on = acts.index('setParameter("Ligand_isPresent",1)')         # measure: ligand added
    i_meas = next(i for i, a in enumerate(acts) if 'sample_times' in a and 'suffix=>"receptor"' in a)
    assert i_off < i_equil < i_on < i_meas, acts
    # carry-over: no resetConcentrations between the equilibration and the measurement
    assert 'resetConcentrations()' not in acts[i_equil:i_meas + 1], acts[i_equil:i_meas + 1]
    assert [s[1] for s in model.suffixes] == ['receptor']           # equilibration is unmeasured
    assert not model.mutants                                        # both conditions consumed inline
    # the 6 ex.5 rate constants are bare-id free params (ADR-0034); receptor.exp drives the grid
    assert {v.name for v in cfg.variables} == {'KD1', 'km1', 'K2RT', 'km2', 'kphos', 'kdephos'}
    assert cfg.exp_data['receptor_v2']['receptor'].indvar == 'time'


def test_receptor_v2_exports_a_petab_clean_preequilibration_problem(tmp_path):
    """``examples/receptor/receptor_v2`` EXPORTS to a petablint-clean PEtab v2 problem
    (ADR-0052, #441 Phase 2): a leading ``time = -inf`` pre-equilibration period + a
    ``time = 0`` measurement period, plus the two conditions; the measurements are tagged by
    the experiment id. Export only -- the import / round-trip is Phase 3 (#442), so this is the
    export-clean half of the validation tier for receptor (BNG2.pl ``--check`` via the petablint
    oracle; no bngsim)."""
    out = tmp_path / 'receptor_petab'
    export_job(RECEPTOR_V2_CONF, out)
    # The two-period Experiment: -inf equilibration (noligand) -> time=0 measurement (withligand).
    exps = _tsv_rows(out / 'experiments.tsv')
    assert [(e['experimentId'], e['time'], e['conditionId']) for e in exps] == [
        ('receptor', '-inf', 'cond_noligand'),
        ('receptor', '0', 'cond_withligand')]
    conds = {(c['conditionId'], c['targetId'], c['targetValue'])
             for c in _tsv_rows(out / 'conditions.tsv')}
    assert ('cond_noligand', 'Ligand_isPresent', '0') in conds
    assert ('cond_withligand', 'Ligand_isPresent', '1') in conds
    # receptor.exp's RLbonds/pR columns are the measured observables, tagged by the experiment.
    meas = _tsv_rows(out / 'measurements.tsv')
    assert {m['experimentId'] for m in meas} == {'receptor'}
    assert {m['observableId'] for m in meas} == {'obs_RLbonds', 'obs_pR'}
    assert _petab_validation_errors(out / 'problem.yaml') == []


def test_receptor_round_trips_through_preequilibration(tmp_path):
    """``examples/receptor/receptor_v2`` makes the full PEtab v2 round trip (ADR-0052, #442
    Phase 3): export -> import recovers the ``preequilibrate:`` experiment -> re-export is
    byte-identical, and the round trip is fit-preserving. The import is the multi-period
    inversion Phase 3 adds: the two-period ``-inf``/``0`` Experiment is read back as
    ``experiment: receptor, preequilibrate: noligand, condition: withligand``, NOT flattened to
    a plain conditioned time course (the pre-#442 bug). Backend-free (BNG2.pl ``--check`` via the
    petablint oracle; the fit-preserving score runs the objective over a synthetic trajectory)."""
    petab1, imp, petab2 = _round_trip(RECEPTOR_V2_CONF, tmp_path / 'receptor_rt')

    # (a) the first export is petablint-clean (the Phase-2 guarantee, re-asserted here).
    assert _petab_validation_errors(petab1 / 'problem.yaml') == []

    # (b) import recovers the pre-equilibration experiment line -- preequilibrate: before
    # condition: (the fitter grammar / receptor_v2.conf authoring order), not a flattened
    # `condition: withligand` time course that drops the -inf equilibration period.
    exp_lines = [ln for ln in (imp / 'imported.conf').read_text().splitlines()
                 if ln.startswith('experiment:')]
    assert exp_lines == [
        'experiment: receptor, preequilibrate: noligand, condition: withligand, '
        'method: ode, data: receptor.exp']

    # (c) the re-export reproduces the identical two-period shape (the double round trip).
    assert [(e['experimentId'], e['time'], e['conditionId'])
            for e in _tsv_rows(petab2 / 'experiments.tsv')] == [
        ('receptor', '-inf', 'cond_noligand'),
        ('receptor', '0', 'cond_withligand')]
    assert _petab_validation_errors(petab2 / 'problem.yaml') == []

    # (d) fit-preserving: the synthetic trajectory scores identically through the original and
    # the re-imported objective (the equilibration phase is preserved, so the fit is the same).
    original = _score(_build_cfg(RECEPTOR_V2_CONF))
    reimported = _score(_build_cfg(imp / 'imported.conf'))
    assert reimported == pytest.approx(original)


# --------------------------------------------------------------------------- #
# Fit-parameter perturbation in a pre-equilibration period (#443, Phase 2.x): the
# surrogate-base <p>__REF split (ADR-0027) composed onto the two-period pre-equilibration shape
# (ADR-0052). The receptor case (above) perturbs a *fixed* parameter; here a pre-equilibration
# condition perturbs a *fit* parameter, so M is non-empty and every period re-pins it. Both a
# measured-condition variant and a wash-out variant round-trip fit-preserving, backend-free.
# --------------------------------------------------------------------------- #
_PREEQUIL_FIT_MODEL = """\
begin model
begin parameters
  k     1.0
  flag  1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 10
end seed species
begin observables
  Molecules A_tot A()
end observables
begin functions
  deg() k*flag
end functions
begin reaction rules
  A() -> 0 deg()
end reaction rules
end model
"""


def _write_preequil_fit_job(d, washout=False):
    """A pre-equilibration job whose pre-equilibration condition perturbs the FIT parameter ``k``
    (so M = {k}). ``washout`` drops the measurement ``condition:`` (the measurement period then
    re-pins ``k`` via the synthesized ``cond_wildtype`` base -- #443)."""
    (d / 'm.bngl').write_text(_PREEQUIL_FIT_MODEL)
    (d / 'relax.exp').write_text(
        '# time\tA_tot\tA_tot_SD\n0\t10\t0.5\n1\t6\t0.5\n2\t4\t0.5\n')
    conf = d / 'job.conf'
    meas = '' if washout else 'condition: meas, perturbations: flag = 1\n'
    meas_field = '' if washout else ', condition: meas'
    conf.write_text(
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        'model: m.bngl\n'
        'condition: pre,  perturbations: k = 0.5\n'
        + meas
        + f'experiment: relax, preequilibrate: pre{meas_field}, data: relax.exp\n'
        'uniform_var = k 0.1 10\n'
        'population_size = 4\nmax_iterations = 1\nverbosity = 0\n')
    return conf


@pytest.mark.parametrize('washout', [False, True], ids=['measured', 'washout'])
def test_fit_parameter_preequilibration_round_trips(washout, tmp_path):
    """A pre-equilibration condition perturbing a FIT parameter round-trips fit-preserving
    through export -> import -> re-export (#443), mirroring the receptor round trip (#442). The
    fit param ``k`` is split to ``k__REF`` in the parameter table; the equilibration period sets
    ``k = 0.5`` and the measurement period re-pins ``k = k__REF`` (or the synthesized
    ``cond_wildtype`` base for the wash-out). Backend-free (BNG2.pl ``--check`` via petablint;
    the synthetic-trajectory score is the fit-preservation oracle)."""
    conf = _write_preequil_fit_job(tmp_path, washout=washout)
    petab1, imp, petab2 = _round_trip(conf, tmp_path / 'rt')

    # (a) both exports are petablint-clean (the Phase-2.x guarantee).
    assert _petab_validation_errors(petab1 / 'problem.yaml') == []
    assert _petab_validation_errors(petab2 / 'problem.yaml') == []

    # (b) the surrogate split: k is removed from the parameter table (renamed k__REF).
    pids = {r['parameterId'] for r in _tsv_rows(petab1 / 'parameters.tsv')}
    assert 'k__REF' in pids and 'k' not in pids

    # (c) import recovers the pre-equilibration experiment line (preequilibrate: before the
    # measurement condition:, or no condition: for the wash-out), not a flattened time course.
    exp_lines = [ln for ln in (imp / 'imported.conf').read_text().splitlines()
                 if ln.startswith('experiment:')]
    assert len(exp_lines) == 1
    assert 'preequilibrate: pre' in exp_lines[0]
    assert ('condition: meas' in exp_lines[0]) is (not washout)

    # (d) fit-preserving: the synthetic trajectory scores identically through the original and
    # the re-imported objective.
    original = _score(_build_cfg(conf))
    reimported = _score(_build_cfg(imp / 'imported.conf'))
    assert reimported == pytest.approx(original)
