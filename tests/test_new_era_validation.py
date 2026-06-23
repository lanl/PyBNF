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

The deferred ``receptor`` example (a multi-phase pre-equilibration protocol the new-era
surface does not express -- ADR-0028/0025) is recorded as a skipped case below;
``examples/receptor/NEW_ERA_NOTE.md`` explains why.
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
# per_observable_noise (Tier 0): per-observable noise families on the new-era surface.
# Its fitted Laplace scale (`fit b_y`) is a documented EXPORT deferral, so this case is
# covered by config-build + scoring rather than an export round trip.
# --------------------------------------------------------------------------- #
PON_CONF = EXAMPLES / 'per_observable_noise' / 'per_observable_noise_v2.conf'


class TestPerObservableNoiseExample:

    @pytest.fixture(scope='class')
    def cfg(self):
        return _build_cfg(PON_CONF)

    def test_two_distinct_per_observable_specs_loaded(self, cfg):
        from pybnf import noise
        x_family, x_source = cfg.obj._spec_for('x')
        y_family, y_source = cfg.obj._spec_for('y')
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

    def test_export_defers_the_fit_sigma_source(self, tmp_path):
        # The fitted Laplace scale is a documented export boundary (a free-parameter sigma needs
        # the noise parameter wired into the PEtab parameter table -- a later chunk, tracked in
        # #439); the exporter raises rather than emit a malformed problem. Keeps the example
        # honest. When #439 lands, replace this with a positive export round-trip.
        with pytest.raises(NotImplementedError, match='fit'):
            export_job(PON_CONF, tmp_path / 'petab')


# --------------------------------------------------------------------------- #
# Deferred example (documented, not a failure): receptor needs a multi-phase
# pre-equilibration protocol the new-era surface does not express (ADR-0028/0025).
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="receptor needs pre-equilibration (equilibrate without ligand, flip "
                         "Ligand_isPresent on, then measure) -- a multi-phase protocol the "
                         "new-era surface defers (ADR-0028/0025, tracked in #440), and "
                         "receptor.exp has no _SD columns. Dropped from the edition-2 example "
                         "set per #436; see examples/receptor/NEW_ERA_NOTE.md.")
def test_receptor_is_a_deferred_preequilibration_case():
    pass
