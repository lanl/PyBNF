"""Unit + round-trip tests for the PEtab v2 *importer* read path (#407; ADR-0032).

The importer is the inverse of the exporter, so its strongest oracle is the exporter
itself: a PyBNF job exported to a PEtab v2 problem and imported back must reproduce the
*problem* exactly. The contract, by strength of oracle:

1. **The byte-equal round trip (the dominant oracle).** ``export -> import -> re-export``
   reproduces the PEtab problem files byte-for-byte (``parameters.tsv`` /
   ``observables.tsv`` / ``measurements.tsv`` / ``conditions.tsv`` / ``experiments.tsv`` /
   ``problem.yaml`` + the BNGL model). The *problem* round-trips; the *recipe* (job_type /
   method / settings) is supplied and is deliberately NOT part of the identity. We compare
   the re-exported problem (not the conf) to avoid conf-formatting noise.
2. **The imported conf is well-formed.** ``parse.ploop`` parses it and it declares the model.
3. **The reconstructed data is exact.** The imported ``.exp`` reproduces the source ``.exp``
   cell-for-cell (the long<->wide pivot inverse).
4. **The external oracle.** The imported-then-re-exported demo problem passes petab's full
   ``default_validation_tasks`` (so the importer emits a genuinely valid PEtab problem,
   not merely one byte-equal to a valid one).
5. **The documented boundaries raise** (an unsupported model language, a PyBNF-less prior
   family, a PEtab-inexpressible noise distribution, a per-measurement placeholder, replicate
   rows) -- mirroring the export side. (SBML now imports, ADR-0036; the expression
   observableFormula becomes a measurement model evaluated post-simulation.)
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pybnf.data import Data
from pybnf.parse import ploop
from pybnf.printing import PybnfError
from pybnf.petab import (
    export_job,
    import_job,
    read_observable_table,
    read_parameter_table,
    read_problem_yaml,
)
from pybnf.petab._bngl import parse_model
from pybnf.petab.import_ import _condition_and_preequilibrate
from pybnf.petab.conditions import (
    PetabConditionRow,
    PetabExperimentRow,
    build_experiment_conditions,
    conditions_from_rows,
    drop_synthesized_wildtype,
    read_condition_table,
    read_experiment_table,
)
from pybnf.petab.measurements import (
    PetabMeasurementRow,
    data_from_measurement_rows,
    measurement_param_bindings,
    measurement_rows_from_data,
    noise_parameter_ids_by_observable,
    observable_parameters_by_observable,
    read_measurement_table,
    reconstruct_dose_responses,
    reconstruct_preequilibrated_dose_responses,
    row_varying_noise_ids,
    row_varying_observable_ids,
)

DEMO_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'demo'
DEMO_CONF = DEMO_DIR / 'demo_bng_v2.conf'
DEMO_MODEL = 'parabola_v2.bngl'

# A crafted PEtab v2 problem exercising the ADR-0044 per-measurement placeholder reduction
# (an observableParameters scale in the observableFormula + an expression noiseFormula).
SCALING_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'scaling_v2'

# A crafted PEtab v2 problem exercising the ADR-0045 row-varying per-measurement noise frontier
# (a noiseParameters id that differs across an observable's rows -> a per-data-point binding).
ROWSIGMA_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'rowsigma_v2'

# A crafted PEtab v2 problem exercising the ADR-0045 row-varying per-measurement OBSERVABLE
# frontier (an observableParameters scale that differs across rows -> a per-data-point
# PerMeasurementModel evaluated in the objective's prediction step, #428 Phase 2b).
OBSSCALE_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'obsscale_v2'

# Three crafted PEtab v2 problems exercising the ADR-0075 observableParameters/noiseParameters
# completions (issue #495): a noiseParameters id that resolves to a FIXED parameter (Oliveira ->
# a constant sigma); a MULTI-token, row-varying noiseParameters product (Fiedler -> a
# PerMeasurementFormulaSigma over two placeholders); and a prediction-dependent affine
# noiseFormula (Raia -> a PredictionFormulaSigma whose sigma scales with the simulated output).
FIXEDSIGMA_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'fixedsigma_v2'
MULTISIGMA_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'multisigma_v2'
PREDSIGMA_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'predsigma_v2'

# A real-world (externally authored) PEtab v2 problem -- the regression oracle for the
# table readers, decoupled from the model (see the fixture's SOURCE.md).
BOEHM_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'boehm_v2'

# A two-parameter-kind model for the conditioned fixture: v1/v2/v3 fit, s fixed (so a
# condition on v1 exercises the surrogate-base rename and one on s the precomputed path).
_PARABOLA2_BNGL = """\
begin model
  begin parameters
    v1 0.5
    v2 1
    v3 3
    s 2
  end parameters
  begin molecule types
    counter()
  end molecule types
  begin seed species
    counter() -10
  end seed species
  begin observables
    Molecules x counter()
  end observables
  begin functions
    y()=s*((v1*(x^2))+(v2*x)+v3)
  end functions
  begin reaction rules
    0->counter() 1
  end reaction rules
end model

begin actions
  generate_network({overwrite=>1})
  simulate({method=>"ode",t_start=>0,t_end=>2,n_steps=>2,suffix=>"par1",print_functions=>1})
end actions
"""

_HEAD = f'edition = 2\njob_type = de\nobjective = chi_sq\nmodel: {DEMO_MODEL}\n'
_PARAMS_U = ('uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
             'uniform_var = v3 0 10\n')


def _roundtrip(tmp_path, conf_text, extra_files=None, model_name=DEMO_MODEL):
    """Run ``export -> import -> re-export`` for ``conf_text``.

    Returns ``(petab1, imported, petab2, conf)``: the first PEtab problem, the imported
    job directory, the re-exported PEtab problem, and the imported ``.conf`` path.
    """
    src = tmp_path / 'src'
    src.mkdir()
    for name, text in (extra_files or {}).items():
        (src / name).write_text(text)
    # Fall back to the demo model/data for anything the fixture did not provide itself.
    if not (src / model_name).exists():
        shutil.copy(DEMO_DIR / model_name, src / model_name)
    if not (src / 'par1.exp').exists():
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
    (src / 'job.conf').write_text(conf_text)

    petab1, imported, petab2 = tmp_path / 'petab1', tmp_path / 'imported', tmp_path / 'petab2'
    export_job(src / 'job.conf', petab1)
    import_job(petab1 / 'problem.yaml', imported)
    conf = imported / 'imported.conf'
    export_job(conf, petab2)
    return petab1, imported, petab2, conf


def _tsv_rows(path):
    """Read a TSV into a list of dict rows (a tiny stdlib reader for assertions)."""
    import csv
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _assert_problem_round_trips(petab1, petab2):
    """Every file in the first PEtab problem is reproduced byte-for-byte by the second."""
    names = sorted(f.name for f in petab1.iterdir())
    assert names == sorted(f.name for f in petab2.iterdir())
    for name in names:
        assert (petab1 / name).read_text() == (petab2 / name).read_text(), \
            f'{name} differs after export -> import -> re-export'


# ---------------------------------------------------------------------------
# 1-3. The MVP round trip: the demo (chi_sq / uniform, single wildtype experiment)
# ---------------------------------------------------------------------------

class TestImportDemoRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(tmp_path_factory.mktemp('demo'), DEMO_CONF.read_text())

    def test_problem_round_trips_byte_for_byte(self, imported):
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_imported_conf_parses_and_declares_the_model(self, imported):
        _, _, _, conf = imported
        with open(conf) as fh:
            d = ploop(fh.readlines())
        assert DEMO_MODEL in d['models']
        assert d['objective'] == 'chi_sq'
        assert d['job_type'] == 'de'

    def test_imported_conf_declares_bare_free_params(self, imported):
        # New-era binds by id (ADR-0034): the conf declares the bare model parameter ids
        # as free parameters -- no '__FREE' marker.
        _, _, _, conf = imported
        text = conf.read_text()
        assert '__FREE' not in text
        for name in ('v1', 'v2', 'v3'):
            assert f'uniform_var = {name} 0 10' in text

    def test_imported_exp_matches_the_source_cell_for_cell(self, imported):
        _, imported_dir, _, _ = imported
        exp = next(imported_dir.glob('*.exp'))
        recon = Data(file_name=str(exp))
        source = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        for col in ('time', 'x', 'y', 'x_SD', 'y_SD'):
            assert np.allclose(recon[col], source[col]), col

    def test_imported_model_is_carried_verbatim(self, imported):
        petab1, imported_dir, _, _ = imported
        model = (imported_dir / DEMO_MODEL).read_text()
        # New-era binds by id (ADR-0034): the model is carried verbatim from the PEtab
        # problem -- bare ids, no '__FREE' marker -- and keeps the measurement model.
        assert model == (petab1 / DEMO_MODEL).read_text()   # byte-identical to the PEtab model
        assert '__FREE' not in model
        assert 'v1 0.5' in model                             # the real nominal, carried through
        assert 'y()=v1*(x^2)+(v2*x)+v3' in model

    def test_imported_problem_passes_full_petab_validation(self, imported):
        # The external oracle: the re-exported problem loads + validates via the real
        # petablint path (same check the export suite runs), proving the importer emits a
        # genuinely valid PEtab problem, not merely one byte-equal to a valid one.
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        _, _, petab2, _ = imported
        problem = Problem.from_yaml(str(petab2 / 'problem.yaml'))
        assert type(problem.model).__name__ == 'BnglModel'
        errors = [type(t).__name__ for t in default_validation_tasks
                  if (i := t.run(problem)) is not None
                  and getattr(i, 'level', None) == ValidationIssueSeverity.ERROR]
        assert errors == []


# ---------------------------------------------------------------------------
# Parameter-valued condition targetValue round trip (ADR-0076): a condition that sets a fixed
# model entity to the value of a FREE parameter -- a per-condition estimated initial condition
# (the Bertozzi/Bruno shape) -- exports as ``targetValue = <param>`` and re-imports byte-for-byte.
# ---------------------------------------------------------------------------

_PARAM_REF_MODEL = """\
begin model
  begin parameters
    v1 0.5
    v2 1
    v3 3
    s 2
  end parameters
  begin molecule types
    counter()
  end molecule types
  begin seed species
    counter() -10
  end seed species
  begin observables
    Molecules x counter()
  end observables
  begin functions
    y()=s*((v1*(x^2))+(v2*x)+v3)
  end functions
  begin reaction rules
    0->counter() 1
  end reaction rules
end model
"""


class TestImportParamRefConditionRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        conf = (
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: pref.bngl\n'
            'condition: cA, perturbations: s = s_A\n'
            'experiment: wt, data: wt.exp\n'
            'experiment: ea, condition: cA, data: ca.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\nuniform_var = s_A 0 10\n')
        extra = {
            'pref.bngl': _PARAM_REF_MODEL,
            'wt.exp': '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n',
            'ca.exp': '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n',
        }
        return _roundtrip(tmp_path_factory.mktemp('paramref'), conf, extra_files=extra,
                          model_name='pref.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_condition_targetvalue_is_the_referenced_param(self, imported):
        petab1, _, _, _ = imported
        rows = _tsv_rows(petab1 / 'conditions.tsv')
        cells = {(r['conditionId'], r['targetId']): r['targetValue'] for r in rows}
        assert cells[('cond_cA', 's')] == 's_A'

    def test_imported_conf_emits_the_param_reference_verbatim(self, imported):
        _, _, _, conf = imported
        d = ploop(conf.read_text().splitlines(keepends=True))
        # The recovered condition carries the parameter-reference value as a string (not a float).
        assert d[('condition', 'cA')] == (None, [('s', '=', 's_A')])
        assert 'uniform_var = s_A 0 10' in conf.read_text()   # s_A recovered as a free parameter


# ---------------------------------------------------------------------------
# #503 (follow-up left by #496): two PEtab observables that map to ONE model column with
# per-observable noise (Bertozzi_PNAS2020's y_I_NY / y_I_CA both measure I_, each with its
# own estimated sigma). The importer currently keys each per-observable noise_model override
# by the shared model COLUMN (import_.py::_per_observable_directives), so it emits two
# colliding `noise_model <column>` lines that parse.ploop rejects. XFAIL until #503 lands;
# a runnable repro + the real problem live in dev/petab-503-repro/. Recommended fix:
# materialize a per-observableId column (an identity measurement model) for a shared-column
# observable, so each dataset keeps its own column + noise line. See the #503 kickoff.
# ---------------------------------------------------------------------------

_SHARED_COL_MODEL = """\
begin model
  begin parameters
    kdeg 0.5
  end parameters
  begin molecule types
    Z()
  end molecule types
  begin seed species
    Z() 100
  end seed species
  begin observables
    Molecules z Z()
  end observables
  begin reaction rules
    Z() -> 0 kdeg
  end reaction rules
end model
"""


def test_shared_column_observables_with_per_observable_noise_import_and_parse(tmp_path):
    """Two observables measuring the same model output `z` in different experiments, each with
    its own estimated sigma, must import into a *parseable* conf with two distinct noise
    sources. Regression for #503 (ADR-0077): each shared-column observable is materialized to
    its own observableId column via an identity measurement model, so the per-observable
    `noise_model` overrides key by the distinct observableId instead of colliding on the one
    shared model column. Simulator-free (import reads the BNGL text with stdlib scanners; the
    failure was at parse.ploop, so no model load / BNG is needed).
    Mirrors dev/petab-503-repro/make_and_repro.py."""
    (tmp_path / 'm.bngl').write_text(_SHARED_COL_MODEL)
    (tmp_path / 'observables.tsv').write_text(
        'observableId\tobservableFormula\tnoiseFormula\tnoisePlaceholders\n'
        'obs_a\tz\tsd_a\t\n'
        'obs_b\tz\tsd_b\t\n')
    (tmp_path / 'parameters.tsv').write_text(
        'parameterId\tlowerBound\tupperBound\tnominalValue\testimate\n'
        'kdeg\t0.01\t10\t0.5\ttrue\n'
        'sd_a\t0.1\t100\t1.0\ttrue\n'
        'sd_b\t0.1\t100\t2.0\ttrue\n')
    (tmp_path / 'measurements.tsv').write_text(
        'experimentId\tobservableId\tmeasurement\ttime\n'
        'ea\tobs_a\t90\t0\n'
        'ea\tobs_a\t55\t1\n'
        'eb\tobs_b\t88\t0\n'
        'eb\tobs_b\t50\t1\n')
    (tmp_path / 'experiments.tsv').write_text(
        'experimentId\ttime\tconditionId\n'
        'ea\t0\t\n'
        'eb\t0\t\n')
    (tmp_path / 'problem.yaml').write_text(
        'format_version: 2.0.0\n'
        'parameter_files:\n- parameters.tsv\n'
        'model_files:\n  m:\n    location: m.bngl\n    language: bngl\n'
        'observable_files:\n- observables.tsv\n'
        'measurement_files:\n- measurements.tsv\n'
        'experiment_files:\n- experiments.tsv\n'
        'condition_files: []\nmapping_files: []\n')

    out = tmp_path / 'imported'
    import_job(tmp_path / 'problem.yaml', out, job_type='de')
    conf = (out / 'imported.conf').read_text()
    ploop(conf.splitlines(keepends=True))          # no longer raises: noise keys by obsId
    assert 'sd_a' in conf and 'sd_b' in conf       # both per-observable sigmas survive, distinct
    lines = conf.splitlines()
    # Each shared-column observable is materialized to its own obsId column via an identity
    # measurement model, so the noise overrides key by obsId (obs_a/obs_b), never the shared `z`.
    assert 'observable: obs_a, formula: z' in lines
    assert 'observable: obs_b, formula: z' in lines
    assert 'noise_model obs_a = gaussian, sigma = fit sd_a' in lines
    assert 'noise_model obs_b = gaussian, sigma = fit sd_b' in lines
    assert not any(line.startswith('noise_model z ') for line in lines)


class TestImportSharedColumnObservablesRoundTrip:
    """The materialized shared-column form (#503, ADR-0077) survives a byte round trip.

    A PyBNF job with two observables measuring one model output ``z`` -- each its own identity
    measurement model with its own estimated sigma -- exports to two ``observables.tsv`` rows
    (both ``observableFormula = z``), and ``export -> import -> re-export`` reproduces the PEtab
    problem byte-for-byte: the importer re-detects the shared entity and re-materializes the two
    per-observableId columns, closing the loop the sibling standalone test opens (import + parse)."""

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        conf = (
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: shared.bngl\n'
            'observable: obs_a, formula: z\n'
            'observable: obs_b, formula: z\n'
            'noise_model obs_a = gaussian, sigma = fit sd_a\n'
            'noise_model obs_b = gaussian, sigma = fit sd_b\n'
            'experiment: ea, data: ea.exp\n'
            'experiment: eb, data: eb.exp\n'
            'uniform_var = kdeg 0.01 10\n'
            'uniform_var = sd_a 0.1 100\n'
            'uniform_var = sd_b 0.1 100\n')
        extra = {
            'shared.bngl': _SHARED_COL_MODEL,
            'ea.exp': '# time obs_a\n0\t90\n1\t55\n',
            'eb.exp': '# time obs_b\n0\t88\n1\t50\n',
        }
        return _roundtrip(tmp_path_factory.mktemp('sharedcol'), conf, extra_files=extra,
                          model_name='shared.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_two_observables_share_the_model_column(self, imported):
        # The source PEtab really is two observables on the ONE model entity z (the shape
        # that collided before #503), each carrying its own estimated sigma.
        petab1, _, _, _ = imported
        rows = _tsv_rows(petab1 / 'observables.tsv')
        assert {r['observableId'] for r in rows} == {'obs_a', 'obs_b'}
        assert {r['observableFormula'] for r in rows} == {'z'}
        assert {r['noiseFormula'] for r in rows} == {'sd_a', 'sd_b'}

    def test_reimport_materializes_distinct_columns_and_noise(self, imported):
        # The re-imported conf keeps each observable on its own obsId column with its own
        # per-observable noise -- never a single colliding `noise_model z`.
        _, _, _, conf = imported
        lines = conf.read_text().splitlines()
        assert 'observable: obs_a, formula: z' in lines
        assert 'observable: obs_b, formula: z' in lines
        assert 'noise_model obs_a = gaussian, sigma = fit sd_a' in lines
        assert 'noise_model obs_b = gaussian, sigma = fit sd_b' in lines
        assert not any(l.startswith('noise_model z ') for l in lines)


# ---------------------------------------------------------------------------
# Dose-response (parameter_scan) round trip (ADR-0046): N steady-state Conditions +
# Experiments measured at time=inf <-> a single swept-axis .exp + a parameter_scan
# experiment. The inverse of the exporter's dose-response emission.
# ---------------------------------------------------------------------------

# A tiny birth-death model whose swept parameter L and fitted parameter kd are both model
# parameters and observable resp is a model observable (the export only reads the entity
# surface, so the steady-state physics is inert in the round trip).
_DR_MODEL = """begin model
begin parameters
  L   1.0
  kd  5.0
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules  resp  A()
end observables
begin reaction rules
  birth: 0 -> A()    L
  death: A() -> 0    kd
end reaction rules
end model
"""

_DR_CONF = (
    'edition = 2\njob_type = de\nobjective = sos\n'
    'model: dr.bngl\n'
    'experiment: dr, data: dose.exp\n'
    'uniform_var = kd 0.1 10\n')

_DR_DOSE_EXP = '# L resp\n1\t0.5\n2\t1\n5\t2.5\n'


class TestImportDoseResponseRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(
            tmp_path_factory.mktemp('dr'), _DR_CONF,
            extra_files={'dr.bngl': _DR_MODEL, 'dose.exp': _DR_DOSE_EXP},
            model_name='dr.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        # The strong oracle: export a dose-response, import (the N time=inf conditions become
        # one swept-axis .exp + a parameter_scan experiment), re-export byte-for-byte.
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_first_export_is_a_steady_state_dose_response(self, imported):
        # Sanity on the source PEtab: N conditions setting L + measurements at time=inf.
        petab1, _, _, _ = imported
        conds = _tsv_rows(petab1 / 'conditions.tsv')
        assert all(c['targetId'] == 'L' for c in conds)
        meas = _tsv_rows(petab1 / 'measurements.tsv')
        assert all(m['time'] == 'inf' for m in meas)

    def test_reconstructed_exp_is_a_swept_axis_grid(self, imported):
        # The importer rebuilds a single .exp whose column 0 is the swept parameter L (its
        # values the doses) and whose observable column carries the per-dose measurements.
        _, imported_dir, _, _ = imported
        recon = Data(file_name=str(imported_dir / 'dr.exp'))
        assert recon.indvar == 'L'
        assert list(recon['L']) == [1.0, 2.0, 5.0]
        assert list(recon['resp']) == [0.5, 1.0, 2.5]

    def test_imported_conf_is_a_parameter_scan_with_no_t_end(self, imported):
        # A steady-state scan: the parameter_scan type is inferred from the .exp's swept axis
        # (no `type:`), and there is no `t_end:` (it runs to steady state, PEtab time=inf).
        _, _, _, conf = imported
        text = conf.read_text()
        assert 'experiment: dr' in text
        assert 't_end' not in text
        assert 'condition:' not in text   # the doses are the scan axis, not condition: lines
        with open(conf) as fh:
            d = ploop(fh.readlines())
        assert ('experiment', 'dr') in d

    def test_imported_conf_loads_as_a_configuration(self, imported, monkeypatch):
        # The fitter accepts the imported conf: it synthesizes a steady-state ParamScan over
        # the reconstructed doses (the Phase-1 keystone, end to end from a PEtab problem).
        from pybnf.config import Configuration
        _, imported_dir, _, conf = imported
        monkeypatch.chdir(imported_dir)
        c = Configuration(ploop(conf.read_text().splitlines(keepends=True)))
        scan = next(a for a in c.models['dr'].actions if 'parameter_scan' in a)
        assert 'steady_state=>1' in scan and 'par_scan_vals=>[1.0,2.0,5.0]' in scan
        assert c.exp_data['dr']['dr'].indvar == 'L'

    def test_imported_problem_passes_full_petab_validation(self, imported):
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        _, _, petab2, _ = imported
        problem = Problem.from_yaml(str(petab2 / 'problem.yaml'))
        errors = [type(t).__name__ for t in default_validation_tasks
                  if (i := t.run(problem)) is not None
                  and getattr(i, 'level', None) == ValidationIssueSeverity.ERROR]
        assert errors == []

    def test_fixed_endpoint_scan_round_trips_with_t_end(self, tmp_path_factory):
        # A finite t_end: dose-response: measurements at the finite time, the imported conf
        # carries `t_end:` and the problem round-trips byte-for-byte.
        conf = _DR_CONF.replace('experiment: dr, data: dose.exp',
                                'experiment: dr, type: parameter_scan, t_end: 250, data: dose.exp')
        petab1, imported_dir, petab2, conf_path = _roundtrip(
            tmp_path_factory.mktemp('drfixed'), conf,
            extra_files={'dr.bngl': _DR_MODEL, 'dose.exp': _DR_DOSE_EXP},
            model_name='dr.bngl')
        meas = _tsv_rows(petab1 / 'measurements.tsv')
        assert all(m['time'] == '250' for m in meas)
        assert 't_end: 250' in conf_path.read_text()
        _assert_problem_round_trips(petab1, petab2)


# ---------------------------------------------------------------------------
# Dose-response replicates (#903): a dose point is one PEtab experiment measured at one time,
# so its replicates are repeated observable rows under one experimentId. The importer deals
# them into <name>.exp / <name>_rep<k>.exp, as it does a time course's (ADR-0039). Before #903
# it pivoted each dose into a single row, each replicate overwriting the one before, so only
# the last replicate reached the fit.
# ---------------------------------------------------------------------------

_DR_DOSE2_EXP = '# L resp\n1\t0.7\n2\t1.4\n5\t3.5\n'
_DR_DOSE_OF = {'dr_0': 1.0, 'dr_1': 2.0, 'dr_2': 5.0}


def _dose_measurements(measurements_tsv):
    """``(dose, measurement)`` for every row of a PEtab measurements table, the dose read from
    the row's experimentId -- the oracle's view of the data, straight from the PEtab table and
    independent of the importer's reconstruction."""
    return [(_DR_DOSE_OF[r['experimentId']], float(r['measurement']))
            for r in _tsv_rows(measurements_tsv)]


def _half_sse_steady_state(points, kd):
    """PyBNF's ``sos`` by hand -- half the sum of squared residuals -- of the birth-death
    steady state ``resp = L / kd`` over ``(L, measurement)`` points."""
    return 0.5 * sum((L / kd - y) ** 2 for L, y in points)


def _score_steady_state_scan(cfg, kd):
    """The loaded job's objective for the ANALYTIC steady state ``resp = L / kd`` at the three
    doses (a simulator-free evaluate over every replicate the job loaded)."""
    import types
    sim = Data.from_columns(np.array([[L, L / kd] for L in (1.0, 2.0, 5.0)]), ['L', 'resp'],
                            indvar='L')
    return cfg.obj.evaluate_multiple({'dr': {'dr': sim}}, cfg.exp_data,
                                     [types.SimpleNamespace(name='kd', value=kd)])


class TestDoseResponseReplicates:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        conf = _DR_CONF.replace('data: dose.exp', 'data: dose.exp, dose2.exp')
        return _roundtrip(
            tmp_path_factory.mktemp('dr_rep'), conf,
            extra_files={'dr.bngl': _DR_MODEL, 'dose.exp': _DR_DOSE_EXP,
                         'dose2.exp': _DR_DOSE2_EXP},
            model_name='dr.bngl')

    def test_export_tags_both_replicates_with_the_dose_ids(self, imported):
        # Sanity on the source PEtab: six rows, each dose id carrying one row per replicate.
        petab1, _, _, _ = imported
        assert [(m['experimentId'], m['measurement'])
                for m in _tsv_rows(petab1 / 'measurements.tsv')] == [
            ('dr_0', '0.5'), ('dr_1', '1'), ('dr_2', '2.5'),
            ('dr_0', '0.7'), ('dr_1', '1.4'), ('dr_2', '3.5')]

    def test_each_replicate_imports_to_its_own_exp(self, imported):
        _, imported_dir, _, conf = imported
        assert 'experiment: dr, method: ode, data: dr.exp, dr_rep2.exp' in conf.read_text()
        for name, source in (('dr.exp', _DR_DOSE_EXP), ('dr_rep2.exp', _DR_DOSE2_EXP)):
            recon = Data(file_name=str(imported_dir / name))
            rows = [line.split() for line in source.splitlines()[1:]]
            assert recon.indvar == 'L'
            assert list(recon['L']) == [float(r[0]) for r in rows]
            assert list(recon['resp']) == [float(r[1]) for r in rows]

    def test_problem_round_trips_byte_for_byte(self, imported):
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_closed_form_kd_counts_all_six_measurements(self, imported):
        # Oracle (the issue's): least squares for resp = L/kd has the closed form
        # kd = sum(L^2) / sum(L*resp). Over the six PEtab rows that is 60/36; the reconstructed
        # .exp files must hold exactly those rows, so they give the same kd -- not the 30/21 of
        # the last replicate alone, which is all the import kept before #903.
        petab1, imported_dir, _, _ = imported
        rows = _dose_measurements(petab1 / 'measurements.tsv')
        assert sum(L * L for L, _ in rows) / sum(L * y for L, y in rows) == pytest.approx(60 / 36)
        recon = [Data(file_name=str(imported_dir / f)) for f in ('dr.exp', 'dr_rep2.exp')]
        points = [(L, y) for d in recon for L, y in zip(d['L'], d['resp'])]
        assert sorted(points) == sorted(rows)

    def test_imported_objective_sums_every_measurement(self, imported, monkeypatch):
        # Oracle: the imported job's objective at a fixed kd equals the hand sum over ALL six
        # measurement rows of the PEtab table, so it is minimised at 60/36.
        petab1, imported_dir, _, _ = imported
        rows = _dose_measurements(petab1 / 'measurements.tsv')
        cfg = _load_conf(imported_dir, monkeypatch)
        for kd in (1.0, 30 / 21, 60 / 36, 3.0):
            assert _score_steady_state_scan(cfg, kd) == pytest.approx(
                _half_sse_steady_state(rows, kd))
        assert (_score_steady_state_scan(cfg, 60 / 36)
                < min(_score_steady_state_scan(cfg, 60 / 36 * f) for f in (0.99, 1.01)))

    def test_external_triplicates_keep_all_nine_measurements(self, imported, tmp_path,
                                                             monkeypatch):
        # The issue's case B: an external problem measuring each dose in triplicate imports as
        # three grids, and its objective sums all nine rows (it kept three before #903).
        petab1, _, _, _ = imported
        problem = tmp_path / 'petab'
        shutil.copytree(petab1, problem)
        lines = ['observableId\texperimentId\ttime\tmeasurement']
        for eid, vals in (('dr_0', (1, 2, 3)), ('dr_1', (10, 20, 30)), ('dr_2', (100, 200, 300))):
            lines += [f'obs_resp\t{eid}\tinf\t{v}' for v in vals]
        (problem / 'measurements.tsv').write_text('\n'.join(lines) + '\n')
        out = import_job(problem / 'problem.yaml', tmp_path / 'out')
        assert ('experiment: dr, method: ode, data: dr.exp, dr_rep2.exp, dr_rep3.exp'
                in (out / 'imported.conf').read_text())
        grids = [Data(file_name=str(out / f)) for f in ('dr.exp', 'dr_rep2.exp', 'dr_rep3.exp')]
        assert [list(g['resp']) for g in grids] == [[1, 10, 100], [2, 20, 200], [3, 30, 300]]
        rows = _dose_measurements(problem / 'measurements.tsv')
        assert len(rows) == 9
        cfg = _load_conf(out, monkeypatch)
        assert _score_steady_state_scan(cfg, 0.05) == pytest.approx(
            _half_sse_steady_state(rows, 0.05))

    def test_fixed_endpoint_scan_keeps_both_replicates(self, tmp_path):
        # The finite-time sibling (a t_end: scan, dose ids <stem>_<i>) goes through the same
        # pivot: both replicates import and the problem round-trips byte-for-byte.
        conf = _DR_CONF.replace('experiment: dr, data: dose.exp',
                                'experiment: dr, type: parameter_scan, t_end: 250, '
                                'data: dose.exp, dose2.exp')
        petab1, _, petab2, conf_path = _roundtrip(
            tmp_path, conf, extra_files={'dr.bngl': _DR_MODEL, 'dose.exp': _DR_DOSE_EXP,
                                         'dose2.exp': _DR_DOSE2_EXP},
            model_name='dr.bngl')
        assert [m['time'] for m in _tsv_rows(petab1 / 'measurements.tsv')] == ['250'] * 6
        assert ('experiment: dr, method: ode, t_end: 250, data: dr.exp, dr_rep2.exp'
                in conf_path.read_text())
        _assert_problem_round_trips(petab1, petab2)

    def test_ragged_dose_replicates_deal_into_a_partial_second_grid(self):
        # Dose 1 measured twice, doses 2 and 5 once: the first grid is the full scan, the
        # second holds only the repeated dose (the time-course dealing rule, ADR-0039).
        inf = float('inf')

        def row(eid, value):
            return PetabMeasurementRow(observable_id='obs_resp', time=inf, measurement=value,
                                       experiment_id=eid)
        conds = [PetabConditionRow(f'cond_{eid}', 'L', str(dose))
                 for eid, dose in _DR_DOSE_OF.items()]
        exps = [PetabExperimentRow(eid, 0.0, f'cond_{eid}') for eid in _DR_DOSE_OF]
        (dr,), remaining, _, _ = reconstruct_dose_responses(
            [row('dr_0', 0.5), row('dr_1', 1.0), row('dr_2', 2.5), row('dr_0', 0.7)],
            conds, exps, {'obs_resp': 'resp'})
        assert remaining == []
        first, second = dr['datas']
        assert list(first['L']) == [1, 2, 5] and list(first['resp']) == [0.5, 1.0, 2.5]
        assert list(second['L']) == [1] and list(second['resp']) == [0.7]

    def test_preequilibrated_scan_keeps_each_replicate_and_its_noise(self, tmp_path):
        # The pre-equilibrated dose-response path (ADR-0062) had the same overwrite. Two
        # replicates with DIFFERENT per-point sigmas (a chi_sq _SD column -> noiseParameters)
        # import as two grids, each keeping its own values and sigmas, and round-trip.
        conf = (_PDR_CONF.replace('objective = sos', 'objective = chi_sq')
                .replace('data: dose.exp', 'data: dose.exp, dose2.exp'))
        rep1 = '# L resp resp_SD\n1\t0.5\t0.1\n2\t1\t0.2\n5\t2.5\t0.3\n'
        rep2 = '# L resp resp_SD\n1\t0.7\t0.4\n2\t1.4\t0.5\n5\t3.5\t0.6\n'
        petab1, imported_dir, petab2, conf_path = _roundtrip(
            tmp_path, conf, extra_files={'m.bngl': _PDR_MODEL, 'dose.exp': rep1,
                                         'dose2.exp': rep2},
            model_name='m.bngl')
        assert [m['noiseParameters'] for m in _tsv_rows(petab1 / 'measurements.tsv')] == [
            '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
        assert 'data: scan.exp, scan_rep2.exp' in conf_path.read_text()
        for name, source in (('scan.exp', rep1), ('scan_rep2.exp', rep2)):
            recon = Data(file_name=str(imported_dir / name))
            rows = [[float(c) for c in line.split()] for line in source.splitlines()[1:]]
            for j, col in enumerate(('L', 'resp', 'resp_SD')):
                assert list(recon[col]) == [r[j] for r in rows], (name, col)
        _assert_problem_round_trips(petab1, petab2)


# ---------------------------------------------------------------------------
# Steady state with NO swept axis (#521, ADR-0086): a PEtab problem measured only at
# ``time = inf`` (Blasi_CellSystems2016's shape). Distinct from the dose-response above,
# which is also at time=inf but reconstructs a swept axis from its N conditions; here
# there is a single condition, so the measurement is a plain equilibrium observation and
# the imported experiment is a steady-state relaxation, not a scan.
# ---------------------------------------------------------------------------

# Birth-death with an analytic equilibrium A_tot -> k_prod/k_deg = 1.5.
_SS_MODEL = """begin model
begin parameters
  k_prod  3.0
  k_deg   2.0
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules A_tot A()
end observables
begin reaction rules
  birth: 0 -> A()   k_prod
  death: A() -> 0   k_deg
end reaction rules
end model
"""

_SS_CONF = ('edition = 2\njob_type = de\nobjective = sos\n'
            'model: ss.bngl\n'
            'experiment: eq, data: eq.exp\n'
            'uniform_var = k_prod 0.1 10\n')

_SS_EXP = '# time A_tot\ninf\t1.5\n'


class TestImportSteadyStateRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(tmp_path_factory.mktemp('ss'), _SS_CONF,
                          extra_files={'ss.bngl': _SS_MODEL, 'eq.exp': _SS_EXP},
                          model_name='ss.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_export_writes_the_petab_steady_state_time(self, imported):
        # PyBNF's `time = inf` .exp cell IS PEtab's steady-state measurement time, so the
        # export is a straight pass-through -- no special casing on either side.
        petab1, _, _, _ = imported
        meas = _tsv_rows(petab1 / 'measurements.tsv')
        assert [m['time'] for m in meas] == ['inf']

    def test_reconstructed_exp_keeps_the_infinite_time(self, imported):
        _, imported_dir, _, _ = imported
        recon = Data(file_name=str(imported_dir / 'experiment1.exp'))
        assert recon.indvar == 'time'
        assert np.isposinf(recon['time'][0])
        assert list(recon['A_tot']) == [1.5]

    def test_imported_conf_loads_as_a_steady_state_relaxation(self, imported, monkeypatch):
        # The keystone: the imported conf now LOADS (it raised OverflowError deriving a step
        # count from t=inf before #521) and synthesizes a steady-state simulate rather than a
        # time course or a scan.
        from pybnf.config import Configuration
        _, imported_dir, _, conf = imported
        monkeypatch.chdir(imported_dir)
        c = Configuration(ploop(conf.read_text().splitlines(keepends=True)))
        sim = next(a for a in c.models['ss'].actions if a.startswith('simulate'))
        assert 'steady_state=>1' in sim and 'n_steps=>1' in sim
        assert 'parameter_scan' not in ''.join(c.models['ss'].actions)

    def test_explicit_type_steady_state_also_exports(self, tmp_path_factory):
        # A hand-written conf may state `type: steady_state`; the exporter takes the same
        # time-course route for it (the .exp time already IS PEtab's inf).
        conf = _SS_CONF.replace('experiment: eq, data: eq.exp',
                                'experiment: eq, type: steady_state, data: eq.exp')
        petab1, _, petab2, _ = _roundtrip(
            tmp_path_factory.mktemp('ss_explicit'), conf,
            extra_files={'ss.bngl': _SS_MODEL, 'eq.exp': _SS_EXP}, model_name='ss.bngl')
        assert [m['time'] for m in _tsv_rows(petab1 / 'measurements.tsv')] == ['inf']
        _assert_problem_round_trips(petab1, petab2)


# ---------------------------------------------------------------------------
# Pre-equilibration: a PEtab v2 two-period Experiment (a leading time=-inf
# steady-state period under the pre-equilibration condition + a time=0 measurement
# period under the measurement condition) imports as a new-era `preequilibrate:`
# experiment and round-trips byte-for-byte (ADR-0052, #442 Phase 3).
# ---------------------------------------------------------------------------

# A birth-death model whose decay is gated by a 0/1 flag (the receptor Ligand_isPresent idiom):
# flag is a FIXED model parameter the two conditions perturb (M empty -- no surrogate split), k
# is the bare-id fit parameter.
_PREEQUIL_MODEL = """begin model
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

_PREEQUIL_CONF = (
    'edition = 2\njob_type = de\nobjective = sos\n'
    'model: m.bngl\n'
    'condition: pre,  perturbations: flag = 0\n'
    'condition: meas, perturbations: flag = 1\n'
    'experiment: relax, preequilibrate: pre, condition: meas, data: relax.exp\n'
    'uniform_var = k 0.1 10\n')

_PREEQUIL_EXP = '# time A_tot\n0\t10\n1\t6\n2\t4\n'


class TestImportPreequilibrationRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(
            tmp_path_factory.mktemp('preequil'), _PREEQUIL_CONF,
            extra_files={'m.bngl': _PREEQUIL_MODEL, 'relax.exp': _PREEQUIL_EXP},
            model_name='m.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        # The strong oracle: export a pre-equilibration (the two-period -inf/0 Experiment),
        # import (recovering preequilibrate: from the multi-period structure), re-export
        # byte-for-byte -- the experiments/conditions/measurements tables all reproduce.
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_first_export_is_a_two_period_experiment(self, imported):
        # Sanity on the source PEtab: the leading -inf equilibration period (cond_pre) precedes
        # the time=0 measurement period (cond_meas).
        petab1, _, _, _ = imported
        assert [(r['experimentId'], r['time'], r['conditionId'])
                for r in _tsv_rows(petab1 / 'experiments.tsv')] == [
            ('relax', '-inf', 'cond_pre'),
            ('relax', '0', 'cond_meas')]

    def test_imported_conf_recovers_the_preequilibrate_experiment(self, imported):
        # The crux of #442: the two-period structure is read back as a single preequilibrate:
        # experiment (preequilibrate: before condition:, the fitter grammar order), NOT
        # flattened to a `condition: meas` time course that drops the -inf period (the pre-#442
        # bug -- the flat experiment->condition map let the last row win).
        _, _, _, conf = imported
        text = conf.read_text()
        assert ('experiment: relax, preequilibrate: pre, condition: meas, '
                'method: ode, data: relax.exp') in text
        assert 'condition: pre, perturbations: flag = 0' in text
        assert 'condition: meas, perturbations: flag = 1' in text

    def test_imported_conf_synthesizes_the_two_phase_action(self, imported, monkeypatch):
        # The fitter accepts the imported conf and synthesizes the equilibrate -> perturb ->
        # measure two-phase action (the pre-equilibration keystone, end to end from a PEtab
        # problem). Backend-free: BNG2.pl -v validates the model; no bngsim, no simulation.
        from pybnf.config import Configuration
        _, imported_dir, _, conf = imported
        monkeypatch.chdir(imported_dir)
        c = Configuration(ploop(conf.read_text().splitlines(keepends=True)))
        acts = c.models['m'].actions
        assert any('steady_state=>1' in a for a in acts)   # the unmeasured equilibration phase
        assert 'setParameter("flag",0)' in acts            # equilibrate under pre (flag=0)
        assert 'setParameter("flag",1)' in acts            # measure under meas (flag=1)


# ---------------------------------------------------------------------------
# Pre-equilibrated dose-response (ADR-0062, #477): N two-period Experiments (a -inf
# pre-equilibration period + a multi-condition measurement period applying a shared wash
# condition AND a per-dose swept-parameter condition), with species setConcentration wash
# targets aliased through the mapping table, imports as a new-era preequilibrate: +
# condition: parameter_scan experiment and round-trips byte-for-byte.
# ---------------------------------------------------------------------------

_PDR_MODEL = """begin model
begin parameters
  L   1.0
  kd  5.0
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 0
  B() 0
end seed species
begin observables
  Molecules  resp  A()
end observables
begin reaction rules
  birth: 0 -> A()    L
  death: A() -> 0    kd
end reaction rules
end model
"""

_PDR_DOSE_EXP = '# L resp\n1\t0.5\n2\t1\n5\t2.5\n'

_PDR_CONF = (
    'edition = 2\njob_type = de\nobjective = sos\n'
    'model: m.bngl\n'
    'condition: incubate, perturbations: "A()" = 100\n'
    'condition: wash, perturbations: "A()" = 0, "B()" = L*kd\n'
    'experiment: scan, preequilibrate: incubate, condition: wash, '
    'type: parameter_scan, t_end: 500, data: dose.exp\n'
    'uniform_var = kd 0.1 10\n')


class TestImportPreequilibratedDoseResponseRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(
            tmp_path_factory.mktemp('pdr'), _PDR_CONF,
            extra_files={'m.bngl': _PDR_MODEL, 'dose.exp': _PDR_DOSE_EXP},
            model_name='m.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        # The strong oracle: export the pre-equilibrated scan (N two-period Experiments + the
        # species mapping table), import (recovering preequilibrate:/condition: + the swept axis),
        # re-export byte-for-byte -- conditions/experiments/measurements/mapping all reproduce.
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_first_export_carries_the_mapping_table(self, imported):
        petab1, _, _, _ = imported
        assert [(m['petabEntityId'], m['modelEntityId'])
                for m in _tsv_rows(petab1 / 'mapping.tsv')] == [
            ('species_A', 'A()'), ('species_B', 'B()')]

    def test_imported_conf_recovers_the_preequilibrated_scan(self, imported):
        # The two-period-per-dose structure reads back as ONE preequilibrate: + condition:
        # parameter_scan experiment; the species targets recover their quoted BNGL patterns
        # (numeric wash + the param-expression competitor) via the mapping table.
        _, _, _, conf = imported
        text = conf.read_text()
        assert 'condition: incubate, perturbations: "A()" = 100' in text
        assert 'condition: wash, perturbations: "A()" = 0, "B()" = L*kd' in text
        assert ('experiment: scan, preequilibrate: incubate, condition: wash, '
                'method: ode, t_end: 500') in text

    def test_reconstructed_exp_is_a_swept_axis_grid(self, imported):
        # The N doses become a single swept-axis .exp (column 0 the swept parameter L, its values
        # the doses; the observable column carries the per-dose measurements).
        _, imported_dir, _, _ = imported
        data = Data(file_name=str(imported_dir / 'scan.exp'))
        assert data.indvar == 'L'
        assert [data.data[i, data.cols['L']] for i in range(data.data.shape[0])] == [1, 2, 5]

    def test_imported_conf_synthesizes_the_preincubate_wash_scan(self, imported, monkeypatch):
        # The fitter accepts the imported conf and synthesizes the equilibrate -> wash -> scan
        # protocol: an unmeasured steady-state simulate, the species washes (a number + the
        # dose-tracking expression), a saveConcentrations, then a reset_conc parameter_scan.
        from pybnf.config import Configuration
        _, imported_dir, _, conf = imported
        monkeypatch.chdir(imported_dir)
        c = Configuration(ploop(conf.read_text().splitlines(keepends=True)))
        acts = c.models['m'].actions
        assert any('steady_state=>1' in a and '_preequil' in a for a in acts)  # equilibration
        assert 'setConcentration("A()",100)' in acts       # incubate (a species amount)
        assert 'setConcentration("A()",0)' in acts          # wash to zero
        assert 'setConcentration("B()","L*kd")' in acts      # dose-tracking competitor expression
        scan = next(a for a in acts if 'parameter_scan' in a)
        assert 'par_scan_vals=>[1.0,2.0,5.0]' in scan and 'reset_conc=>1' in scan

    def test_imported_problem_passes_full_petab_validation(self, imported):
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        _, _, petab2, _ = imported
        problem = Problem.from_yaml(str(petab2 / 'problem.yaml'))
        errors = [type(t).__name__ for t in default_validation_tasks
                  if (i := t.run(problem)) is not None
                  and getattr(i, 'level', None) == ValidationIssueSeverity.ERROR]
        assert errors == []

    def test_steady_state_pdr_round_trips_without_t_end(self, tmp_path_factory):
        # A steady-state pre-equilibrated scan (no t_end:) round-trips: measurements at time=inf,
        # the imported conf omits t_end: (the swept-axis .exp infers the parameter_scan type).
        conf = _PDR_CONF.replace(', type: parameter_scan, t_end: 500', ', type: parameter_scan')
        petab1, _, petab2, imported_conf = _roundtrip(
            tmp_path_factory.mktemp('pdr_ss'), conf,
            extra_files={'m.bngl': _PDR_MODEL, 'dose.exp': _PDR_DOSE_EXP}, model_name='m.bngl')
        _assert_problem_round_trips(petab1, petab2)
        assert 't_end:' not in imported_conf.read_text()


class TestPreequilibrationPeriodGrouping:
    """White-box on the multi-period resolver (`_condition_and_preequilibrate`, ADR-0052/#442):
    a single period is a plain time course; a leading time=-inf steady-state period + a finite
    measurement period is a pre-equilibration; only steady-state -inf equilibration is in scope
    (Phase 1/2), so a finite leading period or >2 periods raises rather than silently flattens."""

    def _row(self, time, cid):
        return PetabExperimentRow('relax', time, cid)

    def test_single_period_is_a_plain_time_course(self):
        # One period -> the measurement condition, no pre-equilibration.
        assert _condition_and_preequilibrate([self._row(0.0, 'cond_meas')], 'relax') == \
            ('meas', None)

    def test_leading_minus_inf_is_a_preequilibration(self):
        # The -inf period's condition is preequilibrate:, the time=0 period's is condition:
        # (sorted by time, so the rows can arrive in either order).
        periods = [self._row(0.0, 'cond_meas'), self._row(float('-inf'), 'cond_pre')]
        periods.sort(key=lambda r: r.time)
        assert _condition_and_preequilibrate(periods, 'relax') == ('meas', 'pre')

    def test_wash_out_measurement_period_drops_the_condition(self):
        # A blank measurement conditionId (a wash-out, Phase 2's empty time=0 period) -> no
        # condition:, just preequilibrate:.
        periods = [self._row(float('-inf'), 'cond_pre'), self._row(0.0, '')]
        assert _condition_and_preequilibrate(periods, 'relax') == (None, 'pre')

    def test_finite_leading_equilibration_period_is_deferred(self):
        # A FINITE leading period is fixed-time equilibration (ADR-0052 "Out"), not steady state;
        # refuse rather than flatten to the last period.
        periods = [self._row(100.0, 'cond_pre'), self._row(200.0, 'cond_meas')]
        with pytest.raises(NotImplementedError, match='fixed-time equilibration'):
            _condition_and_preequilibrate(periods, 'relax')

    def test_more_than_two_periods_is_deferred(self):
        periods = [self._row(float('-inf'), 'cond_pre'), self._row(0.0, 'cond_meas'),
                   self._row(50.0, 'cond_late')]
        with pytest.raises(NotImplementedError, match='more than'):
            _condition_and_preequilibrate(periods, 'relax')


def _fit_imported(out, monkeypatch, seed=1, **settings):
    """Fit the job imported into ``out`` through the real bngsim backend, with the dask layer
    faked as in the recovery tier, and return ``(best objective, best-fit dict)``."""
    from pybnf.config import Configuration
    from . import recovery_harness as H
    H.require_bng2pl()
    H.install(monkeypatch)
    monkeypatch.chdir(out)
    overrides = {'bngl_backend': 'bngsim', 'random_seed': seed, 'refine': 1,
                 'population_size': 10, 'max_iterations': 20, 'delete_old_files': 1,
                 'wall_time_sim': 0, 'output_dir': str(out / 'fit_out'), **settings}
    lines = [line for line in (out / 'imported.conf').read_text().splitlines()
             if line.replace(' ', '').split('=')[0] not in overrides]
    lines += [f'{key} = {value}' for key, value in overrides.items()]
    conf = Configuration(ploop([line + '\n' for line in lines]))
    alg = H.build(conf, 'de')
    H.drive(alg)
    alg = H.refine(alg, conf)
    return alg.trajectory.best_score(), alg.trajectory.best_fit()


# ---------------------------------------------------------------------------
# A plain dose point is ONE period applying ONE condition (#904). The plain dose detector read
# a flat {experimentId: conditionId} map in which the last experiments-table row won, so a
# pre-equilibrated dose -- the experiment__<pre>___<sim> shape petab1to2 writes -- imported
# with its pre-equilibration dropped, and two conditions applied at once imported as the last
# one. The model is dA/dt = L + flag - k*A, A(0) = 0.
# ---------------------------------------------------------------------------

_PERIODS_MODEL = """begin model
begin parameters
  k 1
  L 1
  flag 1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules A_tot A()
end observables
begin functions
  synth() L + flag
end functions
begin reaction rules
  0 -> A() synth()
  A() -> 0 k
end reaction rules
end model
"""


def _write_periods_problem(root, conditions, experiments, measurements):
    """A PEtab v2 problem on :data:`_PERIODS_MODEL` (``k`` estimated, ``obs_A = A_tot``, sigma
    1) from the rows of its conditions / experiments / measurements tables. Returns the yaml."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'model.bngl').write_text(_PERIODS_MODEL)
    tables = {
        'parameters.tsv': (['parameterId', 'lowerBound', 'upperBound', 'nominalValue', 'estimate'],
                           [('k', 0.1, 10, 1, 'true')]),
        'observables.tsv': (['observableId', 'observableFormula', 'noiseFormula',
                             'noiseDistribution'], [('obs_A', 'A_tot', 1, 'normal')]),
        'conditions.tsv': (['conditionId', 'targetId', 'targetValue'], conditions),
        'experiments.tsv': (['experimentId', 'time', 'conditionId'], experiments),
        'measurements.tsv': (['observableId', 'experimentId', 'time', 'measurement'],
                             measurements),
    }
    for name, (header, rows) in tables.items():
        (root / name).write_text('\n'.join('\t'.join(str(c) for c in r)
                                           for r in [header, *rows]) + '\n')
    (root / 'problem.yaml').write_text(
        'format_version: 2.0.0\n'
        + ''.join(f'{key}:\n  - {name}\n' for key, name in (
            ('parameter_files', 'parameters.tsv'), ('observable_files', 'observables.tsv'),
            ('measurement_files', 'measurements.tsv'), ('condition_files', 'conditions.tsv'),
            ('experiment_files', 'experiments.tsv')))
        + 'model_files:\n  model:\n    location: model.bngl\n    language: bngl\n')
    return root / 'problem.yaml'


def _preequilibrated_dose_value(L, k=1.0, pre_L=5.0, t=1.0):
    """The closed form the data are made from: equilibrate at ``L = pre_L`` (flag = 1, so
    ``A = (pre_L + 1)/k``), switch to dose ``L`` at t = 0, read ``A`` at ``t``."""
    return (L + 1) / k + ((pre_L + 1) / k - (L + 1) / k) * np.exp(-k * t)


# The petab1to2 shape of issue #904 (a): one experiment per dose, each a -inf period under 'pre'
# (L = 5) then a time-0 period under 'dose_<i>', measured once at t = 1.
_PETAB1TO2_CONDITIONS = [('pre', 'L', 5)] + [(f'dose_{i}', 'L', i) for i in (1, 2, 3)]
_PETAB1TO2_EXPERIMENTS = [(f'experiment__pre___dose_{i}', t, c) for i in (1, 2, 3)
                          for t, c in (('-inf', 'pre'), ('0', f'dose_{i}'))]
_PETAB1TO2_MEASUREMENTS = [('obs_A', f'experiment__pre___dose_{i}', 1,
                            repr(float(_preequilibrated_dose_value(i)))) for i in (1, 2, 3)]


class TestDosePointIsOnePeriod:

    @pytest.fixture(scope='class')
    def petab1to2_shape(self, tmp_path_factory):
        root = tmp_path_factory.mktemp('p1to2')
        yaml = _write_periods_problem(root / 'problem', _PETAB1TO2_CONDITIONS,
                                      _PETAB1TO2_EXPERIMENTS, _PETAB1TO2_MEASUREMENTS)
        return import_job(yaml, root / 'out')

    def test_petab1to2_shape_imports_as_a_preequilibrated_scan(self, petab1to2_shape):
        # The pre-equilibration is kept: one scan over L under 'preequilibrate: pre', read at
        # t = 1. Before #904 this imported as a plain 't_end: 1' scan with no preequilibrate:.
        text = (petab1to2_shape / 'imported.conf').read_text()
        assert ('experiment: experiment__pre___dose, preequilibrate: pre, method: ode, '
                't_end: 1, data: experiment__pre___dose.exp') in text
        assert 'condition: pre, perturbations: L = 5' in text
        assert 'condition: dose_' not in text          # each dose is the scan axis
        data = Data(file_name=str(petab1to2_shape / 'experiment__pre___dose.exp'))
        assert data.indvar == 'L' and list(data['L']) == [1, 2, 3]
        assert np.allclose(data['A_tot'], [_preequilibrated_dose_value(L) for L in (1, 2, 3)])

    def test_imported_scan_equilibrates_then_scans_the_doses(self, petab1to2_shape,
                                                             monkeypatch):
        cfg = _load_conf(petab1to2_shape, monkeypatch)
        acts = cfg.models['model'].actions
        i_pre = acts.index('setParameter("L",5)')
        i_equil = next(i for i, a in enumerate(acts) if 'steady_state=>1' in a
                       and '_preequil' in a)
        i_scan = next(i for i, a in enumerate(acts) if a.startswith('parameter_scan'))
        assert i_pre < i_equil < i_scan
        assert 'par_scan_vals=>[1.0,2.0,3.0]' in acts[i_scan]
        assert 't_end=>1.0' in acts[i_scan] and 'reset_conc=>1' in acts[i_scan]

    @pytest.mark.bngsim
    @pytest.mark.newera
    def test_imported_fit_recovers_k(self, petab1to2_shape, monkeypatch):
        # Oracle: the data are the closed form at k = 1, so the imported fit must reproduce
        # them and return k = 1. Before #904 the pre-equilibration was dropped and the fit
        # ran to the lower bound (k = 0.1, objective 2.44).
        best, fit = _fit_imported(petab1to2_shape, monkeypatch)
        assert best < 1e-8
        assert fit['k'] == pytest.approx(1.0, rel=1e-3)

    def test_dose_condition_another_experiment_applies_stays_a_condition(self, tmp_path):
        # A dose condition the scan absorbs, but that an unclaimed time course also applies,
        # must stay a condition: line or that experiment would name an undefined condition.
        yaml = _write_periods_problem(
            tmp_path / 'problem', _PETAB1TO2_CONDITIONS,
            _PETAB1TO2_EXPERIMENTS + [('tc', '0', 'dose_2')],
            _PETAB1TO2_MEASUREMENTS + [('obs_A', 'tc', t, 1.0) for t in (0, 1, 2)])
        text = (import_job(yaml, tmp_path / 'out') / 'imported.conf').read_text()
        assert 'experiment: experiment__pre___dose, preequilibrate: pre,' in text
        assert 'experiment: tc, condition: dose_2, method: ode, data: tc.exp' in text
        assert 'condition: dose_2, perturbations: L = 2' in text
        assert 'condition: dose_1' not in text and 'condition: dose_3' not in text

    def test_dose_condition_that_is_also_the_preequilibration_stays_a_condition(
            self, tmp_path, monkeypatch):
        # This scan equilibrates under dose_3 itself, so dose_3 is both a dose and the scan's
        # preequilibrate: condition; consuming it as a dose would leave preequilibrate: dangling.
        exps = [(f'e_{i}', t, c) for i in (1, 2, 3)
                for t, c in (('-inf', 'dose_3'), ('0', f'dose_{i}'))]
        meas = [('obs_A', f'e_{i}', 1, repr(float(_preequilibrated_dose_value(i, pre_L=3.0))))
                for i in (1, 2, 3)]
        out = import_job(_write_periods_problem(tmp_path / 'problem', _PETAB1TO2_CONDITIONS[1:],
                                                exps, meas), tmp_path / 'out')
        text = (out / 'imported.conf').read_text()
        assert 'experiment: e, preequilibrate: dose_3, method: ode, t_end: 1, data: e.exp' in text
        assert 'condition: dose_3, perturbations: L = 3' in text
        assert 'condition: dose_1' not in text and 'condition: dose_2' not in text
        assert 'setParameter("L",3)' in _load_conf(out, monkeypatch).models['model'].actions

    def test_experiments_that_are_not_one_scan_import_each_on_its_own(self, tmp_path,
                                                                       monkeypatch):
        # Two steady-state points sharing the stem 'cell' but pre-equilibrated under DIFFERENT
        # conditions are not one scan. Each imports exactly as its own preequilibrate: +
        # condition: experiment (the time-course path), rather than being refused as ambiguous.
        yaml = _write_periods_problem(
            tmp_path / 'problem',
            [('preA', 'L', 5), ('preB', 'L', 7), ('d1', 'L', 1), ('d2', 'L', 2)],
            [('cell_1', '-inf', 'preA'), ('cell_1', '0', 'd1'),
             ('cell_2', '-inf', 'preB'), ('cell_2', '0', 'd2')],
            [('obs_A', 'cell_1', 'inf', 2.0), ('obs_A', 'cell_2', 'inf', 3.0)])
        out = import_job(yaml, tmp_path / 'out')
        text = (out / 'imported.conf').read_text()
        assert 'experiment: cell_1, preequilibrate: preA, condition: d1, method: ode' in text
        assert 'experiment: cell_2, preequilibrate: preB, condition: d2, method: ode' in text
        cfg = _load_conf(out, monkeypatch)                       # and the conf loads
        assert set(cfg.exp_data['model']) == {'cell_1', 'cell_2'}

    def test_two_conditions_in_one_period_are_refused(self, tmp_path):
        # Issue #904 (b): 'lowflag' (flag = 0.5) and 'high' (L = 2) applied together at t = 0.
        # PyBNF applies one condition per period, so the import refuses, naming the experiment
        # and both conditions. Before #904 it imported as a scan over L = 2 with flag left at 1.
        yaml = _write_periods_problem(
            tmp_path / 'problem', [('lowflag', 'flag', 0.5), ('high', 'L', 2)],
            [('two', '0', 'lowflag'), ('two', '0', 'high')], [('obs_A', 'two', 'inf', 2.5)])
        with pytest.raises(NotImplementedError,
                           match=r"Experiment 'two' applies 2 conditions at the same time "
                                 r"\(0\): \['lowflag', 'high'\]"):
            import_job(yaml, tmp_path / 'out')

    def test_plain_detector_does_not_claim_a_two_period_experiment(self):
        # White-box: a -inf + time-0 experiment whose LAST row sets one numeric target is not a
        # plain dose point, whatever that row sets; its rows are left for the next detector.
        rows = [PetabMeasurementRow(observable_id='obs_A', time=float('inf'), measurement=1.0,
                                    experiment_id='e')]
        conds = [PetabConditionRow('pre', 'L', '5'), PetabConditionRow('dose', 'L', '1')]
        exps = [PetabExperimentRow('e', float('-inf'), 'pre'), PetabExperimentRow('e', 0.0, 'dose')]
        drs, remaining, cids, eids = reconstruct_dose_responses(rows, conds, exps,
                                                                {'obs_A': 'A_tot'})
        assert drs == [] and remaining == rows and cids == set() and eids == set()

    @pytest.mark.parametrize('target, claimed', [('L', True), ('species_A', False)])
    def test_a_species_amount_is_never_the_swept_axis(self, target, claimed):
        # White-box on the other-source rule: the lone time-0 condition is a dose only when it
        # sets a model PARAMETER. A condition setting a species amount (a mapping-table id) is a
        # bolus a parameter_scan cannot sweep, so the experiment is left to the time-course
        # path, which applies it as a setConcentration after the equilibration.
        rows = [PetabMeasurementRow(observable_id='obs_A', time=float('inf'), measurement=5.0,
                                    experiment_id='e')]
        conds = [PetabConditionRow('pre', 'L', '5'), PetabConditionRow('bolus', target, '7')]
        exps = [PetabExperimentRow('e', float('-inf'), 'pre'),
                PetabExperimentRow('e', 0.0, 'bolus')]
        scans, remaining, _, eids = reconstruct_preequilibrated_dose_responses(
            rows, conds, exps, {'obs_A': 'A_tot'}, sweepable={'k', 'L', 'flag'})
        assert bool(scans) is claimed and (eids == {'e'}) is claimed
        assert remaining == ([] if claimed else rows)


def _imported_objective_at(out, monkeypatch, conf_name='imported.conf', **values):
    """The objective of the job imported into ``out`` (or of the conf ``conf_name`` there) at a
    fixed parameter vector, through the real bngsim backend: BNG2.pl generates the network, each
    model is simulated once, and the conf's own objective scores every data file it loaded."""
    from pybnf.config import Configuration
    from pybnf.pset import PSet
    from . import recovery_harness as H
    H.require_bng2pl()
    monkeypatch.chdir(out)
    overrides = {'bngl_backend': 'bngsim', 'population_size': 4, 'max_iterations': 1,
                 'delete_old_files': 1, 'wall_time_sim': 0, 'output_dir': str(out / 'eval_out')}
    lines = [line for line in (out / conf_name).read_text().splitlines()
             if line.replace(' ', '').split('=')[0] not in overrides]
    lines += [f'{key} = {value}' for key, value in overrides.items()]
    conf = Configuration(ploop([line + '\n' for line in lines]))
    alg = H.build(conf, 'de')
    pset = PSet([v.set_value(values[v.name]) for v in alg.variables])
    folder = out / 'eval_out' / 'sim'
    folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(folder)
    simdata = {m.name: m.copy_with_param_set(pset).execute(str(folder), 'eval', 0)
               for m in alg.model_list}
    return conf.obj.evaluate_multiple(simdata, conf.exp_data, pset, conf.constraints)


class TestRaggedPreequilibratedScanThroughTheSimulator:
    """Independent review of #903 x #904: a pre-equilibrated dose-response in the petab1to2 shape
    whose doses are measured a different number of times -- dose 1 once, dose 2 three times,
    dose 3 twice -- so the second and third replicate grids ([2, 3] and [2]) are not a prefix
    of the first ([1, 2, 3]). The author's ragged test is white-box on the pivot and the
    objective tests score an analytic simulation; this one scores the imported job through
    bngsim against a scipy integration of the PEtab problem itself."""

    # (dose, measurement) per measurement row, all read at t = 1 after equilibrating at L = 5.
    ROWS = [(1, 3.5), (2, 4.0), (2, 4.1), (2, 4.25), (3, 4.8), (3, 4.7)]

    @pytest.fixture
    def imported(self, tmp_path):
        eid = 'experiment__pre___dose_{}'.format
        yaml = _write_periods_problem(
            tmp_path / 'problem', _PETAB1TO2_CONDITIONS,
            _PETAB1TO2_EXPERIMENTS, [('obs_A', eid(L), 1, y) for L, y in self.ROWS])
        return import_job(yaml, tmp_path / 'out')

    @staticmethod
    def _petab_objective(rows, k, pre_L=5.0, flag=1.0):
        """Half the sum of squared residuals (sigma 1) over every PEtab measurement row, with
        PEtab v2 period semantics: the -inf period under 'pre' is the steady state
        (pre_L + flag)/k, then the time-0 period sets L to the dose and dA/dt = L + flag - k*A
        is integrated by scipy to the row's time."""
        from scipy.integrate import solve_ivp
        total = 0.0
        for dose, y in rows:
            sol = solve_ivp(lambda _t, a, d=dose: d + flag - k * a, (0.0, 1.0),
                            [(pre_L + flag) / k], rtol=1e-10, atol=1e-12)
            total += 0.5 * (sol.y[0, -1] - y) ** 2
        return total

    def test_ragged_replicates_deal_into_non_prefix_grids(self, imported):
        stem = 'experiment__pre___dose'
        assert (f'experiment: {stem}, preequilibrate: pre, method: ode, t_end: 1, data: '
                f'{stem}.exp, {stem}_rep2.exp, {stem}_rep3.exp') in (
                    imported / 'imported.conf').read_text()
        grids = [Data(file_name=str(imported / f'{stem}{suffix}.exp'))
                 for suffix in ('', '_rep2', '_rep3')]
        assert [list(g['L']) for g in grids] == [[1, 2, 3], [2, 3], [2]]
        assert sorted((L, y) for g in grids for L, y in zip(g['L'], g['A_tot'])) == sorted(
            self.ROWS)

    @pytest.mark.bngsim
    @pytest.mark.newera
    @pytest.mark.parametrize('k', [0.6, 1.0, 1.7])
    def test_imported_objective_is_the_petab_objective_over_every_row(self, imported, k,
                                                                      monkeypatch):
        assert _imported_objective_at(imported, monkeypatch, k=k) == pytest.approx(
            self._petab_objective(self.ROWS, k), rel=1e-5)


class TestPinnedDoseConditionsImportAsOneScan:
    """#892 (fixed in the exporter separately) makes every per-dose condition of a plain
    dose-response re-pin the fit-and-perturbed parameters, ``cond_dr_0: L = 1, kd = kd__REF``.
    After import a pin reads ``kd = kd``, the identity, but the dose detector counted it as a
    second target and re-imported every dose as its own one-time time course -- which fails to
    load for a ``t_end:`` scan (BNG2.pl needs 3 sample times). The importer now drops every pin
    before any condition is classified. The pinned tables are written here by hand, the way the
    #892 exporter writes them (each pin right after its dose row)."""

    DOSE_EXP = '# L resp\n1\t0.5\n2\t1\n5\t2.5\n'
    TC_EXP = '# time resp\n0.5\t0.2\n1\t0.3\n2\t0.4\n'

    def _job(self, root, t_end):
        """A t_end: (or steady-state) scan over L, plus a time course under 'fast' (kd * 2), so
        kd is fit AND perturbed: the surrogate set M = {kd}."""
        root.mkdir()
        scan = f'type: parameter_scan, t_end: {t_end}, ' if t_end else ''
        (root / 'dr.bngl').write_text(_DR_MODEL)
        (root / 'dose.exp').write_text(self.DOSE_EXP)
        (root / 'tc.exp').write_text(self.TC_EXP)
        (root / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = sos\nmodel: dr.bngl\n'
            'condition: fast, perturbations: kd * 2\n'
            f'experiment: dr, {scan}data: dose.exp\n'
            'experiment: tc, condition: fast, data: tc.exp\n'
            'uniform_var = kd 0.1 10\n')
        return root / 'job.conf'

    def _pinned_import(self, tmp_path, t_end):
        conf = self._job(tmp_path / 'src', t_end)
        petab = tmp_path / 'petab'
        export_job(conf, petab)
        lines = []
        for line in (petab / 'conditions.tsv').read_text().splitlines():
            lines.append(line)
            if line.startswith('cond_dr_'):
                lines.append(line.split('\t')[0] + '\tkd\tkd__REF')
        (petab / 'conditions.tsv').write_text('\n'.join(lines) + '\n')
        assert ('cond_dr_0', 'kd', 'kd__REF') in {
            (r['conditionId'], r['targetId'], r['targetValue'])
            for r in _tsv_rows(petab / 'conditions.tsv')}
        assert 'kd__REF' in (petab / 'parameters.tsv').read_text()   # M = {kd}
        return tmp_path / 'src', import_job(petab / 'problem.yaml', tmp_path / 'out')

    @staticmethod
    def _closed_form_objective(kd, t_end):
        """Half the SSR by hand: dA/dt = L - kd*A, A(0) = 0, so the scan reads
        (L/kd)(1 - exp(-kd t_end)) (L/kd at steady state), and the time course under 'fast'
        (L = 1, rate 2 kd) reads (1/(2 kd))(1 - exp(-2 kd t))."""
        total = 0.0
        for L, y in ((1, 0.5), (2, 1.0), (5, 2.5)):
            pred = L / kd if t_end is None else L / kd * (1 - np.exp(-kd * t_end))
            total += 0.5 * (pred - y) ** 2
        for t, y in ((0.5, 0.2), (1, 0.3), (2, 0.4)):
            total += 0.5 * (1 / (2 * kd) * (1 - np.exp(-2 * kd * t)) - y) ** 2
        return total

    @pytest.mark.parametrize('t_end', [0.5, None])
    def test_pinned_dose_conditions_import_as_one_scan(self, t_end, tmp_path, monkeypatch):
        _src, out = self._pinned_import(tmp_path, t_end)
        text = (out / 'imported.conf').read_text()
        tend = f't_end: {t_end}, ' if t_end else ''
        assert f'experiment: dr, method: ode, {tend}data: dr.exp' in text
        assert 'experiment: tc, condition: fast, method: ode, data: tc.exp' in text
        assert 'condition: fast, perturbations: kd * 2' in text
        assert 'condition: dr_' not in text               # each dose is the scan axis again
        data = Data(file_name=str(out / 'dr.exp'))
        assert data.indvar == 'L' and list(data['L']) == [1, 2, 5]
        _load_conf(out, monkeypatch)                        # and the conf loads

    @pytest.mark.bngsim
    @pytest.mark.newera
    @pytest.mark.parametrize('t_end', [0.5, None])
    def test_pinned_scan_scores_the_source_jobs_objective(self, t_end, tmp_path, monkeypatch):
        # Oracle: at fixed kd the imported job scores what the source job scores, and both equal
        # the hand closed form over every measurement.
        src, out = self._pinned_import(tmp_path, t_end)
        for kd in (0.8, 1.5):
            expected = self._closed_form_objective(kd, t_end)
            assert _imported_objective_at(out, monkeypatch, kd=kd) == pytest.approx(
                expected, rel=1e-5)
            assert _imported_objective_at(src, monkeypatch, conf_name='job.conf',
                                          kd=kd) == pytest.approx(expected, rel=1e-5)


class TestReplicateFileNamesAreDistinct:
    """Independent review of #903: every experiment writes its data as ``<name>.exp`` and
    ``<name>_rep<k>.exp``, so an experiment whose experimentId is literally ``<other>_rep2`` and a
    replicated experiment ``<other>`` wanted the same file. The later write won, and one
    experiment was fitted against the other's measurements with no error (the imported objective
    of the scan case at k = 1 was 0.06 against 0.2826 for the PEtab problem). #903 extended the
    replicate naming to dose-response scans; the time-course case is the older ADR-0039 sibling.
    Every file name now comes from one registry (``_DataFileNames``): each experiment keeps its
    ``<name>.exp``, and a replicate whose name is taken moves to the first free ``_<n>`` suffix."""

    CASES = {
        # A replicated steady-state scan 's' (doses s_1, s_2) next to a time course 's_rep2'.
        'scan': ([('d1', 'L', 1), ('d2', 'L', 2), ('c', 'L', 3)],
                 [('s_1', '0', 'd1'), ('s_2', '0', 'd2'), ('s_rep2', '0', 'c')],
                 [('obs_A', 's_1', 'inf', 2.1), ('obs_A', 's_1', 'inf', 1.9),
                  ('obs_A', 's_2', 'inf', 3.1), ('obs_A', 's_2', 'inf', 2.8)]),
        # A replicated time course 's' next to a time course 's_rep2'.
        'time course': ([('c', 'L', 3), ('c2', 'L', 2)],
                        [('s', '0', 'c2'), ('s_rep2', '0', 'c')],
                        [('obs_A', 's', t, v + t) for v in (1.0, 1.1) for t in (0.5, 1, 2)]),
    }
    # Each case's second replicate of 's', as written: (independent-variable column, values).
    SECOND_REPLICATE = {'scan': ('L', [1, 2], [1.9, 2.8]),
                        'time course': ('time', [0.5, 1, 2], [1.6, 2.1, 3.1])}
    S_REP2 = [('obs_A', 's_rep2', t, 1.0 + t) for t in (0.5, 1, 2)]

    def _import(self, replicated, tmp_path):
        conditions, experiments, measurements = self.CASES[replicated]
        yaml = _write_periods_problem(tmp_path / 'problem', conditions, experiments,
                                      measurements + self.S_REP2)
        out = import_job(yaml, tmp_path / 'out')
        files = {}
        for line in (out / 'imported.conf').read_text().splitlines():
            if line.startswith('experiment:'):
                name = line.split(',')[0].split(':')[1].strip()
                files[name] = [f.strip() for f in line.split('data:')[1].split(',')]
        return out, files

    @pytest.mark.parametrize('replicated', ['scan', 'time course'])
    def test_no_experiment_reads_another_experiments_data_file(self, replicated, tmp_path):
        out, files = self._import(replicated, tmp_path)
        assert set(files['s']).isdisjoint(files['s_rep2'])
        tc = Data(file_name=str(out / files['s_rep2'][0]))
        assert tc.indvar == 'time' and list(tc['A_tot']) == [1.5, 2.0, 3.0]
        # The time course keeps its own name; the scan's replicate moves to the next free one,
        # and the conf's data: line names the file actually written.
        assert files['s_rep2'] == ['s_rep2.exp'] and files['s'] == ['s.exp', 's_rep2_2.exp']
        indvar, xs, ys = self.SECOND_REPLICATE[replicated]
        second = Data(file_name=str(out / 's_rep2_2.exp'))
        assert second.indvar == indvar and list(second[indvar]) == xs
        assert list(second['A_tot']) == ys

    @staticmethod
    def _petab_objective(conditions, experiments, measurements, k):
        """Half the sum of squared residuals (sigma 1) over every PEtab measurement row, from the
        closed forms of dA/dt = L + flag - k*A with A(0) = 0 and flag = 1: the steady state
        (L + 1)/k, and (L + 1)/k * (1 - exp(-k t)) at a finite time."""
        L_of = {cid: value for cid, _target, value in conditions}
        condition_of = {eid: cid for eid, _time, cid in experiments}
        total = 0.0
        for _obs, eid, t, y in measurements:
            steady = (L_of[condition_of[eid]] + 1) / k
            predicted = steady if t == 'inf' else steady * (1 - np.exp(-k * float(t)))
            total += 0.5 * (predicted - y) ** 2
        return total

    @pytest.mark.bngsim
    @pytest.mark.newera
    @pytest.mark.parametrize('replicated', ['scan', 'time course'])
    def test_imported_objective_is_the_petab_objective(self, replicated, tmp_path, monkeypatch):
        # Oracle: the imported job, simulated through bngsim, scores every PEtab measurement row
        # once (0.2826 for the scan case at k = 1; 0.06 before the fix).
        out, _files = self._import(replicated, tmp_path)
        conditions, experiments, measurements = self.CASES[replicated]
        for k in (0.7, 1.0):
            assert _imported_objective_at(out, monkeypatch, k=k) == pytest.approx(
                self._petab_objective(conditions, experiments, measurements + self.S_REP2, k),
                rel=1e-5)

    def test_names_that_differ_only_in_case_get_distinct_files(self, tmp_path):
        # 'S.exp' and 's.exp' are one file on a case-insensitive filesystem (macOS, Windows).
        yaml = _write_periods_problem(
            tmp_path / 'problem', [('c', 'L', 3)], [('S', '0', 'c'), ('s', '0', 'c')],
            [('obs_A', eid, t, v + t) for eid, v in (('S', 1.0), ('s', 2.0)) for t in (1, 2)])
        text = (import_job(yaml, tmp_path / 'out') / 'imported.conf').read_text()
        assert 'experiment: S, condition: c, method: ode, data: S.exp' in text
        assert 'experiment: s, condition: c, method: ode, data: s_2.exp' in text
        assert list(Data(file_name=str(tmp_path / 'out' / 's_2.exp'))['A_tot']) == [3.0, 4.0]

    def test_two_experiments_that_would_share_a_name_are_refused(self, tmp_path):
        # A scan named 's' (doses s_1, s_2) and a time course whose experimentId is 's' would
        # both be experiment 's' in the conf, which names each experiment once.
        yaml = _write_periods_problem(
            tmp_path / 'problem', [('d1', 'L', 1), ('d2', 'L', 2), ('c', 'L', 3)],
            [('s_1', '0', 'd1'), ('s_2', '0', 'd2'), ('s', '0', 'c')],
            [('obs_A', 's_1', 'inf', 2.0), ('obs_A', 's_2', 'inf', 3.0)]
            + [('obs_A', 's', t, 1.0 + t) for t in (1, 2)])
        with pytest.raises(PybnfError, match=r"PEtab experiment 's' and the dose-response scan "
                                             r"'s' would both import as the PyBNF experiment 's'"):
            import_job(yaml, tmp_path / 'out')


# ---------------------------------------------------------------------------
# A cond_wildtype carrying real targets (#905). The exporter's synthesized base condition
# cond_wildtype holds only surrogate base pins (p = p__REF), the identity after import. The
# importer dropped EVERY cond_wildtype, so a condition of that id with real targets -- another
# tool's, or a PyBNF condition named 'wildtype' exported before the name was reserved -- was
# lost and its experiments fitted against the unperturbed model. Now only a pins-only
# cond_wildtype is dropped; one with real targets imports under its literal id.
# ---------------------------------------------------------------------------

_WT_MODEL = """begin model
begin parameters
  k 1
  L 1
  A0 10
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() A0
end seed species
begin observables
  Molecules Atot A()
end observables
begin functions
  rate() = k*L
end functions
begin reaction rules
  A() -> 0 rate()
end reaction rules
end model
"""


def _wt_exp(L):
    """Exact data at k = 1: A(t) = 10 exp(-k L t) on t = 0, 0.2, ..., 2."""
    return '# time Atot\n' + ''.join(f'{0.2 * i:g}\t{float(10 * np.exp(-L * 0.2 * i))!r}\n'
                                     for i in range(11))


@pytest.fixture
def real_target_wildtype(tmp_path):
    """The pre-#905 export of the issue's job: experiment 'wt' applies L = 2 under a condition
    named 'wildtype', written as conditionId cond_wildtype. Built by exporting the job under
    another name and renaming the id, since the exporter now refuses the name."""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'm.bngl').write_text(_WT_MODEL)
    (src / 'wt.exp').write_text(_wt_exp(2.0))
    (src / 'kd.exp').write_text(_wt_exp(0.5))
    (src / 'job.conf').write_text(
        'edition = 2\njob_type = de\nobjective = sos\nmodel: m.bngl\n'
        'condition: wtcond, perturbations: L = 2\n'
        'condition: knockdown, perturbations: L = 0.5\n'
        'experiment: wt, condition: wtcond, data: wt.exp\n'
        'experiment: kd, condition: knockdown, data: kd.exp\n'
        'loguniform_var = k 0.01 100\n')
    petab = tmp_path / 'petab'
    export_job(src / 'job.conf', petab)
    for name in ('conditions.tsv', 'experiments.tsv'):
        (petab / name).write_text((petab / name).read_text().replace('cond_wtcond',
                                                                     'cond_wildtype'))
    return petab


def _condition_and_experiment_lines(conf):
    return {line for line in conf.read_text().splitlines()
            if line.startswith(('condition:', 'experiment:'))}


class TestRealTargetWildtypeCondition:

    def test_real_target_cond_wildtype_imports_under_its_literal_id(self, real_target_wildtype,
                                                                   tmp_path):
        assert ('cond_wildtype', 'L', '2') in {
            (r['conditionId'], r['targetId'], r['targetValue'])
            for r in _tsv_rows(real_target_wildtype / 'conditions.tsv')}
        out = import_job(real_target_wildtype / 'problem.yaml', tmp_path / 'out')
        assert _condition_and_experiment_lines(out / 'imported.conf') == {
            'condition: cond_wildtype, perturbations: L = 2',
            'condition: knockdown, perturbations: L = 0.5',
            'experiment: wt, condition: cond_wildtype, method: ode, data: wt.exp',
            'experiment: kd, condition: knockdown, method: ode, data: kd.exp'}

    def test_reexport_writes_cond_cond_wildtype_and_is_stable(self, real_target_wildtype,
                                                              tmp_path):
        # The imported name re-exports as cond_cond_wildtype (no clash with the reserved id),
        # which imports back under the same name: import -> export -> import -> export is a
        # fixed point after the first import.
        first = import_job(real_target_wildtype / 'problem.yaml', tmp_path / 'imp1')
        export_job(first / 'imported.conf', tmp_path / 'exp1')
        assert ('cond_cond_wildtype', 'L', '2') in {
            (r['conditionId'], r['targetId'], r['targetValue'])
            for r in _tsv_rows(tmp_path / 'exp1' / 'conditions.tsv')}
        assert ('wt', 'cond_cond_wildtype') in {
            (r['experimentId'], r['conditionId'])
            for r in _tsv_rows(tmp_path / 'exp1' / 'experiments.tsv')}
        second = import_job(tmp_path / 'exp1' / 'problem.yaml', tmp_path / 'imp2')
        assert (_condition_and_experiment_lines(second / 'imported.conf')
                == _condition_and_experiment_lines(first / 'imported.conf'))
        export_job(second / 'imported.conf', tmp_path / 'exp2')
        _assert_problem_round_trips(tmp_path / 'exp1', tmp_path / 'exp2')

    @pytest.mark.bngsim
    @pytest.mark.newera
    def test_imported_fit_recovers_k(self, real_target_wildtype, tmp_path, monkeypatch):
        # Oracle: the data are exact at k = 1 with L = 2 for wt, so the fit returns k = 1.
        # Before #905 wt was simulated at the default L = 1 and the fit returned k = 1.361,
        # the minimum of the wrong problem.
        out = import_job(real_target_wildtype / 'problem.yaml', tmp_path / 'out')
        best, fit = _fit_imported(out, monkeypatch, population_size=12)
        assert best < 1e-8
        assert fit['k'] == pytest.approx(1.0, rel=1e-3)

    def test_pins_are_dropped_and_real_targets_kept(self):
        # The pre-#905 exporter's cond_wildtype for a user 'wildtype' condition with a fit
        # parameter perturbed: its own surrogate op plus a base pin for the other M parameter.
        rows = [PetabConditionRow('cond_wildtype', 'j', 'j__REF'),
                PetabConditionRow('cond_wildtype', 'k', 'k__REF * 2'),
                PetabConditionRow('cond_c', 'j', 'j__REF * 3'),
                PetabConditionRow('cond_c', 'k', 'k__REF')]
        exps = [PetabExperimentRow('wt', 0.0, 'cond_wildtype')]
        kept_rows, kept_exps = drop_synthesized_wildtype(rows, exps, {'j', 'k'})
        assert kept_rows == rows and kept_exps == exps
        assert conditions_from_rows(kept_rows, {'j', 'k'}) == {
            'cond_wildtype': [('k', '*', 2.0)], 'c': [('j', '*', 3.0)]}

    def test_pins_only_cond_wildtype_still_means_no_condition(self):
        # The exporter's own base (pins only) is still machinery: its rows go, and every
        # period that applied it -- a wildtype time course, a wash-out -- reads as blank.
        rows = [PetabConditionRow('cond_wildtype', 'k', 'k__REF'),
                PetabConditionRow('cond_c', 'k', 'k__REF * 2')]
        exps = [PetabExperimentRow('wt', 0.0, 'cond_wildtype'),
                PetabExperimentRow('w', float('-inf'), 'cond_c'),
                PetabExperimentRow('w', 0.0, 'cond_wildtype')]
        kept_rows, kept_exps = drop_synthesized_wildtype(rows, exps, {'k'})
        assert kept_rows == [rows[1]]
        assert [(e.experiment_id, e.time, e.condition_id) for e in kept_exps] == [
            ('wt', 0.0, ''), ('w', float('-inf'), 'cond_c'), ('w', 0.0, '')]
        assert _condition_and_preequilibrate(
            [e for e in kept_exps if e.experiment_id == 'w'], 'w') == (None, 'c')

    @pytest.mark.parametrize('ids', [('cond_wildtype', 'cond_cond_wildtype'), ('cond_a', 'a')])
    def test_two_ids_that_import_as_one_name_are_refused(self, ids):
        rows = [PetabConditionRow(ids[0], 'L', '2'), PetabConditionRow(ids[1], 'L', '3')]
        with pytest.raises(PybnfError, match=rf"PEtab conditions '{ids[0]}' and '{ids[1]}' "
                                             r"both import as the PyBNF condition"):
            conditions_from_rows(rows, set())

    def test_colliding_ids_that_no_experiment_applies_do_not_block_the_import(self, tmp_path):
        # Independent review of #905: cond_a and a would import under one name, but neither
        # reaches the fit, so there is nothing to merge wrongly. The time course's own condition
        # must import exactly; of the unused pair only the first in table order is kept.
        yaml = _write_periods_problem(
            tmp_path / 'problem', [('cond_a', 'L', 2), ('a', 'L', 3), ('c', 'L', 4)],
            [('tc', '0', 'c')], [('obs_A', 'tc', t, 1.0 + t) for t in (0.5, 1, 2)])
        text = (import_job(yaml, tmp_path / 'out') / 'imported.conf').read_text()
        assert 'experiment: tc, condition: c, method: ode, data: tc.exp' in text
        assert 'condition: c, perturbations: L = 4' in text
        assert 'condition: a, perturbations: L = 2' in text
        assert 'L = 3' not in text

    @pytest.mark.parametrize('applied', ['cond_a', 'a'])
    def test_the_one_applied_id_of_a_colliding_pair_is_imported(self, applied, tmp_path):
        # Exactly one of the pair is applied: it imports under the shared name with its OWN
        # target, and the unused one is left out (it cannot change the fit).
        yaml = _write_periods_problem(
            tmp_path / 'problem', [('cond_a', 'L', 2), ('a', 'L', 3)],
            [('tc', '0', applied)], [('obs_A', 'tc', t, 1.0 + t) for t in (0.5, 1, 2)])
        text = (import_job(yaml, tmp_path / 'out') / 'imported.conf').read_text()
        value = {'cond_a': 2, 'a': 3}[applied]
        assert 'experiment: tc, condition: a, method: ode, data: tc.exp' in text
        assert f'condition: a, perturbations: L = {value}' in text
        assert f'L = {5 - value}' not in text

    def test_two_applied_ids_of_a_colliding_pair_are_refused(self, tmp_path):
        # Both of the pair are applied -- one as a pre-equilibration condition -- so importing
        # them under one name would merge their targets.
        yaml = _write_periods_problem(
            tmp_path / 'problem', [('cond_a', 'L', 2), ('a', 'L', 3)],
            [('tc', '-inf', 'a'), ('tc', '0', 'cond_a')],
            [('obs_A', 'tc', t, 1.0 + t) for t in (0.5, 1, 2)])
        with pytest.raises(PybnfError, match=r"PEtab conditions 'cond_a' and 'a' both import as "
                                             r"the PyBNF condition 'a', and experiments apply "
                                             r"each of them"):
            import_job(yaml, tmp_path / 'out')


# ---------------------------------------------------------------------------
# Replicates: a PEtab measurements table with repeated (experiment, observable,
# time) rows imports as one experiment binding N replicate .exp files (ADR-0039).
# ---------------------------------------------------------------------------

def _replicate_exp_text(delta):
    """A homogeneous replicate of the demo ``par1.exp``: the same ``(time, x, y)`` grid with
    the ``x``/``y`` measurements shifted by ``delta`` (the ``_SD`` columns unchanged). Same
    grid, different values -- exactly what PEtab stacks as repeated measurement rows."""
    data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
    headers = [data.headers[i] for i in range(len(data.headers))]
    lines = ['# ' + '\t'.join(headers)]
    for i in range(data.data.shape[0]):
        cells = [('%g' % (data.data[i, j] + (delta if h in ('x', 'y') else 0.0)))
                 for j, h in enumerate(headers)]
        lines.append('\t'.join(cells))
    return '\n'.join(lines) + '\n'


class TestReplicateRoundTrip:

    REPLICATE_CONF = (
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        f'model: {DEMO_MODEL}\n'
        'experiment: par1, data: par1.exp, par1b.exp\n'
        + _PARAMS_U +
        'population_size = 20\nmax_iterations = 30\nverbosity = 2\n')

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(
            tmp_path_factory.mktemp('replicate'), self.REPLICATE_CONF,
            extra_files={'par1b.exp': _replicate_exp_text(0.5)})

    def test_problem_round_trips_byte_for_byte(self, imported):
        # The dominant oracle: a two-replicate experiment exports to repeated measurement
        # rows, imports by dealing them into two grids, and re-exports byte-identically.
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_measurements_carry_both_replicates(self, imported):
        # Two replicates of a 21-point, two-observable (x, y) grid -> 2 * 21 * 2 data rows.
        petab1, _, _, _ = imported
        rows = read_measurement_table(petab1 / 'measurements.tsv')
        assert len(rows) == 2 * 21 * 2

    def test_imported_experiment_binds_two_exp_files(self, imported):
        # One experiment, two .exp files (the synthesized base name + its _rep2 sibling),
        # both on the experiment's data: list -- the inverse of the forward stacking.
        _, imported_dir, _, conf = imported
        exps = sorted(p.name for p in imported_dir.glob('*.exp'))
        assert len(exps) == 2 and any(name.endswith('_rep2.exp') for name in exps)
        with open(conf) as fh:
            d = ploop(fh.readlines())
        (_, name), fields = next(it for it in d.items()
                                 if isinstance(it[0], tuple) and it[0][0] == 'experiment')
        assert len(fields['data']) == 2

    def test_reconstructed_replicates_match_the_sources(self, imported):
        # Dealing keeps each replicate's values in its own grid: the two reconstructed .exp
        # files reproduce the two source grids cell-for-cell (order-independent: match the
        # one whose x column aligns with each source's).
        _, imported_dir, _, _ = imported
        base = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        expected = [
            {c: base[c] for c in ('time', 'x', 'y', 'x_SD', 'y_SD')},          # par1.exp
            {'time': base['time'], 'x': base['x'] + 0.5, 'y': base['y'] + 0.5,  # par1b.exp
             'x_SD': base['x_SD'], 'y_SD': base['y_SD']},
        ]
        recon = [Data(file_name=str(p)) for p in imported_dir.glob('*.exp')]
        assert len(recon) == 2
        for exp in expected:
            match = next(r for r in recon if np.allclose(r['x'], exp['x']))
            for col, vals in exp.items():
                assert np.allclose(match[col], vals), col


# ---------------------------------------------------------------------------
# Ragged replicates (issue #494): a measurement table whose replicates cover DIFFERENT
# observable subsets reconstructs to per-replicate .exp files with different column sets
# (ADR-0039 deals the extra obs_x-only replicate into a second, x-only grid). Loading that
# imported conf must not raise on the mismatched columns -- the replicates stack onto the
# union of columns, NaN-filling the cells the x-only replicate does not measure.
# ---------------------------------------------------------------------------

class TestRaggedReplicateImport:

    @pytest.fixture
    def imported_ragged(self, tmp_path):
        # Export the demo, then append a replicate that measures ONLY obs_x (a second
        # occurrence of each obs_x cell, none for func_y) -- the ragged shape #494 hit in
        # the PEtab benchmark collection (Armistead_CellDeathDis2024 et al.).
        petab = tmp_path / 'petab'
        export_job(DEMO_CONF, petab)
        mfile = petab / 'measurements.tsv'
        lines = mfile.read_text().splitlines()
        obs_x_rows = [ln for ln in lines[1:] if ln.startswith('obs_x\t')]
        assert obs_x_rows
        mfile.write_text('\n'.join(lines + obs_x_rows) + '\n')
        return import_job(petab / 'problem.yaml', tmp_path / 'out')

    def test_reconstructs_a_ragged_second_replicate(self, imported_ragged):
        # The full grid keeps the base name; the x-only replicate is the _rep2 sibling.
        exps = {p.name: Data(file_name=str(p)) for p in imported_ragged.glob('*.exp')}
        assert any(n.endswith('_rep2.exp') for n in exps)
        rep2 = next(d for n, d in exps.items() if n.endswith('_rep2.exp'))
        assert 'x' in rep2.cols and 'y' not in rep2.cols   # ragged: obs_x only

    def test_imported_ragged_conf_loads_and_pads_to_the_union(self, imported_ragged,
                                                              monkeypatch):
        # The crash path #494 reports: loading the imported conf stacked the ragged .exp
        # files and rejected their mismatched columns. It now union-pads instead.
        from pybnf import config as config_mod
        monkeypatch.chdir(imported_ragged)
        cfg = config_mod.Configuration(
            ploop((imported_ragged / 'imported.conf').read_text().splitlines(keepends=True)))
        stacked = next(iter(cfg.exp_data.values()))['experiment1']
        # Union columns; the x-only replicate's y / y_SD rows are NaN, x is measured throughout.
        assert set(stacked.cols) == {'time', 'x', 'y', 'x_SD', 'y_SD'}
        n = stacked.data.shape[0]
        assert np.isfinite(stacked['x']).sum() == n           # x measured in every row
        assert np.isfinite(stacked['y']).sum() == n // 2      # only the full replicate has y
        assert np.isnan(stacked['y']).sum() == n // 2         # the x-only replicate pads y


# ---------------------------------------------------------------------------
# Multi-model round trip (ADR-0041, #430): a two-model BNGL job exports to a PEtab problem
# with two model_files entries + a modelId column on measurements, imports back to a conf
# declaring both models (each experiment naming its model), and re-exports byte-for-byte.
# The mixed BNGL + SBML round trip lives in test_petab_sbml_layer.py (it needs RoadRunner +
# the petab math layer); this is the dependency-free BNGL-only case.
# ---------------------------------------------------------------------------

# A second BNGL model (distinct stem/parameters/observable/function) for the two-model job.
_GROWTH_BNGL = """\
begin model
  begin parameters
    a1 0.5
    a2 2
  end parameters
  begin molecule types
    cnt()
  end molecule types
  begin seed species
    cnt() 5
  end seed species
  begin observables
    Molecules p cnt()
  end observables
  begin functions
    q()=a1*p+a2
  end functions
  begin reaction rules
    0->cnt() 1
  end reaction rules
end model
"""


class TestImportMultiModelRoundTrip:

    CONF = (
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        f'model: {DEMO_MODEL}\n'
        'model: growth_v2.bngl\n'
        f'experiment: pa, model: {DEMO_MODEL}, data: pa.exp\n'
        'experiment: gr, model: growth_v2.bngl, data: gr.exp\n'
        + _PARAMS_U +
        'uniform_var = a1 0 10\nuniform_var = a2 0 10\n')

    EXTRA = {
        'growth_v2.bngl': _GROWTH_BNGL,
        'pa.exp': (DEMO_DIR / 'par1.exp').read_text(),
        'gr.exp': ('# time\tp\tq\tp_SD\tq_SD\n'
                   + ''.join(f'{t}\t{5 + t}\t{0.5 * (5 + t) + 2}\t1\t1\n' for t in range(5))),
    }

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(tmp_path_factory.mktemp('mm'), self.CONF, extra_files=self.EXTRA)

    def test_problem_round_trips_byte_for_byte(self, imported):
        # The dominant oracle: two models, each experiment's modelId stamped on its rows;
        # import recovers the model->experiment link and re-exports byte-identically.
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)

    def test_imported_conf_declares_both_models_and_per_experiment_model(self, imported):
        _, imported_dir, _, conf = imported
        text = conf.read_text()
        assert f'model: {DEMO_MODEL}' in text and 'model: growth_v2.bngl' in text
        with open(conf) as fh:
            d = ploop(fh.readlines())
        assert d['models'] == {DEMO_MODEL, 'growth_v2.bngl'}
        # Each experiment names the model it simulates (recovered from the rows' modelId).
        exp_models = {fields['model'][0] if isinstance(fields.get('model'), list)
                      else fields.get('model')
                      for k, fields in d.items()
                      if isinstance(k, tuple) and k[0] == 'experiment'}
        assert exp_models == {DEMO_MODEL, 'growth_v2.bngl'}

    def test_both_models_carried_verbatim(self, imported):
        petab1, imported_dir, _, _ = imported
        for name in (DEMO_MODEL, 'growth_v2.bngl'):
            assert (imported_dir / name).read_text() == (petab1 / name).read_text()

    def test_each_experiment_reconstructs_its_own_data(self, imported):
        _, imported_dir, _, _ = imported
        # parabola measures x/y; growth measures p/q -- each .exp carries only its columns.
        recon = {p.name: Data(file_name=str(p)) for p in imported_dir.glob('*.exp')}
        assert len(recon) == 2
        cols = [set(d.cols) for d in recon.values()]
        assert {'time', 'x', 'y', 'x_SD', 'y_SD'} in cols
        assert {'time', 'p', 'q', 'p_SD', 'q_SD'} in cols

    def test_imported_multimodel_conf_loads_as_a_configuration(self, imported, monkeypatch):
        # The imported conf is a genuinely runnable multi-model job (ADR-0028/0034 already
        # run it; ADR-0041 verifies the round trip emits a loadable one): both models load,
        # each experiment's data binds to the model it names, and the union of free
        # parameters binds across the two models. Simulator-free (no fit).
        from pybnf import config as config_mod
        _, imported_dir, _, conf = imported
        monkeypatch.chdir(imported_dir)
        cfg = config_mod.Configuration(ploop(conf.read_text().splitlines(keepends=True)))
        assert set(cfg.models) == {'parabola_v2', 'growth_v2'}
        assert set(cfg.exp_data) == {'parabola_v2', 'growth_v2'}   # data bound per-model
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 'a1', 'a2'}


class TestImportMultiModelCondition:
    """A model-scoped ``condition:`` in a multi-model job round-trips (#444 item 4,
    ADR-0041 addendum). PEtab conditions are model-agnostic (no modelId column); a PyBNF
    condition belongs to ONE model, and the fitter *requires* ``model:`` on a condition
    when the job declares more than one model. So the importer recovers the condition's
    owning model from the experiment that applies it -- without this the imported conf
    raised ``Condition '<name>' does not name a model, but the job declares 2 models``."""

    _EXTRA = {
        'growth_v2.bngl': _GROWTH_BNGL,
        'pa.exp': (DEMO_DIR / 'par1.exp').read_text(),
        'gr.exp': ('# time\tp\tq\tp_SD\tq_SD\n'
                   + ''.join(f'{t}\t{5 + t}\t{0.5 * (5 + t) + 2}\t1\t1\n' for t in range(5))),
    }

    def _conf(self, pert, growth_fit):
        # growth's a1 is FIXED when only a2 is declared fit (numeric condition target);
        # declaring a1 fit too exercises the surrogate (`a1__REF`) path (ADR-0027).
        return (
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            f'model: {DEMO_MODEL}\nmodel: growth_v2.bngl\n'
            f'condition: hi, model: growth_v2.bngl, perturbations: {pert}\n'
            f'experiment: pa, model: {DEMO_MODEL}, data: pa.exp\n'
            'experiment: gr, model: growth_v2.bngl, condition: hi, data: gr.exp\n'
            + _PARAMS_U + ''.join(f'uniform_var = {p} 0 10\n' for p in growth_fit))

    @pytest.mark.parametrize('pert,growth_fit', [
        ('a1 = 5', ['a2']),          # fixed target -> numeric condition value
        ('a1 / 2', ['a1', 'a2']),    # fit target  -> surrogate `a1__REF / 2`
    ])
    def test_model_scoped_condition_round_trips_and_loads(self, tmp_path, monkeypatch,
                                                          pert, growth_fit):
        petab1, imported, petab2, conf = _roundtrip(
            tmp_path, self._conf(pert, growth_fit), extra_files=self._EXTRA)
        # Problem round-trips byte-for-byte (the condition's model: doesn't alter PEtab --
        # PEtab conditions are model-agnostic, so conditions.tsv is identical either way).
        _assert_problem_round_trips(petab1, petab2)
        # The condition recovered its owning model from experiment `gr`.
        assert f'condition: hi, model: growth_v2.bngl, perturbations: {pert}' in conf.read_text()
        # And the multi-model conf now LOADS (the bug: it raised without the model: ref).
        from pybnf import config as config_mod
        monkeypatch.chdir(imported)
        cfg = config_mod.Configuration(ploop(conf.read_text().splitlines(keepends=True)))
        assert set(cfg.models) == {'parabola_v2', 'growth_v2'}

    def test_condition_shared_across_models_is_refused(self, tmp_path):
        """A PEtab condition applied by experiments on *different* models has no PyBNF
        representation (a condition belongs to one model) -> a clear boundary error,
        not a silently-unfittable conf."""
        from pybnf.petab.import_ import _write_conf, ImportedExperiment
        exps = [
            ImportedExperiment('e1', 'shared', None, ['e1.exp'], 'parabola_v2.bngl', None, None),
            ImportedExperiment('e2', 'shared', None, ['e2.exp'], 'growth_v2.bngl', None, None),
        ]
        with pytest.raises(NotImplementedError, match='different models'):
            _write_conf(
                tmp_path / 'x.conf', model_filenames=['parabola_v2.bngl', 'growth_v2.bngl'],
                job_type='de', objective_directives=['objective = chi_sq'],
                free_param_lines=[], conditions={'shared': [('a1', '=', 5.0)]},
                experiments=exps, measurement_models=[], method='ode', method_overrides={},
                settings={'population_size': 10, 'max_iterations': 5, 'verbosity': 1},
                multi=False)


# ---------------------------------------------------------------------------
# Extensions: prior catalog, objective family, conditions, emit-all
# ---------------------------------------------------------------------------

class TestImportExtensionsRoundTrip:

    def test_loguniform_prior_round_trips(self, tmp_path):
        petab1, _, petab2, conf = _roundtrip(
            tmp_path, _HEAD + 'experiment: par1, data: par1.exp\n'
            'loguniform_var = v1 0.1 10\nloguniform_var = v2 0.1 10\n'
            'loguniform_var = v3 0.1 10\n')
        _assert_problem_round_trips(petab1, petab2)
        assert 'loguniform_var = v1 0.1 10' in conf.read_text()

    @pytest.mark.parametrize('line,dist,params', [
        ('cauchy_var = {p} 0 1', 'cauchy', '0;1'),
        ('gamma_var = {p} 2 3', 'gamma', '2;3'),
        ('exponential_var = {p} 0.5', 'exponential', '0.5'),
        ('chisquare_var = {p} 4', 'chisquare', '4'),
        ('rayleigh_var = {p} 1.5', 'rayleigh', '1.5'),
    ])
    def test_catalog_prior_family_round_trips(self, tmp_path, line, dist, params):
        # The five v2 catalog families (#417), each bidirectional: a native *_var line exports
        # to its PEtab priorDistribution + priorParameters and imports back byte-for-byte. The
        # one-parameter families (exponential/chisquare/rayleigh) exercise the one-number form.
        body = '\n'.join(line.format(p=p) for p in ('v1', 'v2', 'v3')) + '\n'
        petab1, _, petab2, conf = _roundtrip(
            tmp_path, _HEAD + 'experiment: par1, data: par1.exp\n' + body)
        _assert_problem_round_trips(petab1, petab2)
        params_tsv = (petab1 / 'parameters.tsv').read_text()
        assert dist in params_tsv and params in params_tsv
        assert line.format(p='v1') in conf.read_text()

    @pytest.mark.parametrize('dist,params,p1,p2', [
        ('cauchy', (0.0, 2.0), 0.0, 2.0),
        ('gamma', (2.0, 3.0), 2.0, 3.0),
        ('exponential', (0.5,), 0.5, None),
        ('chisquare', (4.0,), 4.0, None),
        ('rayleigh', (1.5,), 1.5, None),
    ])
    def test_bounded_catalog_prior_imports_as_truncated(self, dist, params, p1, p2):
        # A real PEtab catalog prior carries finite bounds (PEtab requires them on estimated
        # parameters), so it imports as a *truncated* FreeParameter: the family's parameters
        # (p1[/p2]) plus the [lb, ub] reflecting box (ADR-0020). This is the import-direction
        # unit oracle; the native .conf has no truncation grammar for these families, so the
        # bounded form does not byte-round-trip through a re-export (only the unbounded form
        # does, above) -- the same pre-existing limitation normal/laplace have.
        from pybnf.petab.parameters import PetabParameterRow, free_parameter_from_row
        row = PetabParameterRow(parameter_id='k', estimate=True, lower_bound=0.0,
                                upper_bound=50.0, prior_distribution=dist,
                                prior_parameters=params)
        fp = free_parameter_from_row(row)
        assert fp.type == f'{dist}_var'
        assert fp.p1 == p1 and fp.p2 == p2
        assert fp.bounded and fp.trunc_lb == 0.0 and fp.trunc_ub == 50.0

    @pytest.mark.parametrize('objective', ['chi_sq', 'sos', 'sod', 'ave_norm_sos'])
    def test_objective_family_round_trips(self, tmp_path, objective):
        petab1, _, petab2, conf = _roundtrip(
            tmp_path, f'edition = 2\njob_type = de\nobjective = {objective}\n'
            f'model: {DEMO_MODEL}\nexperiment: par1, data: par1.exp\n' + _PARAMS_U)
        _assert_problem_round_trips(petab1, petab2)
        # The objective token is recovered from the observables' noise columns.
        assert f'objective = {objective}' in conf.read_text()

    @pytest.mark.parametrize('family,param', [('gaussian', 'sigma'), ('laplace', 'scale')])
    def test_uniform_fixed_sigma_recovers_as_noise_model_line(self, tmp_path, family, param):
        # A uniform non-unit fixed sigma is named by no sugar token (sos/sod are the unit
        # case, ave_norm_sos the column-mean case), so it recovers as the symmetric inverse
        # of the exporter's whole-fit noise_model line -- and round-trips byte-for-byte.
        petab1, _, petab2, conf = _roundtrip(
            tmp_path,
            f'edition = 2\njob_type = de\nmodel: {DEMO_MODEL}\n'
            f'noise_model = {family}, {param} = fix_at 2.5\n'
            'experiment: par1, data: par1.exp\n' + _PARAMS_U)
        _assert_problem_round_trips(petab1, petab2)
        text = conf.read_text()
        assert f'noise_model = {family}, {param} = fix_at 2.5' in text
        assert 'objective =' not in text     # a noise_model line, not an objective token

    def test_free_parameter_sigma_recovers_as_fit_noise_model_line(self, tmp_path):
        # A bare-id noiseFormula naming an estimated parameter recovers as a 'fit'
        # noise_model line, connecting observables<->parameters by name. Import-only: the
        # exporter raises on a fit sigma, so there is no byte-equal round trip -- this is
        # external-problem territory. Built by pointing a sos export's constant noiseFormula
        # (no noiseParameters) at one shared estimated sigma parameter.
        src = tmp_path / 'src'
        src.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\nobjective = sos\nmodel: {DEMO_MODEL}\n'
            'experiment: par1, data: par1.exp\n' + _PARAMS_U)
        petab = src / 'petab'
        export_job(src / 'job.conf', petab)
        obs = (petab / 'observables.tsv').read_text()
        (petab / 'observables.tsv').write_text(obs.replace('\t1\tnormal', '\tnoise_sd\tnormal'))
        params = (petab / 'parameters.tsv').read_text()
        (petab / 'parameters.tsv').write_text(params + 'noise_sd\ttrue\t0\t10\n')

        out = import_job(petab / 'problem.yaml', tmp_path / 'out')
        text = (out / 'imported.conf').read_text()
        # New-era binds by id (ADR-0034): the shared sigma id 'noise_sd' connects to its
        # emitted bare free parameter 'noise_sd' (a nuisance -- it matches no model id).
        assert 'noise_model = gaussian, sigma = fit noise_sd' in text
        assert 'uniform_var = noise_sd 0 10' in text
        assert 'objective =' not in text
        with open(out / 'imported.conf') as fh:
            d = ploop(fh.readlines())
        assert ('uniform_var', 'noise_sd') in d

    def test_conditions_round_trip(self, tmp_path):
        extra = {
            'parabola2.bngl': _PARABOLA2_BNGL,
            'wt.exp': '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n',
            'dbl.exp': '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n',
            'scl.exp': '# time x y x_SD y_SD\n0\t-10\t430\t1\t1\n1\t-9\t345\t1\t1\n',
        }
        petab1, _, petab2, conf = _roundtrip(
            tmp_path,
            'edition = 2\njob_type = de\nobjective = chi_sq\nmodel: parabola2.bngl\n'
            'condition: doubled, perturbations: v1 * 2\n'
            'condition: scaled, perturbations: s * 5\n'
            'experiment: wt, data: wt.exp\n'
            'experiment: dbl, condition: doubled, data: dbl.exp\n'
            'experiment: scl, condition: scaled, data: scl.exp\n' + _PARAMS_U,
            extra_files=extra, model_name='parabola2.bngl')
        _assert_problem_round_trips(petab1, petab2)
        text = conf.read_text()
        # The fit-and-perturbed v1 recovers its relative op; the fixed s*5 recovers as the
        # precomputed absolute set (s = 10), which re-exports to the same PEtab value.
        assert 'condition: doubled, perturbations: v1 * 2' in text
        assert 'condition: scaled, perturbations: s = 10' in text
        assert 'condition: wt' not in text   # the synthesized wildtype base is not a condition:

    def test_method_is_emitted_per_experiment(self, tmp_path):
        # The simulation method is supplied (not recovered) on every experiment line.
        src = tmp_path / 'src'
        src.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(_HEAD + 'experiment: par1, data: par1.exp\n' + _PARAMS_U)
        export_job(src / 'job.conf', src / 'petab')
        import_job(src / 'petab' / 'problem.yaml', src / 'imported',
                   method='ssa')
        assert 'method: ssa' in (src / 'imported' / 'imported.conf').read_text()

    def test_emit_all_writes_one_conf_per_optimizer_and_sampler(self, tmp_path):
        src = tmp_path / 'src'
        src.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(_HEAD + 'experiment: par1, data: par1.exp\n' + _PARAMS_U)
        export_job(src / 'job.conf', src / 'petab')
        import_job(src / 'petab' / 'problem.yaml', src / 'imported', job_type='all')

        import pybnf.algorithms  # noqa: F401 -- populate the registry
        from pybnf.registry import FIT_TYPE_REGISTRY
        expected = {f'imported_{c}.conf' for c, e in FIT_TYPE_REGISTRY.items()
                    if e.family in ('optimizer', 'sampler')}
        confs = {f.name for f in (src / 'imported').glob('*.conf')}
        assert confs == expected
        assert 'imported_check.conf' not in confs    # the checker is excluded
        # every emitted conf parses and names its own job_type
        for conf in (src / 'imported').glob('*.conf'):
            jt = conf.stem[len('imported_'):]
            with open(conf) as fh:
                assert ploop(fh.readlines())['job_type'] == jt


# ---------------------------------------------------------------------------
# Per-observable noise import (ADR-0037) -- the Boehm shape, dependency-free
#
# A crafted BNGL-native problem (bare-name observables, so no petab extra) where each
# observable carries a distinct estimated sigma via a named noiseFormula placeholder bound
# to a constant-per-observable parameter-id noiseParameters. Imports to one per-observable
# noise_model line per observable, with no _SD columns (the sigma is a fit parameter).
# ---------------------------------------------------------------------------

class TestPerObservableNoiseImport:

    def _problem(self, tmp_path):
        prob = tmp_path / 'prob'
        prob.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, prob / DEMO_MODEL)
        (prob / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\n'
            'v1\ttrue\t0\t10\nv2\ttrue\t0\t10\nv3\ttrue\t0\t10\n'
            'sd_x\ttrue\t1e-3\t100\nsd_y\ttrue\t1e-3\t100\n')
        # Bare-name observableFormulas (x, y) + a named noiseFormula placeholder per observable.
        (prob / 'observables.tsv').write_text(
            'observableId\tobservableFormula\tnoiseFormula\tnoisePlaceholders\tnoiseDistribution\n'
            'obs_x\tx\tsigma_x\tsigma_x\tnormal\n'
            'obs_y\ty\tsigma_y\tsigma_y\tlaplace\n')
        rows = ''.join(
            f'{oid}\texp1\t{t}\t{v}\t{pid}\n'
            for oid, pid, vals in (('obs_x', 'sd_x', (1.0, 2.0)), ('obs_y', 'sd_y', (3.0, 4.0)))
            for t, v in zip((0.0, 1.0), vals))
        (prob / 'measurements.tsv').write_text(
            'observableId\texperimentId\ttime\tmeasurement\tnoiseParameters\n' + rows)
        (prob / 'conditions.tsv').write_text('conditionId\n')
        (prob / 'experiments.tsv').write_text('experimentId\ttime\tconditionId\n')
        (prob / 'problem.yaml').write_text(
            'format_version: 2.0.0\n'
            'parameter_files:\n  - parameters.tsv\n'
            'observable_files:\n  - observables.tsv\n'
            'measurement_files:\n  - measurements.tsv\n'
            'condition_files:\n  - conditions.tsv\n'
            'experiment_files:\n  - experiments.tsv\n'
            f'model_files:\n  m:\n    location: {DEMO_MODEL}\n    language: bngl\n')
        return prob / 'problem.yaml'

    def test_distinct_per_observable_sigmas_import_as_noise_model_lines(self, tmp_path):
        # Bare-name observables, so the dependency-free tier reaches this (no petab needed).
        out = import_job(self._problem(tmp_path), tmp_path / 'out')
        text = (out / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text          # the structural base
        # Each observable's own estimated sigma, by the column it measures and its family.
        assert 'noise_model x = gaussian, sigma = fit sd_x' in text
        assert 'noise_model y = laplace, scale = fit sd_y' in text
        # The sigma is a fit parameter, not per-point data -> no _SD columns in the .exp.
        exp = Data(file_name=str(out / 'exp1.exp'))
        assert set(exp.cols) == {'time', 'x', 'y'}
        # And it parses + binds (the sigma ids are recognized nuisances, ADR-0034).
        conf = ploop(text.splitlines(keepends=True))
        assert ('uniform_var', 'sd_x') in conf and ('uniform_var', 'sd_y') in conf


# ---------------------------------------------------------------------------
# Reverse-asset unit tests (the seam, not the orchestrator)
# ---------------------------------------------------------------------------

class TestReverseAssets:

    def test_measurement_pivot_inverts_to_identical_rows(self):
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        column_to_id = {'x': 'obs_x', 'y': 'func_y'}
        rows = measurement_rows_from_data(data, column_to_id, experiment_id='')
        datas = data_from_measurement_rows(rows, {'obs_x': 'x', 'func_y': 'y'})
        # No repeats -> a single reconstructed replicate, re-pivoting to the same rows
        # (the long<->wide inverse). Single-model -> the ('', '') (experimentId, modelId) key.
        assert len(datas[('', '')]) == 1
        again = measurement_rows_from_data(datas[('', '')][0], column_to_id, experiment_id='')
        assert rows == again

    def test_measurement_no_noise_yields_no_sd_columns(self):
        # A fixed/column-mean sigma objective writes no noiseParameters, so no _SD columns
        # are reconstructed (mirrors what a sos/ave_norm_sos re-export reads).
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        rows = measurement_rows_from_data(data, {'x': 'obs_x'}, sd_suffix=None)
        recon = data_from_measurement_rows(rows, {'obs_x': 'x'})[('', '')][0]
        assert set(recon.cols) == {'time', 'x'}

    def test_repeated_observation_deals_into_replicates(self):
        # PEtab models replicates as repeated (observable, time) rows with no replicate
        # index; the importer deals the k-th occurrence into the k-th grid (ADR-0039), the
        # inverse of the forward export's per-replicate stacking. Two stacked copies of one
        # grid reconstruct as two identical replicate Data objects, each re-pivoting to the
        # one grid's rows -- so concatenating them reproduces the doubled long table.
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        rows = measurement_rows_from_data(data, {'x': 'obs_x'})
        reps = data_from_measurement_rows(rows + rows, {'obs_x': 'x'})[('', '')]
        assert len(reps) == 2
        for rep in reps:
            assert measurement_rows_from_data(rep, {'x': 'obs_x'}) == rows
        relaid = (measurement_rows_from_data(reps[0], {'x': 'obs_x'})
                  + measurement_rows_from_data(reps[1], {'x': 'obs_x'}))
        assert relaid == rows + rows

    def test_ragged_replicates_deal_lower_count_into_first_grid(self):
        # A cell measured once goes to the first grid only; a cell measured twice spills a
        # second grid. The first grid is the full one (it sees every cell first).
        def row(oid, t, m):
            return PetabMeasurementRow(observable_id=oid, time=t, measurement=m)
        rows = [row('obs_x', 0.0, 1.0), row('obs_x', 0.0, 2.0),  # x@0 twice
                row('obs_x', 1.0, 3.0)]                            # x@1 once
        reps = data_from_measurement_rows(rows, {'obs_x': 'x'})[('', '')]
        assert len(reps) == 2
        assert np.allclose(reps[0]['x'], [1.0, 3.0])               # full grid, first values
        assert np.allclose(reps[1]['time'], [0.0])                 # only the repeated cell
        assert np.allclose(reps[1]['x'], [2.0])

    def test_noise_parameter_ids_per_observable_classifies_constant_and_row_varying(self):
        # A constant-per-observable parameter id is a per-observable sigma (Phase 1); a
        # row-varying id now routes to the per-measurement binding table (ADR-0045), not an
        # error. An id/numeric MIX is still the deferred frontier.
        def row(oid, t, pid=None, num=None):
            return PetabMeasurementRow(observable_id=oid, time=t, measurement=1.0,
                                       noise_parameter_id=pid, noise_parameters=num)
        ok = [row('a', 0, pid='sd_a'), row('a', 1, pid='sd_a'), row('b', 0, pid='sd_b')]
        assert noise_parameter_ids_by_observable(ok) == {'a': 'sd_a', 'b': 'sd_b'}
        assert row_varying_noise_ids(ok) == set()
        # Differing ids across the rows: excluded from the constant map, surfaced as row-varying.
        rv = [row('a', 0, pid='sd_a'), row('a', 1, pid='sd_a2')]
        assert noise_parameter_ids_by_observable(rv) == {}
        assert row_varying_noise_ids(rv) == {'a'}
        # The per-experiment binding table maps the column's noiseParameter1 placeholder to the
        # row's id, keyed by (experiment, model) and time (ADR-0045).
        assert measurement_param_bindings(rv, {'a': 'ya'}, {'a'}) == {
            ('', ''): {'ya': {'noiseParameter1_a': {0: 'sd_a', 1: 'sd_a2'}}}}
        # An id/numeric mix is still deferred.
        with pytest.raises(NotImplementedError, match='source kind'):
            noise_parameter_ids_by_observable([row('a', 0, pid='sd_a'), row('a', 1, num=2.0)])

    def test_observable_parameters_per_observable_classifies_constant_and_row_varying(self):
        # A constant-per-observable observableParameters tuple reduces to a per-observable
        # scale/offset (ADR-0044); a row-varying tuple (or a row that mixes a value with a
        # blank) routes to the per-measurement binding table instead (ADR-0045), so it is
        # absent from the constant map and present in the row-varying set.
        def row(oid, t, op=()):
            return PetabMeasurementRow(observable_id=oid, time=t, measurement=1.0,
                                       observable_parameters=op)
        ok = [row('a', 0, ('scaling',)), row('a', 1, ('scaling',)),
              row('b', 0, ('s', 'o')), row('c', 0)]   # c blank -> absent from both
        assert observable_parameters_by_observable(ok) == {'a': ('scaling',), 'b': ('s', 'o')}
        assert row_varying_observable_ids(ok) == set()
        rv = [row('a', 0, ('s1',)), row('a', 1, ('s2',))]            # differing per row
        assert observable_parameters_by_observable(rv) == {}
        assert row_varying_observable_ids(rv) == {'a'}
        mixed = [row('a', 0, ('s1',)), row('a', 1)]                  # value mixed with a blank
        assert observable_parameters_by_observable(mixed) == {}
        assert row_varying_observable_ids(mixed) == {'a'}

    def test_conditions_inverse_recovers_perturbations(self):
        exps = [('wt', None), ('dbl', 'doubled'), ('scl', 'scaled')]
        conds = {'doubled': [('v1', '*', 2.0)], 'scaled': [('s', '*', 5.0)]}
        cond_rows, _, surrogate, _ = build_experiment_conditions(
            exps, conds, fit_params={'v1', 'v2', 'v3'}, nominal_of=lambda v: 2.0)
        recovered = conditions_from_rows(cond_rows, surrogate)
        # The fit op recovers exactly; the fixed relative op recovers as its precomputed
        # absolute value (s*5 with nominal 2 -> s = 10); base pins are dropped.
        assert recovered == {'doubled': [('v1', '*', 2.0)], 'scaled': [('s', '=', 10.0)]}

    def test_conditions_from_rows_recovers_parameter_reference(self):
        # A per-condition estimated initial condition (ADR-0076): a targetValue that names a
        # free parameter recovers a parameter-reference perturbation (val a STRING naming it);
        # a fixed-parameter targetValue inlines its nominal value; a number is an absolute set.
        rows = [
            PetabConditionRow('cond_uCA', 'I0_', 'I0_CA'),     # free -> reference (string val)
            PetabConditionRow('cond_uCA', 'N_', '39560000'),   # a plain number
            PetabConditionRow('cond_uCA', 'g_', 'g_fixed'),    # fixed -> inlined nominal value
        ]
        recovered = conditions_from_rows(
            rows, surrogate_params=set(), free_names={'I0_CA'},
            fixed_params={'g_fixed': 0.25})
        assert recovered == {'uCA': [('I0_', '=', 'I0_CA'), ('N_', '=', 39560000.0),
                                     ('g_', '=', 0.25)]}

    def test_conditions_from_rows_multisymbol_expression_still_raises(self):
        # A multi-symbol condition formula is still the deferred sympy-layer boundary; only a
        # single parameter reference (or number) is recovered (ADR-0076).
        rows = [PetabConditionRow('cond_c', 'x', 'a * b + c')]
        with pytest.raises(NotImplementedError, match='expression'):
            conditions_from_rows(rows, surrogate_params=set(), free_names={'a', 'b', 'c'})


# ---------------------------------------------------------------------------
# problem.yaml reader unit + documented boundaries
# ---------------------------------------------------------------------------

class TestProblemYamlReader:

    def test_reads_the_exporter_shape(self, tmp_path):
        export_job(DEMO_CONF, tmp_path / 'petab')
        problem = read_problem_yaml(tmp_path / 'petab' / 'problem.yaml')
        assert problem['parameter_files'] == ['parameters.tsv']
        assert problem['observable_files'] == ['observables.tsv']
        assert problem['measurement_files'] == ['measurements.tsv']
        assert problem['model_file'] == DEMO_MODEL
        assert problem['condition_files'] == [] and problem['experiment_files'] == []

    def test_reads_petab1to2_column0_list_shape(self, tmp_path):
        # The official petab.v2.petab1to2 converter emits table-file lists at column 0
        # (`- item`, YAML-legal) rather than the two-space-indented items our own writer
        # emits. A column-0 list item must be read as belonging to the current section,
        # not treated as a new top-level key -- the latter silently dropped every table
        # file, so the whole problem imported as "has no parameter_files". Regression for
        # the petab1to2 import path (an externally-authored v2 problem.yaml).
        yaml_path = tmp_path / 'problem.yaml'
        yaml_path.write_text(
            'format_version: 2.0.0\n'
            'parameter_files:\n'
            '- parameters.tsv\n'
            'model_files:\n'
            '  model:\n'
            '    location: model.xml\n'
            '    language: sbml\n'
            'measurement_files:\n'
            '- measurements.tsv\n'
            'condition_files:\n'
            '- conditions.tsv\n'
            'experiment_files: []\n'
            'observable_files:\n'
            '- observables.tsv\n'
            'mapping_files: []\n'
            'extensions: {}\n'
        )
        problem = read_problem_yaml(yaml_path)
        assert problem['parameter_files'] == ['parameters.tsv']
        assert problem['observable_files'] == ['observables.tsv']
        assert problem['measurement_files'] == ['measurements.tsv']
        assert problem['condition_files'] == ['conditions.tsv']
        assert problem['experiment_files'] == []
        assert problem['model_file'] == 'model.xml'
        assert problem['model_id'] == 'model'
        assert problem['model_language'] == 'sbml'

    def test_column0_and_indented_lists_read_identically(self, tmp_path):
        # The reader is a strict superset of both list shapes: the same problem written
        # with column-0 (petab1to2) and two-space-indented (our writer) list items parses
        # to the same result. Guards against a future scan change that re-honors only one.
        head = ('format_version: 2.0.0\n'
                'model_files:\n  m:\n    location: m.xml\n    language: sbml\n')
        keys = ('parameter_files', 'observable_files', 'measurement_files')
        col0 = head + ''.join(f'{k}:\n- {k[:1]}.tsv\n' for k in keys)
        indented = head + ''.join(f'{k}:\n  - {k[:1]}.tsv\n' for k in keys)
        (tmp_path / 'col0.yaml').write_text(col0)
        (tmp_path / 'indented.yaml').write_text(indented)
        assert (read_problem_yaml(tmp_path / 'col0.yaml')
                == read_problem_yaml(tmp_path / 'indented.yaml'))


class TestBoundaries:

    @pytest.fixture
    def demo_petab(self, tmp_path):
        out = tmp_path / 'petab'
        export_job(DEMO_CONF, out)
        return out

    def _import_mutated(self, demo_petab, tmp_path, edits):
        prob = tmp_path / 'prob'
        shutil.copytree(demo_petab, prob)
        for name, (old, new) in edits.items():
            text = (prob / name).read_text()
            assert old in text
            (prob / name).write_text(text.replace(old, new))
        return import_job(prob / 'problem.yaml', tmp_path / 'out')

    def test_unsupported_model_language_is_refused(self, demo_petab, tmp_path):
        # BNGL and SBML import (ADR-0036); any other model language is out of scope and
        # refused before any table is read (read_problem_yaml stays a pure reader; the
        # importer holds the policy in _require_supported_model).
        with pytest.raises(NotImplementedError, match="'bngl' or 'sbml'"):
            self._import_mutated(demo_petab, tmp_path,
                                 {'problem.yaml': ('language: bngl', 'language: pysb')})

    @pytest.mark.parametrize('distribution', ['neg_bin', 'log-laplace'])
    def test_petab_inexpressible_noise_is_refused(self, demo_petab, tmp_path, distribution):
        with pytest.raises(NotImplementedError):
            self._import_mutated(demo_petab, tmp_path,
                                 {'observables.tsv': ('normal', distribution)})

    def test_expression_observable_formula_becomes_a_measurement_model(self, demo_petab,
                                                                       tmp_path):
        # An expression observableFormula imports as a measurement model evaluated
        # post-simulation (ADR-0036), NOT a function synthesized into the model: the model is
        # carried verbatim and the conf gains an `observable: obs_x, formula: x + 1` line.
        pytest.importorskip('petab')
        out = self._import_mutated(
            demo_petab, tmp_path, {'observables.tsv': ('obs_x\tx\t', 'obs_x\tx + 1\t')})
        ent = parse_model((out / 'parabola_v2.bngl').read_text())
        assert 'obs_x' not in ent.function_bodies     # NO synthesis into the model
        conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
        meas = {k[1]: v for k, v in conf.items()
                if isinstance(k, tuple) and k[0] == 'measurement'}
        assert meas == {'obs_x': 'x + 1'}             # the measurement-model formula line

    def test_unknown_symbol_in_observable_formula_raises(self, demo_petab, tmp_path):
        # A free symbol that is no model entity is an error, never a silent free parameter.
        pytest.importorskip('petab')
        with pytest.raises(PybnfError, match='not a known model entity'):
            self._import_mutated(demo_petab, tmp_path,
                                 {'observables.tsv': ('obs_x\tx\t', 'obs_x\tx + nope\t')})

    def test_observable_parameter_placeholder_is_deferred(self, demo_petab, tmp_path):
        # A per-measurement observableParameter* placeholder has no PyBNF analogue (the
        # frontier ADR-0035 keeps deferred); it raises pointing there, not synthesizes.
        pytest.importorskip('petab')
        with pytest.raises(NotImplementedError, match='placeholder'):
            self._import_mutated(
                demo_petab, tmp_path,
                {'observables.tsv': ('obs_x\tx\t', 'obs_x\tx*observableParameter1_obs_x\t')})

    def test_unknown_prior_distribution_is_refused(self, demo_petab, tmp_path):
        # An unrecognized priorDistribution spelling is a malformed problem, not a gap.
        prob = tmp_path / 'prob'
        shutil.copytree(demo_petab, prob)
        (prob / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\tpriorDistribution\t'
            'priorParameters\n'
            'v1\ttrue\t0\t10\tnonesuch\t0;1\n'
            'v2\ttrue\t0\t10\t\t\n'
            'v3\ttrue\t0\t10\t\t\n')
        with pytest.raises(PybnfError, match='priorDistribution'):
            import_job(prob / 'problem.yaml', tmp_path / 'out')

    def test_one_sided_truncation_imports_half_bounded(self, demo_petab, tmp_path, monkeypatch):
        # A finite wall on one side with the other covering the support maps to a
        # half-bounded box -- a single reflecting wall, the ub->inf limit of the fold
        # (ADR-0047, #432). gamma [5, inf): a wall at 5, open above.
        from pybnf import config as config_mod
        from pybnf.parse import ploop
        prob = tmp_path / 'prob'
        shutil.copytree(demo_petab, prob)
        (prob / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\tpriorDistribution\t'
            'priorParameters\n'
            'v1\ttrue\t5\tinf\tgamma\t2;3\n'
            'v2\ttrue\t0\t10\t\t\n'
            'v3\ttrue\t0\t10\t\t\n')
        out = import_job(prob / 'problem.yaml', tmp_path / 'out')
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        v1 = next(v for v in cfg.variables if v.name == 'v1')
        assert v1.bounded and v1.has_bounded_support
        assert v1.lower_bound == 5.0 and v1.upper_bound == np.inf


# ---------------------------------------------------------------------------
# A re-injected observableTransformation routes to the native scaled family (issues #499/#509)
#
# The bug: a v1 log10 observable, its transformation dropped by petab1to2, imported as a
# linear gaussian (objective = chi_sq) and scored the WRONG objective (linear residual, no
# Jacobian). With the column re-injected (convert.py) the importer selects the additive scale
# from it -- log10 -> the native lognormal family (Gaussian(LOG10), the base the paper scores
# on), log -> lnnormal (Gaussian(LN), PEtab's natural-log scale). Dependency-free: the demo is
# BNGL, bare-name observables.
# ---------------------------------------------------------------------------

class TestObservableTransformationImport:

    @pytest.fixture
    def demo_petab(self, tmp_path):
        out = tmp_path / 'petab'
        export_job(DEMO_CONF, out)
        return out

    def _import_with_transformation(self, demo_petab, tmp_path, transformation,
                                    distribution=None, noise_formula=None):
        # Append an observableTransformation column (optionally rewriting the distribution /
        # noiseFormula) to the exported demo observables -- the shape the scale-preserving
        # converter produces for a v1 log observable -- then import.
        prob = tmp_path / 'prob'
        shutil.copytree(demo_petab, prob)
        rows = _tsv_rows(prob / 'observables.tsv')
        header = list(rows[0].keys()) + ['observableTransformation']
        lines = ['\t'.join(header)]
        for r in rows:
            if distribution is not None:
                r['noiseDistribution'] = distribution
            if noise_formula is not None:
                r['noiseFormula'] = noise_formula
                r['noisePlaceholders'] = ''     # a constant sigma declares no placeholder
            lines.append('\t'.join([r[c] for c in header[:-1]] + [transformation]))
        (prob / 'observables.tsv').write_text('\n'.join(lines) + '\n')
        return import_job(prob / 'problem.yaml', tmp_path / 'out')

    def test_log10_per_point_imports_as_lognormal(self, demo_petab, tmp_path):
        out = self._import_with_transformation(demo_petab, tmp_path, 'log10')
        conf = (out / 'imported.conf').read_text()
        assert 'objective = lognormal' in conf       # Gaussian(LOG10), the paper's objective
        assert 'objective = chi_sq' not in conf       # NOT the linear (wrong) import

    def test_lin_still_imports_as_chi_sq(self, demo_petab, tmp_path):
        # A lin transformation (the default) is a no-op: the demo still imports as chi_sq,
        # so a linear problem is unchanged by the new column.
        out = self._import_with_transformation(demo_petab, tmp_path, 'lin')
        assert 'objective = chi_sq' in (out / 'imported.conf').read_text()

    def test_log10_constant_sigma_imports_as_lognormal_noise_model_line(self, demo_petab,
                                                                        tmp_path):
        # A fixed non-unit sigma has no sugar token -> the whole-fit lognormal noise_model line
        # (the log10 twin of the gaussian/laplace fix_at case).
        out = self._import_with_transformation(demo_petab, tmp_path, 'log10', noise_formula='2.5')
        conf = (out / 'imported.conf').read_text()
        assert 'noise_model = lognormal, sigma = fix_at 2.5' in conf
        assert 'objective =' not in conf

    def test_imported_lognormal_conf_loads_as_a_configuration(self, demo_petab, tmp_path,
                                                              monkeypatch):
        # The emitted lognormal conf is not just a string match -- it builds a real objective
        # whose noise is the Gaussian additive on the log10 scale (the #499 fix's whole point).
        from pybnf import config as config_mod
        from pybnf.noise import LOG10, Gaussian
        out = self._import_with_transformation(demo_petab, tmp_path, 'log10')
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        assert isinstance(cfg.obj.noise, Gaussian) and cfg.obj.noise.additive_on is LOG10

    def test_log_natural_per_point_imports_as_lnnormal(self, demo_petab, tmp_path):
        # PEtab v1's ``log`` is natural log. It must select the distinct lnnormal token, never
        # alias to PyBNF's log10 ``lognormal`` or the linear Gaussian.
        out = self._import_with_transformation(demo_petab, tmp_path, 'log')
        conf = (out / 'imported.conf').read_text()
        assert 'objective = lnnormal' in conf
        assert 'objective = lognormal' not in conf
        assert 'objective = chi_sq' not in conf

    def test_v2_log_normal_distribution_imports_as_lnnormal(self, demo_petab, tmp_path):
        # Native PEtab v2 spells the same natural-log family in noiseDistribution, without
        # needing the v1 compatibility column.
        out = self._import_with_transformation(
            demo_petab, tmp_path, 'lin', distribution='log-normal')
        assert 'objective = lnnormal' in (out / 'imported.conf').read_text()

    def test_log_natural_constant_sigma_uses_lnnormal_noise_model(self, demo_petab, tmp_path):
        out = self._import_with_transformation(
            demo_petab, tmp_path, 'log', noise_formula='2.5')
        conf = (out / 'imported.conf').read_text()
        assert 'noise_model = lnnormal, sigma = fix_at 2.5' in conf
        assert 'objective =' not in conf

    def test_imported_lnnormal_conf_builds_gaussian_on_ln(self, demo_petab, tmp_path,
                                                           monkeypatch):
        from pybnf import config as config_mod
        from pybnf.noise import LN, Gaussian
        out = self._import_with_transformation(demo_petab, tmp_path, 'log')
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        assert isinstance(cfg.obj.noise, Gaussian) and cfg.obj.noise.additive_on is LN

    def test_log10_laplace_is_refused(self, demo_petab, tmp_path):
        # Both Gaussian log bases have native tokens; a log10 Laplace still does not.
        with pytest.raises(NotImplementedError):
            self._import_with_transformation(demo_petab, tmp_path, 'log10',
                                             distribution='laplace', noise_formula='2.5')

    def test_unknown_transformation_is_refused(self, demo_petab, tmp_path):
        with pytest.raises(PybnfError, match='observableTransformation'):
            self._import_with_transformation(demo_petab, tmp_path, 'ln2')


# ---------------------------------------------------------------------------
# A REAL-WORLD v2 problem imported end to end (the Boehm tutorial; #407, ADR-0037)
#
# Boehm is the headline real-world milestone: the PEtab spec repo's only v2 example,
# externally authored, now imports end to end. It exercises every shape our own exporter
# never writes (sci-notation bounds, a parameterName column, a blank nominalValue, no prior
# columns, a noisePlaceholders column, model_files-first yaml, expression observableFormulas,
# a fixed parameter the SBML lacks (specC17), and a parameter-id noiseParameters that is
# constant per observable). SBML import + the measurement-model layer landed in ADR-0036;
# ADR-0037 closes the last gap -- the constant-per-observable noiseParameters placeholder is
# imported as a per-observable estimated sigma (noise_model <obs> = gaussian, sigma = fit
# sd_<obs>), and the fixed specC17 is inlined into the observableFormula. The recovery tier
# (test_recovery.py) simulates the imported problem at the published optimum.
# See tests/petab_fixtures/boehm_v2/SOURCE.md for provenance + license.
# ---------------------------------------------------------------------------

class TestRealWorldBoehmV2:

    YAML = BOEHM_DIR / 'Boehm_JProteomeRes2014.yaml'

    def test_problem_yaml_reads_model_files_first_ordering(self):
        # Our writer emits model_files LAST; the real v2 yaml lists it FIRST. The
        # order-independent scan must read both identically (and record language: sbml).
        problem = read_problem_yaml(self.YAML)
        assert problem['model_file'] == 'model_Boehm_JProteomeRes2014.xml'
        assert problem['model_id'] == 'model'
        assert problem['model_language'] == 'sbml'
        assert problem['parameter_files'] == ['parameters.tsv']
        assert problem['observable_files'] == ['observables.tsv']
        assert problem['measurement_files'] == ['measurement_data.tsv']
        assert problem['condition_files'] == ['experimental_conditions.tsv']
        assert problem['experiment_files'] == ['experiments.tsv']

    def test_imports_boehm_with_per_observable_noise(self, tmp_path):
        # The full Boehm import (ADR-0037): SBML carried verbatim, each expression
        # observableFormula a measurement model (with the fixed specC17 inlined), and each
        # observable's constant-per-observable parameter-id noiseParameters a per-observable
        # estimated Gaussian sigma.
        pytest.importorskip('petab')
        out = import_job(self.YAML, tmp_path / 'out')
        # The .xml is carried byte-verbatim -- the dynamical model is never edited (ADR-0036).
        assert ((out / 'model_Boehm_JProteomeRes2014.xml').read_text()
                == (BOEHM_DIR / 'model_Boehm_JProteomeRes2014.xml').read_text())
        text = (out / 'imported.conf').read_text()
        # One per-observable noise_model line per observable, each its own estimated sigma.
        for obs in ('pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel'):
            assert f'noise_model {obs} = gaussian, sigma = fit sd_{obs}' in text
        assert 'objective = chi_sq' in text          # the structural whole-fit default
        # The expression observables became measurement-model lines; the fixed specC17
        # (absent from the SBML) was inlined as 0.107, leaving only model entities.
        conf = ploop(text.splitlines(keepends=True))
        meas = {k[1] for k, v in conf.items()
                if isinstance(k, tuple) and k[0] == 'measurement'}
        assert meas == {'pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel'}
        assert 'specC17' not in text and '0.107' in text
        # The 3 sigma parameters are emitted as free (nuisance) parameters alongside the model.
        for sd in ('sd_pSTAT5A_rel', 'sd_pSTAT5B_rel', 'sd_rSTAT5A_rel'):
            assert f'uniform_var = {sd} 1e-05 100000' in text

    def test_imported_boehm_conf_loads_as_a_configuration(self, tmp_path, monkeypatch):
        # The imported conf is a valid end-to-end PyBNF job: the objective carries a
        # per-observable noise override for each observable, the 3 sigma parameters are
        # recognized nuisances (bound to no model id), and the measurement layer builds over
        # the SBML namespace. Simulator-free (no fit) -- the recovery tier runs it.
        pytest.importorskip('petab')
        from pybnf import config as config_mod
        from pybnf.parse import ploop
        out = import_job(self.YAML, tmp_path / 'out')
        monkeypatch.chdir(out)
        conf = config_mod.Configuration(ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        assert set(conf.obj.overrides) == {'pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel'}
        assert conf.obj.required_free_noise_params() == {
            'sd_pSTAT5A_rel', 'sd_pSTAT5B_rel', 'sd_rSTAT5A_rel'}
        assert {m.observable_id for m in conf.obj.measurement.models} == {
            'pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel'}

    def test_imported_boehm_reexports_to_clean_petab(self, tmp_path):
        # #439: Boehm's three estimated per-observable sigmas (`sigma = fit sd_*`) re-export as
        # bare-id noiseFormulae naming estimated parameters -- so the import->export round trip
        # now closes for real-world estimated noise (it used to raise at the `fit` source). The
        # re-exported SBML problem is petablint-clean.
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks
        imp = import_job(self.YAML, tmp_path / 'imp')
        out = tmp_path / 'petab2'
        export_job(imp / 'imported.conf', out)
        # Each expression observable's estimated sigma is the bare noise-parameter id (no
        # per-measurement placeholder); each sigma is an estimated parameter in the table.
        obs = {r['observableId']: r for r in _tsv_rows(out / 'observables.tsv')}
        params = {r['parameterId']: r for r in _tsv_rows(out / 'parameters.tsv')}
        for o in ('pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel'):
            assert obs[o]['noiseFormula'] == f'sd_{o}'
            assert obs[o]['noisePlaceholders'] == ''
            assert params[f'sd_{o}']['estimate'] == 'true'
        # The external oracle: the re-exported (SBML) problem validates via the real petablint
        # path.
        problem = Problem.from_yaml(str(out / 'problem.yaml'))
        errors = [type(t).__name__ for t in default_validation_tasks
                  if (i := t.run(problem)) is not None
                  and getattr(i, 'level', None) == ValidationIssueSeverity.ERROR]
        assert errors == []

    def test_parameter_table_tolerates_real_v2_shapes(self):
        rows = {r.parameter_id: r for r in
                read_parameter_table(BOEHM_DIR / 'parameters.tsv')}
        # Sci-notation bounds parse; the parameterName column is ignored; a blank
        # nominalValue is None; with no prior columns the prior is None (uniform default).
        est = rows['Epo_degradation_BaF3']
        assert est.estimate is True
        assert est.lower_bound == 1e-05 and est.upper_bound == 100000.0
        assert est.nominal_value is None
        assert est.prior_distribution is None and est.prior_parameters == ()
        # A fixed parameter: estimate=false, blank bounds -> None, a numeric nominalValue.
        fixed = rows['ratio']
        assert fixed.estimate is False
        assert fixed.lower_bound is None and fixed.upper_bound is None
        assert fixed.nominal_value == 0.693

    def test_observable_table_records_expression_formula_and_bare_sigma(self):
        rows = {r.observable_id: r for r in
                read_observable_table(BOEHM_DIR / 'observables.tsv')}
        row = rows['pSTAT5A_rel']
        # The expression observableFormula is recorded verbatim (not evaluated); the
        # bare-id noiseFormula and the extra noisePlaceholders column are tolerated.
        assert row.observable_formula.startswith('(100 * pApB')
        assert '/' in row.observable_formula        # a real expression, not a bare name
        assert row.noise_formula == 'pSTAT5A_rel_sigma'
        assert row.noise_distribution == 'normal'
        assert set(rows) == {'pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel'}

    def test_condition_and_experiment_tables_parse(self):
        conds = read_condition_table(BOEHM_DIR / 'experimental_conditions.tsv')
        assert [(c.condition_id, c.target_id, c.target_value) for c in conds] == \
            [('epo_bolus', 'Epo_concentration', '1.25E-07')]
        exps = read_experiment_table(BOEHM_DIR / 'experiments.tsv')
        assert [(e.experiment_id, e.time, e.condition_id) for e in exps] == \
            [('epo_stimulation', 0.0, 'epo_bolus')]

    def test_measurement_table_records_parameter_id_noise_parameters(self):
        # Boehm's noiseParameters column carries a parameter id (a placeholder override),
        # not a number. The reader now records it on noise_parameter_id (numeric stays None),
        # and the per-observable summary maps each observable to its constant sigma id
        # (ADR-0037).
        rows = read_measurement_table(BOEHM_DIR / 'measurement_data.tsv')
        a = next(r for r in rows if r.observable_id == 'pSTAT5A_rel')
        assert a.noise_parameter_id == 'sd_pSTAT5A_rel' and a.noise_parameters is None
        assert noise_parameter_ids_by_observable(rows) == {
            'pSTAT5A_rel': 'sd_pSTAT5A_rel',
            'pSTAT5B_rel': 'sd_pSTAT5B_rel',
            'rSTAT5A_rel': 'sd_rSTAT5A_rel'}


# ---------------------------------------------------------------------------
# Per-measurement placeholder reduction (ADR-0044, #428 Phase 1)
#
# A crafted PEtab v2 problem (tests/petab_fixtures/scaling_v2/) whose value is constant per
# observable: an observableParameters scale substituted into the observableFormula
# (obs_sx = scaling * x), and an expression noiseFormula (obs_y: 0.1 + 0.05*slope) that
# becomes a FormulaSigma. Import-only (the exporter does not emit these): oracled by import
# correctness + a simulator-free score against a hand-built trajectory.
# ---------------------------------------------------------------------------

class TestPlaceholderReductionImport:

    @pytest.fixture(scope='class')
    def out(self, tmp_path_factory):
        d = tmp_path_factory.mktemp('scaling')
        return import_job(SCALING_DIR / 'problem.yaml', d / 'out')

    def test_observable_parameters_substitute_into_observable_formula(self, out):
        text = (out / 'imported.conf').read_text()
        # The observableParameter1_obs_sx placeholder is substituted by its constant token
        # 'scaling' (a free parameter), leaving the measurement model 'scaling*x' (ADR-0044).
        line = next(l for l in text.splitlines() if l.startswith('observable: obs_sx'))
        assert line.startswith('observable: obs_sx, formula:')
        assert 'scaling' in line and 'observableParameter' not in line
        # obs_y's observableFormula is the bare model function y -> no measurement model line.
        assert 'observable: obs_y' not in text

    def test_expression_noise_formula_imports_as_a_formula_sigma(self, out):
        text = (out / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text                       # the structural base
        assert 'noise_model obs_sx = gaussian, sigma = fix_at 0.5' in text
        # The expression noiseFormula (placeholder substituted) -> a 'formula' source over the
        # 'slope' free parameter, on the column obs_y measures (the bare model function y).
        nline = next(l for l in text.splitlines()
                     if l.startswith('noise_model y =') and 'formula' in l)
        assert 'slope' in nline and 'noiseParameter' not in nline

    def test_exp_columns_have_no_sd(self, out):
        # The sigmas are a fixed constant / a formula, not per-point data -> no _SD columns.
        exp = Data(file_name=str(out / 'epo.exp'))
        assert set(exp.cols) == {'time', 'obs_sx', 'y'}

    def test_imported_conf_loads_and_scores(self, out, monkeypatch):
        # The imported job is runnable: it loads as a Configuration (the scaling/slope
        # nuisances are recognized, not flagged as orphan typos), the measurement layer
        # materializes scaling*x, and the FormulaSigma feeds 0.1+0.05*slope to the y term.
        # Scored simulator-free against a hand-built trajectory and a hand-derived Gaussian NLL.
        import types
        from pybnf import config as config_mod
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 'scaling', 'slope'}
        assert [m.observable_id for m in cfg.obj.measurement.models] == ['obs_sx']
        assert cfg.obj.measurement.models[0].formula == 'scaling*x'

        # A trajectory carrying the model columns x (observable) and y (function). With
        # scaling=3 the layer materializes obs_sx = 3*x = [-30,-27,-24] vs data [-20,-18,-16]
        # (residuals -10,-9,-8); y matches the data exactly (residual 0).
        sim = Data.from_columns(
            np.array([[0., -10., 43.], [1., -9., 34.5], [2., -8., 27.]]),
            ['time', 'x', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in
                {'v1': 0.5, 'v2': 1., 'v3': 3., 'scaling': 3., 'slope': 1.}.items()]
        score = cfg.obj.evaluate_multiple({'scaling_model': {'epo': sim}}, cfg.exp_data, pset)
        # obs_sx (fixed sigma 0.5, no normalizer): sum (3x-data)^2/(2*0.5^2) = 245/0.5 = 490.
        # obs_y (FormulaSigma 0.1+0.05*1 = 0.15, estimated): residual 0 + 3*log(0.15) normalizer.
        assert score == pytest.approx(490.0 + 3 * float(np.log(0.15)))
        # The materialized scale is live: the layer added obs_sx = scaling * x to the sim data.
        assert np.allclose(sim['obs_sx'], [-30., -27., -24.])

    def test_row_varying_observable_parameters_imports_as_per_measurement_model(self, tmp_path):
        # A row-varying observableParameters scale (a different scale per timepoint) is no longer
        # deferred (ADR-0045): the observableFormula KEEPS its placeholder and the per-row scale
        # tokens ride a measurement_params sidecar, bound per data point by a PerMeasurementModel.
        # A numeric per-row scale keeps the fixture self-contained (no extra declared parameters).
        prob = tmp_path / 'prob'
        shutil.copytree(SCALING_DIR, prob)
        (prob / 'measurements.tsv').write_text(
            'observableId\texperimentId\ttime\tmeasurement\tobservableParameters\tnoiseParameters\n'
            'obs_sx\tepo\t0\t-20\t2\t\n'
            'obs_sx\tepo\t1\t-18\t3\t\n'      # a different (numeric) scale on the second row
            'obs_sx\tepo\t2\t-16\t2\t\n'
            'obs_y\tepo\t0\t43\t\tslope\n'
            'obs_y\tepo\t1\t34.5\t\tslope\n'
            'obs_y\tepo\t2\t27\t\tslope\n')
        out = import_job(prob / 'problem.yaml', tmp_path / 'out')
        text = (out / 'imported.conf').read_text()
        # The placeholder is KEPT in the observable measurement-model line (not substituted away).
        assert 'observable: obs_sx, formula: observableParameter1_obs_sx * x' in text
        exp_line = next(l for l in text.splitlines() if l.startswith('experiment:'))
        assert 'measurement_params:' in exp_line
        # The sidecar carries the per-row numeric scale, keyed by the materialized column obs_sx.
        from pybnf.petab._measurement_params import read_measurement_params
        table = read_measurement_params(out / 'epo_measparams.tsv')
        assert table['obs_sx'] == {'observableParameter1_obs_sx': {0.0: '2', 1.0: '3', 2.0: '2'}}


# ---------------------------------------------------------------------------
# Row-varying per-measurement noise (ADR-0045, #428 Phase 2)
#
# A crafted PEtab v2 problem (tests/petab_fixtures/rowsigma_v2/) whose obs_y noiseParameters id
# DIFFERS across rows (sd_lo, sd_hi, sd_lo): a per-timepoint estimated sigma with no per-observable
# analogue. On import it is bound per data point from a measurement_params sidecar and scored by a
# PerMeasurementFormulaSigma. Import-only: oracled by import correctness + a simulator-free score
# against a hand-derived NLL where the per-row sigma differs (a broadcast bug is caught).
# ---------------------------------------------------------------------------

class TestRowVaryingNoiseImport:

    @pytest.fixture(scope='class')
    def out(self, tmp_path_factory):
        d = tmp_path_factory.mktemp('rowsigma')
        return import_job(ROWSIGMA_DIR / 'problem.yaml', d / 'out')

    def test_row_varying_noise_imports_as_per_measurement_formula(self, out):
        text = (out / 'imported.conf').read_text()
        # obs_y's row-varying noiseParameters id stays a placeholder formula on column y; obs_x's
        # fixed sigma is the constant path (the two coexist under a structural base objective).
        assert 'objective = chi_sq' in text
        assert 'noise_model x = gaussian, sigma = fix_at 0.5' in text
        assert 'noise_model y = gaussian, sigma = formula noiseParameter1_obs_y' in text
        # The experiment line references the per-measurement binding sidecar (ADR-0045).
        exp_line = next(l for l in text.splitlines() if l.startswith('experiment:'))
        assert 'measurement_params: epo_measparams.tsv' in exp_line

    def test_sidecar_carries_the_per_row_ids(self, out):
        from pybnf.petab._measurement_params import read_measurement_params
        table = read_measurement_params(out / 'epo_measparams.tsv')
        # Keyed by the data COLUMN (y), the placeholder, and time -> the row's estimated id.
        assert table == {'y': {'noiseParameter1_obs_y': {0.0: 'sd_lo', 1.0: 'sd_hi', 2.0: 'sd_lo'}}}

    def test_imported_conf_loads_and_scores_with_per_row_sigma(self, out, monkeypatch):
        # The imported job loads (sd_lo/sd_hi recognized as binding-table nuisances, not orphan
        # typos), attaches the binding table to the exp Data, and scores each obs_y point with its
        # OWN sigma. Scored simulator-free against a hand-derived Gaussian NLL.
        import types
        from pybnf import config as config_mod
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 'sd_lo', 'sd_hi'}
        # The per-data-point binding table rode the sidecar onto the experiment's exp Data.
        epo = cfg.exp_data['rowsigma_model']['epo']
        assert epo.measurement_params == {'y': {'noiseParameter1_obs_y': ['sd_lo', 'sd_hi', 'sd_lo']}}

        # A trajectory whose obs_y differs from the data by residuals (1, 2, 2); obs_x matches.
        sim = Data.from_columns(
            np.array([[0., -10., 44.], [1., -9., 36.5], [2., -8., 29.]]),
            ['time', 'x', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in
                {'v1': 0.5, 'v2': 1., 'v3': 3., 'sd_lo': 0.5, 'sd_hi': 2.}.items()]
        score = cfg.obj.evaluate_multiple({'rowsigma_model': {'epo': sim}}, cfg.exp_data, pset)
        # obs_x (fixed sigma 0.5): residual 0 -> 0. obs_y (estimated Gaussian, per-row sigma):
        #   t0 sd_lo=0.5 res 1 -> 1/(2*.25) + log(.5) = 2 + log(.5)
        #   t1 sd_hi=2.0 res 2 -> 4/(2*4)   + log(2)  = 0.5 + log(2)
        #   t2 sd_lo=0.5 res 2 -> 4/(2*.25) + log(.5) = 8 + log(.5)
        expected = 10.5 + 2 * float(np.log(0.5)) + float(np.log(2.0))
        assert score == pytest.approx(expected)
        # A bug that broadcast a single sigma (sd_lo) over the column would score differently.
        broadcast = 18.0 + 3 * float(np.log(0.5))
        assert not np.isclose(score, broadcast)

    def test_per_measurement_sigma_source_survives_pickle(self, out, monkeypatch):
        # The objective (carrying the PerMeasurementFormulaSigma) is scattered to dask workers;
        # the lambdify callable is dropped + rebuilt worker-side, and the binding table rides the
        # exp Data, so a round-tripped objective scores identically (ADR-0045).
        import pickle
        import types
        from pybnf import config as config_mod
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        obj = pickle.loads(pickle.dumps(cfg.obj))
        exp = {m: {s: pickle.loads(pickle.dumps(d)) for s, d in sd.items()}
               for m, sd in cfg.exp_data.items()}
        sim = Data.from_columns(
            np.array([[0., -10., 44.], [1., -9., 36.5], [2., -8., 29.]]),
            ['time', 'x', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in
                {'v1': 0.5, 'v2': 1., 'v3': 3., 'sd_lo': 0.5, 'sd_hi': 2.}.items()]
        score = obj.evaluate_multiple({'rowsigma_model': {'epo': sim}}, exp, pset)
        assert score == pytest.approx(10.5 + 2 * float(np.log(0.5)) + float(np.log(2.0)))

    def test_row_varying_noise_with_numeric_mix_is_deferred(self, tmp_path):
        # An observable whose noiseParameters MIXES a parameter id with a numeric per-point value
        # across rows is a per-row source-kind change -> still the deferred frontier (ADR-0045).
        prob = tmp_path / 'prob'
        shutil.copytree(ROWSIGMA_DIR, prob)
        (prob / 'measurements.tsv').write_text(
            'observableId\texperimentId\ttime\tmeasurement\tobservableParameters\tnoiseParameters\n'
            'obs_x\tepo\t0\t-10\t\t\n'
            'obs_y\tepo\t0\t43\t\tsd_lo\n'
            'obs_y\tepo\t1\t34.5\t\t0.7\n')      # an id on one row, a number on the next
        with pytest.raises(NotImplementedError, match='source kind'):
            import_job(prob / 'problem.yaml', tmp_path / 'out')


# ---------------------------------------------------------------------------
# Row-varying per-measurement OBSERVABLE scale (ADR-0045, #428 Phase 2b)
#
# A crafted PEtab v2 problem (tests/petab_fixtures/obsscale_v2/) whose obs_y observableParameters
# scale DIFFERS across rows (s_lo, s_hi, s_lo): a per-timepoint estimated scale with no
# per-observable analogue. It cannot be pre-materialized as a column by the MeasurementLayer, so
# on import the observableFormula KEEPS its placeholder and the per-row scale ids ride a
# measurement_params sidecar; at score time it is a PerMeasurementModel evaluated per data point
# in the objective's prediction step (the genuine ADR-0036 contract change). Import-only oracle:
# import correctness + a simulator-free score against a hand-derived NLL where the per-row scale
# differs (a single-broadcast bug is caught).
# ---------------------------------------------------------------------------
class TestRowVaryingObservableImport:

    @pytest.fixture(scope='class')
    def out(self, tmp_path_factory):
        d = tmp_path_factory.mktemp('obsscale')
        return import_job(OBSSCALE_DIR / 'problem.yaml', d / 'out')

    def test_row_varying_observable_imports_as_per_measurement_model(self, out):
        text = (out / 'imported.conf').read_text()
        # obs_y's row-varying observableParameters scale stays a placeholder measurement model
        # (NOT substituted); obs_x's bare observable with its fixed sigma is the constant path.
        assert 'objective = chi_sq' in text
        assert 'noise_model x = gaussian, sigma = fix_at 0.5' in text
        assert 'noise_model obs_y = gaussian, sigma = fix_at 1' in text
        assert 'observable: obs_y, formula: observableParameter1_obs_y * y' in text
        # The experiment line references the per-measurement binding sidecar (ADR-0045).
        exp_line = next(l for l in text.splitlines() if l.startswith('experiment:'))
        assert 'measurement_params: epo_measparams.tsv' in exp_line

    def test_sidecar_carries_the_per_row_scale_ids(self, out):
        from pybnf.petab._measurement_params import read_measurement_params
        table = read_measurement_params(out / 'epo_measparams.tsv')
        # Keyed by the materialized measurement-model COLUMN (obs_y), the placeholder, and time.
        assert table == {'obs_y': {'observableParameter1_obs_y': {0.0: 's_lo', 1.0: 's_hi',
                                                                   2.0: 's_lo'}}}

    def test_imported_conf_loads_and_scores_with_per_row_scale(self, out, monkeypatch):
        # The imported job loads (s_lo/s_hi recognized as binding-table nuisances, not orphan
        # typos), attaches the binding table to the exp Data, and scales each obs_y prediction by
        # its OWN per-row scale in _prediction. Scored simulator-free against a hand-derived NLL.
        import types
        from pybnf import config as config_mod
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 's_lo', 's_hi'}
        # The per-data-point binding table rode the sidecar onto the experiment's exp Data, and
        # the row-varying observable is registered on the objective (not pre-materialized).
        epo = cfg.exp_data['obsscale_model']['epo']
        assert epo.measurement_params == {'obs_y': {'observableParameter1_obs_y':
                                                    ['s_lo', 's_hi', 's_lo']}}
        assert set(cfg.obj._per_measurement_models) == {'obs_y'}

        # A trajectory whose y differs from nominal (43, 34.5, 27) by (+1, +2, +2); obs_x matches.
        sim = Data.from_columns(
            np.array([[0., -10., 44.], [1., -9., 36.5], [2., -8., 29.]]),
            ['time', 'x', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in
                {'v1': 0.5, 'v2': 1., 'v3': 3., 's_lo': 2., 's_hi': 3.}.items()]
        score = cfg.obj.evaluate_multiple({'obsscale_model': {'epo': sim}}, cfg.exp_data, pset)
        # obs_x (bare, fixed sigma 0.5): residual 0 -> 0. obs_y (scale * y, fixed sigma 1):
        #   t0 s_lo=2 pred 2*44=88   vs 86    res 2 -> 4/2 = 2
        #   t1 s_hi=3 pred 3*36.5=109.5 vs 103.5 res 6 -> 36/2 = 18
        #   t2 s_lo=2 pred 2*29=58   vs 54    res 4 -> 16/2 = 8
        assert score == pytest.approx(28.0)
        # A bug that broadcast a single scale (s_lo) over the column would score differently:
        #   t1 would be 2*36.5=73 vs 103.5, res -30.5 -> a much larger total.
        broadcast = (2. ** 2 + 30.5 ** 2 + 4. ** 2) / 2.
        assert not np.isclose(score, broadcast)

    def test_per_measurement_model_survives_pickle(self, out, monkeypatch):
        # The objective (carrying the PerMeasurementModel) is scattered to dask workers; the
        # lambdify callable is dropped + rebuilt worker-side, and the binding table rides the exp
        # Data, so a round-tripped objective scores identically (ADR-0045).
        import pickle
        import types
        from pybnf import config as config_mod
        monkeypatch.chdir(out)
        cfg = config_mod.Configuration(
            ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        obj = pickle.loads(pickle.dumps(cfg.obj))
        exp = {m: {s: pickle.loads(pickle.dumps(d)) for s, d in sd.items()}
               for m, sd in cfg.exp_data.items()}
        sim = Data.from_columns(
            np.array([[0., -10., 44.], [1., -9., 36.5], [2., -8., 29.]]),
            ['time', 'x', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in
                {'v1': 0.5, 'v2': 1., 'v3': 3., 's_lo': 2., 's_hi': 3.}.items()]
        score = obj.evaluate_multiple({'obsscale_model': {'epo': sim}}, exp, pset)
        assert score == pytest.approx(28.0)

    def test_row_varying_observable_on_column_joint_objective_is_deferred(self, out, monkeypatch):
        # A row-varying observable scale is bound per data point in _prediction, which the
        # column-joint kl / wasserstein objectives do not have -> a clean deferral (ADR-0045).
        # Swap the per-point objective + its per-observable noise lines for a profile objective.
        from pybnf import config as config_mod
        monkeypatch.chdir(out)
        kept = [l for l in (out / 'imported.conf').read_text().splitlines()
                if not l.startswith('objective =') and not l.startswith('noise_model ')]
        lines = []
        for l in kept:
            lines.append(l)
            if l.startswith('job_type'):
                lines.append('profile_objective = kl')
        with pytest.raises(NotImplementedError, match='column-joint'):
            config_mod.Configuration(ploop((l + '\n' for l in lines)))


# ---------------------------------------------------------------------------
# observableParameters / noiseParameters placeholder completions (ADR-0075, issue #495)
#
# Three crafted PEtab v2 problems, each import-only (oracled by import correctness + a
# simulator-free score against a hand-derived NLL), covering the three gaps #495 named:
#   * fixedsigma_v2 (Oliveira) -- a noiseParameters id that is FIXED -> a constant sigma;
#   * multisigma_v2 (Fiedler)  -- a MULTI-token, row-varying noiseParameters product ->
#                                  a PerMeasurementFormulaSigma over two placeholders;
#   * predsigma_v2  (Raia)     -- a prediction-dependent affine noiseFormula -> a
#                                  PredictionFormulaSigma whose sigma scales with the sim output.
# The shared model is the deterministic parabola y = v1*x^2 + v2*x + v3.
# ---------------------------------------------------------------------------

# A fixed trajectory whose obs_y differs from the data (43, 34.5, 27) by residuals (1, 2, 2).
_SIM_Y = np.array([[0., 44.], [1., 36.5], [2., 29.]])


def _score(cfg, model_name, pset_values):
    """Score the fixed _SIM_Y trajectory under ``pset_values`` (a simulator-free evaluate)."""
    import types
    sim = Data.from_columns(_SIM_Y.copy(), ['time', 'y'], indvar='time')
    pset = [types.SimpleNamespace(name=n, value=v) for n, v in pset_values.items()]
    return cfg.obj.evaluate_multiple({model_name: {'epo': sim}}, cfg.exp_data, pset)


def _load_conf(out, monkeypatch):
    from pybnf import config as config_mod
    monkeypatch.chdir(out)
    return config_mod.Configuration(
        ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))


class TestFixedNoiseParamImport:
    """A noiseParameters id resolving to a FIXED parameter imports as a constant sigma (Oliveira,
    ADR-0075) -- not a `fit` free sigma the .conf never declares."""

    @pytest.fixture(scope='class')
    def out(self, tmp_path_factory):
        return import_job(FIXEDSIGMA_DIR / 'problem.yaml', tmp_path_factory.mktemp('fixed') / 'out')

    def test_fixed_noise_id_imports_as_constant_sigma(self, out):
        text = (out / 'imported.conf').read_text()
        # sd_c (estimate=false, value 2) inlines as a fixed sigma, NOT `fit sd_c`.
        assert 'noise_model = gaussian, sigma = fix_at 2' in text
        assert 'sd_c' not in text                       # neither a fit source nor a variable line

    def test_imported_conf_loads_and_scores_with_fixed_sigma(self, out, monkeypatch):
        cfg = _load_conf(out, monkeypatch)
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3'}   # sd_c is not a free param
        score = _score(cfg, 'fixedsigma_model', {'v1': .5, 'v2': 1., 'v3': 3.})
        # A fixed-scale Gaussian drops the normalizer: sum of res^2/(2*sigma^2), sigma = 2.
        expected = float(np.sum(np.array([1., 2., 2.]) ** 2 / (2 * 2. ** 2)))
        assert score == pytest.approx(expected)


class TestMultiTokenRowVaryingNoiseImport:
    """A multi-token noiseParameters product whose per-row scale differs imports as a
    PerMeasurementFormulaSigma over BOTH placeholders, bound per data point (Fiedler, ADR-0075)."""

    @pytest.fixture(scope='class')
    def out(self, tmp_path_factory):
        return import_job(MULTISIGMA_DIR / 'problem.yaml', tmp_path_factory.mktemp('multi') / 'out')

    def test_multi_token_noise_imports_as_per_measurement_formula(self, out):
        text = (out / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text
        assert ('noise_model y = gaussian, sigma = formula '
                'noiseParameter1_obs_y * noiseParameter2_obs_y') in text
        exp_line = next(l for l in text.splitlines() if l.startswith('experiment:'))
        assert 'measurement_params: epo_measparams.tsv' in exp_line

    def test_sidecar_binds_both_noise_placeholders_per_row(self, out):
        from pybnf.petab._measurement_params import read_measurement_params
        table = read_measurement_params(out / 'epo_measparams.tsv')
        # Both noiseParameter1 (the row-varying scale) and noiseParameter2 (the shared sigma)
        # are bound per data point, keyed by the data column y (ADR-0075).
        assert table == {'y': {
            'noiseParameter1_obs_y': {0.0: 's_lo', 1.0: 's_hi', 2.0: 's_lo'},
            'noiseParameter2_obs_y': {0.0: 'sig', 1.0: 'sig', 2.0: 'sig'}}}

    def test_imported_conf_loads_and_scores_with_per_row_product_sigma(self, out, monkeypatch):
        cfg = _load_conf(out, monkeypatch)
        # s_lo/s_hi/sig are recognized as binding-table nuisances, not orphan typos.
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 's_lo', 's_hi', 'sig'}
        epo = cfg.exp_data['multisigma_model']['epo']
        assert epo.measurement_params == {'y': {
            'noiseParameter1_obs_y': ['s_lo', 's_hi', 's_lo'],
            'noiseParameter2_obs_y': ['sig', 'sig', 'sig']}}
        score = _score(cfg, 'multisigma_model',
                       {'v1': .5, 'v2': 1., 'v3': 3., 's_lo': .5, 's_hi': 1., 'sig': 2.})
        # Estimated Gaussian, sigma_i = scale_i * sig: [1, 2, 1]; residuals [1, 2, 2] -> +log sigma.
        sig = np.array([.5 * 2, 1. * 2, .5 * 2])
        res = np.array([1., 2., 2.])
        assert score == pytest.approx(float(np.sum(res ** 2 / (2 * sig ** 2) + np.log(sig))))
        # A bug that dropped the second token (sigma = scale alone) would score differently.
        wrong = np.array([.5, 1., .5])
        assert not np.isclose(score, float(np.sum(res ** 2 / (2 * wrong ** 2) + np.log(wrong))))


class TestReplicateSpecificMeasurementParamsImport:
    """Repeated ``(observable, time)`` rows retain distinct placeholder tokens (issue #508).

    This is the minimal Fiedler shape: gel/replicate 1 binds ``s_lo`` at every time, while
    gel/replicate 2 binds ``s_hi`` at those same cells. Before ADR-0083 the sidecar's bare time
    key let the second gel overwrite the first, orphaning ``s_lo`` and silently scoring both
    replicate files with ``s_hi``.
    """

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        fixture = tmp_path_factory.mktemp('replicate_params') / 'problem'
        shutil.copytree(MULTISIGMA_DIR, fixture)
        (fixture / 'observables.tsv').write_text(
            'observableId\tobservableFormula\tnoiseFormula\tobservablePlaceholders\t'
            'noisePlaceholders\tnoiseDistribution\n'
            'obs_y\tobservableParameter1_obs_y * y\t'
            'noiseParameter1_obs_y * noiseParameter2_obs_y\t'
            'observableParameter1_obs_y\t'
            'noiseParameter1_obs_y;noiseParameter2_obs_y\tnormal\n')
        # PEtab encodes replicates as repeated cells. The first occurrence of every time is gel 1
        # (s_lo), the second gel 2 (s_hi), exactly the dealing order used to build the two .exp
        # files. Fiedler binds the gel scale through BOTH placeholder families.
        (fixture / 'measurements.tsv').write_text(
            'observableId\texperimentId\ttime\tmeasurement\tobservableParameters\tnoiseParameters\n'
            'obs_y\tepo\t0\t43\ts_lo\ts_lo;sig\n'
            'obs_y\tepo\t0\t42\ts_hi\ts_hi;sig\n'
            'obs_y\tepo\t1\t34.5\ts_lo\ts_lo;sig\n'
            'obs_y\tepo\t1\t32.5\ts_hi\ts_hi;sig\n'
            'obs_y\tepo\t2\t27\ts_lo\ts_lo;sig\n'
            'obs_y\tepo\t2\t25\ts_hi\ts_hi;sig\n')
        out = import_job(fixture / 'problem.yaml', fixture.parent / 'out')
        return fixture, out

    def test_import_writes_two_exp_files_and_a_replicate_aware_sidecar(self, imported):
        _fixture, out = imported
        text = (out / 'imported.conf').read_text()
        exp_line = next(l for l in text.splitlines() if l.startswith('experiment:'))
        assert 'data: epo.exp, epo_rep2.exp' in exp_line
        assert (out / 'epo_measparams.tsv').read_text().splitlines()[0].split('\t') == [
            'replicate', 'column', 'time', 'placeholder', 'token']

        from pybnf.petab._measurement_params import read_measurement_params
        table = read_measurement_params(out / 'epo_measparams.tsv')
        assert table == {'obs_y': {
            'observableParameter1_obs_y': {
                (0, 0.0): 's_lo', (0, 1.0): 's_lo', (0, 2.0): 's_lo',
                (1, 0.0): 's_hi', (1, 1.0): 's_hi', (1, 2.0): 's_hi'},
            'noiseParameter1_obs_y': {
                (0, 0.0): 's_lo', (0, 1.0): 's_lo', (0, 2.0): 's_lo',
                (1, 0.0): 's_hi', (1, 1.0): 's_hi', (1, 2.0): 's_hi'},
            'noiseParameter2_obs_y': {
                (0, 0.0): 'sig', (0, 1.0): 'sig', (0, 2.0): 'sig',
                (1, 0.0): 'sig', (1, 1.0): 'sig', (1, 2.0): 'sig'}}}

    def test_config_attaches_each_replicates_tokens_and_scores_them(self, imported, monkeypatch):
        _fixture, out = imported
        cfg = _load_conf(out, monkeypatch)
        # Both gel scales are recognized as used nuisances; neither is orphaned (#508 reproducer).
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 's_lo', 's_hi', 'sig'}
        epo = cfg.exp_data['multisigma_model']['epo']
        assert epo.measurement_params == {'obs_y': {
            'observableParameter1_obs_y': ['s_lo'] * 3 + ['s_hi'] * 3,
            'noiseParameter1_obs_y': ['s_lo'] * 3 + ['s_hi'] * 3,
            'noiseParameter2_obs_y': ['sig'] * 6}}

        score = _score(cfg, 'multisigma_model',
                       {'v1': .5, 'v2': 1., 'v3': 3., 's_lo': .5, 's_hi': 1., 'sig': 2.})
        sigma = np.array([1., 1., 1., 2., 2., 2.])
        prediction = np.r_[.5 * _SIM_Y[:, 1], _SIM_Y[:, 1]]
        residual = prediction - np.array([43., 34.5, 27., 42., 32.5, 25.])
        expected = float(np.sum(residual ** 2 / (2 * sigma ** 2) + np.log(sigma)))
        assert score == pytest.approx(expected)

    def test_export_restores_each_replicate_placeholder_tokens(self, imported, tmp_path, monkeypatch):
        _fixture, out = imported
        monkeypatch.chdir(out)
        exported = export_job('imported.conf', tmp_path / 'exported')
        rows = read_measurement_table(exported / 'measurements.tsv')
        assert [row.observable_parameters for row in rows] == \
            [('s_lo',)] * 3 + [('s_hi',)] * 3
        assert [row.noise_param_tokens for row in rows] == \
            [('s_lo', 'sig')] * 3 + [('s_hi', 'sig')] * 3

    def test_legacy_four_column_sidecar_still_shares_tokens_across_replicates(self, tmp_path):
        from pybnf.petab._measurement_params import (
            measurement_params_for_replicate,
            read_measurement_params,
            write_measurement_params,
        )
        sidecar = tmp_path / 'legacy.tsv'
        original = {'y': {'noiseParameter1_obs_y': {0.0: 'shared', 1.0: 'shared'}}}
        write_measurement_params(original, sidecar)
        assert sidecar.read_text().splitlines()[0] == 'column\ttime\tplaceholder\ttoken'
        loaded = read_measurement_params(sidecar)
        assert measurement_params_for_replicate(loaded, 0) == original
        assert measurement_params_for_replicate(loaded, 1) == original


class TestPredictionDependentNoiseImport:
    """An affine noiseFormula whose sigma scales with the simulated output imports as a
    PredictionFormulaSigma; sigma reads the current simulation, its coefficients the PSet
    (Raia, ADR-0075)."""

    @pytest.fixture(scope='class')
    def out(self, tmp_path_factory):
        return import_job(PREDSIGMA_DIR / 'problem.yaml', tmp_path_factory.mktemp('pred') / 'out')

    def test_affine_prediction_noise_imports_as_prediction_formula(self, out):
        text = (out / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text
        # The two noiseParameters tokens substitute in by index; y is a model entity, so the
        # source is prediction_formula (not the free-parameter-only `formula`).
        line = next(l for l in text.splitlines() if l.startswith('noise_model y'))
        assert 'prediction_formula sd_abs + sd_rel*y' in line

    def test_imported_conf_loads_and_scores_against_sim_based_nll(self, out, monkeypatch):
        cfg = _load_conf(out, monkeypatch)
        assert {v.name for v in cfg.variables} == {'v1', 'v2', 'v3', 'sd_abs', 'sd_rel'}
        score = _score(cfg, 'predsigma_model',
                       {'v1': .5, 'v2': 1., 'v3': 3., 'sd_abs': .5, 'sd_rel': .1})
        # sigma_i = sd_abs + sd_rel * y_SIM_i = 0.5 + 0.1*[44, 36.5, 29] = [4.9, 4.15, 3.4].
        sig = 0.5 + 0.1 * _SIM_Y[:, 1]
        res = np.array([1., 2., 2.])
        assert score == pytest.approx(float(np.sum(res ** 2 / (2 * sig ** 2) + np.log(sig))))
        # A bug that evaluated sigma at the MEASURED value (43, 34.5, 27) scores differently.
        sig_data = 0.5 + 0.1 * np.array([43., 34.5, 27.])
        assert not np.isclose(score, float(np.sum(res ** 2 / (2 * sig_data ** 2) + np.log(sig_data))))

    def test_prediction_noise_source_survives_pickle(self, out, monkeypatch):
        # The objective carrying the PredictionFormulaSigma is scattered to dask workers; the
        # lambdify callable is dropped + rebuilt worker-side (ADR-0075), so a round-tripped
        # objective scores identically.
        import pickle
        import types
        cfg = _load_conf(out, monkeypatch)
        obj = pickle.loads(pickle.dumps(cfg.obj))
        exp = {m: {s: pickle.loads(pickle.dumps(d)) for s, d in sd.items()}
               for m, sd in cfg.exp_data.items()}
        sim = Data.from_columns(_SIM_Y.copy(), ['time', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in
                {'v1': .5, 'v2': 1., 'v3': 3., 'sd_abs': .5, 'sd_rel': .1}.items()]
        score = obj.evaluate_multiple({'predsigma_model': {'epo': sim}}, exp, pset)
        sig = 0.5 + 0.1 * _SIM_Y[:, 1]
        res = np.array([1., 2., 2.])
        assert score == pytest.approx(float(np.sum(res ** 2 / (2 * sig ** 2) + np.log(sig))))

    def test_prediction_formula_over_free_params_only_is_rejected(self, tmp_path, monkeypatch):
        # prediction_formula must reference a model output; a σ over free parameters alone should
        # use `formula` instead (ADR-0075) -- config raises a pointed error at load.
        from pybnf import config as config_mod
        prob = tmp_path / 'prob'
        shutil.copytree(PREDSIGMA_DIR, prob)
        out = import_job(prob / 'problem.yaml', tmp_path / 'out')
        text = (out / 'imported.conf').read_text().replace(
            'prediction_formula sd_abs + sd_rel*y', 'prediction_formula sd_abs + sd_rel')
        monkeypatch.chdir(out)   # monkeypatch restores the cwd on teardown (no leak into later tests)
        with pytest.raises(PybnfError, match='references no model output'):
            config_mod.Configuration(ploop(text.splitlines(keepends=True)))
