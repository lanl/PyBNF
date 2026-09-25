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
from pybnf.petab._tsv import num
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

# The #894 column-mean fixture and its three objective oracles live with the exporter tests.
from .test_petab_export import (
    _COLUMN_MEAN_SPELLINGS,
    _DECAY_MODEL,
    _DECAY_TIMES,
    _decay_series,
    _exp_text,
    _petab_nll,
    _pybnf_objective,
    _write_decay_job,
    _write_ragged_job,
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


# ---------------------------------------------------------------------------
# #896: a fixed-duration equilibration (``equil_t_end: T``) exports as a leading PEtab period at
# time -T and imports back as ``preequilibrate:`` + ``equil_t_end: T``: the same protocol, the
# same objective at a fixed parameter vector, and a byte-equal re-export.
# ---------------------------------------------------------------------------

# dA/dt = kp - k*flag*A, A(0) = 10 (the #896 reproduction model).
_FIXED_EQUIL_MODEL = _PREEQUIL_MODEL.replace(
    'begin parameters\n  k     1.0', 'begin parameters\n  kp    5\n  k     1.0').replace(
    '  A() -> 0 deg()', '  0 -> A() kp\n  A() -> 0 deg()')

_FIXED_EQUIL_CONF = (
    'edition = 2\njob_type = de\nobjective = sos\nmodel: m.bngl\n'
    'condition: pre,  perturbations: flag = 2\n'
    'condition: meas, perturbations: flag = 1\n'
    'experiment: relax, preequilibrate: pre, condition: meas, equil_t_end: 0.1, '
    'data: relax.exp\n'
    'uniform_var = k 0.1 10\n')

_FIXED_EQUIL_EXP = '# time A_tot\n0\t8.64\n1\t6.34\n2\t5.49\n'


def _simulate_and_score(conf_path, k, monkeypatch):
    """PyBNF's own simulation (BNG2.pl) of the job at ``conf_path`` at ``k``, scored with the
    job's own objective. Returns ``(A_tot over the measured phase, objective value)``."""
    import os
    from pybnf.parse import load_config
    from pybnf.pset import PSet
    monkeypatch.chdir(conf_path.parent)
    text = conf_path.read_text()
    if 'population_size' not in text:
        text += 'population_size = 4\nmax_iterations = 1\n'
    (conf_path.parent / 'sim.conf').write_text(text)
    conf = load_config('sim.conf')
    name, model = next(iter(conf.models.items()))
    ps = PSet([v.set_value(k) for v in conf.variables])
    folder = f'sim_{k}'
    os.makedirs(folder, exist_ok=True)
    out = model.copy_with_param_set(ps).execute(folder, folder, 60)
    data = out['relax']
    return (list(data.data[:, data.cols['A_tot']]),
            conf.obj.evaluate_multiple({name: out}, conf.exp_data, ps))


class TestImportFixedDurationEquilibrationRoundTrip:

    @pytest.fixture(scope='class')
    def imported(self, tmp_path_factory):
        return _roundtrip(
            tmp_path_factory.mktemp('fixed_equil'), _FIXED_EQUIL_CONF,
            extra_files={'m.bngl': _FIXED_EQUIL_MODEL, 'relax.exp': _FIXED_EQUIL_EXP},
            model_name='m.bngl')

    def test_problem_round_trips_byte_for_byte(self, imported):
        petab1, _, petab2, _ = imported
        _assert_problem_round_trips(petab1, petab2)
        assert [(r['time'], r['conditionId']) for r in _tsv_rows(petab1 / 'experiments.tsv')] \
            == [('-0.1', 'cond_pre'), ('0', 'cond_meas')]

    def test_imported_conf_carries_equil_t_end(self, imported):
        # The -0.1 leading period reads back as the fixed duration, not a refusal (the old
        # "fixed-time equilibration is deferred") and not a steady state.
        _, _, _, conf = imported
        assert ('experiment: relax, preequilibrate: pre, condition: meas, method: ode, '
                'equil_t_end: 0.1, data: relax.exp') in conf.read_text()

    def test_imported_conf_synthesizes_the_fixed_duration_equilibration(self, imported,
                                                                        monkeypatch):
        from pybnf.config import Configuration
        _, imported_dir, _, conf = imported
        monkeypatch.chdir(imported_dir)
        acts = Configuration(ploop(conf.read_text().splitlines(keepends=True))).models['m'].actions
        equil = [a for a in acts if '_preequil' in a]
        assert equil == ['simulate({method=>"ode",t_start=>0,t_end=>0.1,n_steps=>1,'
                         'suffix=>"relax_preequil",print_functions=>1})']

    def test_original_and_imported_jobs_score_the_same(self, imported, monkeypatch):
        # Oracle: the native job and its round-tripped import simulate the same trajectory and
        # score the same objective at fixed k, and the trajectory is the hand-derived fixed 0.1
        # equilibration (A(0) = 2.5/k + (10 - 2.5/k)*exp(-0.2k), then relaxation to 5/k).
        import math
        from .recovery_harness import require_bng2pl
        require_bng2pl()
        petab1, imported_dir, _, conf = imported
        src = petab1.parent / 'src'
        for k in (1.0, 0.4):
            native_a, native_obj = _simulate_and_score(src / 'job.conf', k, monkeypatch)
            imported_a, imported_obj = _simulate_and_score(conf, k, monkeypatch)
            a0 = 2.5 / k + (10 - 2.5 / k) * math.exp(-0.2 * k)
            closed = [5 / k + (a0 - 5 / k) * math.exp(-k * t) for t in (0, 1, 2)]
            np.testing.assert_allclose(native_a, closed, rtol=1e-5)
            np.testing.assert_allclose(imported_a, native_a, rtol=1e-12)
            assert imported_obj == pytest.approx(native_obj, rel=1e-12)
            # edition 2 reads `objective = sos` as a unit-sigma Gaussian: 1/2 * sum(residual^2).
            data = (8.64, 6.34, 5.49)
            assert native_obj == pytest.approx(
                0.5 * sum((p - d) ** 2 for p, d in zip(closed, data)), rel=1e-4, abs=1e-7)

    def test_fit_parameter_perturbation_round_trips_with_equil_t_end(self, tmp_path_factory):
        # The surrogate split (ADR-0027, #443) composes with the -T period: the equilibration
        # condition sets the FIT parameter k (so k is exported as k__REF and re-pinned on the
        # measured period) and the fixed duration survives the round trip alongside it.
        conf = _FIXED_EQUIL_CONF.replace('condition: pre,  perturbations: flag = 2',
                                         'condition: pre,  perturbations: k = 0.5')
        petab1, _, petab2, imported_conf = _roundtrip(
            tmp_path_factory.mktemp('fixed_equil_fit'), conf,
            extra_files={'m.bngl': _FIXED_EQUIL_MODEL, 'relax.exp': _FIXED_EQUIL_EXP},
            model_name='m.bngl')
        _assert_problem_round_trips(petab1, petab2)
        assert {(r['conditionId'], r['targetId'], r['targetValue'])
                for r in _tsv_rows(petab1 / 'conditions.tsv')} == {
            ('cond_pre', 'k', '0.5'), ('cond_meas', 'k', 'k__REF'), ('cond_meas', 'flag', '1')}
        text = imported_conf.read_text()
        assert 'condition: pre, perturbations: k = 0.5' in text
        assert 'preequilibrate: pre, condition: meas, method: ode, equil_t_end: 0.1' in text

    def test_preequilibrated_scan_round_trips_its_equil_t_end(self, tmp_path_factory,
                                                              monkeypatch):
        # The ADR-0062 sibling: each dose's -7200 leading period reads back as ONE scan experiment
        # carrying equil_t_end: 7200, and the fitter synthesizes a fixed 7200 equilibration.
        conf = _PDR_CONF.replace(', t_end: 500,', ', t_end: 500, equil_t_end: 7200,')
        petab1, imported_dir, petab2, imported_conf = _roundtrip(
            tmp_path_factory.mktemp('pdr_fixed'), conf,
            extra_files={'m.bngl': _PDR_MODEL, 'dose.exp': _PDR_DOSE_EXP}, model_name='m.bngl')
        _assert_problem_round_trips(petab1, petab2)
        assert {r['time'] for r in _tsv_rows(petab1 / 'experiments.tsv')} == {'-7200', '0'}
        text = imported_conf.read_text()
        assert ('experiment: scan, preequilibrate: incubate, condition: wash, method: ode, '
                't_end: 500, equil_t_end: 7200, data: scan.exp') in text
        from pybnf.config import Configuration
        monkeypatch.chdir(imported_dir)
        acts = Configuration(ploop(text.splitlines(keepends=True))).models['m'].actions
        assert any('t_end=>7200' in a and '_preequil' in a and 'steady_state' not in a
                   for a in acts)

    def test_time_dependent_model_is_refused(self, tmp_path_factory):
        # PEtab runs the leading period on [-T, 0]; PyBNF would run equil_t_end on [0, T] and
        # restart the clock, so a model reading time() would see a different protocol.
        petab1, _, _, _ = _roundtrip(
            tmp_path_factory.mktemp('fixed_equil_time'), _FIXED_EQUIL_CONF,
            extra_files={'m.bngl': _FIXED_EQUIL_MODEL, 'relax.exp': _FIXED_EQUIL_EXP},
            model_name='m.bngl')
        (petab1 / 'm.bngl').write_text(
            _FIXED_EQUIL_MODEL.replace('deg() k*flag', 'deg() k*flag*(1 + time())'))
        with pytest.raises(NotImplementedError,
                           match="Experiment 'relax'.*reads the simulation time"):
            import_job(petab1 / 'problem.yaml', petab1.parent / 'imported_time')

    def test_scan_with_a_finite_leading_period_not_followed_by_time_zero_is_not_a_scan(self):
        # Only the exporter's -T / 0 shape is a fixed-duration pre-equilibrated scan; a leading
        # finite period followed by one at t=1 would shift the dose period, so it is not read as one.
        from pybnf.petab.measurements import reconstruct_preequilibrated_dose_responses
        meas = [PetabMeasurementRow('obs_resp', 20.0, 1.0, experiment_id='scan_0')]
        conds = [PetabConditionRow('cond_scan_0', 'L', '1'),
                 PetabConditionRow('cond_pre', 'kd', '2')]
        for times in ((-5.0, 1.0), (-5.0, 0.0)):
            exps = [PetabExperimentRow('scan_0', times[0], 'cond_pre'),
                    PetabExperimentRow('scan_0', times[1], 'cond_scan_0')]
            scans, *_ = reconstruct_preequilibrated_dose_responses(
                meas, conds, exps, {'obs_resp': 'resp'}, sweepable={'L', 'kd'})
            if times[1] == 0.0:
                assert [(s['preequilibrate'], s['equil_t_end']) for s in scans] == [('pre', 5.0)]
            else:
                assert scans == []

    def test_scan_with_a_blank_fixed_duration_period_is_refused(self):
        # No condition to carry equil_t_end on -> refused, not imported as an un-equilibrated scan.
        from pybnf.petab.measurements import reconstruct_preequilibrated_dose_responses
        meas = [PetabMeasurementRow('obs_resp', 20.0, 1.0, experiment_id='scan_0')]
        conds = [PetabConditionRow('cond_scan_0', 'L', '1')]
        exps = [PetabExperimentRow('scan_0', -5.0, ''),
                PetabExperimentRow('scan_0', 0.0, 'cond_scan_0')]
        with pytest.raises(NotImplementedError, match="'scan_0'.*fixed-duration equilibration"):
            reconstruct_preequilibrated_dose_responses(meas, conds, exps, {'obs_resp': 'resp'},
                                                       sweepable={'L', 'kd'})

    def test_scan_whose_doses_equilibrate_for_different_durations_is_ambiguous(self):
        from pybnf.petab.measurements import reconstruct_preequilibrated_dose_responses
        meas = [PetabMeasurementRow('obs_resp', 20.0, 1.0, experiment_id=f'scan_{i}')
                for i in range(2)]
        conds = [PetabConditionRow(f'cond_scan_{i}', 'L', str(i + 1)) for i in range(2)]
        exps = [PetabExperimentRow('scan_0', -5.0, 'cond_pre'),
                PetabExperimentRow('scan_0', 0.0, 'cond_scan_0'),
                PetabExperimentRow('scan_1', -7.0, 'cond_pre'),
                PetabExperimentRow('scan_1', 0.0, 'cond_scan_1')]
        with pytest.raises(PybnfError, match='ONE duration'):
            reconstruct_preequilibrated_dose_responses(meas, conds, exps, {'obs_resp': 'resp'},
                                                       sweepable={'L', 'kd'})

    @staticmethod
    def _retime_measurements(petab_dir, retime):
        """Rewrite measurements.tsv, mapping each row's time string through ``retime`` (a row it
        maps to ``None`` is kept unchanged)."""
        rows = _tsv_rows(petab_dir / 'measurements.tsv')
        header = list(rows[0])
        lines = ['\t'.join(header)]
        for r in rows:
            r = dict(r, time=retime(r['time']) or r['time'])
            lines.append('\t'.join(r[h] for h in header))
        (petab_dir / 'measurements.tsv').write_text('\n'.join(lines) + '\n')

    def test_measurement_inside_the_fixed_duration_period_is_refused(self, tmp_path_factory):
        # A PEtab measurement at a time in [-T, 0) is taken DURING the equilibration period, which
        # PEtab lint accepts. PyBNF's `preequilibrate:` + `equil_t_end:` equilibration is
        # unmeasured and its measured phase starts at the intervention (t = 0), so the point has
        # no PyBNF representation. Imported as-is it landed on the measured phase's time grid,
        # where bngsim starts integrating at the earliest sample time: every measurement of the
        # experiment was then scored 0.05 time units late, with no error. Refused instead.
        petab1, _, _, _ = _roundtrip(
            tmp_path_factory.mktemp('fixed_equil_early'), _FIXED_EQUIL_CONF,
            extra_files={'m.bngl': _FIXED_EQUIL_MODEL, 'relax.exp': _FIXED_EQUIL_EXP},
            model_name='m.bngl')
        self._retime_measurements(petab1, lambda t: '-0.05' if t == '0' else None)
        assert '-0.05' in [r['time'] for r in _tsv_rows(petab1 / 'measurements.tsv')]
        with pytest.raises(NotImplementedError,
                           match=r"Experiment 'relax'.*-0\.05.*inside its fixed-duration "
                                 r"equilibration period"):
            import_job(petab1 / 'problem.yaml', petab1.parent / 'imported_early')
        # A measurement at exactly 0 is the post-intervention state PyBNF measures: still imported.
        self._retime_measurements(petab1, lambda t: '0' if t == '-0.05' else None)
        import_job(petab1 / 'problem.yaml', petab1.parent / 'imported_zero')

    def test_scan_measured_inside_the_fixed_duration_period_is_refused(self, tmp_path_factory):
        # The pre-equilibrated scan sibling: every dose read at t = -100, inside the -7200
        # equilibration (before the wash and the dose are even applied). It imported as a scan
        # with `t_end: -100`, which BNG2.pl runs as no integration at all and bngsim rejects.
        conf = _PDR_CONF.replace(', t_end: 500,', ', t_end: 500, equil_t_end: 7200,')
        petab1, _, _, _ = _roundtrip(
            tmp_path_factory.mktemp('pdr_fixed_early'), conf,
            extra_files={'m.bngl': _PDR_MODEL, 'dose.exp': _PDR_DOSE_EXP}, model_name='m.bngl')
        self._retime_measurements(petab1, lambda t: '-100' if t == '500' else None)
        with pytest.raises(NotImplementedError,
                           match=r"Experiment 'scan_0'.*-100.*inside its fixed-duration "
                                 r"equilibration period"):
            import_job(petab1 / 'problem.yaml', petab1.parent / 'imported_early')


class TestPreequilibrationPeriodGrouping:
    """White-box on the multi-period resolver (`_condition_and_preequilibrate`, ADR-0052/#442):
    a single period is a plain time course; a leading time=-inf steady-state period + a finite
    measurement period is a pre-equilibration; a leading finite time=-T period + a measurement
    period at exactly 0 is a fixed-duration one (#896); any other finite leading period or >2
    periods raises rather than silently flattens."""

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

    def test_finite_leading_period_not_followed_by_time_zero_is_refused(self):
        # A finite leading period is a fixed-duration equilibration only in the exporter's shape
        # (-T, then the measured period at exactly 0 -- #896); any other finite pair would shift
        # the data times or the equilibration's duration, so refuse rather than flatten it.
        for times in ((100.0, 200.0), (-5.0, 1.0), (0.0, 3.0)):
            periods = [self._row(times[0], 'cond_pre'), self._row(times[1], 'cond_meas')]
            with pytest.raises(NotImplementedError, match='exactly time=0'):
                _condition_and_preequilibrate(periods, 'relax')

    def test_finite_leading_period_before_time_zero_is_a_fixed_duration_preequilibration(self):
        # #896: a leading period at -T followed by the measured period at 0 is preequilibrate:
        # with equil_t_end: T (the duration is read separately, by _fixed_equilibration_time).
        from pybnf.petab.import_ import _fixed_equilibration_time
        periods = [self._row(-7.5, 'cond_pre'), self._row(0.0, 'cond_meas')]
        assert _condition_and_preequilibrate(periods, 'relax') == ('meas', 'pre')
        assert _fixed_equilibration_time(periods) == 7.5
        # the steady-state shape carries no duration
        assert _fixed_equilibration_time(
            [self._row(float('-inf'), 'cond_pre'), self._row(0.0, 'cond_meas')]) is None

    def test_fixed_duration_period_with_a_blank_condition_is_refused(self):
        # An equilibration at the model defaults has no preequilibrate: condition to carry the
        # duration, so it is refused rather than imported as no equilibration at all (#896).
        periods = [self._row(-7.5, ''), self._row(0.0, 'cond_meas')]
        with pytest.raises(NotImplementedError, match='fixed-duration equilibration period'):
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


class TestPinnedDoseConditionsAreNotDoses:
    """A documented limitation, pinned until surrogate pins are handled exactly (a separate
    issue). #892 (fixed in the exporter separately) makes every per-dose condition of a plain
    dose-response re-pin the fit-and-perturbed parameters, ``cond_dr_0: L = 1, kd = kd__REF``.
    The dose detector reads a condition with its pins, so such a condition has two targets and is
    not a dose: each dose re-imports as its own conditioned experiment. At steady state that is
    the same fit; a ``t_end:`` scan fails loudly at load, since BNG2.pl needs three sample times.
    Dropping the pins first was tried and withdrawn: a pin is the identity only when its
    surrogate is estimated and no earlier period changed the parameter. The pinned tables are
    written here by hand, the way the #892 exporter writes them (each pin after its dose row)."""

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

    def test_a_pinned_steady_state_scan_imports_one_experiment_per_dose(self, tmp_path,
                                                                        monkeypatch):
        _src, out = self._pinned_import(tmp_path, None)
        text = (out / 'imported.conf').read_text()
        for i, (L, y) in enumerate(((1, 0.5), (2, 1.0), (5, 2.5))):
            assert (f'experiment: dr_{i}, condition: dr_{i}, method: ode, data: dr_{i}.exp'
                    in text)
            assert f'condition: dr_{i}, perturbations: L = {L}' in text
            data = Data(file_name=str(out / f'dr_{i}.exp'))
            assert np.isposinf(data['time'][0]) and list(data['resp']) == [y]
        assert 'experiment: dr,' not in text
        _load_conf(out, monkeypatch)                        # and the conf loads

    @pytest.mark.bngsim
    @pytest.mark.newera
    def test_a_pinned_steady_state_scan_scores_the_source_jobs_objective(self, tmp_path,
                                                                         monkeypatch):
        # Oracle: at fixed kd the imported job scores what the source job scores, and both equal
        # the hand closed form over every measurement.
        src, out = self._pinned_import(tmp_path, None)
        for kd in (0.8, 1.5):
            expected = self._closed_form_objective(kd, None)
            assert _imported_objective_at(out, monkeypatch, kd=kd) == pytest.approx(
                expected, rel=1e-5)
            assert _imported_objective_at(src, monkeypatch, conf_name='job.conf',
                                          kd=kd) == pytest.approx(expected, rel=1e-5)

    def test_a_pinned_t_end_scan_fails_loudly_at_load(self, tmp_path, monkeypatch):
        _src, out = self._pinned_import(tmp_path, 0.5)
        text = (out / 'imported.conf').read_text()
        assert 'experiment: dr_0, condition: dr_0, method: ode, data: dr_0.exp' in text
        with pytest.raises(PybnfError, match='requires 3 or more'):
            _load_conf(out, monkeypatch)


class TestPinAfterPreequilibrationRestoresTheParameter:
    """Independent review: a pin ``k = k__REF`` is the identity only when nothing earlier in the
    experiment changed k. Here the -inf period sets k = 2, and the measured period sets L to the
    dose and re-pins k = k__REF. Under PEtab v2 the measured period runs with k back at its
    estimate. ``conditions_from_rows`` drops the pin, and PyBNF carries a pre-equilibration
    setting into the measured phase (ADR-0052), so the imported job measures with k = 2 and k has
    no effect on its objective.

    With one measured time per experiment the import fails loudly at load, as on main: the dose
    condition keeps its pin, so it is not a dose, and each experiment is a one-time time course,
    which BNG2.pl cannot sample. With three measured times the experiments import as time courses
    and score 8.54 at k = 0.7 against 1.73 for the PEtab problem, silently, on main too. That is
    pre-existing and waits on the issue that handles pins exactly."""

    @staticmethod
    def _petab_A(t, dose, k):
        """Closed form under PEtab v2 period semantics: the -inf period (L = 5, k = 2) leaves
        A = 3. The measured period sets L = dose and restores k, so A relaxes from 3 toward
        (dose + 1)/k."""
        steady = (dose + 1) / k
        return steady + (3.0 - steady) * np.exp(-k * t)

    def _import(self, times, tmp_path):
        conditions = [('pre', 'L', 5), ('pre', 'k', 2)]
        experiments, measurements = [], []
        for dose in (1, 2, 3):
            conditions += [(f'd_{dose}', 'L', dose), (f'd_{dose}', 'k', 'k__REF')]
            experiments += [(f'scan_{dose}', '-inf', 'pre'), (f'scan_{dose}', '0', f'd_{dose}')]
            measurements += [('obs_A', f'scan_{dose}', t,
                              round(float(self._petab_A(t, dose, 1.0)) + 0.05 * dose, 6))
                             for t in times]
        root = tmp_path / 'problem'
        yaml = _write_periods_problem(root, conditions, experiments, measurements)
        (root / 'parameters.tsv').write_text(
            'parameterId\tlowerBound\tupperBound\tnominalValue\testimate\n'
            'k__REF\t0.1\t10\t1\ttrue\n')
        return import_job(yaml, tmp_path / 'out'), measurements

    def test_one_measured_time_fails_loudly_at_load(self, tmp_path, monkeypatch):
        out, _measurements = self._import((1.0,), tmp_path)
        text = (out / 'imported.conf').read_text()
        assert 'experiment: scan_1, preequilibrate: pre, condition: d_1, method: ode' in text
        with pytest.raises(PybnfError, match='requires 3 or more'):
            _load_conf(out, monkeypatch)

    @pytest.mark.bngsim
    @pytest.mark.newera
    @pytest.mark.xfail(strict=True, reason=(
        "pre-existing on main, awaiting the issue that handles surrogate pins exactly: "
        "conditions_from_rows drops the measured period's pin k = k__REF, so the time courses "
        "keep the pre-equilibration's k = 2"))
    def test_time_courses_score_the_petab_objective(self, tmp_path, monkeypatch):
        out, measurements = self._import((0.5, 1.0, 2.0), tmp_path)
        for k in (0.7, 1.3):
            expected = sum(0.5 * (self._petab_A(t, int(eid.split('_')[1]), k) - y) ** 2
                           for _obs, eid, t, y in measurements)
            assert _imported_objective_at(out, monkeypatch, k=k) == pytest.approx(
                expected, rel=1e-5)


class TestPinnedSteadyStateConditionIsNotADose:
    """Independent review: if the pins were dropped before dose detection, a steady-state
    experiment whose condition has one real target besides its pins would look like a dose
    point. Main gives that reading only to jobs with no fit-and-perturbed parameter. Two jobs the
    exporter writes, each with a fit-and-perturbed parameter (k, under 'fast'), import and load,
    as they do on main; with the pins dropped both failed, loudly.

    * Two steady-state experiments ss_1 and ss_2 under conditions that set different parameters
      are taken for one scan 'ss' and refused as ambiguous.
    * A steady-state experiment and a time course that share a condition: the steady state is
      taken for a scan, the scan consumes the condition, and the time course then names a
      condition the conf does not define, so the conf fails to load.

    Main refuses both shapes, in the same two ways, when the job has no fit-and-perturbed
    parameter. The pins used to shield these jobs from that."""

    SOURCES = {
        'two steady states with one stem': (
            'condition: hiL, perturbations: L = 3\n'
            'condition: noflag, perturbations: flag = 0\n'
            'experiment: ss_1, condition: hiL, data: ss_1.exp\n'
            'experiment: ss_2, condition: noflag, data: ss_2.exp\n'),
        'a condition shared with a time course': (
            'condition: hiL, perturbations: L = 3\n'
            'experiment: ss_1, condition: hiL, data: ss_1.exp\n'
            'experiment: tc2, condition: hiL, data: tc.exp\n'),
    }

    @pytest.mark.parametrize('shape', list(SOURCES))
    def test_exported_job_imports_and_loads(self, shape, tmp_path, monkeypatch):
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'model.bngl').write_text(_PERIODS_MODEL)
        (src / 'ss_1.exp').write_text('# time A_tot\ninf\t2.1\n')
        (src / 'ss_2.exp').write_text('# time A_tot\ninf\t0.9\n')
        (src / 'tc.exp').write_text('# time A_tot\n0.5\t0.4\n1\t0.7\n2\t0.9\n')
        (src / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = sos\nmodel: model.bngl\n'
            'condition: fast, perturbations: k * 2\n' + self.SOURCES[shape]
            + 'experiment: tc, condition: fast, data: tc.exp\nuniform_var = k 0.1 10\n')
        export_job(src / 'job.conf', tmp_path / 'petab')
        assert ('cond_hiL', 'k', 'k__REF') in {
            (r['conditionId'], r['targetId'], r['targetValue'])
            for r in _tsv_rows(tmp_path / 'petab' / 'conditions.tsv')}     # M = {k}: pinned
        out = import_job(tmp_path / 'petab' / 'problem.yaml', tmp_path / 'out')
        _load_conf(out, monkeypatch)


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

    @staticmethod
    def _pins_only_pair_problem(root, applied):
        """cond_a only re-pins the fit parameter (k = k__REF, the identity after import); a sets
        L = 3 and re-pins k too. Both import as the condition 'a'. e1 applies cond_a, and e2
        applies a when ``applied`` names it. k is estimated through its surrogate k__REF."""
        experiments = [('e1', '0', 'cond_a')] + ([('e2', '0', 'a')] if 'a' in applied else [])
        yaml = _write_periods_problem(
            root, [('cond_a', 'k', 'k__REF'), ('a', 'L', 3), ('a', 'k', 'k__REF')], experiments,
            [('obs_A', eid, t, 1.0 + t) for eid, _time, _cid in experiments for t in (0.5, 1, 2)])
        (root / 'parameters.tsv').write_text(
            'parameterId\tlowerBound\tupperBound\tnominalValue\testimate\n'
            'k__REF\t0.1\t10\t1\ttrue\n')
        return yaml

    def test_a_pins_only_id_applied_with_its_colliding_pair_is_refused(self, tmp_path):
        # Independent review of the follow-up: the pins were dropped before the collision check,
        # so cond_a had no rows left and was not seen to collide. e1 was imported under 'a' with
        # a's L = 3, and the imported job scored 5.10 at k = 0.7 against 0.63 for the PEtab
        # problem (e1 at L = 1), with no error. A pins-only id still takes part in the check.
        yaml = self._pins_only_pair_problem(tmp_path / 'problem', ('cond_a', 'a'))
        with pytest.raises(PybnfError, match=r"PEtab conditions 'cond_a' and 'a' both import as "
                                             r"the PyBNF condition 'a', and experiments apply "
                                             r"each of them"):
            import_job(yaml, tmp_path / 'out')

    def test_a_pins_only_id_does_not_take_an_unused_ids_targets(self, tmp_path, monkeypatch):
        # Only cond_a is applied, so it is the id kept under the name 'a' and the unused a (L = 3)
        # is left out. cond_a has no target left once its pin is dropped, so e1 names a condition
        # the conf does not define, and the conf refuses to load. That is loud; the follow-up
        # gave e1 the unused condition's L = 3 instead and fitted it (4.53 at k = 0.7 against
        # 0.056 for the PEtab problem).
        yaml = self._pins_only_pair_problem(tmp_path / 'problem', ('cond_a',))
        out = import_job(yaml, tmp_path / 'out')
        assert 'L = 3' not in (out / 'imported.conf').read_text()
        with pytest.raises(PybnfError, match=r"Experiment 'e1' references condition 'a'"):
            _load_conf(out, monkeypatch)


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

    def test_relative_condition_round_trips_against_its_own_model(self, tmp_path):
        # #897: two models give the fixed parameter L different values (a: 1, b: 5); the
        # condition `L * 2` belongs to b. It must round-trip as b's doubled value, L = 10 -- the
        # base the fitter uses -- not a's (L = 2, which moved the optimum from k = 1 to k = 4.2).
        model = _GROWTH_BNGL.replace('    a2 2\n', '    a2 2\n    L 1\n')
        assert '    L 1\n' in model
        conf = ('edition = 2\njob_type = de\nobjective = chi_sq\n'
                f'model: {DEMO_MODEL}\nmodel: growth_v2.bngl\n'
                'condition: dbl, model: growth_v2.bngl, perturbations: L * 2\n'
                f'experiment: pa, model: {DEMO_MODEL}, data: pa.exp\n'
                'experiment: gr, model: growth_v2.bngl, condition: dbl, data: gr.exp\n'
                + _PARAMS_U + 'uniform_var = a2 0 10\n')
        demo_with_l = (DEMO_DIR / DEMO_MODEL).read_text().replace(
            'begin parameters', 'begin parameters\n    L 5', 1)
        petab1, _, petab2, imported_conf = _roundtrip(
            tmp_path, conf, extra_files={**self._EXTRA, 'growth_v2.bngl': model,
                                         DEMO_MODEL: demo_with_l})
        _assert_problem_round_trips(petab1, petab2)
        assert ('condition: dbl, model: growth_v2.bngl, perturbations: L = 2'
                in imported_conf.read_text())
        # ...and with the models' values swapped, the same condition folds to 10.
        (tmp_path / 'swapped').mkdir()
        petab1, _, _, imported_conf = _roundtrip(
            tmp_path / 'swapped', conf,
            extra_files={**self._EXTRA, 'growth_v2.bngl': model.replace('    L 1', '    L 5'),
                         DEMO_MODEL: demo_with_l.replace('    L 5', '    L 1')})
        assert ('condition: dbl, model: growth_v2.bngl, perturbations: L = 10'
                in imported_conf.read_text())


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
# A column-mean sigma comes back as column_mean only when it IS each experiment's own mean
# (#894). The exporter writes it per experiment (a constant, or each row's noiseParameters);
# the importer compares every value with the mean of the experiment it will belong to in the
# imported job, and anything else stays a fixed sigma. The old check compared a constant with
# one mean pooled over every experiment, so a problem with a pooled sigma (the pre-#894
# export, or any foreign problem that happens to use one) came back as ave_norm_sos and was
# fitted with per-experiment weights it never had.
# ---------------------------------------------------------------------------

_CM_SS_MODEL = _SS_MODEL.replace('  k_deg   2.0\n', '  k_deg   2.0\n  u       1\n')

# Each experiment shape with two experiments of different magnitude under ave_norm_sos:
# (conf, extra files, model file, {petab experimentId -> the .exp files of its PyBNF experiment}).
_CM_SHAPES = {
    'preequilibration': (
        _PREEQUIL_CONF.replace('objective = sos', 'objective = ave_norm_sos')
        + 'experiment: relax10, preequilibrate: pre, condition: meas, data: relax10.exp\n',
        {'m.bngl': _PREEQUIL_MODEL, 'relax.exp': _PREEQUIL_EXP,
         'relax10.exp': '# time A_tot\n0\t100\n1\t60\n2\t40\n'},
        'm.bngl', lambda row: {'relax': ['relax.exp'], 'relax10': ['relax10.exp']}[
            row['experimentId']]),
    # A two-target condition, so the conditioned steady state is not read as a one-dose scan.
    'steady_state': (
        'edition = 2\njob_type = de\nobjective = ave_norm_sos\nmodel: ss.bngl\n'
        'condition: slow, perturbations: k_deg = 0.2, u = 2\n'
        'experiment: eq, data: eq.exp\nexperiment: eq_slow, condition: slow, data: slow.exp\n'
        'uniform_var = k_prod 0.1 10\n',
        {'ss.bngl': _CM_SS_MODEL, 'eq.exp': _SS_EXP, 'slow.exp': '# time A_tot\ninf\t15\n'},
        'ss.bngl', lambda row: {'': ['eq.exp'], 'eq_slow': ['slow.exp']}[row['experimentId']]),
    'preequilibrated_dose_response': (
        _PDR_CONF.replace('objective = sos', 'objective = ave_norm_sos')
        + 'experiment: scan2, preequilibrate: incubate, condition: wash, '
          'type: parameter_scan, t_end: 500, data: dose2.exp\n',
        {'m.bngl': _PDR_MODEL, 'dose.exp': _PDR_DOSE_EXP,
         'dose2.exp': '# L resp\n1\t5\n2\t10\n5\t25\n'},
        'm.bngl', lambda row: {'scan': ['dose.exp'], 'scan2': ['dose2.exp']}[
            row['experimentId'].rsplit('_', 1)[0]]),
    # Two wildtype experiments on two models share experimentId ''; the modelId tells them apart.
    'multi_model': (
        'edition = 2\njob_type = de\nobjective = ave_norm_sos\n'
        'model: decay.bngl\nmodel: decay2.bngl\n'
        'experiment: lo, model: decay.bngl, data: lo.exp\n'
        'experiment: big, model: decay2.bngl, data: hi.exp\n'
        'uniform_var = k 0.05 3.0\n',
        {'decay.bngl': _DECAY_MODEL, 'decay2.bngl': _DECAY_MODEL,
         'lo.exp': _exp_text('time', _DECAY_TIMES, _decay_series(100, 0.5)),
         'hi.exp': _exp_text('time', _DECAY_TIMES, _decay_series(1000, 0.7))},
        'decay.bngl', lambda row: {'decay': ['lo.exp'], 'decay2': ['hi.exp']}[row['modelId']]),
}


def _column_mean_of_files(src, files):
    """The mean of the measured column (column 1) over an experiment's .exp files, by numpy."""
    return np.concatenate([np.atleast_2d(np.loadtxt(src / f))[:, 1] for f in files]).mean()


def _pooled_form(petab, sigma):
    """Rewrite an exported problem so its one observable carries the constant ``sigma`` and
    no per-row noise -- what the pre-#894 exporter wrote."""
    obs = _tsv_rows(petab / 'observables.tsv')
    header = list(obs[0])
    for r in obs:
        r['noiseFormula'], r['noisePlaceholders'] = repr(float(sigma)), ''
    (petab / 'observables.tsv').write_text(
        '\t'.join(header) + '\n' + ''.join('\t'.join(r[h] for h in header) + '\n' for r in obs))
    lines = (petab / 'measurements.tsv').read_text().splitlines()
    assert lines[0].split('\t')[-1] == 'noiseParameters'
    (petab / 'measurements.tsv').write_text(
        '\n'.join([lines[0]] + [ln.rsplit('\t', 1)[0] + '\t' for ln in lines[1:]]) + '\n')


class TestColumnMeanSigmaImport:
    """#894: the importer's side of the per-experiment column-mean sigma."""

    RECOVERED = {
        'ave_norm_sos': 'objective = ave_norm_sos',
        'gaussian_override': 'objective = ave_norm_sos',    # its only observable -> uniform
        'laplace_whole_fit': 'noise_model = laplace, scale = column_mean',
        'lnnormal_whole_fit': 'noise_model = lnnormal, sigma = column_mean',
    }

    def _decay_round_trip(self, tmp_path, noise_lines):
        src = tmp_path / 'src'
        src.mkdir()
        conf = _write_decay_job(src, noise_lines)
        petab1, imported, petab2 = tmp_path / 'petab1', tmp_path / 'imported', tmp_path / 'petab2'
        export_job(conf, petab1)
        import_job(petab1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', petab2)
        return conf, petab1, imported, petab2

    @pytest.mark.parametrize('spelling', sorted(RECOVERED))
    def test_round_trip_restores_the_column_mean(self, tmp_path, spelling, monkeypatch):
        # Time courses (one with two replicate files), a conditioned time course and a
        # dose-response scan: the per-row export comes back as a column_mean sigma, the
        # rebuilt _SD companions are gone (column_mean reads no data column, and the fitter
        # refuses one nothing reads), the re-export is byte-identical, and the imported job
        # scores exactly as the source job does at three k.
        noise_lines = _COLUMN_MEAN_SPELLINGS[spelling][0]
        conf, petab1, imported, petab2 = self._decay_round_trip(tmp_path, noise_lines)
        text = (imported / 'imported.conf').read_text()
        assert self.RECOVERED[spelling] in text.splitlines()
        assert 'fix_at' not in text and 'read_exp_file' not in text
        for exp in imported.glob('*.exp'):
            assert not any(c.endswith('_SD') for c in Data(file_name=str(exp)).cols), exp.name
        _assert_problem_round_trips(petab1, petab2)
        for k in (0.4, 0.6, 0.9):
            assert _pybnf_objective(imported / 'imported.conf', k, monkeypatch) == \
                pytest.approx(_pybnf_objective(conf, k, monkeypatch), rel=1e-12)

    @pytest.mark.parametrize('shape', sorted(_CM_SHAPES))
    def test_every_experiment_shape_round_trips(self, tmp_path, shape):
        # Each row carries its own PyBNF experiment's mean (numpy over that experiment's .exp
        # files: a pre-equilibrated scan's doses all share the scan's mean, two models' wildtype
        # experiments keep theirs), and the import restores ave_norm_sos byte-for-byte.
        conf_text, files, model_name, files_of = _CM_SHAPES[shape]
        petab1, imported, petab2, conf = _roundtrip(
            tmp_path, conf_text, extra_files=files, model_name=model_name)
        src = tmp_path / 'src'
        rows = _tsv_rows(petab1 / 'measurements.tsv')
        means = {float(r['noiseParameters']) for r in rows}
        assert len(means) == 2           # two experiments, two different means
        for r in rows:
            assert float(r['noiseParameters']) == pytest.approx(
                _column_mean_of_files(src, files_of(r)), rel=1e-15)
        assert 'objective = ave_norm_sos' in conf.read_text().splitlines()
        for exp in imported.glob('*.exp'):
            assert not any(c.endswith('_SD') for c in Data(file_name=str(exp)).cols), exp.name
        _assert_problem_round_trips(petab1, petab2)

    def test_per_observable_column_mean_comes_back_as_its_own_line(self, tmp_path, monkeypatch):
        # Two observables: Obs_A's sigma is its column mean (per row -- the experiments differ),
        # Obs_B keeps the unit sigma of sos. The import restores one line per observable and
        # drops both rebuilt _SD companions -- Obs_B's is all NaN, and a leftover one would
        # make the fitter refuse the data -- so the imported job scores like the source.
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'decay.bngl').write_text(_DECAY_MODEL.replace(
            '  Molecules  Obs_A  A()\n', '  Molecules  Obs_A  A()\n  Molecules  Obs_B  A()\n'))
        for name, amp, rate in (('lo', 100, 0.5), ('hi', 1000, 0.7)):
            a, b = _decay_series(amp, rate), _decay_series(1.1 * amp, rate)
            (src / f'{name}.exp').write_text('# time Obs_A Obs_B\n' + ''.join(
                f'{t!r}\t{float(x)!r}\t{float(y)!r}\n' for t, x, y in zip(_DECAY_TIMES, a, b)))
        conf = src / 'job.conf'
        conf.write_text(
            'edition = 2\njob_type = de\nmodel: decay.bngl\nobjective = sos\n'
            'noise_model Obs_A = gaussian, sigma = column_mean\n'
            'condition: high, perturbations: scale = 10\n'
            'experiment: lo, data: lo.exp\nexperiment: hi, condition: high, data: hi.exp\n'
            'uniform_var = k 0.05 3.0\npopulation_size = 12\nmax_iterations = 5\n')
        petab1, imported, petab2 = tmp_path / 'petab1', tmp_path / 'imported', tmp_path / 'petab2'
        export_job(conf, petab1)
        import_job(petab1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', petab2)
        lines = (imported / 'imported.conf').read_text().splitlines()
        assert 'noise_model Obs_A = gaussian, sigma = column_mean' in lines
        assert 'noise_model Obs_B = gaussian, sigma = fix_at 1' in lines
        for exp in imported.glob('*.exp'):
            assert not any(c.endswith('_SD') for c in Data(file_name=str(exp)).cols), exp.name
        _assert_problem_round_trips(petab1, petab2)
        for k in (0.4, 0.9):
            assert _pybnf_objective(imported / 'imported.conf', k, monkeypatch) == \
                pytest.approx(_pybnf_objective(conf, k, monkeypatch), rel=1e-12)

    def test_a_pooled_constant_is_a_fixed_sigma_not_a_column_mean(self, tmp_path, monkeypatch):
        # A constant sigma equal to the mean pooled over experiments of different magnitude
        # (the pre-#894 export; a foreign problem could carry one too) is NO experiment's
        # column mean. It must import as that fixed sigma: libpetab's likelihood of the problem
        # and the imported job's objective then differ only by a constant. Imported as
        # ave_norm_sos, the job would weight each experiment by its own mean instead.
        pytest.importorskip('petab.v2')
        conf, petab1, _, _ = self._decay_round_trip(tmp_path, 'objective = ave_norm_sos\n')
        values = [np.atleast_2d(np.loadtxt(conf.parent / f))[:, 1]
                  for f in ('lo.exp', 'lo_rep.exp', 'hi.exp', 'scan.exp')]
        pooled = float(np.concatenate(values).mean())
        _pooled_form(petab1, pooled)
        out = import_job(petab1 / 'problem.yaml', tmp_path / 'pooled_import')
        text = (out / 'imported.conf').read_text()
        assert 'ave_norm_sos' not in text and 'column_mean' not in text
        assert f'noise_model = gaussian, sigma = fix_at {num(pooled)}' in text.splitlines()
        ks = (0.4, 0.6, 0.9)
        offsets = [_petab_nll(petab1, k) - _pybnf_objective(out / 'imported.conf', k, monkeypatch)
                   for k in ks]
        assert offsets == pytest.approx([offsets[0]] * 3, rel=0, abs=1e-8)

    def test_a_row_off_its_experiment_mean_stays_a_per_point_sigma(self, tmp_path, monkeypatch):
        # Every number must match: one row's noiseParameters 1% off its experiment's mean and
        # the observable is not a column mean. It stays a per-point sigma (chi_sq, reading
        # the rebuilt _SD column), which scores exactly what the PEtab problem says.
        pytest.importorskip('petab.v2')
        _conf, petab1, _, _ = self._decay_round_trip(tmp_path, 'objective = ave_norm_sos\n')
        lines = (petab1 / 'measurements.tsv').read_text().splitlines()
        head, sigma = lines[5].rsplit('\t', 1)
        lines[5] = f'{head}\t{float(sigma) * 1.01!r}'
        (petab1 / 'measurements.tsv').write_text('\n'.join(lines) + '\n')
        out = import_job(petab1 / 'problem.yaml', tmp_path / 'off_import')
        text = (out / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text.splitlines() and 'column_mean' not in text
        assert any('Obs_A_SD' in Data(file_name=str(p)).cols for p in out.glob('*.exp'))
        ks = (0.4, 0.6, 0.9)
        offsets = [_petab_nll(petab1, k) - _pybnf_objective(out / 'imported.conf', k, monkeypatch)
                   for k in ks]
        assert offsets == pytest.approx([offsets[0]] * 3, rel=0, abs=1e-8)

    def test_two_wildtype_experiments_merge_and_stay_per_point(self, tmp_path, monkeypatch):
        # Two wildtype time courses on one model share PEtab experimentId '' (nothing tells
        # them apart), so the import rebuilds them as ONE experiment with two replicates, whose
        # mean is neither of theirs. Their per-row means are therefore not that experiment's
        # column mean: they stay per-point sigmas, and the imported job still scores exactly
        # what the exported problem and the source fit do.
        pytest.importorskip('petab.v2')
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'decay.bngl').write_text(_DECAY_MODEL)
        (src / 'lo.exp').write_text(_exp_text('time', _DECAY_TIMES, _decay_series(100, 0.5)))
        (src / 'lo2.exp').write_text(_exp_text('time', _DECAY_TIMES, _decay_series(30, 0.4)))
        conf = src / 'job.conf'
        conf.write_text(
            'edition = 2\njob_type = de\nmodel: decay.bngl\nobjective = ave_norm_sos\n'
            'experiment: lo, data: lo.exp\nexperiment: lo2, data: lo2.exp\n'
            'uniform_var = k 0.05 3.0\npopulation_size = 12\nmax_iterations = 5\n')
        petab1, imported, petab2 = tmp_path / 'petab1', tmp_path / 'imported', tmp_path / 'petab2'
        export_job(conf, petab1)
        import_job(petab1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', petab2)
        text = (imported / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text.splitlines() and 'column_mean' not in text
        _assert_problem_round_trips(petab1, petab2)
        ks = (0.4, 0.6, 0.9)
        fit = [_pybnf_objective(conf, k, monkeypatch) for k in ks]
        back = [_pybnf_objective(imported / 'imported.conf', k, monkeypatch) for k in ks]
        assert back == pytest.approx(fit, rel=1e-12)
        offsets = [_petab_nll(petab1, k) - f for k, f in zip(ks, fit)]
        assert offsets == pytest.approx([offsets[0]] * 3, rel=0, abs=1e-8)

    def test_a_small_magnitude_sigma_is_compared_relatively(self, tmp_path):
        # Data of order 1e-11 with a fixed sigma five times their mean. The old comparison
        # had an absolute floor of 1e-9, under which the two "matched" and the problem came
        # back as ave_norm_sos -- a sigma five times too small, weights 25 times too large.
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'decay.bngl').write_text(_DECAY_MODEL)
        ys = [1e-12 * y for y in _decay_series(100, 0.5)]
        (src / 'lo.exp').write_text(_exp_text('time', _DECAY_TIMES, ys))
        sigma = 5 * float(np.mean(ys))
        (src / 'job.conf').write_text(
            'edition = 2\njob_type = de\nmodel: decay.bngl\n'
            f'noise_model = gaussian, sigma = fix_at {sigma!r}\n'
            'experiment: lo, data: lo.exp\nuniform_var = k 0.05 3.0\n')
        petab1, petab2 = tmp_path / 'petab1', tmp_path / 'petab2'
        export_job(src / 'job.conf', petab1)
        out = import_job(petab1 / 'problem.yaml', tmp_path / 'imported')
        text = (out / 'imported.conf').read_text()
        assert 'ave_norm_sos' not in text
        assert f'noise_model = gaussian, sigma = fix_at {num(sigma)}' in text.splitlines()
        export_job(out / 'imported.conf', petab2)
        _assert_problem_round_trips(petab1, petab2)

    @pytest.mark.parametrize('edit', ['each_dose_its_own_value', 'one_dose_off'])
    def test_doses_with_different_sigmas_stay_per_point(self, tmp_path, edit, monkeypatch):
        # Added in independent review. The doses of a scan are separate PEtab experiments that
        # import into ONE PyBNF experiment, so their sigmas must all equal that one scan's mean
        # before the import may say column_mean. Two edits of the scan rows break that: every
        # dose carries its own measurement (the mean of its own one-row PEtab experiment), or
        # one dose is 1% off the scan mean while the others keep it. Either way the scan stays
        # a per-point sigma, and libpetab's likelihood of the problem and the imported job's
        # objective differ only by a constant.
        pytest.importorskip('petab.v2')
        _conf, petab1, _, _ = self._decay_round_trip(tmp_path, 'objective = ave_norm_sos\n')
        lines = (petab1 / 'measurements.tsv').read_text().splitlines()
        head = lines[0].split('\t')
        rows = [ln.split('\t') for ln in lines[1:]]
        eid, meas = head.index('experimentId'), head.index('measurement')
        scan_rows = [r for r in rows if r[eid].startswith('scan_')]
        assert len(scan_rows) == 3
        if edit == 'each_dose_its_own_value':
            for r in scan_rows:
                r[-1] = r[meas]
        else:
            scan_rows[1][-1] = repr(float(scan_rows[1][-1]) * 1.01)
        (petab1 / 'measurements.tsv').write_text(
            '\n'.join(['\t'.join(head)] + ['\t'.join(r) for r in rows]) + '\n')
        out = import_job(petab1 / 'problem.yaml', tmp_path / 'edited_import')
        text = (out / 'imported.conf').read_text()
        assert 'objective = chi_sq' in text.splitlines() and 'column_mean' not in text
        assert 'Obs_A_SD' in Data(file_name=str(out / 'scan.exp')).cols
        ks = (0.4, 0.6, 0.9)
        offsets = [_petab_nll(petab1, k) - _pybnf_objective(out / 'imported.conf', k, monkeypatch)
                   for k in ks]
        assert offsets == pytest.approx([offsets[0]] * 3, rel=0, abs=1e-8)

    def test_ragged_replicates_round_trip_to_the_column_mean(self, tmp_path, monkeypatch):
        # Added in independent review. Ragged replicates on different time grids, NaN cells, and
        # one table mixing the constant form (Obs_A, one experiment) with the per-row form
        # (Obs_B, two experiments). The import regroups the replicate rows by time, so its mean
        # is summed in another order than the export's; both observables must still come back
        # as ONE ave_norm_sos line, with no _SD companion left for the fitter to refuse, and the
        # imported job must score exactly like the source job.
        src = tmp_path / 'src'
        src.mkdir()
        conf = _write_ragged_job(src)
        petab1, imported = tmp_path / 'petab1', tmp_path / 'imported'
        export_job(conf, petab1)
        import_job(petab1 / 'problem.yaml', imported)
        lines = (imported / 'imported.conf').read_text().splitlines()
        assert 'objective = ave_norm_sos' in lines
        assert not any('noise_model' in ln for ln in lines)
        for exp in imported.glob('*.exp'):
            assert not any(c.endswith('_SD') for c in Data(file_name=str(exp)).cols), exp.name
        for k in (0.4, 0.6, 0.9):
            assert _pybnf_objective(imported / 'imported.conf', k, monkeypatch) == \
                pytest.approx(_pybnf_objective(conf, k, monkeypatch), rel=1e-12)


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
            exps, conds, fit_params={'v1', 'v2', 'v3'}, nominal_of=lambda _c, _v: 2.0)
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

    # #902: the reader keeps EVERY file listed under a key, reads the one-line flow form
    # (`key: [a.tsv, b.tsv]`) on every key, and refuses what it cannot read. Each shape below
    # is a valid PEtab v2 problem.yaml; the oracle is PyYAML, the parser libpetab reads it with.
    _VALID_SHAPES = {
        'block_lists_two_files_each': (
            'format_version: 2.0.0\n'
            'parameter_files:\n  - parameters.tsv\n  - parameters2.tsv\n'
            'observable_files:\n  - observables.tsv\n  - observables2.tsv\n'
            'measurement_files:\n  - measurements.tsv\n  - measurements2.tsv\n'
            'condition_files:\n  - conditions.tsv\n  - conditions2.tsv\n'
            'experiment_files:\n  - experiments.tsv\n  - experiments2.tsv\n'
            'mapping_files:\n  - mapping.tsv\n  - mapping2.tsv\n'
            'model_files:\n  m:\n    location: m.bngl\n    language: bngl\n'),
        'petab1to2_column0_lists': (
            'format_version: 2.0.0\n'
            'id: split_problem\n'
            'model_files:\n  m:\n    location: m.xml\n    language: sbml\n'
            'parameter_files:\n- parameters.tsv\n'
            'measurement_files:\n- measurements.tsv\n- measurements2.tsv\n'
            'condition_files:\n- conditions.tsv\n- conditions2.tsv\n'
            'experiment_files:\n- experiments.tsv\n'
            'observable_files:\n- observables.tsv\n- observables2.tsv\n'
            'mapping_files: []\n'
            'extensions: {}\n'),
        'flow_lists_on_every_key': (
            'format_version: 2.0.0\n'
            'parameter_files: [parameters.tsv, parameters2.tsv]\n'
            'observable_files: [observables.tsv]\n'
            'measurement_files: [measurements.tsv, measurements2.tsv]\n'
            'condition_files: [conditions.tsv]\n'
            'experiment_files: [experiments.tsv]\n'
            'mapping_files: [mapping.tsv]\n'
            'model_files:\n  m:\n    location: m.bngl\n    language: bngl\n'),
        'quotes_comments_trailing_comma': (
            '# A problem written by hand.\n'
            'format_version: "2.0.0"   # quoted\n'
            "parameter_files: ['parameters.tsv', \"parameters 2.tsv\",]  # trailing comma\n"
            "observable_files:\n  - 'observables.tsv'   # a comment after an item\n"
            'measurement_files:\n  - "measurements.tsv"\n  - m#2.tsv\n'
            'condition_files: []\n'
            "model_files:\n  m:   # the only model\n    location: 'm.bngl'\n    language: bngl\n"),
        'two_models_listed_first': (
            '---\n'
            'format_version: 2.0.0\n'
            'model_files:\n'
            '    first:\n        language: bngl\n        location: a.bngl\n'
            '    second:\n        location: b.xml\n        language: sbml\n'
            'parameter_files:\n    - parameters.tsv\n'
            'observable_files: [observables.tsv]\n'
            'measurement_files:\n    - measurements_a.tsv\n    - measurements_b.tsv\n'),
        'byte_order_mark': (
            '\ufeffparameter_files:\n- parameters.tsv\n'
            'format_version: 2.0.0\n'
            'observable_files:\n- observables.tsv\n'
            'measurement_files:\n- measurements.tsv\n'
            'model_files:\n  m:\n    location: m.bngl\n    language: bngl\n'),
    }

    @pytest.mark.parametrize('shape', sorted(_VALID_SHAPES))
    def test_reads_every_valid_shape_as_pyyaml_does(self, tmp_path, shape):
        yaml = pytest.importorskip('yaml')
        text = self._VALID_SHAPES[shape]
        (tmp_path / 'problem.yaml').write_text(text)
        got = read_problem_yaml(tmp_path / 'problem.yaml')
        want = yaml.safe_load(text)
        for key in ('parameter_files', 'observable_files', 'measurement_files',
                    'condition_files', 'experiment_files', 'mapping_files'):
            assert got[key] == (want.get(key) or []), key
        assert ([(m['model_id'], m['location'], m['language']) for m in got['models']]
                == [(mid, m['location'], m['language'])
                    for mid, m in want['model_files'].items()])

    def test_flow_list_on_an_optional_key_is_read(self, tmp_path):
        # Before #902 the flow form on condition/experiment/mapping keys read as an EMPTY
        # list with no message, so the whole table was dropped from the import.
        (tmp_path / 'problem.yaml').write_text(self._VALID_SHAPES['flow_lists_on_every_key'])
        problem = read_problem_yaml(tmp_path / 'problem.yaml')
        assert problem['condition_files'] == ['conditions.tsv']
        assert problem['experiment_files'] == ['experiments.tsv']
        assert problem['mapping_files'] == ['mapping.tsv']
        assert problem['measurement_files'] == ['measurements.tsv', 'measurements2.tsv']

    _BASE = ('format_version: 2.0.0\n'
             'parameter_files:\n  - parameters.tsv\n'
             'observable_files:\n  - observables.tsv\n'
             'measurement_files:\n  - measurements.tsv\n'
             'model_files:\n  m:\n    location: m.bngl\n    language: bngl\n')

    @pytest.mark.parametrize('extra, match', [
        # A scalar where the schema requires a list (libpetab refuses it too).
        ('condition_files: conditions.tsv\n', "'condition_files' must be a list of files"),
        # A flow list continued on the next line.
        ('condition_files: [conditions.tsv,\n  conditions2.tsv]\n',
         "'condition_files' must be a list of files"),
        # A plain line where a '- file' item belongs.
        ('experiment_files:\n  experiments.tsv\n',
         "'experiment_files' must be a list of files, one '- <file>' line each"),
        # A nested list as an item.
        ('mapping_files:\n  - [a.tsv, b.tsv]\n', 'uses YAML syntax this reader does not read'),
        # The v1 singular spelling, or any key PEtab v2 does not define.
        ('condition_file: conditions.tsv\n', r"\['condition_file'\], which PEtab v2 does not"),
        # A key given twice (PyYAML would silently keep the second list).
        ('measurement_files:\n  - measurements2.tsv\n',
         "gives the key 'measurement_files' twice"),
        # The same file twice under one key (its rows would be read twice).
        ('condition_files: [conditions.tsv, conditions.tsv]\n',
         r"lists \['conditions.tsv'\] more than once under condition_files"),
    ])
    def test_unreadable_shape_is_refused(self, tmp_path, extra, match):
        (tmp_path / 'problem.yaml').write_text(self._BASE + extra)
        with pytest.raises(PybnfError, match=match):
            read_problem_yaml(tmp_path / 'problem.yaml')

    @pytest.mark.parametrize('model_block, match', [
        ('model_files: {m: {location: m.bngl, language: bngl}}\n', 'YAML flow form'),
        ('model_files:\n  m: {location: m.bngl, language: bngl}\n',
         "cannot read the model_files entry 'm: {location"),
        ('model_files:\n  m:\n    location: a.bngl\n  m:\n    location: b.bngl\n',
         "declares the model 'm' twice"),
    ])
    def test_unreadable_model_files_is_refused(self, tmp_path, model_block, match):
        # A flow-form or repeated model entry used to be skipped: with two models, the skipped
        # one simply vanished from the import.
        base = self._BASE.split('model_files:')[0]
        (tmp_path / 'problem.yaml').write_text(base + model_block)
        with pytest.raises(PybnfError, match=match):
            read_problem_yaml(tmp_path / 'problem.yaml')

    def test_petab_v1_problem_is_named_as_such(self, tmp_path):
        (tmp_path / 'problem.yaml').write_text(
            'format_version: 1\nparameter_file: parameters.tsv\nproblems:\n'
            '  - sbml_files: [model.xml]\n    measurement_files: [measurements.tsv]\n')
        with pytest.raises(PybnfError, match='declares format_version 1') as err:
            read_problem_yaml(tmp_path / 'problem.yaml')
        assert 'petab1to2_preserve_scale' in err.value.message

    # Review of #902. The PEtab v2 schema lets a model entry carry fields beyond location and
    # language (it sets no additionalProperties: false there), and such a field may hold a
    # nested mapping with a `location:` key of its own. The reader took that nested key for
    # the model's location, so the import silently used another model file. And a problem file
    # opening with a YAML directive or closing with the `...` end marker (PyYAML's
    # explicit_end) read on main but was refused. Oracle: PyYAML, which libpetab reads with.
    _MORE_VALID_SHAPES = {
        'nested_field_after_location': _BASE.replace(
            '    language: bngl\n',
            '    language: bngl\n    provenance:\n      location: variant.bngl\n'
            '      language: sbml\n'),
        'nested_field_before_location': _BASE.replace(
            '    location: m.bngl\n',
            '    provenance:\n      location: variant.bngl\n    location: m.bngl\n'),
        'nested_list_at_field_indent': _BASE.replace(
            '    language: bngl\n', '    language: bngl\n    tags:\n    - location: x\n'),
        'document_end_marker': _BASE + '...\n',
        'yaml_directive': '%YAML 1.1\n---\n' + _BASE,
    }

    @pytest.mark.parametrize('shape', sorted(_MORE_VALID_SHAPES))
    def test_more_valid_shapes_read_as_pyyaml_does(self, tmp_path, shape):
        yaml = pytest.importorskip('yaml')
        text = self._MORE_VALID_SHAPES[shape]
        (tmp_path / 'problem.yaml').write_text(text)
        got = read_problem_yaml(tmp_path / 'problem.yaml')
        want = yaml.safe_load(text)
        assert ([(m['model_id'], m['location'], m['language']) for m in got['models']]
                == [(mid, m['location'], m['language'])
                    for mid, m in want['model_files'].items()]
                == [('m', 'm.bngl', 'bngl')])
        for key in _TABLE_KEYS:
            assert got[key] == (want.get(key) or []), key

    @pytest.mark.parametrize('model_block, match', [
        # A location continued on a second line (a multi-line YAML scalar).
        ('model_files:\n  m:\n    location: m\n      .bngl\n    language: bngl\n',
         "the location of model 'm' continues on the line '.bngl'"),
        # A field indented differently from the fields before it (not valid YAML).
        ('model_files:\n  m:\n    location: m.bngl\n   language: bngl\n',
         "the line 'language: bngl' of model 'm' is not indented like the fields"),
    ])
    def test_unreadable_model_field_is_refused(self, tmp_path, model_block, match):
        base = self._BASE.split('model_files:')[0]
        (tmp_path / 'problem.yaml').write_text(base + model_block)
        with pytest.raises(PybnfError, match=match):
            read_problem_yaml(tmp_path / 'problem.yaml')

    def test_second_document_after_end_marker_is_refused(self, tmp_path):
        (tmp_path / 'problem.yaml').write_text(self._BASE + '...\n---\nid: second\n')
        with pytest.raises(PybnfError, match=r"more than one YAML document: the line '---'"):
            read_problem_yaml(tmp_path / 'problem.yaml')

    def test_nested_location_does_not_replace_the_model(self, tmp_path):
        # End to end: a variant model file sits beside the real one and a nested field names
        # it. libpetab imports bateman_chain.bngl; the import must copy and name that model.
        petab_v2 = pytest.importorskip('petab.v2')
        root = tmp_path / 'p'
        shutil.copytree(TUTORIAL_PETAB_DIR, root)
        (root / 'variant.bngl').write_text(
            (root / 'bateman_chain.bngl').read_text().replace('k2  0.25', 'k2  9.99'))
        yaml_path = root / 'problem.yaml'
        yaml_path.write_text(yaml_path.read_text().replace(
            '    language: bngl\n',
            '    language: bngl\n    provenance:\n      location: variant.bngl\n'))
        oracle = petab_v2.Problem.from_yaml(str(yaml_path))
        (model_file,) = oracle.config.model_files.values()
        assert str(model_file.location) == 'bateman_chain.bngl'
        out = import_job(yaml_path, tmp_path / 'out')
        conf = (out / 'imported.conf').read_text()
        assert 'model: bateman_chain.bngl' in conf and 'variant.bngl' not in conf
        assert sorted(f.name for f in out.glob('*.bngl')) == ['bateman_chain.bngl']


# ---------------------------------------------------------------------------
# A table split over several files (#902). PEtab v2 types every *_files key as a list, and
# libpetab reads a problem by chaining every listed file's rows in list order. The importer
# read only the first file of each list, so a split problem was fitted to part of its data.
# Oracles: libpetab reading the same split problem, and the unsplit problem's own import.
# ---------------------------------------------------------------------------

TUTORIAL_PETAB_DIR = (Path(__file__).resolve().parents[1] / 'examples' / 'tutorial'
                      / '12_petab_roundtrip' / 'petab')

_TABLE_KEYS = ('parameter_files', 'observable_files', 'measurement_files',
               'condition_files', 'experiment_files', 'mapping_files')


def _split_tutorial_problem(root, flow=False):
    """The #902 reproduction: tutorial 12's problem with its measurements split over two files
    (obs_Obs_A rows in the first, obs_Obs_B/C in the second) and its two estimated parameters
    one per file. ``flow`` writes the two lists in the one-line ``[a, b]`` form."""
    shutil.copytree(TUTORIAL_PETAB_DIR, root)
    head, *rows = (root / 'measurements.tsv').read_text().splitlines()
    (root / 'measurements.tsv').write_text(
        '\n'.join([head] + [r for r in rows if r.split('\t')[0] == 'obs_Obs_A']) + '\n')
    (root / 'measurements2.tsv').write_text(
        '\n'.join([head] + [r for r in rows if r.split('\t')[0] != 'obs_Obs_A']) + '\n')
    (root / 'parameters.tsv').write_text(
        'parameterId\testimate\tlowerBound\tupperBound\nk1\ttrue\t0.05\t3\n')
    (root / 'parameters2.tsv').write_text(
        'parameterId\testimate\tlowerBound\tupperBound\nk2\ttrue\t0.02\t2\n')
    if flow:
        lists = ('parameter_files: [parameters.tsv, parameters2.tsv]\n'
                 'measurement_files: [measurements.tsv, measurements2.tsv]\n')
    else:
        lists = ('parameter_files:\n  - parameters.tsv\n  - parameters2.tsv\n'
                 'measurement_files:\n  - measurements.tsv\n  - measurements2.tsv\n')
    (root / 'problem.yaml').write_text(
        'format_version: 2.0.0\n' + lists + 'observable_files:\n  - observables.tsv\n'
        'model_files:\n  bateman_chain:\n    location: bateman_chain.bngl\n    language: bngl\n')
    return root / 'problem.yaml'


def _split_every_table(src, dst):
    """Copy the PEtab problem at ``src`` to ``dst`` with EVERY table split over two files.

    Rows are divided by the id in the table's first column: the first half of the distinct ids
    (in order of appearance) go to ``<table>.tsv``, the rest to ``<table>_2.tsv`` (header only
    when the table has one id), row order kept. Every row of one id stays in one file, so the
    split problem is the same problem -- which libpetab confirms in the tests below."""
    shutil.copytree(src, dst)
    problem = read_problem_yaml(src / 'problem.yaml')
    lines = ['format_version: 2.0.0\n']
    for key in _TABLE_KEYS:
        if not problem[key]:
            continue
        (name,) = problem[key]
        head, *rows = (src / name).read_text().splitlines()
        ids = list(dict.fromkeys(r.split('\t')[0] for r in rows))
        first = set(ids[:(len(ids) + 1) // 2])
        second_name = name.replace('.tsv', '_2.tsv')
        (dst / name).write_text(
            '\n'.join([head] + [r for r in rows if r.split('\t')[0] in first]) + '\n')
        (dst / second_name).write_text(
            '\n'.join([head] + [r for r in rows if r.split('\t')[0] not in first]) + '\n')
        lines.append(f'{key}:\n  - {name}\n  - {second_name}\n')
    lines.append('model_files:\n')
    for m in problem['models']:
        lines.append(f"  {m['model_id']}:\n    location: {m['location']}\n"
                     f"    language: {m['language']}\n")
    (dst / 'problem.yaml').write_text(''.join(lines))
    return dst / 'problem.yaml'


def _imported_files(out):
    """``{name: text}`` of an imported job's conf, data and sidecar files."""
    return {f.name: f.read_text() for f in sorted(out.iterdir())
            if f.suffix in ('.conf', '.exp', '.tsv')}


class TestSplitTableFiles:

    @pytest.mark.parametrize('flow', [False, True], ids=['block_lists', 'flow_lists'])
    def test_split_problem_imports_what_libpetab_reads(self, tmp_path, flow):
        # The independent oracle: libpetab reads the split problem as 63 measurements and two
        # estimated parameters and finds it valid. On main the import kept only the 21 rows of
        # the first measurement file and declared only k1 (k2 silently held at its model
        # value); with the flow form it refused with "has no parameter_files".
        petab_v2 = pytest.importorskip('petab.v2')
        from petab.v2.lint import lint_problem
        yaml_path = _split_tutorial_problem(tmp_path / 'split', flow=flow)
        oracle = petab_v2.Problem.from_yaml(str(yaml_path))
        assert not lint_problem(oracle)
        out = import_job(yaml_path, tmp_path / 'imported')

        conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
        free = [k[1] for k in conf if isinstance(k, tuple) and k[0].endswith('_var')]
        assert free == list(oracle.x_free_ids) == ['k1', 'k2']

        # Every (observable column, time, value) libpetab reads is in the imported data, and
        # nothing else is.
        column_of = {o.id: str(o.formula) for o in oracle.observables}
        want = sorted((column_of[m.observable_id], float(m.time), float(m.measurement))
                      for m in oracle.measurements)
        data = Data(file_name=str(out / 'experiment1.exp'))
        got = sorted((col, float(t), float(v))
                     for col in data.cols if col != 'time'
                     for t, v in zip(data['time'], data[col]) if not np.isnan(v))
        assert len(want) == 63
        assert got == want

    def test_split_problem_imports_like_the_unsplit_one(self, tmp_path):
        # Splitting a table over two files changes nothing about the problem, so the import
        # must be byte-identical to the unsplit tutorial problem's. Dependency-free.
        whole = import_job(TUTORIAL_PETAB_DIR / 'problem.yaml', tmp_path / 'whole')
        split = import_job(_split_tutorial_problem(tmp_path / 'split'), tmp_path / 'split_out')
        assert _imported_files(split) == _imported_files(whole)

    def test_files_with_different_columns_import_like_one_file(self, tmp_path):
        # Review of #902. Each file of a split table has its own header: the second file may
        # order its columns differently and leave out an optional column that is blank
        # anyway. Here tutorial 20 (per-row observableParameters and noiseParameters) is
        # split so that the second measurement file drops the empty experimentId column and
        # reorders the rest, the second parameter file reorders its columns, and the second
        # measurement file also repeats a row of the first with another value (a replicate
        # across files). Oracles: libpetab reads the split problem as the same measurements
        # and free parameters as the one-file problem, and the import is byte-identical.
        petab_v2 = pytest.importorskip('petab.v2')
        src = (Path(__file__).resolve().parents[1] / 'examples' / 'tutorial'
               / '20_petab_observable_parameters')

        def table(path):
            head, *rows = path.read_text().splitlines()
            cols = head.split('\t')
            return cols, [dict(zip(cols, r.split('\t'))) for r in rows]

        def write(path, cols, rows):
            path.write_text('\n'.join(['\t'.join(cols)]
                                      + ['\t'.join(r[c] for c in cols) for r in rows]) + '\n')

        m_cols, m_rows = table(src / 'measurements.tsv')
        p_cols, p_rows = table(src / 'parameters.tsv')
        assert 'experimentId' in m_cols and not any(r['experimentId'] for r in m_rows)
        first = [r for r in m_rows if r['observableId'] == 'obs_B']
        replicate = dict(first[1], measurement='33.5')
        second = [r for r in m_rows if r['observableId'] != 'obs_B'] + [replicate]
        assert first and len(second) > 1

        split = tmp_path / 'split'
        shutil.copytree(src, split)
        write(split / 'measurements.tsv', m_cols, first)
        write(split / 'measurements2.tsv', ['measurement', 'noiseParameters', 'time',
                                            'observableParameters', 'observableId'], second)
        write(split / 'parameters.tsv', p_cols, p_rows[:3])
        write(split / 'parameters2.tsv', ['upperBound', 'parameterId', 'lowerBound', 'estimate'],
              p_rows[3:])
        (split / 'problem.yaml').write_text((src / 'problem.yaml').read_text().replace(
            '  - measurements.tsv\n', '  - measurements.tsv\n  - measurements2.tsv\n').replace(
            '  - parameters.tsv\n', '  - parameters.tsv\n  - parameters2.tsv\n'))
        # The same problem in one file per table: the rows in the same order.
        whole = tmp_path / 'whole'
        shutil.copytree(src, whole)
        write(whole / 'measurements.tsv', m_cols, first + second)

        def libpetab_view(yaml_path):
            problem = petab_v2.Problem.from_yaml(str(yaml_path))
            return (sorted((m.observable_id, float(m.time), float(m.measurement),
                            tuple(map(str, m.observable_parameters)),
                            tuple(map(str, m.noise_parameters)))
                           for m in problem.measurements),
                    list(problem.x_free_ids))

        split_view = libpetab_view(split / 'problem.yaml')
        assert split_view == libpetab_view(whole / 'problem.yaml')
        assert len(split_view[0]) == len(m_rows) + 1
        a = import_job(split / 'problem.yaml', tmp_path / 'split_out')
        b = import_job(whole / 'problem.yaml', tmp_path / 'whole_out')
        assert _imported_files(a) == _imported_files(b)
        assert 'experiment1_rep2.exp' in _imported_files(a)
        conf = ploop((a / 'imported.conf').read_text().splitlines(keepends=True))
        free = [k[1] for k in conf if isinstance(k, tuple) and k[0].endswith('_var')]
        assert sorted(free) == sorted(split_view[1])

    # Exported jobs that between them populate all six tables: conditions/experiments (a
    # surrogate-base condition with several target rows), the mapping table (a pre-equilibrated
    # dose response, whose experiments have several period rows), and two models.
    _JOBS = {
        'conditions': (
            'edition = 2\njob_type = de\nobjective = chi_sq\nmodel: parabola2.bngl\n'
            'condition: doubled, perturbations: v1 * 2\n'
            'condition: scaled, perturbations: s * 5\n'
            'experiment: wt, data: wt.exp\n'
            'experiment: dbl, condition: doubled, data: dbl.exp\n'
            'experiment: scl, condition: scaled, data: scl.exp\n' + _PARAMS_U,
            {'parabola2.bngl': _PARABOLA2_BNGL,
             'wt.exp': '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n',
             'dbl.exp': '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n',
             'scl.exp': '# time x y x_SD y_SD\n0\t-10\t430\t1\t1\n1\t-9\t345\t1\t1\n'},
            'parabola2.bngl'),
        'mapping_and_periods': (
            _PDR_CONF, {'m.bngl': _PDR_MODEL, 'dose.exp': _PDR_DOSE_EXP}, 'm.bngl'),
        'two_models': (
            TestImportMultiModelRoundTrip.CONF, TestImportMultiModelRoundTrip.EXTRA, DEMO_MODEL),
    }

    @pytest.mark.parametrize('job', sorted(_JOBS))
    def test_every_table_split_imports_like_the_unsplit_one(self, tmp_path, job):
        conf_text, extra, model_name = self._JOBS[job]
        petab1, whole, _petab2, _conf = _roundtrip(
            tmp_path, conf_text, extra_files=extra, model_name=model_name)
        split_yaml = _split_every_table(petab1, tmp_path / 'split')
        listed = read_problem_yaml(split_yaml)
        split_keys = [k for k in _TABLE_KEYS if listed[k]]
        assert len(split_keys) >= 3 and all(len(listed[k]) == 2 for k in split_keys)
        split = import_job(split_yaml, tmp_path / 'split_out')
        assert _imported_files(split) == _imported_files(whole)

    @pytest.mark.parametrize('job', sorted(_JOBS))
    def test_libpetab_reads_the_split_problem_as_the_unsplit_one(self, tmp_path, job):
        # The split helper is itself checked against the oracle: libpetab reads the split and
        # the unsplit problem as the same entities and the same measurements, so the identity
        # above compares two imports of one problem.
        petab_v2 = pytest.importorskip('petab.v2')
        conf_text, extra, model_name = self._JOBS[job]
        petab1, _whole, _petab2, _conf = _roundtrip(
            tmp_path, conf_text, extra_files=extra, model_name=model_name)
        split_yaml = _split_every_table(petab1, tmp_path / 'split')
        import re
        a = petab_v2.Problem.from_yaml(str(petab1 / 'problem.yaml'))
        b = petab_v2.Problem.from_yaml(str(split_yaml))

        def entities(problem, attr):
            # Each entity's full content, minus its row position inside its own file. A
            # condition's numeric targetValue is compared as a number: libpetab types a column
            # per file, so '2' reads as Integer(2) beside an expression and Float(2) without one.
            if attr == 'conditions':
                return sorted((c.id, ch.target_id,
                               float(ch.target_value) if ch.target_value.is_number
                               else str(ch.target_value))
                              for c in problem.conditions for ch in c.changes)
            return sorted(re.sub(r'index=\d+', '', repr(e)) for e in getattr(problem, attr))

        for attr in ('parameters', 'observables', 'measurements', 'conditions',
                     'experiments', 'mappings'):
            assert entities(a, attr) == entities(b, attr), attr
        assert len(b.measurements) == len(a.measurements) > 0

    # A repeated id: each is a duplicate libpetab's lint_problem reports as an error, and the
    # import used to either never see it (it sat in a later file) or let one row silently win.
    @pytest.mark.parametrize('table, edit, match', [
        ('parameters2.tsv', 'k1\ttrue\t0.05\t3\n',
         r"parameterId 'k1' more than once \(in both parameters.tsv and parameters2.tsv\)"),
        ('parameters.tsv', 'k1\ttrue\t0.05\t3\n',
         r"parameterId 'k1' more than once \(twice in parameters.tsv\)"),
    ])
    def test_repeated_parameter_is_refused(self, tmp_path, table, edit, match):
        yaml_path = _split_tutorial_problem(tmp_path / 'split')
        with open(yaml_path.parent / table, 'a') as fh:
            fh.write(edit)
        with pytest.raises(PybnfError, match=match):
            import_job(yaml_path, tmp_path / 'out')
        _assert_libpetab_reports_duplicate(yaml_path, 'Parameter table contains duplicate IDs')

    def test_observable_in_two_files_is_refused(self, tmp_path):
        yaml_path = _split_tutorial_problem(tmp_path / 'split')
        (yaml_path.parent / 'observables2.tsv').write_text(
            'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n'
            'obs_Obs_A\tObs_B\t1\tnormal\n')
        yaml_path.write_text(yaml_path.read_text().replace(
            '  - observables.tsv\n', '  - observables.tsv\n  - observables2.tsv\n'))
        with pytest.raises(PybnfError, match=r"observableId 'obs_Obs_A' more than once "
                                             r"\(in both observables.tsv and observables2.tsv\)"):
            import_job(yaml_path, tmp_path / 'out')
        _assert_libpetab_reports_duplicate(yaml_path, 'Observable table contains duplicate IDs')

    @pytest.mark.parametrize('table, row, match, lint', [
        # A condition's target rows split between two files: two definitions of one id.
        ('conditions', 'cond_incubate\tspecies_B\t1\n',
         r"conditionId 'cond_incubate' more than once \(in both conditions.tsv and "
         r"conditions_2.tsv\)", 'Condition table contains duplicate IDs'),
        ('experiments', 'scan_0\t1\tcond_scan_0\n',
         r"experimentId 'scan_0' more than once \(in both experiments.tsv and "
         r"experiments_2.tsv\)", 'Experiment table contains duplicate IDs'),
        ('mapping', 'species_A\tB()\n',
         r"petabEntityId 'species_A' more than once \(in both mapping.tsv and mapping_2.tsv\)",
         'Mapping table contains non-unique IDs'),
    ])
    def test_id_defined_in_two_files_is_refused(self, tmp_path, table, row, match, lint):
        petab1, _whole, _petab2, _conf = _roundtrip(
            tmp_path, _PDR_CONF, extra_files={'m.bngl': _PDR_MODEL,
                                                     'dose.exp': _PDR_DOSE_EXP},
            model_name='m.bngl')
        split_yaml = _split_every_table(petab1, tmp_path / 'split')
        with open(split_yaml.parent / f'{table}_2.tsv', 'a') as fh:
            fh.write(row)
        with pytest.raises(PybnfError, match=match):
            import_job(split_yaml, tmp_path / 'out')
        _assert_libpetab_reports_duplicate(split_yaml, lint)


def _assert_libpetab_reports_duplicate(yaml_path, message):
    """The oracle for a duplicate-id refusal: libpetab's lint reports the same problem as an
    error, so the importer refuses exactly what PEtab itself calls invalid."""
    petab_v2 = pytest.importorskip('petab.v2')
    from petab.v2.lint import CheckMappingTable, CheckUniquePrimaryKeys
    from petab.v2.lint import ValidationIssueSeverity
    problem = petab_v2.Problem.from_yaml(str(yaml_path))
    issues = [task.run(problem) for task in (CheckUniquePrimaryKeys(), CheckMappingTable())]
    errors = [str(i) for i in issues
              if i is not None and i.level == ValidationIssueSeverity.ERROR]
    assert any(message in e for e in errors), errors


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

    def test_fixed_parameters_that_agree_leave_the_sbml_untouched(self, tmp_path, capsys):
        # #907: Boehm fixes two parameters. `ratio` is a model parameter whose table value
        # (0.693) is the SBML's own, and `specC17` is not a model entity at all (it is inlined
        # into a formula). Neither edits the model, so no override is printed or listed.
        pytest.importorskip('petab')
        out = import_job(self.YAML, tmp_path / 'out')
        assert ((out / 'model_Boehm_JProteomeRes2014.xml').read_bytes()
                == (BOEHM_DIR / 'model_Boehm_JProteomeRes2014.xml').read_bytes())
        assert 'Fixed model parameters' not in (out / 'imported.conf').read_text()
        assert 'PEtab import' not in capsys.readouterr().out

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


# ---------------------------------------------------------------------------
# A fixed (estimate=false) MODEL parameter takes the table's nominalValue (#907, ADR-0149)
#
# The issue's reproduction is fixedsigma_v2 with v3 fixed at 10 in parameters.tsv while the
# model file says `v3 3`, and measurements exact for (v1, v2, v3) = (0.5, 1, 10). PEtab gives
# the table precedence, so the imported copy of the model must carry v3 = 10. Before the fix
# the importer skipped the row and the job simulated v3 = 3: a check job reported 18.375
# instead of 0, and a fit bent v1/v2 to make up the constant.
# ---------------------------------------------------------------------------

_FIXED_V3_PARAMETERS = (
    'parameterId\tparameterName\tlowerBound\tupperBound\tnominalValue\testimate\n'
    'v1\tv1\t0\t10\t0.5\ttrue\n'
    'v2\tv2\t0\t10\t1\ttrue\n'
    '{v3_row}\n'
    'sd_c\tfixed noise\t\t\t2\tfalse\n')
_X = np.array([-10., -9., -8.])            # the counter x at t = 0, 1, 2
_EXACT_AT_V3_10 = (50., 41.5, 34.)         # y = 0.5 x^2 + x + 10
_EXACT_AT_V3_3 = (43., 34.5, 27.)          # the fixture's own data: y = 0.5 x^2 + x + 3


def _fixed_v3_problem(tmp_path, v3='10', model_text=None, data=_EXACT_AT_V3_10,
                      v3_row=None):
    """fixedsigma_v2 with v3 fixed at ``v3`` in the table (the model file says 3 unless
    ``model_text`` replaces it) and measurements ``data``; returns the problem.yaml path."""
    prob = tmp_path / 'prob'
    shutil.copytree(FIXEDSIGMA_DIR, prob)
    row = v3_row if v3_row is not None else f'v3\tv3\t\t\t{v3}\tfalse'
    (prob / 'parameters.tsv').write_text(_FIXED_V3_PARAMETERS.format(v3_row=row))
    (prob / 'measurements.tsv').write_text(
        'observableId\texperimentId\ttime\tmeasurement\tobservableParameters\tnoiseParameters\n'
        + ''.join(f'obs_y\tepo\t{t}\t{y}\t\tsd_c\n' for t, y in enumerate(data)))
    if model_text is not None:
        (prob / 'fixedsigma_model.bngl').write_text(model_text)
    return prob / 'problem.yaml'


def _hand_objective(v1, v2, v3, data, sigma=2.):
    """The fixed-sigma Gaussian objective PyBNF reports, computed by hand (no PyBNF):
    sum(res^2) / (2 sigma^2) over y = v1 x^2 + v2 x + v3."""
    y = v1 * _X ** 2 + v2 * _X + v3
    return float(np.sum((np.asarray(data) - y) ** 2) / (2 * sigma ** 2))


def _bng_objective(job_dir, values, monkeypatch, conf='imported.conf'):
    """Simulate the job's single BNGL model at the free-parameter ``values`` with BNG2.pl and
    score it with the job's own objective: what a ``job_type = check`` run prints, at any
    parameter vector."""
    from pybnf.config import Configuration
    from pybnf.pset import PSet
    monkeypatch.chdir(job_dir)
    cfg = Configuration(ploop((job_dir / conf).read_text().splitlines(keepends=True)))
    (name, model), = cfg.models.items()
    ps = PSet([v.set_value(values[v.name]) for v in cfg.variables])
    sim_dir = job_dir / 'sim'
    sim_dir.mkdir(exist_ok=True)
    ds = model.copy_with_param_set(ps).execute(str(sim_dir), 'probe', 60)
    return cfg.obj.evaluate_multiple({name: ds}, cfg.exp_data, ps)


class TestFixedModelParameterImport:
    """#907: an estimate=false row naming a BNGL model parameter is written into the imported
    model copy, marked, listed in the conf header and printed; a model that already agrees is
    carried byte-for-byte."""

    MARKED = '    v3 10  # PEtab parameters.tsv: estimate=false, nominalValue 10 (model file: 3)'

    def test_the_table_value_is_written_into_the_model_copy_and_marked(self, tmp_path, capsys):
        out = import_job(_fixed_v3_problem(tmp_path), tmp_path / 'out')
        source = (FIXEDSIGMA_DIR / 'fixedsigma_model.bngl').read_text().splitlines()
        copy = (out / 'fixedsigma_model.bngl').read_text().splitlines()
        # Exactly one line differs: the v3 line, rewritten to the table's value and annotated.
        assert len(copy) == len(source)
        assert [(a, b) for a, b in zip(source, copy) if a != b] == [('    v3 3', self.MARKED)]
        assert parse_model('\n'.join(copy)).parameters['v3'] == '10'
        # The source problem is never touched.
        assert (tmp_path / 'prob' / 'fixedsigma_model.bngl').read_text() == '\n'.join(source) + '\n'
        # The conf names the override in its header and declares no v3 line of its own.
        text = (out / 'imported.conf').read_text()
        assert '#   v3 = 10 in fixedsigma_model.bngl (the model file had 3)' in text
        assert not any(ln.split('#')[0].strip().endswith(('v3 10', 'v3'))
                       for ln in text.splitlines() if not ln.startswith('#'))
        # And the import says so on the console, one line per override.
        printed = [ln for ln in capsys.readouterr().out.splitlines() if 'PEtab import' in ln]
        assert printed == ['PEtab import: parameters.tsv fixes v3 = 10 (estimate=false); '
                           'fixedsigma_model.bngl has 3, so the imported copy uses 10.']

    def test_libpetab_fixed_values_are_the_values_in_the_model_copy(self, tmp_path):
        # The external oracle: libpetab reads the problem and reports each fixed parameter's
        # nominal value and which ids are model parameters; the imported copy, read back both
        # with PyBNF's reader and with a plain regex, must carry exactly those values.
        pytest.importorskip('petab.v2')
        import re

        from petab.v2 import Problem
        yaml = _fixed_v3_problem(tmp_path)
        problem = Problem.from_yaml(str(yaml))
        fixed = problem.get_x_nominal_dict(free=False)
        assert fixed == {'v3': 10.0, 'sd_c': 2.0}
        model_params = set(problem.model.get_valid_parameters_for_parameter_table())
        out = import_job(yaml, tmp_path / 'out')
        text = (out / 'fixedsigma_model.bngl').read_text()
        ours = parse_model(text).parameters
        for pid in fixed.keys() & model_params:
            assert float(ours[pid]) == fixed[pid]
            m = re.search(rf'^\s*{pid}\s+([^\s#]+)', text, re.M)
            assert float(m.group(1)) == fixed[pid]
        assert fixed.keys() & model_params == {'v3'}

    def test_an_agreeing_model_is_carried_byte_for_byte(self, tmp_path, capsys):
        # 3.0 in the table, 3 in the model: equal as numbers, so no edit, no header, no print.
        out = import_job(_fixed_v3_problem(tmp_path, v3='3.0', data=_EXACT_AT_V3_3),
                         tmp_path / 'out')
        assert ((out / 'fixedsigma_model.bngl').read_bytes()
                == (FIXEDSIGMA_DIR / 'fixedsigma_model.bngl').read_bytes())
        assert 'Fixed model parameters' not in (out / 'imported.conf').read_text()
        assert 'PEtab import' not in capsys.readouterr().out

    def test_an_expression_right_hand_side_is_replaced_by_the_constant(self, tmp_path):
        # PEtab fixes the parameter at a constant, so an expression is replaced even when it
        # evaluates to the table's value: it would follow v1 if v1 moved.
        model = (FIXEDSIGMA_DIR / 'fixedsigma_model.bngl').read_text().replace(
            '    v3 3\n', '    v3 = 6*v1\n')
        out = import_job(_fixed_v3_problem(tmp_path, v3='3', model_text=model,
                                           data=_EXACT_AT_V3_3), tmp_path / 'out')
        copy = (out / 'fixedsigma_model.bngl').read_text()
        assert ('    v3 = 3  # PEtab parameters.tsv: estimate=false, nominalValue 3 '
                '(model file: 6*v1)\n') in copy
        assert copy.replace(
            '    v3 = 3  # PEtab parameters.tsv: estimate=false, nominalValue 3 '
            '(model file: 6*v1)\n', '    v3 = 6*v1\n') == model

    def test_a_multi_model_problem_is_edited_in_every_model_that_declares_the_parameter(
            self, tmp_path):
        # The parameters table is global: v3 fixed at 7 applies to both models that declare
        # v3 (parabola says 3, the second model says 5), each copy edited and marked.
        second = _GROWTH_BNGL.replace('    a2 2\n', '    a2 2\n    v3 5\n')
        conf = ('edition = 2\njob_type = de\nobjective = chi_sq\n'
                f'model: {DEMO_MODEL}\nmodel: growth_v2.bngl\n'
                f'experiment: pa, model: {DEMO_MODEL}, data: pa.exp\n'
                'experiment: gr, model: growth_v2.bngl, data: gr.exp\n'
                'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = a1 0 10\n')
        src = tmp_path / 'src'
        src.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        (src / 'growth_v2.bngl').write_text(second)
        (src / 'pa.exp').write_text((DEMO_DIR / 'par1.exp').read_text())
        (src / 'gr.exp').write_text(TestImportMultiModelRoundTrip.EXTRA['gr.exp'])
        (src / 'job.conf').write_text(conf)
        petab1 = export_job(src / 'job.conf', tmp_path / 'petab1')
        assert [r['parameterId'] for r in _tsv_rows(petab1 / 'parameters.tsv')] == [
            'v1', 'v2', 'a1']
        (petab1 / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\tnominalValue\n'
            'v1\ttrue\t0\t10\t\nv2\ttrue\t0\t10\t\na1\ttrue\t0\t10\t\n'
            'v3\tfalse\t\t\t7\n')
        out = import_job(petab1 / 'problem.yaml', tmp_path / 'out')
        for name, old in ((DEMO_MODEL, '3'), ('growth_v2.bngl', '5')):
            source = (petab1 / name).read_text().splitlines()
            copy = (out / name).read_text().splitlines()
            diff = [(a, b) for a, b in zip(source, copy) if a != b]
            assert diff == [(f'    v3 {old}',
                             f'    v3 7  # PEtab parameters.tsv: estimate=false, nominalValue 7 '
                             f'(model file: {old})')], name
        text = (out / 'imported.conf').read_text()
        assert f'#   v3 = 7 in {DEMO_MODEL} (the model file had 3)' in text
        assert '#   v3 = 7 in growth_v2.bngl (the model file had 5)' in text

    def test_a_fixed_row_without_a_nominal_value_is_refused(self, tmp_path):
        yaml = _fixed_v3_problem(tmp_path, v3_row='v3\tv3\t\t\t\tfalse')
        with pytest.raises(PybnfError, match="'v3' has estimate=false but no nominalValue"):
            import_job(yaml, tmp_path / 'out')

    def test_a_fixed_row_naming_a_bngl_observable_is_refused(self, tmp_path):
        # `x` is the model's observable; PEtab's BNGL loader admits only parameters to the
        # parameters table, so the row is malformed rather than silently ignored.
        yaml = _fixed_v3_problem(tmp_path, v3_row='v3\tv3\t\t\t10\tfalse\nx\tx\t\t\t1\tfalse')
        with pytest.raises(PybnfError, match="'x' is an observable, not a parameter"):
            import_job(yaml, tmp_path / 'out')

    @pytest.mark.parametrize('action', [
        'setParameter("v3", 3)',
        "parameter_scan({parameter=>'v3', par_min=>1, par_max=>2, n_scan_pts=>2})",
    ])
    def test_a_parameter_the_model_actions_set_is_refused(self, tmp_path, action):
        # BNG2.pl runs the model file's actions after it loads the model, so an action that
        # sets v3 would undo the value written into `begin parameters`.
        model = ((FIXEDSIGMA_DIR / 'fixedsigma_model.bngl').read_text()
                 + f'\nbegin actions\n  {action}\nend actions\n')
        yaml = _fixed_v3_problem(tmp_path, model_text=model)
        with pytest.raises(NotImplementedError,
                           match="'v3' has estimate=false with nominalValue 10, but an action "
                                 "in the model fixedsigma_model.bngl"):
            import_job(yaml, tmp_path / 'out')

    def test_a_non_finite_nominal_value_on_a_model_parameter_is_refused(self, tmp_path):
        yaml = _fixed_v3_problem(tmp_path, v3='inf')
        with pytest.raises(PybnfError, match="'v3' has estimate=false with nominalValue inf, "
                                             "which is not a finite number"):
            import_job(yaml, tmp_path / 'out')

    @pytest.mark.bionetgen
    @pytest.mark.parametrize('data, expected', [
        (_EXACT_AT_V3_10, 0.0),                  # the issue: main reported 18.375
        (_EXACT_AT_V3_3, 3 * 7. ** 2 / (2 * 2. ** 2)),   # each point 7 above the data
    ])
    def test_a_check_evaluation_scores_the_table_value(self, tmp_path, monkeypatch, data,
                                                       expected):
        # What `job_type = check` prints: the model file's own v1/v2 with the table's v3.
        out = import_job(_fixed_v3_problem(tmp_path, data=data), tmp_path / 'out')
        assert expected == _hand_objective(0.5, 1., 10., data)
        assert _bng_objective(out, {'v1': 0.5, 'v2': 1.}, monkeypatch) == pytest.approx(
            expected, abs=1e-9)

    @pytest.mark.bionetgen
    def test_the_objective_is_minimized_where_bounded_least_squares_puts_it(
            self, tmp_path, monkeypatch):
        # An independent solver: with v3 fixed at 10 the model is linear in (v1, v2), so
        # scipy's bounded linear least squares gives the PEtab problem's best fit. PyBNF's
        # imported job must score 0 there and the hand formula anywhere else.
        from scipy.optimize import lsq_linear
        data = np.asarray(_EXACT_AT_V3_10)
        sol = lsq_linear(np.column_stack([_X ** 2, _X]), data - 10., bounds=(0, 10)).x
        np.testing.assert_allclose(sol, [0.5, 1.0], atol=1e-8)
        out = import_job(_fixed_v3_problem(tmp_path), tmp_path / 'out')
        at_sol = _bng_objective(out, {'v1': sol[0], 'v2': sol[1]}, monkeypatch)
        assert at_sol == pytest.approx(0., abs=1e-9)
        # The point the pre-fix importer's fit found (v3 = 3 in the model) is ~18 away.
        for v1, v2 in ((0.4, 0.8), (0.47479, 0.00064)):
            assert _bng_objective(out, {'v1': v1, 'v2': v2}, monkeypatch) == pytest.approx(
                _hand_objective(v1, v2, 10., data), rel=1e-9)

    @pytest.mark.bionetgen
    @pytest.mark.bngsim
    def test_the_model_a_fit_simulates_scores_the_table_value(self, tmp_path, monkeypatch):
        # A fit does not simulate the config's BNGLModel through BNG2.pl: the algorithm turns
        # it into bngsim's network model, generated by BNG2.pl from the model copy. That
        # sibling path must see v3 = 10 as well.
        from pybnf import algorithms
        from pybnf.config import Configuration
        from pybnf.pset import PSet
        out = import_job(_fixed_v3_problem(tmp_path), tmp_path / 'out')
        monkeypatch.chdir(out)
        cfg = Configuration(ploop((out / 'imported.conf').read_text().splitlines(keepends=True)))
        (out / cfg.config['output_dir']).mkdir(parents=True)
        alg = algorithms.DifferentialEvolution(cfg)
        monkeypatch.chdir(out)                # the constructor moves into output_dir
        (model,) = alg.model_list
        assert type(model).__name__ == 'BngsimModel'
        ps = PSet([v.set_value({'v1': 0.4, 'v2': 0.8}[v.name]) for v in cfg.variables])
        (out / 'net_sim').mkdir()
        ds = model.copy_with_param_set(ps).execute(str(out / 'net_sim'), 'probe', 60)
        assert alg.objective.evaluate_multiple({model.name: ds}, alg.exp_data, ps) == \
            pytest.approx(_hand_objective(0.4, 0.8, 10., _EXACT_AT_V3_10), rel=1e-9)

    def test_every_conf_of_an_all_job_types_import_lists_the_override(self, tmp_path):
        out = import_job(_fixed_v3_problem(tmp_path), tmp_path / 'out', job_type='all')
        confs = sorted(out.glob('imported_*.conf'))
        assert len(confs) > 1
        for conf in confs:
            assert ('#   v3 = 10 in fixedsigma_model.bngl (the model file had 3)'
                    in conf.read_text()), conf.name

    @pytest.mark.bionetgen
    def test_export_then_import_is_the_same_problem(self, tmp_path, monkeypatch):
        # Re-exporting the imported job writes the edited model and NO estimate=false row
        # (the exporter never writes one: a parameter absent from the table takes the model
        # file's value, which is now the table's). Re-importing that is the same problem:
        # nothing left to override, the model carried byte-for-byte, the same objective.
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import lint_problem
        imp1 = import_job(_fixed_v3_problem(tmp_path), tmp_path / 'imp1')
        petab2 = export_job(imp1 / 'imported.conf', tmp_path / 'petab2')
        assert [r['parameterId'] for r in _tsv_rows(petab2 / 'parameters.tsv')] == ['v1', 'v2']
        assert self.MARKED in (petab2 / 'fixedsigma_model.bngl').read_text().splitlines()
        assert not lint_problem(Problem.from_yaml(str(petab2 / 'problem.yaml'))).has_errors()
        imp2 = import_job(petab2 / 'problem.yaml', tmp_path / 'imp2')
        assert ((imp2 / 'fixedsigma_model.bngl').read_bytes()
                == (petab2 / 'fixedsigma_model.bngl').read_bytes())
        assert 'Fixed model parameters' not in (imp2 / 'imported.conf').read_text()
        point = {'v1': 0.4, 'v2': 0.8}
        first = _bng_objective(imp1, point, monkeypatch)
        assert first == pytest.approx(_hand_objective(0.4, 0.8, 10., _EXACT_AT_V3_10), rel=1e-9)
        assert _bng_objective(imp2, point, monkeypatch) == first


# Spellings of an action that sets v3 which BNG2.pl 2.9.3 executes (its actions reader is
# `^\s*(\w+)\s*\((.*)\);?\s*$`, and it evaluates the options as a Perl hash, where a key may
# be quoted), and which the first version of the #907 gate did not recognise.
_V3_ACTION_SPELLINGS = [
    'setParameter ("v3", 7)',
    'setParameter\t("v3", 7)',
    'parameter_scan({"parameter"=>"v3", par_min=>6, par_max=>7, n_scan_pts=>2, '
    'method=>"ode", t_end=>1, n_steps=>1})',
]


class TestFixedModelParameterImportEdges:
    """#907, found in review: the ways the override could still fail to take effect, or
    reach a file it must not touch."""

    @pytest.mark.parametrize('action', _V3_ACTION_SPELLINGS)
    def test_every_spelling_of_an_action_that_sets_it_is_refused(self, tmp_path, action):
        # Let through, the action runs after the model is read and undoes the edit: the job
        # then reports that the copy uses v3 = 10 while it simulates 7 (with `setParameter
        # ("v3", 3)` a check job scored 18.375 instead of 0).
        model = ((FIXEDSIGMA_DIR / 'fixedsigma_model.bngl').read_text()
                 + f'\nbegin actions\n  {action}\nend actions\n')
        yaml = _fixed_v3_problem(tmp_path, model_text=model)
        with pytest.raises(NotImplementedError,
                           match="'v3' has estimate=false with nominalValue 10, but an action "
                                 "in the model fixedsigma_model.bngl"):
            import_job(yaml, tmp_path / 'out')
        assert not (tmp_path / 'out' / 'fixedsigma_model.bngl').exists()

    @pytest.mark.bionetgen
    @pytest.mark.parametrize('action', _V3_ACTION_SPELLINGS)
    def test_bng2_runs_each_of_those_spellings(self, tmp_path, action):
        # The oracle for the refusal above: BNG2.pl itself executes each spelling, and
        # writeModel afterwards records v3 = 7, not the model file's 3.
        import re
        import subprocess
        bng2 = shutil.which('BNG2.pl')
        model = ((FIXEDSIGMA_DIR / 'fixedsigma_model.bngl').read_text()
                 + '\nbegin actions\n  generate_network({overwrite=>1})\n'
                 + f'  {action}\n  writeModel({{prefix=>"after"}})\nend actions\n')
        (tmp_path / 'm.bngl').write_text(model)
        subprocess.run([bng2, 'm.bngl'], cwd=tmp_path, check=True, capture_output=True)
        written = (tmp_path / 'after.bngl').read_text()
        assert float(re.search(r'^\s*v3\s+(\S+)', written, re.M).group(1)) == 7.0

    @pytest.mark.parametrize('layout', ['in-place', 'sibling-models-dir'])
    def test_a_copy_that_would_overwrite_the_source_model_is_refused(self, tmp_path, layout):
        # A problem may name its model by a relative path that leaves the problem directory,
        # and the copy is written at the same relative path under out_dir. When that lands on
        # the source file itself, writing the edited copy would change the user's model
        # (PEtab says a parameter absent from some other parameters table takes the model
        # file's value, so a later import of a sibling problem would silently use 10).
        yaml = _fixed_v3_problem(tmp_path)
        prob = yaml.parent
        if layout == 'in-place':
            source, out = prob / 'fixedsigma_model.bngl', prob
        else:
            (tmp_path / 'models').mkdir()
            source = tmp_path / 'models' / 'fixedsigma_model.bngl'
            shutil.move(prob / 'fixedsigma_model.bngl', source)
            yaml.write_text(yaml.read_text().replace(
                'location: fixedsigma_model.bngl',
                'location: ../models/fixedsigma_model.bngl'))
            out = tmp_path / 'out'
        before = source.read_bytes()
        with pytest.raises(PybnfError, match='fixedsigma_model.bngl.*is the source model file'):
            import_job(yaml, out)
        assert source.read_bytes() == before

    def test_an_in_place_import_that_edits_nothing_is_still_accepted(self, tmp_path):
        # The refusal above is only for a copy that differs from its source: a model that
        # already agrees with the table imports in place as it did before #907.
        yaml = _fixed_v3_problem(tmp_path, v3='3', data=_EXACT_AT_V3_3)
        source = yaml.parent / 'fixedsigma_model.bngl'
        before = source.read_bytes()
        import_job(yaml, yaml.parent)
        assert source.read_bytes() == before
        assert (yaml.parent / 'imported.conf').exists()

    @pytest.mark.bionetgen
    def test_a_fixed_surrogate_base_value_is_simulated(self, tmp_path, monkeypatch):
        # A PyBNF job that fits v3 and sets it to 20 in one condition exports v3 as the
        # surrogate v3__REF, pinned in every other experiment by cond_wildtype (v3 = v3__REF).
        # Fixing v3__REF at 10 in parameters.tsv is the PEtab edit #907 is about; libpetab
        # simulates experiment `a` at v3 = 10. Data exact for (0.5, 1, 10) and (0.5, 1, 20).
        src = tmp_path / 'src'
        src.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        for name, v3 in (('a', 10.), ('b', 20.)):
            y = 0.5 * _X ** 2 + _X + v3
            (src / f'{name}.exp').write_text(
                '# time\ty\ty_SD\n' + ''.join(f'{t}\t{float(v)!r}\t1\n' for t, v in enumerate(y)))
        (src / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            f'model: {DEMO_MODEL}\ncondition: hi, perturbations: v3 = 20\n'
            'experiment: a, data: a.exp\nexperiment: b, condition: hi, data: b.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 100\n')
        petab1 = export_job(src / 'job.conf', tmp_path / 'petab1')
        assert [r['parameterId'] for r in _tsv_rows(petab1 / 'parameters.tsv')] == [
            'v1', 'v2', 'v3__REF']
        (petab1 / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\tnominalValue\n'
            'v1\ttrue\t0\t10\t\nv2\ttrue\t0\t10\t\nv3__REF\tfalse\t\t\t10\n')
        out = import_job(petab1 / 'problem.yaml', tmp_path / 'out')
        # chi_sq at the exact point is 0; with v3 = 3 in experiment a it is 3 * 7^2 / 2.
        assert _bng_objective(out, {'v1': 0.5, 'v2': 1.}, monkeypatch) == pytest.approx(
            0., abs=1e-9)


class TestBnglSetParameterValues:
    """The line editor behind #907 (``_bngl.set_parameter_values``): every parameter-line
    shape the reader accepts, only the edited lines change, line endings kept."""

    NOTE = staticmethod(lambda name, old: f'NOTE {name} was {old}')

    def test_each_line_shape_is_rewritten_in_place(self):
        from pybnf.petab._bngl import set_parameter_values
        text = ('begin model\r\n'
                'begin parameters\r\n'
                '    L 1\r\n'
                '\tM = 2 # ligand dose\n'
                '  1 N\t3\n'
                '  lab: P 2*L\n'
                '    Q 4  \\\n'
                '      + 1 # tail\n'
                '    R 5.0\n'
                '    S 6\n'
                'end parameters\n'
                'end model')
        new, changed = set_parameter_values(
            text, {'L': 10, 'M': 2.5, 'N': 30, 'P': 2, 'Q': 5, 'R': 5}, self.NOTE)
        assert changed == {'L': '1', 'M': '2', 'N': '3', 'P': '2*L', 'Q': '4        + 1'}
        assert new == ('begin model\r\n'
                       'begin parameters\r\n'
                       '    L 10  # NOTE L was 1\r\n'
                       '\tM = 2.5  # NOTE M was 2  # ligand dose\n'
                       '  1 N\t30  # NOTE N was 3\n'
                       '  lab: P 2  # NOTE P was 2*L\n'
                       '    Q 5  # NOTE Q was 4        + 1  # tail\n'
                       '    R 5.0\n'           # equal as a number: untouched
                       '    S 6\n'             # not named: untouched
                       'end parameters\n'
                       'end model')
        values = parse_model(new).parameters
        assert {k: float(values[k]) for k in 'LMNPQRS'} == {
            'L': 10, 'M': 2.5, 'N': 30, 'P': 2, 'Q': 5, 'R': 5, 'S': 6}

    def test_the_written_value_reads_back_as_the_same_float(self):
        from pybnf.petab._bngl import set_parameter_values
        v = 0.1 + 0.2
        new, _ = set_parameter_values('begin parameters\n k 1\nend parameters\n', {'k': v},
                                      self.NOTE)
        assert float(parse_model(new).parameters['k']) == v

    def test_nothing_to_change_returns_the_text_itself(self):
        from pybnf.petab._bngl import set_parameter_values
        text = 'begin parameters\n k 1e0\nend parameters\n'
        assert set_parameter_values(text, {'k': 1.0, 'absent': 2.0}, self.NOTE) == (text, {})


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
