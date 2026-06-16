"""Unit tests for the PEtab v2 *exporter* (#407, exporter-first; ADR-0025).

The exporter reads a working PyBNF/BNGL job and serializes it to a PEtab v2 problem.
Its contracts, by strength of oracle:

1. **The external oracle, at model level.** petab's own validation, run on the whole
   problem loaded via ``Problem.from_yaml`` after ``register_bngl()`` installs the
   ``BnglModel`` loader (ADR-0026). The exported ``demo`` problem must pass **every**
   ``default_validation_task`` -- including the model-cross checks (``CheckModel`` et
   al.) ADR-0025 had to exclude -- with no errors.
2. **The measurement pivot is exact.** Every long measurement cell equals the source
   wide ``.exp`` ``Data`` cell -- the wide<->long round trip.
3. **Reverse-asset round trip.** A uniform ``PetabParameterRow`` -> ``FreeParameter``
   (importer) -> ``PetabParameterRow`` (exporter) is the identity: the two-adapter
   proof in the export direction.
4. **Model correspondence** (what the table oracle can't check for BNGL): every
   ``observableFormula`` is a model observable/function name; every ``parameterId`` is
   a model parameter; the PEtab-clean model drops ``__FREE`` and ``begin actions``.
5. **The documented chunk-1 boundaries** raise (dose-response, non-uniform prior,
   no-``_SD`` noise, SBML model, non-``chi_sq`` objective).
"""

from pathlib import Path

import numpy as np
import pytest

from pybnf.data import Data
from pybnf.petab.conditions import (
    build_dose_response_conditions,
    build_mutant_conditions,
    mutation_target_value,
    surrogate_name,
)
from pybnf.petab.export import (
    clean_model_for_petab,
    export_job,
)
from pybnf.petab.measurements import (
    dose_response_measurement_rows,
    measurement_rows_from_data,
)
from pybnf.petab.observables import petab_observable_row
from pybnf.petab.parameters import (
    PetabParameterRow,
    free_parameter_from_row,
    petab_parameter_row,
)
from pybnf.pset import FreeParameter

DEMO_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'demo'
DEMO_CONF = DEMO_DIR / 'demo_bng.conf'


def _tsv_rows(path):
    """Read a TSV into a list of dict rows (a tiny stdlib reader for assertions)."""
    import csv
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


# ---------------------------------------------------------------------------
# 3. Reverse-asset round trip + unit behavior
# ---------------------------------------------------------------------------

class TestReverseAssets:

    @pytest.mark.parametrize('lb,ub', [(0.0, 10.0), (-5.0, 5.0), (1.0, 2.5)])
    def test_uniform_parameter_round_trips(self, lb, ub):
        # PetabParameterRow -> FreeParameter (import) -> PetabParameterRow (export)
        # is the identity on the bounded-uniform subset: the two-adapter proof.
        row = PetabParameterRow(parameter_id='k', estimate=True, lower_bound=lb,
                                upper_bound=ub, prior_distribution=None)
        fp = free_parameter_from_row(row)
        assert petab_parameter_row(fp) == row

    def test_parameter_id_strips_free_marker(self):
        fp = FreeParameter('k1__FREE', 'uniform_var', 1.0, 2.0)
        assert petab_parameter_row(fp).parameter_id == 'k1'

    def test_explicit_parameter_id_overrides_strip(self):
        fp = FreeParameter('kase__FREE', 'uniform_var', 1.0, 2.0)
        assert petab_parameter_row(fp, parameter_id='kase').parameter_id == 'kase'

    def test_non_uniform_prior_export_not_implemented(self):
        fp = FreeParameter('k__FREE', 'normal_var', 0.0, 1.0)
        with pytest.raises(NotImplementedError):
            petab_parameter_row(fp)

    @pytest.mark.parametrize('kind,prefix', [('observable', 'obs_'), ('function', 'func_')])
    def test_observable_row_prefix_formula_and_placeholder(self, kind, prefix):
        row = petab_observable_row('z', kind, 'normal', sd_from_data=True)
        assert row.observable_id == f'{prefix}z'
        assert row.observable_formula == 'z'          # bare model name, never a body
        assert row.noise_distribution == 'normal'
        # _SD noise -> a declared placeholder bound by the measurements' noiseParameters
        assert row.noise_formula == f'noiseParameter1_{prefix}z'
        assert row.noise_placeholders == row.noise_formula

    def test_observable_without_sd_not_implemented(self):
        with pytest.raises(NotImplementedError):
            petab_observable_row('z', 'observable', 'normal', sd_from_data=False)

    def test_measurement_pivot_values_and_noise(self):
        arr = np.array([[0.0, 1.0, 0.5], [1.0, 2.0, 0.7]])
        data = Data.from_columns(arr, ['time', 'a', 'a_SD'])
        rows = measurement_rows_from_data(data, {'a': 'obs_a'}, experiment_id='')
        assert [(r.time, r.measurement, r.noise_parameters) for r in rows] == [
            (0.0, 1.0, 0.5), (1.0, 2.0, 0.7)]
        assert {r.observable_id for r in rows} == {'obs_a'}
        assert all(r.experiment_id == '' for r in rows)

    def test_measurement_skips_nan_cells(self):
        arr = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0]])
        data = Data.from_columns(arr, ['time', 'a'])
        rows = measurement_rows_from_data(data, {'a': 'obs_a'})
        assert [r.time for r in rows] == [0.0, 2.0]   # the NaN row is dropped

    def test_dose_response_exp_not_implemented(self):
        # Independent variable is a swept parameter, not time -> deferred chunk.
        arr = np.array([[1.0, 12.0], [100.0, 88.0]])
        data = Data.from_columns(arr, ['L_total', 'y'])
        with pytest.raises(NotImplementedError):
            measurement_rows_from_data(data, {'y': 'obs_y'})


# ---------------------------------------------------------------------------
# 1/2/4. The demo job exported end-to-end
# ---------------------------------------------------------------------------

class TestExportDemo:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        out = tmp_path_factory.mktemp('petab_demo')
        export_job(DEMO_CONF, out)
        return out

    def test_writes_all_artifacts(self, exported):
        for name in ('parameters.tsv', 'observables.tsv', 'measurements.tsv',
                     'problem.yaml', 'parabola.bngl'):
            assert (exported / name).is_file()

    def test_parameters(self, exported):
        rows = _tsv_rows(exported / 'parameters.tsv')
        assert {r['parameterId'] for r in rows} == {'v1', 'v2', 'v3'}
        assert all(r['estimate'] == 'true' for r in rows)
        assert all((r['lowerBound'], r['upperBound']) == ('0', '10') for r in rows)

    def test_observables_map_to_model_entities(self, exported):
        rows = _tsv_rows(exported / 'observables.tsv')
        by_id = {r['observableId']: r for r in rows}
        assert set(by_id) == {'obs_x', 'func_y'}
        # observableFormula is the bare model name (x is an observable, y a function).
        assert by_id['obs_x']['observableFormula'] == 'x'
        assert by_id['func_y']['observableFormula'] == 'y'
        assert all(r['noiseDistribution'] == 'normal' for r in rows)
        assert all(r['noisePlaceholders'] == r['noiseFormula'] for r in rows)

    def test_measurement_values_match_exp_exactly(self, exported):
        # The strong oracle: every long cell equals the wide .exp Data cell.
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        col_of = {'obs_x': 'x', 'func_y': 'y'}
        rows = _tsv_rows(exported / 'measurements.tsv')
        assert len(rows) == data.data.shape[0] * 2     # 21 times x {x, y}
        for r in rows:
            i = int(round(float(r['time'])))            # time == row index here
            col = col_of[r['observableId']]
            assert float(r['measurement']) == data.data[i, data.cols[col]]
            assert float(r['noiseParameters']) == data.data[i, data.cols[col + '_SD']]
            assert r['experimentId'] == ''              # base time-course, "model as is"

    def test_clean_model_drops_free_and_actions_keeps_measurement_model(self, exported):
        text = (exported / 'parabola.bngl').read_text()
        assert '__FREE' not in text                     # PEtab estimates v1/v2/v3 directly
        assert 'begin actions' not in text              # PEtab drives simulation
        assert 'y()=v1*(x^2)+(v2*x)+v3' in text         # the function (measurement model) survives
        assert 'Molecules x counter()' in text          # the observable survives

    def test_problem_yaml_references_bngl_model(self, exported):
        text = (exported / 'problem.yaml').read_text()
        assert 'format_version: 2.0.0' in text
        assert 'language: bngl' in text
        assert 'location: parabola.bngl' in text

    def test_full_petab_validation_is_clean(self, exported):
        # 1. The external oracle, now at MODEL level: load the whole problem via
        # Problem.from_yaml (the real petablint path -- exercises model_factory ->
        # BnglModel.from_file -> BNG2.pl --check), run ALL tasks.
        pytest.importorskip('petab.v2')  # the v2 typed-table API the oracle needs
        import petab.v1.models as models
        import petab.v2.core as v2core
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        # register_bngl() teaches a stock petab about BNGL, and is a no-op on a
        # petab that ships it natively (#420 Step B) -- so this same path validates
        # against both. We dogfood the fork: when petab is native (bngl known and
        # our wrapper never installed), the loaded model must be petab's OWN
        # BnglModel, not our local stand-in.
        native = ('bngl' in models.known_model_types
                  and not hasattr(v2core, '_pybnf_orig_model_factory'))
        from pybnf.petab.bngl_model import register_bngl
        register_bngl()

        problem = Problem.from_yaml(str(exported / 'problem.yaml'))
        assert type(problem.model).__name__ == 'BnglModel'  # the BNGL loader ran
        if native:
            assert type(problem.model).__module__.startswith('petab.')

        errors = []
        for task in default_validation_tasks:
            issue = task.run(problem)
            if issue is not None and getattr(issue, 'level', None) == \
                    ValidationIssueSeverity.ERROR:
                errors.append((type(task).__name__, issue.message))
        assert errors == []


# ---------------------------------------------------------------------------
# 5. Chunk-1 boundaries raise (documented in code, not silently mis-exported)
# ---------------------------------------------------------------------------

class TestBoundaries:

    def test_sbml_model_not_implemented(self, tmp_path):
        # demo_xml.conf references parabola.xml -> BNGL-only chunk raises.
        with pytest.raises(NotImplementedError):
            export_job(DEMO_DIR / 'demo_xml.conf', tmp_path)

    def test_non_chi_sq_objective_not_implemented(self, tmp_path):
        conf = tmp_path / 'job.conf'
        conf.write_text(
            f"model = {DEMO_DIR / 'parabola.bngl'} : {DEMO_DIR / 'par1.exp'}\n"
            "fit_type = de\n"
            "objfunc = sos\n"
            "uniform_var = v1__FREE 0 10\n")
        with pytest.raises(NotImplementedError):
            export_job(conf, tmp_path / 'out')


class TestCleanModelUnit:

    def test_strips_free_marker_and_actions_block(self):
        src = (
            "begin model\n begin parameters\n  k1 k1__FREE\n end parameters\n"
            "end model\n\nbegin actions\n simulate({})\nend actions\n")
        out = clean_model_for_petab(src, {'k1__FREE': 3.0})
        assert 'k1 3' in out
        assert '__FREE' not in out
        assert 'begin actions' not in out and 'simulate' not in out
        assert 'begin model' in out                     # the model body is untouched


# ---------------------------------------------------------------------------
# 4'. The BnglModel adapter ABC, unit-tested directly (ADR-0026 -- the model-level
#     guarantees the table oracle now checks externally, asserted method by method).
# ---------------------------------------------------------------------------

class TestBnglModel:

    @pytest.fixture
    def model(self, tmp_path):
        # A BnglModel parsed from the exported (cleaned, numeric-nominal) demo model --
        # the same file Problem.from_yaml loads.
        pytest.importorskip('petab')
        from pybnf.petab.bngl_model import BnglModel
        from pybnf.petab.export import export_job
        out = tmp_path / 'p'
        export_job(DEMO_CONF, out)
        return BnglModel.from_file(out / 'parabola.bngl')

    def test_parameter_ids_and_values(self, model):
        assert set(model.get_parameter_ids()) == {'v1', 'v2', 'v3'}
        assert model.get_parameter_value('v1') == 5.0
        assert dict(model.get_free_parameter_ids_with_values()) == \
            {'v1': 5.0, 'v2': 5.0, 'v3': 5.0}

    def test_get_parameter_value_unknown_raises_valueerror(self, model):
        with pytest.raises(ValueError):
            model.get_parameter_value('nope')

    def test_has_entity_spans_full_declared_namespace(self, model):
        # parameter, observable, global function, molecule type -- all model entities.
        for ent in ('v1', 'x', 'y', 'counter'):
            assert model.has_entity_with_id(ent)
        # prefixed PEtab ids + an unknown are NOT model entities (no shadow).
        for non in ('obs_x', 'func_y', 'nope'):
            assert not model.has_entity_with_id(non)

    def test_symbol_allowed_is_the_paramlist_only(self, model):
        # parameters u observables u global functions (verified vs BNG2.pl).
        for sym in ('x', 'y', 'v1'):
            assert model.symbol_allowed_in_observable_formula(sym)
        # a molecule type is an entity but NOT a formula symbol; prefixed ids aren't either.
        for non in ('counter', 'obs_x', 'func_y', 'nope'):
            assert not model.symbol_allowed_in_observable_formula(non)

    def test_is_state_variable_is_seed_species_only(self, model):
        assert model.is_state_variable('counter()')   # the concrete seed species
        assert not model.is_state_variable('v1')       # a parameter is not a species
        assert not model.is_state_variable('x')        # nor is an observable

    def test_expression_valued_parameter_is_not_evaluated(self):
        pytest.importorskip('petab')
        from pybnf.petab._bngl import parse_model
        from pybnf.petab.bngl_model import BnglModel
        ent = parse_model(
            "begin parameters\n base 2\n k_on 2*base\nend parameters\n")
        model = BnglModel(ent, model_id='m')
        assert model.get_parameter_value('base') == 2.0   # numeric RHS -> float
        with pytest.raises(NotImplementedError):           # expression RHS -> confined
            model.get_parameter_value('k_on')


class TestRegisterBngl:

    @staticmethod
    def _petab_is_native():
        """True iff petab already supports BNGL (the #420 Step B fork) and we
        have not installed our own wrapper -- i.e. register_bngl is a no-op."""
        import petab.v1.models as models
        import petab.v2.core as v2core
        return ('bngl' in models.known_model_types
                and not hasattr(v2core, '_pybnf_orig_model_factory'))

    def test_native_support_makes_register_a_noop(self):
        # Against a petab that ships BNGL natively (the dogfooded fork branch),
        # register_bngl must leave model_factory untouched so the native loader
        # wins -- the collapse-to-no-op of ADR-0026.
        pytest.importorskip('petab.v2')
        import petab.v2.core as v2core
        from pybnf.petab.bngl_model import register_bngl

        if not self._petab_is_native():
            pytest.skip("petab does not support BNGL natively in this env")

        before = v2core.model_factory
        register_bngl()
        assert v2core.model_factory is before                 # not rebound
        assert not hasattr(v2core, '_pybnf_orig_model_factory')  # no sentinel

    def test_idempotent_guarded_rebind(self):
        # Against a stock petab without native BNGL, register_bngl installs an
        # idempotent wrapper that routes 'bngl' and delegates everything else.
        pytest.importorskip('petab.v2')
        import petab.v1.models as models
        import petab.v2.core as v2core
        from pybnf.petab.bngl_model import register_bngl

        if self._petab_is_native():
            pytest.skip("petab supports BNGL natively; register_bngl is a no-op")

        register_bngl()
        wrapper = v2core.model_factory
        captured = v2core._pybnf_orig_model_factory
        assert wrapper is not captured            # wrapper installed, original captured
        assert 'bngl' in models.known_model_types

        register_bngl()                            # second call must not re-wrap
        assert v2core._pybnf_orig_model_factory is captured
        assert v2core.model_factory is not captured


# ---------------------------------------------------------------------------
# 6. Chunk 2: Mutants and dose-response -> conditions/experiments (ADR-0027)
#
# Two tests-local synthetic fixtures (a real PyBNF job in tmp_path), each exercising one
# feature family. The Mutants fixture covers BOTH surrogate paths: a *fit* parameter
# mutated relatively (v1*2 -> the surrogate v1__REF*2) and a *fixed* parameter mutated
# relatively (s*5 -> precomputed). The dose-response fixture is one Condition+Experiment
# per measured dose. Both must pass the FULL petab task set (ADR-0026 oracle), now
# genuinely exercising the condition/experiment checks.
# ---------------------------------------------------------------------------

_PARABOLA2_BNGL = """\
begin model
  begin parameters
    v1 v1__FREE
    v2 v2__FREE
    v3 v3__FREE
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

_DOSERESP_BNGL = """\
begin model
  begin parameters
    v1 v1__FREE
    v2 v2__FREE
    L 1
  end parameters
  begin molecule types
    A()
  end molecule types
  begin seed species
    A() L
  end seed species
  begin observables
    Molecules a A()
  end observables
  begin functions
    resp()=v1*a+v2
  end functions
  begin reaction rules
    A()->0 1
  end reaction rules
end model

begin actions
  generate_network({overwrite=>1})
  parameter_scan({parameter=>"L",par_min=>1,par_max=>5,n_scan_pts=>5,t_end=>100,suffix=>"dr"})
end actions
"""


def _write_mutant_fixture(d):
    """A parabola job with a fit-param Mutant (v1*2) and a fixed-param Mutant (s*5)."""
    (d / 'parabola2.bngl').write_text(_PARABOLA2_BNGL)
    (d / 'par1.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n2\t-8\t54\t1\t1\n')
    (d / 'par1fitmut.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n2\t-8\t108\t1\t1\n')
    (d / 'par1fixmut.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t430\t1\t1\n1\t-9\t345\t1\t1\n2\t-8\t270\t1\t1\n')
    conf = d / 'mut.conf'
    conf.write_text(
        'model = parabola2.bngl : par1.exp\n'
        'fit_type = de\nobjfunc = chi_sq\n'
        'uniform_var = v1__FREE 0 10\nuniform_var = v2__FREE 0 10\n'
        'uniform_var = v3__FREE 0 10\n'
        'mutant = parabola2 fitmut v1*2 : par1fitmut.exp\n'
        'mutant = parabola2 fixmut s*5 : par1fixmut.exp\n')
    return conf


def _write_dose_fixture(d):
    """A dose-response job: a Parameter Scan of the input L, swept-axis .exp."""
    (d / 'doseresp.bngl').write_text(_DOSERESP_BNGL)
    (d / 'dr.exp').write_text(
        '# L a resp a_SD resp_SD\n'
        '1\t0.4\t3.1\t0.1\t0.2\n2\t0.8\t5.2\t0.1\t0.2\n3\t1.2\t7.0\t0.1\t0.2\n'
        '4\t1.6\t9.3\t0.1\t0.2\n5\t2.0\t11.1\t0.1\t0.2\n')
    conf = d / 'dose.conf'
    conf.write_text(
        'model = doseresp.bngl : dr.exp\n'
        'fit_type = de\nobjfunc = chi_sq\n'
        'uniform_var = v1__FREE 0 10\nuniform_var = v2__FREE 0 10\n'
        'param_scan = model:doseresp, param:L, min:1, max:5, step:1, time:100, suffix:dr\n')
    return conf


def _assert_petab_clean(exported):
    """The full-task petab oracle: load via Problem.from_yaml, assert zero ERRORs."""
    pytest.importorskip('petab.v2')
    from petab.v2 import Problem
    from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

    from pybnf.petab.bngl_model import register_bngl
    register_bngl()
    problem = Problem.from_yaml(str(exported / 'problem.yaml'))
    assert type(problem.model).__name__ == 'BnglModel'   # the BNGL loader ran
    errors = []
    for task in default_validation_tasks:
        issue = task.run(problem)
        if issue is not None and getattr(issue, 'level', None) == \
                ValidationIssueSeverity.ERROR:
            errors.append((type(task).__name__, issue.message))
    assert errors == []


class TestExportMutants:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = tmp_path_factory.mktemp('mut_job')
        conf = _write_mutant_fixture(src)
        out = src / 'petab'
        export_job(conf, out)
        return out

    def test_writes_conditions_and_experiments(self, exported):
        for name in ('conditions.tsv', 'experiments.tsv'):
            assert (exported / name).is_file()
        text = (exported / 'problem.yaml').read_text()
        assert 'condition_files' in text and 'experiment_files' in text

    def test_fit_mutated_parameter_is_renamed_to_surrogate(self, exported):
        rows = _tsv_rows(exported / 'parameters.tsv')
        ids = {r['parameterId'] for r in rows}
        assert ids == {'v1__REF', 'v2', 'v3'}     # v1 (fit + mutated) -> v1__REF
        assert 'v1' not in ids                     # never in BOTH tables

    def test_conditions_cells(self, exported):
        rows = _tsv_rows(exported / 'conditions.tsv')
        cells = {(r['conditionId'], r['targetId']): r['targetValue'] for r in rows}
        # base experiment pins the removed fit param to its base value
        assert cells[('cond_par1', 'v1')] == 'v1__REF'
        # relative op on a fit param -> symbolic in the surrogate
        assert cells[('cond_par1fitmut', 'v1')] == 'v1__REF * 2'
        # the fixmut experiment still pins v1 (it doesn't mutate it) ...
        assert cells[('cond_par1fixmut', 'v1')] == 'v1__REF'
        # ... and precomputes the relative op on the fixed param (nominal 2 * 5 = 10)
        assert cells[('cond_par1fixmut', 's')] == '10'

    def test_experiments_one_period_at_zero(self, exported):
        rows = _tsv_rows(exported / 'experiments.tsv')
        by_id = {r['experimentId']: r for r in rows}
        assert set(by_id) == {'par1', 'par1fitmut', 'par1fixmut'}
        assert all(r['time'] == '0' for r in rows)
        assert by_id['par1fitmut']['conditionId'] == 'cond_par1fitmut'

    def test_measurements_tagged_by_experiment(self, exported):
        rows = _tsv_rows(exported / 'measurements.tsv')
        eids = {r['experimentId'] for r in rows}
        # M is non-empty, so the base time-course is a named experiment (not '')
        assert eids == {'par1', 'par1fitmut', 'par1fixmut'}
        assert all(r['experimentId'] != '' for r in rows)

    def test_full_petab_validation_is_clean(self, exported):
        _assert_petab_clean(exported)


class TestExportDoseResponse:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = tmp_path_factory.mktemp('dose_job')
        conf = _write_dose_fixture(src)
        out = src / 'petab'
        export_job(conf, out)
        return out

    def test_one_condition_per_dose(self, exported):
        rows = _tsv_rows(exported / 'conditions.tsv')
        assert [(r['conditionId'], r['targetId'], r['targetValue']) for r in rows] == [
            ('cond_dr_0', 'L', '1'), ('cond_dr_1', 'L', '2'), ('cond_dr_2', 'L', '3'),
            ('cond_dr_3', 'L', '4'), ('cond_dr_4', 'L', '5')]

    def test_one_experiment_per_dose(self, exported):
        rows = _tsv_rows(exported / 'experiments.tsv')
        assert {r['experimentId'] for r in rows} == {f'dr_{i}' for i in range(5)}
        assert all(r['time'] == '0' for r in rows)

    def test_measurement_time_is_the_scan_time(self, exported):
        # The independent axis is the swept parameter; the measurement time is the scan's
        # fixed simulation time (100), not a data column.
        rows = _tsv_rows(exported / 'measurements.tsv')
        assert all(r['time'] == '100' for r in rows)
        # each dose row -> its own experiment, value carried from the .exp cell
        by = {(r['observableId'], r['experimentId']): r['measurement'] for r in rows}
        assert by[('obs_a', 'dr_0')] == '0.4'
        assert by[('func_resp', 'dr_4')] == '11.1'

    def test_swept_parameter_not_renamed(self, exported):
        # L is fixed (not fit), so no surrogate: the fit params stay as model names.
        ids = {r['parameterId'] for r in _tsv_rows(exported / 'parameters.tsv')}
        assert ids == {'v1', 'v2'}

    def test_full_petab_validation_is_clean(self, exported):
        _assert_petab_clean(exported)


class TestConditionMappingUnit:

    def test_absolute_set_is_the_bare_number(self):
        assert mutation_target_value('=', 5.0) == '5'

    def test_relative_op_on_fit_target_is_symbolic(self):
        assert mutation_target_value('*', 2.0, surrogate='v1__REF') == 'v1__REF * 2'
        assert mutation_target_value('/', 4.0, surrogate='k__REF') == 'k__REF / 4'

    @pytest.mark.parametrize('op,val,nominal,expected', [
        ('*', 5.0, 2.0, '10'), ('/', 4.0, 8.0, '2'),
        ('+', 3.0, 2.0, '5'), ('-', 1.0, 2.0, '1')])
    def test_relative_op_on_fixed_target_is_precomputed(self, op, val, nominal, expected):
        assert mutation_target_value(op, val, nominal=nominal) == expected

    def test_relative_op_on_expression_nominal_raises(self):
        with pytest.raises(NotImplementedError):
            mutation_target_value('*', 2.0, nominal=None)

    def test_surrogate_marker_is_double_underscore(self):
        assert surrogate_name('v1') == 'v1__REF'

    def test_build_mutant_conditions_surrogate_set_and_base(self):
        # v1 is fit + mutated -> surrogate; s is fixed -> precomputed; base gets pinned.
        muts = [('m1', [('v1', '*', 2.0)], 'b_m1'),
                ('m2', [('s', '*', 5.0)], 'b_m2')]
        cond, exp, surrogate, base_id = build_mutant_conditions(
            'b', muts, fit_params={'v1', 'v2'}, nominal_of=lambda v: 2.0)
        assert surrogate == {'v1'}            # only the fit-and-mutated param
        assert base_id == 'b'                  # named base (M non-empty)
        cells = {(r.condition_id, r.target_id): r.target_value for r in cond}
        assert cells[('cond_b', 'v1')] == 'v1__REF'        # base pin
        assert cells[('cond_b_m1', 'v1')] == 'v1__REF * 2'  # surrogate op
        assert cells[('cond_b_m2', 'v1')] == 'v1__REF'      # pinned in the other mutant
        assert cells[('cond_b_m2', 's')] == '10'            # precomputed fixed op
        assert {e.experiment_id for e in exp} == {'b', 'b_m1', 'b_m2'}

    def test_build_mutant_conditions_fixed_only_leaves_base_unnamed(self):
        # No fit param mutated -> empty M, base stays "model as is" ('').
        muts = [('m1', [('s', '=', 0.0)], 'b_m1')]
        cond, exp, surrogate, base_id = build_mutant_conditions(
            'b', muts, fit_params={'v1'}, nominal_of=lambda v: 1.0)
        assert surrogate == set()
        assert base_id == ''
        assert {e.experiment_id for e in exp} == {'b_m1'}  # no base experiment row

    def test_build_dose_response_conditions(self):
        cond, exp, eids = build_dose_response_conditions('dr', 'L', [1.0, 2.0], 100.0)
        assert eids == ['dr_0', 'dr_1']
        assert [(r.condition_id, r.target_id, r.target_value) for r in cond] == [
            ('cond_dr_0', 'L', '1'), ('cond_dr_1', 'L', '2')]
        assert all(e.time == 0.0 for e in exp)


class TestDoseResponseMeasurementPivot:

    def test_each_row_is_its_own_experiment_at_scan_time(self):
        arr = np.array([[1.0, 0.4], [2.0, 0.8]])
        data = Data.from_columns(arr, ['L', 'a'])
        rows = dose_response_measurement_rows(
            data, {'a': 'obs_a'}, ['dr_0', 'dr_1'], scan_time=100.0)
        assert [(r.experiment_id, r.time, r.measurement) for r in rows] == [
            ('dr_0', 100.0, 0.4), ('dr_1', 100.0, 0.8)]


class TestChunk2Boundaries:

    def test_mutants_and_dose_response_together_not_implemented(self, tmp_path):
        _write_mutant_fixture(tmp_path)
        conf = tmp_path / 'mut.conf'
        conf.write_text(conf.read_text() +
                        'param_scan = model:parabola2, param:s, min:1, max:2, '
                        'step:1, time:10, suffix:sc\n')
        with pytest.raises(NotImplementedError):
            export_job(conf, tmp_path / 'out')

    def test_scanning_a_fit_parameter_not_implemented(self, tmp_path):
        _write_dose_fixture(tmp_path)
        # Sweep v1, which is a fit parameter -> would overlap parameter table.
        (tmp_path / 'dr.exp').write_text(
            '# v1 a resp a_SD resp_SD\n1\t0.4\t3.1\t0.1\t0.2\n2\t0.8\t5.2\t0.1\t0.2\n')
        conf = tmp_path / 'dose.conf'
        conf.write_text(
            'model = doseresp.bngl : dr.exp\nfit_type = de\nobjfunc = chi_sq\n'
            'uniform_var = v1__FREE 0 10\nuniform_var = v2__FREE 0 10\n'
            'param_scan = model:doseresp, param:v1, min:1, max:2, step:1, '
            'time:100, suffix:dr\n')
        with pytest.raises(NotImplementedError):
            export_job(conf, tmp_path / 'out')

    def test_dose_response_without_param_scan_not_implemented(self, tmp_path):
        _write_dose_fixture(tmp_path)
        # Strip the param_scan: a swept-axis .exp with no action to source its time.
        conf = tmp_path / 'dose.conf'
        conf.write_text(
            'model = doseresp.bngl : dr.exp\nfit_type = de\nobjfunc = chi_sq\n'
            'uniform_var = v1__FREE 0 10\nuniform_var = v2__FREE 0 10\n')
        with pytest.raises(NotImplementedError):
            export_job(conf, tmp_path / 'out')

    def test_mutant_target_not_a_model_entity_raises(self, tmp_path):
        from pybnf.printing import PybnfError
        _write_mutant_fixture(tmp_path)
        conf = tmp_path / 'mut.conf'
        conf.write_text(
            'model = parabola2.bngl : par1.exp\nfit_type = de\nobjfunc = chi_sq\n'
            'uniform_var = v1__FREE 0 10\nuniform_var = v2__FREE 0 10\n'
            'uniform_var = v3__FREE 0 10\n'
            'mutant = parabola2 m nope=0 : par1fitmut.exp\n')
        with pytest.raises(PybnfError):
            export_job(conf, tmp_path / 'out')

    def test_relative_op_on_expression_valued_fixed_param_not_implemented(self, tmp_path):
        # s has an expression RHS, so a relative op on it can't be precomputed.
        bngl = _PARABOLA2_BNGL.replace('    s 2\n', '    base 1\n    s 2*base\n')
        (tmp_path / 'parabola2.bngl').write_text(bngl)
        (tmp_path / 'par1.exp').write_text(
            '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n')
        (tmp_path / 'par1fixmut.exp').write_text(
            '# time x y x_SD y_SD\n0\t-10\t430\t1\t1\n1\t-9\t345\t1\t1\n')
        conf = tmp_path / 'mut.conf'
        conf.write_text(
            'model = parabola2.bngl : par1.exp\nfit_type = de\nobjfunc = chi_sq\n'
            'uniform_var = v1__FREE 0 10\nuniform_var = v2__FREE 0 10\n'
            'uniform_var = v3__FREE 0 10\n'
            'mutant = parabola2 fixmut s*5 : par1fixmut.exp\n')
        with pytest.raises(NotImplementedError):
            export_job(conf, tmp_path / 'out')
