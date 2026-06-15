"""Unit tests for the PEtab v2 *exporter* (#407, exporter-first; ADR-0025).

The exporter reads a working PyBNF/BNGL job and serializes it to a PEtab v2 problem.
Its contracts, by strength of oracle:

1. **The external table oracle.** petab's own validation, run on a *model-less*
   ``petab.v2.Problem`` built from the emitted tables (``petablint`` cannot load a
   BNGL model in petab 0.8.2, so we run the ~13 table-level tasks and skip the ~5
   model-cross ones). The exported ``demo`` tables must validate with **no errors**.
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
from pybnf.petab.export import (
    clean_model_for_petab,
    export_job,
)
from pybnf.petab.measurements import measurement_rows_from_data
from pybnf.petab.observables import petab_observable_row
from pybnf.petab.parameters import (
    PetabParameterRow,
    free_parameter_from_row,
    petab_parameter_row,
)
from pybnf.pset import FreeParameter

DEMO_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'demo'
DEMO_CONF = DEMO_DIR / 'demo_bng.conf'

# Validation tasks that need a loaded model; excluded because petab 0.8.2 has no BNGL
# loader. Their job (formula/parameter names are model entities) is checked by the
# model-correspondence tests instead (ADR-0025).
_MODEL_TASKS = {
    'CheckModel', 'CheckObservablesDoNotShadowModelEntities',
    'CheckAllParametersPresentInParameterTable', 'CheckInitialChangeSymbols',
    'CheckValidConditionTargets'}


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

    def test_table_level_petab_validation_is_clean(self, exported):
        # 1. The external oracle: model-less Problem + the table-level validation tasks.
        pytest.importorskip('petab.v2')  # the v2 typed-table API the oracle needs (ADR-0025)
        from petab.v2 import Problem
        from petab.v2.core import (
            MeasurementTable,
            ObservableTable,
            ParameterTable,
        )
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        problem = Problem(
            models=[],
            observable_tables=[ObservableTable.from_tsv(str(exported / 'observables.tsv'))],
            measurement_tables=[MeasurementTable.from_tsv(str(exported / 'measurements.tsv'))],
            parameter_tables=[ParameterTable.from_tsv(str(exported / 'parameters.tsv'))],
        )
        errors = []
        for task in default_validation_tasks:
            if type(task).__name__ in _MODEL_TASKS:
                continue
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
