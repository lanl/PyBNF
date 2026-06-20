"""Unit tests for the PEtab v2 *exporter* (#407/#423; ADR-0025/0027/0028).

The exporter reads a working PyBNF/BNGL job *on the new-era surface* (model: /
experiment: / data: / condition: / observable:) and serializes it to a PEtab v2 problem.
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
   a model parameter; the PEtab-clean model carries bare ids (new-era binds by id,
   ADR-0034) and drops ``begin actions``.
5. **The documented boundaries** raise (parameter-scan #426 deferral, legacy linkage
   refused, non-uniform prior, no-``_SD`` noise, SBML model, PEtab-inexpressible objective).
"""

from pathlib import Path

import numpy as np
import pytest

from pybnf.data import Data
from pybnf.petab.conditions import (
    build_dose_response_conditions,
    build_experiment_conditions,
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
# The new-era (edition 2) demo conf: the PEtab v2 exporter requires a modern config and
# reads the new-era data surface (model: / experiment: / data:), not the legacy linkage
# (ADR-0028 Chunk 5). demo_bng.conf stays legacy for the fitter / install check;
# demo_bng_v2.conf is its fully new-era twin (model: parabola_v2.bngl -- parabola.bngl
# without a begin actions block, since the action is synthesized from the experiment).
DEMO_CONF = DEMO_DIR / 'demo_bng_v2.conf'
DEMO_MODEL = 'parabola_v2.bngl'


def _tsv_rows(path):
    """Read a TSV into a list of dict rows (a tiny stdlib reader for assertions)."""
    import csv
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _petab_validation_errors(problem_yaml):
    """Load a problem via the real petablint path and return its ERROR-level issues.

    The model-level external oracle: ``Problem.from_yaml`` exercises
    ``model_factory -> BnglModel -> BNG2.pl --check`` (after ``register_bngl()``,
    which is a no-op on a petab that ships BNGL natively, #420 Step B), then runs
    **every** ``default_validation_task`` -- the model-cross checks included. Returns
    ``[(task, message), ...]``; empty means a clean problem.
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

    def test_parameter_id_defaults_to_the_free_parameter_name(self):
        # New-era binds by id (ADR-0034): the free parameter's name IS the parameterId,
        # carried through verbatim -- no marker to strip.
        fp = FreeParameter('k1', 'uniform_var', 1.0, 2.0)
        assert petab_parameter_row(fp).parameter_id == 'k1'

    def test_explicit_parameter_id_overrides_the_name(self):
        # The exporter passes parameter_id explicitly for a fit-and-mutated parameter
        # (renamed to its <p>__REF surrogate); the explicit id wins over the name.
        fp = FreeParameter('k', 'uniform_var', 1.0, 2.0)
        assert petab_parameter_row(fp, parameter_id='k__REF').parameter_id == 'k__REF'

    @pytest.mark.parametrize('lb,ub', [(0.5, 10.0), (1e-3, 1e3), (2.0, 2.5)])
    def test_loguniform_parameter_round_trips(self, lb, ub):
        # log-uniform states its family (PEtab's default uniform is *linear*); the
        # bounds are linear and round-trip exactly -- no ln10 scaling on a uniform.
        row = PetabParameterRow(parameter_id='k', estimate=True, lower_bound=lb,
                                upper_bound=ub, prior_distribution='log-uniform',
                                prior_parameters=(lb, ub))
        fp = free_parameter_from_row(row)
        assert fp.type == 'loguniform_var'
        assert petab_parameter_row(fp) == row

    @pytest.mark.parametrize('dist,keyword', [
        ('normal', 'normal_var'), ('laplace', 'laplace_var'),
        ('log-normal', 'lognormal_var'), ('log-laplace', 'loglaplace_var')])
    def test_location_scale_prior_round_trips(self, dist, keyword):
        # Two-sided bounds truncate the unbounded family (ADR-0020); the log families
        # carry their (loc, scale) in natural log, which round-trips through the
        # ln10 conversion to within floating-point (not bit-exact).
        row = PetabParameterRow(parameter_id='k', estimate=True, lower_bound=0.5,
                                upper_bound=20.0, prior_distribution=dist,
                                prior_parameters=(1.0, 0.5))
        fp = free_parameter_from_row(row)
        assert fp.type == keyword
        out = petab_parameter_row(fp)
        assert out.parameter_id == 'k' and out.prior_distribution == dist
        assert (out.lower_bound, out.upper_bound) == (0.5, 20.0)
        assert out.prior_parameters == pytest.approx((1.0, 0.5))

    def test_unbounded_location_scale_writes_blank_bounds(self):
        # A conf-built lognormal has no truncation grammar yet (#417) -> unbounded ->
        # blank PEtab bounds, prior fully carried by priorParameters.
        fp = FreeParameter('k', 'lognormal_var', 0.0, 1.0)
        row = petab_parameter_row(fp)
        assert row.lower_bound is None and row.upper_bound is None
        assert row.prior_distribution == 'log-normal'

    def test_no_prior_keyword_export_not_implemented(self):
        # var / logvar are a flat improper prior -- not a PEtab probability family.
        fp = FreeParameter('k', 'logvar', 1.0, 0.1)
        with pytest.raises(NotImplementedError):
            petab_parameter_row(fp)

    @pytest.mark.parametrize('kind,prefix', [('observable', 'obs_'), ('function', 'func_')])
    def test_observable_row_prefix_formula_and_placeholder(self, kind, prefix):
        # A per-point _SD source -> a declared placeholder bound by noiseParameters.
        row = petab_observable_row('z', kind, 'normal', ('placeholder', None))
        assert row.observable_id == f'{prefix}z'
        assert row.observable_formula == 'z'          # bare model name, never a body
        assert row.noise_distribution == 'normal'
        assert row.noise_formula == f'noiseParameter1_{prefix}z'
        assert row.noise_placeholders == row.noise_formula

    def test_observable_row_constant_sigma_is_inline_no_placeholder(self):
        # A fixed/column-mean sigma -> a numeric noiseFormula, no placeholder declared.
        row = petab_observable_row('z', 'observable', 'laplace', ('constant', 1.0))
        assert row.noise_formula == '1'               # num() drops the trailing .0
        assert row.noise_distribution == 'laplace'
        assert row.noise_placeholders is None

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
                     'problem.yaml', DEMO_MODEL):
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

    def test_clean_model_drops_actions_keeps_measurement_model(self, exported):
        text = (exported / DEMO_MODEL).read_text()
        assert '__FREE' not in text                     # new-era binds by id (trivially true)
        assert 'v1 0.5' in text                          # the real nominal carried verbatim
        assert 'begin actions' not in text              # PEtab drives simulation
        assert 'y()=v1*(x^2)+(v2*x)+v3' in text         # the function (measurement model) survives
        assert 'Molecules x counter()' in text          # the observable survives

    def test_problem_yaml_references_bngl_model(self, exported):
        text = (exported / 'problem.yaml').read_text()
        assert 'format_version: 2.0.0' in text
        assert 'language: bngl' in text
        assert f'location: {DEMO_MODEL}' in text

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
# The dominant real prior: a log-uniform job exported end-to-end (#423 finding D:
# loguniform_var is the corpus's most common non-trivial prior, 1327 uses).
# ---------------------------------------------------------------------------

class TestExportLogUniform:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        import shutil
        src = tmp_path_factory.mktemp('loguniform_src')
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\nobjective = chi_sq\n'
            f'model: {DEMO_MODEL}\n'
            'experiment: par1, data: par1.exp\n'
            'loguniform_var = v1 0.1 10\n'
            'loguniform_var = v2 0.1 10\n'
            'loguniform_var = v3 0.1 10\n')
        out = tmp_path_factory.mktemp('loguniform_out')
        export_job(src / 'job.conf', out)
        return out

    def test_parameters_state_the_log_uniform_prior(self, exported):
        rows = _tsv_rows(exported / 'parameters.tsv')
        by_id = {r['parameterId']: r for r in rows}
        assert set(by_id) == {'v1', 'v2', 'v3'}
        for r in by_id.values():
            assert r['priorDistribution'] == 'log-uniform'   # NOT PEtab's default uniform
            assert r['priorParameters'] == '0.1;10'
            assert (r['lowerBound'], r['upperBound']) == ('0.1', '10')

    def test_full_petab_validation_is_clean(self, exported):
        assert _petab_validation_errors(exported / 'problem.yaml') == []


# ---------------------------------------------------------------------------
# The objective family: the Gaussian/Laplace likelihoods PEtab v2 can express
# (#423 finding B/2 -- only chi_sq mapped before; sos/sod/ave_norm_sos now do).
# ---------------------------------------------------------------------------

class TestExportObjectiveFamily:

    def _export(self, tmp_path_factory, objfunc):
        import shutil
        src = tmp_path_factory.mktemp('objfam_src')
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\nobjective = {objfunc}\n'
            f'model: {DEMO_MODEL}\n'
            'experiment: par1, data: par1.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\n')
        out = tmp_path_factory.mktemp('objfam_out')
        export_job(src / 'job.conf', out)
        return out

    @pytest.mark.parametrize('objfunc,distribution', [
        ('sos', 'normal'), ('sod', 'laplace')])
    def test_fixed_sigma_objfunc_is_constant_noise_formula(
            self, tmp_path_factory, objfunc, distribution):
        out = self._export(tmp_path_factory, objfunc)
        rows = _tsv_rows(out / 'observables.tsv')
        for r in rows:
            assert r['noiseDistribution'] == distribution   # gaussian->normal, laplace
            assert r['noiseFormula'] == '1'                  # fix_at 1, inline
            assert r['noisePlaceholders'] == ''              # no per-point placeholder
        # and no dangling noiseParameters override in the measurements
        meas = _tsv_rows(out / 'measurements.tsv')
        assert all(m['noiseParameters'] == '' for m in meas)
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_ave_norm_sos_is_the_column_mean(self, tmp_path_factory):
        out = self._export(tmp_path_factory, 'ave_norm_sos')
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        by_id = {r['observableId']: r for r in _tsv_rows(out / 'observables.tsv')}
        col_of = {'obs_x': 'x', 'func_y': 'y'}
        for oid, r in by_id.items():
            assert r['noiseDistribution'] == 'normal'
            assert float(r['noiseFormula']) == pytest.approx(np.average(data[col_of[oid]]))
            assert r['noisePlaceholders'] == ''
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_chi_sq_still_uses_the_sd_placeholder(self, tmp_path_factory):
        # The existing _SD path is unchanged: per-point placeholder + noiseParameters.
        out = self._export(tmp_path_factory, 'chi_sq')
        rows = _tsv_rows(out / 'observables.tsv')
        for r in rows:
            assert r['noiseDistribution'] == 'normal'
            assert r['noiseFormula'] == f"noiseParameter1_{r['observableId']}"
            assert r['noisePlaceholders'] == r['noiseFormula']
        meas = _tsv_rows(out / 'measurements.tsv')
        assert all(m['noiseParameters'] != '' for m in meas)
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_modern_whole_fit_noise_model_line_exports(self, tmp_path_factory):
        # The modern ADR-0031 surface (not legacy objfunc): a whole-fit noise_model line.
        import shutil
        src = tmp_path_factory.mktemp('nm_src')
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\n'
            f'model: {DEMO_MODEL}\n'
            'experiment: par1, data: par1.exp\n'
            'noise_model = laplace, scale = fix_at 1\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\n')
        out = tmp_path_factory.mktemp('nm_out')
        export_job(src / 'job.conf', out)
        rows = _tsv_rows(out / 'observables.tsv')
        assert all(r['noiseDistribution'] == 'laplace' for r in rows)
        assert all(r['noiseFormula'] == '1' for r in rows)
        assert _petab_validation_errors(out / 'problem.yaml') == []


# ---------------------------------------------------------------------------
# Chunk 5a: the exporter reads the new-era data surface (model: / experiment: / data: /
# observable:) as transcription, not the legacy linkage (ADR-0028). The demo twin
# (TestExportDemo, DEMO_CONF) already exercises a wildtype time course end-to-end through
# the full oracle; these tests cover the new behaviours: replicates, the observable:
# override, and the parameter-scan deferral (#426). Conditioned experiments are Chunk 5b.
# ---------------------------------------------------------------------------

class TestExportNewEra:

    def _src(self, tmp_path_factory, name='newera'):
        import shutil
        src = tmp_path_factory.mktemp(name)
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        return src

    _HEAD = ('edition = 2\njob_type = de\nobjective = chi_sq\n'
             f'model: {DEMO_MODEL}\n')
    _PARAMS = ('uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
               'uniform_var = v3 0 10\n')

    def test_replicates_become_repeated_measurement_rows(self, tmp_path_factory):
        src = self._src(tmp_path_factory, 'reps')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: par1, data: par1.exp, par1.exp\n' + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        rows = _tsv_rows(out / 'measurements.tsv')
        # Two identical replicate files -> twice the single-file rows (n times x {x, y}).
        assert len(rows) == data.data.shape[0] * 2 * 2
        assert all(r['experimentId'] == '' for r in rows)   # wildtype, M empty
        _assert_petab_clean(out)

    def test_observable_override_remaps_a_renamed_column(self, tmp_path_factory):
        src = self._src(tmp_path_factory, 'obsoverride')
        # The data's x column is headed 'x_meas'; observable: x, column: x_meas rewires it
        # (and its _SD companion) so it classifies as the model observable obs_x.
        (src / 'renamed.exp').write_text(
            '# time x_meas y x_meas_SD y_SD\n'
            '0\t-10\t43\t1\t1\n1\t-9\t34.5\t1\t1\n2\t-8\t27\t1\t1\n')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: par1, data: renamed.exp\n'
            'observable: x, column: x_meas\n' + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        ids = {r['observableId'] for r in _tsv_rows(out / 'observables.tsv')}
        assert ids == {'obs_x', 'func_y'}    # x_meas classified as obs_x after the rename
        _assert_petab_clean(out)

    def test_missing_observable_override_column_raises(self, tmp_path_factory):
        from pybnf.printing import PybnfError
        src = self._src(tmp_path_factory, 'obsmiss')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: par1, data: par1.exp\n'
            'observable: x, column: nope\n' + self._PARAMS)
        with pytest.raises(PybnfError, match='nope'):
            export_job(src / 'job.conf', src / 'out')

    def test_inferred_parameter_scan_experiment_is_deferred(self, tmp_path_factory):
        # A non-time independent variable -> parameter_scan, deferred (#426): a fully
        # new-era conf cannot author the scan endpoint time, so export never sees one.
        src = self._src(tmp_path_factory, 'pscan')
        (src / 'dose.exp').write_text('# dose x y\n1\t-10\t43\n2\t-9\t34.5\n')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: dr, data: dose.exp\n' + self._PARAMS)
        with pytest.raises(NotImplementedError, match='#426'):
            export_job(src / 'job.conf', src / 'out')

    def test_explicit_type_parameter_scan_is_deferred(self, tmp_path_factory):
        src = self._src(tmp_path_factory, 'pscan2')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: dr, type: parameter_scan, data: par1.exp\n'
            + self._PARAMS)
        with pytest.raises(NotImplementedError, match='#426'):
            export_job(src / 'job.conf', src / 'out')

    def test_undefined_referenced_condition_raises(self, tmp_path_factory):
        from pybnf.printing import PybnfError
        src = self._src(tmp_path_factory, 'undefcond')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: par1, condition: nope, data: par1.exp\n'
            + self._PARAMS)
        with pytest.raises(PybnfError, match='nope'):
            export_job(src / 'job.conf', src / 'out')


# ---------------------------------------------------------------------------
# Chunk 5b: new-era condition: / conditioned experiment: -> conditions/experiments
# tables (the surrogate-base machinery of ADR-0027 generalized to named conditions +
# named experiments that reference them; build_experiment_conditions).
# ---------------------------------------------------------------------------

def _write_newera_condition_fixture(d):
    """A new-era job: a fit-param condition (v1*2 -> surrogate) and a fixed-param
    condition (s*5 -> precomputed), a wildtype experiment + two conditioned experiments.
    Reuses the chunk-2 model (its begin actions block is stripped on export)."""
    (d / 'parabola2.bngl').write_text(_PARABOLA2_BNGL)
    (d / 'wt.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n2\t-8\t54\t1\t1\n')
    (d / 'dbl.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n2\t-8\t108\t1\t1\n')
    (d / 'scl.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t430\t1\t1\n1\t-9\t345\t1\t1\n2\t-8\t270\t1\t1\n')
    conf = d / 'job.conf'
    conf.write_text(
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        'model: parabola2.bngl\n'
        'condition: doubled, perturbations: v1 * 2\n'
        'condition: scaled, perturbations: s * 5\n'
        'experiment: wt, data: wt.exp\n'
        'experiment: dbl, condition: doubled, data: dbl.exp\n'
        'experiment: scl, condition: scaled, data: scl.exp\n'
        'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
        'uniform_var = v3 0 10\n')
    return conf


class TestExportNewEraConditions:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = tmp_path_factory.mktemp('newera_cond')
        conf = _write_newera_condition_fixture(src)
        out = src / 'petab'
        export_job(conf, out)
        return out

    def test_writes_conditions_and_experiments(self, exported):
        for name in ('conditions.tsv', 'experiments.tsv'):
            assert (exported / name).is_file()
        text = (exported / 'problem.yaml').read_text()
        assert 'condition_files' in text and 'experiment_files' in text

    def test_fit_perturbed_param_is_renamed_to_surrogate(self, exported):
        ids = {r['parameterId'] for r in _tsv_rows(exported / 'parameters.tsv')}
        assert ids == {'v1__REF', 'v2', 'v3'}   # v1 (fit + perturbed) -> v1__REF
        assert 'v1' not in ids                    # never in both tables

    def test_condition_cells(self, exported):
        rows = _tsv_rows(exported / 'conditions.tsv')
        cells = {(r['conditionId'], r['targetId']): r['targetValue'] for r in rows}
        # relative op on a fit param -> symbolic in the surrogate
        assert cells[('cond_doubled', 'v1')] == 'v1__REF * 2'
        # the scaled condition still pins v1 (it doesn't perturb it) ...
        assert cells[('cond_scaled', 'v1')] == 'v1__REF'
        # ... and precomputes the relative op on the fixed param (nominal 2 * 5 = 10)
        assert cells[('cond_scaled', 's')] == '10'
        # the synthesized wildtype base pins the removed fit param to its base value
        assert cells[('cond_wildtype', 'v1')] == 'v1__REF'

    def test_experiments_reference_their_conditions(self, exported):
        by_id = {r['experimentId']: r for r in _tsv_rows(exported / 'experiments.tsv')}
        assert set(by_id) == {'wt', 'dbl', 'scl'}
        assert by_id['wt']['conditionId'] == 'cond_wildtype'
        assert by_id['dbl']['conditionId'] == 'cond_doubled'
        assert by_id['scl']['conditionId'] == 'cond_scaled'
        assert all(r['time'] == '0' for r in by_id.values())

    def test_measurements_tagged_by_experiment(self, exported):
        eids = {r['experimentId'] for r in _tsv_rows(exported / 'measurements.tsv')}
        # M non-empty, so even the wildtype experiment is named (not '').
        assert eids == {'wt', 'dbl', 'scl'}
        assert '' not in eids

    def test_full_petab_validation_is_clean(self, exported):
        _assert_petab_clean(exported)

    def test_shared_condition_rows_emitted_once(self, tmp_path_factory):
        # Two experiments referencing the same condition -> its rows emitted once (the new
        # degree of freedom over legacy, which couldn't share a mutant across datasets).
        src = tmp_path_factory.mktemp('shared_cond')
        (src / 'parabola2.bngl').write_text(_PARABOLA2_BNGL)
        for f in ('a.exp', 'b.exp'):
            (src / f).write_text(
                '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n')
        conf = src / 'job.conf'
        conf.write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: parabola2.bngl\n'
            'condition: doubled, perturbations: v1 * 2\n'
            'experiment: ea, condition: doubled, data: a.exp\n'
            'experiment: eb, condition: doubled, data: b.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\n')
        out = src / 'petab'
        export_job(conf, out)
        crows = _tsv_rows(out / 'conditions.tsv')
        # No wildtype experiment here, so no cond_wildtype: just the one shared condition.
        assert [(r['conditionId'], r['targetId']) for r in crows] == [
            ('cond_doubled', 'v1')]
        erows = _tsv_rows(out / 'experiments.tsv')
        assert {r['experimentId'] for r in erows} == {'ea', 'eb'}
        assert all(r['conditionId'] == 'cond_doubled' for r in erows)
        _assert_petab_clean(out)

    def test_condition_target_not_a_model_entity_raises(self, tmp_path_factory):
        from pybnf.printing import PybnfError
        src = tmp_path_factory.mktemp('badtarget')
        (src / 'parabola2.bngl').write_text(_PARABOLA2_BNGL)
        (src / 'e.exp').write_text('# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n')
        conf = src / 'job.conf'
        conf.write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: parabola2.bngl\n'
            'condition: bad, perturbations: nope = 0\n'
            'experiment: e, condition: bad, data: e.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\n')
        with pytest.raises(PybnfError, match='nope'):
            export_job(conf, src / 'out')

    def test_relative_op_on_expression_valued_fixed_param_raises(self, tmp_path_factory):
        # s has an expression RHS, so a relative op (s*5) in a condition can't be
        # precomputed to a number -> the surrogate mapping raises (NotImplementedError).
        src = tmp_path_factory.mktemp('exprnom')
        (src / 'parabola2.bngl').write_text(
            _PARABOLA2_BNGL.replace('    s 2\n', '    base 1\n    s 2*base\n'))
        (src / 'e.exp').write_text('# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n')
        conf = src / 'job.conf'
        conf.write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: parabola2.bngl\n'
            'condition: scaled, perturbations: s * 5\n'
            'experiment: e, condition: scaled, data: e.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\n')
        with pytest.raises(NotImplementedError):
            export_job(conf, src / 'out')


class TestBuildExperimentConditions:

    def test_surrogate_set_and_wildtype_base(self):
        exps = [('wt', None), ('dbl', 'doubled'), ('scl', 'scaled')]
        conds = {'doubled': [('v1', '*', 2.0)], 'scaled': [('s', '*', 5.0)]}
        cond, exp, surrogate, eids = build_experiment_conditions(
            exps, conds, fit_params={'v1', 'v2', 'v3'}, nominal_of=lambda v: 2.0)
        assert surrogate == {'v1'}                 # only the fit-and-perturbed param
        cells = {(r.condition_id, r.target_id): r.target_value for r in cond}
        assert cells[('cond_doubled', 'v1')] == 'v1__REF * 2'   # surrogate op
        assert cells[('cond_scaled', 'v1')] == 'v1__REF'        # pinned in the other cond
        assert cells[('cond_scaled', 's')] == '10'              # precomputed fixed op
        assert cells[('cond_wildtype', 'v1')] == 'v1__REF'      # synthesized base pin
        assert eids == {'wt': 'wt', 'dbl': 'dbl', 'scl': 'scl'}
        assert {e.experiment_id: e.condition_id for e in exp} == {
            'wt': 'cond_wildtype', 'dbl': 'cond_doubled', 'scl': 'cond_scaled'}

    def test_no_surrogate_leaves_wildtype_model_as_is(self):
        # Only a fixed param perturbed -> empty M -> wildtype experimentId '' (no row).
        exps = [('wt', None), ('scl', 'scaled')]
        conds = {'scaled': [('s', '*', 5.0)]}
        cond, exp, surrogate, eids = build_experiment_conditions(
            exps, conds, fit_params={'v1'}, nominal_of=lambda v: 2.0)
        assert surrogate == set()
        assert eids == {'wt': '', 'scl': 'scl'}
        assert {e.experiment_id for e in exp} == {'scl'}   # no wildtype experiment row
        cells = {(r.condition_id, r.target_id): r.target_value for r in cond}
        assert cells == {('cond_scaled', 's'): '10'}

    def test_shared_condition_rows_emitted_once(self):
        exps = [('ea', 'doubled'), ('eb', 'doubled')]
        conds = {'doubled': [('v1', '*', 2.0)]}
        cond, exp, surrogate, eids = build_experiment_conditions(
            exps, conds, fit_params={'v1'}, nominal_of=lambda v: 1.0)
        assert [(r.condition_id, r.target_id) for r in cond] == [('cond_doubled', 'v1')]
        assert {e.experiment_id for e in exp} == {'ea', 'eb'}

    def test_unused_condition_emits_nothing(self):
        # A condition referenced by no experiment contributes no rows and no surrogate.
        exps = [('e', 'used')]
        conds = {'used': [('s', '=', 0.0)], 'unused': [('v1', '*', 2.0)]}
        cond, exp, surrogate, eids = build_experiment_conditions(
            exps, conds, fit_params={'v1'}, nominal_of=lambda v: 1.0)
        assert surrogate == set()                       # v1 only in the unused condition
        assert {r.condition_id for r in cond} == {'cond_used'}

    def test_wildtype_named_condition_clashes(self):
        from pybnf.printing import PybnfError
        exps = [('wt', None), ('e', 'wildtype')]
        conds = {'wildtype': [('v1', '*', 2.0)]}
        with pytest.raises(PybnfError, match='wildtype'):
            build_experiment_conditions(exps, conds, fit_params={'v1'},
                                        nominal_of=lambda v: 1.0)


# ---------------------------------------------------------------------------
# 5. Documented boundaries raise (in code, not silently mis-exported). The confs are
# all on the new-era surface (model: / experiment: / data:), since the exporter now
# refuses the legacy linkage (ADR-0028 Chunk 5c).
# ---------------------------------------------------------------------------

def _boundary_conf(tmp_path, body):
    """A new-era boundary conf using the absolute DEMO model/data (most boundaries raise
    before any file is read; the few that reach the data read find real files). ``body`` is
    the boundary-triggering line(s) -- objective/noise + a free parameter."""
    conf = tmp_path / 'job.conf'
    conf.write_text(
        f"edition = 2\njob_type = de\n"
        f"model: {DEMO_DIR / DEMO_MODEL}\n"
        f"experiment: e, data: {DEMO_DIR / 'par1.exp'}\n"
        + body)
    return conf


class TestBoundaries:

    def test_sbml_model_not_implemented(self, tmp_path):
        # demo_xml.conf references parabola.xml -> BNGL-only chunk (or legacy) raises.
        with pytest.raises(NotImplementedError):
            export_job(DEMO_DIR / 'demo_xml.conf', tmp_path)

    def test_legacy_edition_job_is_refused(self, tmp_path):
        # PEtab v2 interop is new-era only: a legacy conf (no edition, legacy objfunc)
        # is refused -- the user must modernize (edition >= 2 + the objective surface).
        conf = tmp_path / 'job.conf'
        conf.write_text(
            f"model = {DEMO_DIR / 'parabola.bngl'} : {DEMO_DIR / 'par1.exp'}\n"
            "fit_type = de\nobjfunc = chi_sq\n"
            "uniform_var = v1__FREE 0 10\n")
        with pytest.raises(NotImplementedError):
            export_job(conf, tmp_path / 'out')

    def test_legacy_data_linkage_is_refused(self, tmp_path):
        # 'Refuse legacy everything' (ADR-0028 Chunk 5c): under edition 2 a legacy data
        # binding (model = X : Y.exp, no experiment:) is refused -- the exporter reads only
        # the new-era experiment:/data: surface.
        conf = tmp_path / 'job.conf'
        conf.write_text(
            f"edition = 2\njob_type = de\nobjective = chi_sq\n"
            f"model = {DEMO_DIR / 'parabola.bngl'} : {DEMO_DIR / 'par1.exp'}\n"
            "uniform_var = v1__FREE 0 10\n")
        with pytest.raises(NotImplementedError, match='new-era'):
            export_job(conf, tmp_path / 'out')

    def test_mixed_legacy_and_new_era_is_refused(self, tmp_path):
        # A legacy mutant alongside the new experiment: surface is refused (it would
        # otherwise be silently ignored on export).
        conf = tmp_path / 'job.conf'
        conf.write_text(
            f"edition = 2\njob_type = de\nobjective = chi_sq\n"
            f"model: {DEMO_DIR / DEMO_MODEL}\n"
            f"experiment: e, data: {DEMO_DIR / 'par1.exp'}\n"
            f"mutant = parabola_v2 m v1*2 : {DEMO_DIR / 'par1.exp'}\n"
            "uniform_var = v1__FREE 0 10\n")
        with pytest.raises(NotImplementedError, match='mix'):
            export_job(conf, tmp_path / 'out')

    def test_modern_edition_without_objective_is_refused(self, tmp_path):
        # New era has no implicit chi_sq default: an edition-2 job must name its objective.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(tmp_path, "uniform_var = v1 0 10\n"),
                       tmp_path / 'out')

    @pytest.mark.parametrize('objfunc', ['neg_bin', 'neg_bin_dynamic', 'score'])
    def test_petab_inexpressible_objective_not_implemented(self, tmp_path, objfunc):
        # neg_bin was removed from PEtab v2; score (the direct_pass successor) is not a
        # likelihood. All are named on the modern `objective` key.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path, f"objective = {objfunc}\nuniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_free_parameter_sigma_objective_not_implemented(self, tmp_path):
        # chi_sq_dynamic's free sigma needs the noise parameter wired into the PEtab
        # parameter table -- a deferred sigma-source path (raised at column classification).
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path, "objective = chi_sq_dynamic\nuniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_legacy_free_marker_free_parameter_is_rejected(self, tmp_path):
        # New-era binds by id (ADR-0034): a free parameter must be a model parameter id.
        # A legacy '*__FREE' name matches no id in the bare model -> the migration error
        # (mirroring config._check_variable_correspondence_modern), with a bind-by-id hint.
        from pybnf.printing import PybnfError
        with pytest.raises(PybnfError, match='__FREE'):
            export_job(_boundary_conf(
                tmp_path, "objective = chi_sq\nuniform_var = v1__FREE 0 10\n"),
                tmp_path / 'out')

    @pytest.mark.parametrize('token', ['kl', 'wasserstein'])
    def test_profile_objective_not_implemented(self, tmp_path, token):
        # Column-joint profile objectives have no per-observable PEtab noise -- they must
        # raise, NOT silently fall through to a default.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path, f"profile_objective = {token}\nuniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_per_observable_noise_model_override_not_implemented(self, tmp_path):
        # A per-observable noise_model override is a later chunk -- raise, not default.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path,
                "noise_model = gaussian, sigma = fix_at 1\n"
                "noise_model x = gaussian, sigma = fix_at 2\n"
                "uniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_mean_centered_noise_model_not_implemented(self, tmp_path):
        # PEtab v2 is median-only; a mean-centered noise model has no representation.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path,
                "noise_model = gaussian, sigma = fix_at 1, location = mean\n"
                "uniform_var = v1 0 10\n"),
                tmp_path / 'out')


class TestCleanModelUnit:

    def test_drops_actions_block_keeps_bare_model_verbatim(self):
        # New-era binds by id (ADR-0034): the model already carries bare ids with real
        # nominals, so clean = strip the actions block and carry the rest verbatim.
        src = (
            "begin model\n begin parameters\n  k1 3\n end parameters\n"
            "end model\n\nbegin actions\n simulate({})\nend actions\n")
        out = clean_model_for_petab(src)
        assert 'k1 3' in out                             # the real nominal carried verbatim
        assert '__FREE' not in out
        assert 'begin actions' not in out and 'simulate' not in out
        assert 'begin model' in out                     # the model body is untouched

    def test_legacy_free_marker_is_rejected(self):
        # A model still carrying a legacy __FREE marker was not modernized; the exporter
        # refuses it (rather than ship a dangling v1__FREE symbol into PEtab) -- ADR-0034.
        from pybnf.printing import PybnfError
        src = "begin model\n begin parameters\n  k1 k1__FREE\n end parameters\nend model\n"
        with pytest.raises(PybnfError, match='__FREE'):
            clean_model_for_petab(src)


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
        return BnglModel.from_file(out / DEMO_MODEL)

    def test_parameter_ids_and_values(self, model):
        # The exported model is carried verbatim (ADR-0034), so the parameter values are
        # the model's real nominals (parabola_v2.bngl: v1 0.5, v2 1, v3 3), not the old
        # synthetic bounds-midpoint the marker-strip used to substitute.
        assert set(model.get_parameter_ids()) == {'v1', 'v2', 'v3'}
        assert model.get_parameter_value('v1') == 0.5
        assert dict(model.get_free_parameter_ids_with_values()) == \
            {'v1': 0.5, 'v2': 1.0, 'v3': 3.0}

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
# Shared fixtures for the new-era condition tests (above) + the surrogate-mapping unit
# tests (below). _PARABOLA2_BNGL has a fit parameter (v1) and a fixed one (s) so a
# condition can perturb each kind; its begin actions block is stripped on export, so it
# doubles as proof the exporter ignores legacy actions in a new-era job. _assert_petab_clean
# is the full-task oracle (ADR-0026).
# ---------------------------------------------------------------------------

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
