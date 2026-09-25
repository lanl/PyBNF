"""Unit tests for the PEtab v2 *exporter* (#407/#423; ADR-0025/0027/0028).

The exporter reads a working PyBNF/BNGL job *on the new-era surface* (model: /
experiment: / data: / condition: / observable:) and serializes it to a PEtab v2 problem.
Its contracts, by strength of oracle:

1. **The external oracle, at model level.** petab's own validation, run on the whole
   problem loaded via ``Problem.from_yaml`` through petab's native ``BnglModel``
   loader (ADR-0026; upstream since petab 0.9.0, #591). The exported ``demo`` problem
   must pass **every**
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
    read_parameter_table,
    write_parameter_table,
)
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter

DEMO_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'demo'
# The new-era (edition 2) demo conf: the PEtab v2 exporter requires a modern config and
# reads the new-era data surface (model: / experiment: / data:), not the legacy linkage
# (ADR-0028 Chunk 5). demo_bng.conf stays legacy for the fitter / install check;
# demo_bng_v2.conf is its fully new-era twin (model: parabola_v2.bngl -- parabola.bngl
# without a begin actions block, since the action is synthesized from the experiment).
DEMO_CONF = DEMO_DIR / 'demo_bng_v2.conf'
DEMO_MODEL = 'parabola_v2.bngl'

# The three row-varying / per-observable round-trip fixtures (ADR-0044/0045, #428): a
# per-observable FormulaSigma (scaling_v2), a row-varying noise id (rowsigma_v2), and a
# row-varying observable scale (obsscale_v2). The importer is oracled in test_petab_import.py;
# here they exercise the closing export half (import -> export -> re-import preserves the fit).
FIXTURE_DIR = Path(__file__).resolve().parents[1] / 'tests' / 'petab_fixtures'
SCALING_DIR = FIXTURE_DIR / 'scaling_v2'
ROWSIGMA_DIR = FIXTURE_DIR / 'rowsigma_v2'
OBSSCALE_DIR = FIXTURE_DIR / 'obsscale_v2'
FIXEDSIGMA_DIR = FIXTURE_DIR / 'fixedsigma_v2'
MULTISIGMA_DIR = FIXTURE_DIR / 'multisigma_v2'
PREDSIGMA_DIR = FIXTURE_DIR / 'predsigma_v2'

# The tutorial's own PEtab priors problem (lesson 15): three parameters whose finite bounds
# TRUNCATE an unbounded family, which the importer can only write as `parameter:` records --
# the shape the exporter used to drop on the floor (#733).
PRIORS_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'tutorial' / '15_petab_priors'


def _tsv_rows(path):
    """Read a TSV into a list of dict rows (a tiny stdlib reader for assertions)."""
    import csv
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _petab_validation_errors(problem_yaml):
    """Load a problem via the real petablint path and return its ERROR-level issues.

    The model-level external oracle: ``Problem.from_yaml`` exercises
    ``model_factory -> BnglModel -> BNG2.pl --check`` through petab's native BNGL
    loader (upstream since petab 0.9.0; #420 Step B, #591), then runs
    **every** ``default_validation_task`` -- the model-cross checks included. Returns
    ``[(task, message), ...]``; empty means a clean problem.
    """
    pytest.importorskip('petab.v2')
    from petab.v2 import Problem
    from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

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

    @pytest.mark.parametrize('keyword,p1,p2', [
        ('uniform_var', 0.0, 10.0), ('loguniform_var', 0.1, 10.0)])
    def test_unbounded_uniform_export_not_implemented(self, keyword, p1, p2):
        # The native 'u' flag turns the reflecting box OFF: the box seeds the first
        # population and the search is then free to leave it. PEtab's lowerBound/upperBound
        # are hard constraints, so writing [p1, p2] into them would export a constraint the
        # job explicitly declined -- refused rather than written (#736).
        with pytest.raises(NotImplementedError, match='unbounded'):
            petab_parameter_row(FreeParameter('k', keyword, p1, p2, bounded=False))
        # the same declaration with its box ON is untouched
        row = petab_parameter_row(FreeParameter('k', keyword, p1, p2, bounded=True))
        assert (row.lower_bound, row.upper_bound) == (p1, p2)

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

    def test_observable_row_inlines_function_body_when_supplied(self):
        # The inlining override (ADR-0035): a function column may carry its body as the
        # observableFormula instead of the bare name; without the override it stays bare.
        row = petab_observable_row('z', 'function', 'normal', ('constant', 1.0),
                                   observable_formula='(a + b)/c')
        assert row.observable_id == 'func_z'
        assert row.observable_formula == '(a + b)/c'
        bare = petab_observable_row('z', 'function', 'normal', ('constant', 1.0))
        assert bare.observable_formula == 'z'         # default is still the bare name

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

    def test_inlining_emits_function_body_observable_stays_bare(self, tmp_path):
        # inline_functions (ADR-0035): the FUNCTION column emits its body as an expression
        # observableFormula; the OBSERVABLE column stays bare (not an algebraic expression).
        pytest.importorskip('petab')
        import sympy as sp
        from petab.v2.math import sympify_petab
        out = tmp_path / 'inl'
        export_job(DEMO_CONF, out, inline_functions=True)
        by_id = {r['observableId']: r for r in _tsv_rows(out / 'observables.tsv')}
        assert by_id['obs_x']['observableFormula'] == 'x'        # observable: still bare
        # func_y's formula is y()'s body (v1*(x^2)+(v2*x)+v3), sympy-equal but inlined.
        inlined = by_id['func_y']['observableFormula']
        assert inlined != 'y'
        assert sp.simplify(sympify_petab(inlined)
                           - sympify_petab('v1*x^2 + v2*x + v3')) == 0

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
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        # petab loads `language: bngl` through its own native BnglModel (upstream
        # since petab 0.9.0; #420 Step B, #591): the loaded model must be petab's.
        problem = Problem.from_yaml(str(exported / 'problem.yaml'))
        assert type(problem.model).__name__ == 'BnglModel'  # the BNGL loader ran
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

    def test_column_mean_noise_formula_skips_unmeasured_points(self, tmp_path_factory):
        """#707: the exported column-mean sigma averages the OBSERVED points only.

        PEtab's own round-trip writes 'nan' for a point an observable does not have
        (see the exporter's measurement rows), so a sparse multi-observable data file is
        exactly what comes back through this path. A plain average over the raw column
        put 'nan' in noiseFormula -- an unusable PEtab problem, and on reimport a sigma
        that matches no objective."""
        import shutil
        src = tmp_path_factory.mktemp('colmean_sparse_src')
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        # par1.exp with x unmeasured on the odd rows -- y keeps its full time course.
        rows = [r for r in (DEMO_DIR / 'par1.exp').read_text().splitlines() if r.strip()]
        sparse = [rows[0]]
        for i, row in enumerate(rows[1:]):
            fields = row.split()
            if i % 2:
                fields[1] = 'nan'          # x unmeasured at this time point
            sparse.append('\t'.join(fields))
        (src / 'par1.exp').write_text('\n'.join(sparse) + '\n')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\nobjective = ave_norm_sos\n'
            f'model: {DEMO_MODEL}\n'
            'experiment: par1, data: par1.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
            'uniform_var = v3 0 10\n')
        out = tmp_path_factory.mktemp('colmean_sparse_out')
        export_job(src / 'job.conf', out)

        data = Data(file_name=str(src / 'par1.exp'))
        by_id = {r['observableId']: r for r in _tsv_rows(out / 'observables.tsv')}
        for oid, col in (('obs_x', 'x'), ('func_y', 'y')):
            column = data[col]
            expected = np.average(column[~np.isnan(column)])
            written = float(by_id[oid]['noiseFormula'])
            assert not np.isnan(written)          # the regression: 'nan' was written here
            assert written == pytest.approx(expected)
        # The sparse column really is sparse -- otherwise this test proves nothing.
        assert np.isnan(data['x']).any()
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

    def test_lnnormal_exports_as_petab_log_normal(self, tmp_path_factory):
        # The explicit natural-log member is an exact mapping; PyBNF's bare lognormal remains
        # log10 and deliberately has no PEtab-v2 home. Make the demo's fitted columns positive,
        # as required by the lognormal support, so the external PEtab oracle validates the result.
        import shutil
        src = tmp_path_factory.mktemp('lnnormal_src')
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        columns = [c for c, _i in sorted(data.cols.items(), key=lambda item: item[1])]
        rows_text = ['# ' + ' '.join(columns)]
        for row in data.data:
            vals = [abs(value) + 1.0 if c not in ('time',) and not c.endswith('_SD') else value
                    for c, value in zip(columns, row)]
            rows_text.append(' '.join(str(value) for value in vals))
        (src / 'positive.exp').write_text('\n'.join(rows_text) + '\n')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\nobjective = lnnormal\nmodel: {DEMO_MODEL}\n'
            'experiment: positive, data: positive.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10\n')
        out = tmp_path_factory.mktemp('lnnormal_out')
        export_job(src / 'job.conf', out)
        rows = _tsv_rows(out / 'observables.tsv')
        assert all(r['noiseDistribution'] == 'log-normal' for r in rows)
        assert all(r['noiseFormula'] == f"noiseParameter1_{r['observableId']}" for r in rows)
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

    def test_whole_fit_formula_sigma_round_trips(self, tmp_path_factory):
        # A whole-fit FormulaSigma (ADR-0044/0045): sigma = an expression over a model
        # parameter. It exports as the observables.tsv noiseFormula verbatim and round-trips
        # export -> import -> re-export byte-for-byte (the import collapses the uniform
        # expression back to a whole-fit noise_model line; ADR-0045).
        import shutil
        from pybnf.petab.import_ import import_job
        expr = '0.1 + 0.05 * v1'
        src = tmp_path_factory.mktemp('formula_src')
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)
        shutil.copy(DEMO_DIR / 'par1.exp', src / 'par1.exp')
        (src / 'job.conf').write_text(
            f'edition = 2\njob_type = de\n'
            f'model: {DEMO_MODEL}\n'
            'experiment: par1, data: par1.exp\n'
            f'noise_model = gaussian, sigma = formula {expr}\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10\n')
        out1 = tmp_path_factory.mktemp('formula_out1')
        export_job(src / 'job.conf', out1)
        rows = _tsv_rows(out1 / 'observables.tsv')
        assert all(r['noiseDistribution'] == 'normal' for r in rows)
        assert all(r['noiseFormula'] == expr for r in rows)      # the expression, verbatim
        assert all(r['noisePlaceholders'] == '' for r in rows)   # no per-measurement placeholder
        assert _petab_validation_errors(out1 / 'problem.yaml') == []

        # Import recovers a whole-fit formula line (the uniform-expression collapse, ADR-0045),
        # and a re-export reproduces the observables table byte-for-byte.
        imp = import_job(out1 / 'problem.yaml', tmp_path_factory.mktemp('formula_imp'))
        conf_text = (imp / 'imported.conf').read_text()
        assert f'noise_model = gaussian, sigma = formula {expr}' in conf_text
        out2 = tmp_path_factory.mktemp('formula_out2')
        export_job(imp / 'imported.conf', out2)
        assert (out2 / 'observables.tsv').read_text() == (out1 / 'observables.tsv').read_text()


# ---------------------------------------------------------------------------
# Per-observable + row-varying export round trip (ADR-0044/0045, #428 Milestone 2): the closing
# half of the per-measurement placeholder frontier. The importer (test_petab_import.py) recovers
# three crafted problems -- a per-observable FormulaSigma (scaling_v2), a row-varying noise id
# (rowsigma_v2), and a row-varying observable scale (obsscale_v2); each now RE-EXPORTS to a valid
# PEtab v2 problem that re-imports to the SAME fit. Export is lossy on naming (a model function y
# the source named obs_y re-exports as func_y; a no-condition experiment loses its name), so the
# oracle is the tightest fit-preserving one rather than byte-equality: petablint-clean re-export +
# re-import scoring identically to the importer's own hand-derived NLL.
# ---------------------------------------------------------------------------

class TestExportRowVaryingRoundTrip:

    # case -> (fixture_dir, model_id, pset, sim trajectory (time, x, y), expected NLL). The pset,
    # trajectory, and expected score mirror each fixture's import oracle in test_petab_import.py.
    CASES = {
        'scaling': (SCALING_DIR, 'scaling_model',
                    {'v1': 0.5, 'v2': 1., 'v3': 3., 'scaling': 3., 'slope': 1.},
                    [[0., -10., 43.], [1., -9., 34.5], [2., -8., 27.]],
                    490.0 + 3 * float(np.log(0.15))),
        'rowsigma': (ROWSIGMA_DIR, 'rowsigma_model',
                     {'v1': 0.5, 'v2': 1., 'v3': 3., 'sd_lo': 0.5, 'sd_hi': 2.},
                     [[0., -10., 44.], [1., -9., 36.5], [2., -8., 29.]],
                     10.5 + 2 * float(np.log(0.5)) + float(np.log(2.0))),
        'obsscale': (OBSSCALE_DIR, 'obsscale_model',
                     {'v1': 0.5, 'v2': 1., 'v3': 3., 's_lo': 2., 's_hi': 3.},
                     [[0., -10., 44.], [1., -9., 36.5], [2., -8., 29.]],
                     28.0),
    }

    def _round_trip(self, case, tmp_path):
        """import fixture -> conf1 -> export -> petab2 -> re-import -> conf2; the artifacts."""
        from pybnf.petab.import_ import import_job
        fixture_dir = self.CASES[case][0]
        imp1, pet2, imp2 = tmp_path / 'imp1', tmp_path / 'pet2', tmp_path / 'imp2'
        import_job(fixture_dir / 'problem.yaml', imp1)
        export_job(imp1 / 'imported.conf', pet2)
        import_job(pet2 / 'problem.yaml', imp2)
        return imp1, pet2, imp2

    @pytest.mark.parametrize('case', list(CASES))
    def test_source_fixture_is_petab_valid(self, case):
        # The source fixtures are valid PEtab v2 (they declare their observable/noise
        # placeholders -- the import path detects placeholders by pattern, but a fixture that
        # claims to be a PEtab problem must pass petab's own validator too).
        assert _petab_validation_errors(self.CASES[case][0] / 'problem.yaml') == []

    @pytest.mark.parametrize('case', list(CASES))
    def test_reexport_is_petab_valid(self, case, tmp_path):
        # The re-exported problem passes petab's full default_validation_tasks (the external
        # oracle): the per-observable noise, the retargeted placeholders, and the per-row
        # observableParameters / noiseParameters columns are all valid PEtab v2.
        _imp1, pet2, _imp2 = self._round_trip(case, tmp_path)
        assert _petab_validation_errors(pet2 / 'problem.yaml') == []

    @pytest.mark.parametrize('case', list(CASES))
    def test_round_trip_preserves_the_fit(self, case, tmp_path, monkeypatch):
        # The end-to-end oracle: the re-imported conf scores a known trajectory identically to the
        # importer's hand-derived NLL, so the row-varying binding (per-row sigma / scale) and the
        # per-observable sigma sources survive export -> re-import unchanged.
        import types

        from pybnf import config as config_mod
        from pybnf.parse import ploop
        _fdir, model, psetd, arr, expected = self.CASES[case]
        _imp1, _pet2, imp2 = self._round_trip(case, tmp_path)
        monkeypatch.chdir(imp2)
        cfg = config_mod.Configuration(
            ploop((imp2 / 'imported.conf').read_text().splitlines(keepends=True)))
        (expname,) = cfg.exp_data[model]              # the experiment name was canonicalized
        sim = Data.from_columns(np.array(arr), ['time', 'x', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in psetd.items()]
        score = cfg.obj.evaluate_multiple({model: {expname: sim}}, cfg.exp_data, pset)
        assert score == pytest.approx(expected)

    def test_per_observable_formula_sigma_round_trips(self, tmp_path):
        # scaling_v2: a per-observable FormulaSigma (the y column) coexisting with a per-observable
        # constant (the obs_sx measurement-model column) -- the per-observable gate the export
        # lifts. Each column carries its own noiseFormula; obs_sx's observableParameters scale was
        # substituted away at import (ADR-0044), so no measurements observableParameters column.
        _imp1, pet2, _imp2 = self._round_trip('scaling', tmp_path)
        noise = {r['observableId']: r['noiseFormula']
                 for r in _tsv_rows(pet2 / 'observables.tsv')}
        assert noise == {'obs_sx': '0.5', 'func_y': '0.05*slope + 0.1'}
        assert 'observableParameters' not in _tsv_rows(pet2 / 'measurements.tsv')[0]

    def test_row_varying_noise_emits_per_row_noise_parameters(self, tmp_path):
        # rowsigma_v2: the per-row sigma id rides the measurements noiseParameters column (the
        # binding-table sidecar back to PEtab, ADR-0045). y is a model FUNCTION, so its
        # observableId is func_y and the kept placeholder is retargeted to match it.
        _imp1, pet2, _imp2 = self._round_trip('rowsigma', tmp_path)
        obs = {r['observableId']: r for r in _tsv_rows(pet2 / 'observables.tsv')}
        assert obs['func_y']['noiseFormula'] == 'noiseParameter1_func_y'
        assert obs['func_y']['noisePlaceholders'] == 'noiseParameter1_func_y'
        rows = [r for r in _tsv_rows(pet2 / 'measurements.tsv')
                if r['observableId'] == 'func_y']
        assert [r['noiseParameters'] for r in rows] == ['sd_lo', 'sd_hi', 'sd_lo']

    def test_row_varying_observable_emits_observable_parameters(self, tmp_path):
        # obsscale_v2: the per-row scale id rides the measurements observableParameters column;
        # the measurement-model observableFormula keeps its placeholder verbatim (its id obs_y is
        # carried through, so no retarget needed).
        _imp1, pet2, _imp2 = self._round_trip('obsscale', tmp_path)
        obs = {r['observableId']: r for r in _tsv_rows(pet2 / 'observables.tsv')}
        assert obs['obs_y']['observableFormula'] == 'observableParameter1_obs_y * y'
        rows = [r for r in _tsv_rows(pet2 / 'measurements.tsv')
                if r['observableId'] == 'obs_y']
        assert [r['observableParameters'] for r in rows] == ['s_lo', 's_hi', 's_lo']


class TestExportNewNoiseShapesRoundTrip:
    """The ADR-0075 noise shapes export back to PEtab and re-import to the same fit (#502) --
    the follow-up half of the import completions the import tests (``test_petab_import.py``)
    already gate. Three fixtures, each the export peer of an import oracle:

    * ``fixedsigma_v2`` (Oliveira) -- a fixed noiseParameters id imported to ``fix_at 2``, which
      exports as an inline constant noiseFormula (the pre-existing constant arm; a regression
      guard that the completion did not disturb it);
    * ``multisigma_v2`` (Fiedler) -- a **multi-token** row-varying noiseParameters product, whose
      sidecar must re-emit BOTH ``noiseParameter${n}`` tokens per row (``s_lo;sig``) -- the
      ``_column_placeholder_series`` single-series gap this closes;
    * ``predsigma_v2`` (Raia) -- a **prediction-dependent** ``sigma = prediction_formula sd_abs +
      sd_rel*y``, which exports verbatim as a noiseFormula that re-imports as prediction_formula
      because ``y`` is a model entity (the new exporter arm).

    The oracle is fit-preserving round trip: the re-imported conf scores the same fixed
    trajectory against each fixture's hand-derived NLL (from ``test_petab_import.py``), so the
    noise source survives export -> re-import unchanged, plus the re-exported problem passes
    petab's own validator.
    """

    # case -> (fixture_dir, model, pset, sim (time, y) trajectory, expected NLL). sd differs by
    # ROW for multisigma (per-gel scale) and predsigma (sigma scales with the prediction), so a
    # dropped token / a sigma-at-the-measurement bug scores differently. Mirrors the import oracle.
    _SIM = [[0., 44.], [1., 36.5], [2., 29.]]           # residuals (1, 2, 2) vs data (43, 34.5, 27)
    _RES = np.array([1., 2., 2.])
    CASES = {
        'fixedsigma': (FIXEDSIGMA_DIR, 'fixedsigma_model',
                       {'v1': .5, 'v2': 1., 'v3': 3.},
                       float(np.sum(_RES ** 2 / (2 * 2. ** 2)))),          # fixed sigma=2, no normalizer
        'multisigma': (MULTISIGMA_DIR, 'multisigma_model',
                       {'v1': .5, 'v2': 1., 'v3': 3., 's_lo': .5, 's_hi': 1., 'sig': 2.},
                       float(np.sum(_RES ** 2 / (2 * np.array([1., 2., 1.]) ** 2)
                                    + np.log([1., 2., 1.])))),             # sigma_i = scale_i * sig
        'predsigma': (PREDSIGMA_DIR, 'predsigma_model',
                      {'v1': .5, 'v2': 1., 'v3': 3., 'sd_abs': .5, 'sd_rel': .1},
                      float(np.sum(_RES ** 2 / (2 * (0.5 + 0.1 * np.array([44., 36.5, 29.])) ** 2)
                                   + np.log(0.5 + 0.1 * np.array([44., 36.5, 29.]))))),  # sigma_i = f(y_sim_i)
    }

    def _round_trip(self, case, tmp_path):
        """import fixture -> conf1 -> export -> petab2 -> re-import -> conf2; the artifacts."""
        from pybnf.petab.import_ import import_job
        fixture_dir = self.CASES[case][0]
        imp1, pet2, imp2 = tmp_path / 'imp1', tmp_path / 'pet2', tmp_path / 'imp2'
        import_job(fixture_dir / 'problem.yaml', imp1)
        export_job(imp1 / 'imported.conf', pet2)
        import_job(pet2 / 'problem.yaml', imp2)
        return imp1, pet2, imp2

    @pytest.mark.parametrize('case', list(CASES))
    def test_source_fixture_is_petab_valid(self, case):
        assert _petab_validation_errors(self.CASES[case][0] / 'problem.yaml') == []

    @pytest.mark.parametrize('case', list(CASES))
    def test_reexport_is_petab_valid(self, case, tmp_path):
        # The re-exported problem passes petab's full default_validation_tasks (the external
        # oracle): the multi-token noiseParameters and the prediction-dependent noiseFormula are
        # valid PEtab v2.
        _imp1, pet2, _imp2 = self._round_trip(case, tmp_path)
        assert _petab_validation_errors(pet2 / 'problem.yaml') == []

    @pytest.mark.parametrize('case', list(CASES))
    def test_round_trip_preserves_the_fit(self, case, tmp_path, monkeypatch):
        # The end-to-end oracle: the re-imported conf scores the fixed trajectory identically to
        # the importer's hand-derived NLL, so the multi-token binding and the prediction-dependent
        # sigma source survive export -> re-import unchanged.
        import types

        from pybnf import config as config_mod
        from pybnf.parse import ploop
        _fdir, model, psetd, expected = self.CASES[case]
        _imp1, _pet2, imp2 = self._round_trip(case, tmp_path)
        monkeypatch.chdir(imp2)
        cfg = config_mod.Configuration(
            ploop((imp2 / 'imported.conf').read_text().splitlines(keepends=True)))
        (expname,) = cfg.exp_data[model]              # the experiment name was canonicalized
        sim = Data.from_columns(np.array(self._SIM), ['time', 'y'], indvar='time')
        pset = [types.SimpleNamespace(name=n, value=v) for n, v in psetd.items()]
        score = cfg.obj.evaluate_multiple({model: {expname: sim}}, cfg.exp_data, pset)
        assert score == pytest.approx(expected)

    def test_multi_token_noise_emits_both_tokens_per_row(self, tmp_path):
        # multisigma_v2: the row-varying multi-token noiseParameters product re-emits BOTH
        # noiseParameter${n} tokens per row, semicolon-joined (the sidecar single-series gap #502
        # closes). y is a model FUNCTION, so its observableId is func_y and both kept placeholders
        # are retargeted to match it.
        _imp1, pet2, _imp2 = self._round_trip('multisigma', tmp_path)
        obs = {r['observableId']: r for r in _tsv_rows(pet2 / 'observables.tsv')}
        assert obs['func_y']['noiseFormula'] == 'noiseParameter1_func_y * noiseParameter2_func_y'
        assert obs['func_y']['noisePlaceholders'] == \
            'noiseParameter1_func_y;noiseParameter2_func_y'
        rows = [r for r in _tsv_rows(pet2 / 'measurements.tsv')
                if r['observableId'] == 'func_y']
        assert [r['noiseParameters'] for r in rows] == ['s_lo;sig', 's_hi;sig', 's_lo;sig']

    def test_prediction_dependent_noise_exports_verbatim_formula(self, tmp_path):
        # predsigma_v2: the prediction-dependent sigma exports as a plain noiseFormula (the direct
        # mirror of a FormulaSigma) -- its coefficients declared estimated, the model entity y kept
        # as a model reference, no per-row noiseParameters. Re-imports as prediction_formula.
        _imp1, pet2, imp2 = self._round_trip('predsigma', tmp_path)
        obs = {r['observableId']: r for r in _tsv_rows(pet2 / 'observables.tsv')}
        assert obs['func_y']['noiseFormula'] == 'sd_abs + sd_rel*y'
        assert obs['func_y']['noisePlaceholders'] == ''
        rows = [r for r in _tsv_rows(pet2 / 'measurements.tsv')
                if r['observableId'] == 'func_y']
        assert [r['noiseParameters'] for r in rows] == ['', '', '']
        assert 'sigma = prediction_formula sd_abs + sd_rel*y' in \
            (imp2 / 'imported.conf').read_text()


# ---------------------------------------------------------------------------
# The fit's declared start point survives the export (#719). #583 made a PEtab
# nominalValue the imported fit's start_point; the reverse direction dropped it twice --
# the conf reader never looked at the start_point lines, and the writer had no
# nominalValue column -- so a round trip silently moved the fit back to a sampled draw.
# ---------------------------------------------------------------------------

def _demo_job(tmp_path, extra='', replace=None):
    """A copy of the demo job whose conf has ``extra`` appended (a start_point line).

    ``replace`` is an optional ``(old, new)`` pair applied to the conf text first, used to
    swap a positional ``uniform_var`` line for the ``parameter:`` record spelling of it.
    """
    import shutil
    job = tmp_path / 'job'
    shutil.copytree(DEMO_DIR, job)
    conf = job / DEMO_CONF.name
    text = conf.read_text()
    if replace is not None:
        old_line, new_line = replace
        assert old_line in text                  # a silent no-op would hollow out the test
        text = text.replace(old_line, new_line)
    conf.write_text(text + extra)
    return conf


class TestStartPointRoundTrip:

    # -- the writer, in isolation --------------------------------------------------

    def test_written_nominal_value_reads_back(self, tmp_path):
        # The unit the loss lived in: a row carrying a nominal must survive the TSV.
        # write_parameter_table built a four-field record and read_parameter_table found
        # no nominalValue column, so the value died between them (#719).
        fp = FreeParameter('v1', 'uniform_var', 0.0, 10.0, value=0.5)
        row = petab_parameter_row(fp)
        path = tmp_path / 'parameters.tsv'
        write_parameter_table([row], path)
        assert 'nominalValue' in path.read_text().splitlines()[0].split('\t')
        back = read_parameter_table(path)[0]
        assert back == row                       # the whole row, not just the nominal
        assert free_parameter_from_row(back).value == 0.5

    def test_no_nominal_keeps_the_four_column_shape(self, tmp_path):
        # The column is optional in PEtab v2 and stays absent when nothing declares a
        # start, so a job with no start point exports byte-for-byte as it did before.
        row = petab_parameter_row(FreeParameter('v1', 'uniform_var', 0.0, 10.0))
        path = tmp_path / 'parameters.tsv'
        write_parameter_table([row], path)
        assert path.read_text() == 'parameterId\testimate\tlowerBound\tupperBound\nv1\ttrue\t0\t10\n'

    def test_partial_nominals_write_blank_cells(self, tmp_path):
        # A start point is partial by design (ADR-0117): declaring one parameter's start
        # must not invent a nominal for the others. The column appears once any row needs
        # it, and the rows that do not carry one write an empty cell, which reads back None.
        rows = [PetabParameterRow('a', True, 0., 10., nominal_value=None),
                PetabParameterRow('b', True, 0., 10., nominal_value=4.),
                PetabParameterRow('c', True, 0., 10., nominal_value=None)]
        path = tmp_path / 'parameters.tsv'
        write_parameter_table(rows, path)
        assert [r.nominal_value for r in read_parameter_table(path)] == [None, 4.0, None]

    def test_nominal_and_prior_columns_coexist(self, tmp_path):
        # Both optional chunks at once: nominalValue precedes the prior pair (PEtab v2's
        # own column order), and neither displaces the other's cells.
        fp = FreeParameter('k', 'loguniform_var', 1e-3, 1e3, value=1.0)
        path = tmp_path / 'parameters.tsv'
        write_parameter_table([petab_parameter_row(fp)], path)
        header = path.read_text().splitlines()[0].split('\t')
        assert header == ['parameterId', 'estimate', 'lowerBound', 'upperBound',
                          'nominalValue', 'priorDistribution', 'priorParameters']
        (back,) = read_parameter_table(path)
        assert (back.nominal_value, back.prior_distribution) == (1.0, 'log-uniform')
        assert back.prior_parameters == pytest.approx((1e-3, 1e3))

    # -- the conf reader ------------------------------------------------------------

    def test_conf_start_point_becomes_a_nominal_value(self, tmp_path):
        # The plain "publish my job as PEtab" path: a native conf's start_point line is the
        # fit's declared start, and it exports as that parameter's nominalValue. The
        # parameters it says nothing about keep an empty cell.
        export_job(_demo_job(tmp_path, '\nstart_point = v1 0.7\n'), tmp_path / 'out')
        nominal = {r['parameterId']: r['nominalValue']
                   for r in _tsv_rows(tmp_path / 'out' / 'parameters.tsv')}
        assert nominal == {'v1': '0.7', 'v2': '', 'v3': ''}

    def test_conf_without_a_start_point_writes_no_nominal_column(self, tmp_path):
        # The unchanged path: the demo conf declares no start, so the exported table keeps
        # its four-column shape rather than gaining a column of blanks.
        export_job(DEMO_CONF, tmp_path / 'out')
        assert 'nominalValue' not in (tmp_path / 'out' / 'parameters.tsv').read_text()

    def test_out_of_box_start_point_is_refused(self, tmp_path):
        # PEtab cannot express a nominalValue outside the row's own bounds either -- the
        # importer refuses exactly that on the way back in. Refused as a PybnfError, not the
        # bare OutOfBoundsException, which pybnf.main reports as "an unknown error" (#583).
        with pytest.raises(PybnfError, match='out of bounds'):
            export_job(_demo_job(tmp_path, '\nstart_point = v1 42\n'), tmp_path / 'out')

    def test_start_point_for_an_undeclared_parameter_is_refused(self, tmp_path):
        # Silently dropping it is the failure this path exists to remove; config.py refuses
        # the same line when the job is run.
        with pytest.raises(PybnfError, match='unknown parameter'):
            export_job(_demo_job(tmp_path, '\nstart_point = nope 1\n'), tmp_path / 'out')

    def test_partial_nominal_problem_is_petab_valid(self, tmp_path):
        # The external oracle on the shape the fixtures never produce: an estimated row with
        # a blank nominalValue beside rows that carry one.
        pytest.importorskip('petab.v2')
        export_job(_demo_job(tmp_path, '\nstart_point = v1 0.7\n'), tmp_path / 'out')
        assert _petab_validation_errors(tmp_path / 'out' / 'problem.yaml') == []

    # -- the full round trip, the harm as reported ----------------------------------

    def test_petab_nominal_survives_import_export_reimport(self, tmp_path):
        # fixedsigma_v2 ships nominalValue 0.5/1/3. The first import turns them into
        # start_point lines (#583); before #719 the export dropped them and the second
        # import produced a conf with none, so the fit began from a sampled draw instead of
        # the point the published problem states -- with no warning and no error.
        from pybnf.petab.import_ import import_job
        imp1, pet2, imp2 = tmp_path / 'imp1', tmp_path / 'pet2', tmp_path / 'imp2'
        import_job(FIXEDSIGMA_DIR / 'problem.yaml', imp1)
        export_job(imp1 / 'imported.conf', pet2)
        import_job(pet2 / 'problem.yaml', imp2)

        nominal = {r['parameterId']: r['nominalValue']
                   for r in _tsv_rows(pet2 / 'parameters.tsv')}
        assert nominal == {'v1': '0.5', 'v2': '1', 'v3': '3'}

        def starts(conf):
            return sorted(line.strip() for line in conf.read_text().splitlines()
                          if line.startswith('start_point'))
        assert starts(imp2 / 'imported.conf') == starts(imp1 / 'imported.conf')
        assert starts(imp2 / 'imported.conf') == [
            'start_point = v1 0.5', 'start_point = v2 1', 'start_point = v3 3']

    def test_reimported_conf_starts_the_fit_at_the_published_point(self, tmp_path, monkeypatch):
        # The user-visible fact behind the tsv cells: the twice-round-tripped conf loads to a
        # Configuration whose resolved start point is the published one. Configuration is the
        # consumer every start-point optimizer reads (ADR-0117), so this is the end of the chain.
        from pybnf import config as config_mod
        from pybnf.parse import ploop
        from pybnf.petab.import_ import import_job
        imp1, pet2, imp2 = tmp_path / 'imp1', tmp_path / 'pet2', tmp_path / 'imp2'
        import_job(FIXEDSIGMA_DIR / 'problem.yaml', imp1)
        export_job(imp1 / 'imported.conf', pet2)
        import_job(pet2 / 'problem.yaml', imp2)
        monkeypatch.chdir(imp2)
        cfg = config_mod.Configuration(
            ploop((imp2 / 'imported.conf').read_text().splitlines(keepends=True)))
        assert cfg.start_point == {'v1': 0.5, 'v2': 1.0, 'v3': 3.0}



# ---------------------------------------------------------------------------
# The edition-2 `parameter:` record is a free-parameter declaration the exporter reads
# (#733). It used to match no branch of the `(keyword, name)` filter, so it was skipped
# rather than refused and the whole parameter vanished from the exported problem -- taking
# every TRUNCATED prior with it, since a record is the only grammar carrying lower/upper
# and is exactly what the importer emits for one (ADR-0020/0043/0047).
# ---------------------------------------------------------------------------

_V1_VAR = 'uniform_var = v1 0 10'


class TestExportParameterRecord:

    def test_record_exports_the_same_row_as_the_positional_line(self, tmp_path):
        # The two-adapter proof on the declaration grammars: one parameter written both
        # ways must reach PEtab as the same row. Before #733 the record's row was simply
        # absent from the table, with no warning and no exception.
        export_job(DEMO_CONF, tmp_path / 'positional')
        export_job(_demo_job(tmp_path, replace=(
            _V1_VAR, 'parameter: v1, prior: uniform, lower: 0, upper: 10')),
            tmp_path / 'record')
        assert (tmp_path / 'record' / 'parameters.tsv').read_text() == \
            (tmp_path / 'positional' / 'parameters.tsv').read_text()

    def test_record_only_conf_exports_every_parameter(self, tmp_path):
        # A conf written entirely on the edition-2 surface used to export nothing at all:
        # every record was skipped, so `free_params` came back empty and the user got
        # "No exportable free parameters found", naming only the *_var keywords -- pointing
        # away from the syntax the edition-2 docs teach.
        conf = _demo_job(tmp_path, replace=(
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10',
            'parameter: v1, prior: uniform, lower: 0, upper: 10\n'
            'parameter: v2, prior: uniform, lower: 0, upper: 10\n'
            'parameter: v3, prior: uniform, lower: 0, upper: 10'))
        export_job(conf, tmp_path / 'out')
        rows = _tsv_rows(tmp_path / 'out' / 'parameters.tsv')
        assert {r['parameterId'] for r in rows} == {'v1', 'v2', 'v3'}
        assert all((r['lowerBound'], r['upperBound']) == ('0', '10') for r in rows)

    def test_records_and_var_lines_keep_declaration_order(self, tmp_path):
        # The two spellings may be mixed, and the table follows the conf's own order --
        # the record is read in the same pass, not appended after.
        conf = _demo_job(tmp_path, replace=(
            'uniform_var = v2 0 10',
            'parameter: v2, prior: uniform, lower: 0, upper: 10'))
        export_job(conf, tmp_path / 'out')
        assert [r['parameterId'] for r in _tsv_rows(tmp_path / 'out' / 'parameters.tsv')] == \
            ['v1', 'v2', 'v3']

    # -- truncated priors, the shape only a record can carry -------------------------

    def test_truncated_priors_survive_import_export_reimport(self, tmp_path):
        # The tutorial's own PEtab priors problem: kon (log-normal), koff (gamma) and R0
        # (normal) all carry finite bounds that TRUNCATE an unbounded family, so the
        # importer writes each as a `parameter:` record. Before #733 the export kept only
        # L0 -- the one plain uniform_var -- and silently dropped the other three, which is
        # three quarters of a published problem.
        from pybnf.petab.import_ import import_job
        imp1, pet2, imp2 = tmp_path / 'imp1', tmp_path / 'pet2', tmp_path / 'imp2'
        import_job(PRIORS_DIR / 'problem.yaml', imp1)
        export_job(imp1 / 'imported.conf', pet2)
        import_job(pet2 / 'problem.yaml', imp2)

        rows = {r['parameterId']: r for r in _tsv_rows(pet2 / 'parameters.tsv')}
        assert set(rows) == {'kon', 'koff', 'R0', 'L0'}
        assert rows['R0']['priorDistribution'] == 'normal'
        assert (rows['R0']['lowerBound'], rows['R0']['upperBound']) == ('1', '100')
        assert rows['koff']['priorDistribution'] == 'gamma'
        assert rows['L0']['priorDistribution'] == ''       # a plain box needs no prior

        # and the declarations come back identical, record grammar included
        def declarations(conf):
            return [line.strip() for line in conf.read_text().splitlines()
                    if line.startswith(('parameter:', 'uniform_var', 'loguniform_var'))]
        assert declarations(imp2 / 'imported.conf') == declarations(imp1 / 'imported.conf')
        assert sum(d.startswith('parameter:') for d in declarations(imp1 / 'imported.conf')) == 3

    def test_reexported_truncated_prior_problem_is_petab_valid(self, tmp_path):
        # The external oracle on a shape the exporter could not previously emit at all.
        pytest.importorskip('petab.v2')
        from pybnf.petab.import_ import import_job
        import_job(PRIORS_DIR / 'problem.yaml', tmp_path / 'imp1')
        export_job(tmp_path / 'imp1' / 'imported.conf', tmp_path / 'pet2')
        # the count first: a table that dropped three of the four parameters is still
        # perfectly valid PEtab, so validity alone is not the guard here
        assert len(_tsv_rows(tmp_path / 'pet2' / 'parameters.tsv')) == 4
        assert _petab_validation_errors(tmp_path / 'pet2' / 'problem.yaml') == []

    def test_half_bounded_truncation_writes_an_explicit_infinity(self, tmp_path):
        # ADR-0047: one finite wall and an open side, the ub->inf limit of the two-sided
        # fold. It reaches PEtab as a finite lowerBound and an infinite upperBound.
        export_job(_demo_job(tmp_path, replace=(
            _V1_VAR, 'parameter: v1, prior: normal, mean: 1, sd: 2, lower: 0, upper: inf')),
            tmp_path / 'out')
        row = {r['parameterId']: r for r in _tsv_rows(tmp_path / 'out' / 'parameters.tsv')}['v1']
        assert (row['lowerBound'], row['upperBound']) == ('0', 'inf')
        assert (row['priorDistribution'], row['priorParameters']) == ('normal', '1;2')

    # -- the start point, through the record spelling (#719) --------------------------

    def test_record_initial_value_becomes_a_nominal_value(self, tmp_path):
        # initial_value: is the record's spelling of the start point, so it lands in the
        # same nominalValue cell a `start_point` line does.
        export_job(_demo_job(tmp_path, replace=(
            _V1_VAR,
            'parameter: v1, prior: uniform, lower: 0, upper: 10, initial_value: 0.7')),
            tmp_path / 'out')
        nominal = {r['parameterId']: r['nominalValue']
                   for r in _tsv_rows(tmp_path / 'out' / 'parameters.tsv')}
        assert nominal == {'v1': '0.7', 'v2': '', 'v3': ''}

    def test_agreeing_start_point_spellings_are_accepted(self, tmp_path):
        # Both spellings for one parameter is fine when they say the same thing -- a record
        # naming its own start plus a start_point line restating it (ADR-0117).
        export_job(_demo_job(
            tmp_path, extra='\nstart_point = v1 0.7\n', replace=(
                _V1_VAR,
                'parameter: v1, prior: uniform, lower: 0, upper: 10, initial_value: 0.7')),
            tmp_path / 'out')
        row = {r['parameterId']: r for r in _tsv_rows(tmp_path / 'out' / 'parameters.tsv')}
        assert row['v1']['nominalValue'] == '0.7'

    def test_contradictory_start_point_spellings_are_refused(self, tmp_path):
        # Disagreement has no defensible winner, so it is refused rather than resolved --
        # the rule Configuration._load_start_point applies when the job is run.
        conf = _demo_job(tmp_path, extra='\nstart_point = v1 0.9\n', replace=(
            _V1_VAR,
            'parameter: v1, prior: uniform, lower: 0, upper: 10, initial_value: 0.7'))
        with pytest.raises(PybnfError, match='contradictory start point'):
            export_job(conf, tmp_path / 'out')

    def test_out_of_box_initial_value_is_refused(self, tmp_path):
        # As a PybnfError, not the bare OutOfBoundsException the FreeParameter constructor
        # raises -- which pybnf.main reports as "an unknown error ... please report this
        # bug" on a config the user wrote (#583).
        conf = _demo_job(tmp_path, replace=(
            _V1_VAR,
            'parameter: v1, prior: uniform, lower: 0, upper: 10, initial_value: 42'))
        with pytest.raises(PybnfError, match='out of bounds'):
            export_job(conf, tmp_path / 'out')

    # -- the boundaries a record reaches by a different spelling ----------------------

    def test_no_prior_record_is_refused(self, tmp_path):
        # A record with no prior and no box is the edition-2 spelling of `var` -- a flat
        # improper prior, which is not a PEtab probability family. Refused in code, which
        # is the exporter's contract for everything it cannot write.
        conf = _demo_job(tmp_path, replace=(_V1_VAR, 'parameter: v1, initial_value: 3'))
        with pytest.raises(NotImplementedError, match='no-prior point start'):
            export_job(conf, tmp_path / 'out')

    @pytest.mark.parametrize('record,built', [
        ('parameter: v1, prior: normal, parameter_scale: ln, mean: 0, sd: 1',
         'lnnormal_var'),                        # PEtab has no natural-log sampling scale
        ('parameter: v1, prior: student_t, df: 3, location: 1, scale: 2',
         'student_t_var'),                       # no three-parameter PEtab family
        ('parameter: v1, prior: cauchy, parameter_scale: log10, location: 0, scale: 1',
         'logcauchy_var'),                       # PEtab defines no log- form for cauchy
    ])
    def test_petab_inexpressible_record_prior_is_refused(self, tmp_path, record, built):
        # A record resolves to an ordinary *_var keyword, so it meets the same boundary the
        # positional line does -- and the message names the keyword it actually built.
        conf = _demo_job(tmp_path, replace=(_V1_VAR, record))
        with pytest.raises(NotImplementedError, match=built):
            export_job(conf, tmp_path / 'out')


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

    def test_undefined_referenced_condition_raises(self, tmp_path_factory):
        from pybnf.printing import PybnfError
        src = self._src(tmp_path_factory, 'undefcond')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: par1, condition: nope, data: par1.exp\n'
            + self._PARAMS)
        with pytest.raises(PybnfError, match='nope'):
            export_job(src / 'job.conf', src / 'out')


# ---------------------------------------------------------------------------
# ADR-0046: a dose-response (parameter_scan) experiment -> N steady-state Conditions +
# Experiments measured at time=inf (the dual of the time-course export).
# ---------------------------------------------------------------------------

# A tiny birth-death model whose swept parameter L (a dose) and fitted parameter kd are
# both model parameters; the observable resp is a model observable. The export only reads
# the model's entity surface (params / observables), so the steady-state physics is inert
# here -- it is the round-trip recovery test that exercises the actual KINSOL scan.
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


class TestExportDoseResponse:

    def _src(self, tmp_path_factory, name='dr'):
        src = tmp_path_factory.mktemp(name)
        (src / 'dr.bngl').write_text(_DR_MODEL)
        (src / 'dose.exp').write_text('# L resp\n1\t0.5\n2\t1\n5\t2.5\n')
        return src

    _HEAD = 'edition = 2\njob_type = de\nobjective = sos\nmodel: dr.bngl\n'
    _PARAMS = 'uniform_var = kd 0.1 10\n'

    def test_steady_state_scan_exports_conditions_and_inf_measurements(self, tmp_path_factory):
        # No t_end: => steady state. Each dose row -> a Condition setting L + an Experiment;
        # the observable column -> measurements at time=inf (PEtab's steady-state convention).
        src = self._src(tmp_path_factory, 'drss')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: dr, data: dose.exp\n' + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)

        conds = _tsv_rows(out / 'conditions.tsv')
        assert [(c['conditionId'], c['targetId'], c['targetValue']) for c in conds] == [
            ('cond_dr_0', 'L', '1'), ('cond_dr_1', 'L', '2'), ('cond_dr_2', 'L', '5')]
        exps = _tsv_rows(out / 'experiments.tsv')
        assert [e['experimentId'] for e in exps] == ['dr_0', 'dr_1', 'dr_2']
        assert [e['conditionId'] for e in exps] == ['cond_dr_0', 'cond_dr_1', 'cond_dr_2']

        meas = _tsv_rows(out / 'measurements.tsv')
        assert all(m['time'] == 'inf' for m in meas)          # steady state => time=inf
        assert {m['observableId'] for m in meas} == {'obs_resp'}
        assert [(m['experimentId'], m['measurement']) for m in meas] == [
            ('dr_0', '0.5'), ('dr_1', '1'), ('dr_2', '2.5')]
        # The swept-axis column L is the scan axis -- never a measurement.
        assert 'obs_L' not in {m['observableId'] for m in meas}
        _assert_petab_clean(out)

    def test_fixed_endpoint_scan_uses_finite_measurement_time(self, tmp_path_factory):
        # An explicit t_end: makes it a fixed-endpoint scan -> a finite measurement time.
        src = self._src(tmp_path_factory, 'drfixed')
        (src / 'job.conf').write_text(
            self._HEAD + 'experiment: dr, type: parameter_scan, t_end: 500, data: dose.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        meas = _tsv_rows(out / 'measurements.tsv')
        assert all(m['time'] == '500' for m in meas)
        _assert_petab_clean(out)

    def test_scan_with_named_condition_raises(self, tmp_path_factory):
        # A dose-response already makes each dose its own condition; combining it with a
        # named condition has no export route (ADR-0046).
        src = self._src(tmp_path_factory, 'drcond')
        (src / 'job.conf').write_text(
            self._HEAD + 'condition: c, perturbations: kd * 2\n'
            'experiment: dr, type: parameter_scan, condition: c, data: dose.exp\n'
            + self._PARAMS)
        with pytest.raises(NotImplementedError, match='dose'):
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


# ---------------------------------------------------------------------------
# ADR-0076: a parameter-valued condition targetValue -- a per-condition estimated initial
# condition. A condition that sets a fixed model entity to the value of a FREE parameter
# (``s = s_A``) exports as ``targetValue = s_A`` (PEtab-legal: the referenced id is a
# parameter-table entry, the target is not, so the surrogate split is not needed).
# ---------------------------------------------------------------------------

def _write_param_ref_condition_fixture(d):
    """A new-era job whose condition sets the fixed model parameter ``s`` to the value of a
    free parameter ``s_A`` (bound to no model entity -- the Bertozzi/Bruno per-condition
    estimated-IC shape), plus a wildtype and a conditioned experiment."""
    (d / 'parabola2.bngl').write_text(_PARABOLA2_BNGL)
    (d / 'wt.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n1\t-9\t69\t1\t1\n')
    (d / 'ca.exp').write_text(
        '# time x y x_SD y_SD\n0\t-10\t172\t1\t1\n1\t-9\t138\t1\t1\n')
    conf = d / 'job.conf'
    conf.write_text(
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        'model: parabola2.bngl\n'
        'condition: cA, perturbations: s = s_A\n'
        'experiment: wt, data: wt.exp\n'
        'experiment: ea, condition: cA, data: ca.exp\n'
        'uniform_var = v1 0 10\nuniform_var = v2 0 10\n'
        'uniform_var = v3 0 10\nuniform_var = s_A 0 10\n')
    return conf


class TestExportParamRefCondition:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = tmp_path_factory.mktemp('param_ref_cond')
        conf = _write_param_ref_condition_fixture(src)
        out = src / 'petab'
        export_job(conf, out)
        return out

    def test_condition_cell_is_the_referenced_parameter(self, exported):
        rows = _tsv_rows(exported / 'conditions.tsv')
        cells = {(r['conditionId'], r['targetId']): r['targetValue'] for r in rows}
        # The fixed target `s` is set to the free-parameter id verbatim (no surrogate split:
        # `s` is not in the parameter table, `s_A` is not a condition target).
        assert cells[('cond_cA', 's')] == 's_A'

    def test_referenced_param_is_an_estimated_parameter(self, exported):
        ids = {r['parameterId'] for r in _tsv_rows(exported / 'parameters.tsv')}
        # s_A binds no model entity, yet is admitted as an estimated parameter (a nuisance the
        # condition references), and s (the fixed target) never enters the parameter table.
        assert 's_A' in ids and 's' not in ids

    def test_full_petab_validation_is_clean(self, exported):
        _assert_petab_clean(exported)


# ---------------------------------------------------------------------------
# ADR-0052 (#441, Phase 2): a new-era pre-equilibration experiment (preequilibrate:) ->
# a PEtab v2 TWO-PERIOD Experiment: a leading time=-inf steady-state period under the
# pre-equilibration condition + a time=0 measurement period under the measurement condition,
# plus the two conditions. The measurements are tagged by experimentId at their data times
# (>= 0); the -inf equilibration period carries none. (The fitter/recovery of the actual
# carry-over physics lives in test_preequilibration.py / test_recovery.py; export reads only
# the entity surface, so the steady-state physics is inert here.)
# ---------------------------------------------------------------------------

# A birth-death model whose decay is gated by a 0/1 flag (the receptor func()*Ligand_isPresent
# idiom): flag is a FIXED model parameter the conditions perturb (so M is empty -- no surrogate
# split), k is the bare-id fit parameter, A_tot is a model observable.
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


class TestExportPreequilibration:

    def _src(self, tmp_path_factory, name='preequil'):
        src = tmp_path_factory.mktemp(name)
        (src / 'm.bngl').write_text(_PREEQUIL_MODEL)
        (src / 'relax.exp').write_text('# time A_tot\n0\t10\n1\t6\n2\t4\n')
        return src

    _HEAD = 'edition = 2\njob_type = de\nobjective = sos\nmodel: m.bngl\n'
    _PARAMS = 'uniform_var = k 0.1 10\n'

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = self._src(tmp_path_factory)
        (src / 'job.conf').write_text(
            self._HEAD
            + 'condition: pre,  perturbations: flag = 0\n'
            + 'condition: meas, perturbations: flag = 1\n'
            + 'experiment: relax, preequilibrate: pre, condition: meas, data: relax.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        return out

    def test_experiments_table_is_two_periods_in_order(self, exported):
        # The -inf pre-equilibration period precedes the time=0 measurement period (ADR-0052).
        rows = _tsv_rows(exported / 'experiments.tsv')
        assert [(r['experimentId'], r['time'], r['conditionId']) for r in rows] == [
            ('relax', '-inf', 'cond_pre'),
            ('relax', '0', 'cond_meas')]

    def test_both_conditions_are_emitted(self, exported):
        rows = _tsv_rows(exported / 'conditions.tsv')
        assert {(r['conditionId'], r['targetId'], r['targetValue']) for r in rows} == {
            ('cond_pre', 'flag', '0'), ('cond_meas', 'flag', '1')}

    def test_measurements_are_tagged_by_the_experiment(self, exported):
        meas = _tsv_rows(exported / 'measurements.tsv')
        assert {m['experimentId'] for m in meas} == {'relax'}
        assert {m['observableId'] for m in meas} == {'obs_A_tot'}
        # the equilibration (-inf) period carries no measurement: every row is at a data time >= 0
        assert all(float(m['time']) >= 0 for m in meas)
        assert sorted(float(m['time']) for m in meas) == [0.0, 1.0, 2.0]

    def test_full_petab_validation_is_clean(self, exported):
        assert _petab_validation_errors(exported / 'problem.yaml') == []

    def test_wash_out_without_measurement_condition_uses_an_empty_measurement_period(
            self, tmp_path_factory):
        # preequilibrate: but no measurement condition: -> equilibrate under the named condition,
        # then measure at the model default (an empty conditionId on the time=0 period). M is
        # empty (flag is a fixed param), so no base condition is needed -- petablint-clean.
        src = self._src(tmp_path_factory, 'washout')
        (src / 'job.conf').write_text(
            self._HEAD
            + 'condition: pre, perturbations: flag = 0\n'
            + 'experiment: relax, preequilibrate: pre, data: relax.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        rows = _tsv_rows(out / 'experiments.tsv')
        assert [(r['experimentId'], r['time'], r['conditionId']) for r in rows] == [
            ('relax', '-inf', 'cond_pre'),
            ('relax', '0', '')]
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_fit_parameter_perturbation_in_preequilibration_uses_the_surrogate_split(
            self, tmp_path_factory):
        # A pre-equilibration condition perturbing a FIT parameter (k) takes the ADR-0027
        # surrogate-base <p>__REF split on the multi-period shape (#443): k is removed from the
        # parameter table (renamed k__REF), the pre-equilibration period sets k = 0.5 (an
        # absolute set on the surrogate), and the measurement period re-pins k = k__REF (its
        # base value) alongside its own fixed-param perturbation.
        src = self._src(tmp_path_factory, 'preequil_fit')
        (src / 'job.conf').write_text(
            self._HEAD
            + 'condition: pre,  perturbations: k = 0.5\n'
            + 'condition: meas, perturbations: flag = 1\n'
            + 'experiment: relax, preequilibrate: pre, condition: meas, data: relax.exp\n'
            + self._PARAMS)
        out = src / 'out'
        export_job(src / 'job.conf', out)
        # k is fit-and-perturbed -> the parameter table carries the surrogate k__REF, not k.
        pids = {r['parameterId'] for r in _tsv_rows(out / 'parameters.tsv')}
        assert 'k__REF' in pids and 'k' not in pids
        conds = {(r['conditionId'], r['targetId'], r['targetValue'])
                 for r in _tsv_rows(out / 'conditions.tsv')}
        assert conds == {
            ('cond_pre', 'k', '0.5'),          # surrogate absolute set on the equilibration period
            ('cond_meas', 'k', 'k__REF'),      # re-pinned at base on the measurement period
            ('cond_meas', 'flag', '1')}        # the measurement period's fixed-param perturbation
        exps = _tsv_rows(out / 'experiments.tsv')
        assert [(e['experimentId'], e['time'], e['conditionId']) for e in exps] == [
            ('relax', '-inf', 'cond_pre'),
            ('relax', '0', 'cond_meas')]
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_wash_out_with_a_fit_parameter_re_pins_M_via_the_synthesized_base(
            self, tmp_path_factory):
        # A wash-out (no measurement condition:) whose pre-equilibration condition perturbs a FIT
        # parameter (k) makes M non-empty, so the measurement period must re-pin k at base. With
        # no user-named measurement condition the exporter synthesizes cond_wildtype = {k =
        # k__REF} and the time=0 period references it (#443) -- petablint-clean, and the importer
        # reads cond_wildtype back as a wash-out (no condition:).
        src = self._src(tmp_path_factory, 'washout_fit')
        (src / 'job.conf').write_text(
            self._HEAD
            + 'condition: pre, perturbations: k = 0.5\n'
            + 'experiment: relax, preequilibrate: pre, data: relax.exp\n'
            + self._PARAMS)
        out = src / 'out'
        export_job(src / 'job.conf', out)
        pids = {r['parameterId'] for r in _tsv_rows(out / 'parameters.tsv')}
        assert 'k__REF' in pids and 'k' not in pids
        conds = {(r['conditionId'], r['targetId'], r['targetValue'])
                 for r in _tsv_rows(out / 'conditions.tsv')}
        assert conds == {
            ('cond_pre', 'k', '0.5'),
            ('cond_wildtype', 'k', 'k__REF')}   # synthesized base re-pins M on the measurement period
        exps = _tsv_rows(out / 'experiments.tsv')
        assert [(e['experimentId'], e['time'], e['conditionId']) for e in exps] == [
            ('relax', '-inf', 'cond_pre'),
            ('relax', '0', 'cond_wildtype')]
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_preequilibrated_scan_perturbing_a_fit_parameter_is_refused(self, tmp_path_factory):
        # A pre-equilibrated dose-response (ADR-0062) requires an empty surrogate set M: the
        # surrogate split is not yet combined with the multi-condition dose period, so a
        # pre-equilibration/wash condition that perturbs a FIT parameter (k) is refused with a
        # clear boundary (the swept L is a model param, not fit; k is the fit knob it perturbs).
        src = self._src(tmp_path_factory, 'preequil_scan_fit')
        (src / 'dose.exp').write_text('# L A_tot\n1\t1\n2\t2\n4\t4\n')
        (src / 'job.conf').write_text(
            self._HEAD
            + 'condition: pre, perturbations: k = 0.5\n'
            + 'experiment: relax, preequilibrate: pre, type: parameter_scan, data: dose.exp\n'
            + self._PARAMS)
        with pytest.raises(NotImplementedError, match='empty surrogate set|fit and perturbed'):
            export_job(src / 'job.conf', src / 'out')

    def test_preequilibration_with_a_species_wash_exports(self, tmp_path_factory):
        # A plain (time-course) pre-equilibration whose measurement condition is a species
        # setConcentration wash (ADR-0062): the wash target is a species amount emitted via the
        # mapping table, not a bogus parameter target. The model gains a seed species Lig() the
        # wash zeroes; petablint-clean.
        src = tmp_path_factory.mktemp('preequil_species')
        (src / 'm.bngl').write_text(_PREEQUIL_MODEL.replace(
            'begin molecule types\n  A()\nend molecule types',
            'begin molecule types\n  A()\n  Lig()\nend molecule types').replace(
            'begin seed species\n  A() 10\nend seed species',
            'begin seed species\n  A() 10\n  Lig() 5\nend seed species'))
        (src / 'relax.exp').write_text('# time A_tot\n0\t10\n1\t6\n2\t4\n')
        (src / 'job.conf').write_text(
            self._HEAD
            + 'condition: pre,  perturbations: flag = 0\n'
            + 'condition: wash, perturbations: "Lig()" = 0\n'
            + 'experiment: relax, preequilibrate: pre, condition: wash, data: relax.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        maps = _tsv_rows(out / 'mapping.tsv')
        assert [(m['petabEntityId'], m['modelEntityId']) for m in maps] == [('species_Lig', 'Lig()')]
        conds = {(r['conditionId'], r['targetId'], r['targetValue'])
                 for r in _tsv_rows(out / 'conditions.tsv')}
        assert ('cond_wash', 'species_Lig', '0') in conds
        assert _petab_validation_errors(out / 'problem.yaml') == []


# ---------------------------------------------------------------------------
# ADR-0062 (#477): a new-era PRE-EQUILIBRATED dose-response (preincubate -> wash -> dose-scan) ->
# N two-period PEtab v2 Experiments (a -inf pre-equilibration period + a multi-condition
# measurement period applying the shared wash condition AND a per-dose swept-parameter condition),
# with species setConcentration wash targets aliased through the mapping table. The combination of
# ADR-0052 (pre-equilibration) and ADR-0046 (dose-response).
# ---------------------------------------------------------------------------

# A birth-death model whose swept parameter L (a dose) and fitted parameter kd are model
# parameters; A()/B() are seed species the incubate/wash setConcentration targets. resp = A().
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


class TestExportPreequilibratedDoseResponse:

    def _src(self, tmp_path_factory, name='pdr'):
        src = tmp_path_factory.mktemp(name)
        (src / 'm.bngl').write_text(_PDR_MODEL)
        (src / 'dose.exp').write_text('# L resp\n1\t0.5\n2\t1\n5\t2.5\n')
        return src

    _HEAD = 'edition = 2\njob_type = de\nobjective = sos\nmodel: m.bngl\n'
    _PARAMS = 'uniform_var = kd 0.1 10\n'
    # incubate: a species set to a NUMBER (the -inf period is subject to CheckInitialChangeSymbols,
    # so its target values must be numeric / parameter-table only). wash: a species zeroed AND a
    # species set to a param-EXPRESSION in the swept parameter (the dose-tracking competitor idiom)
    # -- the measurement period is unconstrained.
    _CONDS = ('condition: incubate, perturbations: "A()" = 100\n'
              'condition: wash, perturbations: "A()" = 0, "B()" = L*kd\n')

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = self._src(tmp_path_factory)
        (src / 'job.conf').write_text(
            self._HEAD + self._CONDS
            + 'experiment: scan, preequilibrate: incubate, condition: wash, '
              'type: parameter_scan, t_end: 500, data: dose.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        return out

    def test_each_dose_is_a_two_period_multi_condition_experiment(self, exported):
        # Per dose: a -inf pre-equilibration period (cond_incubate) + a measurement period applying
        # BOTH the shared wash and the per-dose swept-parameter condition (two rows at time 0).
        rows = _tsv_rows(exported / 'experiments.tsv')
        assert [(r['experimentId'], r['time'], r['conditionId']) for r in rows] == [
            ('scan_0', '-inf', 'cond_incubate'), ('scan_0', '0', 'cond_wash'),
            ('scan_0', '0', 'cond_scan_0'),
            ('scan_1', '-inf', 'cond_incubate'), ('scan_1', '0', 'cond_wash'),
            ('scan_1', '0', 'cond_scan_1'),
            ('scan_2', '-inf', 'cond_incubate'), ('scan_2', '0', 'cond_wash'),
            ('scan_2', '0', 'cond_scan_2')]

    def test_species_targets_go_through_the_mapping_table(self, exported):
        # Each BNGL species pattern is aliased to a species_<...> id in the mapping table, and the
        # conditions target that id (a numeric wash and a param-expression, verbatim).
        maps = {(m['petabEntityId'], m['modelEntityId']) for m in _tsv_rows(exported / 'mapping.tsv')}
        assert maps == {('species_A', 'A()'), ('species_B', 'B()')}
        conds = {(r['conditionId'], r['targetId'], r['targetValue'])
                 for r in _tsv_rows(exported / 'conditions.tsv')}
        assert ('cond_incubate', 'species_A', '100') in conds
        assert ('cond_wash', 'species_A', '0') in conds
        assert ('cond_wash', 'species_B', 'L*kd') in conds
        # Each dose is its own single-target condition on the swept parameter L.
        assert ('cond_scan_0', 'L', '1') in conds
        assert ('cond_scan_2', 'L', '5') in conds

    def test_measurements_are_at_the_scan_time_per_dose(self, exported):
        meas = _tsv_rows(exported / 'measurements.tsv')
        assert all(m['time'] == '500' for m in meas)             # finite t_end scan
        assert [(m['experimentId'], m['measurement']) for m in meas] == [
            ('scan_0', '0.5'), ('scan_1', '1'), ('scan_2', '2.5')]
        assert 'obs_L' not in {m['observableId'] for m in meas}  # the swept axis is not measured

    def test_problem_yaml_lists_the_mapping_file(self, exported):
        assert 'mapping_files:\n  - mapping.tsv\n' in (exported / 'problem.yaml').read_text()

    def test_full_petab_validation_is_clean(self, exported):
        assert _petab_validation_errors(exported / 'problem.yaml') == []

    def test_steady_state_scan_uses_inf_measurement_time(self, tmp_path_factory):
        # No t_end: => steady state, so the measurement time is inf (PEtab's steady-state
        # convention), the -inf pre-equilibration period still preceding it.
        src = self._src(tmp_path_factory, 'pdr_ss')
        (src / 'job.conf').write_text(
            self._HEAD + self._CONDS
            + 'experiment: scan, preequilibrate: incubate, condition: wash, '
              'type: parameter_scan, data: dose.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        meas = _tsv_rows(out / 'measurements.tsv')
        assert all(m['time'] == 'inf' for m in meas)
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_wash_free_scan_has_a_single_condition_measurement_period(self, tmp_path_factory):
        # No measurement condition: (equilibrate, then scan with no intervention) -> the
        # measurement period carries only the per-dose condition (a single-condition period).
        src = self._src(tmp_path_factory, 'pdr_nowash')
        (src / 'job.conf').write_text(
            self._HEAD + 'condition: incubate, perturbations: "A()" = 100\n'
            + 'experiment: scan, preequilibrate: incubate, type: parameter_scan, data: dose.exp\n'
            + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        rows = [(r['experimentId'], r['time'], r['conditionId'])
                for r in _tsv_rows(out / 'experiments.tsv')]
        assert ('scan_0', '-inf', 'cond_incubate') in rows
        assert ('scan_0', '0', 'cond_scan_0') in rows
        assert not any(r[0] == 'scan_0' and r[2] == 'cond_wash' for r in rows)
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_species_condition_on_a_plain_time_course_is_refused(self, tmp_path_factory):
        # A species setConcentration is inline-only within a pre-equilibration protocol (ADR-0062);
        # applied to a plain time-course experiment it has no export route (the fitter rejects it too).
        src = self._src(tmp_path_factory, 'pdr_tc_species')
        (src / 'tc.exp').write_text('# time resp\n0\t0\n1\t0.5\n2\t1\n')
        (src / 'job.conf').write_text(
            self._HEAD + 'condition: wash, perturbations: "A()" = 0\n'
            + 'experiment: tc, condition: wash, data: tc.exp\n'
            + self._PARAMS)
        with pytest.raises(NotImplementedError, match='inline-only|pre-equilibration protocol'):
            export_job(src / 'job.conf', src / 'out')


# ---------------------------------------------------------------------------
# `perturbations: none` (#906, ADR-0150): a condition that changes nothing. As `preequilibrate:`
# its -inf period is the model as is -- a blank conditionId, or cond_wildtype re-pinning a
# non-empty surrogate set M -- and as the measured `condition:` it exports exactly as an omitted
# one. It never becomes a conditionId of its own (PEtab has no zero-row condition).
# ---------------------------------------------------------------------------

class TestExportUnperturbedCondition:

    _HEAD = TestExportPreequilibration._HEAD
    _PARAMS = TestExportPreequilibration._PARAMS

    def _export(self, tmp_path_factory, name, body, files=()):
        src = tmp_path_factory.mktemp(name)
        (src / 'm.bngl').write_text(_PREEQUIL_MODEL)
        (src / 'relax.exp').write_text('# time A_tot\n0\t10\n1\t6\n2\t4\n')
        for fname, text in files:
            (src / fname).write_text(text)
        (src / 'job.conf').write_text(self._HEAD + body + self._PARAMS)
        out = src / 'petab'
        export_job(src / 'job.conf', out)
        return out

    @staticmethod
    def _experiments(out):
        return [(r['experimentId'], r['time'], r['conditionId'])
                for r in _tsv_rows(out / 'experiments.tsv')]

    @staticmethod
    def _conditions(out):
        path = out / 'conditions.tsv'
        return ({(r['conditionId'], r['targetId'], r['targetValue']) for r in _tsv_rows(path)}
                if path.exists() else set())

    def test_none_preequilibration_is_a_blank_minus_inf_period(self, tmp_path_factory):
        out = self._export(tmp_path_factory, 'none_pre', (
            'condition: basal, perturbations: none\n'
            'condition: meas, perturbations: flag = 2\n'
            'experiment: relax, preequilibrate: basal, condition: meas, data: relax.exp\n'))
        assert self._experiments(out) == [('relax', '-inf', ''), ('relax', '0', 'cond_meas')]
        assert self._conditions(out) == {('cond_meas', 'flag', '2')}
        assert _petab_validation_errors(out / 'problem.yaml') == []
        # libpetab's own reading: a pre-equilibration period with an empty condition list.
        from petab.v2 import Problem
        (experiment,) = Problem.from_yaml(str(out / 'problem.yaml')).experiments
        assert [(p.time, list(p.condition_ids)) for p in experiment.periods] == [
            (float('-inf'), []), (0.0, ['cond_meas'])]

    def test_none_preequilibration_re_pins_a_fit_and_perturbed_parameter(self, tmp_path_factory):
        # Another experiment perturbs the fit parameter k, so M = {k}: k is renamed k__REF in the
        # parameter table and every period must set k, the -inf one included -- through the same
        # base condition a wildtype experiment gets.
        out = self._export(tmp_path_factory, 'none_pre_fit', (
            'condition: basal, perturbations: none\n'
            'condition: meas, perturbations: flag = 2\n'
            'condition: fast, perturbations: k * 2\n'
            'experiment: relax, preequilibrate: basal, condition: meas, data: relax.exp\n'
            'experiment: other, condition: fast, data: relax.exp\n'))
        assert {r['parameterId'] for r in _tsv_rows(out / 'parameters.tsv')} == {'k__REF'}
        assert ('relax', '-inf', 'cond_wildtype') in self._experiments(out)
        conditions = self._conditions(out)
        assert ('cond_wildtype', 'k', 'k__REF') in conditions
        assert ('cond_meas', 'k', 'k__REF') in conditions
        assert not any(cid == 'cond_basal' for cid, _t, _v in conditions)
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_none_measured_condition_exports_as_an_omitted_one(self, tmp_path_factory):
        # Byte-for-byte the problem an experiment with no condition exports, as a plain time
        # course, as a wash-out after a pre-equilibration, and on a plain dose-response scan
        # (which refuses a real named condition).
        dose = ('dose.exp', '# flag A_tot\n1\t10\n2\t5\n')
        for label, with_none, without in (
                ('tc', 'experiment: relax, condition: basal, data: relax.exp\n',
                 'experiment: relax, data: relax.exp\n'),
                ('washout', 'condition: pre, perturbations: flag = 0\n'
                            'experiment: relax, preequilibrate: pre, condition: basal, '
                            'data: relax.exp\n',
                 'condition: pre, perturbations: flag = 0\n'
                 'experiment: relax, preequilibrate: pre, data: relax.exp\n'),
                ('scan', 'experiment: scan, condition: basal, data: dose.exp\n',
                 'experiment: scan, data: dose.exp\n')):
            a = self._export(tmp_path_factory, f'none_meas_{label}',
                             'condition: basal, perturbations: none\n' + with_none, [dose])
            b = self._export(tmp_path_factory, f'omitted_{label}', without, [dose])
            assert sorted(f.name for f in a.iterdir()) == sorted(f.name for f in b.iterdir())
            for f in a.iterdir():
                assert f.read_text() == (b / f.name).read_text(), (label, f.name)

    def test_none_preequilibrated_scan_is_a_blank_minus_inf_period(self, tmp_path_factory):
        out = self._export(tmp_path_factory, 'none_pdr', (
            'condition: basal, perturbations: none\n'
            'experiment: scan, preequilibrate: basal, t_end: 1, data: dose.exp\n'),
            [('dose.exp', '# flag A_tot\n1\t10\n2\t5\n')])
        assert self._experiments(out) == [
            ('scan_0', '-inf', ''), ('scan_0', '0', 'cond_scan_0'),
            ('scan_1', '-inf', ''), ('scan_1', '0', 'cond_scan_1')]
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_an_empty_condition_is_never_written_as_a_condition_id(self):
        # The builders' own net: a condition with no rows cannot be expressed in PEtab.
        from pybnf.petab.conditions import _condition_rows_for
        with pytest.raises(PybnfError, match="would be exported with no rows"):
            _condition_rows_for('cond_basal', [], set(), lambda v: None)


# ---------------------------------------------------------------------------
# Multi-model export (ADR-0041, #430): a job with more than one model: each experiment names
# the model it simulates; the model id is stamped on its measurement rows' modelId (the column
# is omitted single-model), free parameters bind across the union of every model's ids, and
# problem.yaml lists every model in its own language (BNGL + SBML may mix). The dominant oracle
# is the byte-equal round trip (test_petab_import.py / test_petab_sbml_layer.py); here we assert
# the export shape + the unit boundaries.
# ---------------------------------------------------------------------------

# A second BNGL model (distinct stem, distinct parameters/observable/function) for the
# two-model fixtures: p/q where parabola has x/y, a1/a2 where parabola has v1/v2/v3.
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

# A conflict model: x is a FUNCTION here, where parabola_v2 has x as an OBSERVABLE -- so the
# shared column 'x' classifies differently across the two models (the cross-model conflict).
_CONFLICT_BNGL = """\
begin model
  begin parameters
    a1 1
  end parameters
  begin molecule types
    m()
  end molecule types
  begin seed species
    m() 1
  end seed species
  begin observables
    Molecules z m()
  end observables
  begin functions
    x()=a1*z
  end functions
  begin reaction rules
    0->m() 1
  end reaction rules
end model
"""


def _petab_multimodel_validation_errors(problem_yaml):
    """ERROR-severity petablint issues for a **multi-model** problem (the partial oracle).

    libpetab-python's validation framework is not yet multi-model-aware (libpetab#392): the
    model-cross tasks (``CheckModel``, ``CheckValidConditionTargets``, ...) access
    ``Problem.model`` (singular) and raise ``ValueError`` on a >1-model problem, and petablint
    emits only a WARNING on multiple models -- so the byte-equal round trip is the primary
    oracle (ADR-0041). This still runs every task that *can* run (asserting the multi-model
    problem loads, both models parsed), skipping only the ones that raise the multi-model
    ``ValueError``, and returns any genuine ERROR among the rest."""
    pytest.importorskip('petab.v2')
    from petab.v2 import Problem
    from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

    problem = Problem.from_yaml(str(problem_yaml))
    assert len(problem.models) > 1                 # the multi-model problem loaded
    errors = []
    for task in default_validation_tasks:
        try:
            issue = task.run(problem)
        except ValueError as exc:
            if 'more than one model' in str(exc):
                continue                            # libpetab#392: not multi-model-aware yet
            raise
        if issue is not None and getattr(issue, 'level', None) == \
                ValidationIssueSeverity.ERROR:
            errors.append((type(task).__name__, issue.message))
    return errors


def _write_two_model_bngl_job(d):
    """A two-model BNGL job: parabola_v2 (v1/v2/v3, observable x + function y) and growth
    (a1/a2, observable p + function q), one wildtype experiment each naming its model."""
    import shutil
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEMO_DIR / DEMO_MODEL, d / DEMO_MODEL)
    (d / 'growth_v2.bngl').write_text(_GROWTH_BNGL)
    shutil.copy(DEMO_DIR / 'par1.exp', d / 'pa.exp')
    (d / 'gr.exp').write_text(
        '# time\tp\tq\tp_SD\tq_SD\n'
        + ''.join(f'{t}\t{5 + t}\t{0.5 * (5 + t) + 2}\t1\t1\n' for t in range(5)))
    conf = d / 'job.conf'
    conf.write_text(
        'edition = 2\njob_type = de\nobjective = chi_sq\n'
        f'model: {DEMO_MODEL}\n'
        'model: growth_v2.bngl\n'
        f'experiment: pa, model: {DEMO_MODEL}, data: pa.exp\n'
        'experiment: gr, model: growth_v2.bngl, data: gr.exp\n'
        'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10\n'
        'uniform_var = a1 0 10\nuniform_var = a2 0 10\n')
    return conf


class TestExportMultiModel:

    @pytest.fixture(scope='class')
    def exported(self, tmp_path_factory):
        src = tmp_path_factory.mktemp('mm_export')
        conf = _write_two_model_bngl_job(src)
        out = src / 'petab'
        export_job(conf, out)
        return out

    def test_problem_yaml_lists_both_models_each_in_its_language(self, exported):
        text = (exported / 'problem.yaml').read_text()
        # Two model_files entries, declaration order, each its own language (here both bngl).
        assert text.index('parabola_v2:') < text.index('growth_v2:')   # declaration order
        assert 'location: parabola_v2.bngl' in text
        assert 'location: growth_v2.bngl' in text
        assert text.count('language: bngl') == 2

    def test_measurements_carry_the_modelid_link(self, exported):
        rows = _tsv_rows(exported / 'measurements.tsv')
        assert 'modelId' in rows[0]               # the optional column is present (multi-model)
        # Each experiment's rows are stamped with its model's stem; both wildtype, so
        # experimentId stays '' and the modelId is what distinguishes them (ADR-0041).
        by_model = {r['modelId'] for r in rows}
        assert by_model == {'parabola_v2', 'growth_v2'}
        assert all(r['experimentId'] == '' for r in rows)
        # The right observables landed under the right model.
        assert {r['observableId'] for r in rows if r['modelId'] == 'parabola_v2'} == \
            {'obs_x', 'func_y'}
        assert {r['observableId'] for r in rows if r['modelId'] == 'growth_v2'} == \
            {'obs_p', 'func_q'}

    def test_observables_cover_every_model(self, exported):
        ids = {r['observableId'] for r in _tsv_rows(exported / 'observables.tsv')}
        assert ids == {'obs_x', 'func_y', 'obs_p', 'func_q'}

    def test_parameters_union_across_models(self, exported):
        ids = {r['parameterId'] for r in _tsv_rows(exported / 'parameters.tsv')}
        assert ids == {'v1', 'v2', 'v3', 'a1', 'a2'}   # the union of both models' free params

    def test_partial_petab_validation_has_no_errors(self, exported):
        # The model-cross checks raise on multi-model (libpetab#392); the rest pass clean.
        assert _petab_multimodel_validation_errors(exported / 'problem.yaml') == []

    def test_stem_collision_across_models_raises(self, tmp_path):
        from pybnf.printing import PybnfError
        # Two model files sharing a stem ('m') would collide on the modelId / output filename.
        (tmp_path / 'm.bngl').write_text(_PARABOLA2_BNGL)
        (tmp_path / 'm.xml').write_text('<sbml/>\n')
        (tmp_path / 'e.exp').write_text('# time x y x_SD y_SD\n0\t-10\t86\t1\t1\n')
        (tmp_path / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: m.bngl\nmodel: m.xml\n'
            'experiment: e, model: m.bngl, data: e.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10\n')
        with pytest.raises(PybnfError, match='stem'):
            export_job(tmp_path / 'job.conf', tmp_path / 'out')

    def test_unnamed_experiment_under_multiple_models_is_ambiguous(self, tmp_path):
        from pybnf.printing import PybnfError
        src = tmp_path / 'src'
        conf = _write_two_model_bngl_job(src)
        # Strip the model: field off the 'pa' experiment -> ambiguous under 2 models.
        conf.write_text(conf.read_text().replace(
            f'experiment: pa, model: {DEMO_MODEL}, data: pa.exp',
            'experiment: pa, data: pa.exp'))
        with pytest.raises(PybnfError, match='does not name a model'):
            export_job(conf, tmp_path / 'out')

    def test_free_parameter_in_no_model_is_a_typo(self, tmp_path):
        from pybnf.printing import PybnfError
        src = tmp_path / 'src'
        conf = _write_two_model_bngl_job(src)
        # 'nope' is a parameter id of neither model -> the union typo check fires.
        conf.write_text(conf.read_text() + 'uniform_var = nope 0 10\n')
        with pytest.raises(PybnfError, match='nope'):
            export_job(conf, tmp_path / 'out')

    def test_cross_model_observable_conflict_raises(self, tmp_path):
        import shutil

        from pybnf.printing import PybnfError
        src = tmp_path / 'src'
        src.mkdir()
        shutil.copy(DEMO_DIR / DEMO_MODEL, src / DEMO_MODEL)   # x is an observable
        (src / 'conflict.bngl').write_text(_CONFLICT_BNGL)      # x is a function
        (src / 'a.exp').write_text('# time\tx\tx_SD\n0\t-10\t1\n1\t-9\t1\n')
        (src / 'b.exp').write_text('# time\tx\tx_SD\n0\t3\t1\n1\t4\t1\n')
        (src / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            f'model: {DEMO_MODEL}\nmodel: conflict.bngl\n'
            f'experiment: ea, model: {DEMO_MODEL}, data: a.exp\n'
            'experiment: eb, model: conflict.bngl, data: b.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10\n'
            'uniform_var = a1 0 10\n')
        with pytest.raises(PybnfError, match='inconsistently'):
            export_job(src / 'job.conf', tmp_path / 'out')


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

    @pytest.mark.parametrize('line', [
        'uniform_var = v1 0 10 u', 'loguniform_var = v1 0.1 10 u'])
    def test_unbounded_flag_is_refused(self, tmp_path, line):
        # End to end: the exporter used to read p1/p2 and drop the flag, so this conf --
        # which declares a search free to leave [0, 10] -- exported to a table byte-identical
        # to the bounded spelling, silently constraining the fit a re-import would run
        # (#736). Both flag-bearing families raise.
        conf = _demo_job(tmp_path, replace=(_V1_VAR, line))
        with pytest.raises(NotImplementedError, match='unbounded'):
            export_job(conf, tmp_path / 'out')

    def test_explicit_bounded_flag_exports_as_the_bare_declaration(self, tmp_path):
        # The other half of the flag is unchanged: 'b' means what leaving it off means, and
        # passing it through the FreeParameter must not perturb the emitted table.
        export_job(DEMO_CONF, tmp_path / 'bare')
        export_job(_demo_job(tmp_path, replace=(_V1_VAR, _V1_VAR + ' b')),
                   tmp_path / 'flagged')
        assert (tmp_path / 'flagged' / 'parameters.tsv').read_text() == \
            (tmp_path / 'bare' / 'parameters.tsv').read_text()

    def test_modern_edition_without_objective_is_refused(self, tmp_path):
        # New era has no implicit chi_sq default: an edition-2 job must name its objective.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(tmp_path, "uniform_var = v1 0 10\n"),
                       tmp_path / 'out')

    def test_constraint_experiment_is_refused(self, tmp_path):
        # BPSL constraints (.con/.prop) are PyBNF-native qualitative fitting with no core-PEtab
        # representation (ADR-0028 addendum): an experiment whose data: carries one is refused
        # rather than mis-exported (dropping it would export a different, weaker fit). The
        # fitter still runs the job; only export is refused.
        (tmp_path / 'c.prop').write_text('x > 0 always weight 1\n')
        conf = tmp_path / 'job.conf'
        conf.write_text(
            f"edition = 2\njob_type = de\nobjective = chi_sq\n"
            f"model: {DEMO_DIR / DEMO_MODEL}\n"
            f"experiment: e, data: {DEMO_DIR / 'par1.exp'}, {tmp_path / 'c.prop'}\n"
            "uniform_var = v1 0 10\n")
        with pytest.raises(NotImplementedError, match='constraint'):
            export_job(conf, tmp_path / 'out')

    def test_per_observable_normalization_is_refused(self, tmp_path):
        # Normalization (peak/init/zero/unit) is a PyBNF *prediction* transform -- a
        # whole-trajectory reduction of a predicted observable with no PEtab v2 operator
        # (ADR-0053, #444). An experiment carrying it is refused rather than mis-exported:
        # silently dropping it would emit a problem that scores the raw, un-normalized
        # columns -- a different objective (the same fail-loud stance as cumulative).
        with pytest.raises(NotImplementedError, match='normaliz'):
            export_job(_boundary_conf(
                tmp_path, "objective = chi_sq\nuniform_var = v1 0 10\nnormalization x = peak\n"),
                tmp_path / 'out')

    def test_whole_fit_normalization_is_refused(self, tmp_path):
        # The whole-fit form (normalization = <type>) is equally inexpressible in PEtab.
        with pytest.raises(NotImplementedError, match='normaliz'):
            export_job(_boundary_conf(
                tmp_path, "objective = chi_sq\nuniform_var = v1 0 10\nnormalization = init\n"),
                tmp_path / 'out')

    def test_floor_normalization_is_refused(self, tmp_path):
        # ADR-0066 (#479): floor (x + rho*max(x)) is a whole-series offset, equally non-pointwise.
        with pytest.raises(NotImplementedError, match='normaliz'):
            export_job(_boundary_conf(
                tmp_path, "objective = chi_sq\nuniform_var = v1 0 10\nnormalization x = floor 0.03\n"),
                tmp_path / 'out')

    def test_analytic_scale_is_refused(self, tmp_path):
        # A whole-fit `scale` compiles `normalization` to None (scale is not a Data transform),
        # so the export check must key off the `analytic_scale` config, not `normalization` alone.
        with pytest.raises(NotImplementedError, match='normaliz'):
            export_job(_boundary_conf(
                tmp_path, "objective = chi_sq\nuniform_var = v1 0 10\nnormalization = scale\n"),
                tmp_path / 'out')

    @pytest.mark.parametrize('objfunc', ['neg_bin', 'neg_bin_dynamic', 'score'])
    def test_petab_inexpressible_objective_not_implemented(self, tmp_path, objfunc):
        # neg_bin was removed from PEtab v2; score (the direct_pass successor) is not a
        # likelihood. All are named on the modern `objective` key.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path, f"objective = {objfunc}\nuniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_implicit_dynamic_sigma_objective_not_implemented(self, tmp_path):
        # A declared `fit` noise scale now exports (#439); chi_sq_dynamic is the remaining
        # boundary because its free sigma is an IMPLICIT `sigma__FREE` the user never declared,
        # so there are no bounds/prior to write as a PEtab estimated parameter.
        with pytest.raises(NotImplementedError, match='sigma__FREE'):
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

    def test_per_observable_noise_model_override_exports_per_column(self, tmp_path):
        # A per-observable noise_model override now exports (ADR-0045): each column takes its
        # own sigma source (its override, else the whole-fit base), no longer a deferred raise.
        out = tmp_path / 'out'
        export_job(_boundary_conf(
            tmp_path,
            "noise_model = gaussian, sigma = fix_at 1\n"     # whole-fit base -> column y
            "noise_model x = gaussian, sigma = fix_at 2\n"   # override -> column x
            "uniform_var = v1 0 10\n"),
            out)
        noise = {r['observableId']: r['noiseFormula'] for r in _tsv_rows(out / 'observables.tsv')}
        assert noise == {'obs_x': '2', 'func_y': '1'}        # the override on x, the base on y
        assert _petab_validation_errors(out / 'problem.yaml') == []

    def test_mean_centered_noise_model_not_implemented(self, tmp_path):
        # PEtab v2 is median-only; a mean-centered noise model has no representation.
        with pytest.raises(NotImplementedError):
            export_job(_boundary_conf(
                tmp_path,
                "noise_model = gaussian, sigma = fix_at 1, location = mean\n"
                "uniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_time_error_marginalization_not_implemented(self, tmp_path):
        # A time_error clause (ADR-0112, #587) says the reported times are not exact: the
        # objective integrates each observation over a prior on its true sampling time, and
        # config.py swaps the whole objective for a MarginalizedTimeObjective. A PEtab
        # measurements row carries one exact time, so there is nothing to write it as --
        # exporting emitted a problem scored at the nominal times by the ordinary likelihood,
        # a different statistical model, and said nothing (#738). Its sibling in the same
        # noise_model grammar, `cumulative`, was guarded here all along.
        with pytest.raises(NotImplementedError, match='time_error'):
            export_job(_boundary_conf(
                tmp_path,
                "noise_model = gaussian, sigma = fix_at 1, "
                "time_error = truncated_normal, sigma_t = fix_at 0.5\n"
                "uniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_per_observable_time_error_is_also_refused(self, tmp_path):
        # The whole-fit form is the only one a job can currently run (_maybe_marginalize_time
        # defers per-observable time priors), but the exporter reads the RAW config, so the
        # refusal keys on the ('time_error', observable) key rather than on the shape that
        # happens to be reachable today -- and names the observable it found.
        with pytest.raises(NotImplementedError, match=r"time_error|\['x'\]"):
            export_job(_boundary_conf(
                tmp_path,
                "noise_model = gaussian, sigma = fix_at 1\n"
                "noise_model x = gaussian, sigma = fix_at 2, "
                "time_error = uniform, sigma_t = fix_at 0.5\n"
                "uniform_var = v1 0 10\n"),
                tmp_path / 'out')

    def test_the_same_job_without_the_time_clause_exports(self, tmp_path):
        # The control: everything else about the conf is exportable, so the refusal above is
        # the time clause and not some other property of the fixture.
        export_job(_boundary_conf(
            tmp_path,
            "noise_model = gaussian, sigma = fix_at 1\n"
            "uniform_var = v1 0 10\n"),
            tmp_path / 'out')
        assert (tmp_path / 'out' / 'parameters.tsv').is_file()

    def test_cumulative_prediction_transform_not_implemented(self, tmp_path):
        # The cumulative->incident differencing (ADR-0051, #418) is a PyBNF prediction
        # transform with no PEtab v2 representation: refuse rather than silently export a
        # problem that scores the raw cumulative columns (a different objective). The
        # offending observable is named in the error.
        with pytest.raises(NotImplementedError, match='cumulative'):
            export_job(_boundary_conf(
                tmp_path,
                "noise_model = gaussian, sigma = fix_at 1\n"
                "noise_model x = gaussian, sigma = fix_at 2, cumulative\n"
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

    def test_generate_network_cap_survives_actions_strip(self):
        # #485: generate_network is a network-definition directive, not a simulation action.
        # Its finiteness cap (max_stoich / max_agg / max_iter) is what keeps a rule-based
        # network finite, so the cleaner keeps the directive verbatim while dropping the
        # simulate action (PEtab drives simulation from the measurement times instead).
        src = (
            "begin model\n begin parameters\n  k1 3\n end parameters\nend model\n\n"
            "begin actions\n"
            "  generate_network({overwrite=>1,max_stoich=>{EGF=>4,EGFR=>4}})\n"
            '  simulate({method=>"ode",t_end=>2})\n'
            "end actions\n")
        out = clean_model_for_petab(src)
        assert 'generate_network({overwrite=>1,max_stoich=>{EGF=>4,EGFR=>4}})' in out
        assert 'max_stoich' in out                        # the cap itself, verbatim
        assert 'simulate' not in out                      # the simulation action is dropped
        assert 'k1 3' in out                              # the model body is untouched
        # Idempotent: re-cleaning an already-clean model keeps the cap unchanged.
        assert clean_model_for_petab(out) == out

    def test_simulation_only_actions_block_fully_dropped(self):
        # A block with no generate_network line (simulation-only) disappears entirely, exactly
        # as the whole-block strip did before -- no network-definition directive to keep.
        src = ("begin model\nend model\n\nbegin actions\n"
               " simulate({})\n parameter_scan({})\nend actions\n")
        out = clean_model_for_petab(src)
        assert 'begin actions' not in out
        assert 'simulate' not in out and 'parameter_scan' not in out

    def test_commented_generate_network_is_not_kept(self):
        # A commented-out '# generate_network(...)' is documentation, not a live directive: it
        # must not resurrect the block (consistent with pset.py's BNGLModel scanner, #473).
        src = ("begin model\nend model\n\nbegin actions\n"
               "# generate_network({overwrite=>1})\n simulate({})\nend actions\n")
        out = clean_model_for_petab(src)
        assert 'begin actions' not in out and 'generate_network' not in out

    def test_parse_model_captures_function_bodies(self):
        # Phase A (ADR-0035): parse_model records each global function's body verbatim;
        # function_names is exactly the body keys; a function WITH arguments (not the
        # inlinable zero-arg convention) is named but yields an empty, non-inlinable body.
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin functions\n f() = (a + b)/c\n h = a*2\n g(x) = x^2\nend functions\n')
        assert ent.function_bodies == {'f': '(a + b)/c', 'h': 'a*2', 'g': ''}
        assert ent.function_names == frozenset({'f', 'h', 'g'})


# ---------------------------------------------------------------------------
# 4''. Grammar hardening (#437): the reader is pinned to the BNGL reference in
#      BNG_vscode_extension/docs/bngl-grammar.md -- block aliases and the seed-
#      species '$' clamp are the cases the canonical-only scanner used to miss.
# ---------------------------------------------------------------------------

class TestBnglGrammarHardening:

    def test_seed_species_dollar_clamp_is_stripped(self):
        # SeedSpeciesDefn = ["$"], Species, WS, MathExpression -- the '$' fixes the
        # concentration but is not part of the species identity, so the enumerated
        # state variable is the bare pattern (attached '$A()' and spaced '$ A()' both).
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin seed species\n $A() 100\n $ B() 0\n C() 5\nend seed species\n')
        assert ent.seed_species == frozenset({'A()', 'B()', 'C()'})
        assert '$A()' not in ent.seed_species        # the marker never leaks into the id

    def test_rejected_block_aliases_are_not_honored(self):
        # The grammar doc lists `molecules`/`rules` as aliases, but BNG2.pl 2.9.3
        # REJECTS both ("Could not process block type"); only `species` is real.
        # Matching the reference implementation, the reader must NOT treat
        # `begin molecules`/`begin rules` as their canonical blocks (else it would
        # accept models BNG2.pl refuses).
        from pybnf.petab._bngl import parse_model
        ent = parse_model('begin molecules\n A()\n B(x)\nend molecules\n')
        assert ent.molecule_type_names == frozenset()   # `molecules` not an alias

    def test_line_continuation_is_joined(self):
        # A trailing `\` continues the logical line (BNG2.pl readFile). Without
        # joining, a continued parameter reads as the value '\' (the corpus bug
        # this closes); the join concatenates directly -- no space -- so a token
        # split across the break rejoins (`1e\`+`3` -> `1e3`), matching BNG2.pl.
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin parameters\n'
            '  minusb = \\\n(p4-1)/(p4*(1+p2))\n'   # continued expression value
            '  r 1e\\\n3\n'                          # token split -> 1e3, no space
            '  a = 1+\\\n2+\\\n3\n'                  # chained continuation
            'end parameters\n')
        assert ent.parameters['minusb'] == '(p4-1)/(p4*(1+p2))'
        assert ent.parameters['r'] == '1e3'
        assert ent.parameters['a'] == '1+2+3'

    def test_backslash_in_comment_is_not_a_continuation(self):
        # BNG2.pl strips the comment before testing for a trailing `\`, so a `\`
        # living inside a comment must not swallow the next line.
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin parameters\n k 1 # note \\\n j 2\nend parameters\n')
        assert ent.parameters == {'k': '1', 'j': '2'}

    def test_indexed_declarations(self):
        # Legacy `.net`-style leading index (LineLabel = {Digit}, WS): the index
        # must not be read as the name. (corpus: example1/egfr_tutorial/Chattaraj)
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin parameters\n 1 L0 1\n 2 R0 2\nend parameters\n'
            'begin seed species\n 1 A() 100\n 2 B() 50\nend seed species\n')
        assert ent.parameters == {'L0': '1', 'R0': '2'}    # index dropped, real names
        assert ent.seed_species == frozenset({'A()', 'B()'})

    def test_labeled_seed_species(self):
        # Named line label (LineLabel = Name, ":"): `CD14: CD14(...)` -- the label,
        # which here even equals the molecule name, must not be read as the species.
        # (corpus: An_2009). Label is stripped before the `$` clamp.
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin seed species\n'
            ' CD14: CD14(TLR4,MD2) v1\n'
            ' clamp: $MD2(x~0) v2\n'
            'end seed species\n')
        assert ent.seed_species == frozenset({'CD14(TLR4,MD2)', 'MD2(x~0)'})

    def test_line_label_does_not_over_strip(self):
        # A normal `name value` param and an `@compartment:` species must be left
        # alone -- a compartment prefix carries `@`, so it is not a bare Name label.
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin parameters\n NA = 6.02e23\n k1 1.0\nend parameters\n'
            'begin seed species\n @PM:Rec() 100\nend seed species\n')
        assert ent.parameters == {'NA': '6.02e23', 'k1': '1.0'}
        assert ent.seed_species == frozenset({'@PM:Rec()'})   # compartment kept

    def test_seed_species_block_alias(self):
        # `begin species` is BNG's short alias for `begin seed species` -- and the
        # '$' clamp is stripped under the alias spelling too.
        from pybnf.petab._bngl import parse_model
        ent = parse_model('begin species\n $A() 100\n B() 0\nend species\n')
        assert ent.seed_species == frozenset({'A()', 'B()'})

    def test_alias_does_not_shadow_the_canonical_block(self):
        # The `species` alias must not swallow the *other* block whose name it is a
        # substring of: `seed species` and `molecule types` stay distinct namespaces.
        from pybnf.petab._bngl import parse_model
        ent = parse_model(
            'begin molecule types\n Counter()\nend molecule types\n'
            'begin seed species\n $Counter() 1\nend seed species\n')
        assert ent.molecule_type_names == frozenset({'Counter'})
        assert ent.seed_species == frozenset({'Counter()'})

    def test_bnglmodel_state_variable_ignores_the_clamp(self):
        # The whole point at the ABC seam: a clamped seed species is still a state
        # variable under its bare id (is_state_variable drives CheckModel's species
        # cross-checks in petablint).
        pytest.importorskip('petab')
        from petab.v1.models.bngl_model import BnglModel, parse_bngl
        model = BnglModel(
            parse_bngl('begin seed species\n $A() 100\nend seed species\n'),
            model_id='m')
        assert model.is_state_variable('A()')
        assert not model.is_state_variable('$A()')


# ---------------------------------------------------------------------------
# 4'. petab's native BnglModel, pinned at the ABC seam the exporter relies on
#     (ADR-0026). PyBNF's local adapter and its register_bngl() monkeypatch were
#     retired in #591 once petab 0.9.0 shipped the loader upstream; these are the
#     model-level semantics the table oracle checks externally, asserted method by
#     method, as verified against BNG2.pl when the adapter was written.
# ---------------------------------------------------------------------------

class TestNativeBnglModel:

    @pytest.fixture
    def model(self, tmp_path):
        # petab's BnglModel parsed from the exported (cleaned, numeric-nominal) demo
        # model -- the same file Problem.from_yaml loads.
        pytest.importorskip('petab')
        from petab.v1.models.bngl_model import BnglModel
        from pybnf.petab.export import export_job
        out = tmp_path / 'p'
        export_job(DEMO_CONF, out)
        return BnglModel.from_file(out / DEMO_MODEL)

    def test_petab_ships_the_bngl_loader_natively(self):
        # The #591 trigger, pinned: stock petab (>= 0.9.0) knows `bngl`. Were this
        # to fail, the installed petab predates the native loader and every
        # `language: bngl` oracle in this file would raise "Unknown model format".
        pytest.importorskip('petab')
        import petab.v1.models as models
        assert 'bngl' in models.known_model_types

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

    def test_a_parameter_defined_by_others_never_reports_a_value(self, tmp_path):
        # The demo model above is all literals, so nothing here pinned what petab
        # does with `kon  koff/Kd`. That is the case PyBNF depends on: a parameter
        # defined by other parameters has no value the model file can settle,
        # because a PEtab parameter table may override or estimate what it depends
        # on, and a value handed out here would reach a simulator as a constant
        # that overrides the model's own expression. PyBNF's own exporter already
        # refuses to give such a parameter a nominal value (_numeric_nominal), and
        # petab must not contradict it. Released petab 0.9.0 refuses with
        # NotImplementedError and PEtab-dev/libpetab-python#517 refuses with
        # ValueError; both leave it out of the free-value list, which is the part
        # that matters. Nothing in pybnf/ calls these two methods, and the petab
        # version is a range that CI re-resolves on every run, so this test is the
        # only thing that would notice petab starting to report a value.
        pytest.importorskip('petab')
        from petab.v1.models.bngl_model import BnglModel
        model_file = tmp_path / 'derived.bngl'
        model_file.write_text(
            'begin model\n'
            'begin parameters\n'
            '  koff  0.1\n'
            '  Kd    5.0\n'
            '  kon   koff/Kd\n'
            'end parameters\n'
            'end model\n'
        )
        model = BnglModel.from_file(model_file)
        free = dict(model.get_free_parameter_ids_with_values())
        assert 'kon' not in free
        assert free['koff'] == 0.1  # a literal is still reported
        with pytest.raises((ValueError, NotImplementedError)):
            model.get_parameter_value('kon')

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

    problem = Problem.from_yaml(str(exported / 'problem.yaml'))
    assert type(problem.model).__name__ == 'BnglModel'   # the BNGL loader ran
    errors = []
    for task in default_validation_tasks:
        issue = task.run(problem)
        if issue is not None and getattr(issue, 'level', None) == \
                ValidationIssueSeverity.ERROR:
            errors.append((type(task).__name__, issue.message))
    assert errors == []


class TestExportPreservesNetworkCap:
    """#485: a ``generate_network`` finiteness cap (``max_stoich`` / ``max_agg`` /
    ``max_iter``) must survive PEtab export. A model that is finite *only* under the cap
    becomes one that network-generates unbounded (a silent hang) if the cap is dropped, so the
    exported model has to carry it. The cap rides in the model's own actions block, so
    ``import_`` -- which copies the model byte-verbatim -- round-trips it for free."""

    CAP = 'generate_network({overwrite=>1,max_stoich=>{counter=>4}})'

    def _capped_fixture(self, d):
        # Reuse the shared parabola2 model, swapping its bare generate_network for a capped one.
        model = _PARABOLA2_BNGL.replace(
            'generate_network({overwrite=>1})', self.CAP)
        assert self.CAP in model                          # guard the fixture edit
        (d / 'parabola2.bngl').write_text(model)
        (d / 'par1.exp').write_text(
            '# time x y x_SD y_SD\n0\t-10\t3\t1\t1\n1\t-9\t2\t1\t1\n2\t-8\t1\t1\t1\n')
        conf = d / 'job.conf'
        conf.write_text(
            'edition = 2\njob_type = de\nobjective = chi_sq\n'
            'model: parabola2.bngl\n'
            'experiment: par1, data: par1.exp\n'
            'uniform_var = v1 0 10\nuniform_var = v2 0 10\nuniform_var = v3 0 10\n')
        return conf

    def test_exported_model_carries_the_cap(self, tmp_path):
        conf = self._capped_fixture(tmp_path)
        out = tmp_path / 'petab'
        export_job(conf, out)
        model = (out / 'parabola2.bngl').read_text()
        assert self.CAP in model                          # cap survives clean_model_for_petab
        assert 'simulate' not in model                    # the simulation action is still dropped

    def test_exported_capped_model_is_petab_valid(self, tmp_path):
        # petab.v2 lint accepts a BNGL model that retains a generate_network line.
        conf = self._capped_fixture(tmp_path)
        out = tmp_path / 'petab'
        export_job(conf, out)
        _assert_petab_clean(out)

    def test_cap_round_trips_through_import(self, tmp_path):
        # The exporter records the cap in the model itself, so the importer (byte-verbatim
        # model copy) recovers it with no new PEtab metadata field.
        pytest.importorskip('petab.v2')
        from pybnf.petab.import_ import import_job
        conf = self._capped_fixture(tmp_path)
        out = tmp_path / 'petab'
        export_job(conf, out)
        imp = import_job(out / 'problem.yaml', tmp_path / 'imp')
        assert self.CAP in (imp / 'parabola2.bngl').read_text()


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
        with pytest.raises(NotImplementedError) as excinfo:
            mutation_target_value('*', 2.0, nominal=None)
        message = str(excinfo.value)
        # The refusal covers both languages: a BNGL parameter whose value is an expression, and
        # an SBML parameter a rule or an initial assignment defines (#795). It used to name BNGL
        # alone, though an SBML model could already reach it.
        assert 'BNGL' in message and 'SBML' in message
        assert 'initial assignment' in message

    def test_the_refusal_names_the_parameter_when_the_caller_knows_it(self):
        with pytest.raises(NotImplementedError, match="'beta_N'"):
            mutation_target_value('*', 2.0, nominal=None, target='beta_N')

    def test_a_relative_condition_on_a_derived_sbml_parameter_refuses(self, tmp_path):
        # End to end through the exporter's own reader: the model file settles no value for
        # beta_N, so the exporter refuses rather than writing the placeholder attribute times two
        # into conditions.tsv, which is what issue #795 reported.
        from pybnf.petab.conditions import mutation_target_value as target_value
        from pybnf.petab.export import _numeric_nominal, _read_model
        xml = tmp_path / 'derived.xml'
        xml.write_text(
            '''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="derived">
    <listOfParameters>
      <parameter id="koff" value="1" constant="true"/>
      <parameter id="Kd" value="2" constant="true"/>
      <parameter id="kon" value="0" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="kon">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/><ci>koff</ci><ci>Kd</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
  </model>
</sbml>
''')
        view = _read_model(xml.read_text(), xml, 'sbml')
        assert _numeric_nominal(view, 'koff') == 1.0      # a literal still has a nominal
        assert _numeric_nominal(view, 'kon') is None      # the placeholder value="0" is gone
        with pytest.raises(NotImplementedError, match="'kon'"):
            target_value('*', 2.0, nominal=_numeric_nominal(view, 'kon'), target='kon')

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
