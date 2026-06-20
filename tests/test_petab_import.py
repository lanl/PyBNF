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
5. **The documented boundaries raise** (SBML model, a PyBNF-less prior family, a
   PEtab-inexpressible noise distribution, replicate rows) -- mirroring the export side.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pybnf.data import Data
from pybnf.parse import ploop
from pybnf.petab import (
    export_job,
    import_job,
    read_observable_table,
    read_parameter_table,
    read_problem_yaml,
)
from pybnf.petab.conditions import (
    build_experiment_conditions,
    conditions_from_rows,
    read_condition_table,
    read_experiment_table,
)
from pybnf.petab.measurements import (
    data_from_measurement_rows,
    measurement_rows_from_data,
    read_measurement_table,
)

DEMO_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'demo'
DEMO_CONF = DEMO_DIR / 'demo_bng_v2.conf'
DEMO_MODEL = 'parabola_v2.bngl'

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

        from pybnf.petab.bngl_model import register_bngl
        register_bngl()
        _, _, petab2, _ = imported
        problem = Problem.from_yaml(str(petab2 / 'problem.yaml'))
        assert type(problem.model).__name__ == 'BnglModel'
        errors = [type(t).__name__ for t in default_validation_tasks
                  if (i := t.run(problem)) is not None
                  and getattr(i, 'level', None) == ValidationIssueSeverity.ERROR]
        assert errors == []


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
# Reverse-asset unit tests (the seam, not the orchestrator)
# ---------------------------------------------------------------------------

class TestReverseAssets:

    def test_measurement_pivot_inverts_to_identical_rows(self):
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        column_to_id = {'x': 'obs_x', 'y': 'func_y'}
        rows = measurement_rows_from_data(data, column_to_id, experiment_id='')
        datas = data_from_measurement_rows(rows, {'obs_x': 'x', 'func_y': 'y'})
        # re-pivoting the reconstructed Data yields the same rows (the long<->wide inverse).
        again = measurement_rows_from_data(datas[''], column_to_id, experiment_id='')
        assert rows == again

    def test_measurement_no_noise_yields_no_sd_columns(self):
        # A fixed/column-mean sigma objective writes no noiseParameters, so no _SD columns
        # are reconstructed (mirrors what a sos/ave_norm_sos re-export reads).
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        rows = measurement_rows_from_data(data, {'x': 'obs_x'}, sd_suffix=None)
        recon = data_from_measurement_rows(rows, {'obs_x': 'x'})['']
        assert set(recon.cols) == {'time', 'x'}

    def test_repeated_observation_is_a_replicate_boundary(self):
        data = Data(file_name=str(DEMO_DIR / 'par1.exp'))
        rows = measurement_rows_from_data(data, {'x': 'obs_x'})
        with pytest.raises(NotImplementedError, match='replicate'):
            data_from_measurement_rows(rows + rows, {'obs_x': 'x'})

    def test_conditions_inverse_recovers_perturbations(self):
        exps = [('wt', None), ('dbl', 'doubled'), ('scl', 'scaled')]
        conds = {'doubled': [('v1', '*', 2.0)], 'scaled': [('s', '*', 5.0)]}
        cond_rows, _, surrogate, _ = build_experiment_conditions(
            exps, conds, fit_params={'v1', 'v2', 'v3'}, nominal_of=lambda v: 2.0)
        recovered = conditions_from_rows(cond_rows, surrogate)
        # The fit op recovers exactly; the fixed relative op recovers as its precomputed
        # absolute value (s*5 with nominal 2 -> s = 10); base pins are dropped.
        assert recovered == {'doubled': [('v1', '*', 2.0)], 'scaled': [('s', '=', 10.0)]}


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

    def test_sbml_model_is_refused(self, demo_petab, tmp_path):
        with pytest.raises(NotImplementedError, match='bngl'):
            self._import_mutated(demo_petab, tmp_path,
                                 {'problem.yaml': ('language: bngl', 'language: sbml')})

    @pytest.mark.parametrize('distribution', ['neg_bin', 'log-normal', 'log-laplace'])
    def test_petab_inexpressible_noise_is_refused(self, demo_petab, tmp_path, distribution):
        with pytest.raises(NotImplementedError):
            self._import_mutated(demo_petab, tmp_path,
                                 {'observables.tsv': ('normal', distribution)})

    def test_expression_observable_formula_is_refused(self, demo_petab, tmp_path):
        # A non-bare observableFormula is the deferred formula layer (ADR-0033): importing
        # it means synthesizing a BNGL function, with no round-trip oracle, so it raises.
        with pytest.raises(NotImplementedError, match='expression'):
            self._import_mutated(demo_petab, tmp_path,
                                 {'observables.tsv': ('obs_x\tx\t', 'obs_x\tx + 1\t')})

    def test_unsupported_prior_family_is_refused(self, demo_petab, tmp_path):
        # A PEtab prior family PyBNF has no Prior for (catalog-parity follow-up).
        prob = tmp_path / 'prob'
        shutil.copytree(demo_petab, prob)
        (prob / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\tpriorDistribution\t'
            'priorParameters\n'
            'v1\ttrue\t0\t10\tcauchy\t0;1\n'
            'v2\ttrue\t0\t10\t\t\n'
            'v3\ttrue\t0\t10\t\t\n')
        with pytest.raises(NotImplementedError, match='cauchy'):
            import_job(prob / 'problem.yaml', tmp_path / 'out')


# ---------------------------------------------------------------------------
# Reader robustness vs a REAL-WORLD v2 problem (the Boehm tutorial; #407 chunk 1)
#
# The byte-equal round trip only feeds the importer problems it itself emitted. This
# class feeds the dependency-free table readers a problem we did NOT emit -- the PEtab
# spec repo's only v2 example -- to prove they tolerate the shapes a real v2 problem uses
# that our exporter never writes (sci-notation bounds, a parameterName column, a blank
# nominalValue, no prior columns, a noisePlaceholders column, model_files-first yaml,
# expression observableFormulas, a parameter-id noiseParameters). It is SBML + expression
# observables, so import_job refuses it cleanly; the readers below are tested directly.
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

    def test_import_refuses_the_sbml_model_early(self, tmp_path):
        # The SBML boundary fires before any table is read (read_problem_yaml stays a pure
        # reader; the importer holds the BNGL-native policy).
        with pytest.raises(NotImplementedError, match='bngl'):
            import_job(self.YAML, tmp_path / 'out')

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

    def test_measurement_table_refuses_parameter_id_noise_parameters(self):
        # Boehm's noiseParameters column carries a parameter id (a placeholder override),
        # not a number -- the deferred placeholder semantics, surfaced as a clean boundary
        # rather than a raw float() ValueError.
        with pytest.raises(NotImplementedError, match='placeholder'):
            read_measurement_table(BOEHM_DIR / 'measurement_data.tsv')
