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
from pybnf.petab.conditions import (
    build_experiment_conditions,
    conditions_from_rows,
    read_condition_table,
    read_experiment_table,
)
from pybnf.petab.measurements import (
    PetabMeasurementRow,
    data_from_measurement_rows,
    measurement_rows_from_data,
    noise_parameter_ids_by_observable,
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

    def test_noise_parameter_ids_per_observable_guards_row_varying(self):
        # A constant-per-observable parameter id is a per-observable sigma; a row-varying id
        # (or a mix with numeric per-point values) is the deferred per-measurement frontier.
        def row(oid, t, pid=None, num=None):
            return PetabMeasurementRow(observable_id=oid, time=t, measurement=1.0,
                                       noise_parameter_id=pid, noise_parameters=num)
        ok = [row('a', 0, pid='sd_a'), row('a', 1, pid='sd_a'), row('b', 0, pid='sd_b')]
        assert noise_parameter_ids_by_observable(ok) == {'a': 'sd_a', 'b': 'sd_b'}
        with pytest.raises(NotImplementedError, match='per-measurement'):
            noise_parameter_ids_by_observable([row('a', 0, pid='sd_a'), row('a', 1, pid='sd_a2')])
        with pytest.raises(NotImplementedError, match='per-measurement'):
            noise_parameter_ids_by_observable([row('a', 0, pid='sd_a'), row('a', 1, num=2.0)])

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

    def test_unsupported_model_language_is_refused(self, demo_petab, tmp_path):
        # BNGL and SBML import (ADR-0036); any other model language is out of scope and
        # refused before any table is read (read_problem_yaml stays a pure reader; the
        # importer holds the policy in _require_supported_model).
        with pytest.raises(NotImplementedError, match="'bngl' or 'sbml'"):
            self._import_mutated(demo_petab, tmp_path,
                                 {'problem.yaml': ('language: bngl', 'language: pysb')})

    @pytest.mark.parametrize('distribution', ['neg_bin', 'log-normal', 'log-laplace'])
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

    def test_one_sided_truncation_is_refused(self, demo_petab, tmp_path):
        # A finite bound on one side with an infinite bound on the other has no finite
        # reflecting box -- the deferred #417 boundary, still raised in code.
        prob = tmp_path / 'prob'
        shutil.copytree(demo_petab, prob)
        (prob / 'parameters.tsv').write_text(
            'parameterId\testimate\tlowerBound\tupperBound\tpriorDistribution\t'
            'priorParameters\n'
            'v1\ttrue\t5\tinf\tgamma\t2;3\n'
            'v2\ttrue\t0\t10\t\t\n'
            'v3\ttrue\t0\t10\t\t\n')
        with pytest.raises(NotImplementedError, match='one-sided'):
            import_job(prob / 'problem.yaml', tmp_path / 'out')


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
