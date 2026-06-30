"""SBML import + the measurement-model observation layer on real backends (#407, ADR-0036).

The oracle for SBML import, weakest -> strongest (the exporter cannot emit SBML, so there is
no export -> import -> re-export round trip; ADR-0036 §8):

1. **Importer** -- a crafted SBML PEtab v2 problem imports to a runnable new-era conf: the
   ``.xml`` carried **verbatim**, the expression ``observableFormula`` emitted as a measurement
   model (``observable: <id>, formula: <expr>``), free parameters bound by id.
2. **Config wiring** -- the conf's measurement-model line builds the layer with the SBML
   expression namespace (species u parameters u compartments) and a fixed-constant snapshot.
3. **Layer on a real RoadRunner trace** -- simulate the SBML model and apply the layer; the
   materialized ``observableFormula`` column equals both a closed-form analytic reference and a
   direct numpy re-computation over the trace's species (lambdify vs hand numpy).
4. **Dual-backend agreement (``-m recovery``, bngsim)** -- the layer's column agrees across
   **RoadRunner and bngsim** (and both match the analytic reference), proving the observation
   layer is backend-agnostic: neither backend exposes a computed observable; the layer does.

The crafted model (controlled here, not vendored Boehm -- which also needs the deferred
placeholder layer) is mass-action decay ``A -> B`` with rate ``cell*k1*A``, so ``A(t) =
A0*exp(-k1 t)`` and ``A + B`` is conserved. The measurement model ``scale*A/(A+B)`` therefore
has the closed form ``scale * exp(-k1 t)`` -- an analytic oracle independent of both the
backend and the layer.

``petab``/``sympy`` is the optional ``pybnf[petab]`` extra (SBML observables are 100%
expressions, so SBML import always needs it); these tests ``importorskip('petab')``.
RoadRunner is a core dependency, so the RoadRunner legs run in the normal tier; the bngsim leg
is gated ``-m recovery`` + ``@pytest.mark.bngsim``.
"""

import csv
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pybnf.measurement import MeasurementLayer, MeasurementModel

# The vendored BioModels EpoR fixture (Becker), whose D2D ``Epo_cells`` / ``Epo_medium``
# observables are SBML assignment rules -- the exact #464 reproduction.
_BECKER_XML = Path(__file__).resolve().parent / 'sbml_files' / 'becker_epor.xml'
_HAS_BECKER = _BECKER_XML.exists()

# A crafted SBML L3V2 model: mass-action A -> B (rate cell*k1*A), with a fixed scale
# parameter. A(t)=10 exp(-k1 t); A+B=10 conserved. Loads in RoadRunner and bngsim alike.
DECAY_SBML = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="decay">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.5" constant="true"/>
      <parameter id="scale" value="100" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>cell</ci><ci>k1</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# The measurement model: scale*A/(A+B). With A+B conserved at 10 and scale=100, this is
# 100*A/10 = 10*A = 100*exp(-k1 t) at the published k1=0.5 (the analytic reference).
OBS_FORMULA = 'scale * A / (A + B)'
K1_TRUE = 0.5
SCALE = 100.0


def _analytic(t):
    return SCALE * np.exp(-K1_TRUE * t)


def _write_sbml_petab_problem(prob):
    """Write a crafted SBML PEtab v2 problem (an expression observable + a fixed-sigma noise)
    into directory ``prob``; the measurements are the analytic reference at t=0,1,2."""
    prob.mkdir(parents=True, exist_ok=True)
    (prob / 'model.xml').write_text(DECAY_SBML)
    (prob / 'parameters.tsv').write_text(
        'parameterId\testimate\tlowerBound\tupperBound\n'
        'k1\ttrue\t0.01\t10\n')
    (prob / 'observables.tsv').write_text(
        'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n'
        f'obs_ratio\t{OBS_FORMULA}\t1\tnormal\n')
    times = [0.0, 1.0, 2.0]
    rows = ''.join(f'obs_ratio\texp1\t{t}\t{_analytic(t)}\n' for t in times)
    (prob / 'measurements.tsv').write_text(
        'observableId\texperimentId\ttime\tmeasurement\n' + rows)
    (prob / 'conditions.tsv').write_text('conditionId\n')
    (prob / 'experiments.tsv').write_text('experimentId\ttime\tconditionId\n')
    (prob / 'problem.yaml').write_text(textwrap.dedent("""\
        format_version: 2.0.0
        parameter_files:
          - parameters.tsv
        observable_files:
          - observables.tsv
        measurement_files:
          - measurements.tsv
        condition_files:
          - conditions.tsv
        experiment_files:
          - experiments.tsv
        model_files:
          model:
            location: model.xml
            language: sbml
        """))
    return prob / 'problem.yaml'


# ---------------------------------------------------------------------------
# 1. Importer: a crafted SBML problem -> a runnable conf + verbatim .xml
# ---------------------------------------------------------------------------

class TestSbmlImport:

    def test_import_carries_xml_verbatim_and_emits_a_measurement_model(self, tmp_path):
        pytest.importorskip('petab')
        from pybnf.parse import ploop
        from pybnf.petab import import_job

        yaml = _write_sbml_petab_problem(tmp_path / 'prob')
        out = import_job(yaml, tmp_path / 'out')

        # The .xml is carried byte-verbatim -- the dynamical model is never edited (ADR-0036).
        assert (out / 'model.xml').read_text() == DECAY_SBML
        conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
        # The expression observable became a measurement-model line (not a model edit).
        meas = {k[1]: v for k, v in conf.items()
                if isinstance(k, tuple) and k[0] == 'measurement'}
        assert meas == {'obs_ratio': OBS_FORMULA}
        # SBML free parameter bound by id (ADR-0034), the .xml model declared, sos recovered.
        assert conf[('uniform_var', 'k1')][:2] == [0.01, 10.0]
        assert conf['model'] == ['model.xml']
        assert conf['objective'] == 'sos'
        # The .exp reconstructs the analytic measurement table cell-for-cell.
        from pybnf.data import Data
        exp = Data(file_name=str(out / 'exp1.exp'))
        np.testing.assert_allclose(exp['obs_ratio'], [_analytic(t) for t in (0., 1., 2.)])

    def test_unsupported_language_still_refused(self, tmp_path):
        # The scope is BNGL + SBML; anything else is refused before any table is read.
        pytest.importorskip('petab')
        from pybnf.petab import import_job
        yaml = _write_sbml_petab_problem(tmp_path / 'prob')
        yaml.write_text(yaml.read_text().replace('language: sbml', 'language: pysb'))
        with pytest.raises(NotImplementedError, match="'bngl' or 'sbml'"):
            import_job(yaml, tmp_path / 'out')


# ---------------------------------------------------------------------------
# 1b. Exporter: a native SBML PyBNF job -> a valid SBML PEtab v2 problem (#429, ADR-0040)
# ---------------------------------------------------------------------------

def _write_sbml_job(d):
    """Write a native SBML PyBNF job (a new-era conf + the verbatim ``.xml`` + a ``.exp``)
    into directory ``d``; return the conf path. The job measures ``obs_ratio`` -- a
    measurement-model formula (``observable: ... formula:``) over the SBML species, the
    exporter's input for emitting an ``observableFormula`` expression."""
    d.mkdir(parents=True, exist_ok=True)
    (d / 'decay.xml').write_text(DECAY_SBML)
    (d / 'meas.exp').write_text('# time\tobs_ratio\n0\t100\n1\t60\n2\t36\n')
    conf = d / 'job.conf'
    conf.write_text(textwrap.dedent(f"""\
        edition = 2
        job_type = de
        objective = sos
        model: decay.xml
        observable: obs_ratio, formula: {OBS_FORMULA}
        experiment: meas, data: meas.exp
        uniform_var = k1 0.01 10
        """))
    return conf


class TestSbmlExport:

    def test_exports_sbml_verbatim_with_observable_formula(self, tmp_path):
        from pybnf.petab import export_job

        out = export_job(_write_sbml_job(tmp_path / 'job'), tmp_path / 'petab')
        # problem.yaml declares the model in its own native language, at the verbatim .xml.
        yaml = (out / 'problem.yaml').read_text()
        assert 'language: sbml' in yaml
        assert 'location: decay.xml' in yaml
        # The .xml is carried byte-verbatim -- the dynamical model is never edited (ADR-0036).
        assert (out / 'decay.xml').read_text() == DECAY_SBML
        # SBML carries no observables, so the measurement-model formula is emitted as the
        # observableFormula (the mirror of the importer's measurement-model line).
        obs = (out / 'observables.tsv').read_text()
        assert 'obs_ratio' in obs and OBS_FORMULA in obs

    def test_sbml_round_trips_byte_for_byte(self, tmp_path):
        # The dominant oracle: a native SBML job exports -> imports (ADR-0036) -> re-exports,
        # reproducing every PEtab file byte-for-byte. Import needs the petab math layer.
        pytest.importorskip('petab')
        from pybnf.petab import export_job, import_job

        conf = _write_sbml_job(tmp_path / 'job')
        petab1 = export_job(conf, tmp_path / 'petab1')
        import_job(petab1 / 'problem.yaml', tmp_path / 'imported')
        petab2 = export_job(tmp_path / 'imported' / 'imported.conf', tmp_path / 'petab2')

        names = sorted(p.name for p in petab1.iterdir())
        assert names == sorted(p.name for p in petab2.iterdir())
        for name in names:
            assert (petab1 / name).read_text() == (petab2 / name).read_text(), \
                f'{name} differs after export -> import -> re-export'

    def test_exported_sbml_problem_passes_petab_validation(self, tmp_path):
        # The external oracle: the exported SBML problem loads + validates via the real
        # petablint path (an SBML model needs libsbml, which petab pulls in).
        pytest.importorskip('petab')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        from pybnf.petab import export_job
        out = export_job(_write_sbml_job(tmp_path / 'job'), tmp_path / 'petab')
        problem = Problem.from_yaml(str(out / 'problem.yaml'))
        errors = [type(t).__name__ for t in default_validation_tasks
                  if (i := t.run(problem)) is not None
                  and getattr(i, 'level', None) == ValidationIssueSeverity.ERROR]
        assert errors == []


# ---------------------------------------------------------------------------
# 1c. Multi-model export mixing BNGL + SBML (ADR-0041, #430): the two-model round trip
# composes with the per-language emit (ADR-0040). A job declares a BNGL model and an SBML
# model, each experiment naming the one it simulates; the exported problem.yaml lists each in
# its own language, the modelId column links each measurement row to its model, and the
# whole thing round-trips export -> import -> re-export byte-for-byte.
# ---------------------------------------------------------------------------

_DEMO_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'demo'
_DEMO_BNGL = 'parabola_v2.bngl'   # v1/v2/v3, observable x + function y


def _write_mixed_job(d):
    """A two-model job mixing a BNGL model (parabola_v2: observable x) and the crafted SBML
    decay model (its obs_ratio measurement model), one experiment each naming its model."""
    import shutil
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(_DEMO_DIR / _DEMO_BNGL, d / _DEMO_BNGL)
    (d / 'decay.xml').write_text(DECAY_SBML)
    (d / 'pa.exp').write_text('# time\tx\n0\t-10\n1\t-9\n2\t-8\n')
    (d / 'dec.exp').write_text('# time\tobs_ratio\n0\t100\n1\t60\n2\t36\n')
    conf = d / 'job.conf'
    conf.write_text(textwrap.dedent(f"""\
        edition = 2
        job_type = de
        objective = sos
        model: {_DEMO_BNGL}
        model: decay.xml
        observable: obs_ratio, formula: {OBS_FORMULA}
        experiment: pa, model: {_DEMO_BNGL}, data: pa.exp
        experiment: dec, model: decay.xml, data: dec.exp
        uniform_var = v1 0 10
        uniform_var = v2 0 10
        uniform_var = v3 0 10
        uniform_var = k1 0.01 10
        """))
    return conf


class TestMixedBnglSbmlExport:

    def test_problem_yaml_emits_each_model_in_its_language(self, tmp_path):
        from pybnf.petab import export_job
        out = export_job(_write_mixed_job(tmp_path / 'job'), tmp_path / 'petab')
        yaml = (out / 'problem.yaml').read_text()
        # The BNGL model is PEtab-cleaned (no begin actions), the SBML carried verbatim.
        assert 'location: parabola_v2.bngl' in yaml and 'language: bngl' in yaml
        assert 'location: decay.xml' in yaml and 'language: sbml' in yaml
        assert (out / 'decay.xml').read_text() == DECAY_SBML
        # The SBML observable is the measurement-model formula; the BNGL one is the bare name.
        obs = {r['observableId']: r['observableFormula']
               for r in csv.DictReader(open(out / 'observables.tsv'), delimiter='\t')}
        assert obs == {'obs_x': 'x', 'obs_ratio': OBS_FORMULA}
        # The modelId column links each measurement row to its model.
        rows = list(csv.DictReader(open(out / 'measurements.tsv'), delimiter='\t'))
        assert {r['modelId'] for r in rows} == {'parabola_v2', 'decay'}

    def test_mixed_round_trips_byte_for_byte(self, tmp_path):
        # The dominant oracle: a mixed BNGL + SBML job exports -> imports (the SBML expression
        # observable becomes a measurement model, ADR-0036) -> re-exports byte-identically.
        pytest.importorskip('petab')
        from pybnf.petab import export_job, import_job

        conf = _write_mixed_job(tmp_path / 'job')
        petab1 = export_job(conf, tmp_path / 'petab1')
        import_job(petab1 / 'problem.yaml', tmp_path / 'imported')
        petab2 = export_job(tmp_path / 'imported' / 'imported.conf', tmp_path / 'petab2')

        names = sorted(p.name for p in petab1.iterdir())
        assert names == sorted(p.name for p in petab2.iterdir())
        for name in names:
            assert (petab1 / name).read_text() == (petab2 / name).read_text(), \
                f'{name} differs after export -> import -> re-export'

    def test_imported_mixed_conf_declares_both_models(self, tmp_path):
        pytest.importorskip('petab')
        from pybnf.parse import ploop
        from pybnf.petab import export_job, import_job

        conf = _write_mixed_job(tmp_path / 'job')
        export_job(conf, tmp_path / 'petab1')
        import_job(tmp_path / 'petab1' / 'problem.yaml', tmp_path / 'imported')
        text = (tmp_path / 'imported' / 'imported.conf').read_text()
        assert 'model: parabola_v2.bngl' in text and 'model: decay.xml' in text
        # The SBML expression observable rides the measurement-model layer (ADR-0036).
        d = ploop(text.splitlines(keepends=True))
        meas = {k[1]: v for k, v in d.items()
                if isinstance(k, tuple) and k[0] == 'measurement'}
        assert meas == {'obs_ratio': OBS_FORMULA}
        # Both model files carried verbatim into the imported job.
        assert (tmp_path / 'imported' / 'decay.xml').read_text() == DECAY_SBML


# ---------------------------------------------------------------------------
# 2. Config wiring: an SBML conf's formula line builds the layer (SBML namespace)
# ---------------------------------------------------------------------------

def _sbml_config(tmp_path, *, backend='roadrunner'):
    """A real Configuration for the crafted SBML model + the measurement-model line."""
    import os

    from pybnf import config as config_mod
    from pybnf.parse import ploop
    (tmp_path / 'decay.xml').write_text(DECAY_SBML)
    (tmp_path / 'meas.exp').write_text(
        '# time\tobs_ratio\n'
        + ''.join(f'{t}\t{_analytic(t)}\n' for t in (0.0, 1.0, 2.0)))
    conf_text = textwrap.dedent(f"""\
        edition = 2
        job_type = de
        objective = sos
        sbml_backend = {backend}
        model: decay.xml
        observable: obs_ratio, formula: {OBS_FORMULA}
        experiment: meas, data: meas.exp
        uniform_var = k1 0.01 10
        population_size = 4
        max_iterations = 1
        verbosity = 0
        """)
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


class TestSbmlConfigWiring:

    def test_config_builds_layer_with_sbml_namespace(self, tmp_path):
        pytest.importorskip('petab')
        conf = _sbml_config(tmp_path)
        assert conf.obj.measurement and len(conf.obj.measurement) == 1
        mm = conf.obj.measurement.models[0]
        assert mm.observable_id == 'obs_ratio'
        # species u parameters u compartments (the SBML ParamList analogue, ADR-0036).
        assert mm.allowed_symbols == {'A', 'B', 'k1', 'scale', 'cell'}
        # scale + cell are fixed -> constants; k1 is free -> resolved from the PSet.
        assert mm.constants == {'scale': 100.0, 'cell': 1.0}


# A crafted SBML model with an assignment-rule variable: ``total := A + B`` (a value-less
# parameter assigned every step). The forward A->B kinetics are irrelevant -- the test only
# builds the Configuration, which is where the #464 namespace check lives.
RULE_SBML = DECAY_SBML.replace(
    '      <parameter id="scale" value="100" constant="true"/>\n',
    '      <parameter id="scale" value="100" constant="true"/>\n'
    '      <parameter id="total" constant="false"/>\n').replace(
    '    </listOfReactions>\n',
    '    </listOfReactions>\n'
    '    <listOfRules>\n'
    '      <assignmentRule variable="total">\n'
    '        <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
    '          <apply><plus/><ci> A </ci><ci> B </ci></apply>\n'
    '        </math>\n'
    '      </assignmentRule>\n'
    '    </listOfRules>\n')


def _rule_config(tmp_path, formula):
    """Build a Configuration for ``RULE_SBML`` whose measurement formula is ``formula``.
    Raises (the #464 path) when ``formula`` references the assignment-rule variable ``total``."""
    import os

    from pybnf import config as config_mod
    from pybnf.parse import ploop
    (tmp_path / 'ruled.xml').write_text(RULE_SBML)
    (tmp_path / 'meas.exp').write_text('# time\tobs\n0\t1.0\n1\t1.0\n')
    conf_text = textwrap.dedent(f"""\
        edition = 2
        job_type = de
        objective = sos
        sbml_backend = roadrunner
        model: ruled.xml
        observable: obs, formula: {formula}
        experiment: meas, data: meas.exp
        uniform_var = k1 0.01 10
        population_size = 4
        max_iterations = 1
        verbosity = 0
        """)
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


class TestAssignmentRuleFormulaRejected:
    """A measurement formula that references an SBML assignment-rule variable is rejected
    **at config build** with a pointed, reconstruct-from-species message -- it must not pass
    validation and then fail mid-fit in ``MeasurementModel.materialize`` (#464)."""

    def test_referencing_an_assignment_rule_var_fails_fast(self, tmp_path):
        pytest.importorskip('petab')
        from pybnf.printing import PybnfError
        with pytest.raises(PybnfError) as exc:
            _rule_config(tmp_path, 'total')
        msg = str(exc.value)
        # Names the offending symbol, calls out the assignment-rule cause, and points at the
        # species to rebuild from (the rule's <ci> referents).
        assert "'total'" in msg and 'assignment-rule' in msg
        assert 'A' in msg and 'B' in msg
        assert '#464' in msg

    def test_reconstructing_from_species_still_builds(self, tmp_path):
        # The fail-fast is targeted, not a blanket ban: the working reconstruction over the
        # species the rule is defined on builds the layer normally, and the rule target is
        # absent from the allowed namespace.
        pytest.importorskip('petab')
        conf = _rule_config(tmp_path, 'A + B')
        mm = conf.obj.measurement.models[0]
        assert mm.observable_id == 'obs'
        assert {'A', 'B'} <= mm.allowed_symbols
        assert 'total' not in mm.allowed_symbols


@pytest.mark.skipif(not _HAS_BECKER, reason='becker_epor.xml fixture missing')
class TestBeckerAssignmentRuleRejected:
    """The issue's exact reproduction on the vendored BioModels EpoR fixture: a formula naming
    the D2D ``Epo_cells`` assignment-rule observable is rejected at config build, naming the
    species (``Epo_EpoRi + dEpoi``) the sibling smoke test (#462) reconstructs it from."""

    def test_becker_epo_cells_rejected_at_config_build(self, tmp_path):
        pytest.importorskip('petab')
        from pybnf.parse import ploop
        from pybnf import config as config_mod
        from pybnf.printing import PybnfError
        (tmp_path / 'becker_tc.exp').write_text(
            '# time\tEpo_cells\tEpo_cells_SD\n0\t1.0\t0.1\n10\t1.0\t0.1\n')
        conf_text = textwrap.dedent(f"""\
            edition = 2
            job_type = trf
            objective = chi_sq
            sbml_backend = roadrunner
            model: {_BECKER_XML}
            observable: Epo_cells, formula: Epo_cells
            experiment: tc, data: becker_tc.exp
            uniform_var = kon 1e-5 1e-2
            uniform_var = koff 1e-3 1e-1
            population_size = 4
            max_iterations = 1
            verbosity = 0
            """)
        import os
        home = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(PybnfError) as exc:
                config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
        finally:
            os.chdir(home)
        msg = str(exc.value)
        assert "'Epo_cells'" in msg and 'assignment-rule' in msg
        # The reconstruction hint names the underlying species of the D2D observable.
        assert 'Epo_EpoRi' in msg and 'dEpoi' in msg


# ---------------------------------------------------------------------------
# 2c. Antimony (.ant) config wiring: the formula namespace is built from the SBML
#     the model converts to at load, not from parsing the .ant as BNGL (#463)
# ---------------------------------------------------------------------------

# A crafted Antimony model: A -> B decay with an assignment-rule observable ``total := A + B``.
CRAFT_ANT = """\
model craft
  species A = 10, B = 0;
  k1 = 0.5;
  J0: A -> B; k1*A;
  total := A + B;
end
"""

_BECKER_ANT = Path(__file__).resolve().parent / 'sbml_files' / 'becker_epor.ant'
_HAS_BECKER_ANT = _BECKER_ANT.exists()


def _ant_config(tmp_path, formula, *, model='craft.ant', model_text=CRAFT_ANT,
                obs_col='obs', free=(('k1', 0.01, 10),)):
    """Build a Configuration for an Antimony model whose measurement formula is ``formula``.

    ``.ant`` is a bngsim-only path (no roadrunner-antimony loader), so callers gate on
    ``@pytest.mark.bngsim_antimony``. Lines are assembled explicitly (not via a dedented
    template) so a multi-line free-parameter block keeps its indentation."""
    import os

    from pybnf import config as config_mod
    from pybnf.parse import ploop
    if model_text is not None:
        (tmp_path / model).write_text(model_text)
    (tmp_path / 'meas.exp').write_text(f'# time\t{obs_col}\n0\t10\n1\t6\n')
    lines = ['edition = 2', 'job_type = de', 'objective = sos', 'sbml_backend = bngsim',
             f'model: {model}', f'observable: {obs_col}, formula: {formula}',
             'experiment: meas, data: meas.exp']
    lines += [f'uniform_var = {n} {lo} {hi}' for n, lo, hi in free]
    lines += ['population_size = 4', 'max_iterations = 1', 'verbosity = 0']
    conf_text = '\n'.join(lines) + '\n'
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


@pytest.mark.bngsim_antimony
class TestAntimonyConfigWiring:
    """A measurement formula over an Antimony model's species must validate: a ``.ant`` model
    is converted to SBML at load, so its species/parameter namespace is fully available --
    building the namespace by parsing the ``.ant`` as BNGL (the old path) wrongly rejected
    every species as "not a known model entity" (#463). The routed SBML path also inherits the
    assignment-rule exclusion (#464)."""

    def test_ant_species_formula_builds_the_layer(self, tmp_path):
        pytest.importorskip('petab')
        conf = _ant_config(tmp_path, 'A + B')
        assert conf.obj.measurement and len(conf.obj.measurement) == 1
        mm = conf.obj.measurement.models[0]
        assert mm.observable_id == 'obs'
        # The species the .ant declares are in the namespace (the #463 regression: before, the
        # .ant parsed as BNGL and the species were absent, so this formula was rejected).
        assert {'A', 'B'} <= mm.allowed_symbols

    def test_ant_assignment_rule_var_rejected_via_sbml_path(self, tmp_path):
        # The Antimony ``total := A + B`` becomes an SBML assignmentRule, so routing .ant
        # through parse_sbml carries the #464 exclusion: referencing it fails fast at load.
        pytest.importorskip('petab')
        from pybnf.printing import PybnfError
        with pytest.raises(PybnfError) as exc:
            _ant_config(tmp_path, 'total')
        msg = str(exc.value)
        assert "'total'" in msg and 'assignment-rule' in msg and '#464' in msg

    @pytest.mark.skipif(not _HAS_BECKER_ANT, reason='becker_epor.ant fixture missing')
    def test_becker_ant_species_formula_builds(self, tmp_path):
        # The issue's exact reproduction: the becker .ant scored through a measurement formula
        # over two of its species (the D2D Epo_cells observable, reconstructed). Before #463
        # this raised "'Epo_EpoRi' is not a known model entity"; now it builds end to end.
        pytest.importorskip('petab')
        conf = _ant_config(
            tmp_path, 'Epo_EpoRi + dEpoi', model=str(_BECKER_ANT), model_text=None,
            obs_col='obs_cells', free=(('kon', 1e-5, 1e-2), ('koff', 1e-3, 1e-1)))
        mm = conf.obj.measurement.models[0]
        assert mm.observable_id == 'obs_cells'
        assert {'Epo_EpoRi', 'dEpoi'} <= mm.allowed_symbols
        # The assignment-rule observables are excluded from the .ant namespace too (#464).
        assert 'Epo_cells' not in mm.allowed_symbols


# ---------------------------------------------------------------------------
# 3. The layer on a real RoadRunner SBML trace (core dependency; normal tier)
# ---------------------------------------------------------------------------

def _simulate(model_cls, xml_path):
    """Simulate the crafted model at k1=0.5 over t=0..4 (step 0.5) and return its Data."""
    from pybnf.pset import FreeParameter, PSet, TimeCourse
    ps = PSet([FreeParameter('k1', 'uniform_var', 0, 1, value=K1_TRUE)])
    model = model_cls(str(xml_path), str(xml_path), pset=ps,
                      actions=(TimeCourse({'time': '4', 'step': '0.5'}),))
    ds = model.execute(str(xml_path.parent), 'decay', 0)
    layer = MeasurementLayer([
        MeasurementModel('obs_ratio', OBS_FORMULA, {'A', 'B', 'scale'},
                         constants={'scale': SCALE})])
    layer.apply({model.name: ds}, {'k1': K1_TRUE})
    return ds[next(iter(ds))]


class TestLayerOnRoadRunnerTrace:

    def test_materialized_column_matches_analytic_and_hand_numpy(self, tmp_path):
        pytest.importorskip('petab')
        from pybnf.pset import SbmlModelNoTimeout
        (tmp_path / 'decay.xml').write_text(DECAY_SBML)
        data = _simulate(SbmlModelNoTimeout, tmp_path / 'decay.xml')

        assert 'obs_ratio' in data.cols
        t = data['time']
        # (a) matches the closed-form reference (whole pipeline, ODE tolerance)
        np.testing.assert_allclose(data['obs_ratio'], _analytic(t), rtol=1e-4)
        # (b) the layer's lambdify output is exactly a direct numpy recomputation over the
        # SAME trace's species (lambdify vs hand numpy -- isolates the layer from the ODE).
        np.testing.assert_allclose(
            data['obs_ratio'], SCALE * data['A'] / (data['A'] + data['B']), rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Dual-backend agreement (-m recovery, bngsim): the layer is backend-agnostic
# ---------------------------------------------------------------------------

@pytest.mark.recovery
@pytest.mark.bngsim
class TestDualBackendLayer:

    def test_layer_agrees_across_roadrunner_and_bngsim(self, tmp_path):
        """The measurement layer's computed ``observableFormula`` column agrees across
        RoadRunner and bngsim (and both match the analytic reference). Neither backend
        exposes a computed observable -- the layer materializes it identically over each
        backend's species trajectory, which is the whole point of ADR-0036."""
        pytest.importorskip('petab')
        from pybnf.bngsim_sbml_model import BngsimSbmlModelNoTimeout
        from pybnf.pset import SbmlModelNoTimeout
        (tmp_path / 'decay.xml').write_text(DECAY_SBML)

        rr = _simulate(SbmlModelNoTimeout, tmp_path / 'decay.xml')
        bg = _simulate(BngsimSbmlModelNoTimeout, tmp_path / 'decay.xml')

        t = rr['time']
        np.testing.assert_allclose(rr['time'], bg['time'])
        # Both backends' layer columns match the analytic reference (ODE tolerance)...
        np.testing.assert_allclose(rr['obs_ratio'], _analytic(t), rtol=1e-4)
        np.testing.assert_allclose(bg['obs_ratio'], _analytic(t), rtol=1e-4)
        # ...and agree with each other -- the backend-agnostic guarantee.
        np.testing.assert_allclose(rr['obs_ratio'], bg['obs_ratio'], rtol=1e-3)


# ---------------------------------------------------------------------------
# 5. The real-world Boehm problem reproduces its published data (-m recovery)
#
# The crafted decay model above isolates the layer with an analytic oracle; this closes
# the loop on the HEADLINE milestone -- the externally-authored Boehm v2 problem imported
# end to end (ADR-0037). Simulated at the published optimum (the SBML's embedded parameter
# values) on RoadRunner, the imported measurement layer's materialized observable columns
# track the published measurement table (the fit IS the optimum), and agree across
# RoadRunner and bngsim. Opt-in (-m recovery): a real stiff-ODE simulation of the full
# model, not a unit-scale crafted one.
# ---------------------------------------------------------------------------

_BOEHM_DIR = Path(__file__).resolve().parent / 'petab_fixtures' / 'boehm_v2'
_BOEHM_OBS = ('pSTAT5A_rel', 'pSTAT5B_rel', 'rSTAT5A_rel')
# The estimated model parameters (the 3 sd_* sigmas do not enter the trajectory).
_BOEHM_FREE = ('Epo_degradation_BaF3', 'k_exp_hetero', 'k_exp_homo',
               'k_imp_hetero', 'k_imp_homo', 'k_phos')


def _boehm_measurement_data():
    rows = list(csv.DictReader(open(_BOEHM_DIR / 'measurement_data.tsv'), delimiter='\t'))
    return {oid: (np.array([float(r['time']) for r in rows if r['observableId'] == oid]),
                  np.array([float(r['measurement']) for r in rows if r['observableId'] == oid]))
            for oid in _BOEHM_OBS}


def _simulate_boehm(model_cls, out_dir):
    """Import Boehm into ``out_dir``, simulate the SBML at the published optimum on
    ``model_cls``, apply the imported measurement layer, and return the materialized Data.

    The published optimum is the SBML's embedded parameter ``value`` attributes (read via the
    stdlib ``_sbml`` constants snapshot); the measurement models come from the imported conf's
    ``observable: <id>, formula: <expr>`` lines (specC17 already inlined, ADR-0037)."""
    from pybnf.parse import ploop
    from pybnf.petab import import_job
    from pybnf.petab._sbml import parse_model as parse_sbml
    from pybnf.pset import FreeParameter, PSet, TimeCourse
    out = import_job(_BOEHM_DIR / 'Boehm_JProteomeRes2014.yaml', out_dir)
    xml = str(out / 'model_Boehm_JProteomeRes2014.xml')
    ent = parse_sbml(Path(xml).read_text())
    consts, namespace = ent.constants, set(ent.namespace_symbols)
    layer_consts = {n: v for n, v in consts.items() if n not in _BOEHM_FREE}
    conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
    models = [MeasurementModel(k[1], v, namespace, layer_consts)
              for k, v in conf.items() if isinstance(k, tuple) and k[0] == 'measurement']
    pset = PSet([FreeParameter(n, 'uniform_var', 1e-6, 1e6, value=consts[n])
                 for n in _BOEHM_FREE])
    model = model_cls(xml, xml, pset=pset, actions=(TimeCourse({'time': '240', 'step': '2.5'}),))
    ds = model.execute(str(out), 'boehm', 0)
    MeasurementLayer(models).apply({model.name: ds}, {n: consts[n] for n in _BOEHM_FREE})
    return ds[next(iter(ds))]


@pytest.mark.recovery
class TestBoehmRecovery:

    def test_roadrunner_reproduces_published_data(self, tmp_path):
        pytest.importorskip('petab')
        from pybnf.pset import SbmlModelNoTimeout
        data = _simulate_boehm(SbmlModelNoTimeout, tmp_path / 'out')
        for oid, (t, y) in _boehm_measurement_data().items():
            assert oid in data.cols                       # the layer materialized the column
            pred = np.interp(t, data['time'], data[oid])
            # The model is the published optimum, so the materialized column tracks the data
            # (to its fitted noise): a high correlation and a residual small vs the data range.
            assert np.corrcoef(pred, y)[0, 1] > 0.9
            assert np.sqrt(np.mean((pred - y) ** 2)) < 0.15 * (y.max() - y.min())

    @pytest.mark.bngsim
    def test_layer_agrees_across_roadrunner_and_bngsim_on_boehm(self, tmp_path):
        """Neither backend exposes Boehm's computed observables; the measurement layer
        materializes them identically over each backend's species trajectory (ADR-0036),
        even on a real stiff model with assignment/initial-assignment rules."""
        pytest.importorskip('petab')
        from pybnf.bngsim_sbml_model import BngsimSbmlModelNoTimeout
        from pybnf.pset import SbmlModelNoTimeout
        rr = _simulate_boehm(SbmlModelNoTimeout, tmp_path / 'rr')
        bg = _simulate_boehm(BngsimSbmlModelNoTimeout, tmp_path / 'bg')
        for oid in _BOEHM_OBS:
            on_rr_grid = np.interp(rr['time'], bg['time'], bg[oid])
            np.testing.assert_allclose(on_rr_grid, rr[oid], rtol=1e-3, atol=1e-3)
