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
