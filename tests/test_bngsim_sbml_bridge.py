import os
import subprocess
import sys
import textwrap
from pathlib import Path
import types

import numpy as np
import numpy.testing as npt
import pytest

from .context import config, parse, printing, pset
import pybnf.bngsim_sbml_model as bngsim_sbml_model


# A tiny SBML model whose species S0 initial concentration is set by an
# initialAssignment referencing parameter k_init (S0(0) = 2*k_init). Used to
# pin the #415 detect-and-fallback: fitting k_init must recompute the species
# initial, which the in-place set_param path cannot do, so the bridge must fall
# back to the reload path.
_IA_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_test">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S0" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S1" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_init" value="5" constant="true"/>
      <parameter id="kf" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S0">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>k_init</ci></apply></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S0" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="S1" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>kf</ci><ci>S0</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _write_ia_model(tmp_path):
    xml_path = tmp_path / 'ia_test.xml'
    xml_path.write_text(_IA_SBML)
    return str(xml_path)


def _amount_species_sbml(compartment_attrs, extra_rules=""):
    """SBML with an amount-based catalyst species S2 in compartment c.

    S2 is hasOnlySubstanceUnits (an amount); bngsim stores it as a concentration
    (amount / size), so setting it in place must divide by the compartment size
    to match a reload. Used to pin the unit-conversion + variable-volume
    handling (#415).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="amt">
    <listOfCompartments><compartment id="c" {compartment_attrs}/></listOfCompartments>
    <listOfSpecies>
      <species id="S0" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S1" compartment="c" initialConcentration="0"  hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="c" initialAmount="4" hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.01" constant="true"/>
      <parameter id="vol" value="2" constant="true"/>
      <parameter id="k_init" value="3" constant="true"/>
    </listOfParameters>
    {extra_rules}
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S0" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="S1" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>kf</ci><ci>S0</ci><ci>S2</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# Amount-based species in a constant, non-unit (size 2) compartment.
_AMOUNT_CONST_VOL_SBML = _amount_species_sbml('size="2" constant="true"')

# Amount-based species S2 (size-2 compartment) whose initial AMOUNT is driven by
# an initialAssignment S2 = 2*k_init. Exercises unit conversion AND initial
# recompute together: reload bakes initialAmount=2*k_init -> concentration k_init.
_AMOUNT_IA_SBML = _amount_species_sbml(
    'size="2" constant="true"',
    extra_rules='<listOfInitialAssignments><initialAssignment symbol="S2">'
    '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>k_init</ci></apply></math>'
    '</initialAssignment></listOfInitialAssignments>',
)

# Same, but the compartment volume is set by an assignmentRule c := vol, so the
# unit conversion would be parameter-dependent -> must fall back to reload.
_AMOUNT_VAR_VOL_SBML = _amount_species_sbml(
    'size="2" constant="false"',
    extra_rules='<listOfRules><assignmentRule variable="c">'
    '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>vol</ci></math>'
    '</assignmentRule></listOfRules>',
)


# Like _IA_SBML, but S0's initial reads parameter P, which is itself defined by
# an assignmentRule P := 3*k_init. So S0(0) = 3*k_init only after resolving the
# rule -- exercises the transitive dependency walk and libSBML's rule-aware
# initialAssignment evaluation (#415).
_CHAINED_IA_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="chained_ia">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S0" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S1" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_init" value="5" constant="true"/>
      <parameter id="P" value="0" constant="false"/>
      <parameter id="kf" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="P">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>3</cn><ci>k_init</ci></apply></math>
      </assignmentRule>
    </listOfRules>
    <listOfInitialAssignments>
      <initialAssignment symbol="S0">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>P</ci></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S0" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="S1" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>kf</ci><ci>S0</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _raf_xml_path():
    return str(Path(__file__).resolve().parent / 'bngl_files' / 'raf.xml')


def _raf_params():
    return [
        pset.FreeParameter('K3', 'uniform_var', 2000., 10000., 8000.),
        pset.FreeParameter('K5', 'uniform_var', 0.1, 1., 0.3),
    ]


def _raf_params_mutant():
    return [
        pset.FreeParameter('K3', 'uniform_var', 2000., 10000., 2000.),
        pset.FreeParameter('K5', 'uniform_var', 0.1, 1., 0.3),
    ]


def _model_loader_config(sbml_backend='roadrunner'):
    xml_path = _raf_xml_path()
    cfg = object.__new__(config.Configuration)
    cfg._data_map = {}
    cfg.config = {
        'models': {xml_path},
        xml_path: [],
        'delete_old_files': 1,
        'wall_time_sim': 0,
        'sbml_integrator': 'cvode',
        'sbml_backend': sbml_backend,
        'smoothing': 1,
        'parallelize_models': 1,
        'fit_type': 'check',
    }
    return cfg


def test_parse_accepts_sbml_backend():
    assert parse.parse('sbml_backend = bngsim') == ['sbml_backend', 'bngsim']


def test_parse_accepts_sbml_ssa_strict():
    # parse.parse returns the raw string token; type coercion happens
    # downstream in Configuration based on parse.numkeys_int membership.
    assert parse.parse('sbml_ssa_strict = 0') == ['sbml_ssa_strict', '0']
    assert parse.parse('sbml_ssa_strict = 1') == ['sbml_ssa_strict', '1']
    assert 'sbml_ssa_strict' in parse.numkeys_int


def test_default_config_sets_sbml_ssa_strict_to_one():
    assert config.Configuration.default_config()['sbml_ssa_strict'] == 1


@pytest.mark.bngsim_sbml
def test_config_propagates_sbml_ssa_strict_default_to_bngsim_model():
    cfg = _model_loader_config(sbml_backend='bngsim')
    cfg.config['sbml_ssa_strict'] = 1

    models = cfg._load_models()

    assert models['raf'].strict_ssa is True


@pytest.mark.bngsim_sbml
def test_config_propagates_sbml_ssa_strict_override_to_bngsim_model():
    cfg = _model_loader_config(sbml_backend='bngsim')
    cfg.config['sbml_ssa_strict'] = 0

    models = cfg._load_models()

    assert models['raf'].strict_ssa is False


def test_config_rejects_sbml_ssa_strict_under_roadrunner_backend():
    cfg = object.__new__(config.Configuration)
    cfg.models = {}
    cfg.config = {
        'bng_command': '',
        'sbml_backend': 'roadrunner',
        'sbml_integrator': 'cvode',
        'sbml_ssa_strict': 0,
    }

    with pytest.raises(printing.PybnfError, match='sbml_ssa_strict.*bngsim'):
        cfg._load_simulators()


def test_make_simulator_passes_strict_ssa_for_ssa_method(monkeypatch):
    captured = {}

    class _FakeSimulator:
        def __init__(self, model, **kwargs):
            captured['model'] = model
            captured['kwargs'] = kwargs

    fake_bngsim = types.SimpleNamespace(
        Simulator=_FakeSimulator,
        SsaValidationError=None,
    )
    monkeypatch.setattr(bngsim_sbml_model, 'bngsim', fake_bngsim)

    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    model.strict_ssa = False
    model._make_simulator(object(), 'ssa')

    assert captured['kwargs'] == {'method': 'ssa', 'strict_ssa': False}


def test_make_simulator_omits_strict_ssa_for_ode_method(monkeypatch):
    captured = {}

    class _FakeSimulator:
        def __init__(self, model, **kwargs):
            captured['kwargs'] = kwargs

    fake_bngsim = types.SimpleNamespace(
        Simulator=_FakeSimulator,
        SsaValidationError=None,
    )
    monkeypatch.setattr(bngsim_sbml_model, 'bngsim', fake_bngsim)

    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    model.strict_ssa = False
    model._make_simulator(object(), 'ode')

    assert captured['kwargs'] == {'method': 'ode'}


def test_sensitivity_entity_namespace_global_params_and_species_ic():
    """The gradient router's bind-by-id namespace (#448/#455): the SBML backend reports its
    global model parameters as the parameter axis and each species as its own bare initializer,
    so a free parameter named for a global param routes to ``sensitivity_params`` and one named
    for a species routes to ``sensitivity_ic`` keyed by that species."""
    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    model._global_param_names = ('kAB', 'kBA')
    model._species_names = ('A', 'B', 'C')

    param_ids, species_initializers = model.sensitivity_entity_namespace()

    assert param_ids == ['kAB', 'kBA']
    assert species_initializers == [('A', 'A'), ('B', 'B'), ('C', 'C')]


def test_make_simulator_threads_sensitivity_request_for_ode(monkeypatch):
    """On the gradient path the SBML Simulator is built with the request's
    ``sensitivity_params`` / ``sensitivity_ic`` (#455); the scalar path (no request) is
    byte-identical (the prior tests pin the empty case)."""
    captured = {}

    class _FakeSimulator:
        def __init__(self, model, **kwargs):
            captured['kwargs'] = kwargs

    monkeypatch.setattr(bngsim_sbml_model, 'bngsim',
                        types.SimpleNamespace(Simulator=_FakeSimulator, SsaValidationError=None))

    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    model._sensitivity_request = bngsim_sbml_model._SensitivityRequest(params=['kAB'], ic=['A'])
    model._make_simulator(object(), 'ode')

    assert captured['kwargs'] == {'method': 'ode', 'sensitivity_params': ['kAB'],
                                  'sensitivity_ic': ['A']}


def test_sensitivity_request_refuses_stochastic_method():
    """Forward output sensitivities are deterministic-ODE only (#447/#455): an ssa action under an
    active gradient request is a pointed PyBNF-level error, not a backend traceback."""
    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    model.name = 'm'
    model.strict_ssa = True
    model._sensitivity_request = bngsim_sbml_model._SensitivityRequest(params=['kAB'], ic=[])
    with pytest.raises(printing.PybnfError, match='deterministic ODE'):
        model._make_simulator(object(), 'ssa')


def test_config_routes_xml_to_roadrunner_by_default():
    cfg = _model_loader_config()

    models = cfg._load_models()

    assert isinstance(models['raf'], pset.SbmlModelNoTimeout)
    assert not isinstance(models['raf'], bngsim_sbml_model.BngsimSbmlModelNoTimeout)


def test_config_routes_xml_to_sbmlmodel_when_wall_time_sim_positive():
    """RoadRunner path: wall_time_sim>0 must select SbmlModel (subprocess
    wrapper with timeout), not SbmlModelNoTimeout. Pins the post-#382
    invariant that only the RR path branches on wall_time_sim — the BNGsim
    path always uses NoTimeout regardless."""
    cfg = _model_loader_config()
    cfg.config['wall_time_sim'] = 60

    models = cfg._load_models()

    # SbmlModel is a subclass of SbmlModelNoTimeout, so check the concrete type.
    assert type(models['raf']) is pset.SbmlModel


@pytest.mark.bngsim_sbml
def test_config_routes_xml_to_bngsim_notimeout_regardless_of_wall_time_sim():
    """BNGsim SBML path always uses NoTimeout — wall_time_sim is enforced
    in-process via SimulationTimeout (issue #382)."""
    cfg = _model_loader_config(sbml_backend='bngsim')
    cfg.config['wall_time_sim'] = 60

    models = cfg._load_models()

    assert isinstance(models['raf'], bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    # Bngsim SBML path must not promote to a subprocess-timeout-wrapped class.
    assert not isinstance(models['raf'], pset.SbmlModel)


@pytest.mark.bngsim_sbml
def test_config_routes_xml_to_bngsim_when_requested():
    cfg = _model_loader_config(sbml_backend='bngsim')

    models = cfg._load_models()

    assert isinstance(models['raf'], bngsim_sbml_model.BngsimSbmlModelNoTimeout)


def test_config_rejects_bngsim_backend_without_support(monkeypatch):
    cfg = _model_loader_config(sbml_backend='bngsim')
    monkeypatch.setattr(config, 'BNGSIM_HAS_SBML', False)
    monkeypatch.setattr(config, 'BNGSIM_SBML_ERROR', 'python-libsbml is not installed')

    with pytest.raises(printing.PybnfError, match='python-libsbml is not installed'):
        cfg._load_models()


def test_subprocess_pybnf_no_bngsim_rejects_sbml_backend_bngsim():
    """`sbml_backend = bngsim` must fail at config load when PYBNF_NO_BNGSIM=1.

    Uses a subprocess because PYBNF_NO_BNGSIM is read by `_bngsim_caps` at
    import time; an in-process monkeypatch wouldn't exercise the realistic
    failure shape (`BNGSIM_SBML_ERROR` falls back to the env-var message).
    """
    xml_path = str(Path(__file__).resolve().parent / 'bngl_files' / 'raf.xml')
    script = textwrap.dedent('''
        import sys
        from pybnf import config, printing
        from pybnf._bngsim_caps import BNGSIM_AVAILABLE, BNGSIM_SBML_ERROR
        assert BNGSIM_AVAILABLE is False, 'env var did not disable bngsim'
        assert 'PYBNF_NO_BNGSIM' in BNGSIM_SBML_ERROR, (
            'BNGSIM_SBML_ERROR did not surface env-var reason: ' + repr(BNGSIM_SBML_ERROR)
        )
        xml_path = __XML_PATH__
        cfg = object.__new__(config.Configuration)
        cfg._data_map = {}
        cfg.config = {
            'models': {xml_path},
            xml_path: [],
            'delete_old_files': 1,
            'wall_time_sim': 0,
            'sbml_integrator': 'cvode',
            'sbml_backend': 'bngsim',
            'smoothing': 1,
            'parallelize_models': 1,
            'fit_type': 'check',
        }
        try:
            cfg._load_models()
        except printing.PybnfError as exc:
            msg = str(exc)
            assert 'sbml_backend = bngsim' in msg, msg
            assert 'PYBNF_NO_BNGSIM' in msg, msg
            print('OK')
            sys.exit(0)
        sys.exit('expected PybnfError but none was raised')
    ''').replace('__XML_PATH__', repr(xml_path))
    env = os.environ.copy()
    env['PYBNF_NO_BNGSIM'] = '1'
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        'stdout=%r stderr=%r' % (result.stdout, result.stderr)
    )
    assert 'OK' in result.stdout


def test_config_rejects_non_cvode_integrator_for_bngsim_backend():
    cfg = object.__new__(config.Configuration)
    cfg.models = {'raf': object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)}
    cfg.config = {
        'bng_command': '',
        'sbml_backend': 'bngsim',
        'sbml_integrator': 'rk4',
    }

    with pytest.raises(printing.PybnfError, match='cvode, gillespie'):
        cfg._load_simulators()


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_timecourse_matches_existing_expectations(tmp_path):
    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.TimeCourse({'time': '1000', 'step': '10'})
    model = bngsim_sbml_model.BngsimSbmlModel(
        xml_path,
        xml_path,
        pset=ps,
        actions=(action,),
    )

    result = model.execute(str(tmp_path), 'raf_test_exec', 1000)
    dat = result['time_course']

    assert abs(dat['RIRI'][-1] - 2.94514) < 0.01
    assert abs(dat['R'][-1] - 0.358949) < 0.01
    assert dat.cols['time'] == 0


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_param_scan_matches_existing_expectations(tmp_path):
    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.ParamScan({'param': 'K3', 'min': '500', 'max': '10000', 'step': '500', 'time': '1000'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path,
        xml_path,
        pset=ps,
        actions=(action,),
    )

    result = model.execute(str(tmp_path), 'raf_test_scan', 1000)
    dat = result['param_scan']

    assert dat.indvar == 'K3'
    assert abs(dat['I'][0] - 0.236666) < 0.01
    assert abs(dat['R'][-1] - 0.315964) < 0.01


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_param_scan_carries_dose_axis_sensitivities(tmp_path):
    """A gradient-path SBML dose-response scan carries ``∂species/∂θ`` stacked down the dose
    axis (#476) -- the SBML/Antimony twin of the net-backend plumbing.

    Each swept ``K3`` point is an independent, reset-to-seed ODE run, so its final-row forward
    sensitivity w.r.t. the fitted ``K5`` is well-posed; the scan stacks those rows into
    ``Data.output_sensitivities`` (``species:`` selectors, one row per dose). Validated against
    a central finite difference of the scan's own species columns w.r.t. ``K5``."""
    xml_path = _raf_xml_path()
    action = pset.ParamScan(
        {'param': 'K3', 'min': '2000', 'max': '8000', 'step': '2000', 'time': '1000'})

    def _run_scan(k5, with_sensitivities):
        ps = pset.PSet([
            pset.FreeParameter('K3', 'uniform_var', 2000., 10000., 8000.),
            pset.FreeParameter('K5', 'uniform_var', 0.1, 1., k5),
        ])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,))
        if with_sensitivities:
            model.enable_output_sensitivities(params=['K5'])
        return model.execute(str(tmp_path), 'raf_scan_sens', 1000)['param_scan']

    k5_0 = 0.3
    data = _run_scan(k5_0, True)
    sens = data.output_sensitivities
    assert sens is not None
    assert sens.param_names == ['K5']
    assert sens.ic_species == []
    # dose axis first, one param column, one selector per species column.
    n_doses = data.data.shape[0]
    assert sens.d_param.shape == (n_doses, len(sens.selectors), 1)
    assert all(s.startswith('species:') for s in sens.selectors)

    # Central finite difference of each scan species column w.r.t. K5, per dose row.
    h = 1e-4
    hi = _run_scan(k5_0 + h, False)
    lo = _run_scan(k5_0 - h, False)
    for col in ('R', 'I'):
        fd = (hi.data[:, hi.cols[col]] - lo.data[:, lo.cols[col]]) / (2.0 * h)
        got = sens.slice_for('species:%s' % col)[:, 0]
        np.testing.assert_allclose(got, fd, rtol=1e-3, atol=1e-3)


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_param_scan_scalar_path_carries_no_sensitivities(tmp_path):
    """With the gradient path inactive, an SBML dose-response scan has no sensitivity payload
    (scalar path byte-identical)."""
    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.ParamScan(
        {'param': 'K3', 'min': '2000', 'max': '8000', 'step': '2000', 'time': '1000'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,))
    data = model.execute(str(tmp_path), 'raf_scan_scalar', 1000)['param_scan']
    assert data.output_sensitivities is None


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_mutant_matches_existing_expectations(tmp_path):
    xml_path = _raf_xml_path()
    mut = pset.Mutation('K3', '*', 4)
    mutset = pset.MutationSet((mut,), suffix='k3x4')
    ps = pset.PSet(_raf_params_mutant())
    action = pset.TimeCourse({'time': '1000', 'step': '10'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path,
        xml_path,
        pset=ps,
        actions=(action,),
    )
    model.add_mutant(mutset)

    result = model.execute(str(tmp_path), 'raf_test_mut', 1000)
    dat = result['time_coursek3x4']

    assert abs(dat['RIRI'][-1] - 2.94514) < 0.01
    assert abs(dat['R'][-1] - 0.358949) < 0.01
    assert dat.cols['time'] == 0


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_ode_matches_roadrunner_on_raf(tmp_path):
    """Numerical parity for SBML ODE (cvode) on the existing raf.xml.

    The SSA parity test (test_bngsim_ssa_replaces_rr.py) already covers
    stochastic equivalence; this guards the deterministic path so that any
    future flip of the SBML default surfaces drift before it lands."""
    xml_path = _raf_xml_path()
    action = pset.TimeCourse({'time': '1000', 'step': '10', 'method': 'ode'})
    empty_pset = pset.PSet([])
    rr_model = pset.SbmlModelNoTimeout(
        xml_path, xml_path, pset=empty_pset, actions=(action,), integrator='cvode',
    )
    bn_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=empty_pset, actions=(action,), integrator='cvode',
    )

    rr = rr_model.execute(str(tmp_path), 'raf_rr_ode', 1000)['time_course']
    bn = bn_model.execute(str(tmp_path), 'raf_bn_ode', 1000)['time_course']

    npt.assert_allclose(bn.data[:, 0], rr.data[:, 0], rtol=0, atol=1e-9)
    for species in ('R', 'RIRI', 'I'):
        assert species in rr.cols and species in bn.cols, species
        npt.assert_allclose(bn[species], rr[species], rtol=1e-3, atol=1e-6)


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_species_ic_scan_matches_roadrunner(tmp_path):
    xml_path = _raf_xml_path()
    action = pset.ParamScan({'param': 'I', 'min': '10', 'max': '50', 'step': '10', 'time': '1000'})
    empty_pset = pset.PSet([])
    rr_model = pset.SbmlModelNoTimeout(xml_path, xml_path, pset=empty_pset, actions=(action,))
    bngsim_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path,
        xml_path,
        pset=empty_pset,
        actions=(action,),
    )

    rr_data = rr_model.execute(str(tmp_path), 'raf_rr_ic_scan', 1000)['param_scan']
    bngsim_data = bngsim_model.execute(str(tmp_path), 'raf_bngsim_ic_scan', 1000)['param_scan']

    assert rr_data.indvar == 'I_0'
    assert bngsim_data.indvar == 'I_0'
    npt.assert_allclose(bngsim_data.data[:, 0], rr_data.data[:, 0], rtol=0, atol=1e-12)
    npt.assert_allclose(bngsim_data['R'], rr_data['R'], rtol=1e-3, atol=1e-4)
    npt.assert_allclose(bngsim_data['RIRI'], rr_data['RIRI'], rtol=1e-3, atol=1e-4)


# ── wall_time_sim trip-path tests (issue #374) ──────────────────────────────


class _FakeSimulationTimeout(RuntimeError):
    def __init__(self, message, *, timeout, elapsed):
        super().__init__(message)
        self.timeout = float(timeout)
        self.elapsed = float(elapsed)


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_model_timeout_reraises_failedsimulationerror(tmp_path, monkeypatch, caplog):
    """sim.run raising SimulationTimeout becomes FailedSimulationError."""
    timeout_seen = {}

    class FakeSimulator:
        def __init__(self, engine_model, **kw):
            self.method = kw.get('method', 'ode')

        def run(self, *args, **kwargs):
            timeout_seen['timeout'] = kwargs.get('timeout')
            raise _FakeSimulationTimeout(
                'wall_time_sim exceeded', timeout=0.25, elapsed=0.40,
            )

    fake_bngsim = types.ModuleType('bngsim')
    fake_bngsim.Simulator = FakeSimulator
    fake_bngsim.SimulationTimeout = _FakeSimulationTimeout
    fake_bngsim.Model = bngsim_sbml_model.bngsim.Model
    # Preserve other attributes the SBML bridge may read at module scope
    for attr in ('SsaValidationError',):
        if hasattr(bngsim_sbml_model.bngsim, attr):
            setattr(fake_bngsim, attr, getattr(bngsim_sbml_model.bngsim, attr))

    monkeypatch.setattr(bngsim_sbml_model, 'bngsim', fake_bngsim)

    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.TimeCourse({'time': '1000', 'step': '10'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,),
    )

    caplog.set_level('WARNING')
    with pytest.raises(pset.FailedSimulationError):
        model.execute(str(tmp_path), 'sbml_trip_test', 0.25)

    assert timeout_seen.get('timeout') == 0.25, (
        f"timeout=0.25 was not forwarded; got {timeout_seen}"
    )
    log_text = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'wall_time_sim' in log_text


# ── #415: engine model loaded once, cloned per evaluation ───────────────────


@pytest.fixture
def _clear_engine_cache():
    """Reset the process-level engine template cache around a test."""
    bngsim_sbml_model._ENGINE_TEMPLATE_CACHE.clear()
    yield
    bngsim_sbml_model._ENGINE_TEMPLATE_CACHE.clear()


@pytest.mark.bngsim_sbml
def test_engine_template_loaded_once_across_evaluations(tmp_path, _clear_engine_cache):
    """The bngsim engine model (and its analytical Jacobian) is loaded once and
    reused across many objective evaluations, not re-derived per evaluation.

    Regression guard for #415: prior to the fix, execute() reloaded the model
    (libSBML parse + SymPy Jacobian) on every parameter set.
    """
    xml_path = _raf_xml_path()
    load_count = {'n': 0}
    orig_loader = bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text

    def _counting_loader(self, text):
        load_count['n'] += 1
        return orig_loader(self, text)

    action = pset.TimeCourse({'time': '100', 'step': '2'})
    base = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet(_raf_params()), actions=(action,),
    )
    # raf fits only rate constants (K3, K5), which do not feed any species
    # initial, so every evaluation takes the fast cached-clone path.
    assert base._changes_touch_initials() is False
    assert base._needs_structural_reload() is False

    bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text = _counting_loader
    try:
        last = None
        for value in (4000., 6000., 8000., 9000.):
            ps = pset.PSet([
                pset.FreeParameter('K3', 'uniform_var', 2000., 10000., value),
                pset.FreeParameter('K5', 'uniform_var', 0.1, 1., 0.3),
            ])
            model = base.copy_with_param_set(ps)
            last = model.execute(str(tmp_path), f'raf_load_count_{int(value)}', 100)
    finally:
        bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text = orig_loader

    assert load_count['n'] == 1, (
        f'engine model was loaded {load_count["n"]} times across 4 evaluations; '
        'expected exactly 1 (cached + cloned thereafter)'
    )
    # The different K3 values must still produce different trajectories -- the
    # cached model is cloned and set_param'd, not frozen at the first value.
    assert last is not None and 'time_course' in last


def _force_full_reload(model):
    """Coerce a model onto the full structural-reload path (the algebraicRule
    sentinel), to obtain the reference result the fast paths must match."""
    model._initial_dep_names = None
    assert model._needs_structural_reload() is True
    return model


@pytest.mark.bngsim_sbml
def test_fast_path_matches_full_reload_numerically(tmp_path, _clear_engine_cache):
    """The in-place cached-clone path is numerically identical to a full reload
    when no species initial is affected (#415)."""
    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.TimeCourse({'time': '1000', 'step': '10'})

    fast_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,),
    )
    assert fast_model._changes_touch_initials() is False
    fast = fast_model.execute(str(tmp_path), 'raf_fast', 1000)['time_course']

    reload_model = _force_full_reload(bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,),
    ))
    slow = reload_model.execute(str(tmp_path), 'raf_slow', 1000)['time_course']

    npt.assert_allclose(fast.data, slow.data, rtol=0, atol=0)


@pytest.mark.bngsim_sbml
def test_param_driven_initial_recomputed_without_reload(tmp_path, _clear_engine_cache):
    """Fitting a parameter that sets a species' initial via an initialAssignment
    recomputes that initial *in place* -- reusing the cached engine model (and
    its Jacobian), not reloading. The recomputed initial must track the
    parameter and match a full reload exactly (#415)."""
    xml_path = _write_ia_model(tmp_path)
    action = pset.TimeCourse({'time': '5', 'step': '5', 'method': 'ode'})

    probe = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )
    # k_init feeds S0's initialAssignment, so a change to it touches initials,
    # but this is handled in place (no full reload).
    assert probe._initial_dep_names == {'k_init'}
    assert probe._initial_expr_species == {'S0'}
    assert probe._needs_structural_reload() is False

    # The engine model is loaded exactly once across evaluations even though the
    # initial is parameter-driven (the recompute path does not reload).
    load_count = {'n': 0}
    orig_loader = bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text

    def _counting_loader(self, text):
        load_count['n'] += 1
        return orig_loader(self, text)

    bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text = _counting_loader
    results = {}
    try:
        for k in (5.0, 10.0, 7.0):
            ps = pset.PSet([pset.FreeParameter('k_init', 'uniform_var', 1., 20., k)])
            model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
                xml_path, xml_path, pset=ps, actions=(action,),
            )
            assert model._changes_touch_initials() is True
            results[k] = model.execute(str(tmp_path), f'ia_{int(k)}', 5)['time_course']
    finally:
        bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text = orig_loader

    assert load_count['n'] == 1, (
        f'engine model was loaded {load_count["n"]} times; the parameter-driven '
        'initial must be recomputed in place, not by reloading'
    )
    # S0(t=0) tracks 2*k_init (10, 20, 14), proving the initial was recomputed.
    s0_at_zero = {k: float(d['S0'][0]) for k, d in results.items()}
    assert abs(s0_at_zero[5.0] - 10.0) < 1e-9, s0_at_zero
    assert abs(s0_at_zero[10.0] - 20.0) < 1e-9, s0_at_zero
    assert abs(s0_at_zero[7.0] - 14.0) < 1e-9, s0_at_zero

    # Full numerical parity against a forced full reload, per parameter value.
    for k in (5.0, 10.0, 7.0):
        ps = pset.PSet([pset.FreeParameter('k_init', 'uniform_var', 1., 20., k)])
        ref_model = _force_full_reload(bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        ))
        ref = ref_model.execute(str(tmp_path), f'ia_ref_{int(k)}', 5)['time_course']
        npt.assert_allclose(results[k].data, ref.data, rtol=0, atol=0)


@pytest.mark.bngsim_sbml
def test_initial_assignment_independent_param_uses_fast_path(tmp_path, _clear_engine_cache):
    """A model with an initialAssignment still uses the fast path when the
    *fitted* parameter does not feed any species initial (#415)."""
    xml_path = _write_ia_model(tmp_path)
    action = pset.TimeCourse({'time': '5', 'step': '5', 'method': 'ode'})
    # Fit kf (a rate constant), which does not feed S0's initialAssignment.
    ps = pset.PSet([pset.FreeParameter('kf', 'uniform_var', 0.01, 1., 0.2)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,),
    )
    assert model._changes_touch_initials() is False
    assert model._needs_structural_reload() is False
    data = model.execute(str(tmp_path), 'ia_fast', 5)['time_course']
    # Base k_init=5 -> S0(0)=10 regardless of kf.
    assert abs(float(data['S0'][0]) - 10.0) < 1e-9


@pytest.mark.bngsim_sbml
def test_chained_assignment_rule_initial_recomputed(tmp_path, _clear_engine_cache):
    """A species initial that reads a parameter through an assignmentRule chain
    (S0 = P, P := 3*k_init) is recomputed in place and matches a full reload.
    Exercises the transitive dependency walk + libSBML's rule resolution (#415).
    """
    xml_path = tmp_path / 'chained.xml'
    xml_path.write_text(_CHAINED_IA_SBML)
    xml_path = str(xml_path)
    action = pset.TimeCourse({'time': '5', 'step': '5', 'method': 'ode'})

    probe = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )
    # k_init reaches S0's initial through the assignmentRule-defined P.
    assert 'k_init' in probe._initial_dep_names
    assert probe._initial_expr_species == {'S0'}

    for k in (5.0, 9.0):
        ps = pset.PSet([pset.FreeParameter('k_init', 'uniform_var', 1., 20., k)])
        fast_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        )
        assert fast_model._needs_structural_reload() is False
        fast = fast_model.execute(str(tmp_path), f'chain_{int(k)}', 5)['time_course']
        # S0(0) = P = 3*k_init.
        assert abs(float(fast['S0'][0]) - 3.0 * k) < 1e-9

        ref_model = _force_full_reload(bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        ))
        ref = ref_model.execute(str(tmp_path), f'chain_ref_{int(k)}', 5)['time_course']
        npt.assert_allclose(fast.data, ref.data, rtol=0, atol=0)


@pytest.mark.bngsim_sbml
def test_recompute_falls_back_to_reload_without_libsbml_transform(tmp_path, _clear_engine_cache, monkeypatch):
    """If libSBML's expandInitialAssignments is unavailable, the parameter-driven
    initial case falls back to a full reload to stay correct (#415)."""
    monkeypatch.setattr(bngsim_sbml_model, '_HAS_EXPAND_INITIAL_ASSIGNMENTS', False)
    xml_path = _write_ia_model(tmp_path)
    action = pset.TimeCourse({'time': '5', 'step': '5', 'method': 'ode'})

    load_seen = {'n': 0}
    orig_loader = bngsim_sbml_model.BngsimSbmlModelNoTimeout._load_bngsim_model_from_text

    def _counting_loader(self, text):
        load_seen['n'] += 1
        return orig_loader(self, text)

    monkeypatch.setattr(
        bngsim_sbml_model.BngsimSbmlModelNoTimeout,
        '_load_bngsim_model_from_text', _counting_loader,
    )
    last = None
    for k in (10.0, 15.0, 7.0):
        ps = pset.PSet([pset.FreeParameter('k_init', 'uniform_var', 1., 20., k)])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        )
        last = model.execute(str(tmp_path), f'ia_noexpand_{int(k)}', 5)['time_course']
    # Result is still correct (S0(0) = 2*k_init = 14) via the reload fallback...
    assert abs(float(last['S0'][0]) - 14.0) < 1e-9
    # ...and the fallback reloads per evaluation (3 reloads, no template caching),
    # unlike the in-place recompute path which would load exactly once.
    assert load_seen['n'] == 3


@pytest.mark.bngsim_sbml
def test_amount_species_nonunit_compartment_matches_reload(tmp_path, _clear_engine_cache):
    """An amount-based species in a constant non-unit compartment is set in
    place with the right unit conversion (amount / size), matching a full
    reload (#415)."""
    xml_path = tmp_path / 'amt_const.xml'
    xml_path.write_text(_AMOUNT_CONST_VOL_SBML)
    xml_path = str(xml_path)
    action = pset.TimeCourse({'time': '3', 'step': '3', 'method': 'ode'})

    probe = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )
    assert probe._unsafe_volume is False
    assert probe._species_unit_factor['S2'] == 0.5   # 1 / size(=2)
    assert probe._species_unit_factor['S0'] == 1.0   # concentration-based

    for amt in (4.0, 6.0):
        # Fit the amount-based species S2 directly (PyBNF value is an amount).
        ps = pset.PSet([pset.FreeParameter('S2', 'uniform_var', 1., 10., amt)])
        fast_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        )
        assert fast_model._needs_structural_reload() is False
        fast = fast_model.execute(str(tmp_path), f'amt_fast_{int(amt)}', 3)['time_course']

        ref_model = _force_full_reload(bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        ))
        ref = ref_model.execute(str(tmp_path), f'amt_ref_{int(amt)}', 3)['time_course']
        npt.assert_allclose(fast.data, ref.data, rtol=0, atol=0)


@pytest.mark.bngsim_sbml
def test_variable_volume_compartment_forces_reload(tmp_path, _clear_engine_cache):
    """An amount-based species in a non-constant-volume compartment can't be
    safely unit-converted in place, so the bridge falls back to a full reload --
    and the result stays correct (#415)."""
    xml_path = tmp_path / 'amt_var.xml'
    xml_path.write_text(_AMOUNT_VAR_VOL_SBML)
    xml_path = str(xml_path)
    action = pset.TimeCourse({'time': '3', 'step': '3', 'method': 'ode'})

    probe = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )
    assert probe._unsafe_volume is True
    assert probe._needs_structural_reload() is True

    ps = pset.PSet([pset.FreeParameter('S2', 'uniform_var', 1., 10., 6.0)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,),
    )
    fast = model.execute(str(tmp_path), 'amt_var', 3)['time_course']
    ref_model = _force_full_reload(bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=ps, actions=(action,),
    ))
    ref = ref_model.execute(str(tmp_path), 'amt_var_ref', 3)['time_course']
    npt.assert_allclose(fast.data, ref.data, rtol=0, atol=0)


@pytest.mark.bngsim_sbml
def test_amount_species_initial_assignment_recompute_with_units(tmp_path, _clear_engine_cache):
    """Unit conversion AND initial recompute together: an amount-based species
    (size-2 compartment) whose initialAmount is set by initialAssignment
    2*k_init. Recomputed in place, matching a full reload (#415)."""
    xml_path = tmp_path / 'amt_ia.xml'
    xml_path.write_text(_AMOUNT_IA_SBML)
    xml_path = str(xml_path)
    action = pset.TimeCourse({'time': '3', 'step': '3', 'method': 'ode'})

    probe = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )
    assert probe._initial_expr_species == {'S2'}
    assert 'k_init' in probe._initial_dep_names
    assert probe._species_unit_factor['S2'] == 0.5
    assert probe._unsafe_volume is False

    for k in (3.0, 5.0):
        ps = pset.PSet([pset.FreeParameter('k_init', 'uniform_var', 1., 10., k)])
        fast_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        )
        assert fast_model._needs_structural_reload() is False
        assert fast_model._changes_touch_initials() is True
        fast = fast_model.execute(str(tmp_path), f'amt_ia_{int(k)}', 3)['time_course']
        # S2(0) concentration = initialAmount(2*k_init) / size(2) = k_init.
        assert abs(float(fast['S2'][0]) - k) < 1e-9

        ref_model = _force_full_reload(bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml_path, xml_path, pset=ps, actions=(action,),
        ))
        ref = ref_model.execute(str(tmp_path), f'amt_ia_ref_{int(k)}', 3)['time_course']
        npt.assert_allclose(fast.data, ref.data, rtol=0, atol=0)


# ── #469/#470: SBML/Antimony fits honor the experiment's measurement grid ────
#
# A one-step decay A -> B with rate k*A, so A(t) = 100*exp(-k t). The experiment's
# measurement times are threaded into bngsim's sample_times; before the fix the
# SBML path simulated a uniform integer grid and non-grid times (e.g. t=0.5) were
# never in the output, so scoring raised / scored inf.
_DECAY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


class _CapturingSim:
    def __init__(self):
        self.captured = {}

    def run(self, **kwargs):
        self.captured.update(kwargs)
        return object()


def test_run_simulation_threads_sample_times_into_bngsim_run(monkeypatch):
    """Regression for #469/#470: when the action carries explicit measurement
    points, ``_run_simulation`` passes them to bngsim's ``run(sample_times=...)``
    (with a matching t_span/n_points) instead of a uniform grid."""
    sim = _CapturingSim()
    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    monkeypatch.setattr(model, '_make_simulator', lambda engine_model, method: sim)

    model._run_simulation(object(), 2.5, 3, method='ode',
                          sample_times=[0.0, 0.5, 1.5, 2.5])

    assert sim.captured['sample_times'] == [0.0, 0.5, 1.5, 2.5]
    assert sim.captured['t_span'] == (0.0, 2.5)
    assert sim.captured['n_points'] == 4


def test_run_simulation_uniform_grid_without_sample_times(monkeypatch):
    """The legacy uniform-grid path is unchanged: no ``sample_times`` is passed
    when the action carries none (issue #469/#470 must not alter the default)."""
    sim = _CapturingSim()
    model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    monkeypatch.setattr(model, '_make_simulator', lambda engine_model, method: sim)

    model._run_simulation(object(), 8.0, 9, method='ode')

    assert 'sample_times' not in sim.captured
    assert sim.captured['t_span'] == (0.0, 8.0)
    assert sim.captured['n_points'] == 9


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_timecourse_honors_non_integer_sample_times(tmp_path):
    """End-to-end regression for #469/#470: an SBML time course whose experiment
    supplies non-integer measurement times outputs at exactly those times, so a
    point at t=0.5 is present and scores correctly (previously the uniform integer
    grid dropped it and scoring failed)."""
    xml_path = tmp_path / 'decay.xml'
    xml_path.write_text(_DECAY_SBML)
    xml_path = str(xml_path)

    times = [0.5, 1.5, 2.5]
    action = pset.TimeCourse({}, explicit_points=times)
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )

    data = model.execute(str(tmp_path), 'decay_nonint', 10)['time_course']

    # explicit_points forces a leading t=0, so the sim outputs {0, 0.5, 1.5, 2.5}.
    sim_times = list(data['time'])
    for t in (0.0, 0.5, 1.5, 2.5):
        assert any(abs(st - t) < 1e-9 for st in sim_times), (t, sim_times)
    # A(t) = 100*exp(-0.5 t) at each non-integer measurement time.
    for t in times:
        idx = min(range(len(sim_times)), key=lambda i: abs(sim_times[i] - t))
        npt.assert_allclose(data['A'][idx], 100.0 * np.exp(-0.5 * t), rtol=1e-4)


@pytest.mark.bngsim_sbml
def test_bngsim_sbml_param_scan_honors_explicit_values(tmp_path):
    """Companion regression for #469/#470: an SBML parameter scan whose experiment
    supplies explicit (non-grid) swept values sweeps exactly those values rather
    than a uniform linspace grid."""
    xml_path = tmp_path / 'decay.xml'
    xml_path.write_text(_DECAY_SBML)
    xml_path = str(xml_path)

    values = [0.3, 0.55, 0.8]
    action = pset.ParamScan({'param': 'k', 'time': '2'}, explicit_points=values)
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path, xml_path, pset=pset.PSet([]), actions=(action,),
    )

    data = model.execute(str(tmp_path), 'decay_scan', 10)['param_scan']

    assert data.indvar == 'k'
    npt.assert_allclose(sorted(data['k']), values, rtol=0, atol=1e-12)
    # A(t=2) = 100*exp(-2k) at each swept k, in swept order.
    order = np.argsort(data['k'])
    npt.assert_allclose(
        np.asarray(data['A'])[order],
        100.0 * np.exp(-2.0 * np.asarray(values)),
        rtol=1e-4,
    )
