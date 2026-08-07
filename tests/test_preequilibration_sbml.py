"""Pre-equilibration on the bngsim SBML/Antimony backend (ADR-0052, #547).

ADR-0052 deferred pre-equilibration on the SBML side, and ``pset.SbmlModel`` (RoadRunner)
refuses it outright -- but the *bngsim* SBML backend neither refused it nor ran it: it
dropped ``preequilibrate:`` on the floor and simulated the bare model, so every experiment
sharing a pre-equilibration condition produced the SAME trajectory and a dose-response
collapsed to a single dose, silently (#547). These tests pin the protocol that closes that
gap: equilibrate under the ``preequilibrate:`` condition unmeasured, apply the measurement
``condition:``, then measure with the equilibrated state carried over -- one persistent
``bngsim.Simulator``, no reset between the phases.

The oracle throughout is a birth-death model with a closed-form solution. Under
``flag = f`` the ODE is ``A' = k_prod*f - k_deg*A``, so equilibrating at ``f = 1`` gives
``A_ss = k_prod/k_deg`` and a measured phase at ``flag = f`` runs

    A(t) = f*A_ss + (A_ss - f*A_ss)*exp(-k_deg*t).

The signature of the #547 defect is exactly what that formula forbids: two doses agreeing
to full precision.
"""

import math

import numpy as np
import pytest

from .context import config, parse, printing, pset
import pybnf.bngsim_sbml_model as bngsim_sbml_model


pytestmark = pytest.mark.bngsim_sbml


# A(0) = 0; birth at k_prod*flag, death at k_deg*A. ``flag`` is the dose knob the
# conditions turn; ``k_prod``/``k_deg`` are the fittable kinetics.
_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="bd">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_prod" value="3" constant="true"/>
      <parameter id="k_deg" value="2" constant="true"/>
      <parameter id="flag" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="birth" reversible="false" fast="false">
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k_prod</ci><ci>flag</ci><ci>c</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="death" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k_deg</ci><ci>A</ci><ci>c</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

_K_PROD = 3.0
_K_DEG = 2.0
_A_SS = _K_PROD / _K_DEG          # the flag=1 equilibrium, 1.5
_TIMES = [0.0, 0.25, 0.5, 1.0]


def _analytic(dose, times, a_start=_A_SS, k_prod=_K_PROD, k_deg=_K_DEG):
    """A(t) for the measured phase at ``flag = dose``, started from ``a_start``."""
    plateau = dose * k_prod / k_deg
    return np.array([plateau + (a_start - plateau) * math.exp(-k_deg * t) for t in times])


def _xml(tmp_path):
    path = tmp_path / 'bd.xml'
    path.write_text(_SBML)
    return str(path)


def _pset(**overrides):
    values = {'k_prod': _K_PROD, 'k_deg': _K_DEG}
    values.update(overrides)
    return pset.PSet([
        pset.FreeParameter(name, 'uniform_var', 1e-3, 1e3, value)
        for name, value in values.items()
    ])


def _time_course(suffix, dose, times=_TIMES, equil_fixed_time=None, equil_flag=1):
    """The measured phase of a pre-equilibration experiment: equilibrate at
    ``flag = equil_flag``, then measure at ``flag = dose`` over ``times``."""
    action = pset.TimeCourse({'suffix': suffix, 'method': 'ode'}, explicit_points=times)
    action.set_preequilibration([('param', 'flag', equil_flag)], [('param', 'flag', dose)],
                                equil_fixed_time=equil_fixed_time)
    return action


def _run(tmp_path, actions, ps=None, sensitivity_params=None, name='preequil'):
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        _xml(tmp_path), _xml(tmp_path), pset=ps or _pset(), actions=tuple(actions))
    if sensitivity_params is not None:
        model.enable_output_sensitivities(params=list(sensitivity_params))
    return model.execute(str(tmp_path), name, 1000)


# --------------------------------------------------------------------------- #
# The #547 defect itself
# --------------------------------------------------------------------------- #
def test_experiments_sharing_a_preequilibration_condition_do_not_collapse(tmp_path):
    """The regression #547 asks for: three experiments with the SAME ``preequilibrate:``
    condition and DIFFERENT measurement conditions must simulate differently.

    Before the fix all three were byte-identical (the backend ran the unperturbed model
    once per experiment), which is what made Brannmark's eight-dose dose-response collapse
    to a single dose while still reading as a plausible modelling result."""
    doses = (0.0, 1.0, 2.0)
    result = _run(tmp_path, [_time_course(f'dose{i}', d) for i, d in enumerate(doses)])

    curves = [result[f'dose{i}']['A'] for i in range(len(doses))]
    for i, dose in enumerate(doses):
        np.testing.assert_allclose(curves[i], _analytic(dose, _TIMES), rtol=1e-6, atol=1e-8)
    # ...and, stated the way the defect presents: no two doses agree.
    for i in range(len(doses)):
        for j in range(i + 1, len(doses)):
            assert not np.allclose(curves[i], curves[j]), (
                f'doses {doses[i]} and {doses[j]} simulated identically')


def test_the_measured_phase_starts_from_the_equilibrated_state(tmp_path):
    """The carry-over invariant: the measured phase's ``t = 0`` row is the equilibration's
    steady state (A_ss = 1.5), not the model's seed initial condition (A(0) = 0)."""
    result = _run(tmp_path, [_time_course('relax', 0.0)])
    assert result['relax']['A'][0] == pytest.approx(_A_SS, rel=1e-6)


def test_the_equilibration_condition_drives_the_state_carried_in(tmp_path):
    """The ``preequilibrate:`` condition is applied to the FIRST phase: equilibrating at
    ``flag = 2`` doubles the state the measurement starts from."""
    result = _run(tmp_path, [_time_course('at1', 0.0, equil_flag=1),
                             _time_course('at2', 0.0, equil_flag=2)])
    assert result['at1']['A'][0] == pytest.approx(_A_SS, rel=1e-6)
    assert result['at2']['A'][0] == pytest.approx(2 * _A_SS, rel=1e-6)


def test_a_fixed_equilibration_duration_integrates_for_that_long(tmp_path):
    """``equil_t_end:`` runs the equilibration for a fixed interval instead of relaxing to
    equilibrium, so the state carried in is the partway value A(0.5) = A_ss*(1 - e^-1)."""
    result = _run(tmp_path, [_time_course('timed', 0.0, equil_fixed_time=0.5)])
    partway = _A_SS * (1 - math.exp(-_K_DEG * 0.5))
    assert result['timed']['A'][0] == pytest.approx(partway, rel=1e-5)
    np.testing.assert_allclose(
        result['timed']['A'], _analytic(0.0, _TIMES, a_start=partway), rtol=1e-5, atol=1e-8)


def test_a_measured_steady_state_phase_relaxes_to_the_new_equilibrium(tmp_path):
    """A pre-equilibration whose measured phase is itself a steady state (ADR-0086):
    equilibrate at flag=1, switch to flag=2, relax to the NEW equilibrium (3.0)."""
    action = pset.TimeCourse({'suffix': 'ss', 'method': 'ode', 'steady_state': 1})
    action.set_preequilibration([('param', 'flag', 1)], [('param', 'flag', 2)])
    result = _run(tmp_path, [action])
    assert result['ss']['A'][-1] == pytest.approx(2 * _A_SS, rel=1e-5)


def test_a_t0_only_measurement_reads_the_post_intervention_state(tmp_path):
    """A ``t = 0``-only experiment (#510) under pre-equilibration has nothing to integrate:
    its single row is the equilibrated state the intervention left."""
    result = _run(tmp_path, [_time_course('basal', 0.0, times=[0.0])])
    data = result['basal']
    assert data.data.shape[0] == 1
    assert data['A'][0] == pytest.approx(_A_SS, rel=1e-6)


# --------------------------------------------------------------------------- #
# The pre-equilibrated dose-response scan (#474, ADR-0062)
# --------------------------------------------------------------------------- #
def test_a_preequilibrated_scan_resets_each_dose_to_the_carried_state(tmp_path):
    """Each scan point starts from the snapshot of the post-intervention state, not from the
    model's seed initial conditions -- the engine-level form of BNGL's ``saveConcentrations()``
    + ``parameter_scan(reset_conc=>1)``. With production switched off by the intervention,
    dose ``k_deg`` decays A_ss for one time unit; reset to the seed instead and every dose
    would read 0."""
    doses = [1.0, 2.0, 4.0]
    action = pset.ParamScan(
        {'suffix': 'scan', 'method': 'ode', 'param': 'k_deg', 'time': '1'},
        explicit_points=doses)
    action.set_preequilibration([('param', 'flag', 1)], [('param', 'flag', 0)])
    data = _run(tmp_path, [action])['scan']

    assert data.indvar == 'k_deg'
    np.testing.assert_allclose(
        data['A'], [_A_SS * math.exp(-d) for d in doses], rtol=1e-5, atol=1e-8)


# --------------------------------------------------------------------------- #
# The gradient path
# --------------------------------------------------------------------------- #
def test_sensitivities_are_carried_across_the_pre_equilibration_boundary(tmp_path):
    """``dA/dk_prod`` through a pre-equilibration is seeded from the equilibration's own
    ``dA_ss/dk_prod``, not from zero (bngsim's implicit-function-theorem seeding, GH #210).

    Validated twice over: against the closed form ``(1/k_deg)*exp(-k_deg*t)`` for the
    switched-off measured phase, and against a central finite difference of the trajectory."""
    def action():
        return _time_course('relax', 0.0)

    data = _run(tmp_path, [action()], sensitivity_params=['k_prod'])['relax']
    sens = data.output_sensitivities
    assert sens is not None and sens.param_names == ['k_prod']
    got = sens.slice_for('species:A')[:, 0]

    closed_form = np.array([math.exp(-_K_DEG * t) / _K_DEG for t in _TIMES])
    np.testing.assert_allclose(got, closed_form, rtol=1e-5, atol=1e-8)

    h = 1e-5
    hi = _run(tmp_path, [action()], ps=_pset(k_prod=_K_PROD + h))['relax']['A']
    lo = _run(tmp_path, [action()], ps=_pset(k_prod=_K_PROD - h))['relax']['A']
    np.testing.assert_allclose(got, (hi - lo) / (2 * h), rtol=1e-4, atol=1e-6)


def test_a_t0_only_measurement_carries_the_equilibration_sensitivity(tmp_path):
    """The ``t = 0``-only row's derivative is the equilibration's ``dA_ss/dk_prod`` = 1/k_deg
    -- read off the carried state rather than re-derived from a fresh start."""
    data = _run(tmp_path, [_time_course('basal', 0.0, times=[0.0])],
                sensitivity_params=['k_prod'])['basal']
    sens = data.output_sensitivities
    assert sens is not None
    assert sens.slice_for('species:A')[0, 0] == pytest.approx(1.0 / _K_DEG, rel=1e-5)


def test_a_measured_steady_state_phase_carries_its_sensitivity(tmp_path):
    """A steady-state measured phase (ADR-0086) is itself seeded from the equilibration:
    the NEW equilibrium at flag=2 is 2*k_prod/k_deg, so dA/dk_prod = 2/k_deg."""
    action = pset.TimeCourse({'suffix': 'ss', 'method': 'ode', 'steady_state': 1})
    action.set_preequilibration([('param', 'flag', 1)], [('param', 'flag', 2)])
    sens = _run(tmp_path, [action], sensitivity_params=['k_prod'])['ss'].output_sensitivities
    assert sens is not None
    assert sens.slice_for('species:A')[-1, 0] == pytest.approx(2.0 / _K_DEG, rel=1e-4)


def test_a_preequilibrated_scan_carries_dose_axis_sensitivities(tmp_path):
    """Every dose of a pre-equilibrated scan continues the snapshot, so its sensitivity is
    seeded from that state's dA/dk_prod rather than from zero. The snapshot equilibrated at
    the model's own k_deg, so A = (k_prod/k_deg)*e^{-dose*t} and dA/dk_prod = e^{-dose}/k_deg
    for the one-unit read."""
    doses = [1.0, 2.0, 4.0]
    action = pset.ParamScan(
        {'suffix': 'scan', 'method': 'ode', 'param': 'k_deg', 'time': '1'},
        explicit_points=doses)
    action.set_preequilibration([('param', 'flag', 1)], [('param', 'flag', 0)])
    sens = _run(tmp_path, [action], sensitivity_params=['k_prod'])['scan'].output_sensitivities
    assert sens is not None and sens.param_names == ['k_prod']
    np.testing.assert_allclose(
        sens.slice_for('species:A')[:, 0],
        [math.exp(-d) / _K_DEG for d in doses], rtol=1e-4, atol=1e-8)


def test_the_scalar_path_attaches_no_sensitivity_payload(tmp_path):
    """With the gradient path inactive a pre-equilibration carries no tensor."""
    assert _run(tmp_path, [_time_course('relax', 0.0)])['relax'].output_sensitivities is None


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #
def test_a_target_the_model_does_not_declare_is_refused(tmp_path):
    """A condition target this model has no entity for would apply to nothing, leaving the
    phase to simulate an unperturbed model -- the #547 failure mode. Refuse instead."""
    action = pset.TimeCourse({'suffix': 'relax', 'method': 'ode'}, explicit_points=_TIMES)
    action.set_preequilibration([('param', 'nope', 1)], [])
    with pytest.raises(printing.PybnfError, match="does not declare"):
        _run(tmp_path, [action])


def test_an_expression_valued_intervention_is_refused(tmp_path):
    """The BNGL titrated-competitor idiom (a ``setConcentration`` whose value is a parameter
    expression, #474) has no reading on this backend."""
    action = pset.TimeCourse({'suffix': 'relax', 'method': 'ode'}, explicit_points=_TIMES)
    action.set_preequilibration([('param', 'flag', 1)], [('species', 'A', 'k_prod*2')])
    with pytest.raises(printing.PybnfError, match="must be a number"):
        _run(tmp_path, [action])


def test_a_species_intervention_refuses_on_the_gradient_path_only(tmp_path):
    """A mid-protocol species write retires the carried sensitivity matrix, and this backend
    has no seed-row rebuild for it (ADR-0098/0101), so a gradient fit is refused rather than
    given a derivative that is wrong across the intervention. The scalar path is unaffected:
    the write itself is well defined, and the value is what a bolus should produce."""
    def _action():
        act = pset.TimeCourse({'suffix': 'bolus', 'method': 'ode'}, explicit_points=_TIMES)
        act.set_preequilibration([('param', 'flag', 1)], [('species', 'A', 5.0)])
        return act

    with pytest.raises(printing.PybnfError, match="carry forward sensitivities"):
        _run(tmp_path, [_action()], sensitivity_params=['k_prod'])

    # Scalar: the bolus sets A to 5, which then relaxes back toward A_ss.
    curve = _run(tmp_path, [_action()])['bolus']['A']
    assert curve[0] == pytest.approx(5.0, rel=1e-6)
    np.testing.assert_allclose(curve, _analytic(1.0, _TIMES, a_start=5.0), rtol=1e-5, atol=1e-8)


def test_a_species_dose_axis_refuses_on_the_gradient_path_only(tmp_path):
    """The same boundary one phase later: a pre-equilibrated scan whose dose axis is a species
    amount writes the carried state per dose, so the gradient path refuses it. Scalar runs it
    -- each dose is the written amount decaying with production switched off."""
    def _action():
        act = pset.ParamScan(
            {'suffix': 'bolus_scan', 'method': 'ode', 'param': 'A', 'time': '1'},
            explicit_points=[1.0, 2.0])
        act.set_preequilibration([('param', 'flag', 1)], [('param', 'flag', 0)])
        return act

    with pytest.raises(printing.PybnfError, match="carry forward sensitivities"):
        _run(tmp_path, [_action()], sensitivity_params=['k_prod'])

    data = _run(tmp_path, [_action()])['bolus_scan']
    np.testing.assert_allclose(
        data['A'], [d * math.exp(-_K_DEG) for d in (1.0, 2.0)], rtol=1e-5, atol=1e-8)


# --------------------------------------------------------------------------- #
# End to end, through the config surface the corpus uses
# --------------------------------------------------------------------------- #
_EXP = "# time\tA\n0\t1.5\n0.5\t1.0\n1\t0.5\n"

_CONF = """\
edition = 2
job_type = de
objective = sos
population_size = 4
max_iterations = 1
verbosity = 0
sbml_backend = bngsim
model: bd.xml
uniform_var = k_prod 0.1 10
observable: A_obs, formula: A
noise_model A_obs = gaussian, sigma = fix_at 1
condition: basal, perturbations: flag = 1
condition: off, perturbations: flag = 0
condition: high, perturbations: flag = 2
experiment: e_off, preequilibrate: basal, condition: off, method: ode, data: relax.exp
experiment: e_high, preequilibrate: basal, condition: high, method: ode, data: relax.exp
"""


def test_two_conf_experiments_sharing_a_preequilibration_condition_differ(tmp_path, monkeypatch):
    """The corpus shape end to end (Brannmark/Weber): two ``experiment:`` lines with one
    shared ``preequilibrate:`` and different measurement ``condition:``s, built through the
    real config surface. Both conditions are consumed inline, so each experiment is keyed by
    its own name -- and the two trajectories must not coincide."""
    (tmp_path / 'bd.xml').write_text(_SBML)
    (tmp_path / 'relax.exp').write_text(_EXP)
    monkeypatch.chdir(tmp_path)
    conf = config.Configuration(parse.ploop(_CONF.splitlines(keepends=True)))

    model = conf.models['bd'].copy_with_param_set(_pset())
    assert not model.mutants[0].suffix           # both conditions consumed, only wildtype left
    result = model.execute(str(tmp_path), 'conf_e2e', 1000)

    off, high = result['e_off']['A'], result['e_high']['A']
    assert not np.allclose(off, high)
    np.testing.assert_allclose(off, _analytic(0.0, [0.0, 0.5, 1.0]), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(high, _analytic(2.0, [0.0, 0.5, 1.0]), rtol=1e-5, atol=1e-8)
