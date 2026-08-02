"""Gradient-based fitting of an SBML model through the bngsim backend (#482).

The BNGL net backend's gradient path (forward output sensitivities -> residual
Jacobian -> TRF / L-BFGS-B) was complete, but the SBML backend was missing the
#475 scored-suffix sensitivity gate that the gradient runner calls on every
model (``set_scored_suffixes``), so an SBML gradient fit crashed at setup with an
``AttributeError``. That gate now lives on :class:`~pybnf.bngsim_sbml_model.BngsimSbmlModelNoTimeout`
(and, by inheritance, the Antimony backend). These tests lock in the finished path:

* **backend scored gate** -- the SBML twin of the net tests in
  ``test_bngsim_output_sensitivities.py``: the gradient path carries the native
  ``∂(species)/∂θ`` tensor (parameter + initial-condition axes) matched to the
  analytic decay derivative; the scalar path is byte-unchanged; and an
  incidental/unscored stochastic action runs sensitivity-free while a *scored* one
  refuses cleanly;
* **assembly FD oracle** -- central differences of PyBNF's own ``loss(u)`` vs the
  assembled ``gradient(u)`` for a two-experiment (wildtype + a ``k``-scaled
  condition), two-free-parameter SBML fit, exercising both sensitivity axes, the
  per-condition factor, and the cross-experiment sum through the ``species:``
  selectors the SBML backend labels its tensor with;
* **setup regression guard** -- the exact crash site (``_setup_gradient_path`` ->
  ``set_scored_suffixes``) driven on a real SBML ``TRFAlgorithm`` in default CI, so
  the AttributeError cannot silently return (the original bug lived only in the
  opt-in ``recovery`` tier, which is why it slipped through); and
* **end-to-end recovery** (opt-in ``recovery`` tier) -- ``trf`` and ``lbfgs``
  recover a small SBML model's rate + initial condition from zero-noise data,
  including through a measurement-model **formula** observable (the
  ``prediction_sensitivity`` seam, ADR-0036).

The fixture is a tiny exponential-decay SBML model (``dS/dt = -k S``, ``S(0) = S0``)
whose closed-form solution ``S(t) = S0·exp(-k·t)`` gives analytic sensitivity
oracles ``∂S/∂k = -t·S0·exp(-k·t)`` and ``∂S/∂S0 = exp(-k·t)`` -- the SBML peer of
the BNGL ``e2e_ode_decay.net`` the net tests use.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.data import Data
from pybnf.gradient import (
    IC, PARAM, RouteContribution, SeedTerm,
    assemble_gaussian_gradient, route_experiment, route_for_model,
)
from pybnf.gradient import derivative
from pybnf.gradient.derivative import ONE
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import (
    FreeParameter, Mutation, MutationSet, PSet, TimeCourse,
)

from . import recovery_harness as H


# Every test here runs a real bngsim SBML ODE solve, so gate on the SBML backend
# (auto-skips via conftest when it is unavailable). The sensitivity-bearing tests
# additionally need the output_sensitivities feature (guarded per-test below).
pytestmark = pytest.mark.bngsim_sbml

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')


# --- fixture: a tiny exponential-decay SBML model ------------------------------ #
# One species S (a concentration), one global rate k, one degradation reaction
# S -> (rate k*S). S(t) = S0*exp(-k*t). The species id doubles as the bind-by-id
# name for its initial-condition free parameter (ADR-0034): a free parameter named
# 'S' routes to the sensitivity_ic axis keyed by species S; a free parameter named
# 'k' routes to the sensitivity_params axis.
_DECAY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sbml_decay">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

TRUE_K = 0.3
TRUE_S0 = 100.0


def _write_decay_sbml(tmp_path):
    """Write the decay SBML fixture into ``tmp_path`` and return its path."""
    xml = Path(tmp_path) / 'decay.xml'
    xml.write_text(_DECAY_SBML)
    return str(xml)


def _decay_model(tmp_path, *, k=TRUE_K, s0=TRUE_S0, actions=None, integrator='cvode'):
    """A :class:`BngsimSbmlModelNoTimeout` over the decay fixture at ``(k, s0)``.

    Default action is a single ``t_end=10`` unit-step ODE time course (suffix
    ``time_course``); pass ``actions`` to override (e.g. a mixed ode/ssa pair)."""
    xml = _write_decay_sbml(tmp_path)
    ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
               FreeParameter('S', 'uniform_var', 0.0, 1e6, value=s0)])
    if actions is None:
        actions = (TimeCourse({'time': '10', 'step': '1'}),)
    return bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=actions, integrator=integrator)


# --- backend scored gate + sensitivity tensor ---------------------------------- #
@_needs_output_sens
def test_sbml_gradient_path_carries_native_parameter_and_ic_tensor(tmp_path):
    """The gradient path carries ``∂S/∂θ`` on both axes, matched to the analytic decay.

    The SBML timecourse peer of the net ``test_gradient_path_carries_*`` tests: with a
    parameter (``k``) and an initial-condition (``S``) request active, the returned Data
    carries a ``species:S`` sensitivity column on the ``parameter`` and ``ic`` axes,
    equal to ``∂S/∂k = -t·S0·exp(-k·t)`` and ``∂S/∂S0 = exp(-k·t)`` respectively."""
    model = _decay_model(tmp_path)
    model.enable_output_sensitivities(params=['k'], ic=['S'])
    data = model.execute(str(tmp_path), 'decay_grad', 0)['time_course']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['species:S']
    assert sens.param_names == ['k']
    assert sens.ic_species == ['S']

    t = np.asarray(data['time'])
    d_k = sens.slice_for('species:S', axis='parameter')[:, 0]
    d_s0 = sens.slice_for('species:S', axis='ic')[:, 0]
    np.testing.assert_allclose(d_k, -t * TRUE_S0 * np.exp(-TRUE_K * t), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(d_s0, np.exp(-TRUE_K * t), rtol=1e-3, atol=1e-4)


def test_sbml_scalar_path_carries_no_sensitivities(tmp_path):
    """With no sensitivity request, the SBML Data carries no tensor (scalar path intact)."""
    model = _decay_model(tmp_path)
    data = model.execute(str(tmp_path), 'decay_scalar', 0)['time_course']
    assert data.output_sensitivities is None


@_needs_output_sens
def test_sbml_time_zero_only_returns_initial_state_and_sensitivities(tmp_path):
    """A t=0-only condition is a one-row initial-state observation, including on gntr."""
    action = TimeCourse({'suffix': 'initial'}, explicit_points=[0])
    model = _decay_model(tmp_path, actions=(action,))
    model.enable_output_sensitivities(params=['k'], ic=['S'])

    data = model.execute(str(tmp_path), 'decay_initial', 0)['initial']

    np.testing.assert_array_equal(data['time'], [0.0])
    np.testing.assert_array_equal(data['S'], [TRUE_S0])
    sens = data.output_sensitivities
    assert sens is not None
    np.testing.assert_array_equal(
        sens.slice_for('species:S', axis='parameter'), [[0.0]])
    np.testing.assert_array_equal(
        sens.slice_for('species:S', axis='ic'), [[1.0]])


@_needs_output_sens
def test_sbml_time_zero_only_differentiates_initial_assignment(tmp_path):
    """The t=0 tensor includes parameter derivatives baked into SBML initials (#510)."""
    from .test_bngsim_sbml_bridge import _IA_SBML

    xml = tmp_path / 'initial_assignment.xml'
    xml.write_text(_IA_SBML)
    ps = PSet([FreeParameter('k_init', 'uniform_var', 1.0, 20.0, value=5.0)])
    action = TimeCourse({'suffix': 'initial'}, explicit_points=[0])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=ps, actions=(action,))
    model.enable_output_sensitivities(params=['k_init'])

    data = model.execute(str(tmp_path), 'initial_assignment', 0)['initial']

    np.testing.assert_allclose(data['S0'], [10.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        data.output_sensitivities.slice_for('species:S0', axis='parameter'),
        [[2.0]], rtol=1e-7, atol=1e-9)


@_needs_output_sens
def test_sbml_unscored_stochastic_action_runs_sensitivity_free(tmp_path):
    """An UNSCORED ssa diagnostic beside a scored ODE course no longer aborts (#475/#482).

    The gradient fit's scored target is the ODE ``tc`` course; the incidental ``diag`` ssa
    run is never scored against data, so it must run on the ordinary path -- carrying no
    sensitivity tensor -- rather than aborting the whole fit at the ODE-only guard."""
    model = _decay_model(tmp_path, actions=(
        TimeCourse({'time': '5', 'step': '1', 'suffix': 'tc'}),
        TimeCourse({'time': '5', 'step': '1', 'suffix': 'diag', 'method': 'ssa'}),
    ))
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})   # only 'tc' is a scored gradient target

    ds = model.execute(str(tmp_path), 'sbml_mixed', 0)
    # The scored ODE course carries its native parameter tensor...
    assert ds['tc'].output_sensitivities is not None
    assert ds['tc'].output_sensitivities.param_names == ['k']
    # ...while the unscored stochastic diagnostic ran sensitivity-free.
    assert ds['diag'].output_sensitivities is None


@_needs_output_sens
def test_sbml_scored_stochastic_action_refuses_cleanly(tmp_path):
    """A *scored* ssa simulate() under a gradient fit surfaces a PyBNF-level message."""
    model = _decay_model(tmp_path, actions=(
        TimeCourse({'time': '5', 'step': '1', 'suffix': 'tc', 'method': 'ssa'}),
    ))
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})   # the ssa output IS a scored target
    with pytest.raises(PybnfError) as exc:
        model.execute(str(tmp_path), 'sbml_ssa_scored', 0)
    msg = str(exc.value).lower()
    assert 'ode' in msg and 'ssa' in msg


@_needs_output_sens
def test_sbml_unscored_non_ode_kwargs_are_empty_not_refused(tmp_path):
    """The kwargs helper: unscored non-ODE -> {}; scored non-ODE -> refuse; ODE bears."""
    model = _decay_model(tmp_path)
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})

    model._current_action_suffix = 'tc'          # scored
    assert model._sensitivity_request_kwargs('ode') == {'sensitivity_params': ['k']}
    with pytest.raises(PybnfError):
        model._sensitivity_request_kwargs('ssa')  # scored non-ODE still refuses

    model._current_action_suffix = 'diag'        # unscored
    assert model._sensitivity_request_kwargs('ssa') == {}      # runs sensitivity-free
    # An ODE action is always sensitivity-bearing, scored or not (matches the net backend).
    assert model._sensitivity_request_kwargs('ode') == {'sensitivity_params': ['k']}


@_needs_output_sens
def test_sbml_scored_gate_folds_in_condition_suffix(tmp_path):
    """The SBML gate keys an action by its full output suffix ``act.suffix + mut.suffix``.

    The SBML backend's mutant loop is inline in ``execute`` (unlike the net backend's
    separate mutant-model object + ``_sensitivity_offset``), so the condition suffix is
    folded straight into ``_current_action_suffix``. Only that full key is tested against
    the scored set."""
    model = _decay_model(tmp_path)
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc_cond'})   # only the conditioned output is scored

    model._current_action_suffix = 'tc'          # bare wildtype suffix: not scored
    assert model._action_bears_sensitivities() is False
    model._current_action_suffix = 'tc_cond'     # action-suffix + condition-suffix: scored
    assert model._action_bears_sensitivities() is True


def test_sbml_scalar_path_action_never_bears_sensitivities(tmp_path):
    """With the request inactive, no action bears sensitivities (scalar path intact)."""
    model = _decay_model(tmp_path)
    model.set_scored_suffixes({'tc'})            # a scored set with no active request
    model._current_action_suffix = 'tc'
    assert model._action_bears_sensitivities() is False


# --- assembly FD oracle -------------------------------------------------------- #
def _exp_from_species(sim, sigma):
    """Decay-SBML experimental Data from a run's exact (time, S) grid, with a constant
    ``S_SD`` column for the chi_sq fixed-sigma source."""
    t = sim.data[:, sim.cols['time']]
    obs = sim.data[:, sim.cols['S']]
    sd = np.full(len(obs), sigma, float)
    return Data.from_columns(np.column_stack([t, obs, sd]), ['time', 'S', 'S_SD'])


def _assert_fd_matches(tmp_path, xml, tag, *, cond, free, model_params, truth, column,
                       sigma, expect_route=None, rtol=2e-3, atol=2e-3):
    """Central differences of ``loss(u)`` vs the assembled ``gradient(u)`` for free parameters
    that reach the model only through ``cond`` (a per-condition estimated initial condition).

    ``model_params(theta)`` maps the free-parameter values to the model parameter values the
    condition would set, so the forward runs need no mutant machinery; the routing itself is
    still built from ``cond`` against the live model, which is what is under test.
    """
    action = TimeCourse({'time': '10', 'step': '1'})

    def run(values, route=None):
        ps = PSet([FreeParameter(k, 'uniform_var', 0.0, 1e12, value=v)
                   for k, v in values.items()])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml, xml, pset=ps, actions=(action,))
        if route is not None:
            model.enable_output_sensitivities(
                params=route.sensitivity_params, ic=route.sensitivity_ic)
        return model.execute(str(tmp_path), tag, 0)['time_course']

    base = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=PSet([]), actions=(action,))
    names = [p.name for p in free]
    route = route_for_model(base, names, cond)
    if expect_route is not None:
        expect_route(route)

    sim_truth = run(truth)
    t = sim_truth.data[:, sim_truth.cols['time']]
    obs = sim_truth.data[:, sim_truth.cols[column]]
    exp = Data.from_columns(
        np.column_stack([t, obs, np.full(len(obs), sigma, float)]),
        ['time', column, '%s_SD' % column])
    obj = ChiSquareObjective()

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        return obj.evaluate(run(model_params(theta)), exp)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-6
    grad_fd = np.array([
        (loss_at(_bump(u0, j, h)) - loss_at(_bump(u0, j, -h))) / (2.0 * h)
        for j in range(len(free))])

    theta0 = {p.name: p.value for p in free}
    point = route.at_point(theta0)
    sim = run(model_params(theta0), route=point)
    res = assemble_gaussian_gradient(obj, [(sim, exp, point)], free)
    assert res.param_names == names
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=rtol, atol=atol)
    return route


def _bump(u0, j, h):
    u = u0.copy()
    u[j] += h
    return u


@_needs_output_sens
def test_sbml_fd_acceptance_gate(tmp_path):
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the decay
    SBML model -- the SBML twin of ``test_gradient_assembly.test_fd_acceptance_gate``. Two
    experiments (wildtype + a ``k*4`` condition), two free parameters (``k`` on the parameter
    axis, ``S`` on the initial-condition axis), so it exercises both sensitivity axes, the
    per-condition factor, the cross-experiment sum, and the ``species:`` selectors the SBML
    backend labels its tensor with -- the downstream residual/Jacobian assembly for SBML."""
    xml = _write_decay_sbml(tmp_path)

    def run(k, s0, with_sensitivities):
        ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
                   FreeParameter('S', 'uniform_var', 0.0, 1e6, value=s0)])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml, xml, pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
        if with_sensitivities:
            model.enable_output_sensitivities(params=['k'], ic=['S'])
        return model.execute(str(tmp_path), 'fd', 0)['time_course']

    obj = ChiSquareObjective()
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0   # the 'hi' condition: k * 4

    # Synthetic data: each experiment's own simulated trajectory at the *true* params, so
    # residuals at the evaluation point (k=0.4, S0=120) are non-zero -> a non-trivial gradient.
    k_true, s0_true, sigma = 0.3, 100.0, 5.0
    exp_wt = _exp_from_species(run(k_true, s0_true, False), sigma)
    exp_hi = _exp_from_species(run(k_factor * k_true, s0_true, False), sigma)

    # Per-experiment routing (factors): wildtype k=1, condition k=4; S is an unperturbed IC.
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['k'], [('S', 'S')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim_wt = run(theta['k'], theta['S'], False)
        sim_hi = run(k_factor * theta['k'], theta['S'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-5
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)

    sim_wt = run(free[0].value, free[1].value, True)
    sim_hi = run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


# --- #511: per-condition estimated initial conditions through a condition ------ #
# A mini-Bruno: species S's initial value is seeded by a bare initialAssignment from global
# param S0, and two decay channels carry rate multipliers kmult1/kmult2. A condition sets S0
# and BOTH multipliers to the value of free parameters (a per-condition estimated initial
# condition + a shared multiplier, ADR-0076), so the free parameters reach the model ONLY
# through the condition -- S(t) = S0*exp(-(k1*kmult1 + k2*kmult2)*t).
_SEEDED_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="seeded_decay">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="100" constant="true"/>
      <parameter id="k1" value="0.3" constant="true"/>
      <parameter id="k2" value="0.2" constant="true"/>
      <parameter id="kmult1" value="1" constant="true"/>
      <parameter id="kmult2" value="1" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>S0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k1</ci><ci>kmult1</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="r2" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k2</ci><ci>kmult2</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@_needs_output_sens
def test_sbml_ic_seed_map_exposes_bare_initial_assignment(tmp_path):
    """The SBML namespace exposes ``{S0 -> S}`` for the bare ``initialAssignment`` ``S = S0``,
    so the router can compose a per-condition estimated initial condition (ADR-0076, #511)."""
    xml = Path(tmp_path) / 'seeded.xml'
    xml.write_text(_SEEDED_SBML)
    ps = PSet([FreeParameter('k1', 'uniform_var', 0.0, 1e6, value=0.3)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
    param_ids, species, ic_seed_map = model.sensitivity_entity_namespace()
    assert set(param_ids) == {'S0', 'k1', 'k2', 'kmult1', 'kmult2'}
    assert ic_seed_map == {'S0': (SeedTerm(IC, 'S', ONE),)}


@_needs_output_sens
def test_sbml_fd_oracle_free_params_routed_through_a_condition(tmp_path):
    """Central-difference FD of loss(u) vs the assembled gradient(u) when the free parameters
    reach the model ONLY through a condition (ADR-0076, #511): ``s0_free`` sets the IC-seeding
    param S0 (a per-condition estimated initial condition -> IC axis), and ``m_free`` sets BOTH
    rate multipliers at once (a shared multiplier -> the SUM of two parameter-axis columns).
    ``k1`` binds by id. The pre-fix gradient path aborted on this; here its gradient must match
    finite differences on both composed axes."""
    xml_path = Path(tmp_path) / 'seeded.xml'
    xml_path.write_text(_SEEDED_SBML)
    xml = str(xml_path)

    def run(s0, m, k1, with_sensitivities, route=None):
        # The condition sets S0=s0 (seeds S's IC), kmult1=kmult2=m (both channels), k1=k1.
        ps = PSet([FreeParameter('S0', 'uniform_var', 0.0, 1e6, value=s0),
                   FreeParameter('kmult1', 'uniform_var', 0.0, 1e6, value=m),
                   FreeParameter('kmult2', 'uniform_var', 0.0, 1e6, value=m),
                   FreeParameter('k1', 'uniform_var', 0.0, 1e6, value=k1)])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml, xml, pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
        if with_sensitivities:
            model.enable_output_sensitivities(
                params=route.sensitivity_params, ic=route.sensitivity_ic)
        return model.execute(str(tmp_path), 'fd511', 0)['time_course']

    obj = ChiSquareObjective()
    free = [FreeParameter('s0_free', 'uniform_var', 0.0, 1000.0, value=120.0),
            FreeParameter('m_free', 'uniform_var', 0.01, 100.0, value=0.8),
            FreeParameter('k1', 'uniform_var', 0.01, 100.0, value=0.4)]
    names = [p.name for p in free]

    # The routing comes from the live model's namespaces (the SBML ic_seed_map end-to-end): the
    # condition param-refs S0/kmult1/kmult2 to free parameters.
    cond = MutationSet([
        Mutation('S0', '=', 's0_free', is_param_ref=True),
        Mutation('kmult1', '=', 'm_free', is_param_ref=True),
        Mutation('kmult2', '=', 'm_free', is_param_ref=True),
    ], 'c')
    route_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=PSet([FreeParameter('k1', 'uniform_var', 0.0, 1e6, value=0.3)]),
        actions=(TimeCourse({'time': '10', 'step': '1'}),))
    route = route_for_model(route_model, names, cond)
    # s0_free -> IC(S); m_free -> PARAM(kmult1) + PARAM(kmult2) (summed); k1 -> PARAM(k1).
    assert route.sensitivity_ic == ['S']
    assert set(route.sensitivity_params) == {'kmult1', 'kmult2', 'k1'}
    assert len(route.routes['m_free'].contributions) == 2

    # Synthetic data at the true params -> non-zero residuals at the evaluation point.
    s0_true, m_true, k1_true, sigma = 100.0, 1.0, 0.3, 5.0
    exp = _exp_from_species(run(s0_true, m_true, k1_true, False), sigma)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        return obj.evaluate(run(theta['s0_free'], theta['m_free'], theta['k1'], False), exp)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-6
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)

    sim = run(free[0].value, free[1].value, free[2].value, True, route=route)
    res = assemble_gaussian_gradient(obj, [(sim, exp, route)], free)

    assert res.param_names == names
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=2e-3, atol=2e-3)


_NONBARE_SEED_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="nonbare_seed">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="100" constant="true"/>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>S0</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@_needs_output_sens
def test_sbml_non_bare_initial_assignment_seed_carries_its_derivative(tmp_path):
    """A **non-bare** initialAssignment ``S = 2*S0`` has ``d(IC)/d(S0) = 2``: the seed is
    routable and carries that factor rather than refusing (ADR-0076, #530). #511 could only
    express a derivative of 1, so this shape lost the gradient path entirely."""
    xml_path = Path(tmp_path) / 'nonbare.xml'
    xml_path.write_text(_NONBARE_SEED_SBML)
    xml = str(xml_path)
    ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=0.3)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
    _, _, ic_seed_map = model.sensitivity_entity_namespace()
    assert ic_seed_map == {'S0': (SeedTerm(IC, 'S', ('num', 2.0)),)}
    cond = MutationSet([Mutation('S0', '=', 's0_free', is_param_ref=True)], 'c')
    route = route_for_model(model, ['s0_free'], cond)
    assert route.routes['s0_free'].contributions == (RouteContribution(IC, 'S', 2.0),)


_UNDIFFERENTIABLE_SEED_SBML = _NONBARE_SEED_SBML.replace(
    '<apply><times/><cn>2</cn><ci>S0</ci></apply>',
    '<apply><exp/><ci>S0</ci></apply>')


@_needs_output_sens
def test_sbml_seed_outside_the_arithmetic_grammar_is_refused(tmp_path):
    """``S = exp(S0)`` is outside the seed grammar, so S0 stays non-routable and the
    per-condition estimated initial condition refuses rather than guessing (#530)."""
    from pybnf.gradient import GradientNotSupported
    xml_path = Path(tmp_path) / 'undiff.xml'
    xml_path.write_text(_UNDIFFERENTIABLE_SEED_SBML)
    xml = str(xml_path)
    ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=0.3)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
    _, _, ic_seed_map = model.sensitivity_entity_namespace()
    assert ic_seed_map == {'S0': None}
    cond = MutationSet([Mutation('S0', '=', 's0_free', is_param_ref=True)], 'c')
    with pytest.raises(GradientNotSupported, match='cannot differentiate'):
        route_for_model(model, ['s0_free'], cond)


_AMOUNT_SEED_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="amount_seed">
    <listOfCompartments><compartment id="c" size="2" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="100" constant="true"/>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>S0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@_needs_output_sens
def test_sbml_amount_species_seed_folds_the_unit_conversion(tmp_path):
    """A bare initialAssignment on an amount species in a size-2 compartment.

    The assignment sets an *amount* (``hasOnlySubstanceUnits``) and the sensitivity tensor is
    already in PyBNF species-value units -- also an amount here -- so the two volume factors
    cancel and ``d(value)/d(S0)`` is 1 after all. #511 read only the raw unit factor (0.5) and
    refused this seed outright; #530 composes both halves, and the FD oracle below is what
    settles which is right."""
    xml_path = Path(tmp_path) / 'amount.xml'
    xml_path.write_text(_AMOUNT_SEED_SBML)
    xml = str(xml_path)
    ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=0.3)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
    assert model._species_unit_factor['S'] == 0.5              # PyBNF value -> concentration
    assert model._species_assignment_to_concentration['S'] == 0.5   # assignment -> concentration
    _, _, ic_seed_map = model.sensitivity_entity_namespace()
    assert ic_seed_map == {'S0': (SeedTerm(IC, 'S', ONE),)}
    cond = MutationSet([Mutation('S0', '=', 's0_free', is_param_ref=True)], 'c')
    route = route_for_model(model, ['s0_free'], cond)
    assert route.routes['s0_free'].contributions == (RouteContribution(IC, 'S', 1.0),)


# The Bertozzi_PNAS2020 shape, minimised: one estimated parameter I0 seeds TWO species
# initials with opposite signs (I = I0, S = N - I0), and a derived parameter beta = R0*g/N
# -- fixed by an initialAssignment, not by the ODE -- carries R0 and g into the rate law.
_SEIR_LIKE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="seir_like">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="I" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S" compartment="c" initialConcentration="99" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="R" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="N" value="100" constant="true"/>
      <parameter id="I0" value="1" constant="true"/>
      <parameter id="R0" value="2" constant="true"/>
      <parameter id="g" value="0.2" constant="true"/>
      <parameter id="beta" value="0" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="I"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>I0</ci></math></initialAssignment>
      <initialAssignment symbol="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>N</ci><ci>I0</ci></apply></math></initialAssignment>
      <initialAssignment symbol="beta"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><divide/><apply><times/><ci>R0</ci><ci>g</ci></apply><ci>N</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="infect" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
          <speciesReference species="I" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts><speciesReference species="I" stoichiometry="2" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>beta</ci><ci>I</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="recover" reversible="false" fast="false">
        <listOfReactants><speciesReference species="I" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="R" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>g</ci><ci>I</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@_needs_output_sens
def test_sbml_seed_map_spans_several_species_and_a_derived_parameter(tmp_path):
    """The seed map for the Bertozzi shape.

    ``I0`` seeds two species with opposite derivatives (+1 on ``I``, -1 on ``S``); ``R0`` and
    ``g`` seed the *parameter* ``beta``, whose derivatives are point-dependent expressions.
    ``N`` seeds all three. Under #511 every one of these was a flat refusal (#530)."""
    xml_path = Path(tmp_path) / 'seir.xml'
    xml_path.write_text(_SEIR_LIKE_SBML)
    xml = str(xml_path)
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=PSet([]), actions=(TimeCourse({'time': '10', 'step': '1'}),))
    _, _, seeds = model.sensitivity_entity_namespace()
    assert seeds['I0'] == (SeedTerm(IC, 'I', ONE), SeedTerm(IC, 'S', ('num', -1.0)))
    assert seeds['N'][0] == SeedTerm(IC, 'S', ONE)
    # beta's inputs carry point-dependent derivatives, checked by value at a probe point.
    env = {'R0': 3.0, 'g': 0.4, 'N': 50.0}
    for name, expected in (('R0', 0.4 / 50.0), ('g', 3.0 / 50.0)):
        (term,) = [s for s in seeds[name] if s.key == 'beta']
        assert term.target == PARAM
        assert derivative.evaluate(term.node, env) == pytest.approx(expected)


@_needs_output_sens
def test_sbml_fd_oracle_multi_species_seed_and_derived_parameter(tmp_path):
    """FD oracle for the Bertozzi shape: three free parameters reaching the model *only*
    through a condition -- one summing two opposite-signed species IC columns, two chaining
    through a derived parameter with a point-dependent factor (#530).

    ``g`` is the sharp case: it is both a rate constant the ODE reads directly *and* an input
    to ``beta``, so its column is the sum of its own parameter axis and ``beta``'s scaled by
    ``R0/N``. Dropping either half leaves a gradient that still looks plausible."""
    xml_path = Path(tmp_path) / 'seir.xml'
    xml_path.write_text(_SEIR_LIKE_SBML)

    def _check(route):
        assert route.is_point_dependent
        assert set(route.sensitivity_ic) == {'I', 'S'}
        assert 'beta' in route.sensitivity_params
        assert len(route.routes['g_free'].contributions) == 2   # beta chain + g's own axis

    _assert_fd_matches(
        tmp_path, str(xml_path), 'seirfd',
        cond=MutationSet([
            Mutation('I0', '=', 'i0_free', is_param_ref=True),
            Mutation('R0', '=', 'r0_free', is_param_ref=True),
            Mutation('g', '=', 'g_free', is_param_ref=True),
        ], 'c'),
        free=[FreeParameter('i0_free', 'uniform_var', 0.1, 50.0, value=3.0),
              FreeParameter('r0_free', 'uniform_var', 0.5, 8.0, value=2.5),
              FreeParameter('g_free', 'uniform_var', 0.01, 2.0, value=0.25)],
        model_params=lambda theta: {'N': 100.0, 'I0': theta['i0_free'],
                                    'R0': theta['r0_free'], 'g': theta['g_free']},
        truth={'N': 100.0, 'I0': 1.0, 'R0': 2.0, 'g': 0.2},
        column='I', sigma=1.0, expect_route=_check)


@_needs_output_sens
def test_sbml_fd_oracle_amount_species_seed(tmp_path):
    """FD oracle for the case above: the assembled gradient of a free parameter routed through
    an amount-species seed in a size-2 compartment must match central differences (#530)."""
    xml_path = Path(tmp_path) / 'amount.xml'
    xml_path.write_text(_AMOUNT_SEED_SBML)
    _assert_fd_matches(
        tmp_path, str(xml_path), 'amountfd',
        cond=MutationSet([Mutation('S0', '=', 's0_free', is_param_ref=True)], 'c'),
        free=[FreeParameter('s0_free', 'uniform_var', 1.0, 1e4, value=120.0)],
        model_params=lambda theta: {'S0': theta['s0_free'], 'k': 0.3},
        truth={'S0': 100.0, 'k': 0.3}, column='S', sigma=2.0)


# --- setup regression guard (default CI, no full fit) -------------------------- #
def _decay_recovery_config(tmp_path, fit_type, *, observables=None, obs_column='S',
                           obs_formula=None, **overrides):
    """Build a real edition-2 ``Configuration`` for a decay-SBML gradient fit.

    Simulates the model at the true params to write a zero-noise ``.exp`` (the oracle),
    then emits a conf through the real parser with ``sbml_backend = bngsim``. ``observables``
    routes the fit through a measurement-model formula column (ADR-0036)."""
    xml = _write_decay_sbml(tmp_path)
    truth = _decay_model(tmp_path)
    data = truth.execute(str(tmp_path), 'truth', 0)['time_course']
    t = np.asarray(data['time'])
    column = obs_formula(data) if obs_formula is not None else np.asarray(data[obs_column])
    sd = max(0.02 * float(np.max(column)), 1e-6)
    exp = Path(tmp_path) / 'tc.exp'
    exp.write_text('\n'.join(
        ['# time\t%s\t%s_SD' % (obs_column, obs_column)]
        + ['%.12g\t%.12g\t%.12g' % (ti, ci, sd) for ti, ci in zip(t, column)]) + '\n')

    free = {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)}
    return H.make_newera_config(
        tmp_path, xml, str(exp), free, 'tc', fit_type,
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=60,
        sbml_backend='bngsim', observables=observables, **overrides)


@_needs_output_sens
def test_sbml_gradient_setup_declares_scored_suffixes(tmp_path):
    """The exact #482 crash site, guarded in default CI: building a real SBML ``TRFAlgorithm``
    and running its gradient-path setup must declare each model's scored suffixes (the call
    ``set_scored_suffixes`` that used to raise ``AttributeError`` on the SBML backend), enable
    forward sensitivities, and build the per-experiment routings -- WITHOUT running a full fit.

    Lives outside the opt-in ``recovery`` tier on purpose: the original bug was invisible to CI
    precisely because the only SBML gradient coverage was recovery-only."""
    conf = _decay_recovery_config(tmp_path, 'trf')
    alg = H.build(conf, 'trf')
    alg._setup_gradient_path()

    assert alg._routings and ('decay', 'tc') in alg._routings
    for model in alg.model_list:
        assert model._scored_suffixes == {'tc'}          # the gate the crash was missing
        assert model._sensitivity_request is not None    # forward sensitivities activated


# --- end-to-end recovery (opt-in recovery tier) -------------------------------- #
@pytest.mark.recovery
@_needs_output_sens
@pytest.mark.parametrize('fit_type', ['trf', 'lbfgs'])
def test_gradient_recovers_sbml_decay_rate_and_initial_condition(tmp_path, monkeypatch, fit_type):
    """``fit_type = trf`` / ``lbfgs`` recover the decay SBML model's rate + initial condition
    from zero-noise data, through PyBNF's real scheduler -- the SBML peer of the BNGL
    ``test_{trf,lbfgs}_recovers_decay_rate_and_initial_condition`` recovery tests. A tight
    parameter assertion (not a smoke bound), so a wrong residual Jacobian would fail it."""
    H.install(monkeypatch)
    conf = _decay_recovery_config(tmp_path, fit_type)
    alg = H.build(conf, fit_type)
    H.drive(alg)

    rec = H.best_params(alg, ['k', 'S'])
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['S'] - TRUE_S0) / TRUE_S0 < 0.02
    assert alg.trajectory.best_score() < 1e-3


@pytest.mark.recovery
@_needs_output_sens
def test_trf_recovers_sbml_through_formula_observable(tmp_path, monkeypatch):
    """``fit_type = trf`` recovers the decay SBML model scoring a measurement-model **formula**
    observable ``dbl = 2 * S`` (ADR-0036) -- the ``prediction_sensitivity`` seam the issue
    flags as a likely next gap. The ``k``/``S`` forward sensitivities must flow through the
    formula's chain rule (factor 2) into the residual Jacobian for the fit to recover, so a
    broken seam would fail the tight assertion. Needs the ``petab`` math extra for the
    ``observableFormula``."""
    pytest.importorskip('petab')
    H.install(monkeypatch)
    conf = _decay_recovery_config(
        tmp_path, 'trf', observables={'dbl': '2 * S'}, obs_column='dbl',
        obs_formula=lambda data: 2.0 * np.asarray(data['S']))
    assert [m.observable_id for m in conf.obj.measurement.models] == ['dbl']

    alg = H.build(conf, 'trf')
    H.drive(alg)

    rec = H.best_params(alg, ['k', 'S'])
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['S'] - TRUE_S0) / TRUE_S0 < 0.02
    assert alg.trajectory.best_score() < 1e-3
