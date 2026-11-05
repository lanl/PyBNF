"""Independent oracles for the routing of derived parameters and initial-condition seeds.

Each test compares the assembled gradient with central differences of PyBNF's own scalar loss
(a fresh model built and run at each perturbed point, through no sensitivity code), a closed
form, or a published independent value. Where a bngsim build cannot give the right column, the
only acceptable alternative is a refusal that names the cause.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.data import Data
from pybnf.gradient import (
    GradientNotSupported, apply_routings, assemble_gaussian_gradient, route_for_model,
)
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, Mutation, MutationSet, PSet, TimeCourse

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')


def _free(name, value, lo=1e-6, hi=1e6):
    return FreeParameter(name, 'uniform_var', lo, hi, value=value)


def _central(loss, free, h=1e-6):
    theta = {p.name: p.value for p in free}
    out = np.zeros(len(free))
    for j, p in enumerate(free):
        up, dn = dict(theta), dict(theta)
        up[p.name] += h
        dn[p.name] -= h
        out[j] = (loss(up) - loss(dn)) / (2.0 * h)
    return out


def _exp_from(sim, columns, sigma, indvar='time'):
    x = sim.data[:, sim.cols[indvar]]
    blocks, headers = [x], [indvar]
    for name in columns:
        y = sim.data[:, sim.cols[name]]
        blocks += [y, np.full(len(y), sigma)]
        headers += [name, name + '_SD']
    return Data.from_columns(np.column_stack(blocks), headers)


# ================================================================================= SBML ===

def _sbml_run(xml, values, actions, key, route=None):
    ps = PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v) for n, v in values.items()])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(xml, xml, pset=ps, actions=tuple(actions))
    if route is not None:
        model.enable_output_sensitivities(params=route.sensitivity_params,
                                          ic=route.sensitivity_ic)
    return model.execute(str(Path(xml).parent), 'review', 0)[key]


def _sbml_gradient(xml, free, actions, key, exp):
    theta = {p.name: p.value for p in free}
    base = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=PSet([_free(n, v, -1e12, 1e12) for n, v in theta.items()]),
        actions=tuple(actions))
    route = route_for_model(base, [p.name for p in free]).at_point(theta)
    sim = _sbml_run(xml, theta, actions, key, route=route)
    return assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, route)], free).gradient, sim


# lanl/PyBNF#864's reproduction, verbatim: beta := R0*g/N, and both rate laws carry the
# compartment symbol c (size 1), the usual SBML/COPASI form.
_SIR_C_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sir">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="99" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="I" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="N" value="100" constant="true"/>
      <parameter id="R0" value="2" constant="true"/>
      <parameter id="g" value="0.2" constant="true"/>
      <parameter id="beta" value="0" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="beta"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><divide/><apply><times/><ci>R0</ci><ci>g</ci></apply><ci>N</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="infect" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/><speciesReference species="I" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="I" stoichiometry="2" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>beta</ci><ci>S</ci><ci>I</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="recover" reversible="false" fast="false">
        <listOfReactants><speciesReference species="I" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>g</ci><ci>I</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_SIR_DATA = np.array([
    [0, -0.20], [4, 0.17], [8, 4.02], [12, 8.69], [16, 14.18], [20, 15.64],
    [24, 14.60], [28, 11.76], [32, 10.78], [36, 9.15], [40, 4.86]])


@pytest.mark.bngsim_sbml
@_needs_output_sens
@pytest.mark.parametrize('g, published', [(0.12, -1024.0646), (0.1978481, 5.3186)])
def test_sbml_fd_oracle_issue_864_compartment_factor_law(tmp_path, g, published):
    """lanl/PyBNF#864 at the issue's own points. ``beta = R0*g/N`` is held live by bngsim, so
    ``g``'s own column already carries the chain through ``beta``; routing ``beta``'s column on
    top gave dF/dg = -80.17 at g = 0.1978481 (the sign wrong) and -538.99 at g = 0.12. The
    oracle is central differences of PyBNF's chi_sq loss, and the independent scipy values the
    issue published (solve_ivp at rtol = atol = 1e-11)."""
    xml = tmp_path / 'sir_c.xml'
    xml.write_text(_SIR_C_SBML)
    xml = str(xml)
    actions = (TimeCourse({'time': '40', 'step': '4'}),)
    exp = Data.from_columns(
        np.column_stack([_SIR_DATA[:, 0], _SIR_DATA[:, 1], np.full(len(_SIR_DATA), 1.5)]),
        ['time', 'I', 'I_SD'])
    free = [_free('g', g, 0.05, 0.6)]
    obj = ChiSquareObjective()
    fd = _central(lambda v: obj.evaluate(_sbml_run(xml, v, actions, 'time_course'), exp), free)
    grad, _ = _sbml_gradient(xml, free, actions, 'time_course', exp)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(grad, [published], rtol=1e-3, atol=1e-3)


_ALIAS_C_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="alias_c">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_src" value="0.3" constant="true"/>
      <parameter id="k_used" value="0" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k_used"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k_src</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k_used</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_issue_864_pure_alias_under_a_compartment_factor_is_not_doubled(tmp_path):
    """The pure-alias shape of lanl/PyBNF#864: ``k_used := k_src``, law ``c*k_used*S``. The
    source reaches the model only through the alias, so the gradient came back exactly 2x.
    Oracles: central differences of the scalar loss, and the closed form of the column the
    route reads, dS/dk_src = -t * S0 * exp(-k t)."""
    xml = tmp_path / 'alias_c.xml'
    xml.write_text(_ALIAS_C_SBML)
    xml = str(xml)
    actions = (TimeCourse({'time': '10', 'step': '1'}),)
    exp = _exp_from(_sbml_run(xml, {'k_src': 0.3}, actions, 'time_course'), ['S'], 2.0)
    free = [_free('k_src', 0.45, 0.01, 3.0)]
    obj = ChiSquareObjective()
    fd = _central(lambda v: obj.evaluate(_sbml_run(xml, v, actions, 'time_course'), exp), free)
    grad, sim = _sbml_gradient(xml, free, actions, 'time_course', exp)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)
    sens = sim.output_sensitivities
    t = sim.data[:, sim.cols['time']]
    assert sens.param_names == ['k_src']          # the only column the route reads
    col = sens.d_param[:, sens.selectors.index('species:S'), 0]
    np.testing.assert_allclose(col, -t * 100.0 * np.exp(-0.45 * t), rtol=1e-6, atol=1e-9)


# k2 := 2*k1 drives A -> B; k1 also degrades B. A pre-equilibration's measurement condition
# then pins k2, so after the intervention k2 no longer depends on k1.
_PREEQ_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="preeq_derived">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.3" constant="true"/>
      <parameter id="k2" value="0" constant="true"/>
      <parameter id="kb" value="1" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k2"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>k1</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="syn" reversible="false" fast="false">
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>kb</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="conv" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k2</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k1</ci><ci>B</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_preequilibration_measurement_pins_a_live_derived_parameter(tmp_path):
    """The router drops ``k2``'s column because bngsim holds ``k2`` live, so ``k1``'s own
    column must carry the chain until the intervention and stop carrying it after. The run-time
    check runs when the Simulator is built, before the intervention, so it cannot see the pin.
    Either the gradient matches central differences of the scalar loss or the fit is refused
    with ``k2`` named."""
    xml = tmp_path / 'preeq.xml'
    xml.write_text(_PREEQ_SBML)
    xml = str(xml)
    act = TimeCourse({'time': '4', 'step': '1', 'suffix': 'tc'})
    act.set_preequilibration([], [('setParameter', 'k2', 1.5)])
    actions = (act,)
    exp = _exp_from(_sbml_run(xml, {'k1': 0.3}, actions, 'tc'), ['A', 'B'], 0.05)
    free = [_free('k1', 0.35, 0.01, 3.0)]
    obj = ChiSquareObjective()
    fd = _central(lambda v: obj.evaluate(_sbml_run(xml, v, actions, 'tc'), exp), free)
    try:
        grad, _ = _sbml_gradient(xml, free, actions, 'tc', exp)
    except (PybnfError, GradientNotSupported) as exc:
        assert 'k2' in str(exc)
        return
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


_SEEDED_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="seeded_two">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="X" compartment="c" initialConcentration="5" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="100" constant="true"/>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>S0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="r2" reversible="false" fast="false">
        <listOfReactants><speciesReference species="X" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_fd_oracle_one_condition_estimates_a_seed_another_pins_the_species(tmp_path):
    """Condition ``c2`` estimates ``S``'s seed (``S0 = x``, a parameter reference) and
    condition ``c3`` pins ``S = 5``. The sensitivity request is the union over experiments, so
    ``S0``'s column is requested in ``c3`` too, where ``S(0)`` legitimately does not depend on
    it. Oracle: central differences of the summed chi_sq loss over both experiments."""
    xml = tmp_path / 'seeded.xml'
    xml.write_text(_SEEDED_SBML)
    xml = str(xml)
    action = TimeCourse({'time': '10', 'step': '1'})
    c2 = MutationSet([Mutation('S0', '=', 'x', is_param_ref=True)], 'c2')
    c3 = MutationSet([Mutation('S', '=', 5.0)], 'c3')
    keys = ('time_coursec2', 'time_coursec3')
    free = [_free('k', 0.35, 0.01, 10.0), _free('x', 110.0, 1.0, 1000.0)]
    names = [p.name for p in free]

    def build(values):
        ps = PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                   for n, v in values.items()])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(xml, xml, pset=ps, actions=(action,))
        model.add_mutant(c2)
        model.add_mutant(c3)
        return model

    def run(values, routes=None):
        model = build(values)
        if routes is not None:
            apply_routings(model, list(routes.values()))
            model.set_scored_suffixes(set(keys))
        return model.execute(str(tmp_path), 'two_conditions', 0)

    truth = run({'k': 0.3, 'x': 100.0})
    exps = {key: _exp_from(truth[key], ['S'], 2.0) for key in keys}
    obj = ChiSquareObjective()
    fd = _central(lambda v: sum(obj.evaluate(run(v)[key], exps[key]) for key in keys), free)
    theta = {p.name: p.value for p in free}
    base = build(theta)
    routes = {'time_coursec2': route_for_model(base, names, c2),
              'time_coursec3': route_for_model(base, names, c3)}
    ds = run(theta, routes)
    grad = assemble_gaussian_gradient(
        obj, [(ds[key], exps[key], routes[key].at_point(theta)) for key in keys], free).gradient
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-4)


# ================================================================================= .net ===

def _net_model(net, actions, suffixes, values, mutants=()):
    import pybnf.bngsim_model as bngsim_model
    model = bngsim_model.BngsimModel(Path(net).stem, list(actions), list(suffixes), [],
                                     nf=str(net))
    for mut in mutants:
        model.add_mutant(mut)
    model.param_set = PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                            for n, v in values.items()])
    return model


def _net_gradient_and_fd(tmp_path, net, actions, suffixes, key, free, truth, columns, sigma,
                         mutants=(), condition=None, indvar='time'):
    names = [p.name for p in free]
    obj = ChiSquareObjective()

    def run(values, routed=False):
        model = _net_model(net, actions, suffixes, values, mutants)
        route = None
        if routed:
            route = route_for_model(model, names, condition)
            apply_routings(model, [route])
            model.set_scored_suffixes({key})
        return model.execute(str(tmp_path), 'review_net', 60)[key], route

    exp = _exp_from(run(truth)[0], columns, sigma, indvar)
    fd = _central(lambda v: obj.evaluate(run(v)[0], exp), free)
    sim, route = run({p.name: p.value for p in free}, routed=True)
    grad = assemble_gaussian_gradient(
        obj, [(sim, exp, route.at_point({p.name: p.value for p in free}))], free).gradient
    return grad, fd, sim


_EXCHANGE_NET = """begin parameters
    1 A0    10    # Constant
    2 kf    1     # Constant
    3 kr    0.5   # Constant
    4 Atot  10    # Constant
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf #_R1
    2 2 1 kr #_R2
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
"""


@pytest.mark.bngsim
@_needs_output_sens
@pytest.mark.parametrize('witness', ['this build', 'forced to fail'])
def test_net_newton_scan_conserved_species_seeded_by_an_intervention(tmp_path, monkeypatch,
                                                                     witness):
    """A Newton steady-state scan of ``A <-> B`` whose ``A`` comes from
    ``setConcentration("A()", "Atot")``: the steady state moves with ``Atot`` only through the
    conserved total, ``dB*/dAtot = kf/(kf + kr)``. On a build whose Newton solve drops that
    seeding (lanl/bngsim#704) the column comes back zero, so the scan must be refused there --
    the .net initializer is not the only way a fitted parameter seeds a conserved species.
    ``forced to fail`` runs the refusal half on any build."""
    import pybnf.bngsim_model._runtime as runtime
    if witness == 'forced to fail':
        monkeypatch.setattr(runtime, 'steady_state_sensitivity_carries_ic_seeding',
                            lambda: False, raising=False)
    carries = getattr(runtime, 'steady_state_sensitivity_carries_ic_seeding', lambda: True)()
    net = tmp_path / 'exchange.net'
    net.write_text(_EXCHANGE_NET)
    actions = ['setConcentration("A()","Atot")',
               'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],steady_state=>1,'
               'ss_method=>"newton",t_end=>200,suffix=>"dr"})']
    free = [_free('Atot', 12.0), _free('kr', 0.6)]
    run = lambda: _net_gradient_and_fd(  # noqa: E731
        tmp_path, net, actions, [('parameter_scan', 'dr')], 'dr', free,
        {'Atot': 10.0, 'kr': 0.5}, ['Bobs'], 0.1, indvar='kf')
    if not carries:
        with pytest.raises(PybnfError, match=r'lanl/bngsim#704'):
            run()
        return
    grad, fd, sim = run()
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-3)
    sens = sim.output_sensitivities
    kf = sim.data[:, sim.cols['kf']]
    col = sens.d_param[:, sens.selectors.index('observable:Bobs'), sens.param_names.index('Atot')]
    np.testing.assert_allclose(col, kf / (kf + 0.6), rtol=1e-4)


_DERIVED_RATE_NET = """begin parameters
    1 k1    0.3   # Constant
    2 k2    2*k1  # ConstantExpression
    3 kb    1     # Constant
    4 A0    5     # Constant
    5 Btot  2*A0  # ConstantExpression
end parameters
begin species
    1 A() 10
    2 B() 0
    3 C() Btot
end species
begin reactions
    1 0 1 kb #_R1
    2 1 2 k2 #_R2
    3 2 0 k1 #_R3
    4 3 0 k1 #_R4
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
    3 Cobs                 3
end groups
"""

_SIM = 'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"tc"})'


@pytest.mark.bngsim
@_needs_output_sens
def test_net_fd_oracle_fitted_derived_parameter_read_by_a_rate_law(tmp_path):
    """Fitting ``k2`` (a ConstantExpression ``2*k1``) overrides it, so ``k1`` no longer reaches
    the conversion rate. ``k1``'s column must then hold only its own rate law; a build that
    keeps differentiating ``k1`` through the overridden ``k2`` returned it about 20x too large.
    Either the gradient matches central differences or the fit is refused naming ``k2``."""
    net = tmp_path / 'derived_rate.net'
    net.write_text(_DERIVED_RATE_NET)
    try:
        grad, fd, _ = _net_gradient_and_fd(
            tmp_path, net, [_SIM], [('simulate', 'tc')], 'tc',
            [_free('k1', 0.35), _free('k2', 0.8)], {'k1': 0.3, 'k2': 0.6},
            ['Aobs', 'Bobs'], 0.2)
    except (PybnfError, GradientNotSupported) as exc:
        assert 'k2' in str(exc)
        return
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_fd_oracle_relative_condition_on_a_derived_parameter(tmp_path):
    """A condition ``Btot * 2`` on the ConstantExpression ``Btot = 2*A0`` writes ``4*A0`` as a
    constant at every evaluation, so the conditioned experiment's ``C(0)`` still moves with
    ``A0`` -- but the write overrides ``Btot``, so no column carries that. Either the gradient
    matches central differences of the scalar loss or the condition is refused naming
    ``Btot``."""
    net = tmp_path / 'derived_rate.net'
    net.write_text(_DERIVED_RATE_NET)
    cond = MutationSet([Mutation('Btot', '*', 2.0)], 'dbl')
    try:
        grad, fd, _ = _net_gradient_and_fd(
            tmp_path, net, [_SIM], [('simulate', 'tc')], 'tcdbl',
            [_free('A0', 6.0), _free('k1', 0.35)], {'A0': 5.0, 'k1': 0.3},
            ['Cobs'], 0.2, mutants=[cond], condition=cond)
    except (PybnfError, GradientNotSupported) as exc:
        assert 'Btot' in str(exc)
        return
    np.testing.assert_allclose(grad, fd, rtol=1e-4)
