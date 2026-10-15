"""Independent oracles for which experiment a run-time check, or a load-time refusal, is about.

The sensitivity request is the union over every experiment of a model, so a column one
experiment reads is also computed in another that never reads it -- where a condition can
legitimately pin what the column follows. Each test here is a valid configuration of that
kind, checked against central differences of PyBNF's own scalar outputs (fresh models built and
run at each perturbed point, through no sensitivity code).
"""

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.data import Data
from pybnf.gradient import apply_routings, assemble_gaussian_gradient, route_for_model
from pybnf.objective import ChiSquareObjective
from pybnf.pset import FreeParameter, Mutation, MutationSet, ParamScan, PSet, TimeCourse

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

_OBJ = ChiSquareObjective()

# k2 := 2*k1 drives A -> B; k1 also degrades B.
_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="derived_pin">
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

_NET = """begin parameters
    1 k1    0.3   # Constant
    2 k2    2*k1  # ConstantExpression
    3 kb    1     # Constant
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 0 1 kb #_R1
    2 1 2 k2 #_R2
    3 2 0 k1 #_R3
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
"""
_SIM = 'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"tc"})'

# c2 estimates k1 (a parameter reference); c3 pins the derived k2.
_C2 = MutationSet([Mutation('k1', '=', 'x', is_param_ref=True)], 'c2')
_C3 = MutationSet([Mutation('k2', '=', 0.8)], 'c3')


def _pset(values):
    return PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                 for n, v in values.items()])


def _sbml_model(tmp_path, values, actions):
    xml = tmp_path / 'derived_pin.xml'
    xml.write_text(_SBML)
    return bngsim_sbml_model.BngsimSbmlModelNoTimeout(str(xml), str(xml), pset=_pset(values),
                                                       actions=tuple(actions))


def _net_model(tmp_path, values):
    import pybnf.bngsim_model as bngsim_model
    net = tmp_path / 'derived_pin.net'
    net.write_text(_NET)
    model = bngsim_model.BngsimModel('derived_pin', [_SIM], [('simulate', 'tc')], [],
                                     nf=str(net))
    model.param_set = _pset(values)
    return model


@pytest.mark.bngsim
@pytest.mark.bngsim_sbml
@_needs_output_sens
@pytest.mark.parametrize('backend', ['sbml', 'net'])
def test_multiple_shooting_compares_each_segment_with_its_own_conditions_route(tmp_path,
                                                                              backend):
    """Multiple shooting first applies the routings, as every gradient job does, and then
    widens the request to every carried state (``_request_state_sensitivities``). Its segments
    must be checked against the running condition's columns just as an ordinary fit's runs
    are: the ``c3`` segment, where ``x`` reaches nothing, runs; the ``c2`` segment's ``x``
    column (``k1``'s) matches central differences of ``c2``'s scalar trajectory."""
    from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm
    from pybnf.shooting import BngsimSegmentBackend, NetSegmentBackend
    x, h = 0.35, 1e-6
    times = np.linspace(0.0, 2.0, 3)
    action = TimeCourse({'time': '4', 'step': '1'})

    def build(value):
        if backend == 'sbml':
            model = _sbml_model(tmp_path, {'x': value}, [action])
        else:
            model = _net_model(tmp_path, {'x': value})
        model.add_mutant(_C2)
        model.add_mutant(_C3)
        return model

    def scalar_b(value):
        key = 'time_coursec2' if backend == 'sbml' else 'tcc2'
        data = build(value).execute(str(tmp_path), 'ms_scope', 60)[key]
        column = 'B' if backend == 'sbml' else 'Bobs'
        return data.data[:3, data.cols[column]]

    model = build(x)
    apply_routings(model, [route_for_model(model, ['x'], None),
                           route_for_model(model, ['x'], _C2),
                           route_for_model(model, ['x'], _C3)])
    MultipleShootingAlgorithm._request_state_sensitivities(None, model)

    def segment(condition):
        if backend == 'sbml':
            seg = BngsimSegmentBackend(model, action, condition, 'time_course' + condition.suffix)
        else:
            from pybnf.bngsim_model.parsing import _parse_simulate_action
            seg = NetSegmentBackend(model, _parse_simulate_action(_SIM), condition,
                                    'tc' + condition.suffix)
        return seg.simulate(model.param_set, times, None)

    segment(_C3)                                   # a valid run: must not be refused
    data = segment(_C2)
    sens = data.output_sensitivities
    selector = 'species:B' if backend == 'sbml' else 'species:B()'
    column = sens.d_param[:, sens.selectors.index(selector), sens.param_names.index('k1')]
    fd = (scalar_b(x + h) - scalar_b(x - h)) / (2.0 * h)
    np.testing.assert_allclose(column, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.bngsim_sbml
@_needs_output_sens
@pytest.mark.parametrize('how', ['a scan of k2', 'an equilibration condition on k2'])
def test_sbml_write_honoured_in_its_own_experiment_beside_a_condition_moving_its_input(
        tmp_path, how):
    """The wildtype experiment writes the initialAssignment target ``k2`` -- a dose-response
    scan of it, or a pre-equilibration that sets it -- and nothing in that experiment moves
    ``k1``, so the write stands. Condition ``c`` estimates ``k1 = x`` in a time course of its
    own. The gradient over both experiments must match central differences of the summed
    loss; refusing it because ``c`` moves ``k1`` somewhere else refuses a valid configuration."""
    tc = TimeCourse({'time': '4', 'step': '1', 'suffix': 'tc'})
    cond = MutationSet([Mutation('k1', '=', 'x', is_param_ref=True)], 'c')
    if how == 'a scan of k2':
        other = ParamScan({'param': 'k2', 'time': '2', 'suffix': 'dr'},
                          explicit_points=[0.4, 0.8, 1.6])
        scored = {'dr': ('k2', None), 'tcc': ('time', cond)}
    else:
        other = TimeCourse({'time': '4', 'step': '1', 'suffix': 'pe'})
        other.set_preequilibration([('setParameter', 'k2', 1.5)],
                                   [('setParameter', 'kb', 3.0)])
        scored = {'pe': ('time', None), 'tcc': ('time', cond)}

    def build(value):
        model = _sbml_model(tmp_path, {'x': value}, [tc, other])
        model.add_mutant(cond)
        model.set_scored_suffixes(set(scored))
        return model

    def run(value, routes=None):
        model = build(value)
        if routes is not None:
            apply_routings(model, list(routes.values()))
        return model.execute(str(tmp_path), 'honoured', 0)

    truth = run(0.3)
    exps = {}
    for key, (indvar, _) in scored.items():
        sim = truth[key]
        x = sim.data[:, sim.cols[indvar]]
        y = sim.data[:, sim.cols['B']]
        exps[key] = Data.from_columns(np.column_stack([x, y, np.full(len(y), 0.05)]),
                                      [indvar, 'B', 'B_SD'])
    free = [FreeParameter('x', 'uniform_var', 0.01, 3.0, value=0.35)]
    h = 1e-5
    fd = (sum(_OBJ.evaluate(run(0.35 + h)[k], exps[k]) for k in scored)
          - sum(_OBJ.evaluate(run(0.35 - h)[k], exps[k]) for k in scored)) / (2.0 * h)
    base = build(0.35)
    routes = {key: route_for_model(base, ['x'], c) for key, (_, c) in scored.items()}
    ds = run(0.35, routes)
    grad = assemble_gaussian_gradient(
        _OBJ, [(ds[k], exps[k], routes[k].at_point({'x': 0.35})) for k in scored],
        free).gradient
    np.testing.assert_allclose(grad, [fd], rtol=1e-5)
