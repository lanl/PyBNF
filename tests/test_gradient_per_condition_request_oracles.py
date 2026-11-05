"""Independent oracles for per-condition sensitivity requests and condition resolution.

Each test compares the assembled gradient with central differences of PyBNF's own scalar loss
(fresh models built and run at each perturbed point, through no sensitivity code). A refusal
is acceptable only where it names a real cause.
"""

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.data import Data
from pybnf.gradient import apply_routings, assemble_gaussian_gradient, route_for_model
from pybnf.objective import ChiSquareObjective
from pybnf.pset import FreeParameter, Mutation, MutationSet, PSet, TimeCourse

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

_OBJ = ChiSquareObjective()


def _pset(values):
    return PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                 for n, v in values.items()])


def _exp(sim, column, sigma, indvar='time', offset=0.0):
    x = sim.data[:, sim.cols[indvar]]
    y = sim.data[:, sim.cols[column]] + offset
    return Data.from_columns(np.column_stack([x, y, np.full(len(y), sigma)]),
                             [indvar, column, column + '_SD'])


def _job_fit(build, scored, free, truth, column, sigma, indvar=None, offsets=None, h=1e-5):
    """(assembled, central differences), with routes resolved and applied exactly as
    ``GradientOptimizer._setup_gradient_path`` does: scored set first, the condition of each
    scored suffix from ``_condition_for_suffix``, the wildtype route always included."""
    from pybnf.algorithms.optimizers.gradient_base import GradientOptimizer
    names = [p.name for p in free]
    indvar = indvar or {}
    offsets = offsets or {}

    def run(values, routed=False):
        model = build(values)
        model.set_scored_suffixes(set(scored))
        routes = None
        if routed:
            routes = {k: route_for_model(
                model, names, GradientOptimizer._condition_for_suffix(None, model, k))
                for k in scored}
            apply_routings(model, [route_for_model(model, names, None), *routes.values()])
        return model.execute(str(_job_fit.tmp), 'per_condition', 60), routes

    truth_ds, _ = run(truth)
    exps = {k: _exp(truth_ds[k], column, sigma, indvar.get(k, 'time'), offsets.get(k, 0.0))
            for k in scored}
    theta = {p.name: p.value for p in free}
    fd = np.zeros(len(free))
    for j, p in enumerate(free):
        up, dn = dict(theta), dict(theta)
        up[p.name] += h
        dn[p.name] -= h
        fd[j] = (sum(_OBJ.evaluate(run(up)[0][k], exps[k]) for k in scored)
                 - sum(_OBJ.evaluate(run(dn)[0][k], exps[k]) for k in scored)) / (2.0 * h)
    ds, routes = run(theta, routed=True)
    grad = assemble_gaussian_gradient(
        _OBJ, [(ds[k], exps[k], routes[k].at_point(theta)) for k in scored], free).gradient
    return grad, fd, ds, routes


@pytest.fixture(autouse=True)
def _tmp(tmp_path):
    _job_fit.tmp = tmp_path


# A <-> B, B also degraded at kb; A seeded by S0 and B by the derived Btot = 2*S0.
_NET = """begin parameters
    1 S0    10    # Constant
    2 kf    1     # Constant
    3 kr    0.5   # Constant
    4 kb    0.2   # Constant
    5 Btot  2*S0  # ConstantExpression
end parameters
begin species
    1 A() S0
    2 B() Btot
end species
begin reactions
    1 1 2 kf #_R1
    2 2 1 kr #_R2
    3 2 0 kb #_R3
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
"""
_SIM = 'simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>4,suffix=>"%s"})'
_CS = MutationSet([Mutation('S0', '=', 's', is_param_ref=True)], 'cs')    # reaches a seed only
_CK = MutationSet([Mutation('kr', '=', 'x', is_param_ref=True)], 'ck')    # reaches a rate only


def _net_builder(tmp_path, actions, suffixes, mutants):
    def build(values):
        import pybnf.bngsim_model as bngsim_model
        net = tmp_path / 'per_condition.net'
        net.write_text(_NET)
        model = bngsim_model.BngsimModel('per_condition', list(actions), list(suffixes), [],
                                         nf=str(net))
        for mut in mutants:
            model.add_mutant(mut)
        model.param_set = _pset(values)
        return model
    return build


@pytest.mark.bngsim
@_needs_output_sens
@pytest.mark.parametrize('protocol', ['time course', 'equilibrate, setParameter, measure'])
def test_net_each_condition_requests_its_own_columns_and_the_gradient_holds(tmp_path,
                                                                           protocol):
    """Each ``.net`` condition now asks bngsim only for the parameter columns its own route
    reads. ``cs`` reaches the model only through a seed (``S0``, and ``B()`` through the derived
    ``Btot``), ``ck`` only through a rate, and ``kb`` is fitted by id everywhere -- including
    the wildtype, scored beside them. Every experiment's tensor must hold every column its
    route reads, and the gradient must match central differences of the summed loss. The
    carried variant equilibrates under each condition and measures after a ``setParameter``."""
    if protocol == 'time course':
        actions = [_SIM % (4, 'tc')]
        suffixes = [('simulate', 'tc')]
        scored = ['tc', 'tccs', 'tcck']
    else:
        actions = [_SIM % (5, 'eq'), 'setParameter("kf",3)', _SIM % (3, 'tc')]
        suffixes = [('simulate', 'eq'), ('simulate', 'tc')]
        scored = ['tccs', 'tcck']
    free = [FreeParameter('s', 'uniform_var', 1.0, 100.0, value=12.0),
            FreeParameter('x', 'uniform_var', 0.01, 10.0, value=0.6),
            FreeParameter('kb', 'uniform_var', 0.01, 10.0, value=0.25)]
    grad, fd, ds, routes = _job_fit(
        _net_builder(tmp_path, actions, suffixes, [_CS, _CK]), scored, free,
        {'s': 10.0, 'x': 0.5, 'kb': 0.2}, 'Bobs', 0.1)
    for key in scored:
        assert set(routes[key].sensitivity_params) <= set(
            ds[key].output_sensitivities.param_names), key
    # The protocol's saves and resets leave central differences noisy at about 1e-4.
    np.testing.assert_allclose(grad, fd, rtol=3e-4)


_CF = MutationSet([Mutation('kf', '=', 'y', is_param_ref=True)], 'cf')     # binds kf
_C0 = MutationSet([Mutation('kb', '=', 0.4)], 'c0')                        # binds nothing fitted


@pytest.mark.bngsim
@_needs_output_sens
def test_net_carried_scan_where_the_route_reads_no_column(tmp_path):
    """Condition ``cf`` binds ``kf = y`` for a time course; condition ``c0`` runs a time course
    followed by a carried scan of ``kf``. In ``c0`` nothing routes ``y`` anywhere, so its route
    reads no column and the scan of ``kf`` there is not a scan of a differentiated parameter.
    The gradient must match central differences of the summed loss."""
    actions = [_SIM % (2, 'tc'),
               'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],t_start=>0,t_end=>3,'
               'method=>"ode",suffix=>"dr"})']
    suffixes = [('simulate', 'tc'), ('parameter_scan', 'dr')]
    free = [FreeParameter('y', 'uniform_var', 0.01, 10.0, value=1.2)]
    grad, fd, _, _ = _job_fit(_net_builder(tmp_path, actions, suffixes, [_CF, _C0]),
                              ['tccf', 'drc0'], free, {'y': 1.0}, 'Bobs', 0.1,
                              indvar={'drc0': 'kf'})
    np.testing.assert_allclose(grad, fd, rtol=3e-4)


_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="suffix_clash">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.3" constant="true"/>
      <parameter id="kb" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="syn" reversible="false" fast="false">
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>kb</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="conv" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k1</ci><ci>A</ci></apply></math></kineticLaw>
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
def test_sbml_wildtype_experiment_whose_name_ends_with_a_condition_name(tmp_path):
    """A wildtype experiment named ``data`` beside a condition named ``a`` (``k1 = x``) on
    experiment ``e1``. The wildtype does not depend on ``x``, so only ``e1a`` contributes to its
    gradient. Resolving a scored suffix to a condition by "ends with the condition's suffix"
    reads ``data`` as ``dat`` under ``a`` and routes ``x`` onto the wildtype's ``k1`` column.
    The gradient must match central differences of the summed loss whatever the names."""
    xml = tmp_path / 'suffix_clash.xml'
    xml.write_text(_SBML)
    xml = str(xml)
    cond = MutationSet([Mutation('k1', '=', 'x', is_param_ref=True)], 'a')
    actions = (TimeCourse({'time': '4', 'step': '1', 'suffix': 'data'}),
               TimeCourse({'time': '4', 'step': '1', 'suffix': 'e1'}))

    def build(values):
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(xml, xml, pset=_pset(values),
                                                            actions=actions)
        model.add_mutant(cond)
        return model

    free = [FreeParameter('x', 'uniform_var', 0.01, 3.0, value=0.35)]
    grad, fd, _, _ = _job_fit(build, ['data', 'e1a'], free, {'x': 0.3}, 'B', 0.05,
                              offsets={'data': 0.1}, h=1e-6)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)
