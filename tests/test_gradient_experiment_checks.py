"""The gradient path compares each run with its own experiment's route, and refuses at load what
no run can differentiate.

Each FD oracle here is central differences of PyBNF's own scalar loss -- a fresh model built
and run at each perturbed point, through no sensitivity code -- against the gradient assembled
from the routed columns, or a closed form. Each refusal test checks that the configuration is
rejected with an error that names the cause.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pybnf._bngsim_caps as caps
import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf import _bngsim_seeding as seeding
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.bngsim_model import expressions
from pybnf.data import Data
from pybnf.gradient import (
    GradientNotSupported, apply_routings, assemble_gaussian_gradient, route_for_model,
)
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, Mutation, MutationSet, ParamScan, PSet, TimeCourse

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

_OBJ = ChiSquareObjective()


def _free(name, value):
    return FreeParameter(name, 'uniform_var', 1e-6, 1e6, value=value)


def _exp_from(sim, columns, sigma, indvar='time'):
    x = sim.data[:, sim.cols[indvar]]
    blocks, headers = [x], [indvar]
    for name in columns:
        y = sim.data[:, sim.cols[name]]
        blocks += [y, np.full(len(y), sigma)]
        headers += [name, name + '_SD']
    return Data.from_columns(np.column_stack(blocks), headers)


def _gradient_and_fd(tmp_path, build, keys, free, truth, columns, sigma, indvar='time',
                     conditions=None, h=1e-6):
    """(assembled gradient, central differences) of the loss summed over ``keys``."""
    conditions = conditions or {}
    names = [p.name for p in free]

    def run(values, routes=None):
        model = build(values)
        if routes is not None:
            apply_routings(model, list(routes.values()))
            model.set_scored_suffixes(set(keys))
        return model.execute(str(tmp_path), 'experiment_checks', 60)

    truth_ds = run(truth)
    exps = {key: _exp_from(truth_ds[key], columns, sigma, indvar) for key in keys}
    theta = {p.name: p.value for p in free}

    def loss(values):
        ds = run(values)
        return sum(_OBJ.evaluate(ds[key], exps[key]) for key in keys)

    fd = np.zeros(len(free))
    for j, p in enumerate(free):
        up, dn = dict(theta), dict(theta)
        up[p.name] += h
        dn[p.name] -= h
        fd[j] = (loss(up) - loss(dn)) / (2.0 * h)
    base = build(theta)
    routes = {key: route_for_model(base, names, conditions.get(key)) for key in keys}
    ds = run(theta, routes)
    grad = assemble_gaussian_gradient(
        _OBJ, [(ds[key], exps[key], routes[key].at_point(theta)) for key in keys],
        free).gradient
    return grad, fd, ds


def _net_builder(net, actions, suffixes, mutants=()):
    def build(values):
        import pybnf.bngsim_model as bngsim_model
        model = bngsim_model.BngsimModel(Path(net).stem, list(actions), list(suffixes), [],
                                         nf=str(net))
        for mut in mutants:
            model.add_mutant(mut)
        model.param_set = PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                                for n, v in values.items()])
        return model
    return build


def _sbml_model(tmp_path, text, values, actions=(TimeCourse({'time': '10', 'step': '1'}),),
                mutants=()):
    xml = Path(tmp_path) / 'model.xml'
    xml.write_text(text)
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml),
        pset=PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                   for n, v in values.items()]),
        actions=tuple(actions))
    for mut in mutants:
        model.add_mutant(mut)
    return model


# ======================================================= one rule: the experiment's route ===

def test_experiment_columns_reads_the_running_conditions_route():
    """The request is the union over the experiments; a run is compared only on the columns its
    own condition's route reads. Without a record, or for a condition it does not name, the
    whole request is compared -- the side that refuses more."""
    request = SimpleNamespace(params=['k1', 'k2', 'kb'], ic=['A', 'B'])
    columns = {'': (frozenset({'kb'}), frozenset()),
               'c2': (frozenset({'k1'}), frozenset({'B'})),
               'c3': (frozenset(), frozenset())}
    assert seeding.experiment_columns(request, columns, 'c2') == (['k1'], ['B'])
    assert seeding.experiment_columns(request, columns, 'c3') == ([], [])
    assert seeding.experiment_columns(request, columns, '') == (['kb'], [])
    assert seeding.experiment_columns(request, columns, 'c9') == (['k1', 'k2', 'kb'], ['A', 'B'])
    assert seeding.experiment_columns(request, None, 'c2') == (['k1', 'k2', 'kb'], ['A', 'B'])


def test_apply_routings_records_the_columns_each_condition_reads():
    """``apply_routings`` hands the model the union request and, beside it, each condition's
    own columns, keyed by condition suffix (``''`` for the wildtype)."""
    class _Model:
        def enable_output_sensitivities(self, *, params=None, ic=None):
            self.request = (list(params), list(ic))

        def set_experiment_columns(self, columns):
            self.columns = columns

    c2 = MutationSet([Mutation('k1', '=', 'x', is_param_ref=True)], 'c2')
    c3 = MutationSet([Mutation('k2', '=', 0.8)], 'c3')
    routings = [SimpleNamespace(condition=None, sensitivity_params=['kb'], sensitivity_ic=[]),
                SimpleNamespace(condition=c2, sensitivity_params=['k1'], sensitivity_ic=['B']),
                SimpleNamespace(condition=c3, sensitivity_params=[], sensitivity_ic=[])]
    model = _Model()
    apply_routings(model, routings)
    assert model.request == (['kb', 'k1'], ['B'])
    assert model.columns == {'': (frozenset({'kb'}), frozenset()),
                             'c2': (frozenset({'k1'}), frozenset({'B'})),
                             'c3': (frozenset(), frozenset())}


# ===================================================== intervention rows at run-time values ===

def test_intervention_seed_row_reads_a_constant_at_its_run_time_value():
    """``Atot*dose`` written after ``setParameter("dose",3)`` (or at a scan point ``dose = 3``)
    seeds ``dA(0)/dAtot = 3``, not the ``.net`` file's ``dose = 1``. Only a derived parameter
    that still tracks its expression is inlined; an overridden one is a number."""
    definitions = {'Atot': '10', 'dose': '1', 'V': '2*dose'}
    row = expressions._intervention_seed_row
    live = {'Atot': 10.0, 'dose': 3.0, 'V': 6.0}
    assert row('Atot*dose', ['Atot'], definitions, live) == [3.0]
    # V = 2*dose live: d(Atot*V)/dAtot = V = 2*dose, and d/d dose = 2*Atot.
    assert row('Atot*V', ['Atot', 'dose'], definitions, live, derived={'V'}) == [6.0, 20.0]
    # V overridden to 7: a number, which reads nothing, so dose does not reach the amount.
    pinned = {'Atot': 10.0, 'dose': 3.0, 'V': 7.0}
    assert row('Atot*V', ['Atot', 'dose'], definitions, pinned, derived=set()) == [7.0, 0.0]


_DOSE_NET = """begin parameters
    1 A0    10    # Constant
    2 kf    1     # Constant
    3 kr    0.5   # Constant
    4 Atot  10    # Constant
    5 dose  1     # Constant
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
_SIM = 'simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>4,suffix=>"%s"})'


@pytest.mark.bngsim
@_needs_output_sens
@pytest.mark.parametrize('start', ['fresh', 'carried'])
def test_net_fd_oracle_intervention_after_a_set_parameter(tmp_path, start):
    """``setParameter("dose",3)`` then ``setConcentration("A()","Atot*dose")``: ``A`` starts
    the measured phase at ``3*Atot``. The row was taken at the file's ``dose = 1``, so
    ``Atot``'s column came back exactly a third of central differences (445.90 against 1337.70
    fresh, 425.23 against 1275.70 carried, on both builds)."""
    net = Path(tmp_path) / 'dose.net'
    net.write_text(_DOSE_NET)
    actions = ['setParameter("dose",3)', 'setConcentration("A()","Atot*dose")', _SIM % (3, 'tc')]
    suffixes = [('simulate', 'tc')]
    if start == 'carried':
        actions = [_SIM % (2, 'eq')] + actions
        suffixes = [('simulate', 'eq')] + suffixes
    grad, fd, _ = _gradient_and_fd(
        tmp_path, _net_builder(net, actions, suffixes), ['tc'],
        [_free('Atot', 12.0), _free('kr', 0.6)], {'Atot': 10.0, 'kr': 0.5},
        ['Aobs', 'Bobs'], 0.2)
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_newton_refusal_does_not_depend_on_a_zero_seed_value(tmp_path, monkeypatch):
    """``setConcentration("A()","Atot*(dose-1)")`` before a Newton scan whose first dose is 1:
    the seed ``dA(0)/dAtot`` is zero at that point and ``1``, ``2`` at the next. The seed exists
    either way, and a build whose Newton solve drops a conserved species' seeding must be
    refused for it -- a row dropped for being zero at the first point used to let it run."""
    import pybnf.bngsim_model._runtime as runtime
    monkeypatch.setattr(runtime, 'steady_state_sensitivity_carries_ic_seeding', lambda: False)
    net = Path(tmp_path) / 'dose.net'
    net.write_text(_DOSE_NET)
    actions = ['setConcentration("A()","Atot*(dose-1)")',
               'parameter_scan({parameter=>"dose",par_scan_vals=>[1,2,3],steady_state=>1,'
               'ss_method=>"newton",t_end=>200,suffix=>"dr"})']
    model = _net_builder(net, actions, [('parameter_scan', 'dr')])({'Atot': 12.0, 'kr': 0.6})
    apply_routings(model, [route_for_model(model, ['Atot', 'kr'])])
    model.set_scored_suffixes({'dr'})
    with pytest.raises(PybnfError, match=r'respect to Atot, which sets .* of A\(\).*#704'):
        model.execute(str(tmp_path), 'zero_row', 60)


# ============================================================== scans of a fitted parameter ===

_SCAN_NET_ACTION = ('parameter_scan({parameter=>"%s",par_scan_vals=>[0.2,0.5,1],t_start=>0,'
                    't_end=>3,method=>"ode",suffix=>"dr"})')


@pytest.mark.bngsim
@_needs_output_sens
def test_net_scan_of_a_fitted_or_bound_parameter_is_refused_at_load(tmp_path):
    """A scan writes its own value to the swept parameter at every dose, so the loss does not
    move with the fitted value (central difference 0) while its column is the derivative at
    the dose: ``kr`` fitted and scanned assembled -472.32 against 0 on both builds. Refused at
    load, naming the scan and the parameter -- fitted by id, or bound by a condition. An
    unscored scan bears no sensitivities and is left alone."""
    net = Path(tmp_path) / 'dose.net'
    net.write_text(_DOSE_NET)
    build = _net_builder(net, [_SCAN_NET_ACTION % 'kr'], [('parameter_scan', 'dr')])
    with pytest.raises(GradientNotSupported,
                       match=r"parameter_scan with suffix 'dr' sweeps 'kr', which this fit "
                             r"differentiates \(a fitted parameter\)"):
        route_for_model(build({'kr': 0.6, 'kf': 1.2}), ['kr', 'kf'])
    bind = MutationSet([Mutation('kr', '=', 'x', is_param_ref=True)], 'c')
    model = _net_builder(net, [_SCAN_NET_ACTION % 'kr'], [('parameter_scan', 'dr')],
                         mutants=[bind])({'x': 0.6})
    with pytest.raises(GradientNotSupported,
                       match=r"suffix 'drc' sweeps 'kr'.*a condition binds a fitted parameter"):
        route_for_model(model, ['x'], bind)
    unscored = build({'kr': 0.6, 'kf': 1.2})
    unscored.set_scored_suffixes({'tc'})
    route_for_model(unscored, ['kr', 'kf'])


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_scan_of_a_fitted_parameter_is_refused_at_load(tmp_path):
    """The SBML sibling of the ``.net`` refusal: a ``ParamScan`` over a fitted ``k``."""
    scan = ParamScan({'param': 'k', 'min': '0.1', 'max': '1', 'step': '0.3', 'time': '5',
                      'suffix': 'dr'})
    model = _sbml_model(tmp_path, _SPECIES_IA_SBML, {'k': 0.3, 'S0': 100.0}, actions=(scan,))
    with pytest.raises(GradientNotSupported, match=r"suffix 'dr' sweeps 'k'"):
        route_for_model(model, ['k'])
    route_for_model(model, ['S0'])


# ======================================== writes an initialAssignment overwrites (#865 forms) ===

# S := S0 by an initialAssignment; k2 := 2*k1 by another; S decays at k, B is made from A at k2.
_SPECIES_IA_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="assigned">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="100" constant="true"/>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="k1" value="0.3" constant="true"/>
      <parameter id="k2" value="0" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>S0</ci></math></initialAssignment>
      <initialAssignment symbol="k2"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>k1</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
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


def _preequilibrated(equil, measure):
    action = TimeCourse({'time': '4', 'step': '1', 'suffix': 'tc'})
    action.set_preequilibration(equil, measure)
    return action


@pytest.mark.bngsim_sbml
@_needs_output_sens
@pytest.mark.parametrize('how', ['species fitted beside its input',
                                 'relative condition on the species',
                                 'equilibration condition', 'scan of it'])
def test_sbml_write_an_initial_assignment_overwrites_is_refused(tmp_path, how):
    """Every write SBML evaluates an ``initialAssignment`` over while its input moves: the
    species ``S := S0`` fitted beside ``S0`` (assembled dF/dS 3.68 against a central difference
    of 0), a relative condition ``S * 2``, a pre-equilibration's equilibration condition
    ``k2 = 1.5`` and a scan of ``k2`` while ``k1`` is fitted. The scalar path's handling is
    #865; the gradient path refuses each at load, naming the target and the input."""
    if how == 'species fitted beside its input':
        model, free, condition = _sbml_model(tmp_path, _SPECIES_IA_SBML,
                                             {'S': 120.0, 'S0': 110.0}), ['S', 'S0'], None
        target, feeder = 'S', 'S0'
    elif how == 'relative condition on the species':
        condition = MutationSet([Mutation('S', '*', 2.0)], 'dbl')
        model = _sbml_model(tmp_path, _SPECIES_IA_SBML, {'S0': 110.0}, mutants=[condition])
        free, target, feeder = ['S0'], 'S', 'S0'
    elif how == 'equilibration condition':
        action = _preequilibrated([('setParameter', 'k2', 1.5)], [])
        model = _sbml_model(tmp_path, _SPECIES_IA_SBML, {'k1': 0.35}, actions=(action,))
        free, condition, target, feeder = ['k1'], None, 'k2', 'k1'
    else:
        scan = ParamScan({'param': 'k2', 'min': '0.2', 'max': '1', 'step': '0.4', 'time': '4',
                          'suffix': 'dr'})
        model = _sbml_model(tmp_path, _SPECIES_IA_SBML, {'k1': 0.35}, actions=(scan,))
        free, condition, target, feeder = ['k1'], None, 'k2', 'k1'
    with pytest.raises(GradientNotSupported,
                       match=r"defines '%s' by an initialAssignment, .* writes it; the "
                             r"evaluation also changes .*'%s'" % (target, feeder)):
        route_for_model(model, free, condition)


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_writes_an_initial_assignment_keeps_are_not_refused(tmp_path):
    """The narrow side: the species ``S`` fitted with nothing that feeds ``S0`` moving keeps its
    written value, and its gradient matches central differences (4.016); the measurement
    condition of a pre-equilibration is written after the initial values are derived, so it
    stands and the load-time rule does not refuse it."""
    def build(values):
        return _sbml_model(tmp_path, _SPECIES_IA_SBML, values)

    grad, fd, _ = _gradient_and_fd(tmp_path, build, ['time_course'],
                                   [_free('S', 120.0), _free('k', 0.35)],
                                   {'S': 100.0, 'k': 0.3}, ['S'], 2.0, h=1e-5)
    np.testing.assert_allclose(grad, fd, rtol=1e-4)
    action = _preequilibrated([], [('setParameter', 'k2', 1.5)])
    route_for_model(_sbml_model(tmp_path, _SPECIES_IA_SBML, {'k1': 0.35}, actions=(action,)),
                    ['k1'])


# ===================================================================== the run-time floor ===

@pytest.mark.parametrize('version', ['0.15.1', 'built-from-source'])
@pytest.mark.parametrize('backend', ['net', 'sbml'])
def test_gradient_path_refuses_a_bngsim_below_its_floor(tmp_path, monkeypatch, version,
                                                        backend):
    """The gradient path needs bngsim 0.16.0 or later at run time, whatever the installed
    metadata says: a 0.15.x build from source differentiates an overridden derived parameter
    and a retired seed differently. Refused naming the installed version and the floor; a
    scalar fit (which never asks for sensitivities) is unaffected."""
    if not BNGSIM_HAS_OUTPUT_SENS:
        pytest.skip('needs a bngsim build with the output_sensitivities feature')
    monkeypatch.setattr(caps, 'BNGSIM_VERSION', version)
    if backend == 'net':
        import pybnf.bngsim_model as bngsim_model
        model = object.__new__(bngsim_model.BngsimModel)
        model.name = 'm'
    else:
        model = object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)
    with pytest.raises(PybnfError, match=r'%s.*0\.16\.0 or later' % version):
        model.enable_output_sensitivities(params=['k'], ic=[])
    monkeypatch.setattr(caps, 'BNGSIM_VERSION', '0.16.0')
    model.enable_output_sensitivities(params=['k'], ic=[])
    assert model._sensitivity_request.params == ['k']


# ================================================================ multiple-shooting lanes ===

@pytest.mark.bngsim
@_needs_output_sens
def test_net_segment_backend_writes_only_the_parameters_its_model_declares(tmp_path):
    """A lane writes the fit vector into its engine; a free parameter of another model in a
    multi-model fit is skipped by name rather than by swallowing whatever ``set_param`` raises,
    so any other failure reaches the run."""
    from pybnf.bngsim_model.parsing import _parse_simulate_action
    from pybnf.shooting import NetSegmentBackend
    net = Path(tmp_path) / 'dose.net'
    net.write_text(_DOSE_NET)
    action = _SIM % (3, 'tc')
    model = _net_builder(net, [action], [('simulate', 'tc')])(
        {'kr': 0.6, 'other_models_k': 2.0})
    backend = NetSegmentBackend(model, _parse_simulate_action(action), MutationSet(), 'tc')
    data = backend.simulate(model.param_set, np.linspace(0.0, 3.0, 4), None)
    assert np.all(np.isfinite(data.data))
