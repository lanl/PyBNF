"""Independent oracles for overridden derived parameters and scan-point seeding.

Each test compares the assembled gradient with central differences of PyBNF's own scalar loss
(fresh models built and run at each perturbed point, through no sensitivity code), with a
closed form where one exists. Where a bngsim build cannot give the right column, the only
acceptable alternative is a refusal that names the cause.
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

_OBJ = ChiSquareObjective()


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


def _pset(values):
    return PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                 for n, v in values.items()])


def _fit(build, keys, conditions, free, truth, columns, sigma, indvar='time'):
    """(assembled gradient, central differences) summed over the scored ``keys``.

    ``build(values)`` returns a fresh model holding ``values``; ``conditions[key]`` is the
    condition of experiment ``key`` (``None`` for the wildtype)."""
    names = [p.name for p in free]

    def run(values, routes=None):
        model = build(values)
        if routes is not None:
            apply_routings(model, list(routes.values()))
            model.set_scored_suffixes(set(keys))
        return model.execute(str(_fit.tmp), 'override_oracle', 60)

    truth_ds = run(truth)
    exps = {key: _exp_from(truth_ds[key], columns, sigma, indvar) for key in keys}
    fd = _central(lambda v: sum(_OBJ.evaluate(run(v)[key], exps[key]) for key in keys), free)
    theta = {p.name: p.value for p in free}
    base = build(theta)
    routes = {key: route_for_model(base, names, conditions.get(key)) for key in keys}
    ds = run(theta, routes)
    grad = assemble_gaussian_gradient(
        _OBJ, [(ds[key], exps[key], routes[key].at_point(theta)) for key in keys],
        free).gradient
    return grad, fd, ds


@pytest.fixture(autouse=True)
def _tmp(tmp_path):
    _fit.tmp = tmp_path


def _net_builder(net, actions, suffixes, mutants=()):
    def build(values):
        import pybnf.bngsim_model as bngsim_model
        model = bngsim_model.BngsimModel(Path(net).stem, list(actions), list(suffixes), [],
                                         nf=str(net))
        for mut in mutants:
            model.add_mutant(mut)
        model.param_set = _pset(values)
        return model
    return build


_SIM = 'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"tc"})'

# k2 = 2*k1 converts A to B; k1 also degrades B.
_OVERRIDE_NET = """begin parameters
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


@pytest.mark.bngsim
@_needs_output_sens
def test_net_one_condition_pins_a_derived_parameter_another_estimates_its_input(tmp_path):
    """Condition ``c2`` estimates ``k1`` (``k1 = x``), condition ``c3`` pins ``k2 = 0.8``. The
    sensitivity request is the union over experiments, so ``k1``'s column is requested in
    ``c3`` too; but in ``c3`` nothing routes ``x`` to it, so a stale chain through the pinned
    ``k2`` there is never read. The gradient must match central differences of the summed loss
    on every build -- refusing this is refusing a valid configuration."""
    net = tmp_path / 'override.net'
    net.write_text(_OVERRIDE_NET)
    c2 = MutationSet([Mutation('k1', '=', 'x', is_param_ref=True)], 'c2')
    c3 = MutationSet([Mutation('k2', '=', 0.8)], 'c3')
    grad, fd, _ = _fit(_net_builder(net, [_SIM], [('simulate', 'tc')], [c2, c3]),
                       ['tcc2', 'tcc3'], {'tcc2': c2, 'tcc3': c3}, [_free('x', 0.35)],
                       {'x': 0.3}, ['Aobs', 'Bobs'], 0.2)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


_CHAINED_NETS = {
    # k2 is read only through k3 = k2 + 0.1, which the conversion rate reads.
    'through another derived parameter': ("""begin parameters
    1 k1    0.3      # Constant
    2 k2    2*k1     # ConstantExpression
    3 k3    k2+0.1   # ConstantExpression
    4 kb    1        # Constant
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 0 1 kb #_R1
    2 1 2 k3 #_R2
    3 2 0 k1 #_R3
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
""", _SIM, ['Aobs', 'Bobs']),
    # k2 is read only through a global function the conversion rate calls.
    'through a rate-law function': ("""begin parameters
    1 k1    0.3      # Constant
    2 k2    2*k1     # ConstantExpression
    3 kb    1        # Constant
end parameters
begin functions
    1 f() k2*1
end functions
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 0 1 kb #_R1
    2 1 2 f #_R2
    3 2 0 k1 #_R3
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
""", _SIM, ['Aobs', 'Bobs']),
    # k2 is read by no reaction at all, only by a printed function the loss scores.
    'through an output function only': ("""begin parameters
    1 k1    0.3      # Constant
    2 k2    2*k1     # ConstantExpression
    3 kb    1        # Constant
end parameters
begin functions
    1 g() k2*Aobs
end functions
begin species
    1 A() 10
end species
begin reactions
    1 0 1 kb #_R1
    2 1 0 k1 #_R2
end reactions
begin groups
    1 Aobs                 1
end groups
""", _SIM.replace('suffix=>"tc"', 'suffix=>"tc",print_functions=>1'), ['g']),
}


@pytest.mark.bngsim
@_needs_output_sens
@pytest.mark.parametrize('how', list(_CHAINED_NETS))
def test_net_fitted_derived_parameter_reached_indirectly(tmp_path, how):
    """Fitting ``k2 = 2*k1`` beside ``k1`` overrides it. However the loss reaches ``k2`` -- via
    a second derived parameter, a rate-law function, or a printed function -- ``k1``'s column
    must not keep the chain through the overridden ``k2``. Either the gradient matches central
    differences of the scalar loss or the fit is refused naming ``k2``."""
    text, action, columns = _CHAINED_NETS[how]
    net = tmp_path / 'chained.net'
    net.write_text(text)
    try:
        grad, fd, _ = _fit(_net_builder(net, [action], [('simulate', 'tc')]), ['tc'], {},
                           [_free('k1', 0.35), _free('k2', 0.8)], {'k1': 0.3, 'k2': 0.6},
                           columns, 0.2)
    except (PybnfError, GradientNotSupported) as exc:
        assert "'k2'" in str(exc)
        return
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


# A <-> B; the dose-response writes A from a fitted total and the scanned dose.
_DOSE_NET = """begin parameters
    1 kf    1     # Constant
    2 kr    0.5   # Constant
    3 Atot  10    # Constant
    4 dose  1     # Constant
end parameters
begin species
    1 A() 0
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
@pytest.mark.parametrize('method', ['parity', 'newton'])
def test_net_dose_scan_intervention_reads_the_scanned_parameter(tmp_path, method):
    """``setConcentration("A()","Atot*dose")`` before a steady-state scan over ``dose``: each
    point starts at ``A(0) = Atot*dose``, so ``dB*/dAtot = dose*kf/(kf + kr)`` -- the scanned
    value, not the ``.net`` file's ``dose = 1``. Oracles: central differences of the scalar loss
    and that closed form. A Newton scan on a build without lanl/bngsim#704 must be refused."""
    import pybnf.bngsim_model._runtime as runtime
    net = tmp_path / 'dose.net'
    net.write_text(_DOSE_NET)
    newton = 'ss_method=>"newton",' if method == 'newton' else ''
    actions = ['setConcentration("A()","Atot*dose")',
               'parameter_scan({parameter=>"dose",par_scan_vals=>[0.5,2,4],steady_state=>1,'
               '%st_end=>200,suffix=>"dr"})' % newton]
    build = _net_builder(net, actions, [('parameter_scan', 'dr')])
    free = [_free('Atot', 12.0), _free('kr', 0.6)]

    def fit():
        return _fit(build, ['dr'], {}, free, {'Atot': 10.0, 'kr': 0.5}, ['Bobs'], 0.1,
                    indvar='dose')

    if method == 'newton' and not runtime.steady_state_sensitivity_carries_ic_seeding():
        with pytest.raises(PybnfError, match=r'lanl/bngsim#704'):
            fit()
        return
    grad, fd, ds = fit()
    sim = ds['dr']
    sens = sim.output_sensitivities
    dose = sim.data[:, sim.cols['dose']]
    col = sens.d_param[:, sens.selectors.index('observable:Bobs'), sens.param_names.index('Atot')]
    np.testing.assert_allclose(col, dose * 1.0 / (1.0 + 0.6), rtol=1e-4)
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


_SEEDED_EXCHANGE_NET = """begin parameters
    1 A0    10    # Constant
    2 kf    1     # Constant
    3 kr    0.5   # Constant
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
def test_net_scan_over_a_fitted_seed_parameter_gives_it_no_gradient(tmp_path):
    """A time-course scan over ``A0`` overwrites the fitted ``A0`` at every point, so the loss
    does not depend on the fitted value: its central difference is 0, and ``kr``'s is not.
    Either the gradient matches that or the configuration is refused naming ``A0``."""
    net = tmp_path / 'seeded_exchange.net'
    net.write_text(_SEEDED_EXCHANGE_NET)
    scan = ('parameter_scan({parameter=>"A0",par_scan_vals=>[5,10,20],t_start=>0,t_end=>3,'
            'method=>"ode",suffix=>"dr"})')
    try:
        grad, fd, _ = _fit(_net_builder(net, [scan], [('parameter_scan', 'dr')]), ['dr'], {},
                           [_free('A0', 12.0), _free('kr', 0.6)], {'A0': 10.0, 'kr': 0.5},
                           ['Bobs'], 0.1, indvar='A0')
    except (PybnfError, GradientNotSupported) as exc:
        assert 'A0' in str(exc)
        return
    assert fd[0] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-4)


_DERIVED_SBML = """<?xml version="1.0" encoding="UTF-8"?>
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


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_one_condition_pins_a_derived_parameter_another_estimates_its_input(tmp_path):
    """The SBML twin of the ``.net`` case: ``c2`` estimates ``k1`` (``k1 = x``) and ``c3`` pins
    the initialAssignment target ``k2 = 0.8``. Nothing in ``c3`` re-derives ``k2`` (``x``
    reaches ``k1`` only in ``c2``), so the pin stands there and ``c3`` does not depend on
    ``x``. The gradient must match central differences of the summed loss."""
    xml = tmp_path / 'derived_pin.xml'
    xml.write_text(_DERIVED_SBML)
    xml = str(xml)
    action = TimeCourse({'time': '10', 'step': '1'})
    c2 = MutationSet([Mutation('k1', '=', 'x', is_param_ref=True)], 'c2')
    c3 = MutationSet([Mutation('k2', '=', 0.8)], 'c3')

    def build(values):
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(xml, xml, pset=_pset(values),
                                                            actions=(action,))
        model.add_mutant(c2)
        model.add_mutant(c3)
        return model

    grad, fd, _ = _fit(build, ['time_coursec2', 'time_coursec3'],
                       {'time_coursec2': c2, 'time_coursec3': c3}, [_free('x', 0.35)],
                       {'x': 0.3}, ['B'], 0.05)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)
