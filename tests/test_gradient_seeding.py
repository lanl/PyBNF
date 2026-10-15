"""The run model a gradient fit simulates seeds what its routing assumed, on every bngsim build.

The router decides each free parameter's columns from two answers bngsim gives about the
unconfigured model: which derived parameters it holds live, and which initial-condition seeds
``∂x(0)/∂θ`` its parameter axis carries (:mod:`pybnf._bngsim_seeding`). PyBNF's own preparation
of a run model changed both without changing a concentration:

* a write-back of a derived parameter one ulp off overrode its expression (lanl/bngsim#188);
* an unlabelled ``save_concentrations()`` at a fresh state retires every seed on bngsim
  >= lanl/bngsim#982, which every ``.net`` evaluation does at its start, every scan point does,
  every snapshot a protocol takes does, and an SBML preparation does whenever it writes a
  species;
* a multiple-shooting knot was seeded from the parameters on bngsim before #982, though a knot
  does not depend on them.

Each FD oracle here is central differences of PyBNF's own scalar loss -- a fresh model built
and run at each perturbed point, which goes through no sensitivity code at all -- against the
gradient assembled from the routed columns, or a closed form. Each fails on a bngsim build
before this change; which build is named in the test.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf import _bngsim_seeding as seeding
from pybnf._bngsim_caps import (
    BNGSIM_HAS_OUTPUT_SENS,
    steady_state_sensitivity_carries_ic_seeding,
)
from pybnf.data import Data
from pybnf.gradient import (
    PARAM, RouteContribution, apply_routing, apply_routings, assemble_gaussian_gradient,
    route_for_model,
)
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, Mutation, MutationSet, PSet, TimeCourse

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')


def _fd_gradient(loss_at, free, h=1e-5):
    """Central differences of ``loss_at`` over the free parameters' sampling space."""
    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    grad = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)
    return grad


def _exp_from(sim, columns, sigma, indvar='time'):
    """Experimental Data from a run's own grid: ``columns`` with a constant ``_SD`` each."""
    x = sim.data[:, sim.cols[indvar]]
    blocks, headers = [x], [indvar]
    for name in columns:
        y = sim.data[:, sim.cols[name]]
        blocks += [y, np.full(len(y), sigma)]
        headers += [name, name + '_SD']
    return Data.from_columns(np.column_stack(blocks), headers)


def _theta(free, u_vec):
    return {p.name: p.from_sampling_space(u) for p, u in zip(free, u_vec)}


# =============================================================================== .net ===

# One species seeded bare (S() <- S0), one by an expression (A() <- 3*A0, which bngsim
# lowers to a derived carrier), one through a derived parameter (B() <- Btot = 2*A0). All
# three decay at k.
_SEEDED_NET = """begin parameters
    1 S0    100   # Constant
    2 k     0.3   # Constant
    3 A0    5     # Constant
    4 Btot  2*A0  # ConstantExpression
end parameters
begin species
    1 S() S0
    2 A() 3*A0
    3 B() Btot
end species
begin reactions
    1 1 0 k #_R1
    2 2 0 k #_R2
    3 3 0 k #_R3
end reactions
begin groups
    1 Sobs                 1
    2 Aobs                 2
    3 Bobs                 3
end groups
"""

_PROTOCOL = [
    'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"a"})',
    'resetConcentrations()',
    'setConcentration("S()",0)',
    'saveConcentrations()',
    'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"b"})',
    'resetConcentrations()',
    'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"c"})',
]


def _net_model(net, actions, suffixes, values, mutants=()):
    import pybnf.bngsim_model as bngsim_model
    model = bngsim_model.BngsimModel(Path(net).stem, list(actions), list(suffixes), [],
                                     nf=str(net))
    for mut in mutants:
        model.add_mutant(mut)
    model.param_set = PSet([FreeParameter(n, 'uniform_var', 0.0, 1e6, value=v)
                            for n, v in values.items()])
    return model


@pytest.mark.bngsim
@_needs_output_sens
def test_net_fd_oracle_seeded_initial_condition_on_the_parameter_axis(tmp_path):
    """The lanl/bngsim#982 sibling on the ``.net`` backend, in its plainest form.

    ``execute`` re-derives every species initializer and saves the result as the reset target
    (#450). The router routes ``S0`` on its own parameter axis, because bngsim seeds
    ``dS(0)/dS0 = 1`` into it; on bngsim >= #982 that save retired the seed, and ``S0``'s
    column came back exactly zero against a central difference of ``e^{-kt}``-weighted
    residuals. Fails on bngsim ``main`` before this change."""
    net = FIXTURES / 'e2e_ode_decay.net'
    actions = ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})']
    free = [FreeParameter('k', 'uniform_var', 0.01, 10.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 1.0, 1000.0, value=120.0)]

    def run(values, routed=False):
        model = _net_model(net, actions, [('simulate', 'tc')], values)
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free])
            apply_routing(model, routing)
        return model.execute(str(tmp_path), 'seed_net', 60)['tc'], routing

    exp = _exp_from(run({'k': 0.3, 'S0': 100.0})[0], ['Stot'], 5.0)
    grad_fd = _fd_gradient(lambda u: ChiSquareObjective().evaluate(
        run(_theta(free, u))[0], exp), free)
    sim, routing = run({p.name: p.value for p in free}, routed=True)
    assert routing.routes['S0'].contributions == (RouteContribution(PARAM, 'S0', 1.0),)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)
    # And the seeded column itself against its closed form, d Stot/d S0 = e^{-kt}.
    sens = sim.output_sensitivities
    t = sim.data[:, sim.cols['time']]
    col = sens.d_param[:, sens.selectors.index('observable:Stot'), sens.param_names.index('S0')]
    np.testing.assert_allclose(col, np.exp(-0.4 * t), rtol=1e-6)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_fd_oracle_seeding_survives_every_snapshot_a_protocol_takes(tmp_path):
    """Seeds through a reset to PyBNF's own baseline, through a user ``saveConcentrations()``
    of a fresh state, and through the reset back to that snapshot.

    Experiment ``a`` starts from the synced initial conditions; ``b`` from a snapshot whose
    ``S()`` was set to a literal (so ``S0`` no longer reaches it) while ``A()`` and ``B()`` still
    depend on ``A0`` -- through an expression and through a derived parameter; ``c`` resets back
    to that snapshot. Every reset drops bngsim's declarations, and on bngsim >= #982 every
    unlabelled save of a fresh state is seeded from nothing, so each of ``a``, ``b`` and ``c``
    lost ``A0``'s (and ``a`` lost ``S0``'s) seeding before this change. Fails on bngsim
    ``main`` before this change."""
    net = Path(tmp_path) / 'seeded.net'
    net.write_text(_SEEDED_NET)
    suffixes = [('simulate', 'a'), ('simulate', 'b'), ('simulate', 'c')]
    free = [FreeParameter('k', 'uniform_var', 0.01, 10.0, value=0.35),
            FreeParameter('S0', 'uniform_var', 1.0, 1000.0, value=90.0),
            FreeParameter('A0', 'uniform_var', 0.1, 100.0, value=6.0)]
    observables = ['Sobs', 'Aobs', 'Bobs']

    def run(values, routed=False):
        model = _net_model(net, _PROTOCOL, suffixes, values)
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free])
            apply_routing(model, routing)
            model.set_scored_suffixes({'a', 'b', 'c'})
        return model.execute(str(tmp_path), 'seed_protocol', 60), routing

    truth, _ = run({'k': 0.3, 'S0': 100.0, 'A0': 5.0})
    exps = {s: _exp_from(truth[s], observables, 2.0) for s in ('a', 'b', 'c')}

    def loss_at(u):
        ds, _ = run(_theta(free, u))
        return sum(ChiSquareObjective().evaluate(ds[s], exps[s]) for s in exps)

    grad_fd = _fd_gradient(loss_at, free)
    ds, routing = run({p.name: p.value for p in free}, routed=True)
    res = assemble_gaussian_gradient(
        ChiSquareObjective(), [(ds[s], exps[s], routing) for s in exps], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_fd_oracle_a_fitted_derived_parameter_cuts_its_chain(tmp_path):
    """Fitting ``Btot`` (defined as ``2*A0``) overrides its expression, so ``B(0) = Btot`` no
    longer depends on ``A0``. The seeding a re-derived state is declared with must come from a
    model in that same parameter state: read off one where ``Btot`` still tracks ``A0``, it
    would credit ``A0`` with ``B``'s initial value as well. Fails on bngsim ``main`` before this
    change, where the save at the start of the actions left ``A0`` and ``Btot`` no seeding at
    all."""
    net = Path(tmp_path) / 'seeded.net'
    net.write_text(_SEEDED_NET)
    actions = ['simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"a"})']
    free = [FreeParameter('k', 'uniform_var', 0.01, 10.0, value=0.35),
            FreeParameter('A0', 'uniform_var', 0.1, 100.0, value=6.0),
            FreeParameter('Btot', 'uniform_var', 0.1, 100.0, value=9.0)]
    observables = ['Aobs', 'Bobs']

    def run(values, routed=False):
        model = _net_model(net, actions, [('simulate', 'a')], values)
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free])
            apply_routing(model, routing)
        return model.execute(str(tmp_path), 'seed_pinned', 60)['a'], routing

    exp = _exp_from(run({'k': 0.3, 'A0': 5.0, 'Btot': 10.0})[0], observables, 1.0)
    grad_fd = _fd_gradient(lambda u: ChiSquareObjective().evaluate(
        run(_theta(free, u))[0], exp), free)
    sim, routing = run({p.name: p.value for p in free}, routed=True)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_fd_oracle_seed_through_a_condition(tmp_path):
    """The condition sibling: ``S0`` reaches the model only through a condition that sets it to
    a free parameter (ADR-0076), which runs on a mutant copy with its own initial-condition
    sync. Fails on bngsim ``main`` before this change."""
    net = FIXTURES / 'e2e_ode_decay.net'
    actions = ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})']
    cond = MutationSet([Mutation('S0', '=', 's0_free', is_param_ref=True)], 'c')
    free = [FreeParameter('k', 'uniform_var', 0.01, 10.0, value=0.4),
            FreeParameter('s0_free', 'uniform_var', 1.0, 1000.0, value=120.0)]

    def run(values, routed=False):
        model = _net_model(net, actions, [('simulate', 'tc')], values, mutants=[cond])
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free], cond)
            apply_routings(model, [routing])
            model.set_scored_suffixes({'tcc'})
        return model.execute(str(tmp_path), 'seed_cond', 60)['tcc'], routing

    exp = _exp_from(run({'k': 0.3, 's0_free': 100.0})[0], ['Stot'], 5.0)
    grad_fd = _fd_gradient(lambda u: ChiSquareObjective().evaluate(
        run(_theta(free, u))[0], exp), free)
    sim, routing = run({p.name: p.value for p in free}, routed=True)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


# A <-> B with A seeded by A0: the steady state moves with A0 through the conserved total, so
# a steady-state dose-response depends on A0 only through the seed (lanl/bngsim#704).
_EXCHANGE_NET = """begin parameters
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
@pytest.mark.parametrize('scan', [
    'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],t_start=>0,t_end=>3,'
    'method=>"ode",suffix=>"dr"})',
    'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],steady_state=>1,t_end=>200,'
    'suffix=>"dr"})',
    'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],steady_state=>1,'
    'ss_method=>"newton",t_end=>200,suffix=>"dr"})',
], ids=['time-course', 'steady-state', 'newton'])
def test_net_fd_oracle_scan_points_keep_their_seeding(tmp_path, scan):
    """Each dose point of a fresh-from-seed scan is a clone of the run model, re-synced and
    saved, so on bngsim >= #982 it was seeded from nothing: ``d(dose-response)/d A0`` came back
    zero at every point -- on the integrated time course, on the integrate-to-steady-state
    parity scan, and on the KINSOL steady state, which reads the seeding for the conserved total.
    At steady state ``B* = A0 kf/(kf + kr)``. Fails on bngsim ``main`` before this change.

    A bngsim build without lanl/bngsim#704 (0.16.0 is one) gets the KINSOL column wrong on its
    own -- zero -- whatever PyBNF seeds; there the Newton scan must be refused instead."""
    net = Path(tmp_path) / 'exchange.net'
    net.write_text(_EXCHANGE_NET)
    free = [FreeParameter('A0', 'uniform_var', 0.1, 100.0, value=12.0),
            FreeParameter('kr', 'uniform_var', 0.01, 10.0, value=0.6)]

    def run(values, routed=False):
        model = _net_model(net, [scan], [('parameter_scan', 'dr')], values)
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free])
            apply_routing(model, routing)
            model.set_scored_suffixes({'dr'})
        return model.execute(str(tmp_path), 'seed_scan', 60)['dr'], routing

    exp = _exp_from(run({'A0': 10.0, 'kr': 0.5})[0], ['Bobs'], 0.1, indvar='kf')
    if 'newton' in scan and not steady_state_sensitivity_carries_ic_seeding():
        # A bngsim build without lanl/bngsim#704 returns d B*/d A0 = 0 from its Newton solve,
        # so the scan is refused there, naming the parameter and the species, rather than run.
        with pytest.raises(PybnfError, match=r"respect to A0, which sets .* of A\(\)"):
            run({p.name: p.value for p in free}, routed=True)
        return
    grad_fd = _fd_gradient(lambda u: ChiSquareObjective().evaluate(
        run(_theta(free, u))[0], exp), free)
    sim, routing = run({p.name: p.value for p in free}, routed=True)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)
    if 'steady_state' in scan:
        kf = sim.data[:, sim.cols['kf']]
        sens = sim.output_sensitivities
        col = sens.d_param[:, sens.selectors.index('observable:Bobs'),
                           sens.param_names.index('A0')]
        np.testing.assert_allclose(col, kf / (kf + 0.6), rtol=1e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_newton_scan_refusal_is_limited_to_a_seeded_conserved_species(tmp_path, monkeypatch):
    """On a build whose Newton solve drops the seeding of a conserved total, the scan is refused
    when a fitted parameter seeds a species some conservation law holds -- and only then: in the
    decay network nothing is conserved, so ``S0``'s seeding cannot move a steady state and the
    scan runs."""
    import pybnf.bngsim_model._runtime as runtime
    monkeypatch.setattr(runtime, 'steady_state_sensitivity_carries_ic_seeding', lambda: False)
    newton = ('parameter_scan({parameter=>"%s",par_scan_vals=>[0.5,1,2],steady_state=>1,'
              'ss_method=>"newton",t_end=>200,suffix=>"dr"})')

    def gradient_run(net, scanned, values, free):
        model = _net_model(net, [newton % scanned], [('parameter_scan', 'dr')], values)
        apply_routing(model, route_for_model(model, free))
        model.set_scored_suffixes({'dr'})
        return model.execute(str(tmp_path), 'seed_newton', 60)['dr']

    exchange = Path(tmp_path) / 'exchange.net'
    exchange.write_text(_EXCHANGE_NET)
    with pytest.raises(PybnfError, match=r'lanl/bngsim#704'):
        gradient_run(exchange, 'kf', {'A0': 12.0, 'kr': 0.6}, ['A0', 'kr'])
    # kr alone seeds nothing, so the conserved total is not the question.
    gradient_run(exchange, 'kf', {'A0': 12.0, 'kr': 0.6}, ['kr'])
    gradient_run(FIXTURES / 'e2e_ode_decay.net', 'k', {'S0': 120.0, 'k': 0.3}, ['S0'])


@pytest.mark.bngsim
@_needs_output_sens
def test_net_segment_backend_seeds_the_model_start_and_not_a_knot():
    """Multiple shooting on the ``.net`` backend: a segment from the model's own initial state
    depends on ``S0`` (``dS/dS0 = e^{-kt}``), a segment from a knot does not (``0``). The lane's
    preparation saves the synced state, so on bngsim >= #982 the first came back ``0``; bngsim
    before #982 kept the seed through the knot's save, so there the second came back
    ``e^{-k dt}``. Fails on both builds before this change."""
    from pybnf.bngsim_model.parsing import _parse_simulate_action
    from pybnf.shooting import NetSegmentBackend
    net = FIXTURES / 'e2e_ode_decay.net'
    action = 'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})'
    k, s0 = 0.3, 100.0
    model = _net_model(net, [action], [('simulate', 'tc')], {'k': k, 'S0': s0})
    model.enable_output_sensitivities(params=['k', 'S0'], ic=['S()'])
    backend = NetSegmentBackend(model, _parse_simulate_action(action), MutationSet(), 'tc')
    pset = model.param_set

    def s0_column(data):
        sens = data.output_sensitivities
        return sens.d_param[:, sens.selectors.index('species:S()'),
                            sens.param_names.index('S0')]

    times = np.linspace(0.0, 4.0, 5)
    first = backend.simulate(pset, times, None)
    np.testing.assert_allclose(s0_column(first), np.exp(-k * times), rtol=1e-6)
    knot = backend.simulate(pset, np.linspace(4.0, 8.0, 5), {'S()': 42.0})
    np.testing.assert_allclose(s0_column(knot), 0.0, atol=1e-12)
    # The knot's save made it this lane's reset target; a later start from the model's own
    # initial state must still start there, value and seeding both.
    again = backend.simulate(pset, times, None)
    np.testing.assert_allclose(again.data[:, again.cols['S()']], s0 * np.exp(-k * times),
                               rtol=1e-6)
    np.testing.assert_allclose(s0_column(again), np.exp(-k * times), rtol=1e-6)


@pytest.mark.bngsim
@_needs_output_sens
def test_net_run_model_whose_seeding_differs_from_the_route_is_refused(tmp_path, monkeypatch):
    """The run-time check: a fresh-start sensitivity run whose seeding is not the one the
    route was built on stops with the pair named, rather than returning a column missing its
    seeding (here the route is told ``S()`` is not seeded from ``S0`` when it is)."""
    net = FIXTURES / 'e2e_ode_decay.net'
    actions = ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})']
    model = _net_model(net, actions, [('simulate', 'tc')], {'k': 0.3, 'S0': 100.0})
    apply_routing(model, route_for_model(model, ['k', 'S0']))
    monkeypatch.setattr(type(model), 'backend_ic_sensitivity', lambda self: {})
    with pytest.raises(PybnfError, match=r'seeded but assumed not: dS\(\)\(0\)/dS0'):
        model.execute(str(tmp_path), 'seed_refuse', 60)


# =============================================================================== SBML ===

# S's initial value is the bare initialAssignment S = S0; X is a plain species a free
# parameter can bind by id, which makes the preparation write and save a species.
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
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="r2" reversible="false" fast="false">
        <listOfReactants><speciesReference species="X" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# k_used = k_src by an initialAssignment; the rate law reads only k_used.
_ALIASED_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="aliased_param">
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
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k_used</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_ACTION = TimeCourse({'time': '10', 'step': '1'})


def _sbml_model(tmp_path, text, values, name='m.xml', actions=(_ACTION,)):
    xml = Path(tmp_path) / name
    xml.write_text(text)
    ps = PSet([FreeParameter(n, 'uniform_var', 0.0, 1e6, value=v) for n, v in values.items()])
    return bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=ps, actions=tuple(actions))


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_fd_oracle_seed_kept_when_a_species_write_forces_a_save(tmp_path):
    """A free parameter bound to species ``X`` makes the preparation write a species and save,
    and on bngsim >= #982 that save retired ``S``'s seed as well -- ``S0``'s column came back
    zero though nothing about ``S`` was written. Fails on bngsim ``main`` before this change."""
    free = [FreeParameter('k', 'uniform_var', 0.01, 10.0, value=0.32),
            FreeParameter('S0', 'uniform_var', 1.0, 1000.0, value=110.0),
            FreeParameter('X', 'uniform_var', 0.1, 100.0, value=8.0)]

    def run(values, routed=False):
        model = _sbml_model(tmp_path, _SEEDED_SBML, values)
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free])
            apply_routing(model, routing)
        return model.execute(str(tmp_path), 'seed_sbml', 0)['time_course'], routing

    exp = _exp_from(run({'k': 0.3, 'S0': 100.0, 'X': 5.0})[0], ['S', 'X'], 2.0)
    # A step of 1e-4: the X column is small beside the loss, and a smaller step leaves its
    # central difference at the integrator's own noise.
    grad_fd = _fd_gradient(lambda u: ChiSquareObjective().evaluate(
        run(_theta(free, u))[0], exp), free, h=1e-4)
    sim, routing = run({p.name: p.value for p in free}, routed=True)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim_sbml
@_needs_output_sens
@pytest.mark.parametrize('how', ['condition', 'scan', 'pre-equilibration condition'])
def test_sbml_every_species_write_keeps_the_other_seeds(tmp_path, how):
    """Every way a preparation writes a species -- a condition, a scan point, the condition a
    pre-equilibration's first phase starts under -- forces the same save, so each must leave the
    run model seeded exactly as the unconfigured model is. Fails on bngsim ``main`` before this
    change."""
    model = _sbml_model(tmp_path, _SEEDED_SBML, {'k': 0.4, 'S0': 120.0})
    model.enable_output_sensitivities(params=['k', 'S0'], ic=[])
    kwargs = {'condition': {'mut': MutationSet([Mutation('X', '=', 7.0)], 'c')},
              'scan': {'scan_override': ('X', 7.0)},
              'pre-equilibration condition': {'phase_overrides': [('X', 7.0)]}}[how]
    engine = model._engine_model_for_action(**kwargs)
    assert engine.get_concentration('X') == pytest.approx(7.0)
    assert engine.get_concentration('S') == pytest.approx(120.0)
    expected = seeding.seed_pairs(seeding.seed_rows(model._get_engine_template()), ['k', 'S0'])
    assert ('S', 'S0') in expected
    assert seeding.seed_pairs(seeding.seed_rows(engine, ['k', 'S0'])) == expected
    model._make_simulator(engine, 'ode')     # and the run-time check agrees


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_segment_backend_seeds_the_model_start_and_not_a_knot(tmp_path):
    """Multiple shooting on the SBML backend: ``dS/dS0 = e^{-kt}`` from the model's own start,
    ``0`` from a knot. Before this change the lane's preparation wrote ``S`` and saved, so on
    bngsim >= #982 the first came back ``0``; bngsim before #982 kept the seed through the
    knot's save, so there the second came back ``e^{-k dt}``. Fails on both builds before this
    change."""
    from pybnf.shooting import BngsimSegmentBackend
    k, s0 = 0.3, 100.0
    model = _sbml_model(tmp_path, _SEEDED_SBML, {'k': k, 'S0': s0})
    model.enable_output_sensitivities(params=['k', 'S0'], ic=['S', 'X'])
    backend = BngsimSegmentBackend(model, _ACTION, MutationSet(), 'time_course')
    pset = model.param_set

    def s0_column(data):
        sens = data.output_sensitivities
        return sens.d_param[:, sens.selectors.index('species:S'), sens.param_names.index('S0')]

    times = np.linspace(0.0, 4.0, 5)
    first = backend.simulate(pset, times, None)
    np.testing.assert_allclose(s0_column(first), np.exp(-k * times), rtol=1e-6)
    knot = backend.simulate(pset, np.linspace(4.0, 8.0, 5), {'S': 42.0, 'X': 3.0})
    np.testing.assert_allclose(s0_column(knot), 0.0, atol=1e-12)
    again = backend.simulate(pset, times, None)
    np.testing.assert_allclose(again.data[:, again.cols['S']], s0 * np.exp(-k * times),
                               rtol=1e-6)
    np.testing.assert_allclose(s0_column(again), np.exp(-k * times), rtol=1e-6)


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_a_derived_parameter_one_ulp_off_stays_live(tmp_path, monkeypatch):
    """The fragile edge of the write-back: libSBML's value for ``k_used`` one ulp away from
    bngsim's overrode the expression (lanl/bngsim#188), and ``k_src``'s column -- the only one
    the route reads -- then lost the chain and came back zero. A value bngsim already holds to
    rounding is no longer written."""
    model = _sbml_model(tmp_path, _ALIASED_SBML, {'k_src': 0.45})
    apply_routing(model, route_for_model(model, ['k_src']))
    real = model._recompute_initial_assignments

    def one_ulp_off(**kwargs):
        species, params = real(**kwargs)
        return species, {n: np.nextafter(v, np.inf) for n, v in params.items()}

    monkeypatch.setattr(model, '_recompute_initial_assignments', one_ulp_off)
    engine = model._engine_model_for_action()
    assert 'k_used' in seeding.live_parameters(engine)
    model._make_simulator(engine, 'ode')


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_a_live_derived_parameter_overridden_in_the_run_is_refused(tmp_path, monkeypatch):
    """When the recomputed value genuinely differs from bngsim's, the write still happens (it is
    the value the model is meant to have), the derived parameter is overridden, and the route
    -- built on it tracking its expression -- no longer matches the columns. That stops the run,
    naming the parameter, instead of returning ``k_src``'s column without its chain."""
    model = _sbml_model(tmp_path, _ALIASED_SBML, {'k_src': 0.45})
    apply_routing(model, route_for_model(model, ['k_src']))
    real = model._recompute_initial_assignments

    def far_off(**kwargs):
        species, params = real(**kwargs)
        return species, {n: 1.01 * v for n, v in params.items()}

    monkeypatch.setattr(model, '_recompute_initial_assignments', far_off)
    engine = model._engine_model_for_action()
    assert 'k_used' not in seeding.live_parameters(engine)
    with pytest.raises(PybnfError, match="'k_used' \\(routed as tracking its expression"):
        model._make_simulator(engine, 'ode')


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_run_model_whose_seeding_differs_from_the_route_is_refused(tmp_path, monkeypatch):
    """The SBML run-time check: the route is told ``S`` is not seeded from ``S0`` when the run
    model seeds it, so the router would have added the initial-condition term on top of a
    parameter column already carrying it. The run stops with the pair named."""
    model = _sbml_model(tmp_path, _SEEDED_SBML, {'k': 0.4, 'S0': 120.0})
    model.enable_output_sensitivities(params=['k', 'S0'], ic=[])
    monkeypatch.setattr(model, '_template_seed_rows', lambda: {})
    with pytest.raises(PybnfError, match=r'seeded but assumed not: dS\(0\)/dS0'):
        model.execute(str(tmp_path), 'seed_refuse', 0)


@pytest.mark.bngsim_sbml
@_needs_output_sens
def test_sbml_free_parameter_bound_to_an_assigned_parameter_is_refused(tmp_path):
    """``k_used`` is defined by an initialAssignment, so a free parameter of that name writes a
    value the assignment overwrites whenever the initial values are re-derived; the gradient path
    refuses it at setup, naming the parameter."""
    from pybnf.gradient import GradientNotSupported
    model = _sbml_model(tmp_path, _ALIASED_SBML, {'k_src': 0.45})
    with pytest.raises(GradientNotSupported, match="defines 'k_used' by an initialAssignment"):
        route_for_model(model, ['k_src', 'k_used'])


# The Bertozzi shape written in Antimony: I = I0 and S = N - I0 seed two species, and the
# initial assignment beta = R0*g/N derives the parameter the infection rate reads.
_SEIR_ANTIMONY = """model seir_like()
  compartment c = 1;
  species I in c, S in c, R in c;
  N = 100; I0 = 1; R0 = 2; g = 0.2;
  beta = R0*g/N;
  I = I0;
  S = N - I0;
  R = 0;
  infect: S + I -> 2 I; beta*I*S;
  recover: I -> R; g*I;
end
"""


@pytest.mark.bngsim_antimony
@_needs_output_sens
def test_antimony_fd_oracle_derived_parameter_and_seeded_species(tmp_path):
    """The Antimony backend shares the SBML preparation and router, so both bngsim regressions
    reach it: ``R0`` and ``g`` reach the model through the live ``beta`` (doubled on bngsim
    ``main`` before this change), and ``I0`` through two seeded species. Free parameters bind
    by id here, the way an Antimony model is usually fitted."""
    import pybnf.bngsim_antimony_model as bngsim_antimony_model
    path = Path(tmp_path) / 'seir.ant'
    path.write_text(_SEIR_ANTIMONY)
    free = [FreeParameter('I0', 'uniform_var', 0.1, 50.0, value=3.0),
            FreeParameter('R0', 'uniform_var', 0.5, 8.0, value=2.5),
            FreeParameter('g', 'uniform_var', 0.01, 2.0, value=0.25)]

    def run(values, routed=False):
        ps = PSet([FreeParameter(n, 'uniform_var', 0.0, 1e6, value=v) for n, v in values.items()])
        model = bngsim_antimony_model.BngsimAntimonyModelNoTimeout(
            str(path), str(path), pset=ps, actions=(_ACTION,))
        routing = None
        if routed:
            routing = route_for_model(model, [p.name for p in free])
            apply_routing(model, routing)
        return model.execute(str(tmp_path), 'seed_ant', 0)['time_course'], routing

    exp = _exp_from(run({'I0': 1.0, 'R0': 2.0, 'g': 0.2})[0], ['I'], 1.0)
    grad_fd = _fd_gradient(lambda u: ChiSquareObjective().evaluate(
        run(_theta(free, u))[0], exp), free)
    sim, routing = run({p.name: p.value for p in free}, routed=True)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=2e-4, atol=2e-4)


# ============================================================ helpers, no bngsim needed ===

class _FakeEngine:
    """Answers the two seeding questions from dictionaries, and honours declarations unless
    told not to."""

    def __init__(self, rows, honour=True):
        self.rows = {s: dict(r) for s, r in rows.items()}
        self.honour = honour
        self.declared = None

    def effective_ic_sensitivity(self, params=None):
        return {s: {p: v for p, v in r.items() if params is None or p in params}
                for s, r in self.rows.items()
                if any(params is None or p in params for p in r)}

    def declare_ic_sensitivity(self, rows):
        self.declared = rows
        if self.honour:
            self.rows = {s: dict(r) for s, r in rows.items() if r}


def test_holds_value_is_rounding_not_tolerance():
    assert seeding.holds_value(0.45, np.nextafter(0.45, 1.0))
    assert seeding.holds_value(0.0, 0.0)
    assert not seeding.holds_value(0.45, 0.45 * (1 + 1e-9))
    assert not seeding.holds_value(0.0, 1e-300)
    assert not seeding.holds_value(float('nan'), float('nan'))


def test_declare_rows_pins_what_was_not_captured_and_verifies():
    """Every species the engine would still seed but the capture did not is declared empty,
    and a build that does not report the declaration afterwards is refused."""
    engine = _FakeEngine({'S': {'S0': 1.0}, 'X': {'X0': 1.0}})
    seeding.declare_rows(engine, {'S': {'S0': 1.0}}, model_name='m', where='here')
    assert engine.declared == {'S': {'S0': 1.0}, 'X': {}}
    assert seeding.seed_pairs(engine.rows) == {('S', 'S0')}

    deaf = _FakeEngine({}, honour=False)
    with pytest.raises(PybnfError, match=r'assumed seeded but not seeded: dS\(0\)/dS0'):
        seeding.declare_rows(deaf, {'S': {'S0': 1.0}}, model_name='m', where='here')
    seeding.declare_rows(deaf, None, model_name='m', where='here')   # nothing captured


def test_check_seeding_names_each_pair_and_honours_its_limits():
    engine = _FakeEngine({'S': {'S0': 1.0}, 'A': {'A0': 2.0}})
    expected = {'S': {'S0': 1.0}}
    with pytest.raises(PybnfError, match=r'seeded but assumed not: dA\(0\)/dA0'):
        seeding.check_seeding(engine, expected, ['S0', 'A0'], model_name='m', where='here')
    # Not a requested column, not a pair the route decided on, or a species the run assigned:
    # none of those is a mismatch.
    seeding.check_seeding(engine, expected, ['S0'], model_name='m', where='here')
    seeding.check_seeding(engine, expected, ['S0', 'A0'], model_name='m', where='here',
                          pairs={('S', 'S0')})
    seeding.check_seeding(engine, expected, ['S0', 'A0'], model_name='m', where='here',
                          exempt_species={'A'})
