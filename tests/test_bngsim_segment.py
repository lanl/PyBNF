"""The segment integrator: one interval of a bngsim ``.net`` model from a carried species state.

Oracles, none of which goes through the integrator's own arithmetic: closed forms (a first-order
conversion, the same with its rate changed between segments, a clamped ligand's seed); one full
simulation, the integrator's own and ``BngsimModel.execute``'s, at two tolerances four decades
apart, so a disagreement that is not solver error fails the tighter one; the first segment
against ``execute()``, byte for byte; and the ordinary path's own piecewise-constant run
(``setParameter`` and ``continue=>1``).
"""

import copy
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from pybnf.bngsim_model import BngsimModel, BngsimNfModel, SegmentFailed, SegmentIntegrator
from pybnf.printing import PybnfError
from pybnf.pset import BNGLModel, FreeParameter, MutationSet, PSet

bngsim_only = pytest.mark.bngsim

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

# |chain - full| in tolerance units, rtol |y| + atol: 5 to 53 measured (bngsim 0.16.0) on these
# models and segment counts. It is not a general bound: the gap grows with the segments.
TOLERANCE_UNITS = 100.0

CONVERSION = """begin parameters
    1 k   0.7
    2 A0  5
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

# A -> B at k, B -> C at the derived rate k2 = 2 k, and a function of k.
PIECEWISE = """begin parameters
    1 k   0.7
    2 A0  5
    3 k2  2*k
end parameters
begin species
    1 A() A0
    2 B() 0
    3 C() 0
end species
begin reactions
    1 1 2 k
    2 2 3 k2
end reactions
begin groups
    1 Atot 1
    2 Btot 2
    3 Ctot 3
end groups
begin functions
    1 rateA() k*Atot
end functions
"""

# Seasonally forced SIR: its rate reads time(), so a segment on the wrong clock would show.
SEASONAL_SIR = """begin parameters
    1 beta0  1.2
    2 amp    0.3
    3 period 7
    4 gamma  0.4
    5 N      1000
    6 I0     2
end parameters
begin species
    1 S() N-I0
    2 I() I0
    3 R() 0
    4 Cum() 0
end species
begin reactions
    1 1,2 2,2,4 beta_t
    2 2 3 gamma
end reactions
begin groups
    1 Inf 2
    2 CumInf 4
end groups
begin functions
    1 beta_t() beta0*(1+amp*sin(6.283185307179586*time()/period))/N
end functions
"""

# Initial conditions set by C0 three ways (#450): a derived _InitialConc parameter (A), a bare
# name (B) and an inline expression (D); k0 is derived from k0__FREE and f is a function.
SEEDED = """begin parameters
    1 gamma     1/7
    2 k0__FREE  0.5
    3 k0        k0__FREE
    4 two_k0    2*k0
    5 C0        3
    6 _InitialConc1 2*C0
end parameters
begin species
    1 A() _InitialConc1
    2 B() C0
    3 D() 3*C0
    4 E() 1.5
end species
begin reactions
    1 1 2 two_k0
    2 2 3 gamma
end reactions
begin groups
    1 Atot 1
    2 Btot 2
    3 Dtot 3
end groups
begin functions
    1 f() Atot*k0
end functions
"""

# dX/dt = k X^2 blows up at t = 1/(k X0).
BLOWUP = """begin parameters
    1 k   1.0
    2 X0  1.0
end parameters
begin species
    1 X() X0
end species
begin reactions
    1 1,1 1,1,1 k
end reactions
begin groups
    1 Xtot 1
end groups
"""

# Seeds the #450 sync does not set: a clamped species, a parameter named lambda (a Python
# keyword), and C0^2 and ln(D0), which Python does not read.
SYNC_UNSET = """begin parameters
    1 L0      5
    2 lambda  3
    3 C0      2
    4 D0      5
    5 k       0.1
end parameters
begin species
    1 $L()  L0
    2 A()   lambda
    3 B()   C0^2
    4 D()   ln(D0)
    5 E()   0
end species
begin reactions
    1 2 5 k
end reactions
begin groups
    1 Ltot 1
    2 Atot 2
    3 Btot 3
    4 Dtot 4
    5 Etot 5
end groups
"""

# Written by BioNetGen 2.9.2 from a compartmental BNGL model with the clamped ligand $L(r)@EC L0.
CLAMPED_BNG2 = """# Created by BioNetGen 2.9.2
begin parameters
    1 k          0.1  # Constant
    2 L0         3  # Constant
    3 R0         2  # Constant
    4 V_EC       10  # Constant
    5 V_PM       1  # Constant
    6 _rateLaw1  0.05  # Constant
end parameters
begin species
    1 @EC::$L(r) L0
    2 @PM::R(l) R0
    3 @PM::L(r!1)@EC.R(l!1) 0
end species
begin reactions
    1 1,2 3 0.1*k #_R1 unit_conversion=1/V_EC
    2 3 1,2 _rateLaw1 #_reverse__R1
end reactions
begin groups
    1 Ltot                 1,3
    2 Rtot                 2,3
    3 LR                   3
end groups
"""

def _action(t_end=10, n_steps=10, suffix='tc', extra='', start='t_start=>0'):
    return ('simulate({method=>"ode",%s,t_end=>%s,n_steps=>%d,suffix=>"%s",print_functions=>1%s})'
            % (start, t_end, n_steps, suffix, extra))


def _tolerances(atol, rtol):
    return ',atol=>%s,rtol=>%s' % (atol, rtol)


@pytest.fixture(scope='module')
def nets(tmp_path_factory):
    folder = tmp_path_factory.mktemp('segment_nets')
    paths = {'reversible': str(FIXTURES / 'two_species_reversible.net')}
    for name, text in (('conversion', CONVERSION), ('piecewise', PIECEWISE),
                       ('seasonal_sir', SEASONAL_SIR), ('seeded', SEEDED), ('blowup', BLOWUP),
                       ('sync_unset', SYNC_UNSET), ('clamped', CLAMPED_BNG2)):
        path = folder / ('%s.net' % name)
        path.write_text(text)
        paths[name] = str(path)
    return paths


def _model(net, actions, mutants=()):
    actions = [actions] if isinstance(actions, str) else list(actions)
    suffixes = [('simulate', line.split('suffix=>"', 1)[1].split('"', 1)[0])
                for line in actions if line.startswith('simulate(') and 'suffix=>"' in line]
    return BngsimModel(Path(net).stem, actions, suffixes, list(mutants), nf=net)


def _execute(model, names, values, folder, suffix='tc'):
    model.param_set = PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=float(v))
                            for n, v in zip(names, values)])
    return model.execute(str(folder), 'x', None)[suffix]


def _rows(data):
    return np.asarray(data.data, dtype=float)


def _tolerance_units(a, b, atol, rtol):
    """max |a - b| / (rtol |b| + atol) over every column but time."""
    return float(np.max(np.abs(a[:, 1:] - b[:, 1:]) / (rtol * np.abs(b[:, 1:]) + atol)))


def _chain(integrator, values, grid):
    """The rows of a chain of one-interval segments over ``grid``, from the initial state."""
    state = integrator.initial_state(values)
    rows = []
    for t0, t1 in zip(grid[:-1], grid[1:]):
        segment = integrator.integrate(state, values, t0, t1)
        rows.extend(_rows(segment.data)[[0, -1]] if not rows else _rows(segment.data)[-1:])
        state = segment.state
    return np.array(rows)


# ---------------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------------

@bngsim_only
def test_a_chain_of_segments_matches_the_closed_form_of_a_first_order_conversion(nets):
    """A(t) = A0 e^{-kt}, B(t) = A0 (1 - e^{-kt}); each segment reports the carried state at its
    start, the requested times and its end."""
    k, a0 = 0.7, 5.0
    atol, rtol = 1e-12, 1e-10
    integrator = SegmentIntegrator(
        _model(nets['conversion'], _action(extra=_tolerances(atol, rtol))), ['k', 'A0'])
    bound = TOLERANCE_UNITS * (rtol * a0 + atol)
    values = [k, a0]
    state = integrator.initial_state(values)
    np.testing.assert_array_equal(state, [a0, 0.0])
    edges = [0.0, 0.5, 1.7, 3.0, 6.25, 10.0]
    for t0, t1 in zip(edges[:-1], edges[1:]):
        inner = np.linspace(t0, t1, 5)[1:-1]
        segment = integrator.integrate(state, values, t0, t1, times=inner)
        rows, cols = _rows(segment.data), segment.data.cols
        times = rows[:, cols['time']]
        np.testing.assert_array_equal(times, [t0, *inner, t1])
        assert rows[0, cols['Atot']] == state[0]
        np.testing.assert_allclose(rows[:, cols['Atot']], a0 * np.exp(-k * times), rtol=0,
                                   atol=bound)
        np.testing.assert_allclose(rows[:, cols['Btot']], a0 * (1 - np.exp(-k * times)), rtol=0,
                                   atol=bound)
        state = segment.state
    assert integrator.n_integrations == len(edges) - 1


@bngsim_only
@pytest.mark.parametrize('rtol', [1e-6, 1e-10])
@pytest.mark.parametrize('name, names, values, t_end, intervals', [
    ('reversible', ['kf', 'kr'], [0.002, 0.05], 50, 25),
    ('seasonal_sir', ['beta0', 'gamma'], [1.3, 0.35], 60, 60),
], ids=['reversible', 'seasonal_sir'])
def test_a_chain_of_segments_is_one_full_simulation(nets, name, names, values, t_end, intervals,
                                                    rtol, tmp_path):
    """lanl/PyBNF #973: the segment path and the full re-simulation path give the same
    predictions, to solver tolerance."""
    model = _model(nets[name], _action(t_end, intervals, extra=_tolerances(rtol, rtol)))
    integrator = SegmentIntegrator(model, names)
    grid = np.linspace(0.0, t_end, intervals + 1)
    chain = _chain(integrator, values, grid)
    full = _rows(integrator.integrate(integrator.initial_state(values), values, grid[0], grid[-1],
                                      times=grid[1:-1]).data)
    ordinary = _rows(_execute(model, names, values, tmp_path))
    np.testing.assert_array_equal(ordinary[:, 0], grid)
    assert _tolerance_units(chain, full, rtol, rtol) <= TOLERANCE_UNITS
    assert _tolerance_units(chain, ordinary, rtol, rtol) <= TOLERANCE_UNITS


@bngsim_only
@pytest.mark.parametrize('names, values', [
    (['C0'], [4.0]),
    (['k0__FREE', 'C0', 'gamma'], [0.8, 2.5, 0.3]),
], ids=['initial-condition-only', 'rates-and-initial-condition'])
def test_the_first_segment_is_the_ordinary_simulation_byte_for_byte(nets, names, values,
                                                                     tmp_path):
    """The same initial conditions, codegen artifact, tolerances (written as expressions here)
    and columns as ``execute()``."""
    model = _model(nets['seeded'], _action(6, 12, extra=_tolerances('1e-5*1e-5', '2e-9')))
    integrator = SegmentIntegrator(model, names)
    assert integrator._tolerances == {'atol': 1e-5 * 1e-5, 'rtol': 2e-9}
    segment = integrator.integrate(integrator.initial_state(values), values, 0.0, 6.0,
                                   times=np.linspace(0.0, 6.0, 13)[1:-1])
    ordinary = _execute(model, names, values, tmp_path)
    assert segment.data.cols == ordinary.cols
    np.testing.assert_array_equal(_rows(segment.data), _rows(ordinary))


def test_the_quiet_hold_covers_child_loggers_and_lasts_until_the_last_holder_leaves(caplog):
    from pybnf.bngsim_model import segment
    parent, child = logging.getLogger('bngsim'), logging.getLogger('bngsim.sensitivity')
    with caplog.at_level(logging.INFO, logger='bngsim'):
        with segment._QUIET:
            child.info('child during')
            with segment._QUIET:            # a second integration, as from another thread
                parent.info('parent during')
            child.info('child while the first hold stands')
        parent.info('after')
        child.info('child after')
        assert parent.level == logging.INFO     # the level caplog set is back
    assert [r.getMessage() for r in caplog.records if r.name.startswith('bngsim')] == \
        ['after', 'child after']


@bngsim_only
def test_a_segment_keeps_bngsim_quiet_only_while_it_integrates(nets, caplog):
    integrator = SegmentIntegrator(_model(nets['seeded'], _action()), ['C0'])
    with caplog.at_level(logging.INFO, logger='bngsim'):
        integrator.integrate(integrator.initial_state([4.0]), [4.0], 0.0, 1.0)
        logging.getLogger('bngsim').info('afterwards')
    assert [r.getMessage() for r in caplog.records if r.name == 'bngsim'] == ['afterwards']


@bngsim_only
def test_segments_use_the_tolerances_of_the_experiments_action(nets, tmp_path):
    names, values = ['beta0', 'gamma'], [1.3, 0.35]
    grid = np.linspace(0.0, 30.0, 31)
    results = {}
    for tol in ('1e-4', '1e-10'):
        model = _model(nets['seasonal_sir'], _action(30, 30, extra=_tolerances(tol, tol)))
        integrator = SegmentIntegrator(model, names)
        results[tol] = _rows(integrator.integrate(integrator.initial_state(values), values, 0.0,
                                                  30.0, times=grid[1:-1]).data)
        np.testing.assert_array_equal(results[tol], _rows(_execute(model, names, values, tmp_path)))
    assert not np.array_equal(results['1e-4'], results['1e-10'])


@bngsim_only
def test_parameters_changed_between_segments_are_piecewise_constant_parameters(nets, tmp_path):
    """k is 0.7 on [0, 3], 0.2 on [3, 5] and 1.5 on [5, 10], as the kernel would move it: A in
    closed form, and the ordinary path's setParameter and continue=>1 run, which shares no code
    with this one. The derived k2 and the function rateA follow k; A0, written late, does not
    matter."""
    atol, rtol = 1e-12, 1e-10
    tol = _tolerances(atol, rtol)
    names = ['k', 'A0']
    integrator = SegmentIntegrator(_model(nets['piecewise'], _action(10, 10, extra=tol)), names)
    first = integrator.integrate(integrator.initial_state([0.7, 5.0]), [0.7, 5.0], 0.0, 3.0,
                                 times=[1.0, 2.0])
    second = integrator.integrate(first.state, [0.2, 5.0], 3.0, 5.0, times=[4.0])
    third = integrator.integrate(second.state, [1.5, 999.0], 5.0, 10.0, times=[6.0, 7.0, 8.0, 9.0])
    a3 = 5.0 * np.exp(-0.7 * 3.0)
    a5 = a3 * np.exp(-0.2 * 2.0)
    for segment, closed in ((first, lambda t: 5.0 * np.exp(-0.7 * t)),
                            (second, lambda t: a3 * np.exp(-0.2 * (t - 3.0))),
                            (third, lambda t: a5 * np.exp(-1.5 * (t - 5.0)))):
        rows = _rows(segment.data)
        np.testing.assert_allclose(rows[:, segment.data.cols['Atot']],
                                   closed(rows[:, segment.data.cols['time']]),
                                   rtol=0, atol=TOLERANCE_UNITS * (rtol * 5.0 + atol))
    assert _rows(second.data)[0, second.data.cols['rateA']] == pytest.approx(0.2 * a3, rel=1e-8)

    ordinary = _model(nets['piecewise'], [
        _action(3, 3, suffix='p1', extra=tol), 'setParameter("k", 0.2)',
        _action(5, 2, suffix='p2', extra=tol, start='continue=>1'), 'setParameter("k", 1.5)',
        _action(10, 5, suffix='p3', extra=tol, start='continue=>1')])
    ordinary.param_set = PSet([FreeParameter(n, 'uniform_var', 0, 1e3, value=v)
                               for n, v in zip(names, [0.7, 5.0])])
    out = ordinary.execute(str(tmp_path), 'piecewise', None)
    for suffix, segment in (('p1', first), ('p2', second), ('p3', third)):
        assert out[suffix].cols == segment.data.cols
        np.testing.assert_array_equal(_rows(out[suffix])[:, 0], _rows(segment.data)[:, 0])
        assert _tolerance_units(_rows(segment.data), _rows(out[suffix]), atol, rtol) \
            <= TOLERANCE_UNITS


@bngsim_only
def test_a_carried_state_is_integrated_as_given_when_a_parameter_write_would_reseed_it(nets):
    """Writing C0 = 9 with the state C0 = 3 seeds must integrate that state, byte for byte as
    with C0 = 3: the state is written after the parameters, since bngsim re-derives seeded
    species on a parameter write."""
    x0 = np.array([6.0, 3.0, 9.0, 1.5])
    names = ['k0__FREE', 'C0']
    moved = SegmentIntegrator(_model(nets['seeded'], _action(6, 6)), names)
    kept = SegmentIntegrator(_model(nets['seeded'], _action(6, 6)), names)
    a = moved.integrate(x0, [0.5, 9.0], 0.0, 2.0)
    b = kept.integrate(x0, [0.5, 3.0], 0.0, 2.0)
    np.testing.assert_array_equal(_rows(a.data)[0, 1:4], [6.0, 3.0, 9.0])
    np.testing.assert_array_equal(_rows(a.data), _rows(b.data))
    np.testing.assert_array_equal(a.state, b.state)


@bngsim_only
@pytest.mark.parametrize('t0', [-2.5, 1000.25])
def test_a_function_of_time_is_read_at_the_absolute_time_of_a_segment_start(nets, t0):
    integrator = SegmentIntegrator(_model(nets['seasonal_sir'], _action()), ['beta0', 'amp'])
    segment = integrator.integrate(integrator.initial_state([1.3, 0.3]), [1.3, 0.3], t0, t0 + 1.0)
    row = _rows(segment.data)[0]
    assert row[segment.data.cols['time']] == t0
    expected = 1.3 * (1 + 0.3 * np.sin(2 * np.pi * t0 / 7.0)) / 1000.0
    assert row[segment.data.cols['beta_t']] == pytest.approx(expected, rel=1e-12)


@bngsim_only
def test_every_initial_state_follows_its_own_parameters_where_the_sync_sets_nothing(nets,
                                                                                   tmp_path):
    """Three particles in turn on one integrator. Each seed comes from bngsim's re-derivation,
    which the sync switches off on an engine it touched. Oracles: ``execute()`` on a fresh model,
    and ``[L0, lambda, C0^2, ln D0, 0]`` by hand where bngsim derives B and D (0.15 loads an
    inline ``C0^2`` as 0)."""
    import bngsim

    names = ['L0', 'lambda', 'C0', 'D0']
    probe = bngsim.Model.from_net(nets['sync_unset'])
    probe.set_params({'C0': 4.0, 'D0': 7.0})
    probe.reset()
    columns = [0, 1, 2, 3, 4] if probe.get_state()[2] == 16.0 else [0, 1, 4]
    integrator = SegmentIntegrator(_model(nets['sync_unset'], _action(4, 4)), names)
    for values in ([5.0, 3.0, 2.0, 5.0], [7.0, 11.0, 4.0, 7.0], [2.0, 0.5, 3.0, 9.0]):
        l0, lam, c0, d0 = values
        state = integrator.initial_state(values)
        by_hand = np.array([l0, lam, c0 ** 2, math.log(d0), 0.0])
        np.testing.assert_allclose(state[columns], by_hand[columns], rtol=1e-15, atol=0)
        fresh = _execute(_model(nets['sync_unset'], _action(4, 4)), names, values, tmp_path)
        np.testing.assert_array_equal(_rows(fresh)[0, 1:], state)


@bngsim_only
def test_every_initial_state_of_a_clamped_ligand_follows_its_own_dose(nets, tmp_path):
    names = ['L0', 'R0', 'k']
    integrator = SegmentIntegrator(_model(nets['clamped'], _action(20, 20)), names)
    for values in ([3.0, 2.0, 0.1], [7.0, 5.0, 0.1], [11.0, 6.0, 0.1]):
        fresh = _execute(_model(nets['clamped'], _action(20, 20)), names, values, tmp_path)
        np.testing.assert_array_equal(_rows(fresh)[0, 1:3], values[:2])
        np.testing.assert_array_equal(integrator.initial_state(values), [*values[:2], 0.0])


def _engine_snapshot(model):
    engine = model._engine_model
    return [engine.get_param(n) for n in engine.param_names], engine.get_state().tolist()


@bngsim_only
def test_an_integrator_starts_from_the_net_file_and_never_writes_to_its_model(nets, tmp_path):
    """A model whose engine ran under other parameters, one of them (C0) not the integrator's,
    gives the segments a fresh model gives, and its engine is left as it was."""
    names, values = ['k0__FREE'], [0.9]
    used = _model(nets['seeded'], _action(6, 6))
    _execute(used, ['k0__FREE', 'C0'], [0.2, 9.0], tmp_path)
    before = _engine_snapshot(used)
    a = SegmentIntegrator(used, names)
    b = SegmentIntegrator(_model(nets['seeded'], _action(6, 6)), names)
    np.testing.assert_array_equal(a.initial_state(values), [6.0, 3.0, 9.0, 1.5])   # C0 = 3
    np.testing.assert_array_equal(_rows(a.integrate(a.initial_state(values), values, 0, 4).data),
                                  _rows(b.integrate(b.initial_state(values), values, 0, 4).data))
    assert _engine_snapshot(used) == before


@bngsim_only
def test_accepts_the_second_of_two_independent_experiments(nets, tmp_path):
    """Edition 2 resets the parameters and species before each experiment of a model, so the
    second starts from the seed state."""
    model = _model(nets['conversion'], [
        'saveParameters("__pybnf_experiment_start")', 'resetConcentrations()',
        _action(suffix='first'),
        'resetParameters("__pybnf_experiment_start")', 'resetConcentrations()',
        _action(suffix='second')])
    integrator = SegmentIntegrator(model, ['k'], suffix='second')
    segment = integrator.integrate(integrator.initial_state([0.4]), [0.4], 0.0, 10.0,
                                   times=np.linspace(0.0, 10.0, 11)[1:-1])
    np.testing.assert_array_equal(_rows(segment.data),
                                  _rows(_execute(model, ['k'], [0.4], tmp_path, suffix='second')))


# ---------------------------------------------------------------------------------------------
# Copies and failures
# ---------------------------------------------------------------------------------------------

@bngsim_only
def test_a_copy_and_an_unpickled_integrator_share_no_engine(nets):
    """Interleaved calls on an integrator, its copies and its unpickled twin, each under its own
    parameters, give each the result a fresh integrator gives, byte for byte."""
    names = ['beta0', 'gamma']
    original = SegmentIntegrator(_model(nets['seasonal_sir'], _action()), names)
    original.integrate(original.initial_state([1.3, 0.35]), [1.3, 0.35], 0.0, 1.0)
    twin = pickle.loads(pickle.dumps(original))
    duplicate = copy.copy(original)
    deep = copy.deepcopy(original)
    assert twin._engine is None and duplicate._engine is None and deep._engine is None
    points = {'original': [1.3, 0.35], 'duplicate': [0.9, 0.2], 'twin': [2.0, 0.6], 'deep': [1.1, 0.5]}
    integrators = {'original': original, 'duplicate': duplicate, 'twin': twin, 'deep': deep}
    got = {}
    for _round in range(2):
        for key, integrator in integrators.items():
            values = points[key]
            got[key] = integrator.integrate(integrator.initial_state(values), values, 3.0, 9.0)
    assert len({id(i._engine) for i in integrators.values()}) == 4
    for key, values in points.items():
        fresh = SegmentIntegrator(_model(nets['seasonal_sir'], _action()), names)
        expected = fresh.integrate(fresh.initial_state(values), values, 3.0, 9.0)
        np.testing.assert_array_equal(_rows(got[key].data), _rows(expected.data))
        np.testing.assert_array_equal(got[key].state, expected.state)


@bngsim_only
def test_an_integrator_failure_is_a_failed_segment_and_the_next_call_starts_clean(nets):
    """At k = 1, X0 = 1 the solution blows up at t = 1; the next call, at k = 0.1, is what a
    fresh integrator gives."""
    import bngsim

    integrator = SegmentIntegrator(_model(nets['blowup'], _action(2, 2)), ['k'])
    with pytest.raises(SegmentFailed) as failed:
        integrator.integrate(np.array([1.0]), [1.0], 0.0, 2.0)
    assert isinstance(failed.value.__cause__, bngsim.SimulationError)
    after = integrator.integrate(np.array([1.0]), [0.1], 0.0, 2.0)
    fresh = SegmentIntegrator(_model(nets['blowup'], _action(2, 2)), ['k'])
    np.testing.assert_array_equal(_rows(after.data),
                                  _rows(fresh.integrate(np.array([1.0]), [0.1], 0.0, 2.0).data))
    assert after.state[0] == pytest.approx(1.0 / (1.0 - 0.1 * 2.0), rel=1e-6)


@bngsim_only
def test_an_integration_that_ends_at_a_state_that_is_not_finite_is_a_failed_segment(
        nets, monkeypatch):
    """bngsim 0.16 refuses a right-hand side that is not finite itself; the end state stands in
    here for a solver that does not."""
    integrator = SegmentIntegrator(_model(nets['conversion'], _action()), ['k'])
    engine, sim = integrator._integrating_engine()

    class Overflowed:
        def __getattr__(self, name):
            return getattr(engine, name)

        def get_state(self):
            return np.array([np.inf, 0.0])

    monkeypatch.setattr(integrator, '_integrating_engine', lambda: (Overflowed(), sim))
    with pytest.raises(SegmentFailed, match='ended at a state that is not finite'):
        integrator.integrate([5.0, 0.0], [0.7], 0.0, 1.0)


@bngsim_only
def test_a_timeout_is_a_failed_segment(nets):
    import bngsim

    integrator = SegmentIntegrator(_model(nets['blowup'], _action(2, 2)), ['k'], timeout=1e-9)
    with pytest.raises(SegmentFailed) as failed:
        integrator.integrate(np.array([1.0]), [1.0], 0.0, 0.99,
                             times=np.linspace(0.0, 0.99, 20001)[1:-1])
    assert isinstance(failed.value.__cause__, bngsim.SimulationTimeout)


@bngsim_only
def test_an_initial_state_that_is_not_finite_is_a_failed_segment(nets):
    """A() = 2 C0 overflows at C0 = 1e308, a finite value."""
    integrator = SegmentIntegrator(_model(nets['seeded'], _action()), ['C0'])
    np.testing.assert_array_equal(integrator.initial_state([3.0]), [6.0, 3.0, 9.0, 1.5])
    with pytest.raises(SegmentFailed, match=r'initial state .* not finite \(A\(\) = inf'):
        integrator.initial_state([1e308])


@bngsim_only
@pytest.mark.parametrize('error', ['ParameterError', 'ValueError'])
def test_an_error_that_is_not_an_integrator_failure_propagates(nets, monkeypatch, error):
    """Only SimulationError and SimulationTimeout are integrator failures."""
    import bngsim

    raised = getattr(bngsim, error, None) or ValueError

    def fail(self, *args, **kwargs):
        raise raised('not a solver failure')

    integrator = SegmentIntegrator(_model(nets['conversion'], _action()), ['k'])
    state = integrator.initial_state([0.7])
    monkeypatch.setattr(bngsim.Simulator, 'run', fail)
    with pytest.raises(raised, match='not a solver failure') as caught:
        integrator.integrate(state, [0.7], 0.0, 1.0)
    assert not isinstance(caught.value, SegmentFailed)


@bngsim_only
def test_segments_follow_execute_with_codegen_switched_off(nets, monkeypatch, tmp_path):
    monkeypatch.setenv('PYBNF_NO_CODEGEN', '1')
    names, values = ['beta0', 'gamma'], [1.3, 0.35]
    model = _model(nets['seasonal_sir'], _action(30, 30, extra=_tolerances(1e-8, 1e-8)))
    integrator = SegmentIntegrator(model, names)
    assert integrator._model._codegen_kwargs('ode', sensitivities=False) == {'codegen': False}
    ours = integrator.integrate(integrator.initial_state(values), values, 0.0, 30.0,
                                times=np.linspace(0.0, 30.0, 31)[1:-1])
    np.testing.assert_array_equal(_rows(ours.data), _rows(_execute(model, names, values, tmp_path)))


# ---------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------

@bngsim_only
@pytest.mark.parametrize('call, match', [
    (lambda it: it.integrate([5.0], [0.7], 0.0, 1.0), 'one value per species'),
    (lambda it: it.integrate([np.nan, 0.0], [0.7], 0.0, 1.0), r'finite .* A\(\) = nan'),
    (lambda it: it.integrate([5.0, 0.0], [0.7], 0.0, 1.0, times=[0.5, 0.2]), 'increase strictly'),
    (lambda it: it.integrate([5.0, 0.0], [0.7], 0.0, float('inf')), 'times must be finite'),
    (lambda it: it.integrate([5.0, 0.0], [0.7], 0.0, 1.0, times=[[0.5]]), 'one-dimensional'),
], ids=['state-length', 'state-nan', 'unordered-times', 'infinite-time', 'times-shape'])
def test_a_malformed_call_is_refused_before_anything_is_integrated(nets, call, match):
    """The caller's error, not a failed segment a filter would weigh as zero."""
    integrator = SegmentIntegrator(_model(nets['conversion'], _action()), ['k'])
    with pytest.raises(ValueError, match=match):
        call(integrator)
    assert integrator.n_integrations == 0


@bngsim_only
def test_a_call_while_another_runs_is_refused(nets):
    integrator = SegmentIntegrator(_model(nets['conversion'], _action()), ['k'])
    with integrator._busy:
        for call in (lambda: integrator.integrate([5.0, 0.0], [0.7], 0.0, 1.0),
                     lambda: integrator.initial_state([0.7])):
            with pytest.raises(RuntimeError, match='already running a call in another thread'):
                call()
    assert integrator.n_integrations == 0


@pytest.mark.parametrize('cls', [BNGLModel, BngsimNfModel])
def test_refuses_a_model_that_is_not_a_bngsim_net_model(cls):
    """Runs without bngsim: the refusal reads only what the model is."""
    model = object.__new__(cls)
    model.name = 'm'
    with pytest.raises(PybnfError, match=r'Model m \(%s\) is not a bngsim .net model' % cls.__name__):
        SegmentIntegrator(model, [])


@bngsim_only
@pytest.mark.parametrize('method', ['ssa', 'nf'])
def test_refuses_an_experiment_that_is_not_an_ode_integration(nets, method):
    action = 'simulate({method=>"%s",t_end=>10,n_steps=>10,suffix=>"tc"})' % method
    with pytest.raises(PybnfError, match="method '%s'.*only method=>\"ode\"" % method):
        SegmentIntegrator(_model(nets['conversion'], action), ['k'])


@bngsim_only
@pytest.mark.parametrize('field, text', [
    ('continue', ',continue=>1'), ('stop_if', ',stop_if=>"Atot<1"'),
    ('steady_state', ',steady_state=>1')])
def test_refuses_an_action_field_a_segment_does_not_apply(nets, field, text):
    with pytest.raises(PybnfError, match='sets %s, which a segment does not apply' % field):
        SegmentIntegrator(_model(nets['conversion'], _action(extra=text)), ['k'])


@bngsim_only
def test_accepts_an_action_field_the_ordinary_path_does_not_apply(nets, tmp_path):
    """The ordinary path reads continue as bool(int(value)), so continue=>0 is ordinary."""
    names, values = ['k', 'A0'], [0.4, 6.0]
    model = _model(nets['conversion'], _action(extra=',continue=>0'))
    integrator = SegmentIntegrator(model, names)
    segment = integrator.integrate(integrator.initial_state(values), values, 0.0, 10.0,
                                   times=np.linspace(0.0, 10.0, 11)[1:-1])
    np.testing.assert_array_equal(_rows(segment.data), _rows(_execute(model, names, values,
                                                                      tmp_path)))


@bngsim_only
@pytest.mark.parametrize('extra, match', [
    (',atol=>1e-8*oops', r'sets atol=>1e-8\*oops, which does not evaluate'),
    (',continue=>oops', 'sets continue=>oops, which does not evaluate'),
], ids=['tolerance', 'field'])
def test_refuses_a_tolerance_or_field_that_does_not_evaluate(nets, extra, match):
    with pytest.raises(PybnfError, match=match):
        SegmentIntegrator(_model(nets['conversion'], _action(extra=extra)), ['k'])


@bngsim_only
@pytest.mark.parametrize('actions, suffix, match', [
    (['setConcentration("A()", 1)'], None, 'has no simulate'),
    ([_action(suffix='a'), _action(suffix='b')], None, 'has 2 simulate.*a, b'),
    ([_action(suffix='a')], 'b', "no simulate.*suffix 'b'"),
    ([_action(suffix='a'), 'resetConcentrations()', _action(suffix='a')], 'a',
     "more than one simulate.*suffix 'a'"),
], ids=['none', 'several', 'unknown-suffix', 'repeated-suffix'])
def test_refuses_a_simulate_action_it_cannot_identify(nets, actions, suffix, match):
    with pytest.raises(PybnfError, match=match):
        SegmentIntegrator(_model(nets['conversion'], actions), ['k'], suffix=suffix)


@bngsim_only
@pytest.mark.parametrize('actions, moved', [
    (['setParameter("k", 2)', _action(suffix='second')], 'setParameter'),
    ([_action(suffix='first'), _action(suffix='second')], 'suffix=>"first"'),
], ids=['set-parameter', 'earlier-simulate'])
def test_refuses_an_experiment_that_does_not_start_from_the_seed_state(nets, actions, moved):
    with pytest.raises(PybnfError, match="does not start from the model's seed state") as refused:
        SegmentIntegrator(_model(nets['conversion'], actions), ['k'], suffix='second')
    assert moved in str(refused.value)


@bngsim_only
def test_refuses_a_model_with_conditions(nets):
    with pytest.raises(PybnfError, match=r'has conditions \(_dose\)'):
        SegmentIntegrator(_model(nets['conversion'], _action(), mutants=[MutationSet([], '_dose')]),
                          ['k'])


@bngsim_only
@pytest.mark.parametrize('names, match', [
    (['kk'], "has no parameter 'kk'.*primary parameters are gamma, k0__FREE, C0"),
    (['k0'], "'k0' is defined by an expression"),
    (['f'], "'f' holds a function's value"),
], ids=['unknown', 'derived', 'function'])
def test_refuses_a_parameter_it_cannot_write(nets, names, match):
    with pytest.raises(PybnfError, match=match):
        SegmentIntegrator(_model(nets['seeded'], _action()), names)
