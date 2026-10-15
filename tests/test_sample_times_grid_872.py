"""lanl/PyBNF#872: on the bngsim bridge, an action's ``sample_times`` grid is BNG2.pl's.

The bridge used to integrate an action with ``sample_times`` from its first listed time, so the
initial state was reported there and every later row was shifted, and it ignored a ``t_end``
below the last listed time. BNG2.pl's ``run_network`` starts at ``t_start`` and writes that row,
then each listed time after ``t_start`` and before the end, then the end: ``t_end`` when the
action gives one, else the last listed time (``pybnf/bngsim_model/grid.py``).

The oracle is BNG2.pl's own output for the same action, and the closed form of the fixture
(``S' = -k S``, ``S(t_start) = 100``, ``k = 0.3``, so ``S(t) = 100 exp(-k (t - t_start))``). For
network-free actions, which BNG2.pl refuses with ``sample_times``, the oracle is the mean-field
solution of the binding fixture and the run from an explicit ``t = 0``. Skipped where BNG2.pl or
bngsim is unavailable.
"""

import numpy as np
import pytest

from pybnf.data import Data
from pybnf.printing import PybnfError

from . import recovery_harness as R
from .test_bngsim_output_grid import (
    FIXTURES,
    _bng2pl,
    _bngsim_net,
    _bngsim_nf,
    _model_block,
    _times,
)

pytestmark = pytest.mark.bngsim

K = 0.3


def _decay_block():
    return _model_block(FIXTURES / 'e2e_ode_decay.bngl')


def _closed_form(t, t_start=0.0):
    return 100.0 * np.exp(-K * (np.asarray(t, dtype=float) - t_start))


def _run_bng2pl(tmp_path, actions, out):
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})'] + list(actions))
    assert code == 0, log
    return np.loadtxt(tmp_path / out, ndmin=2)


def _run_bridge(tmp_path, actions, suffixes, wanted):
    return _bngsim_net(tmp_path / 'decay.net', actions, suffixes).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)[wanted]


# Each shape: (id, options, BNG2.pl's row times, t_start).
_ODE_SHAPES = [
    ('late_first_time', 'sample_times=>[2,5,10]', [0, 2, 5, 10], 0),
    ('late_first_time_explicit_t_start', 't_start=>0,sample_times=>[10,20,30]', [0, 10, 20, 30], 0),
    ('t_end_below_last', 't_start=>0,t_end=>7,sample_times=>[0,5,10]', [0, 5, 7], 0),
    ('t_end_below_last_late', 't_end=>7,sample_times=>[2,5,10]', [0, 2, 5, 7], 0),
    ('t_end_above_last', 't_start=>0,t_end=>15,sample_times=>[0,5,10]', [0, 5, 10, 15], 0),
    ('t_end_at_last', 't_end=>10,sample_times=>[0,5,10]', [0, 5, 10], 0),
    ('t_start_listed', 't_start=>3,sample_times=>[0,3,5,10]', [3, 5, 10], 3),
    ('t_start_between', 't_start=>5,sample_times=>[2,5,10]', [5, 10], 5),
    ('time_listed_twice', 'sample_times=>[0,5,5,10]', [0, 5, 5, 10], 0),
    ('unsorted', 'sample_times=>[10,2,5]', [0, 2, 5, 10], 0),
]


@pytest.mark.parametrize('options, rows, t_start', [s[1:] for s in _ODE_SHAPES],
                         ids=[s[0] for s in _ODE_SHAPES])
def test_an_ode_sample_times_action_writes_bng2pls_rows(tmp_path, options, rows, t_start):
    """The run starts at t_start and writes it; a listed time at or before t_start, or at or
    after t_end, is dropped; t_end, when given, is the last row, below or above the last listed
    time; a time listed twice is written twice."""
    action = 'simulate({method=>"ode",%s,suffix=>"x"})' % options
    bng = _run_bng2pl(tmp_path, [action], 'decay_x.gdat')
    np.testing.assert_allclose(bng[:, 0], rows, atol=1e-9)      # the oracle's rows, as expected
    np.testing.assert_allclose(bng[:, 1], _closed_form(bng[:, 0], t_start), rtol=1e-5, atol=1e-6)
    out = _run_bridge(tmp_path, [action], ['x'], 'x')
    np.testing.assert_allclose(_times(out), bng[:, 0], atol=1e-9)
    np.testing.assert_allclose(out['Stot'], bng[:, 1], rtol=1e-5, atol=1e-6)


def test_an_ssa_sample_times_action_starts_at_t_start(tmp_path):
    """A stochastic network method takes the same rows, and its first row is the initial state
    at t = 0, not at the first listed time (S0 = 100 molecules)."""
    action = 'simulate({method=>"ssa",sample_times=>[2,5,10],seed=>5,suffix=>"x"})'
    bng = _run_bng2pl(tmp_path, [action], 'decay_x.gdat')
    np.testing.assert_allclose(bng[:, 0], [0, 2, 5, 10], atol=1e-9)
    out = _run_bridge(tmp_path, [action], ['x'], 'x')
    np.testing.assert_allclose(_times(out), bng[:, 0], atol=1e-9)
    assert out['Stot'][0] == 100.0
    assert out['Stot'][1] < 100.0


# Each: (id, the two actions, BNG2.pl's rows of the continued one).
_CONTINUED = [
    ('after_late_sample_times',
     ['simulate({method=>"ode",sample_times=>[1,2,5],suffix=>"a"})',
      'simulate({continue=>1,method=>"ode",t_end=>20,n_steps=>2,suffix=>"x"})'], [12.5, 20]),
    ('after_t_end_below_last',
     ['simulate({method=>"ode",t_end=>7,sample_times=>[0,5,10],suffix=>"a"})',
      'simulate({continue=>1,method=>"ode",t_end=>20,n_steps=>2,suffix=>"x"})'], [13.5, 20]),
    ('sample_times_continued',
     ['simulate({method=>"ode",t_end=>4,n_steps=>2,suffix=>"a"})',
      'simulate({continue=>1,method=>"ode",sample_times=>[2,5,10],suffix=>"x"})'], [5, 10]),
]


@pytest.mark.parametrize('actions, rows', [c[1:] for c in _CONTINUED],
                         ids=[c[0] for c in _CONTINUED])
def test_a_continued_action_starts_on_bng2pls_clock(tmp_path, actions, rows):
    """After a run, the model's clock is the time of its last row (5, or the t_end of 7), so
    ``continue=>1`` divides [5, 20] or [7, 20] into its steps, and a continued sample_times
    action starts at the previous end (4) and drops the listed time before it. The bridge used
    to leave its clock elsewhere. BNG2.pl writes no row at a continued run's start; the bridge
    writes it (scoring matches by time). The values continue one decay from t = 0."""
    bng = _run_bng2pl(tmp_path, actions, 'decay_x.gdat')
    np.testing.assert_allclose(bng[:, 0], rows, atol=1e-9)
    np.testing.assert_allclose(bng[:, 1], _closed_form(bng[:, 0]), rtol=1e-5)
    out = _run_bridge(tmp_path, actions, ['a', 'x'], 'x')
    np.testing.assert_allclose(_times(out)[1:], bng[:, 0], atol=1e-9)
    np.testing.assert_allclose(out['Stot'][1:], bng[:, 1], rtol=1e-5)
    np.testing.assert_allclose(out['Stot'][0], _closed_form(_times(out)[0]), rtol=1e-5)


# Each: (id, actions, BNG2.pl's .scan values by the closed form).
_SCANS = [
    ('late_first_time',
     ['parameter_scan({method=>"ode",parameter=>"k",par_scan_vals=>[0.1,0.3],'
      'sample_times=>[2,5,10],suffix=>"x"})'], 100.0 * np.exp(-np.array([0.1, 0.3]) * 10)),
    ('t_end_below_last',
     ['parameter_scan({method=>"ode",parameter=>"k",par_scan_vals=>[0.1,0.3],t_end=>7,'
      'sample_times=>[2,5,10],suffix=>"x"})'], 100.0 * np.exp(-np.array([0.1, 0.3]) * 7)),
    # A scan after a simulate starts each point from the state the scan found (the carried-state
    # scan): 100 exp(-0.3) at t = 1, then 10 time units at each k.
    ('after_a_simulate',
     ['simulate({method=>"ode",t_end=>1,n_steps=>1,suffix=>"a"})',
      'parameter_scan({method=>"ode",parameter=>"k",par_scan_vals=>[0.1,0.3],'
      'sample_times=>[2,5,10],suffix=>"x"})'],
     100.0 * np.exp(-0.3) * np.exp(-np.array([0.1, 0.3]) * 10)),
]


@pytest.mark.parametrize('actions, expected', [s[1:] for s in _SCANS], ids=[s[0] for s in _SCANS])
def test_a_scan_point_with_sample_times_ends_where_bng2pls_does(tmp_path, actions, expected):
    """A scan point's row is its final state: from t_start to t_end, or to the last listed time
    without one. The bridge used to integrate from the first listed time (8 units, not 10) and to
    the last listed time whatever the t_end."""
    bng = Data(file_name=str(_scan_file(tmp_path, actions)))
    np.testing.assert_allclose(bng['Stot'], expected, rtol=1e-5)
    suffixes = ['a', 'x'] if len(actions) == 2 else ['x']
    out = _run_bridge(tmp_path, actions, suffixes, 'x')
    np.testing.assert_allclose(out['k'], [0.1, 0.3])
    np.testing.assert_allclose(out['Stot'], bng['Stot'], rtol=1e-5)


@pytest.mark.parametrize('options, t_end', [('sample_times=>[2,5,10]', 10.0),
                                            ('t_end=>7,sample_times=>[2,5,10]', 7.0)],
                         ids=['late_first_time', 't_end_below_last'])
def test_a_protocol_scan_runs_its_sample_times_from_t_start(tmp_path, options, t_end):
    """A protocol block's simulate takes the same rows: each point of a ``method=>"protocol"``
    scan ends at the protocol grid's end, reached from t_start. Oracle: BNG2.pl's .scan and the
    closed form 100 exp(-k t_end)."""
    protocol = ['simulate({method=>"ode",%s})' % options]
    scan = 'parameter_scan({method=>"protocol",parameter=>"k",par_scan_vals=>[0.1,0.3],suffix=>"x"})'
    code, log = _bng2pl(tmp_path, 'decay', _decay_block() + 'begin protocol\n  %s\nend protocol\n'
                        % protocol[0], ['generate_network({overwrite=>1})', scan])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'decay_x.scan'))
    np.testing.assert_allclose(bng['Stot'], 100.0 * np.exp(-np.array([0.1, 0.3]) * t_end),
                               rtol=1e-5)
    import pybnf.bngsim_model as bngsim_model
    from pybnf import pset
    model = bngsim_model.BngsimModel('decay', [scan], [('parameter_scan', 'x')], [],
                                     nf=str(tmp_path / 'decay.net'), protocol=protocol)
    model.param_set = pset.PSet([])
    out = model.execute(str(tmp_path / 'run'), 'decay_bngsim', 60)['x']
    np.testing.assert_allclose(out['Stot'], bng['Stot'], rtol=1e-5)


def test_the_gradient_reads_the_same_rows(tmp_path):
    """The gradient path runs the same grid: its sensitivities are on BNG2.pl's rows, from
    t_start. Oracle: dS/dk = -t 100 exp(-0.3 t). Run from the first listed time, as the bridge
    ran it, each row held the sensitivity of a run 2 time units shorter, and the run went on
    past t_end = 7 to 10."""
    import pybnf.bngsim_model as bngsim_model
    from pybnf import pset
    action = 'simulate({method=>"ode",t_end=>7,sample_times=>[2,5,10],suffix=>"x"})'
    model = bngsim_model.BngsimModel('e2e_ode_decay', [action], [('simulate', 'x')], [],
                                     nf=str(FIXTURES / 'e2e_ode_decay.net'))
    model.param_set = pset.PSet([])
    model.enable_output_sensitivities(params=['k'])
    out = model.execute(str(tmp_path / 'run'), 'grad', 60)['x']
    t = _times(out)
    np.testing.assert_allclose(t, [0, 2, 5, 7], atol=1e-9)
    got = out.output_sensitivities.slice_for('observable:Stot')[:, 0]
    np.testing.assert_allclose(got, -t * _closed_form(t), rtol=1e-4, atol=1e-6)


def _scan_file(tmp_path, actions):
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})'] + list(actions))
    assert code == 0, log
    return tmp_path / 'decay_x.scan'


# ---------------------------------------------------------------------------------------------
# Network-free: BNG2.pl's NFsim refuses sample_times; bngsim's runs them on the same rows
# ---------------------------------------------------------------------------------------------

def _nf_run(folder, action, replicate=0):
    """One run of the binding fixture's NF action. The seed is PyBNF's (``stochastic_seed =
    auto``, which overrides a BNGL ``seed=>``), derived from the action's place and suffix and
    from ``replicate``: two actions in the same place get the same seed."""
    model = _bngsim_nf(action, 'tc')
    model._pybnf_replicate_index = replicate
    return model.execute(str(folder), 'nf', 60)['tc']


def _cme_bound(k_on, t, n0=100):
    """The exact mean and standard deviation of ``bound`` at ``t`` for the binding fixture,
    A(b) + B(a) -> A(b!1).B(a!1) at k_on per pair from 100 of each, from its chemical master
    equation: a pure death chain in the free count n, at rate k_on n^2."""
    from scipy.linalg import expm
    n = np.arange(n0 + 1)
    rate = k_on * n.astype(float) ** 2
    generator = np.diag(-rate) + np.diag(rate[1:], 1)
    start = np.zeros(n0 + 1)
    start[n0] = 1.0
    p = expm(generator * t) @ start
    bound = n0 - n
    mean = float(bound @ p)
    return mean, float(np.sqrt(bound ** 2 @ p - mean ** 2))


def test_an_nf_sample_times_action_starts_at_t_start(tmp_path):
    """The rows a network method's would have, the run from t_start included: with the same
    seed, a list that starts after t = 0 gives the same trajectory as one that lists t = 0, and
    its first row is the initial state, at t = 0. A t_end before the last listed time ends it."""
    late = _nf_run(tmp_path / 'late', 'simulate({method=>"nf",sample_times=>[2,5,10],gml=>1000,'
                   'suffix=>"tc"})')
    from_zero = _nf_run(tmp_path / 'zero', 'simulate({method=>"nf",sample_times=>[0,2,5,10],'
                        'gml=>1000,suffix=>"tc"})')
    np.testing.assert_allclose(_times(late), [0, 2, 5, 10], atol=1e-9)
    np.testing.assert_array_equal(late.data, from_zero.data)
    assert late['bound'][0] == 0.0 and late['Afree'][0] == 100.0
    ended = _nf_run(tmp_path / 'ended', 'simulate({method=>"nf",t_end=>7,sample_times=>[0,5,10],'
                    'gml=>1000,suffix=>"tc"})')
    np.testing.assert_allclose(_times(ended), [0, 5, 7], atol=1e-9)


def test_nf_sample_times_reach_the_exact_mean_from_t_zero(tmp_path):
    """Oracle: the chemical master equation of the binding fixture (:func:`_cme_bound`). The
    mean of bound at the last row, t = 10, over 60 replicates of a list starting at t = 2 must
    be the exact E[bound(10)] = 50.1 within three standard errors. Integrated from t = 2, as the
    bridge did, it is E[bound(8)] = 44.5, eleven standard errors away."""
    values = [_nf_run(tmp_path / str(i), 'simulate({method=>"nf",sample_times=>[2,5,10],'
                      'gml=>1000,suffix=>"tc"})', replicate=i)['bound'][-1] for i in range(60)]
    mean, sd = _cme_bound(0.001, 10.0)
    assert abs(np.mean(values) - mean) < 3 * sd / np.sqrt(len(values)), (np.mean(values), mean)


def test_an_nf_scan_with_sample_times_ends_each_point_from_t_start(tmp_path):
    """The network-free scan: each point's final state is reached from t_start. With the same
    seeds a list starting at t = 2 gives the points one that lists t = 0 does, and over 40
    replicates each point's mean is the exact E[bound(10)] at its k_on (:func:`_cme_bound`)."""
    scan = ('parameter_scan({method=>"nf",parameter=>"k_on",par_scan_vals=>[0.001,0.002],'
            'sample_times=>%s,gml=>1000,suffix=>"tc"})')
    np.testing.assert_array_equal(_nf_run(tmp_path / 'late', scan % '[2,5,10]').data,
                                  _nf_run(tmp_path / 'zero', scan % '[0,2,5,10]').data)
    runs = [_nf_run(tmp_path / str(i), scan % '[2,5,10]', replicate=i) for i in range(40)]
    for point, k_on in enumerate((0.001, 0.002)):
        values = [run['bound'][point] for run in runs]
        assert all(run['k_on'][point] == k_on for run in runs)
        mean, sd = _cme_bound(k_on, 10.0)
        assert abs(np.mean(values) - mean) < 3 * sd / np.sqrt(len(values)), (k_on, np.mean(values),
                                                                             mean)


def test_a_continued_nf_action_is_refused_as_bng2pl_refuses_it(tmp_path):
    """The network-free bridge has no clock for ``continue=>1`` to start on; it ignored the
    option and labelled the continued rows from t_start. BNG2.pl refuses it with this message
    (test_load_refusals_996_997_999 runs BNG2.pl on it). ``continue=>0`` is no continuation."""
    from pybnf.bngsim_model.grid import check_output_grids
    first = 'simulate({method=>"nf",t_end=>5,n_steps=>5,gml=>1000,suffix=>"a"})'
    action = ('simulate({method=>"nf",continue=>1,sample_times=>[2,5,10],gml=>1000,'
              'suffix=>"tc"})')
    message = ('Model e2e_nf_binding, simulate action with suffix "tc": '
               "NFsim does not support 'continue' option.")
    with pytest.raises(PybnfError) as info:
        check_output_grids('e2e_nf_binding', [first, action])
    assert info.value.log_message == message
    with pytest.raises(PybnfError) as info:
        _nf_run(tmp_path / 'continued', action)
    assert info.value.log_message == message
    out = _nf_run(tmp_path / 'not', action.replace('continue=>1', 'continue=>0'))
    np.testing.assert_allclose(_times(out), [0, 2, 5, 10], atol=1e-9)


# ---------------------------------------------------------------------------------------------
# Grids the bridge refuses
# ---------------------------------------------------------------------------------------------

def test_a_negative_t_start_with_sample_times_is_refused_as_bng2pl_stops(tmp_path):
    """run_network stops on a sample_times grid that starts before 0. The bridge ran it."""
    from pybnf.bngsim_model.grid import check_output_grids
    action = 'simulate({method=>"ode",t_start=>-1,sample_times=>[2,5,10],suffix=>"x"})'
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})', action])
    assert code != 0 and 'Negative sample times are not allowed.' in log, log
    message = 'Model decay, simulate action with suffix "x": Negative sample times are not allowed.'
    with pytest.raises(PybnfError) as info:
        check_output_grids('decay', [action])
    assert info.value.log_message == message
    with pytest.raises(PybnfError) as info:
        _bngsim_net(FIXTURES / 'e2e_ode_decay.net', [action], ['x']).execute(
            str(tmp_path / 'run'), 'decay_bngsim', 60)
    assert info.value.log_message.endswith('Negative sample times are not allowed.')


def test_a_grid_that_ends_where_it_starts_is_refused(tmp_path):
    """With every listed time at or before t_start, BNG2.pl integrates nothing and writes the
    one row at t_start; the bridge's runs need a later time. Refused when the models are set
    up, and when the action runs."""
    from pybnf.bngsim_model.grid import check_output_grids
    action = 'simulate({method=>"ode",t_start=>12,sample_times=>[2,5,10],suffix=>"x"})'
    bng = _run_bng2pl(tmp_path, [action], 'decay_x.gdat')
    np.testing.assert_allclose(bng, [[12.0, 100.0]])
    message = ('Model decay, simulate action with suffix "x": its sample_times grid ends at '
               't = 10, which is not after t_start = 12')
    with pytest.raises(PybnfError) as info:
        check_output_grids('decay', [action])
    assert info.value.log_message == message
    with pytest.raises(PybnfError) as info:
        _bngsim_net(FIXTURES / 'e2e_ode_decay.net', [action], ['x']).execute(
            str(tmp_path / 'run'), 'decay_bngsim', 60)
    assert info.value.log_message == message.replace('decay', 'e2e_ode_decay', 1)


def test_a_continued_grid_that_ends_before_the_clock_is_refused_when_it_runs(tmp_path):
    """``continue=>1`` takes t_start from the clock, which is known only when the action runs:
    after a run to 10, a continued list ending at 8 has nothing to integrate. BNG2.pl writes no
    row for it, and then stops on the empty output."""
    from pybnf.bngsim_model.grid import check_output_grids
    actions = ['simulate({method=>"ode",sample_times=>[0,5,10],suffix=>"a"})',
               'simulate({continue=>1,method=>"ode",sample_times=>[2,5,8],suffix=>"x"})']
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})'] + actions)
    assert code != 0 and 'simulation will not run' in log, log
    check_output_grids('decay', actions)        # not known at set-up
    with pytest.raises(PybnfError) as info:
        _run_bridge(tmp_path, actions, ['a', 'x'], 'x')
    assert info.value.log_message.endswith('ends at t = 8, which is not after t_start = 10')
    assert 'continue=>1' in info.value.message


def test_edition2_data_before_t_zero_is_refused_when_the_configuration_loads(tmp_path):
    """An edition-2 time course starts at t = 0 (``t_start=>0``), so a datum at a negative time
    has no row on either backend: BNG2.pl's run_network drops it, and the bridge now does too.
    The bridge used to start the run at the negative time instead. Refused at load."""
    from pybnf import config, parse
    model = tmp_path / 'm.bngl'
    model.write_text(_decay_block())
    (tmp_path / 'tc.exp').write_text('# time Stot\n-2 182.2\n1 74.08\n5 22.31\n')
    lines = ['edition = 2', 'model: %s' % model, 'output_dir = %s' % (tmp_path / 'out'),
             'job_type = de', 'objective = sos', 'bngl_backend = bngsim',
             'experiment: tc, data: %s' % (tmp_path / 'tc.exp'), 'uniform_var = k 0.1 1',
             'population_size = 4', 'max_iterations = 1']
    with pytest.raises(PybnfError) as info:
        config.Configuration(parse.ploop([ln + '\n' for ln in lines]))
    assert info.value.log_message == ('The time_course action with suffix "tc" has data at '
                                      't = -2, before t = 0, where the simulation starts')


# ---------------------------------------------------------------------------------------------
# The failure scenario of the issue: a fit recovers the rate on the bridge as on BNG2.pl
# ---------------------------------------------------------------------------------------------

_DECAY_1000 = """begin model
begin parameters
  S0    1000
  kdeg  kdeg__FREE
end parameters
begin molecule types
  S()
end molecule types
begin seed species
  S()  S0
end seed species
begin observables
  Molecules Stot S()
end observables
begin reaction rules
  S() -> 0  kdeg
end reaction rules
end model
begin actions
  generate_network({overwrite=>1})
  simulate({method=>"ode",sample_times=>[10,20,30],suffix=>"tc"})
end actions
"""


@pytest.mark.parametrize('backend', ['auto', 'bionetgen'])
def test_a_de_fit_of_late_sample_times_recovers_the_rate(tmp_path, monkeypatch, backend):
    """lanl/PyBNF#872's own scenario: data 1000 exp(-0.1 t) at t = 10, 20 and 30, a model whose
    action lists those times. On the bridge the fit converged to kdeg = 0.199 with an objective
    of 4e5; BNG2.pl's run recovers 0.1. Both must now."""
    from pybnf.bngsim_model import BngsimModel
    R.require_bng2pl()
    R.install(monkeypatch, real_run_job=True)    # BNG2.pl's runs need the job folders
    model = tmp_path / 'decay.bngl'
    model.write_text(_DECAY_1000)
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time Stot\n' + ''.join('%g %.6f\n' % (t, 1000 * np.exp(-0.1 * t))
                                            for t in (10, 20, 30)))
    monkeypatch.chdir(tmp_path)
    conf = R.make_config(tmp_path, model, exp, {'kdeg__FREE': ('uniform_var', 0.01, 1.0)}, 'de',
                         population_size=8, max_iterations=12, random_seed=7, bngl_backend=backend,
                         wall_time_sim=120)
    alg = R.build(conf, 'de')
    assert isinstance(alg.model_list[0], BngsimModel) == (backend == 'auto')
    R.drive(alg)
    best = alg.trajectory.best_fit()
    assert abs(best['kdeg__FREE'] - 0.1) < 0.003, best['kdeg__FREE']
    assert alg.trajectory.best_score() < 1e3     # it was 4e5 on the bridge
