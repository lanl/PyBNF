"""Second independent review of lanl/PyBNF#1005 (fix round 2a25a39c): the load-time check that
every datum has a row (#872 review), and the state after a ``stop_if`` stop (#1010).

The oracle is BNG2.pl's own output for the same actions (``.gdat`` rows, ``.scan`` rows), and the
closed form of the models. Tests marked ``xfail(strict=True)`` record a finding of the review:
they state what should happen, and fail until it does.
"""

import numpy as np
import pytest

from pybnf import config, pset
from pybnf.data import Data
from pybnf.printing import PybnfError

from .test_bngsim_output_grid import FIXTURES, _bng2pl, _bngsim_net, _model_block, _times

pytestmark = pytest.mark.bngsim

K = 0.3


def _decay_free():
    return _model_block(FIXTURES / 'e2e_ode_decay.bngl').replace('k   0.3', 'k   k__FREE')


# ---------------------------------------------------------------------------------------------
# The data-without-rows refusal and actions that share a suffix
# ---------------------------------------------------------------------------------------------

# Each: (id, actions that write suffix "x", the data times for "x"). In BNG2.pl a continue=>1
# run appends its rows to the output of the suffix it shares (run_network -x), and a later run
# that does not continue writes that output afresh; either way every datum below has a row.
_SHARED = [
    ('continue_appends',
     ['simulate({method=>"ode",sample_times=>[1,2,5],suffix=>"x"})',
      'simulate({method=>"ode",continue=>1,t_end=>10,n_steps=>5,suffix=>"x"})'], [6, 8, 10]),
    ('later_step_grid_writes_afresh',
     ['simulate({method=>"ode",sample_times=>[1,2,5],suffix=>"x"})',
      'simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"x"})'], [4, 8]),
    ('later_list_writes_afresh',
     ['simulate({method=>"ode",sample_times=>[1,2,5],suffix=>"x"})',
      'simulate({method=>"ode",sample_times=>[3,6,9],suffix=>"x"})'], [3, 6, 9]),
    ('continued_list_appends',
     ['simulate({method=>"ode",sample_times=>[1,2,5],suffix=>"x"})',
      'simulate({method=>"ode",continue=>1,t_start=>5,sample_times=>[6,8,10],suffix=>"x"})'],
     [8, 10]),
]


@pytest.mark.parametrize('actions, times', [s[1:] for s in _SHARED], ids=[s[0] for s in _SHARED])
def test_data_written_by_an_action_sharing_the_suffix_loads(tmp_path, monkeypatch, actions,
                                                            times):
    """BNG2.pl writes a row at every datum (the oracle, checked first), and so does the bridge;
    the configuration must load."""
    code, log = _bng2pl(tmp_path, 'decay', _model_block(FIXTURES / 'e2e_ode_decay.bngl'),
                        ['generate_network({overwrite=>1})'] + actions)
    assert code == 0, log
    bng = np.loadtxt(tmp_path / 'decay_x.gdat', ndmin=2)[:, 0]
    assert all(np.any(np.isclose(bng, t, atol=0.)) for t in times), bng
    model = tmp_path / 'm.bngl'
    model.write_text(_decay_free() + 'begin actions\n  generate_network({overwrite=>1})\n'
                     + ''.join('  %s\n' % a for a in actions) + 'end actions\n')
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n' + ''.join('%g %.6f\n' % (t, 100 * np.exp(-K * t))
                                            for t in times))
    monkeypatch.chdir(tmp_path)
    config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'wall_time_sim': 60,
        ('uniform_var', 'k__FREE'): [0.1, 1.0]})


def test_data_only_the_overwritten_action_wrote_is_refused(tmp_path, monkeypatch):
    """The later of two actions sharing a suffix keeps the output (BNG2.pl writes it afresh), so
    data on the earlier action's rows has no row on either backend and is refused."""
    model = tmp_path / 'm.bngl'
    model.write_text(_decay_free() + 'begin actions\n  generate_network({overwrite=>1})\n'
                     '  simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"x"})\n'
                     '  simulate({method=>"ode",sample_times=>[3,6,9],suffix=>"x"})\n'
                     'end actions\n')
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n4 30.119421\n8 9.071795\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError, match='has t = 4, 8'):
        config.Configuration({
            'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
            'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
            'fit_type': 'de', 'objfunc': 'sos', 'wall_time_sim': 60,
            ('uniform_var', 'k__FREE'): [0.1, 1.0]})


def test_data_after_a_shortened_grid_is_still_refused(tmp_path, monkeypatch):
    """The refusal the check exists for: rows 0, 5, 7 (t_end below the last listed time), a
    datum at 10, and no other action writing "x". It names the time and the action's rows."""
    model = tmp_path / 'm.bngl'
    model.write_text(_decay_free() + 'begin actions\n  generate_network({overwrite=>1})\n'
                     '  simulate({method=>"ode",t_end=>7,sample_times=>[0,5,10],suffix=>"x"})\n'
                     'end actions\n')
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n5 22.313016\n10 4.978707\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        config.Configuration({
            'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
            'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
            'fit_type': 'de', 'objfunc': 'sos', 'wall_time_sim': 60,
            ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    assert info.value.log_message == ('Model m, simulate action with suffix "x": the data for "x" '
                                      'has t = 10, where the action writes no row')
    assert 'writes rows at t = 0, 5, 7' in info.value.message


# ---------------------------------------------------------------------------------------------
# #1010: the state after a stop
# ---------------------------------------------------------------------------------------------

#: A sets off the stop; C is in no observable and makes D, which is. So D after the stop shows
#: whether every species, the unobserved C included, was put back at the stop row.
HIDDEN = """begin model
begin parameters
  A0 1000
  kdeg 0.1
  kc 0.3
  kd 1
end parameters
begin molecule types
  A()
  C()
  D()
end molecule types
begin seed species
  A() A0
  C() 100
  D() 0
end seed species
begin observables
  Molecules Atot A()
  Molecules Dtot D()
end observables
begin reaction rules
  A() -> 0 kdeg
  C() -> 0 kc
  C() -> C() + D() kd
end reaction rules
end model
"""
STOP = 'simulate({method=>"ode",t_end=>20,n_steps=>20,stop_if=>"Atot<500",suffix=>"eq"})'


def _hidden_exact(t):
    """A, and D from C(0) = 100: D(t) = kd C0 (1 - exp(-kc t)) / kc, at kdeg 0.1, kc 0.3, kd 1."""
    t = np.asarray(t, dtype=float)
    return 1000 * np.exp(-0.1 * t), 100 * (1 - np.exp(-0.3 * t)) / 0.3


def test_an_unobserved_species_is_put_back_at_the_stop_row(tmp_path):
    """After a stop at t = 7, C (in no observable) must be C(7), not C(20): the following run's D
    is D(7 + t). Oracle: BNG2.pl's rows and the closed form."""
    after = 'simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"x"})'
    code, log = _bng2pl(tmp_path, 'm', HIDDEN, ['generate_network({overwrite=>1})', STOP, after])
    assert code == 0, log
    bng = np.loadtxt(tmp_path / 'm_x.gdat', ndmin=2)
    a, d = _hidden_exact(bng[:, 0] + 7)
    np.testing.assert_allclose(bng[:, 1], a, rtol=1e-6)
    np.testing.assert_allclose(bng[:, 2], d, rtol=1e-6)
    out = _bngsim_net(tmp_path / 'm.net', [STOP, after], ['eq', 'x']).execute(
        str(tmp_path / 'run'), 'm_bngsim', 60)['x']
    np.testing.assert_allclose(_times(out), bng[:, 0], atol=1e-12)
    np.testing.assert_allclose(out['Atot'], bng[:, 1], rtol=1e-6)
    np.testing.assert_allclose(out['Dtot'], bng[:, 2], rtol=1e-6)


def test_a_scan_point_stops_on_its_stop_if_as_in_bng2pl(tmp_path):
    """BNG2.pl runs simulate at each point with the scan's options, stop_if included, and reads
    the point's row from the last row written: A first below 500 at t = 7 (kdeg 0.1) and t = 4
    (kdeg 0.2). The bridge reads A(20). Oracle: BNG2.pl's .scan and the closed form."""
    scan = ('parameter_scan({parameter=>"kdeg",par_scan_vals=>[0.1,0.2],method=>"ode",t_end=>20,'
            'n_steps=>20,stop_if=>"Atot<500",suffix=>"s"})')
    code, log = _bng2pl(tmp_path, 'm', HIDDEN, ['generate_network({overwrite=>1})', scan])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'm_s.scan'))
    np.testing.assert_allclose(bng['Atot'], 1000 * np.exp(-np.array([0.7, 0.8])), rtol=1e-6)
    import pybnf.bngsim_model as bngsim_model
    model = bngsim_model.BngsimModel('m', [scan], [('parameter_scan', 's')], [],
                                     nf=str(tmp_path / 'm.net'))
    model.param_set = pset.PSet([])
    out = model.execute(str(tmp_path / 'run'), 'm_bngsim', 60)['s']
    np.testing.assert_allclose(out['Atot'], bng['Atot'], rtol=1e-6)


@pytest.mark.parametrize('between', [
    ['saveConcentrations()', 'resetConcentrations()'],
    ['saveConcentrations("s")', 'resetConcentrations("s")'],
], ids=['unlabelled_snapshot', 'labelled_snapshot'])
def test_a_snapshot_of_a_stopped_state_is_not_carried_with_stale_derivatives(tmp_path, between):
    """The gradient refusal clears its flag on resetConcentrations(); restoring a snapshot taken
    at the stop row still carries the stopped state, whose dx/dtheta bngsim dropped. That must
    stop with an error, never yield a derivative seeded from nothing."""
    import pybnf.bngsim_model as bngsim_model
    code, log = _bng2pl(tmp_path, 'm', HIDDEN, ['generate_network({overwrite=>1})', STOP])
    assert code == 0, log
    after = 'simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"x"})'
    m = bngsim_model.BngsimModel('m', [STOP] + between + [after],
                                 [('simulate', 'eq'), ('simulate', 'x')], [],
                                 nf=str(tmp_path / 'm.net'))
    m.param_set = pset.PSet([])
    m.enable_output_sensitivities(params=['kdeg'])
    with pytest.raises(PybnfError):
        m.execute(str(tmp_path / 'run'), 'g', 60)
