"""lanl/PyBNF#1010: after a ``simulate`` that stops early, the bngsim bridge leaves the model where
BNG2.pl leaves it.

BNG2.pl reads the model's species and clock back from the last row a run wrote (BNGAction.pm
757-779). run_network stops an ODE run at the first output time at which ``stop_if`` holds, and
a ``steady_state=>1`` run at the first output time it is steady. bngsim cuts its result at the
same row, but it checks ``stop_if`` only after integrating the whole span, so the bridge used to
carry the state at ``t_end`` into every later action, and its clock read ``t_end`` after either
kind of stop. A stochastic method checks ``stop_if`` after every reaction in run_network, so it
stops between output times; the bridge, which checks at output times, hands such an action to
BNG2.pl. BNG2.pl's NFsim ignores ``stop_if``, as the bridge's network-free path does.

The oracle is BNG2.pl's own output for the same actions, and the closed form of the model:
A() -> B() at kdeg = 0.1 from A = 1000, so A(t) = 1000 exp(-0.1 t) and B = 1000 - A.
"""

import numpy as np
import pytest

from pybnf.printing import PybnfError

from . import recovery_harness as R
from .test_bngsim_output_grid import _bng2pl, _bngsim_net, _times

pytestmark = pytest.mark.bngsim

MODEL = """begin model
begin parameters
  A0 1000
  kdeg 0.1
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() A0
  B() 0
end seed species
begin observables
  Molecules Atot A()
  Molecules Btot B()
end observables
begin reaction rules
  A() -> B() kdeg
end reaction rules
end model
"""

#: The issue's steady-state model: made at ksyn, lost at kdeg, so A -> ksyn/kdeg = 100.
STEADY = """begin model
begin parameters
  ksyn 10
  kdeg 0.1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  0 -> A() ksyn
  A() -> 0 kdeg
end reaction rules
end model
"""

STOP = 'simulate({method=>"ode",t_end=>20,n_steps=>20,stop_if=>"Atot<500",suffix=>"eq"})'


def _both(tmp_path, model_text, actions, suffixes):
    """BNG2.pl's ``.gdat`` rows of each suffix, and the bridge's output, for ``actions``."""
    code, log = _bng2pl(tmp_path, 'm', model_text,
                        ['generate_network({overwrite=>1})'] + list(actions))
    assert code == 0, log
    bng = {s: np.loadtxt(tmp_path / ('m_%s.gdat' % s), ndmin=2) for s in suffixes}
    out = _bngsim_net(tmp_path / 'm.net', actions, suffixes).execute(
        str(tmp_path / 'run'), 'm_bngsim', 60)
    return bng, out


def _exact(t):
    a = 1000 * np.exp(-0.1 * np.asarray(t, dtype=float))
    return a, 1000 - a


def test_the_stop_run_itself_writes_bng2pls_rows(tmp_path):
    bng, out = _both(tmp_path, MODEL, [STOP], ['eq'])
    np.testing.assert_allclose(bng['eq'][:, 0], np.arange(8), atol=1e-12)   # stops at t = 7
    np.testing.assert_allclose(_times(out['eq']), bng['eq'][:, 0], atol=1e-12)
    np.testing.assert_allclose(out['eq']['Atot'], bng['eq'][:, 1], rtol=1e-5)


# Each: (id, the action after the stop, BNG2.pl's rows of it, where its decay starts).
_AFTER = [
    ('simulate', 'simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"x"})',
     [0, 1, 2, 3, 4, 5], 7.0),
    ('continue_n_steps', 'simulate({method=>"ode",continue=>1,t_end=>30,n_steps=>5,suffix=>"x"})',
     [11.6, 16.2, 20.8, 25.4, 30], 0.0),
    ('continue_sample_times', 'simulate({method=>"ode",continue=>1,'
                              'sample_times=>[10,15,20,25,30],suffix=>"x"})',
     [10, 15, 20, 25, 30], 0.0),
]


@pytest.mark.parametrize('after, rows, offset', [a[1:] for a in _AFTER], ids=[a[0] for a in _AFTER])
def test_the_action_after_a_stop_starts_from_the_stop_row(tmp_path, after, rows, offset):
    """BNG2.pl continues from A(7), with every species (B too) at the stop row, and a continued
    action starts its clock at 7. The bridge used to continue from A(20) at t = 20: a plain
    simulate fitted the next phase to 1000 exp(-0.1 (20 + t)) instead of exp(-0.1 (7 + t))."""
    bng, out = _both(tmp_path, MODEL, [STOP, after], ['eq', 'x'])
    a, b = _exact(bng['x'][:, 0] + offset)
    np.testing.assert_allclose(bng['x'][:, 0], rows, atol=1e-9)
    np.testing.assert_allclose(bng['x'][:, 1], a, rtol=1e-5)
    np.testing.assert_allclose(bng['x'][:, 2], b, rtol=1e-5)
    x = out['x']
    got = slice(1, None) if 'continue' in after else slice(None)    # the bridge's start row
    np.testing.assert_allclose(_times(x)[got], bng['x'][:, 0], atol=1e-9)
    np.testing.assert_allclose(x['Atot'][got], bng['x'][:, 1], rtol=1e-5)
    np.testing.assert_allclose(x['Btot'][got], bng['x'][:, 2], rtol=1e-5)
    if 'continue' in after:
        assert _times(x)[0] == 7.0 and np.isclose(x['Atot'][0], _exact(7.0)[0], rtol=1e-6)


def test_the_clock_after_a_steady_state_stop_is_its_last_row(tmp_path):
    """The issue's model: both stop at t = 250 at steady state (A = 100), and BNG2.pl's
    continued run over [250, 1100] in 2 steps writes 675 and 1100. The bridge's clock read
    t_end = 1000, so it wrote 1000, 1050 and 1100."""
    actions = ['simulate({method=>"ode",t_end=>1000,n_steps=>100,steady_state=>1,suffix=>"eq"})',
               'simulate({method=>"ode",continue=>1,t_end=>1100,n_steps=>2,suffix=>"x"})']
    bng, out = _both(tmp_path, STEADY, actions, ['eq', 'x'])
    end = bng['eq'][-1, 0]
    assert end < 1000
    np.testing.assert_allclose(_times(out['eq'])[-1], end, atol=1e-9)
    np.testing.assert_allclose(bng['x'][:, 0], [(end + 1100) / 2, 1100], atol=1e-9)
    np.testing.assert_allclose(_times(out['x']), [end, (end + 1100) / 2, 1100], atol=1e-9)
    np.testing.assert_allclose(out['x']['Atot'], 100.0, rtol=1e-4)


def test_a_protocol_scan_continues_from_the_stop_row(tmp_path):
    """The protocol block takes the same path: a stop then a simulate, run at each point of a
    method=>"protocol" scan over kdeg. Oracle: BNG2.pl's .scan, and A(t*) exp(-kdeg 5) with
    A(t*) the first integer t at which A < 500."""
    import pybnf.bngsim_model as bngsim_model
    from pybnf import pset
    from pybnf.data import Data
    protocol = [STOP.replace(',suffix=>"eq"', ''),
                'simulate({method=>"ode",t_end=>5,n_steps=>5})']
    scan = ('parameter_scan({method=>"protocol",parameter=>"kdeg",par_scan_vals=>[0.1,0.2],'
            'suffix=>"s"})')
    code, log = _bng2pl(tmp_path, 'm', MODEL + 'begin protocol\n%s\nend protocol\n'
                        % '\n'.join('  ' + p for p in protocol),
                        ['generate_network({overwrite=>1})', scan])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'm_s.scan'))
    stops = np.array([7.0, 4.0])                        # ln 2 / kdeg, rounded up
    np.testing.assert_allclose(bng['Atot'], 1000 * np.exp(-np.array([0.1, 0.2]) * (stops + 5)),
                               rtol=1e-5)
    model = bngsim_model.BngsimModel('m', [scan], [('parameter_scan', 's')], [],
                                     nf=str(tmp_path / 'm.net'), protocol=protocol)
    model.param_set = pset.PSet([])
    out = model.execute(str(tmp_path / 'run'), 'm_bngsim', 60)['s']
    np.testing.assert_allclose(out['Atot'], bng['Atot'], rtol=1e-5)
    np.testing.assert_allclose(out['Btot'], bng['Btot'], rtol=1e-5)


def test_a_stochastic_stop_if_is_bng2pls(tmp_path):
    """run_network checks a stochastic run's stop condition after every reaction, so it stops,
    and writes its last row, between output times (the oracle: BNG2.pl's last row is not an
    output time). The bridge checks at output times only, so it cannot write those rows or
    carry that state: the action is not the bridge's (auto runs it on BNG2.pl), and the bridge
    refuses it, naming stop_if."""
    from pybnf.bngsim_model.classification import (bngsim_unread_action_options,
                                                   classify_actions_for_bngsim)
    from pybnf.bngsim_model.grid import check_output_grids
    action = STOP.replace('"ode"', '"ssa"')
    code, log = _bng2pl(tmp_path, 'm', MODEL, ['generate_network({overwrite=>1})', action])
    assert code == 0, log
    last = np.loadtxt(tmp_path / 'm_eq.gdat', ndmin=2)[-1, 0]
    assert last != round(last) and last < 20
    actions = ['generate_network({overwrite=>1})', action]
    assert classify_actions_for_bngsim(actions) is None
    assert bngsim_unread_action_options(actions) == [('eq', ['stop_if'])]
    with pytest.raises(PybnfError) as info:
        check_output_grids('m', [action])
    assert info.value.log_message == ('Model m, simulate action with suffix "eq": names stop_if, '
                                      'which the bngsim bridge does not read as BNG2.pl does')
    assert classify_actions_for_bngsim(['generate_network({overwrite=>1})', STOP]) is not None


def test_the_network_free_paths_both_ignore_stop_if(tmp_path):
    """BNG2.pl's NFsim takes no stop condition: its run reaches t_end, as the bridge's does."""
    from pybnf.data import Data
    from .test_bngsim_output_grid import FIXTURES, _bngsim_nf, _model_block
    action = ('simulate({method=>"nf",t_end=>10,n_steps=>10,gml=>1000,stop_if=>"bound>30",'
              'suffix=>"tc"})')
    code, log = _bng2pl(tmp_path, 'binding', _model_block(FIXTURES / 'e2e_nf_binding.bngl'),
                        [action])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'binding_tc.gdat'))
    np.testing.assert_allclose(_times(bng), np.arange(11), atol=1e-9)
    assert bng['bound'][-1] > 30
    out = _bngsim_nf(action, 'tc').execute(str(tmp_path / 'run'), 'nf', 60)['tc']
    np.testing.assert_allclose(_times(out), np.arange(11), atol=1e-9)


def test_the_gradient_path_refuses_to_carry_a_stopped_state(tmp_path):
    """bngsim cannot put a stopped run's dx/dtheta back at the stop row, so a sensitivity run
    that continues from it is refused (with a reset between, it starts afresh and runs)."""
    import pybnf.bngsim_model as bngsim_model
    from pybnf import pset
    code, log = _bng2pl(tmp_path, 'm', MODEL, ['generate_network({overwrite=>1})', STOP])
    assert code == 0, log
    after = 'simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"x"})'

    def model(actions):
        m = bngsim_model.BngsimModel('m', actions, [('simulate', 'eq'), ('simulate', 'x')], [],
                                     nf=str(tmp_path / 'm.net'))
        m.param_set = pset.PSet([])
        m.enable_output_sensitivities(params=['kdeg'])
        return m

    with pytest.raises(PybnfError) as info:
        model([STOP, after]).execute(str(tmp_path / 'run1'), 'g', 60)
    assert 'continues from the state a stop_if run stopped at' in info.value.message
    out = model([STOP, 'resetConcentrations()', after]).execute(str(tmp_path / 'run2'), 'g', 60)
    t = _times(out['x'])
    got = out['x'].output_sensitivities.slice_for('observable:Atot')[:, 0]
    np.testing.assert_allclose(got, -t * 1000 * np.exp(-0.1 * t), rtol=1e-4, atol=1e-6)


def test_the_issues_threshold_fit_scores_as_on_bng2pl(tmp_path, monkeypatch):
    """lanl/PyBNF#1010's scenario: treat when A falls below 500, fit the treatment phase. At
    kdeg = 0.1 the data are BNG2.pl's own rows (A(7) exp(-0.1 t)); the bridge scored 502548
    there, BNG2.pl 0.07."""
    R.require_bng2pl()
    R.install(monkeypatch)
    model = tmp_path / 'stopif.bngl'
    model.write_text(MODEL.replace('kdeg 0.1', 'kdeg kdeg__FREE') + 'begin actions\n'
                     '  generate_network({overwrite=>1})\n  %s\n'
                     '  simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"tc"})\n'
                     'end actions\n' % STOP)
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time Atot\n' + ''.join('%g %.6f\n' % (t, 1000 * np.exp(-0.1 * (7 + t)))
                                            for t in range(6)))
    monkeypatch.chdir(tmp_path)
    conf = R.make_config(tmp_path, model, exp, {'kdeg__FREE': ('uniform_var', 0.0999, 0.1001)},
                         'de', population_size=3, max_iterations=1, random_seed=1)
    alg = R.build(conf, 'de')
    R.drive(alg)
    assert alg.trajectory.best_score() < 1.0
