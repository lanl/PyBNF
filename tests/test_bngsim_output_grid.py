"""The bngsim bridge writes the output grid BNG2.pl writes for a ``simulate`` action (#990 review).

The bridge read ``n_steps`` and ``t_end`` with defaults of its own (100 and 100), so on a model
whose action named ``n_output_steps``, or no step count, it wrote 101 rows where BNG2.pl writes 21,
or 2; and an action without ``t_end``, which BNG2.pl refuses, ran to t = 100. It now follows
``Perl2/BNGAction.pm`` (``pybnf/bngsim_model/grid.py``).

The oracle is BNG2.pl itself: the same action is run through BNG2.pl (``run_network`` for ode,
NFsim for nf) and through the bridge, and the time column of each output, or the refusal and its
message, must agree. Skipped where BNG2.pl or bngsim is unavailable.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pybnf import pset
from pybnf.data import Data
from pybnf.printing import PybnfError

from . import recovery_harness as R

pytestmark = pytest.mark.bngsim

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'


def _model_block(path):
    """The ``begin model ... end model`` text of a fixture."""
    text = path.read_text()
    return text[text.index('begin model'):text.index('end model') + len('end model')] + '\n'


def _bng2pl(folder, name, model_text, actions):
    """Run BNG2.pl on ``model_text`` with ``actions``; ``(returncode, stdout)``."""
    R.require_bng2pl()
    (folder / ('%s.bngl' % name)).write_text(model_text + 'begin actions\n' + ''.join(
        '  %s\n' % a for a in actions) + 'end actions\n')
    cmd = [R._default_bng_command(), '%s.bngl' % name]
    if os.name == 'nt':
        cmd = ['perl'] + cmd
    done = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout + done.stderr


def _times(data):
    return np.asarray(data.data[:, data.cols['time']], dtype=float)


def _bngsim_net(net, actions, suffixes):
    import pybnf.bngsim_model as bngsim_model
    model = bngsim_model.BngsimModel(Path(net).stem, list(actions),
                                     [('simulate', s) for s in suffixes], [], nf=str(net))
    model.param_set = pset.PSet([])
    return model


# Each shape: (suffix, action options, BNG2.pl's row count, BNG2.pl's times).
_ODE_SHAPES = [
    ('out', 't_start=>0,t_end=>20,n_output_steps=>20', 21, np.linspace(0, 20, 21)),
    ('none', 't_start=>0,t_end=>20', 2, np.array([0.0, 20.0])),
    ('both', 't_start=>0,t_end=>20,n_steps=>20,sample_times=>[0,5,10]', 21, np.linspace(0, 20, 21)),
    ('st', 't_start=>0,sample_times=>[0,2.5,5,10]', 4, np.array([0.0, 2.5, 5.0, 10.0])),
    ('stend', 't_start=>0,t_end=>20,sample_times=>[0,5,10]', 4, np.array([0.0, 5.0, 10.0, 20.0])),
    ('steps', 't_start=>0,t_end=>20,n_steps=>20', 21, np.linspace(0, 20, 21)),
]


@pytest.mark.parametrize('suffix, options, rows, times', _ODE_SHAPES,
                         ids=[s[0] for s in _ODE_SHAPES])
def test_an_ode_simulate_writes_bng2pls_grid(tmp_path, suffix, options, rows, times):
    """``n_output_steps`` is ``n_steps``; with no step count there is one step; ``n_steps``
    takes precedence over ``sample_times``; ``sample_times`` gives the grid, with a later
    ``t_end`` appended. Before this, the bridge wrote 101 rows for ``out`` and ``none``."""
    action = 'simulate({method=>"ode",%s,suffix=>"%s"})' % (options, suffix)
    code, log = _bng2pl(tmp_path, 'decay', _model_block(FIXTURES / 'e2e_ode_decay.bngl'),
                        ['generate_network({overwrite=>1})', action])
    assert code == 0, log
    bng = _times(Data(file_name=str(tmp_path / ('decay_%s.gdat' % suffix))))
    assert len(bng) == rows
    np.testing.assert_allclose(bng, times, atol=1e-9)

    out = _bngsim_net(tmp_path / 'decay.net', [action], [suffix]).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)[suffix]
    np.testing.assert_allclose(_times(out), bng, atol=1e-9)
    stot = np.asarray(out.data[:, out.cols['Stot']])
    np.testing.assert_allclose(stot, 100.0 * np.exp(-0.3 * bng), rtol=1e-5)    # S0 exp(-k t)


_ODE_REFUSALS = [
    ('both_step_keys', 't_start=>0,t_end=>20,n_steps=>20,n_output_steps=>20',
     "'n_steps' and 'n_output_steps' are the same thing, cannot define both. Please only "
     "define one ('n_output_steps' is preferred)."),
    ('no_t_end', 't_start=>0,n_steps=>20', 'Parameter t_end must be defined.'),
    ('no_t_end_no_steps', 't_start=>0', 'Parameter t_end must be defined.'),
    ('two_sample_times', 't_start=>0,sample_times=>[0,10]',
     "'sample_times' array must contain 3 or more points."),
]


@pytest.mark.parametrize('name, options, message', _ODE_REFUSALS,
                         ids=[s[0] for s in _ODE_REFUSALS])
def test_an_ode_simulate_bng2pl_refuses_is_refused_with_its_message(tmp_path, name, options,
                                                                    message):
    """Both step keys, and a grid without ``t_end``, are errors in BNG2.pl, which says so; the
    bridge ran them on its own defaults. It now refuses them with BNG2.pl's message, when the
    model's actions are checked (as the fit sets up its models) and when the action runs."""
    from pybnf.bngsim_model.grid import check_output_grids
    action = 'simulate({method=>"ode",%s,suffix=>"x"})' % options
    code, log = _bng2pl(tmp_path, 'decay', _model_block(FIXTURES / 'e2e_ode_decay.bngl'),
                        ['generate_network({overwrite=>1})', action])
    assert code != 0 and message in log, log
    with pytest.raises(PybnfError) as info:
        check_output_grids('decay', [action])
    assert info.value.log_message == 'Model decay, simulate action with suffix "x": ' + message
    model = _bngsim_net(FIXTURES / 'e2e_ode_decay.net', [action], ['x'])
    with pytest.raises(PybnfError) as info:
        model.execute(str(tmp_path / 'run'), 'decay_bngsim', 60)
    assert info.value.log_message.endswith(message)


def _bngsim_nf(action, suffix):
    import pybnf.bngsim_model as bngsim_model
    return bngsim_model.BngsimNfModel(
        'e2e_nf_binding', [action], [('simulate', suffix)], [],
        str(FIXTURES / 'e2e_nf_binding.xml'),
        bngl_model_lines=(FIXTURES / 'e2e_nf_binding.bngl').read_text().splitlines(keepends=True),
        param_names=())


def test_an_nf_simulate_writes_bng2pls_grid(tmp_path):
    """NFsim's ``n_steps`` grid, the one NF shape BNG2.pl runs."""
    action = 'simulate({method=>"nf",t_start=>0,t_end=>10,n_steps=>20,gml=>1000,suffix=>"tc"})'
    code, log = _bng2pl(tmp_path, 'binding', _model_block(FIXTURES / 'e2e_nf_binding.bngl'),
                        [action])
    assert code == 0, log
    bng = _times(Data(file_name=str(tmp_path / 'binding_tc.gdat')))
    np.testing.assert_allclose(bng, np.linspace(0, 10, 21), atol=1e-9)
    out = _bngsim_nf(action, 'tc').execute(str(tmp_path / 'run'), 'nf_grid', 60)['tc']
    np.testing.assert_allclose(_times(out), bng, atol=1e-9)


_NF_REFUSALS = [
    ('n_output_steps', 't_start=>0,t_end=>10,n_output_steps=>20',
     'No simulation output requested: set n_steps>0'),
    ('no_steps', 't_start=>0,t_end=>10', 'No simulation output requested: set n_steps>0'),
    ('no_t_end', 't_start=>0,n_steps=>20', 'Parameter t_end must be defined'),
]


@pytest.mark.parametrize('name, options, message', _NF_REFUSALS, ids=[s[0] for s in _NF_REFUSALS])
def test_an_nf_simulate_bng2pl_refuses_is_refused_with_its_message(tmp_path, name, options,
                                                                  message):
    """BNG2.pl's NFsim reads ``n_steps`` alone: ``n_output_steps`` is not read, there is no
    default step count, and ``t_end`` is required. The bridge ran these on 100 steps to t = 100."""
    action = 'simulate({method=>"nf",%s,gml=>1000,suffix=>"tc"})' % options
    code, log = _bng2pl(tmp_path, 'binding', _model_block(FIXTURES / 'e2e_nf_binding.bngl'),
                        [action])
    assert code != 0 and message in log, log
    with pytest.raises(PybnfError) as info:
        _bngsim_nf(action, 'tc').execute(str(tmp_path / 'run'), 'nf_refused', 60)
    assert info.value.log_message == ('Model e2e_nf_binding, simulate action with suffix "tc": '
                                      + message)


def test_nf_sample_times_stay_a_bngsim_extension():
    """BNG2.pl's NFsim takes no ``sample_times``; bngsim's does, and the edition-2 experiments
    write them (#427), so an NF action with ``sample_times`` and no ``n_steps`` still runs on
    them, and ``n_steps`` beside them still takes precedence."""
    from pybnf.bngsim_model.grid import simulate_output_grid
    times = {'sample_times': ['0', '2.5', '10']}
    assert simulate_output_grid(times, 'x', 'nf_reject', t_start=0.0, evaluate=float) == (
        [0.0, 2.5, 10.0], 10.0, None)
    assert simulate_output_grid({**times, 'n_steps': '4', 't_end': '8'}, 'x', 'nf_reject',
                                t_start=0.0, evaluate=float) == (None, 8.0, 4)


def test_a_scan_without_t_end_is_refused_and_a_protocol_scan_is_not():
    """BNG2.pl runs ``simulate`` at each scan point, which needs ``t_end`` without
    ``sample_times``; a protocol scan runs the protocol block instead."""
    from pybnf.bngsim_model.grid import check_output_grids
    scan = 'parameter_scan({parameter=>"k",par_scan_vals=>[1,2],suffix=>"s"})'
    with pytest.raises(PybnfError) as info:
        check_output_grids('m', [scan])
    assert info.value.log_message == ('Model m, parameter_scan action with suffix "s": '
                                      'Parameter t_end must be defined.')
    check_output_grids('m', [scan.replace('suffix', 't_end=>5,suffix')])
    check_output_grids('m', [scan.replace('parameter=>', 'method=>"protocol",parameter=>')])
    with pytest.raises(PybnfError) as info:     # a point's simulate needs 3 sample times
        check_output_grids('m', [scan.replace('suffix', 'sample_times=>[0,5],suffix')])
    assert info.value.log_message.endswith("'sample_times' array must contain 3 or more points.")
    check_output_grids('m', [scan.replace('suffix', 'sample_times=>[0,5,10],suffix')])


def test_a_fit_on_bngsim_refuses_such_an_action_when_it_sets_up(tmp_path):
    """Through the algorithm's model setup under ``bngl_backend = auto``: the refusal is not
    taken as a reason to fall back to BNG2.pl, which refuses the same action."""
    from pybnf import algorithms, config
    R.require_bng2pl()
    text = (FIXTURES / 'parabola.bngl').read_text().replace('t_end=>10,', '')
    assert 't_end' not in text
    model = tmp_path / 'parabola.bngl'
    model.write_text(text)
    exp = str(FIXTURES / 'par1.exp')
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 2, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {exp}, str(model): [exp], 'fit_type': 'de',
        ('uniform_var', 'v1__FREE'): [0.0, 1.0], ('uniform_var', 'v2__FREE'): [0.0, 1.0],
        ('uniform_var', 'v3__FREE'): [0.0, 1.0]})
    with pytest.raises(PybnfError) as info:
        algorithms.DifferentialEvolution(cfg)
    assert info.value.log_message == ('Model parabola, simulate action with suffix "par1": '
                                      'Parameter t_end must be defined.')


@pytest.mark.parametrize('method', ['ode', 'ssa', 'nf'])
def test_an_equilibration_on_one_step_hands_on_the_state_one_on_a_hundred_does(tmp_path,
                                                                              method):
    """An unscored ``simulate`` with no step count now takes one output step where the bridge
    took 100. The state it hands to the next action must not depend on that: for ode the
    integrator's output points are interpolated, and for ssa and NFsim the output steps draw
    nothing (bngsim carries the pending reaction across an output time, #391). Oracle: the
    measured action after a 1-step and after a 100-step equilibration, from the same seeds.
    (NFsim refuses a step-less action, as BNG2.pl does, so the 1-step run states n_steps=>1.)"""
    if method == 'nf':
        steps = ['simulate({method=>"nf",t_start=>0,t_end=>5,n_steps=>%s,gml=>1000,suffix=>"eq"})',
                 'simulate({method=>"nf",t_start=>0,t_end=>5,n_steps=>10,gml=>1000,suffix=>"tc"})']
    else:
        steps = ['simulate({method=>"%s",t_start=>0,t_end=>20%%s,suffix=>"eq"})' % method,
                 'simulate({method=>"%s",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})' % method]
    measured = []
    for i, eq in enumerate(('1' if method == 'nf' else '', '100' if method == 'nf'
                            else ',n_steps=>100')):
        actions = [steps[0] % eq, steps[1]]
        if method == 'nf':
            import pybnf.bngsim_model as bngsim_model
            model = bngsim_model.BngsimNfModel(
                'e2e_nf_binding', actions, [('simulate', 'eq'), ('simulate', 'tc')], [],
                str(FIXTURES / 'e2e_nf_binding.xml'),
                bngl_model_lines=(FIXTURES / 'e2e_nf_binding.bngl').read_text().splitlines(
                    keepends=True), param_names=())
        else:
            net = FIXTURES / ('e2e_ssa_birthdeath.net' if method == 'ssa' else 'e2e_ode_decay.net')
            model = _bngsim_net(net, actions, ['eq', 'tc'])
        out = model.execute(str(tmp_path / ('run%d' % i)), 'eq%d' % i, 60)
        assert out['eq'].data.shape[0] == (2 if i == 0 else 101)
        measured.append(out['tc'].data)
    if method == 'ode':
        np.testing.assert_allclose(measured[0], measured[1], rtol=1e-8)
    else:
        np.testing.assert_array_equal(measured[0], measured[1])


# Each: (option, the action's options, its method). print_on_stop is unread only beside stop_if
# (#1000), and stop_if only on a stochastic method (#1010), so print_on_stop is shown on ode.
_UNREAD = [
    ('argfile', 'argfile=>"simargs.txt"', 'ssa'),
    ('max_sim_steps', 't_end=>20,n_steps=>20,max_sim_steps=>50', 'ssa'),
    ('output_step_interval', 't_end=>20,n_steps=>20,output_step_interval=>5', 'ssa'),
    ('print_on_stop', 't_end=>20,n_steps=>20,stop_if=>"Stot<50",print_on_stop=>0', 'ode'),
]


@pytest.mark.parametrize('option, options, method', _UNREAD, ids=[u[0] for u in _UNREAD])
def test_an_option_bng2pl_reads_for_the_rows_and_the_bridge_does_not_is_not_the_bridges(
        option, options, method):
    """BNG2.pl reads these to decide an action's rows (an argfile can hold t_end and n_steps);
    the bridge reads none of them. Such an action is classified unsupported, so ``auto`` runs
    the model on BNG2.pl; were it run on the bridge anyway, the refusal says the bridge does not
    read the option -- not that BNG2.pl refuses the action, which it runs (#994 re-review)."""
    from pybnf.bngsim_model.classification import (bngsim_unread_action_options,
                                                   classify_actions_for_bngsim)
    from pybnf.bngsim_model.grid import check_output_grids
    action = 'simulate({method=>"%s",%s,suffix=>"p1"})' % (method, options)
    actions = ['generate_network({overwrite=>1})', action]
    assert classify_actions_for_bngsim(actions) is None
    assert bngsim_unread_action_options(actions) == [('p1', [option])]
    with pytest.raises(PybnfError) as info:
        check_output_grids('decay', [action])
    assert info.value.log_message == ('Model decay, simulate action with suffix "p1": names %s, '
                                      'which the bngsim bridge does not read' % option)
    assert 'BNG2.pl refuses' not in info.value.message
    scan = action.replace('simulate({', 'parameter_scan({parameter=>"k",par_scan_vals=>[1,2],')
    with pytest.raises(PybnfError, match='which the bngsim bridge does not read'):
        check_output_grids('decay', [scan])


def test_print_on_stop_at_its_default_stays_the_bridges():
    from pybnf.bngsim_model import BNGSIM_BACKEND_NET
    from pybnf.bngsim_model.classification import classify_actions_for_bngsim
    actions = ['generate_network({overwrite=>1})',
               'simulate({method=>"ode",t_end=>20,n_steps=>20,stop_if=>"Stot<50",'
               'print_on_stop=>1,suffix=>"p1"})']
    assert classify_actions_for_bngsim(actions) == BNGSIM_BACKEND_NET


@pytest.mark.parametrize('backend', ['auto', 'bngsim'])
def test_an_argfile_model_runs_on_bng2pl_under_auto_and_is_refused_by_name_under_bngsim(
        tmp_path, monkeypatch, backend):
    """Through the fit's model set-up. ``auto`` takes the BNG2.pl subprocess path for the model
    (a plain NetModel, not the bridge); ``bngsim`` refuses, naming the argfile."""
    from pybnf import algorithms, config
    from pybnf.bngsim_model import BngsimModel
    R.require_bng2pl()
    (tmp_path / 'simargs.txt').write_text('t_end 2\nn_steps 8\n')
    text = _model_block(FIXTURES / 'e2e_ode_decay.bngl').replace('k   0.3', 'k   k__FREE')
    assert 'k__FREE' in text
    model = tmp_path / 'decay.bngl'
    model.write_text(text + 'generate_network({overwrite=>1})\n'
                     'simulate({method=>"ode",argfile=>"%s",suffix=>"p1"})\n'
                     % (tmp_path / 'simargs.txt'))
    (tmp_path / 'p1.exp').write_text('# time Stot\n' + ''.join(
        '%g %g\n' % (t, 100 * np.exp(-0.3 * t)) for t in (0.25, 0.5, 1.0)))
    monkeypatch.chdir(tmp_path)
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 2, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(tmp_path / 'p1.exp')},
        str(model): [str(tmp_path / 'p1.exp')], 'fit_type': 'de', 'objfunc': 'sos',
        'bngl_backend': backend, ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    if backend == 'bngsim':
        with pytest.raises(PybnfError) as info:
            algorithms.DifferentialEvolution(cfg)
        assert ('The action with suffix "p1" names argfile, which BNG2.pl reads to decide its '
                'output rows and the bridge does not') in info.value.message
        return
    alg = algorithms.DifferentialEvolution(cfg)
    assert [type(m).__name__ for m in alg.model_list] == ['NetModel']
    assert not isinstance(alg.model_list[0], BngsimModel)
