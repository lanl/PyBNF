"""lanl/PyBNF#1000: classifying a model's actions for the bngsim bridge by the options that decide
their rows (#994).

1. A protocol block's actions were not classified, so a protocol ``simulate`` naming an option the
   bridge does not read (``argfile``, ...) left a model that runs the protocol on the bridge, and
   the set-up then refused it, saying ``auto`` "picks BNG2.pl for it", which it had not. They are
   now classified with the actions when an action runs the protocol block. ``auto`` cannot hand
   such a model to BNG2.pl instead: PyBNF runs a model that simulates a network on BNG2.pl from
   its generated ``.net`` file, which carries no protocol block (#926). So the set-up refuses it,
   and says why.
2. ``print_on_stop=>0`` was treated as unread with or without ``stop_if``. BNG2.pl reads it only
   beside ``stop_if`` (BNGAction.pm 424-432), so without one the action is the bridge's.

The oracle is BNG2.pl: its rows for an action, or its failure on a model.
"""

import subprocess
from subprocess import CalledProcessError

import numpy as np
import pytest

from pybnf import algorithms, config
from pybnf.bngsim_model import BNGSIM_BACKEND_NET, BngsimModel, classify_actions_for_bngsim
from pybnf.bngsim_model.classification import action_lines_run, bngsim_unread_action_options
from pybnf.bngsim_model.grid import check_output_grids
from pybnf.data import Data
from pybnf.printing import PybnfError
from pybnf.pset import BNGLModel, NetModel

from . import recovery_harness as R
from .test_bngsim_output_grid import FIXTURES, _bng2pl, _bngsim_net, _model_block, _times

pytestmark = pytest.mark.bngsim

_PROTOCOL_MODEL = """begin model
begin parameters
  S0  100
  k   %s
  d   1
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
  S() -> 0  k
end reaction rules
end model
begin protocol
  simulate({method=>"ode",%s})
end protocol
begin actions
  generate_network({overwrite=>1})
  parameter_scan({method=>"%s",parameter=>"d",par_scan_vals=>[1,2],t_end=>2,n_steps=>2,suffix=>"s"})
end actions
"""

_ARGFILE = 'argfile=>"simargs.txt"'
_GRID = 't_end=>2,n_steps=>4'


def _protocol_model(tmp_path, protocol_options, method='protocol', k='k__FREE'):
    """A decay model whose protocol block holds one simulate, run by a parameter_scan with
    ``method``, and the data its scan is scored against (the state at t = 2 with k = 0.3)."""
    (tmp_path / 'simargs.txt').write_text('t_end 2\nn_steps 4\n')
    model = tmp_path / 'decay.bngl'
    model.write_text(_PROTOCOL_MODEL % (k, protocol_options, method))
    exp = tmp_path / 's.exp'
    exp.write_text('# d Stot\n1 %.6f\n2 %.6f\n' % ((100 * np.exp(-0.6),) * 2))
    return model, exp


def _set_up(tmp_path, model, exp, backend):
    """The fit's model set-up, ``Algorithm._initialize_models``, through a DE algorithm."""
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'bngl_backend': backend, 'wall_time_sim': 60,
        ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    return algorithms.DifferentialEvolution(cfg)


# ---------------------------------------------------------------------------------------------
# 1. Protocol-block actions
# ---------------------------------------------------------------------------------------------

def test_protocol_actions_are_classified_with_the_actions_that_run_them(tmp_path):
    """The protocol block's lines count when an action runs it, and only then: BNG2.pl runs a
    protocol block only from a ``simulate_protocol`` action or a ``method=>"protocol"`` scan
    (``simulate_protocol``, BNGAction.pm 30-73)."""
    model = BNGLModel(str(_protocol_model(tmp_path, _ARGFILE)[0]))
    assert classify_actions_for_bngsim(model.actions) == BNGSIM_BACKEND_NET
    assert classify_actions_for_bngsim(action_lines_run(model.actions, model.protocol)) is None
    assert bngsim_unread_action_options(model.protocol) == [('time_course', ['argfile'])]
    unrun = BNGLModel(str(_protocol_model(tmp_path, _ARGFILE, method='ode')[0]))
    assert action_lines_run(unrun.actions, unrun.protocol) == list(unrun.actions)
    assert classify_actions_for_bngsim(action_lines_run(unrun.actions, unrun.protocol)) \
        == BNGSIM_BACKEND_NET
    # A protocol nothing runs is not the bridge's to refuse either.
    check_output_grids('decay', action_lines_run(unrun.actions, unrun.protocol))


def test_bng2pl_cannot_run_the_protocol_of_a_model_that_generates_its_network(tmp_path,
                                                                              monkeypatch):
    """The premise of the refusals below, shown on BNG2.pl itself: the model as PyBNF hands it
    to BNG2.pl (a NetModel of the generated .net; PyBNF generates the network of every model
    with a network method, protocol included) loses its protocol block, and BNG2.pl stops at
    the scan's first point (#926). The same model as written runs, argfile and all, and writes
    the state at the argfile's t_end, 100 exp(-0.3 * 2), at each point."""
    R.require_bng2pl()
    bng = tmp_path / 'bng'
    bng.mkdir()
    model_path, _exp = _protocol_model(bng, _ARGFILE, k='0.3')
    done = subprocess.run([R._default_bng_command(), model_path.name], cwd=bng,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    np.testing.assert_allclose(Data(file_name=str(bng / 'decay_s.scan'))['Stot'],
                               100 * np.exp(-0.3 * 2) * np.ones(2), rtol=1e-5)
    model = BNGLModel(str(_protocol_model(tmp_path, _ARGFILE)[0]))
    assert model.generates_network and model.protocol
    monkeypatch.chdir(tmp_path)
    net = NetModel(model.name, model.actions, model.suffixes, model.mutants,
                   nf=str(bng / 'decay.net'))
    net.bng_command = R._default_bng_command()
    folder = tmp_path / 'net'
    folder.mkdir()
    with pytest.raises(CalledProcessError):
        net.execute(str(folder), 'decay_net', 60)
    assert 'begin protocol' not in (folder / 'decay_net.bngl').read_text()
    assert 'ABORT' in (folder / 'decay_net.log').read_text()


@pytest.mark.parametrize('backend', ['auto', 'bngsim'])
def test_a_protocol_action_the_bridge_cannot_read_is_refused_accurately(tmp_path, monkeypatch,
                                                                        backend):
    """``auto`` used to keep such a model on the bridge and refuse the protocol line saying
    ``auto`` "picks BNG2.pl for it". It cannot: BNG2.pl's path drops the protocol (above). Both
    backends now refuse the model, naming the option and saying BNG2.pl's path cannot run it."""
    R.require_bng2pl()
    model, exp = _protocol_model(tmp_path, _ARGFILE)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        _set_up(tmp_path, model, exp, backend)
    message = info.value.message
    assert 'picks BNG2.pl for it' not in message and 'runs it on BNG2.pl' not in message
    if backend == 'auto':
        assert info.value.log_message == (
            'Model decay: bngl_backend = auto simulates it with BNG2.pl, because the bngsim '
            'bridge does not support its actions (an action in the protocol block names '
            'argfile, which the bridge does not read), which cannot run its protocol block')
        assert 'carries no protocol block' in message and '#926' in message
    else:
        assert ('An action in the protocol block names argfile, which BNG2.pl reads to decide '
                "its output rows and the bridge does not; PyBNF's BNG2.pl path cannot run the "
                'model either') in message


def test_a_protocol_the_bridge_reads_stays_on_the_bridge(tmp_path, monkeypatch):
    """The same model with the grid written into the protocol action runs on the bridge under
    ``auto`` (the remedy the refusal above gives)."""
    R.require_bng2pl()
    model, exp = _protocol_model(tmp_path, _GRID)
    monkeypatch.chdir(tmp_path)
    alg = _set_up(tmp_path, model, exp, 'auto')
    assert isinstance(alg.model_list[0], BngsimModel)


def test_an_unrun_protocol_the_bridge_cannot_read_is_not_refused(tmp_path, monkeypatch):
    """A protocol block no action runs is never simulated, on BNG2.pl or on the bridge, so its
    argfile does not move the model off the bridge (the scan here is an ode scan)."""
    R.require_bng2pl()
    model, exp = _protocol_model(tmp_path, _ARGFILE, method='ode')
    monkeypatch.chdir(tmp_path)
    alg = _set_up(tmp_path, model, exp, 'auto')
    assert isinstance(alg.model_list[0], BngsimModel)


def test_bionetgen_refuses_a_protocol_scan_it_cannot_run(tmp_path, monkeypatch):
    """Under ``bngl_backend = bionetgen`` the same NetModel hand-off made every simulation of a
    protocol scan fail (#926), and the fit ended "Fitting complete" at an objective of inf. It is
    refused when the models are set up, before any simulation."""
    R.require_bng2pl()
    model, exp = _protocol_model(tmp_path, _GRID)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        _set_up(tmp_path, model, exp, 'bionetgen')
    assert info.value.log_message == ('Model decay: bngl_backend = bionetgen simulates it with '
                                      'BNG2.pl, which cannot run its protocol block')


def test_a_failed_bridge_set_up_does_not_fall_back_to_a_protocol_bng2pl_cannot_run(
        tmp_path, monkeypatch):
    """When the bridge fails to start under ``auto``, set-up falls back to BNG2.pl; for a model
    whose protocol BNG2.pl's path drops, it refuses instead, naming the failure."""
    from pybnf.algorithms import base
    R.require_bng2pl()

    class Broken(BngsimModel):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('bridge start-up failed on purpose')

    monkeypatch.setattr(base, 'BngsimModel', Broken)
    model, exp = _protocol_model(tmp_path, _GRID)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        _set_up(tmp_path, model, exp, 'auto')
    assert 'bridge start-up failed on purpose' in info.value.log_message
    assert info.value.log_message.endswith('which cannot run its protocol block')


# ---------------------------------------------------------------------------------------------
# 2. print_on_stop without stop_if
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('stop_if, unread', [('', []), (',stop_if=>"Stot<50"', ['print_on_stop'])],
                         ids=['without_stop_if', 'with_stop_if'])
def test_print_on_stop_is_unread_only_beside_stop_if(stop_if, unread):
    action = 'simulate({method=>"ode",t_end=>4,n_steps=>4%s,print_on_stop=>0,suffix=>"p1"})' \
        % stop_if
    actions = ['generate_network({overwrite=>1})', action]
    assert bngsim_unread_action_options(actions) == ([('p1', unread)] if unread else [])
    assert classify_actions_for_bngsim(actions) == (None if unread else BNGSIM_BACKEND_NET)


def test_print_on_stop_without_stop_if_writes_bng2pls_rows_on_the_bridge(tmp_path):
    """BNG2.pl passes print_on_stop to run_network only with a stop condition, so without one it
    writes the rows it writes without the option, and so does the bridge. Oracle: BNG2.pl's own
    rows and the closed form 100 exp(-0.3 t)."""
    action = 'simulate({method=>"ode",t_end=>4,n_steps=>4,print_on_stop=>0,suffix=>"p1"})'
    code, log = _bng2pl(tmp_path, 'decay', _model_block(FIXTURES / 'e2e_ode_decay.bngl'),
                        ['generate_network({overwrite=>1})', action])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'decay_p1.gdat'))
    np.testing.assert_allclose(_times(bng), [0, 1, 2, 3, 4], atol=1e-9)
    check_output_grids('decay', [action])
    out = _bngsim_net(tmp_path / 'decay.net', [action], ['p1']).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)['p1']
    np.testing.assert_allclose(_times(out), _times(bng), atol=1e-9)
    np.testing.assert_allclose(out['Stot'], bng['Stot'], rtol=1e-5)
    np.testing.assert_allclose(out['Stot'], 100 * np.exp(-0.3 * _times(out)), rtol=1e-5)


def test_bngsim_runs_print_on_stop_without_stop_if(tmp_path, monkeypatch):
    """Through the fit's set-up: ``bngl_backend = bngsim`` used to refuse the action."""
    R.require_bng2pl()
    text = _model_block(FIXTURES / 'e2e_ode_decay.bngl').replace('k   0.3', 'k   k__FREE')
    model = tmp_path / 'decay.bngl'
    model.write_text(text + 'generate_network({overwrite=>1})\n'
                     'simulate({method=>"ode",t_end=>4,n_steps=>4,print_on_stop=>0,'
                     'suffix=>"p1"})\n')
    exp = tmp_path / 'p1.exp'
    exp.write_text('# time Stot\n1 74.08\n2 54.88\n')
    monkeypatch.chdir(tmp_path)
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'bngl_backend': 'bngsim',
        ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    alg = algorithms.DifferentialEvolution(cfg)
    assert isinstance(alg.model_list[0], BngsimModel)



# ---------------------------------------------------------------------------------------------
# A simulate_protocol action runs the protocol block too (#1005 review, F2)
# ---------------------------------------------------------------------------------------------

_SIMULATE_PROTOCOL = """begin protocol
  simulate({method=>"ode",t_end=>2,n_steps=>4,suffix=>"p"})
end protocol
begin actions
  generate_network({overwrite=>1})
  simulate({method=>"ode",t_end=>4,n_steps=>4,suffix=>"x"})
  simulate_protocol({});  # the protocol, once
end actions
"""


def _simulate_protocol_model(tmp_path, k='k__FREE'):
    model = tmp_path / 'decay.bngl'
    model.write_text((_PROTOCOL_MODEL[:_PROTOCOL_MODEL.index('begin protocol')] % k)
                     + _SIMULATE_PROTOCOL)
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n2 %.6f\n4 %.6f\n' % (100 * np.exp(-0.6), 100 * np.exp(-1.2)))
    return model, exp


def test_a_simulate_protocol_action_runs_the_protocol_block():
    from pybnf.bngsim_model.classification import runs_protocol_block, runs_simulate_protocol
    for line in ('simulate_protocol({})', '  simulate_protocol({});  # once',
                 'simulate_protocol({prefix=>"p"})'):
        assert runs_simulate_protocol([line]) and runs_protocol_block([line])
    assert not runs_protocol_block(['# simulate_protocol({})',
                                    'simulate({method=>"ode",t_end=>2,n_steps=>2})'])
    assert action_lines_run(['simulate_protocol({})'], ['setParameter("k",1)']) == [
        'simulate_protocol({})', 'setParameter("k",1)']


def test_bng2pl_runs_simulate_protocol_only_from_the_model_as_written(tmp_path, monkeypatch):
    """The oracle for the refusals below: BNG2.pl runs the model as written, the protocol
    included (its simulate writes 100 exp(-0.3 t) at t = 0 ... 2, continuing from the state at
    t = 4), and stops on the model PyBNF hands it (a NetModel of the generated .net file)."""
    R.require_bng2pl()
    bng = tmp_path / 'bng'
    bng.mkdir()
    model_path, _exp = _simulate_protocol_model(bng, k='0.3')
    done = subprocess.run([R._default_bng_command(), model_path.name], cwd=bng,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'executing 1 commands in protocol' in done.stdout
    model = BNGLModel(str(_simulate_protocol_model(tmp_path)[0]))
    assert model.protocol and model.generates_network
    monkeypatch.chdir(tmp_path)
    net = NetModel(model.name, model.actions, model.suffixes, model.mutants,
                   nf=str(bng / 'decay.net'))
    net.bng_command = R._default_bng_command()
    folder = tmp_path / 'net'
    folder.mkdir()
    with pytest.raises(CalledProcessError):
        net.execute(str(folder), 'decay_net', 60)
    assert 'ABORT' in (folder / 'decay_net.log').read_text()


@pytest.mark.parametrize('backend, message', [
    ('auto', 'Model decay: bngl_backend = auto simulates it with BNG2.pl, because the bngsim '
             'bridge does not support its actions, which cannot run its protocol block'),
    ('bionetgen', 'Model decay: bngl_backend = bionetgen simulates it with BNG2.pl, which cannot '
                  'run its protocol block'),
    ('bngsim', 'bngl_backend = bngsim was requested for model decay, but its BNGL actions are not '
               'supported by the bngsim bridge. Its simulate_protocol action runs the protocol '
               'block'),
])
def test_a_simulate_protocol_model_is_refused_where_nothing_runs_it(tmp_path, monkeypatch,
                                                                   backend, message):
    """The bridge does not run simulate_protocol, and PyBNF's BNG2.pl path drops the protocol
    block, so every backend would fail every simulation: refused at set-up, saying why. Under
    auto it used to fall back to BNG2.pl and end "Fitting complete" at inf."""
    R.require_bng2pl()
    model, exp = _simulate_protocol_model(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        _set_up(tmp_path, model, exp, backend)
    assert info.value.log_message.startswith(message)
    if backend == 'auto':
        assert 'simulate_protocol' in ' '.join(info.value.hints or [])
