"""Third independent review of lanl/PyBNF#1005 (bebfcdce): the remedy given for a model that runs
its protocol block with ``simulate_protocol`` (review R4), and the load-time check of data
against the rows of a suffix's kept output when a ``postprocess`` script rewrites the times.

The oracle is BNG2.pl: its failure on the model PyBNF hands it, and its rows for an action.
Tests marked ``xfail(strict=True)`` record a finding of the review: they state what should
happen, and fail until it does.
"""

import subprocess

import numpy as np
import pytest

from pybnf import algorithms, config
from pybnf.printing import PybnfError

from . import recovery_harness as R
from .test_bngsim_output_grid import FIXTURES, _bng2pl, _model_block

pytestmark = pytest.mark.bngsim


def _decay_free():
    return _model_block(FIXTURES / 'e2e_ode_decay.bngl').replace('k   0.3', 'k   k__FREE')


_PROTOCOL = ('begin protocol\n  simulate({method=>"ode",t_end=>5,n_steps=>5})\nend protocol\n'
             'begin actions\n  generate_network({overwrite=>1})\n'
             '  simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"x"})\n'
             '  simulate_protocol({})\nend actions\n')


def _set_up(tmp_path, backend):
    model = tmp_path / 'decay.bngl'
    model.write_text(_decay_free() + _PROTOCOL)
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n2 54.881164\n4 30.119421\n')
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'bngl_backend': backend, 'wall_time_sim': 60,
        ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    return algorithms.DifferentialEvolution(cfg)


def test_bng2pl_stops_on_simulate_protocol_without_its_protocol_block(tmp_path):
    """The premise: PyBNF hands BNG2.pl the generated .net (a NetModel), which has no protocol
    block, and BNG2.pl's simulate_protocol then aborts (BNGAction.pm 36). The model as written,
    protocol included, runs."""
    code, log = _bng2pl(tmp_path, 'decay', _model_block(FIXTURES / 'e2e_ode_decay.bngl'),
                        ['generate_network({overwrite=>1})', 'simulate_protocol({})'])
    assert code != 0 and "Can't use an undefined value as an ARRAY reference" in log, log
    whole = tmp_path / 'whole'
    whole.mkdir()
    (whole / 'decay.bngl').write_text(_model_block(FIXTURES / 'e2e_ode_decay.bngl') + _PROTOCOL)
    done = subprocess.run([R._default_bng_command(), 'decay.bngl'], cwd=whole,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize('backend', ['bionetgen', 'auto'])
def test_a_simulate_protocol_refusal_never_points_at_a_backend_that_refuses_it(tmp_path,
                                                                              monkeypatch,
                                                                              backend):
    """Neither backend runs the action, so the remedy must not send the user to the other one
    (under bionetgen it used to say "Set bngl_backend = bngsim ..., which runs the protocol
    block", and bngsim then refused the action)."""
    R.require_bng2pl()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError) as info:
        _set_up(tmp_path, backend)
    assert 'cannot run its protocol block' in info.value.log_message
    remedy = ' '.join(info.value.hints)
    assert 'Set bngl_backend = bngsim' not in remedy
    assert 'does not run a simulate_protocol action' in remedy
    assert "write the protocol's steps into the actions block" in remedy
    with pytest.raises(PybnfError) as info:
        _set_up(tmp_path, 'bngsim')
    assert 'simulate_protocol' in info.value.message


_MINUTES = '''def postprocess(simdata):
    """The model runs in seconds; report its time in minutes."""
    simdata.data[:, 0] = simdata.data[:, 0] / 60.0
    return simdata
'''


def test_data_on_the_rows_a_postprocess_script_writes_loads(tmp_path, monkeypatch):
    """An n_steps grid over 0..600 s, rescaled to minutes by the documented postprocess hook:
    the scored output has rows at 0, 1, ..., 10, where the data are. Oracle: BNG2.pl's rows,
    passed through the same script."""
    action = 'simulate({method=>"ode",t_end=>600,n_steps=>10,suffix=>"x"})'
    code, log = _bng2pl(tmp_path, 'decay', _model_block(FIXTURES / 'e2e_ode_decay.bngl'),
                        ['generate_network({overwrite=>1})', action])
    assert code == 0, log
    (tmp_path / 'minutes.py').write_text(_MINUTES)
    namespace = {}
    exec(_MINUTES, namespace)

    class _Rows:
        data = np.loadtxt(tmp_path / 'decay_x.gdat', ndmin=2)

    scored = namespace['postprocess'](_Rows).data[:, 0]
    times = [1, 2, 5, 10]
    assert all(np.any(np.isclose(scored, t, atol=0.)) for t in times), scored
    model = tmp_path / 'm.bngl'
    model.write_text(_decay_free() + 'begin actions\n  generate_network({overwrite=>1})\n  %s\n'
                     'end actions\n' % action)
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n' + ''.join('%g %.6f\n' % (t, 100 * np.exp(-0.005 * 60 * t))
                                            for t in times))
    monkeypatch.chdir(tmp_path)
    config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'wall_time_sim': 60,
        'postprocess': [[str(tmp_path / 'minutes.py'), 'x']],
        ('uniform_var', 'k__FREE'): [0.001, 0.02]})
