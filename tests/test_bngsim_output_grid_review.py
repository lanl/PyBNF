"""More action shapes for the bngsim bridge's output grid, each against BNG2.pl itself (review of
lanl/PyBNF#994, #998).

The oracle is BNG2.pl's own output for the same action, plus the closed form of the fixture
(S' = -k S, S(0) = 100, k = 0.3, so S(t) = 100 exp(-k t)) where one applies.
"""

import numpy as np
import pytest

from pybnf.data import Data

from .test_bngsim_output_grid import (
    FIXTURES,
    _bng2pl,
    _bngsim_net,
    _bngsim_nf,
    _model_block,
    _times,
)

pytestmark = pytest.mark.bngsim


def _decay_block():
    return _model_block(FIXTURES / 'e2e_ode_decay.bngl')


def test_a_continued_ode_simulate_writes_bng2pls_grid(tmp_path):
    """``continue=>1`` with no ``t_start`` starts where the previous action stopped, and its
    ``n_output_steps`` divide [t_start, t_end], as BNG2.pl's ``step_size = (t_end -
    t_start)/n_steps`` does. The values continue the same decay: 100 exp(-0.3 t)."""
    first = 'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"a"})'
    second = 'simulate({method=>"ode",continue=>1,t_end=>20,n_output_steps=>5,suffix=>"b"})'
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})', first, second])
    assert code == 0, log
    # A continued run appends: BNG2.pl writes no header and no row at t_start.
    bng = np.loadtxt(tmp_path / 'decay_b.gdat', ndmin=2)
    np.testing.assert_allclose(bng[:, 0], [12.0, 14.0, 16.0, 18.0, 20.0], atol=1e-9)
    np.testing.assert_allclose(bng[:, 1], 100.0 * np.exp(-0.3 * bng[:, 0]), rtol=1e-5)
    out = _bngsim_net(tmp_path / 'decay.net', [first, second], ['a', 'b']).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)['b']
    # The bridge also writes the t_start row (a difference that predates #998, and harmless
    # to scoring, which matches by time); every row BNG2.pl writes is on its grid.
    np.testing.assert_allclose(_times(out), np.concatenate(([10.0], bng[:, 0])), atol=1e-9)
    np.testing.assert_allclose(out['Stot'][1:], bng[:, 1], rtol=1e-5)


def test_a_scan_with_n_output_steps_scores_bng2pls_final_states(tmp_path):
    """``parameter_scan`` runs ``simulate`` at each point; ``n_output_steps`` is a legal step
    key for it, and a scan writes one row per point, the final state. Oracle: BNG2.pl's .scan
    and the closed form 100 exp(-k 5)."""
    scan = ('parameter_scan({parameter=>"k",method=>"ode",par_scan_vals=>[0.1,0.3,0.5],'
            't_start=>0,t_end=>5,n_output_steps=>10,suffix=>"s"})')
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})', scan])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'decay_s.scan'))
    ks = np.array([0.1, 0.3, 0.5])
    np.testing.assert_allclose(bng['Stot'], 100.0 * np.exp(-ks * 5.0), rtol=1e-5)
    out = _bngsim_net(tmp_path / 'decay.net', [scan], ['s']).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)['s']
    np.testing.assert_allclose(out['k'], ks)
    np.testing.assert_allclose(out['Stot'], bng['Stot'], rtol=1e-5)


def test_nf_n_steps_takes_precedence_over_sample_times_as_in_bng2pl(tmp_path):
    """BNG2.pl's NFsim reads ``n_steps`` and ignores ``sample_times`` beside it."""
    action = ('simulate({method=>"nf",t_start=>0,t_end=>10,n_steps=>5,sample_times=>[0,1,2],'
              'gml=>1000,suffix=>"tc"})')
    code, log = _bng2pl(tmp_path, 'binding', _model_block(FIXTURES / 'e2e_nf_binding.bngl'),
                        [action])
    assert code == 0, log
    bng = _times(Data(file_name=str(tmp_path / 'binding_tc.gdat')))
    np.testing.assert_allclose(bng, np.linspace(0, 10, 6), atol=1e-9)
    out = _bngsim_nf(action, 'tc').execute(str(tmp_path / 'run'), 'nf_grid', 60)['tc']
    np.testing.assert_allclose(_times(out), bng, atol=1e-9)


def test_sample_times_after_t_start_are_reached_from_t_start(tmp_path):
    """An edition-1 action whose ``sample_times`` begin after ``t_start`` (0 by default):
    BNG2.pl integrates from t_start and writes it as its first row; the values at the sample
    times are the decay from t = 0. Oracle: the closed form 100 exp(-0.3 t) at 5 and 10, and
    BNG2.pl's own rows. (Edition-2 experiments always put t = 0 first, so they never meet this.)
    It was a strict xfail until lanl/PyBNF#872: the bridge reported the initial state at t = 2."""
    action = 'simulate({method=>"ode",sample_times=>[2,5,10],suffix=>"x"})'
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})', action])
    assert code == 0, log
    bng = np.loadtxt(tmp_path / 'decay_x.gdat', ndmin=2)
    np.testing.assert_allclose(bng[:, 0], [0.0, 2.0, 5.0, 10.0], atol=1e-9)
    np.testing.assert_allclose(bng[:, 1], 100.0 * np.exp(-0.3 * bng[:, 0]), rtol=1e-5)
    out = _bngsim_net(tmp_path / 'decay.net', [action], ['x']).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)['x']
    at = {round(t, 9): v for t, v in zip(_times(out), out['Stot'])}
    for t in (2.0, 5.0, 10.0):
        assert t in at, (t, sorted(at))
        np.testing.assert_allclose(at[t], 100.0 * np.exp(-0.3 * t), rtol=1e-5)
    # ...and the same rows as BNG2.pl, the t_start row included.
    np.testing.assert_allclose(_times(out), bng[:, 0], atol=1e-9)
    np.testing.assert_allclose(out['Stot'], bng[:, 1], rtol=1e-5)


def test_an_argfile_simulate_is_not_refused_as_one_bng2pl_refuses(tmp_path):
    """BNG2.pl's ``simulate`` reads its arguments, ``t_end`` and ``n_steps`` among them, from an
    ``argfile`` (BNGAction.pm). The bridge does not read argfiles, so it cannot know the action's
    grid; but BNG2.pl runs the action, so it must not be refused with the claim that BNG2.pl
    refuses it. (Classified as unsupported, ``auto`` would run it on BNG2.pl.)

    Oracle: BNG2.pl writes the argfile's grid, t_end 2 in 8 steps, and the closed form."""
    from pybnf.bngsim_model.classification import classify_actions_for_bngsim
    from pybnf.bngsim_model.grid import check_output_grids
    from pybnf.printing import PybnfError
    (tmp_path / 'simargs.txt').write_text('t_end 2\nn_steps 8\n')
    action = 'simulate({method=>"ode",argfile=>"%s",suffix=>"p1"})' % (tmp_path / 'simargs.txt')
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})', action])
    assert code == 0, log
    bng = Data(file_name=str(tmp_path / 'decay_p1.gdat'))
    np.testing.assert_allclose(_times(bng), np.linspace(0, 2, 9), atol=1e-9)
    np.testing.assert_allclose(bng['Stot'], 100.0 * np.exp(-0.3 * _times(bng)), rtol=1e-5)
    actions = ['generate_network({overwrite=>1})', action]
    if classify_actions_for_bngsim(actions) is None:
        return                                  # not the bridge's: auto runs it on BNG2.pl
    try:
        check_output_grids('decay', actions)
    except PybnfError as error:
        assert 'BNG2.pl refuses' not in error.message, error.message
