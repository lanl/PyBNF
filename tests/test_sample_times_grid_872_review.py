"""Independent review of lanl/PyBNF#1005 (#872, #1000): more ``sample_times`` shapes against
BNG2.pl, and the cases the review found the change does not cover.

The oracle is BNG2.pl's own output (``.gdat`` for a ``simulate``, ``.scan`` for a
``parameter_scan``) and the closed form of the decay fixture, S(t) = 100 exp(-k (t - t_start)).
The tests marked ``xfail(strict=True)`` record a finding of the review: they state what should
happen, and fail until it does.
"""

import numpy as np
import pytest

from pybnf import config, parse
from pybnf.printing import PybnfError

from . import recovery_harness as R
from .test_bngsim_output_grid import FIXTURES, _bng2pl, _bngsim_net, _model_block, _times

pytestmark = pytest.mark.bngsim

K = 0.3


def _decay_block():
    return _model_block(FIXTURES / 'e2e_ode_decay.bngl')


def _bng2pl_rows(tmp_path, actions, out):
    code, log = _bng2pl(tmp_path, 'decay', _decay_block(),
                        ['generate_network({overwrite=>1})'] + list(actions))
    assert code == 0, log
    return np.loadtxt(tmp_path / out, ndmin=2)


def _bridge(tmp_path, actions, suffixes, wanted):
    return _bngsim_net(tmp_path / 'decay.net', actions, suffixes).execute(
        str(tmp_path / 'run'), 'decay_bngsim', 60)[wanted]


# Each: (id, options, BNG2.pl's row times).
_SHAPES = [
    ('float_times', 'sample_times=>[0.1,0.2,0.3]', [0, 0.1, 0.2, 0.3]),
    ('float_t_end_at_last', 't_end=>0.3,sample_times=>[0.1,0.2,0.3]', [0, 0.1, 0.2, 0.3]),
    ('t_end_arithmetic', 't_end=>2*3.5,sample_times=>[0,5,10]', [0, 5, 7]),
    ('exponent_format', 'sample_times=>[1e0,5E0,1.0e1]', [0, 1, 5, 10]),
    ('n_output_steps_wins', 't_end=>20,n_output_steps=>4,sample_times=>[1,5,10]',
     [0, 5, 10, 15, 20]),
    ('t_end_between_late', 't_end=>6,sample_times=>[1,5,7,9]', [0, 1, 5, 6]),
    ('negative_time_listed', 'sample_times=>[-5,0,5,10]', [0, 5, 10]),
    ('twice_with_t_end', 't_end=>7,sample_times=>[0,5,5,10]', [0, 5, 5, 7]),
    ('t_start_listed_twice', 'sample_times=>[0,0,5,10]', [0, 5, 10]),
]


@pytest.mark.parametrize('options, rows', [s[1:] for s in _SHAPES], ids=[s[0] for s in _SHAPES])
def test_more_sample_times_shapes_write_bng2pls_rows(tmp_path, options, rows):
    """Times that are not integers, a t_end that is an arithmetic expression (known only when
    the action runs), exponent notation, n_output_steps beside sample_times, a negative listed
    time with t_start = 0 (dropped, not refused), and repeated times."""
    action = 'simulate({method=>"ode",%s,suffix=>"x"})' % options
    bng = _bng2pl_rows(tmp_path, [action], 'decay_x.gdat')
    np.testing.assert_allclose(bng[:, 0], rows, atol=1e-12)
    np.testing.assert_allclose(bng[:, 1], 100 * np.exp(-K * bng[:, 0]), rtol=1e-5)
    out = _bridge(tmp_path, [action], ['x'], 'x')
    np.testing.assert_allclose(_times(out), bng[:, 0], atol=1e-12)
    np.testing.assert_allclose(out['Stot'], bng[:, 1], rtol=1e-5)


# Each: (id, scan options, BNG2.pl's (t_start, t_end) for every point).
_SCANS = [
    ('t_start_after_zero', 't_start=>2,sample_times=>[3,5,10]', (2, 10)),
    ('t_end_after_last', 't_end=>12,sample_times=>[3,5,10]', (0, 12)),
    ('n_steps_wins', 't_end=>4,n_steps=>2,sample_times=>[3,5,10]', (0, 4)),
]


@pytest.mark.parametrize('options, span', [s[1:] for s in _SCANS], ids=[s[0] for s in _SCANS])
def test_a_scan_point_runs_bng2pls_span(tmp_path, options, span):
    """Each point's row is the state at the end of its grid, reached from its t_start."""
    action = ('parameter_scan({parameter=>"k",par_scan_vals=>[0.1,0.3],method=>"ode",%s,'
              'suffix=>"s"})' % options)
    bng = _bng2pl_rows(tmp_path, [action], 'decay_s.scan')
    exact = 100 * np.exp(-np.array([0.1, 0.3]) * (span[1] - span[0]))
    np.testing.assert_allclose(bng[:, 1], exact, rtol=1e-5)
    out = _bridge(tmp_path, [action], ['s'], 's')
    np.testing.assert_allclose(out['Stot'], bng[:, 1], rtol=1e-5)


def test_a_chain_of_continued_sample_times_grids_keeps_bng2pls_clock(tmp_path):
    """Three actions: a late-starting grid (clock 5), a continued grid whose t_end is below its
    last listed time (clock 8), then a continued step grid over [8, 10]. BNG2.pl writes 9 and
    10 for the last; the bridge also writes the start row (a difference older than #872)."""
    actions = ['simulate({method=>"ode",sample_times=>[1,2,5],suffix=>"a"})',
               'simulate({continue=>1,method=>"ode",sample_times=>[6,7,9],t_end=>8,suffix=>"b"})',
               'simulate({continue=>1,method=>"ode",t_end=>10,n_steps=>2,suffix=>"x"})']
    bng = _bng2pl_rows(tmp_path, actions, 'decay_x.gdat')
    np.testing.assert_allclose(bng[:, 0], [9, 10], atol=1e-12)
    np.testing.assert_allclose(bng[:, 1], 100 * np.exp(-K * bng[:, 0]), rtol=1e-5)
    out = _bridge(tmp_path, actions, ['a', 'b', 'x'], 'x')
    np.testing.assert_allclose(_times(out), [8, 9, 10], atol=1e-12)
    np.testing.assert_allclose(out['Stot'][1:], bng[:, 1], rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# Findings of the review (strict xfail)
# ---------------------------------------------------------------------------------------------

_DECAY_FREE = _model_block(FIXTURES / 'e2e_ode_decay.bngl').replace('k   0.3', 'k   k__FREE')


def _legacy_set_up(tmp_path, actions, exp_text, backend='auto'):
    model = tmp_path / 'decay.bngl'
    model.write_text(_DECAY_FREE + 'begin actions\n' + ''.join('  %s\n' % a for a in actions)
                     + 'end actions\n')
    exp = tmp_path / 'x.exp'
    exp.write_text(exp_text)
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'bngl_backend': backend, 'wall_time_sim': 60,
        ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    return R.build(cfg, 'de')


def test_data_after_a_sample_times_grid_end_is_refused_at_set_up(tmp_path, monkeypatch):
    """BNG2.pl writes rows 0, 5 and 7 for this action (t_end below the last listed time), as
    the bridge now does, so the datum at t = 10 has no row on either backend. The bridge used
    to write a row at 10. The set-up knows both the rows and the data, so it should refuse."""
    R.require_bng2pl()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PybnfError):
        _legacy_set_up(tmp_path, ['generate_network({overwrite=>1})',
                                  'simulate({method=>"ode",t_end=>7,sample_times=>[0,5,10],'
                                  'suffix=>"x"})'],
                       '# time Stot\n5 22.313016\n10 4.978707\n')


def test_a_simulate_protocol_action_is_refused_where_bng2pl_drops_the_protocol(tmp_path,
                                                                              monkeypatch):
    """BNG2.pl's simulate_protocol action runs the protocol block, as a method=>"protocol"
    scan does. The bridge does not support the action, so auto hands the model to BNG2.pl
    from its .net file, which carries no protocol block, and BNG2.pl stops on every
    simulation ("Can't use an undefined value as an ARRAY reference at BNGAction.pm line
    36"). The set-up should refuse it as it refuses the protocol scan."""
    R.require_bng2pl()
    monkeypatch.chdir(tmp_path)
    model = tmp_path / 'decay.bngl'
    model.write_text(_DECAY_FREE + 'begin protocol\n  simulate({method=>"ode",t_end=>5,'
                     'n_steps=>5})\nend protocol\nbegin actions\n'
                     '  generate_network({overwrite=>1})\n'
                     '  simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"x"})\n'
                     '  simulate_protocol({})\nend actions\n')
    exp = tmp_path / 'x.exp'
    exp.write_text('# time Stot\n2 54.88\n4 30.12\n')
    cfg = config.Configuration({
        'population_size': 4, 'max_iterations': 1, 'output_dir': str(tmp_path / 'out') + '/',
        'models': {str(model)}, 'exp_data': {str(exp)}, str(model): [str(exp)],
        'fit_type': 'de', 'objfunc': 'sos', 'bngl_backend': 'auto', 'wall_time_sim': 60,
        ('uniform_var', 'k__FREE'): [0.1, 1.0]})
    with pytest.raises(PybnfError):
        R.build(cfg, 'de')


_INTEROP = FIXTURES.parent.parent / 'examples' / 'tutorial' / '11_interop'


@pytest.mark.parametrize('model_file, backend', [
    ('decay.bngl', 'bngsim'),
    ('decay.xml', 'bngsim'),
    ('decay.xml', 'roadrunner'),
])
def test_edition2_data_before_t_zero_is_refused_on_every_backend(tmp_path, monkeypatch,
                                                                 model_file, backend):
    """An edition-2 time course starts at t = 0 on every backend (``t=0`` is forced into its
    grid). #1005 refuses a datum before 0 for a BNGL model; the SBML backends still start the
    run at the earliest datum, so A(0) is reported as A0 exp(-2k) rather than A0."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / model_file).write_text((_INTEROP / model_file).read_text())
    (tmp_path / 'neg.exp').write_text('# time\tObs_A\n-2\t100\n1\t60.65\n2\t36.79\n5\t8.21\n')
    lines = ['edition = 2', 'model: %s' % model_file, 'output_dir = out',
             'job_type = de', 'objective = sos', 'bngl_backend = bngsim',
             'sbml_backend = %s' % backend, 'experiment: tc, data: neg.exp',
             'uniform_var = k 0.1 1', 'population_size = 4', 'max_iterations = 1']
    if model_file.endswith('.xml'):
        lines.append('observable: Obs_A, formula: A')
    with pytest.raises(PybnfError):
        cfg = config.Configuration(parse.ploop([ln + '\n' for ln in lines]))
        R.build(cfg, 'de')
