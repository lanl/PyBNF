"""Review tests for #1051 (BNGL arithmetic on the bngsim bridges), written by a reviewer who
did not write the change.

Every expected value comes from outside the code under test: a live BNG2.pl (the XML and the
network file it writes, the TEND it reports, the trajectory run_network computes), bngsim
reading a network file on its own, or a hand calculation. Where BNG2.pl is needed and not
installed, the test skips.
"""

import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf.bngsim_model import expressions
from pybnf.printing import PybnfError
from pybnf.pset import BNGLModel, FailedSimulationError, ModelError, PSet

from ._bngl_differential import _locate_bng2

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'
DECAY_NET = FIXTURES / 'e2e_ode_decay.net'      # S() S0 = 100, S -> 0 at k = 0.3
CARET_BNGL = FIXTURES / 'bngl_math_caret.bngl'  # A() N0 = 1000
CARET_XML = FIXTURES / 'bngl_math_caret.xml'

_DECAY_HEAD = """begin model
begin parameters
  k  0.1
  N0 100
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() N0
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() -> 0 k
end reaction rules
end model
"""


def _bng2_or_skip():
    bng2 = _locate_bng2()
    if bng2 is None:
        pytest.skip('BNG2.pl not found (set BNGPATH)')
    return bng2


def _run_bng2(bng2, folder, name, text):
    (folder / name).write_text(text)
    return subprocess.run(['perl', bng2, name], cwd=folder, check=True, capture_output=True,
                          text=True, timeout=120).stdout


def _run_net_bridge(net_path, actions, suffix, folder):
    model = bngsim_model.BngsimModel(
        Path(net_path).stem, actions, [('simulate', suffix)], [], nf=str(net_path))
    model.param_set = PSet([])
    return model.execute(str(folder), 'review', 60)[suffix]


# ---------------------------------------------------- the network-free parameters block
_FRESH_PARAMETERS = [
    ('p', '3'),
    ('c_frac_exp', '10^-3.5'),
    ('c_right', '2^(3^2)'),
    ('c_left', '(2^3)^2'),
    ('c_neg_power', '-(2^2)'),
    ('c_neg_base', '(-2)^3'),
    ('c_ln', 'exp(ln(10)*2)'),
    ('c_sci_base', '1e-3^2'),
    ('c_ref', '10^p/2'),
    ('c_neg_exp_ref', '2^-p'),
    ('c_rint_neg', 'rint(-2.5)'),
    ('c_if', 'if(p>2, 10^p, 0)'),
    ('c_fold', 'p^0.5^2'),
    ('c_max', 'max(10^-1, 2^-3)'),
    ('c_mixed', '3-10^2*2'),
]


def test_network_free_parameters_match_bng2pl_on_expressions_the_author_did_not_use(tmp_path):
    """Each value the network-free bridge would push into NFsim equals the value BNG2.pl
    writes in its XML for the same parameters block."""
    bng2 = _bng2_or_skip()
    block = '\n'.join('  %s %s' % pair for pair in _FRESH_PARAMETERS)
    text = _DECAY_HEAD.replace('  k  0.1\n', '  k  0.1\n' + block + '\n') + 'writeXML()\n'
    _run_bng2(bng2, tmp_path, 'fresh.bngl', text)
    root = ET.parse(tmp_path / 'fresh.xml').getroot()
    theirs = {p.get('id'): float(p.get('value')) for p in root.iter()
              if p.tag.endswith('Parameter') and p.get('id')}

    ours = expressions._evaluate_bngl_params(
        expressions._parse_bngl_param_block(text.splitlines()))

    for name, _ in _FRESH_PARAMETERS:
        assert ours[name] == pytest.approx(theirs[name], rel=1e-6), name
    # And by hand, for the ones where Python and BNGL part ways.
    assert ours['c_right'] == 512.0 and ours['c_left'] == 64.0
    assert ours['c_fold'] == pytest.approx(3.0)  # (3^0.5)^2, not 3^(0.5^2) = 1.316
    assert ours['c_neg_base'] == -8.0 and ours['c_rint_neg'] == -2.0


# --------------------------------------------- initial amounts read from a network file
@pytest.mark.bngsim
def test_net_initial_amount_is_read_as_the_simulators_of_that_file_read_it(tmp_path):
    """A network file is read by run_network and bngsim, whose ``^`` groups from the right:
    ``2^3^2`` is 2^9 = 512 to both. The bridge re-derives the amount with BNGL's left fold
    (64) and writes that over bngsim's own 512. A refusal would also be acceptable."""
    net = tmp_path / 'chain.net'
    net.write_text(
        'begin parameters\n    1 k  0.1  # Constant\nend parameters\n'
        'begin species\n    1 A() 2^3^2\nend species\n'
        'begin reactions\n    1 1 0 k #_R1\nend reactions\n'
        'begin groups\n    1 Atot  1\nend groups\n')
    import bngsim
    assert bngsim.Model.from_net(str(net)).get_concentration('A()') == 2.0 ** 9
    bng2 = _locate_bng2()
    if bng2 is not None:
        run_network = Path(bng2).parent / 'bin' / 'run_network'
        subprocess.run([str(run_network), '-o', 'rn', '-g', str(net), str(net), '0.001', '1'],
                       cwd=tmp_path, check=True, capture_output=True, timeout=60)
        assert np.loadtxt(tmp_path / 'rn.cdat', ndmin=2)[0, 1] == 2.0 ** 9

    try:
        data = _run_net_bridge(
            net, ['simulate({method=>"ode",t_start=>0,t_end=>1e-9,n_steps=>1,suffix=>"tc"})'],
            'tc', tmp_path)
    except PybnfError:
        return
    assert data.data[0, data.cols['Atot']] == pytest.approx(2.0 ** 9)


# ------------------------------- quoted settings that BNG2.pl does not evaluate as BNGL
def test_bng2pl_reads_a_quoted_power_in_n_steps_and_atol_as_a_plain_number(tmp_path):
    """The premise of the load-time refusal is that a quoted argument is safe because BNG2.pl
    evaluates it as BNGL. That holds for t_end, but BNG2.pl hands n_steps, atol and rtol to
    run_network as text, where "2^3" is 2 and "10^-10" is 10."""
    bng2 = _bng2_or_skip()
    _run_bng2(bng2, tmp_path, 'quoted.bngl', _DECAY_HEAD +
              'generate_network({overwrite=>1})\n'
              'simulate({method=>"ode",t_end=>"2^3",n_steps=>"2^3",prefix=>"steps"})\n'
              'resetConcentrations()\n'
              'simulate({method=>"ode",t_end=>8,n_steps=>1,atol=>"10^-10",rtol=>"10^-10",'
              'prefix=>"tol"})\n')
    steps = np.loadtxt(tmp_path / 'steps.gdat', ndmin=2)
    assert steps[-1, 0] == 8.0          # t_end "2^3" is 8: evaluated
    assert len(steps) == 3              # n_steps "2^3" is 2, not 8: not evaluated
    exact = 100 * math.exp(-0.1 * 8)
    assert abs(np.loadtxt(tmp_path / 'tol.gdat', ndmin=2)[-1, 1] - exact) > 0.1 * exact


@pytest.mark.parametrize('settings', [
    'n_steps=>"2^3"',
    'n_steps=>1,atol=>"10^-10"',
    'n_steps=>1,rtol=>"10^-10"',
])
def test_a_quoted_power_in_a_setting_bng2pl_does_not_evaluate_is_refused(tmp_path, settings):
    """Such a model runs on BNG2.pl with a number nobody wrote (2 steps, atol = 10; see the
    test above), and on the .net bridge with a different one (8 steps, atol = 1e-10)."""
    path = tmp_path / 'm.bngl'
    path.write_text(_DECAY_HEAD + 'begin actions\n'
                    'simulate({method=>"ode",t_end=>8,%s})\nend actions\n' % settings)
    with pytest.raises((ModelError, PybnfError)):
        BNGLModel(str(path), suppress_free_param_error=True)


# --------------------------------------------- a quoted setConcentration on the .net bridge
@pytest.mark.bngsim
@pytest.mark.parametrize('expr, expected', [
    ('"-k^2+50"', (-0.3) ** 2 + 50),  # BNG2.pl writes ((-k)^2)+50 to the network file
    ('"1/(k-0.3)"', None),            # undefined at k = 0.3: a failed simulation
])
def test_net_bridge_set_concentration_is_applied_or_refused_never_skipped(
        tmp_path, expr, expected):
    actions = ['setConcentration("S()", %s)' % expr,
               'simulate({method=>"ode",t_start=>0,t_end=>1e-9,n_steps=>1,suffix=>"tc"})']
    try:
        data = _run_net_bridge(DECAY_NET, actions, 'tc', tmp_path)
    except (PybnfError, FailedSimulationError):
        return
    start = data.data[0, data.cols['Stot']]
    assert expected is not None, 'the action was skipped; S starts at %g' % start
    assert start == pytest.approx(expected)


# ------------------------------------- a quoted setConcentration on the network-free bridge
@pytest.mark.bngsim
@pytest.mark.bngsim_nfsim
def test_network_free_set_concentration_with_spaces_in_the_expression(tmp_path):
    """``"2 * 10^3"`` is 2000 molecules by hand. The bridge's own test writes ``"2*10^3"``."""
    actions = ['setConcentration("A()", "2 * 10^3")',
               'simulate({method=>"nf",t_start=>0,t_end=>1e-6,n_steps=>1,suffix=>"tc"})']
    model = bngsim_model.BngsimNfModel(
        'bngl_math_caret', actions, [('simulate', 'tc')], [], str(CARET_XML),
        bngl_model_lines=CARET_BNGL.read_text().splitlines(keepends=True), param_names=())
    data = model.execute(str(tmp_path), 'spaces', 60)['tc']
    assert data.data[0, data.cols['Atot']] == 2000


# ------------------------------------- a minus sign before a named power has one reading
def test_bionetgen_reads_a_minus_before_a_named_power_one_way(tmp_path):
    """BNG2.pl writes ``-a^2`` to the network file as ``(-a)^2``, so its own arithmetic,
    run_network and bngsim all read 4. Only a minus on a number literal (``-2^2``) is kept
    as written and read two ways."""
    bng2 = _bng2_or_skip()
    text = _DECAY_HEAD.replace('  k  0.1\n', '  k  0.1\n  a  2\n  s  -a^2\n')
    _run_bng2(bng2, tmp_path, 'named.bngl', text + 'generate_network({overwrite=>1})\n')
    net = (tmp_path / 'named.net').read_text()
    assert re.search(r'^\s*\d+\s+s\s+\(-a\)\^2\s', net, re.M)

    ours = expressions._evaluate_bngl_params(
        expressions._parse_bngl_param_block(text.splitlines()))
    assert ours['s'] == 4.0


# --------------------------------------- an unquoted Perl call BNG2.pl and main evaluated
@pytest.mark.bngsim
def test_unquoted_perl_log_in_t_end_matches_bng2pl(tmp_path):
    """BNG2.pl runs an unquoted argument as Perl, where log is the natural log: TEND is
    ln(100) = 4.60517 (checked with BNG2.pl 2.9.3). The bridge on main read it the same way.
    A refusal when the model loads would also be acceptable; a refusal per simulation is not."""
    actions = ['simulate({method=>"ode",t_start=>0,t_end=>log(100),n_steps=>1,suffix=>"tc"})']
    try:
        data = _run_net_bridge(DECAY_NET, actions, 'tc', tmp_path)
    except PybnfError:
        return
    assert data.data[-1, data.cols['time']] == pytest.approx(math.log(100))
