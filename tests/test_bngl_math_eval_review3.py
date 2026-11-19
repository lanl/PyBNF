"""Third-round review tests for #1051, written by a reviewer who did not write the change.

Round 3 refuses, when the bngsim bridge is built, a setConcentration whose species bngsim cannot
find, so that ``bngl_backend = auto`` falls back to BNG2.pl. These tests drive real PyBNF runs
(``python -m pybnf``) and check the result against a hand calculation of exponential decay, and
check the bridge against BNG2.pl's own trajectories. They need BNG2.pl (BNGPATH) and bngsim.
"""

import math
import subprocess
import sys

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf.printing import PybnfError
from pybnf.pset import PSet

from ._bngl_differential import _locate_bng2

pytestmark = pytest.mark.bngsim

_MODEL = """begin model
begin parameters
  k  k__FREE
end parameters
begin molecule types
  A(x,y)
  B()
end molecule types
begin seed species
  A(x,y) 100
  B() 100
end seed species
begin observables
  Molecules Atot A()
  Molecules Btot B()
end observables
begin reaction rules
  A() -> 0 k
  B() -> 0 k
end reaction rules
end model
begin actions
  generate_network({overwrite=>1})
%s
  simulate({method=>"ode",t_end=>10,n_steps=>10,suffix=>"tc"})
end actions
"""

# BNG2.pl finds A(y,x) as A(x,y) and B as B(); bngsim needs the names as the .net writes them.
_SET_BY_NAME = '  setConcentration("A(y,x)", 50)\n  setConcentration("B", 30)'
# The same action split over two lines, and no other action the bridge would refuse.
_SET_CONTINUED = '  setConcentration(\\\n    "A(y,x)", 50)'


def _need_bng2():
    if _locate_bng2() is None:
        pytest.skip('BNG2.pl not found (set BNGPATH)')


def _run_pybnf(tmp_path, actions, backend, b0=30):
    """Fit k to data made by hand from A(0) = 50, B(0) = b0; return (exit code, output, results)."""
    (tmp_path / 'm.bngl').write_text(_MODEL % actions)
    rows = ['# time Atot Btot'] + ['%d %r %r' % (t, 50 * math.exp(-0.1 * t), b0 * math.exp(-0.1 * t))
                                   for t in range(11)]
    (tmp_path / 'tc.exp').write_text('\n'.join(rows) + '\n')
    (tmp_path / 'fit.conf').write_text(
        'output_dir = out\nmodel = m.bngl : tc.exp\nfit_type = de\nobjfunc = sos\n'
        'uniform_var = k__FREE 0.01 1\npopulation_size = 6\nmax_iterations = 2\n'
        'bngl_backend = %s\nverbosity = 1\nparallel_count = 2\n' % backend)
    proc = subprocess.run([sys.executable, '-m', 'pybnf', '-c', 'fit.conf', '-o'], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr, tmp_path / 'out' / 'Results'


def _hand_sos(k, a0, b0, data_b0=30):
    """The sum of squares PyBNF should report at rate k, from A(0) = a0 and B(0) = b0."""
    return sum((50 * math.exp(-0.1 * t) - a0 * math.exp(-k * t)) ** 2
               + (data_b0 * math.exp(-0.1 * t) - b0 * math.exp(-k * t)) ** 2 for t in range(11))


def _best(results):
    line = (results / 'sorted_params_final.txt').read_text().splitlines()[1].split()
    return float(line[1]), float(line[2])  # objective, k


@pytest.mark.parametrize('actions, b0', [(_SET_BY_NAME, 30), (_SET_CONTINUED, 100)],
                         ids=['one line', 'continued line'])
def test_auto_falls_back_to_bng2pl_and_reports_its_objective(tmp_path, actions, b0):
    """Under bngl_backend = auto the model runs, and the objective PyBNF reports at its best k is
    the one computed by hand from the amounts BNG2.pl starts from (A 50, and B 30 or 100). Had
    setConcentration("A(y,x)", 50) been skipped, A would start at 100 and the objective would be
    over 2500 at any k. The continued line is the same action written over two lines."""
    _need_bng2()
    code, output, results = _run_pybnf(tmp_path, actions, 'auto', b0=b0)
    assert code == 0, output[-2000:]
    objective, k = _best(results)
    assert objective == pytest.approx(_hand_sos(k, 50, b0, b0), rel=1e-5)
    assert objective < _hand_sos(k, 100, b0, b0)


def test_explicit_bngsim_stops_and_names_the_species(tmp_path):
    _need_bng2()
    code, output, _ = _run_pybnf(tmp_path, _SET_BY_NAME, 'bngsim')
    assert code != 0
    assert 'A(y,x)' in output and 'not in the network' in output


# ------------------------------------------------ a fixed species, named as the .net names it
_FIXED_NET = """begin parameters
    1 k  0.1  # Constant
end parameters
begin species
    1 $S() 10
    2 A(x,y) 100
end species
begin reactions
    1 2 0 k #_R1
    2 1 1,2 k #_R2
end reactions
begin groups
    1 Atot 2
    2 Stot 1
end groups
"""


def test_a_fixed_species_named_as_the_network_names_it_is_set(tmp_path):
    """The network file names the fixed species ``$S()``, BNG2.pl accepts
    ``setConcentration("$S()", 20)`` and starts S at 20, and bngsim's set_concentration accepts
    the name too. (bngsim lists the species as ``S()``.)"""
    bng2 = _locate_bng2()
    if bng2 is not None:
        (tmp_path / 'fx.bngl').write_text(
            'begin model\nbegin parameters\n  k 0.1\nend parameters\n'
            'begin molecule types\n  A(x,y)\n  S()\nend molecule types\n'
            'begin seed species\n  $S() 10\n  A(x,y) 100\nend seed species\n'
            'begin observables\n  Molecules Atot A()\n  Molecules Stot S()\nend observables\n'
            'begin reaction rules\n  A() -> 0 k\n  S() -> S() + A(x,y) k\nend reaction rules\n'
            'end model\ngenerate_network({overwrite=>1})\nsetConcentration("$S()", 20)\n'
            'simulate({method=>"ode",t_end=>1e-9,n_steps=>1,prefix=>"fs"})\n')
        subprocess.run(['perl', bng2, 'fx.bngl'], cwd=tmp_path, check=True, capture_output=True,
                       timeout=120)
        assert np.loadtxt(tmp_path / 'fs.gdat', ndmin=2)[0, 2] == 20.0
    net = tmp_path / 'fixed.net'
    net.write_text(_FIXED_NET)
    actions = ['setConcentration("$S()", 20)',
               'simulate({method=>"ode",t_start=>0,t_end=>1e-9,n_steps=>1,suffix=>"fs"})']
    model = bngsim_model.BngsimModel('fixed', actions, [('simulate', 'fs')], [], nf=str(net))
    model.param_set = PSet([])
    data = model.execute(str(tmp_path), 'fixed', 60)['fs']
    assert data.data[0, data.cols['Stot']] == pytest.approx(20.0)


# ------------------------------------------ a quoted hash key, which Perl reads as unquoted
_DECAY_NET = """begin parameters
    1 k   0.1  # Constant
end parameters
begin species
    1 A() 100
end species
begin reactions
    1 1 0 k #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize('settings, last_time, rows', [
    ('t_end=>8,"n_steps"=>4', 8.0, 5),                         # BNG2.pl: 4 steps
    ('t_end=>100,n_steps=>100,"stop_if"=>"Atot<31.5"', 12.0, 13),  # by hand: ceil(10 ln(100/31.5))
])
def test_a_quoted_key_is_read_as_bng2pl_reads_it_or_refused(tmp_path, settings, last_time, rows):
    """Perl reads ``"n_steps"=>4`` as ``n_steps=>4``, and BNG2.pl writes 4 steps; it stops a
    ``"stop_if"=>"Atot<31.5"`` run at t = 12. The bridge keeps the quotes in the key, so it does
    not see either setting: it writes 1 step, and runs to t = 100 with no error. Round 3 accepts
    quoted t_start and t_end keys when the model loads. A refusal would also be acceptable."""
    net = tmp_path / 'decay.net'
    net.write_text(_DECAY_NET)
    action = 'simulate({method=>"ode",%s,suffix=>"o"})' % settings
    try:
        model = bngsim_model.BngsimModel('decay', [action], [('simulate', 'o')], [], nf=str(net))
        model.param_set = PSet([])
        data = model.execute(str(tmp_path), 'qkey', 60)['o']
    except PybnfError:
        return
    times = data.data[:, data.cols['time']]
    assert (times[-1], len(times)) == (last_time, rows)
