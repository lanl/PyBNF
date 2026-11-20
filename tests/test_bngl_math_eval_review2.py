"""Second-round review tests for #1051, written by a reviewer who did not write the change.

Every expected value comes from outside the code under test: a live BNG2.pl (the trajectory it
writes, the TEND it reports), run_network, bngsim reading a network file on its own, or a hand
calculation of exponential decay. Where BNG2.pl is needed only to confirm a hand value and is
not installed, that part is skipped.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf.bngsim_model.expressions import _bngl_value, _parse_net_species_initializers
from pybnf.printing import PybnfError
from pybnf.pset import BNGLModel, ModelError, PSet

from ._bngl_differential import _locate_bng2

_DECAY = """begin model
begin parameters
  k  0.1
  N0 100
end parameters
begin molecule types
  A(x,y)
  B()
end molecule types
begin seed species
  A(x,y) N0
  B() N0
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
"""

# The network BNG2.pl 2.9.3 writes for _DECAY, in its format.
_DECAY_NET = """begin parameters
    1 k   0.1  # Constant
    2 N0  100  # Constant
end parameters
begin species
    1 A(x,y) N0
    2 B() N0
end species
begin reactions
    1 1 0 k #_R1
    2 2 0 k #_R2
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""


def _bng2_gdat(tmp_path, actions):
    """The trajectory BNG2.pl writes for _DECAY run with ``actions``, or None without BNG2.pl."""
    bng2 = _locate_bng2()
    if bng2 is None:
        return None
    (tmp_path / 'oracle.bngl').write_text(
        _DECAY + 'generate_network({overwrite=>1})\n' + '\n'.join(actions) + '\n')
    subprocess.run(['perl', bng2, 'oracle.bngl'], cwd=tmp_path, check=True,
                   capture_output=True, timeout=120)
    return np.loadtxt(tmp_path / 'o.gdat', ndmin=2)


def _net_bridge(tmp_path, actions):
    net = tmp_path / 'decay.net'
    net.write_text(_DECAY_NET)
    model = bngsim_model.BngsimModel('decay', actions, [('simulate', 'o')], [], nf=str(net))
    model.param_set = PSet([])
    return model.execute(str(tmp_path), 'review2', 60)['o']


# ------------------------------------- setConcentration by a name BNG2.pl resolves
_SET_BY_NAME = ['setConcentration("A(y,x)", 50)',   # components in another order
                'setConcentration("B", 30)']          # a molecule written without ()


@pytest.mark.bngsim
def test_set_concentration_by_a_name_bng2pl_resolves_is_never_skipped(tmp_path):
    """Round 2 made a setConcentration bngsim cannot place an error. It used to be a warning,
    and the species kept its amount (100 here). Either is acceptable but that."""
    actions = _SET_BY_NAME + [
        'simulate({method=>"ode",t_start=>0,t_end=>1e-9,n_steps=>1,suffix=>"o"})']
    try:
        data = _net_bridge(tmp_path, actions)
    except PybnfError:
        return
    assert data.data[0, data.cols['Atot']] == pytest.approx(50.0)
    assert data.data[0, data.cols['Btot']] == pytest.approx(30.0)


@pytest.mark.bngsim
def test_set_concentration_by_a_name_bng2pl_resolves_matches_bng2pl(tmp_path):
    """BNG2.pl finds the species ``A(y,x)`` and ``B`` name, and starts them at 50 and 30.
    21 of the 132 species names in the corpus's setConcentration actions are of this kind
    (An_2009, the Mitra2019 JNK fits, RasSosGDP, egfr_path, ...), so those models stop on the
    .net bridge, although BNG2.pl runs them."""
    gdat = _bng2_gdat(tmp_path, _SET_BY_NAME + [
        'simulate({method=>"ode",t_end=>1e-9,n_steps=>1,prefix=>"o"})'])
    if gdat is not None:
        assert gdat[0, 1:] == pytest.approx([50.0, 30.0])
    # Adapted by the author after review: the bridge may also refuse such a model when it is
    # built, which is where bngl_backend = auto falls back to BNG2.pl (algorithms/base.py), so
    # the model runs there with BNG2.pl's own species matching. A failure while running is not
    # accepted.
    net = tmp_path / 'decay.net'
    net.write_text(_DECAY_NET)
    actions = _SET_BY_NAME + [
        'simulate({method=>"ode",t_start=>0,t_end=>1e-9,n_steps=>1,suffix=>"o"})']
    try:
        model = bngsim_model.BngsimModel('decay', actions, [('simulate', 'o')], [], nf=str(net))
    except PybnfError as exc:
        assert 'not in the network' in str(exc)
        return
    model.param_set = PSet([])
    data = model.execute(str(tmp_path), 'review2', 60)['o']
    assert data.data[0, data.cols['Atot']] == pytest.approx(50.0)
    assert data.data[0, data.cols['Btot']] == pytest.approx(30.0)


# ------------------------------------------------------- stop_if on the .net bridge
@pytest.mark.bngsim
@pytest.mark.parametrize('condition, threshold', [
    ('Atot<exp(3.45)', math.exp(3.45)),  # a BNGL function
    ('Atot<N0/3', 100.0 / 3),           # a model parameter
])
def test_stop_if_stops_where_bng2pl_stops_or_is_refused(tmp_path, condition, threshold):
    """By hand, Atot = 100 exp(-0.1 t) first falls below the threshold at the output row
    t = ceil(10 ln(100 / threshold)): 12 for exp(3.45) = 31.5 and 11 for N0/3. BNG2.pl stops
    there. The bridge hands the text to bngsim, whose stop condition is Python's eval over the
    observables alone: exp and N0 are unknown names there, the error is logged at debug level,
    and the run goes on to t_end with no error. Refusing such a condition would be acceptable."""
    expected = math.ceil(10 * math.log(100 / threshold))
    action = ('simulate({method=>"ode",t_end=>100,n_steps=>100,stop_if=>"%s",%s})'
              % (condition, '%s'))
    gdat = _bng2_gdat(tmp_path, [action % 'prefix=>"o"'])
    if gdat is not None:
        assert gdat[-1, 0] == expected
    path = tmp_path / 'm.bngl'
    path.write_text(_DECAY + 'begin actions\n' + action % 'suffix=>"o"' + '\nend actions\n')
    try:
        BNGLModel(str(path), suppress_free_param_error=True)
        data = _net_bridge(tmp_path, [action % 'suffix=>"o"'])
    except (ModelError, PybnfError):
        return
    assert data.data[-1, data.cols['time']] == expected


# ------------------------------------------- a quoted key in a t_start / t_end setting
@pytest.mark.parametrize('setting', ['"t_end"=>"2^3"', "'t_end'=>'2^3'"])
def test_a_quoted_t_end_key_is_read_as_bng2pl_reads_it(tmp_path, setting):
    """Perl reads ``"t_end"=>"2^3"`` as ``t_end=>"2^3"``, and BNG2.pl evaluates it: TEND:8,
    and its trajectory ends at t = 8. So the model should load."""
    action = 'simulate({method=>"ode",%s,n_steps=>2,%s})' % (setting, '%s')
    gdat = _bng2_gdat(tmp_path, [action % 'prefix=>"o"'])
    if gdat is not None:
        assert gdat[-1, 0] == 8.0
    path = tmp_path / 'm.bngl'
    path.write_text(_DECAY + 'begin actions\n' + action % 'suffix=>"o"' + '\nend actions\n')
    BNGLModel(str(path), suppress_free_param_error=True)


# ---------------------------------- the 'network' reading against bngsim and run_network
_NETWORK_AMOUNTS = ['10^-2', '(2^3)^2', '2^(3^2)', '(-2)^2', '-(2^2)+10', 'exp(ln(3))',
                    '10^-R*R', '1e2/R^2']


@pytest.mark.bngsim
def test_network_reading_matches_bngsim_and_run_network(tmp_path):
    """Every initial amount the 'network' reading accepts equals the amount bngsim reads from
    the same network file, and the amount run_network starts from."""
    import bngsim
    net = tmp_path / 'amounts.net'
    net.write_text(
        'begin parameters\n    1 R  2  # Constant\n    2 k  0.1  # Constant\nend parameters\n'
        'begin species\n'
        + ''.join('    %d S%d() %s\n' % (i + 1, i, e) for i, e in enumerate(_NETWORK_AMOUNTS))
        + 'end species\nbegin reactions\n    1 1 0 k #_R1\nend reactions\n'
        'begin groups\n    1 S0tot 1\nend groups\n')
    model = bngsim.Model.from_net(str(net))
    params = {name: model.get_param(name) for name in model.param_names}
    ours = [_bngl_value(expr, params, source='network')
            for _, expr in _parse_net_species_initializers(net.read_text().splitlines())]
    theirs = [model.get_concentration('S%d()' % i) for i in range(len(_NETWORK_AMOUNTS))]
    assert ours == pytest.approx(theirs, rel=1e-12)
    bng2 = _locate_bng2()
    if bng2 is not None:
        run_network = Path(bng2).parent / 'bin' / 'run_network'
        subprocess.run([str(run_network), '-o', 'rn', '-g', str(net), str(net), '1e-6', '1'],
                       cwd=tmp_path, check=True, capture_output=True, timeout=60)
        assert np.loadtxt(tmp_path / 'rn.cdat', ndmin=2)[0, 1:] == pytest.approx(ours, rel=1e-9)
