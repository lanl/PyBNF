"""Fourth-round review test for #1051, written by a reviewer who did not write the change.

Round 4 accepts a setConcentration that names a fixed species as the network file writes it
(``$S()``), which bngsim lists as ``S()``. The test sets a fixed source through a quoted
expression over a parameter that an earlier setParameter changed, as a real model does
(``setConcentration("$EGF(r)","egf_tot")``), and checks the bridge against the analytic
solution and, where BNG2.pl is installed, against BNG2.pl's own trajectory.
"""

import math
import subprocess

import numpy as np
import pytest

from pybnf import bngsim_model
from pybnf.pset import PSet

from ._bngl_differential import _locate_bng2

pytestmark = pytest.mark.bngsim

# S is a fixed source: S -> S + A at kp, and A -> 0 at kd. With S held at s and A(0) = 0,
# A(t) = (kp s / kd) (1 - exp(-kd t)).
_NET = """begin parameters
    1 kp    0.1  # Constant
    2 kd    0.5  # Constant
    3 dose  1  # Constant
end parameters
begin species
    1 $S() 10
    2 A() 0
end species
begin reactions
    1 1 1,2 kp #_R1
    2 2 0 kd #_R2
end reactions
begin groups
    1 Atot 2
    2 Stot 1
end groups
"""

_ACTIONS = ['setParameter("dose", 2)',
            'setConcentration("$S()", "dose*10")',
            'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"src"})']


def test_a_fixed_source_set_by_its_network_name_matches_the_analytic_solution(tmp_path):
    s, kp, kd = 20.0, 0.1, 0.5
    times = np.arange(5.0)
    analytic = kp * s / kd * (1 - np.exp(-kd * times))

    bng2 = _locate_bng2()
    if bng2 is not None:
        (tmp_path / 'src.bngl').write_text(
            'begin model\nbegin parameters\n  kp 0.1\n  kd 0.5\n  dose 1\nend parameters\n'
            'begin molecule types\n  S()\n  A()\nend molecule types\n'
            'begin seed species\n  $S() 10\n  A() 0\nend seed species\n'
            'begin observables\n  Molecules Atot A()\n  Molecules Stot S()\nend observables\n'
            'begin reaction rules\n  S() -> S() + A() kp\n  A() -> 0 kd\nend reaction rules\n'
            'end model\ngenerate_network({overwrite=>1})\n'
            + '\n'.join(_ACTIONS).replace('suffix=>"src"', 'prefix=>"bng"') + '\n')
        subprocess.run(['perl', bng2, 'src.bngl'], cwd=tmp_path, check=True,
                       capture_output=True, timeout=120)
        gdat = np.loadtxt(tmp_path / 'bng.gdat', ndmin=2)
        assert gdat[:, 1] == pytest.approx(analytic, rel=1e-6, abs=1e-9)
        assert gdat[:, 2] == pytest.approx(np.full(5, s))

    net = tmp_path / 'source.net'
    net.write_text(_NET)
    model = bngsim_model.BngsimModel('source', _ACTIONS, [('simulate', 'src')], [], nf=str(net))
    model.param_set = PSet([])
    data = model.execute(str(tmp_path), 'source', 60)['src']
    assert data.data[:, data.cols['time']] == pytest.approx(times)
    assert data.data[:, data.cols['Stot']] == pytest.approx(np.full(5, s))
    assert data.data[:, data.cols['Atot']] == pytest.approx(analytic, rel=1e-6, abs=1e-9)
    assert math.isclose(analytic[-1], 4 * (1 - math.exp(-2)))
