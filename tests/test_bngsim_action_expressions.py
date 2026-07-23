"""BNGL action arguments accept arithmetic expressions, not just bare numbers.

``simulate(...)`` / ``parameter_scan(...)`` action arguments are parsed as raw
strings (``_parse_action_value`` preserves scalars as text), so a value written
as ``t_end=>2*5`` reaches the action handlers as the string ``"2*5"``. The
handlers evaluate every numeric argument through ``_eval_numeric`` -- the same
no-builtins safe evaluator ``setParameter``/``setConcentration`` already use --
so an arithmetic expression resolves exactly the way its literal equivalent
would. These tests pin that behaviour: the expression form must reproduce the
literal form. (The prior bare-``float()``/``int()`` parsing raised ``ValueError``
on ``"2*5"``.)
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf import pset


pytestmark = pytest.mark.bngsim


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'


def _suffixes_for(actions):
    """Extract the ``('simulate', <suffix>)`` pairs from action lines."""
    pairs = []
    for a in actions:
        i = a.find('suffix=>"')
        if i == -1:
            continue
        j = a.find('"', i + len('suffix=>"'))
        pairs.append(('simulate', a[i + len('suffix=>"'):j]))
    return pairs


def _decay_model(actions):
    """A BngsimModel over the committed exponential-decay .net fixture."""
    net_path = FIXTURES / 'e2e_ode_decay.net'
    model = bngsim_model.BngsimModel(
        net_path.stem, list(actions), _suffixes_for(actions), [], nf=str(net_path),
    )
    model.param_set = pset.PSet([])
    return model


def test_simulate_action_args_accept_expressions(tmp_path):
    """``t_end``/``n_steps`` written as arithmetic expressions (``2*5``, ``4*5``)
    produce the identical run to their literal equivalents (``10``, ``20``).

    Two fresh models start from identical initial conditions, so any difference
    would be the expression parsing -- and on the pre-change ``float()`` path the
    expression form raised ``ValueError`` outright.
    """
    literal = _decay_model(
        ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>20,suffix=>"tc"})'],
    ).execute(str(tmp_path), 'decay_literal', 60)['tc']
    expr = _decay_model(
        ['simulate({method=>"ode",t_start=>0,t_end=>2*5,n_steps=>4*5,suffix=>"tc"})'],
    ).execute(str(tmp_path), 'decay_expr', 60)['tc']

    lit_t = literal.data[:, literal.cols['time']]
    exp_t = expr.data[:, expr.cols['time']]
    assert exp_t[-1] == pytest.approx(10.0)        # t_end=>2*5
    assert len(exp_t) == len(lit_t)                # n_steps=>4*5
    np.testing.assert_allclose(exp_t, lit_t)

    lit_s = literal.data[:, literal.cols['Stot']]
    exp_s = expr.data[:, expr.cols['Stot']]
    np.testing.assert_allclose(exp_s, lit_s)


def test_parameter_scan_action_args_accept_expressions():
    """The parameter_scan/bifurcate handler evaluates its numeric args through
    ``_eval_numeric`` as well (``t_start``/``t_end``/``print_functions``/
    ``reset_conc``/``seed``). Resolving settings from expression-valued args must
    land each on the expected number; bare ``float()``/``int()`` would raise on
    ``"1+1"``.
    """
    model = _decay_model([])
    ps_params = {
        'parameter': 'k',
        'par_scan_vals': ['1', '2', '3'],
        't_start': '1+1',          # -> 2.0
        't_end': '2*5',            # -> 10.0
        'print_functions': '1*1',  # -> True
        'reset_conc': '0+1',       # -> True
        'seed': '40+2',            # -> 42
        'suffix': 'scan',
    }
    settings = model._resolve_scan_settings(ps_params, False, 0, 60, None)

    assert settings.t_start == pytest.approx(2.0)
    assert settings.t_end == pytest.approx(10.0)
    assert settings.print_funcs is True
    assert settings.reset_conc is True
    assert settings.scan_seed == 42
