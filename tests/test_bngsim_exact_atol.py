"""A tolerance PyBNF chose reaches CVODE as that number, on every bngsim build.

From bngsim 0.17.0 (lanl/bngsim#1420) ``Simulator.run`` reads a bare float ``atol`` as a
per-species *ceiling*: each species takes ``min(atol, rtol * max(|x_i|, floor))`` in the
state the call starts from, unless the run's sensitivities come from CVODES' difference
quotient. ``bngsim.ExactAtol(value)`` keeps the old reading. Every scalar PyBNF hands
bngsim is meant exactly -- ADR-0103's derived scalar, a stated ``sbml_atol``, a BNGL
action's ``atol=>`` as BioNetGen (and BNG2.pl, the sibling backend) reads it -- so each one
travels through :func:`pybnf._bngsim_caps.exact_atol`. Read as a ceiling it ignored the
number on a model whose species sit below one (``atol=>1e-8`` and ``1e-10`` gave the same
run), moved with the fit point's initial conditions, and differed between a gradient fit
and a scalar fit of the same model.

The SBML/Antimony half, with its closed-form oracle, lives beside the rest of the
tolerance tests in ``test_sbml_solver_tolerances.py``. This file holds the helper and the
``.net`` paths: the ordinary simulate action, the segment integrator, and the multiple
shooting ``.net`` backend.
"""

import types
from pathlib import Path

import numpy as np
import pytest

from pybnf import _bngsim_caps, bngsim_model, pset
from pybnf.bngsim_model import SegmentIntegrator
from pybnf.shooting.net_backend import NetSegmentBackend


# --- the helper ----------------------------------------------------------------- #

class _ExactAtol:
    def __init__(self, value):
        self.value = value


def test_a_build_with_exact_atol_gets_one():
    got = _bngsim_caps.exact_atol(1e-8, types.SimpleNamespace(ExactAtol=_ExactAtol))
    assert isinstance(got, _ExactAtol)
    assert got.value == 1e-8 and isinstance(got.value, float)


def test_a_build_without_it_gets_the_float_it_always_read_exactly():
    """bngsim 0.16.0, the floor: a float already meant exactly that number, so the run
    is unchanged there."""
    got = _bngsim_caps.exact_atol(1e-8, types.SimpleNamespace())
    assert got == 1e-8 and type(got) is float


def test_no_bngsim_at_all_gets_the_float(monkeypatch):
    """``module=None`` means the real bngsim, which may itself be absent."""
    monkeypatch.setattr(_bngsim_caps, 'bngsim', None)
    got = _bngsim_caps.exact_atol('1e-6')
    assert got == 1e-6 and type(got) is float


def test_the_flag_and_the_helper_agree_about_the_installed_build():
    got = _bngsim_caps.exact_atol(1e-8)
    if _bngsim_caps.BNGSIM_HAS_EXACT_ATOL:
        assert isinstance(got, _bngsim_caps.bngsim.ExactAtol)
    else:
        assert type(got) is float


# --- the .net paths, against the closed form ------------------------------------- #
# ``e2e_ode_decay.net`` is S -> 0 at rate k, i.e. S(t) = S0 exp(-k t). Seeded at 1e-8, the
# whole trajectory sits at or below a 1e-8 absolute tolerance, which is the shape whose
# accuracy says which tolerance was used. Worst relative error over t in [0, 20], measured:
#
#     atol=>   0.16.0 (float)   0.17.0 ExactAtol   0.17.0 bare float (ceiling)
#     1e-8     6.08e+01         6.08e+01           7.62e-06
#     1e-10    4.62e-01         4.62e-01           7.62e-06
#
# Exact, each stated number gives its own answer and the same one on both builds; as a
# ceiling both collapse onto one run that no stated tolerance would give.
_K, _S0 = 0.3, 1e-8
_T_END = 20.0


@pytest.fixture
def small_decay_net(tmp_path):
    text = (Path(__file__).resolve().parent / 'bngl_files' / 'e2e_ode_decay.net').read_text()
    assert 'S0    100' in text
    net = tmp_path / 'decay_small.net'
    net.write_text(text.replace('S0    100', 'S0    %r' % _S0))
    return net


def _action(atol):
    return ('simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>40,suffix=>"tc",'
            'atol=>%s})' % (_T_END, atol))


def _model(net, atol):
    model = bngsim_model.BngsimModel(net.stem, [_action(atol)], [('simulate', 'tc')], [],
                                     nf=str(net))
    model.param_set = pset.PSet([])
    return model


def _worst_error(t, got):
    exact = _S0 * np.exp(-_K * np.asarray(t))
    return float(np.max(np.abs(np.asarray(got) - exact) / exact))


@pytest.mark.bngsim
def test_a_bngl_action_atol_is_the_tolerance_bionetgen_means(small_decay_net, tmp_path):
    def worst(atol):
        data = _model(small_decay_net, atol).execute(str(tmp_path), 'x', None)['tc']
        return _worst_error(data['time'], data['Stot'])

    loose, tight = worst('1e-8'), worst('1e-10')
    # A ceiling-tightened run lands at ~8e-06 for both; each bound is decades above that.
    assert loose > 1.0, 'atol=>1e-8 was not integrated at 1e-8: %.2e off' % loose
    assert tight > 1e-2, 'atol=>1e-10 was not integrated at 1e-10: %.2e off' % tight
    assert tight < loose, (loose, tight)


@pytest.mark.bngsim
def test_a_segment_integrates_at_the_actions_own_atol(small_decay_net):
    """The segment integrator is the ordinary path's sibling, and starts every segment
    from a state of its own -- exactly where a ceiling read off the start state would hand
    two segments of one action two tolerances."""
    integrator = SegmentIntegrator(_model(small_decay_net, '1e-8'), [])
    grid = np.linspace(0.0, _T_END, 41)
    segment = integrator.integrate(integrator.initial_state([]), [], grid[0], grid[-1],
                                   times=grid[1:-1])
    rows = np.asarray(segment.data.data, dtype=float)
    stot = rows[:, segment.data.cols['Stot']]
    error = _worst_error(rows[:, 0], stot)
    assert error > 1.0, 'the segment did not integrate at atol=>1e-8: %.2e off' % error


def test_the_multiple_shooting_net_backend_states_its_atol_exactly(monkeypatch):
    fake = types.SimpleNamespace(ExactAtol=_ExactAtol)
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim', fake)
    owner = types.SimpleNamespace(sim_params={'atol': '1e-8', 'rtol': '1e-9'}, timeout=None)

    kwargs = NetSegmentBackend._run_kwargs(owner)

    assert isinstance(kwargs['atol'], _ExactAtol) and kwargs['atol'].value == 1e-8
    assert kwargs['rtol'] == 1e-9


def test_the_multiple_shooting_net_backend_on_a_build_without_it(monkeypatch):
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim', types.SimpleNamespace())
    owner = types.SimpleNamespace(sim_params={'atol': '1e-8'}, timeout=None)
    assert NetSegmentBackend._run_kwargs(owner) == {'atol': 1e-8}


# --- a value that is not a tolerance is refused, naming where it came from -------------- #

@pytest.mark.parametrize('value', [-1e-8, float('nan'), float('inf'), '-1e-8'])
def test_a_negative_or_non_finite_atol_is_refused(value):
    """Out of ``ExactAtol`` it was a raw ``ValueError`` that failed every evaluation as "an
    unknown error" on 0.17.0, and on 0.16.0 it ran. Refused on every build, by name."""
    from pybnf.printing import PybnfError
    with pytest.raises(PybnfError, match=r"the action X sets an absolute tolerance of"):
        _bngsim_caps.exact_atol(value, types.SimpleNamespace(ExactAtol=_ExactAtol),
                                where='the action X')
    with pytest.raises(PybnfError):
        _bngsim_caps.exact_atol(value, types.SimpleNamespace())


def test_zero_is_a_tolerance():
    """``atol = 0`` is pure relative error control, which CVODE accepts."""
    assert _bngsim_caps.exact_atol(0, types.SimpleNamespace()) == 0.0


@pytest.mark.bngsim
def test_a_bngl_action_with_a_negative_atol_ends_the_run(small_decay_net, tmp_path):
    from pybnf.printing import PybnfError
    with pytest.raises(PybnfError, match=r'absolute tolerance of -1e-08'):
        _model(small_decay_net, '-1e-8').execute(str(tmp_path), 'x', None)


def test_multiple_shooting_refuses_an_atol_that_does_not_evaluate(monkeypatch):
    """It used to drop such a value and run at bngsim's default."""
    from pybnf.printing import PybnfError
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim', types.SimpleNamespace())
    owner = types.SimpleNamespace(sim_params={'atol': '1e-8*oops'}, timeout=None)
    with pytest.raises(PybnfError, match=r'does not evaluate to a number'):
        NetSegmentBackend._run_kwargs(owner)


# --- no atol=> at all: BNG2.pl's default, exactly ------------------------------------- #
# bngsim 0.17.0 reads its own unstated 1e-8 as a ceiling off each run's start state. On the
# 1e-8 decay that is 7.6e-06 relative error where BNG2.pl's 1e-8 gives 6.08e+01 (BNG2.pl
# 2.9.3 measured S(20) = -1.481247240622e-09, the same as bngsim 0.16.0).

def _unstated(net):
    action = 'simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>40,suffix=>"tc"})' % _T_END
    model = bngsim_model.BngsimModel(net.stem, [action], [('simulate', 'tc')], [], nf=str(net))
    model.param_set = pset.PSet([])
    return model


@pytest.mark.bngsim
def test_a_bngl_action_with_no_atol_integrates_at_bng2pls_default(small_decay_net, tmp_path):
    data = _unstated(small_decay_net).execute(str(tmp_path), 'x', None)['tc']
    error = _worst_error(data['time'], data['Stot'])
    assert error > 1.0, 'an unstated atol was not BNG2.pl\'s exact 1e-8: %.2e off' % error


@pytest.mark.bngsim
def test_a_segment_with_no_atol_integrates_at_bng2pls_default(small_decay_net):
    integrator = SegmentIntegrator(_unstated(small_decay_net), [])
    grid = np.linspace(0.0, _T_END, 41)
    segment = integrator.integrate(integrator.initial_state([]), [], grid[0], grid[-1],
                                   times=grid[1:-1])
    rows = np.asarray(segment.data.data, dtype=float)
    error = _worst_error(rows[:, 0], rows[:, segment.data.cols['Stot']])
    assert error > 1.0, 'an unstated segment atol was not 1e-8: %.2e off' % error


def test_multiple_shooting_with_no_atol_states_bng2pls_default(monkeypatch):
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim',
                        types.SimpleNamespace(ExactAtol=_ExactAtol))
    kwargs = NetSegmentBackend._run_kwargs(types.SimpleNamespace(sim_params={}, timeout=None))
    assert isinstance(kwargs['atol'], _ExactAtol) and kwargs['atol'].value == 1e-8


def test_a_build_without_exact_atol_is_passed_nothing(monkeypatch):
    """bngsim 0.16.0's own default is already exactly 1e-8, so nothing changes there."""
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim', types.SimpleNamespace())
    assert NetSegmentBackend._run_kwargs(
        types.SimpleNamespace(sim_params={}, timeout=None)) == {}
    assert _bngsim_caps.bng2pl_default_atol(types.SimpleNamespace()) is None
