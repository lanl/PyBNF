"""The compiled ODE right-hand side is built once per model, onto the engine model (ADR-0148).

``BngsimModel`` used to call ``bngsim.prepare_codegen(net_path)`` at construction and on every
Dask unpickle, only as a flag, and then asked each per-evaluation ``Simulator`` for
``codegen=True, net_path=...``. Since lanl/bngsim#803 each such request recomputes bngsim's
structural cache key, about 90 ms per ``Simulator`` at 3,749 reactions and 1.4 s at 58,276,
and both calls are deprecated there. Now one ``Simulator`` on the engine model attaches the
artifact, every clone carries it, and a plain run inherits it. A sensitivity run still asks for
``codegen=True``, so a condition that overrides a derived parameter is never served a stale
sensitivity right-hand side (lanl/bngsim#708).

Oracles are closed forms: the decay fixture's Stot(t) = S0 exp(-kt), and for the derived
parameter model, dA/dk1 = 0 once k2 is pinned.
"""

import pickle
import shutil
import warnings

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf import pset
from pybnf.bngsim_model import _runtime

pytestmark = [
    pytest.mark.bngsim,
    pytest.mark.skipif(
        not (shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')),
        reason='codegen needs a C compiler on PATH',
    ),
]

DECAY_NET = '''# Created by BioNetGen 2.9.3
begin parameters
    1 S0    100  # Constant
    2 k     0.3  # Constant
end parameters
begin species
    1 S() S0
end species
begin reactions
    1 1 0 k #_R1
end reactions
begin groups
    1 Stot                 1
end groups
'''
DERIVED_NET = '''# Created by BioNetGen 2.9.3
begin parameters
    1 k1   0.3   # Constant
    2 k2   2*k1  # ConstantExpression
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 k2 #_R1
end reactions
begin groups
    1 Atot                 1
end groups
'''
ACTIONS = ['simulate({method=>"ode",t_start=>0,t_end=>2,n_steps=>4,suffix=>"tc"})']


def _model(tmp_path, text, name):
    net = tmp_path / name
    net.write_text(text)
    model = bngsim_model.BngsimModel(net.stem, list(ACTIONS), [('simulate', 'tc')], [],
                                     nf=str(net))
    model.param_set = pset.PSet([])
    return model


@pytest.fixture
def codegen_spy(monkeypatch):
    """Count every bngsim entry point that builds or looks up a codegen artifact."""
    import bngsim._codegen as cg

    calls = []
    for name in ('prepare_model_codegen', 'prepare_model_codegen_source', 'prepare_codegen'):
        real = getattr(cg, name, None)
        if real is None:
            continue

        def spy(*a, _real=real, _name=name, **k):
            calls.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(cg, name, spy)
    return calls


def test_a_plain_run_inherits_the_artifact_built_once(tmp_path, codegen_spy):
    model = _model(tmp_path, DECAY_NET, 'decay.net')
    assert model._codegen_so, 'the engine model should carry the artifact after construction'

    del codegen_spy[:]
    copy = model.copy_with_param_set(model.param_set)
    sim = _runtime.bngsim.Simulator(copy._engine_model, method='ode',
                                    **copy._codegen_kwargs('ode'),
                                    **copy._sensitivity_request_kwargs('ode'))
    assert sim.codegen_backend in ('cc', 'mir')
    assert codegen_spy == [], 'a plain run rebuilt or looked up its artifact: %s' % codegen_spy

    data = model.execute(str(tmp_path), 'plain', 60)['tc']
    t = data.data[:, data.cols['time']]
    np.testing.assert_allclose(data.data[:, data.cols['Stot']], 100.0 * np.exp(-0.3 * t),
                               rtol=1e-5, atol=1e-6)


def test_no_deprecated_bngsim_codegen_call_is_made(tmp_path, monkeypatch):
    """Neither ``prepare_codegen`` nor ``net_path=`` reaches bngsim, on any version of it."""
    import bngsim

    def refuse(*a, **k):
        raise AssertionError('bngsim.prepare_codegen was called')

    monkeypatch.setattr(bngsim, 'prepare_codegen', refuse, raising=False)
    real = _runtime.bngsim.Simulator

    def no_net_path(*a, **k):
        assert 'net_path' not in k, 'Simulator(net_path=...) was passed'
        return real(*a, **k)

    monkeypatch.setattr(_runtime.bngsim, 'Simulator', no_net_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        model = _model(tmp_path, DECAY_NET, 'decay.net')
        model.execute(str(tmp_path), 'plain', 60)


def test_the_artifact_is_rebuilt_on_unpickle(tmp_path):
    """A Dask worker unpickles the model and attaches its own artifact."""
    model = _model(tmp_path, DECAY_NET, 'decay.net')
    again = pickle.loads(pickle.dumps(model))
    assert again._codegen_so
    assert (getattr(again._engine_model, '_codegen_so_path', '')
            or getattr(again._engine_model, '_codegen_c_source', ''))


def _net_codegen_path_present():
    import bngsim._codegen as cg

    return hasattr(cg, 'generate_rhs_c')


@pytest.mark.xfail(
    _net_codegen_path_present(),
    strict=True,
    raises=AssertionError,
    reason='lanl/bngsim#694: this bngsim compiles .net models through its .net codegen '
           'path, whose sensitivity RHS keeps a pinned derived parameter\'s chain rule '
           'however it is rebuilt',
)
def test_a_sensitivity_run_is_not_served_a_stale_chain_rule(tmp_path):
    """k2 = 2*k1 drives A -> B. A probe attaches a sensitivity artifact to the base engine
    model, as ``analytic_sens_rhs_status`` does and a base condition's run does; then a
    clone pins k2, as a condition does. With k2 pinned, k1 reaches nothing: A = 10 exp(-5t)
    and dA/dk1 = 0. Inheriting the probe's artifact would keep dA/dk2 * dk2/dk1."""
    model = _model(tmp_path, DERIVED_NET, 'derived.net')
    model.enable_output_sensitivities(params=['k1'])
    model.analytic_sens_rhs_status()  # attaches a sensitivity artifact to the base engine

    clone = model._engine_model.clone()
    clone.set_param('k2', 5.0)
    sim = _runtime.bngsim.Simulator(clone, method='ode', **model._codegen_kwargs('ode'),
                                    **model._sensitivity_request_kwargs('ode'))
    t = np.linspace(0.0, 1.0, 5)
    r = sim.run(sample_times=list(t), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], 10.0 * np.exp(-5.0 * t),
                               rtol=1e-7)
    assert np.abs(np.asarray(r.sensitivities)[:, 0, 0]).max() < 1e-9
