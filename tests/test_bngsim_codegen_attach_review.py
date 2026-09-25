"""Independent review of ADR-0148 (one codegen build per model, inherited by every clone).

Written by the reviewer, not the author. The author's tests use a one-reaction decay and a
one-reaction derived-parameter model, and read the right-hand side through ``Simulator``
directly. These go through the PyBNF evaluation path itself (``copy_with_param_set`` then
``execute`` with conditions), on a model whose right-hand side reads derived parameters inside
Functional rate laws and a time-dependent function, and check every number against a closed
system integrated by scipy, which shares no code with bngsim.

The model (``RICH_NET``, written by BNG2.pl 2.9.3 from a BNGL file) has four species:

    A -> B   at  fMM = Vmax / (Km + Atot)          (Vmax = 10*k1, derived)
    B -> C   at  fT  = kdeg * exp(-t / tau)        (kdeg = k2/3, derived; time-dependent)
    C -> D   at  k2                                (k2 = 2*k1, derived)
    0 -> A   at  kin

Conditions pin derived parameters (``k2 = 5``, ``Vmax = 1``) and change an initial
concentration (``A0 = 20``): the cases where one artifact shared by every clone could, in
principle, describe a different model from the clone it runs.
"""

import copy
import pickle
import shutil

import numpy as np
import pytest

from pybnf import bngsim_model, pset
from pybnf.bngsim_model import _runtime

scipy_integrate = pytest.importorskip('scipy.integrate')

pytestmark = [
    pytest.mark.bngsim,
    pytest.mark.skipif(
        not (shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')),
        reason='codegen needs a C compiler on PATH',
    ),
]

RICH_NET = '''# Created by BioNetGen 2.9.3
begin parameters
    1 k1      0.3  # Constant
    2 k2      2*k1  # ConstantExpression
    3 Km      5  # Constant
    4 Vmax    10*k1  # ConstantExpression
    5 kdeg    k2/3  # ConstantExpression
    6 A0      10  # Constant
    7 C0      1  # Constant
    8 tau     0.5  # Constant
    9 kin     0.2  # Constant
end parameters
begin functions
    1 fMM() Vmax/(Km+Atot)
    2 fT() kdeg*exp(((-time())/tau))
    3 fRatio() (Btot/(Ctot+1))+k2
end functions
begin species
    1 A() A0
    2 B() 0
    3 C() C0
    4 D() 0
end species
begin reactions
    1 1 2 fMM #R1
    2 2 3 fT #R2
    3 3 4 k2 #R3
    4 0 1 kin #R4
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
    3 Ctot                 3
    4 Dtot                 4
end groups
'''
ACTIONS = [('simulate({method=>"ode",t_start=>0,t_end=>3,n_steps=>12,suffix=>"tc",'
            'atol=>1e-12,rtol=>1e-10,print_functions=>1})')]
T = np.linspace(0.0, 3.0, 13)
OBS = ('Atot', 'Btot', 'Ctot', 'Dtot')
K1, TAU = 0.7, 1.3
CONDITIONS = {
    '': {},
    '_pk2': {'k2': 5.0},     # pins a derived parameter; kdeg = k2/3 follows it
    '_pV': {'Vmax': 1.0},    # pins a derived parameter read inside a Functional rate law
    '_A0': {'A0': 20.0},     # an initial concentration, re-derived from its seed parameter
}


def _closed_system(k1, tau, k2=None, Vmax=None, A0=10.0):
    """The model's ODEs written out by hand and integrated by scipy.

    An override of ``k2`` is a pin: ``kdeg = k2/3`` follows it, and ``k1`` no longer reaches
    ``k2`` (BioNetGen's setParameter semantics).
    """
    k2 = 2.0 * k1 if k2 is None else k2
    Vmax = 10.0 * k1 if Vmax is None else Vmax
    kdeg, Km, kin = k2 / 3.0, 5.0, 0.2

    def rhs(t, y):
        A, B, C, _ = y
        fMM = Vmax / (Km + A)
        fT = kdeg * np.exp(-t / tau)
        return [kin - fMM * A, fMM * A - fT * B, fT * B - k2 * C, k2 * C]

    sol = scipy_integrate.solve_ivp(rhs, (0.0, T[-1]), [A0, 0.0, 1.0, 0.0], t_eval=T,
                                    method='LSODA', rtol=1e-12, atol=1e-14)
    assert sol.success
    return sol.y.T, k2


def _fd_sensitivities(overrides, h=1e-5):
    """d(observables)/d(k1, tau) by central differences of the closed system."""
    def at(k1, tau):
        return _closed_system(k1, tau, **overrides)[0]

    dk1 = (at(K1 * (1 + h), TAU) - at(K1 * (1 - h), TAU)) / (2 * K1 * h)
    dtau = (at(K1, TAU * (1 + h)) - at(K1, TAU * (1 - h))) / (2 * TAU * h)
    return np.stack([dk1, dtau], axis=-1)


@pytest.fixture
def codegen_env(monkeypatch):
    for name in ('PYBNF_NO_CODEGEN', 'BNGSIM_NO_CODEGEN', 'BNGSIM_CODEGEN_JIT'):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _model(tmp_path):
    net = tmp_path / 'rich.net'
    net.write_text(RICH_NET)
    mutants = [pset.MutationSet([pset.Mutation(name, '=', value)], suffix=suffix)
               for suffix, over in CONDITIONS.items() if suffix
               for name, value in over.items()]
    model = bngsim_model.BngsimModel('rich', list(ACTIONS), [('simulate', 'tc')], mutants,
                                     nf=str(net))
    model.param_set = pset.PSet([
        pset.FreeParameter('k1', 'uniform_var', 0.0, 10.0, value=K1),
        pset.FreeParameter('tau', 'uniform_var', 0.0, 10.0, value=TAU),
    ])
    return model


def _artifact(engine):
    return (getattr(engine, '_codegen_so_path', '') or getattr(engine, '_codegen_c_source', ''))


def _evaluate(model, tmp_path, name):
    return model.copy_with_param_set(model.param_set).execute(str(tmp_path), name, 60)


def _assert_values_match_the_closed_system(ds):
    for suffix, over in CONDITIONS.items():
        data = ds['tc' + suffix]
        expected, k2 = _closed_system(K1, TAU, **over)
        got = np.column_stack([data.data[:, data.cols[o]] for o in OBS])
        label = suffix or 'base'
        np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9,
                                   err_msg=f'condition {label!r}')
        ratio = expected[:, 1] / (expected[:, 2] + 1.0) + k2
        np.testing.assert_allclose(data.data[:, data.cols['fRatio']], ratio, rtol=1e-7,
                                   err_msg=f'fRatio under condition {label!r}')


def test_every_condition_runs_on_the_inherited_artifact_and_gets_the_right_numbers(
        tmp_path, codegen_env):
    """Each condition's clone runs on the base model's one artifact, and each one's
    trajectory and function output match the closed system under that condition."""
    model = _model(tmp_path)
    built = _artifact(model._engine_model)
    assert built, 'the base engine model should carry the artifact after construction'

    real = _runtime.bngsim.Simulator
    seen = []

    def spy(engine, **kwargs):
        sim = real(engine, **kwargs)
        seen.append((kwargs.get('codegen'), sim._codegen_so_path or sim._codegen_c_source))
        return sim

    codegen_env.setattr(_runtime.bngsim, 'Simulator', spy)
    for rep in range(2):  # a second evaluation reuses what the first left behind
        _assert_values_match_the_closed_system(_evaluate(model, tmp_path, f'rep{rep}'))
    assert seen, 'no Simulator was built'
    assert all(codegen is None for codegen, _ in seen), seen
    assert all(artifact == built for _, artifact in seen), (
        f'a condition ran on something other than the inherited artifact: {seen}')


def test_plain_runs_after_a_sensitivity_probe_on_the_base_engine(tmp_path, codegen_env):
    """``analytic_sens_rhs_status`` builds a sensitivity Simulator on the BASE engine model,
    which replaces the artifact every later clone inherits. A plain evaluation afterwards,
    under every condition, must still match the closed system."""
    if not _runtime.BNGSIM_HAS_OUTPUT_SENS:
        pytest.skip('this bngsim has no forward output sensitivities')
    model = _model(tmp_path)
    model.enable_output_sensitivities(params=['k1', 'tau'])
    model.analytic_sens_rhs_status()
    model._sensitivity_request = None   # back to the scalar path
    _assert_values_match_the_closed_system(_evaluate(model, tmp_path, 'after_probe'))


def _net_codegen_path_present():
    # Evaluated at collection, so it must not raise where bngsim is absent; the
    # bngsim marker skips these tests there.
    try:
        import bngsim._codegen as cg
    except ImportError:
        return False
    return hasattr(cg, 'generate_rhs_c')


def _assert_gradient_matches_finite_differences(ds):
    for suffix, over in CONDITIONS.items():
        sens = ds['tc' + suffix].output_sensitivities
        rows = [list(sens.selectors).index('observable:' + o) for o in OBS]
        cols = [list(sens.param_names).index(p) for p in ('k1', 'tau')]
        got = np.asarray(sens.d_param)[:, rows, :][:, :, cols]
        expected = _fd_sensitivities(over)
        scale = np.abs(expected).max()
        worst = np.abs(got - expected).max()
        assert worst <= 1e-6 * scale, (
            f"condition {suffix or 'base'!r}: max |d/dtheta error| = {worst:.3g} "
            f'of a {scale:.3g} scale')


_GRADIENT_XFAIL = pytest.mark.xfail(
    _net_codegen_path_present(),
    strict=True,
    raises=AssertionError,
    reason="lanl/bngsim#694: this bngsim compiles .net models through its .net codegen path, "
           "whose sensitivity RHS keeps a pinned derived parameter's chain rule",
)


@_GRADIENT_XFAIL
def test_gradient_under_conditions_that_pin_derived_parameters(tmp_path, codegen_env):
    """d/dk1 must follow each condition's own dependency graph: with k2 pinned, k1 reaches
    only Vmax; with Vmax pinned, only k2 and kdeg. A probe first, as a gradient fit does."""
    if not _runtime.BNGSIM_HAS_OUTPUT_SENS:
        pytest.skip('this bngsim has no forward output sensitivities')
    model = _model(tmp_path)
    model.enable_output_sensitivities(params=['k1', 'tau'])
    model.analytic_sens_rhs_status()
    for rep in range(2):
        _assert_gradient_matches_finite_differences(_evaluate(model, tmp_path, f'g{rep}'))


@_GRADIENT_XFAIL
def test_gradient_under_pybnf_no_codegen_is_not_served_another_conditions_chain_rule(
        tmp_path, codegen_env):
    """PYBNF_NO_CODEGEN disables the plain build, but a sensitivity run is compiled anyway.
    The base condition's sensitivity Simulator attaches its artifact to the evaluation's
    engine model, and each condition clones that engine model. If the condition's own
    sensitivity run does not ask bngsim to build against the model as it stands, it inherits
    a chain rule in which k1 still reaches the pinned k2 (lanl/bngsim#708)."""
    if not _runtime.BNGSIM_HAS_OUTPUT_SENS:
        pytest.skip('this bngsim has no forward output sensitivities')
    codegen_env.setenv('PYBNF_NO_CODEGEN', '1')
    model = _model(tmp_path)
    assert not model._codegen_so
    model.enable_output_sensitivities(params=['k1', 'tau'])
    _assert_gradient_matches_finite_differences(_evaluate(model, tmp_path, 'nocg'))


def test_copies_do_not_share_an_engine_model_they_mutate(tmp_path, codegen_env):
    """``__copy__`` gives the copy its own clone of the engine model, so no evaluation may
    leave a trace on the base model or on another copy, and a mutant's engine model is its
    own."""
    model = _model(tmp_path)
    base_engine = model._engine_model
    base_k1 = base_engine.get_param('k1')
    assert copy.copy(model)._engine_model is not base_engine

    first = model.copy_with_param_set(model.param_set)
    other = model.copy_with_param_set(pset.PSet([
        pset.FreeParameter('k1', 'uniform_var', 0.0, 10.0, value=2.5),
        pset.FreeParameter('tau', 'uniform_var', 0.0, 10.0, value=TAU),
    ]))
    engines = {id(base_engine), id(first._engine_model), id(other._engine_model)}
    assert len(engines) == 3
    other.execute(str(tmp_path), 'other', 60, with_mutants=False)
    assert base_engine.get_param('k1') == base_k1
    assert first._engine_model.get_param('k1') == base_k1

    mutant = first._get_mutant_model_bngsim(first.mutants[0])
    assert mutant._engine_model is not first._engine_model
    assert _artifact(mutant._engine_model) == _artifact(base_engine)

    # The unchanged first copy still evaluates to the closed system, conditions included.
    _assert_values_match_the_closed_system(first.execute(str(tmp_path), 'first', 60))


def test_deepcopy_and_pickle_still_rebuild_an_independent_engine(tmp_path, codegen_env):
    """``__copy__`` must not leak into ``copy.deepcopy`` or pickling (Dask), which rebuild
    the engine model from the .net file and attach their own artifact."""
    model = _model(tmp_path)
    for again in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert again._engine_model is not model._engine_model
        assert again._codegen_so and _artifact(again._engine_model)
        _assert_values_match_the_closed_system(_evaluate(again, tmp_path, 'again'))
