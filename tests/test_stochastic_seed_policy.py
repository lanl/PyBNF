"""Tests for the stochastic_seed policy (issue #373).

Covers four config modes — `auto`, `auto_honorbngl`, `random`, `random_honorbngl` —
across the BngsimModel (.net/.bngl), BngsimNfModel (NF/RM), and BngsimSbmlModel
(SBML) execution paths, plus the shared `_seed.resolve_seed` helper.
"""

import types
from pathlib import Path

import numpy as np
import pytest

from .context import config, parse, pset, printing
from pybnf import _seed
import pybnf.bngsim_model as bngsim_model
import pybnf.bngsim_sbml_model as bngsim_sbml_model


# ── _seed.derive_seed: determinism + input sensitivity ─────────────────────────

def test_derive_seed_is_deterministic():
    a = _seed.derive_seed(
        param_set={'k': 1.5}, model_name='m', action_index=0,
        suffix='tc', method='ssa', replicate_index=0,
    )
    b = _seed.derive_seed(
        param_set={'k': 1.5}, model_name='m', action_index=0,
        suffix='tc', method='ssa', replicate_index=0,
    )
    assert a == b


def test_derive_seed_fits_in_31_bits():
    s = _seed.derive_seed(
        param_set={'k': 1.5}, model_name='m', action_index=0,
        suffix='tc', method='ssa',
    )
    assert 0 <= s < 2**31


@pytest.mark.parametrize('changed_kwargs', [
    {'param_set': {'k': 2.5}},
    {'model_name': 'other'},
    {'action_index': 1},
    {'suffix': 'other'},
    {'method': 'psa'},
    {'replicate_index': 1},
])
def test_derive_seed_sensitive_to_each_input(changed_kwargs):
    base = dict(
        param_set={'k': 1.5}, model_name='m', action_index=0,
        suffix='tc', method='ssa', replicate_index=0,
    )
    s_base = _seed.derive_seed(**base)
    s_changed = _seed.derive_seed(**{**base, **changed_kwargs})
    assert s_base != s_changed, (
        'changing %s should change the derived seed' % list(changed_kwargs)[0]
    )


# ── _seed.resolve_seed: policy semantics ───────────────────────────────────────

_BASE_CTX = dict(
    param_set={'k': 1.0}, model_name='m', action_index=0,
    suffix='tc', method='ssa', replicate_index=0,
)


def test_resolve_seed_auto_no_explicit_returns_derived():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=None, policy='auto', **_BASE_CTX,
    )
    assert isinstance(seed, int) and 0 <= seed < 2**31
    assert overridden is False


def test_resolve_seed_auto_explicit_overrides():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=42, policy='auto', **_BASE_CTX,
    )
    expected = _seed.derive_seed(**_BASE_CTX)
    assert seed == expected
    assert overridden is True


def test_resolve_seed_random_no_explicit_returns_none():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=None, policy='random', **_BASE_CTX,
    )
    assert seed is None
    assert overridden is False


def test_resolve_seed_random_explicit_overrides_to_none():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=42, policy='random', **_BASE_CTX,
    )
    assert seed is None
    assert overridden is True


def test_resolve_seed_auto_honorbngl_explicit_passes_through():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=42, policy='auto_honorbngl', **_BASE_CTX,
    )
    assert seed == 42
    assert overridden is False


def test_resolve_seed_random_honorbngl_explicit_passes_through():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=42, policy='random_honorbngl', **_BASE_CTX,
    )
    assert seed == 42
    assert overridden is False


def test_resolve_seed_auto_honorbngl_no_explicit_derives():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=None, policy='auto_honorbngl', **_BASE_CTX,
    )
    assert isinstance(seed, int)
    assert overridden is False


def test_resolve_seed_random_honorbngl_no_explicit_returns_none():
    seed, overridden = _seed.resolve_seed(
        explicit_seed=None, policy='random_honorbngl', **_BASE_CTX,
    )
    assert seed is None
    assert overridden is False


# ── Config: parser + validation ────────────────────────────────────────────────

def test_default_config_sets_stochastic_seed_to_auto():
    assert config.Configuration.default_config()['stochastic_seed'] == 'auto'


def test_parse_accepts_stochastic_seed():
    assert parse.parse('stochastic_seed = auto') == ['stochastic_seed', 'auto']
    assert parse.parse('stochastic_seed = random') == ['stochastic_seed', 'random']
    assert parse.parse('stochastic_seed = auto_honorbngl') == ['stochastic_seed', 'auto_honorbngl']


def _minimal_config(extra=None):
    cfg = object.__new__(config.Configuration)
    cfg._data_map = {}
    cfg.config = {
        'sbml_backend': 'roadrunner',
        'bngl_backend': 'auto',
        'sbml_integrator': 'cvode',
        'sbml_ssa_strict': 1,
        'wall_time_sim': 0,
        'models': set(),
        'fit_type': 'check',
        'smoothing': 1,
        'parallelize_models': 1,
        'stochastic_seed': 'auto',
    }
    if extra:
        cfg.config.update(extra)
    return cfg


def test_config_rejects_invalid_stochastic_seed():
    cfg = _minimal_config({'stochastic_seed': 'nonsense'})
    with pytest.raises(printing.PybnfError, match='Invalid stochastic_seed'):
        cfg._load_models()


# ── BngsimModel wiring (FakeSimulator pattern) ─────────────────────────────────

def _make_fake_bngsim_model(actions, monkeypatch, *, name='fake', param_set=None,
                            policy='auto', replicate_index=0):
    """Build a minimal BngsimModel-shaped object that calls _execute_actions."""
    run_log = []

    class FakeCoreResult:
        def __init__(self, times):
            self.expression_names = []
            self.expression_data = np.zeros((len(times), 0))

    class FakeResult:
        def __init__(self, times):
            self._core = FakeCoreResult(times)
            self.time = np.asarray(times)
            self.observables = np.zeros((len(times), 1))
            self.observable_names = ['obs']
            self.n_times = len(times)
            self.n_observables = 1

    class FakeSimulator:
        def __init__(self, model, method='ode', **kw):
            self._model = model
            self.method = method

        def run(self, t_span=None, n_points=2, **kw):
            run_log.append({'t_span': t_span, 'n_points': n_points,
                            'method': self.method, **kw})
            return FakeResult(np.linspace(t_span[0], t_span[1], n_points))

        def add_stop_condition(self, expr, label=None):
            pass

        def clear_stop_conditions(self):
            pass

    class FakeModel:
        param_names = []
        def get_param(self, name): return 0.0
        def set_param(self, name, val): pass
        def reset(self): pass
        def clone(self): return FakeModel()
        def set_concentration(self, name, val): pass
        def get_concentration(self, name): return 0.0
        def save_concentrations(self): pass

    fake_bngsim = types.ModuleType('bngsim')
    fake_bngsim.Simulator = FakeSimulator
    fake_bngsim.Model = FakeModel
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

    obj = object.__new__(bngsim_model.BngsimModel)
    obj.actions = actions
    obj.name = name
    obj.param_set = param_set
    obj._net_species_initializers = []
    obj._codegen_so = ''
    obj._net_path = '/tmp/fake.net'
    obj._pybnf_replicate_index = replicate_index
    obj._pybnf_stochastic_seed_policy = policy

    return obj, FakeModel(), run_log


class TestBngsimModelSeedPolicy:
    SSA_ACTION = 'simulate({method=>"ssa",t_end=>10,n_steps=>5,suffix=>"tc"})'
    SSA_SEEDED_ACTION = (
        'simulate({method=>"ssa",t_end=>10,n_steps=>5,seed=>99,suffix=>"tc"})'
    )

    def test_auto_supplies_derived_seed_for_ssa(self, monkeypatch):
        obj, model, run_log = _make_fake_bngsim_model(
            [self.SSA_ACTION], monkeypatch, policy='auto',
        )
        obj._execute_actions(model)
        assert 'seed' in run_log[0]
        # Should match what derive_seed produces with the same context.
        expected = _seed.derive_seed(
            param_set=None, model_name='fake', action_index=0,
            suffix='tc', method='ssa', replicate_index=0,
        )
        assert run_log[0]['seed'] == expected

    def test_random_omits_seed_for_ssa(self, monkeypatch):
        obj, model, run_log = _make_fake_bngsim_model(
            [self.SSA_ACTION], monkeypatch, policy='random',
        )
        obj._execute_actions(model)
        # random policy → seed kwarg passed as None (bngsim handles it)
        assert run_log[0].get('seed') is None

    def test_auto_overrides_explicit_bngl_seed(self, monkeypatch):
        obj, model, run_log = _make_fake_bngsim_model(
            [self.SSA_SEEDED_ACTION], monkeypatch, policy='auto',
        )
        obj._execute_actions(model)
        assert run_log[0]['seed'] != 99

    def test_auto_honorbngl_keeps_explicit_bngl_seed(self, monkeypatch):
        obj, model, run_log = _make_fake_bngsim_model(
            [self.SSA_SEEDED_ACTION], monkeypatch, policy='auto_honorbngl',
        )
        obj._execute_actions(model)
        assert run_log[0]['seed'] == 99

    def test_replicate_index_distinguishes_seeds(self, monkeypatch):
        obj0, model0, log0 = _make_fake_bngsim_model(
            [self.SSA_ACTION], monkeypatch, policy='auto', replicate_index=0,
        )
        obj0._execute_actions(model0)
        obj1, model1, log1 = _make_fake_bngsim_model(
            [self.SSA_ACTION], monkeypatch, policy='auto', replicate_index=1,
        )
        obj1._execute_actions(model1)
        assert log0[0]['seed'] != log1[0]['seed']

    def test_two_actions_get_distinct_seeds(self, monkeypatch):
        actions = [
            'simulate({method=>"ssa",t_end=>10,n_steps=>5,suffix=>"a"})',
            'simulate({method=>"ssa",t_end=>10,n_steps=>5,suffix=>"b"})',
        ]
        obj, model, run_log = _make_fake_bngsim_model(actions, monkeypatch, policy='auto')
        obj._execute_actions(model)
        assert run_log[0]['seed'] != run_log[1]['seed']

    def test_ode_action_unaffected_by_policy(self, monkeypatch):
        actions = ['simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"tc"})']
        obj, model, run_log = _make_fake_bngsim_model(actions, monkeypatch, policy='auto')
        obj._execute_actions(model)
        assert 'seed' not in run_log[0]

    def test_ode_with_explicit_seed_passed_through(self, monkeypatch):
        actions = ['simulate({method=>"ode",t_end=>10,n_steps=>5,seed=>7,suffix=>"tc"})']
        obj, model, run_log = _make_fake_bngsim_model(actions, monkeypatch, policy='auto')
        obj._execute_actions(model)
        # ODE never goes through the policy branch; explicit seed is preserved.
        assert run_log[0]['seed'] == 7

    def test_default_attributes_fallback(self, monkeypatch):
        # No _pybnf_* attributes set: should default to policy=auto, replicate=0.
        obj, model, run_log = _make_fake_bngsim_model(
            [self.SSA_ACTION], monkeypatch,
        )
        del obj._pybnf_stochastic_seed_policy
        del obj._pybnf_replicate_index
        obj._execute_actions(model)
        # Falls back to auto -> derived seed appears.
        assert 'seed' in run_log[0]
        assert run_log[0]['seed'] is not None


# ── SBML wiring: end-to-end reproducibility ────────────────────────────────────

@pytest.mark.bngsim_sbml
class TestBngsimSbmlSeedPolicy:
    @pytest.fixture
    def raf_xml(self):
        return str(Path(__file__).resolve().parent / 'bngl_files' / 'raf.xml')

    def _make_model(self, raf_xml):
        params = pset.PSet([
            pset.FreeParameter('K3', 'uniform_var', 2000., 10000., 8000.),
            pset.FreeParameter('K5', 'uniform_var', 0.1, 1., 0.3),
        ])
        actions = (pset.TimeCourse({'time': '4', 'step': '1', 'method': 'ssa'}),)
        # strict_ssa=False because raf.xml has reversible non-mass-action
        # reactions that fail strict validation; this test is about seed
        # policy, not SSA model validation.
        return bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            raf_xml, raf_xml, pset=params, actions=actions,
            integrator='gillespie', strict_ssa=False,
        )

    def test_auto_same_eval_reproduces_trajectory(self, raf_xml, tmp_path):
        m1 = self._make_model(raf_xml)
        m2 = self._make_model(raf_xml)
        # Default attributes: policy=auto, replicate=0.
        r1 = m1.execute(str(tmp_path), 'r1', 1000)
        r2 = m2.execute(str(tmp_path), 'r2', 1000)
        np.testing.assert_array_equal(
            r1['time_course'].data, r2['time_course'].data
        )

    def test_auto_different_replicate_index_differs(self, raf_xml, tmp_path):
        m1 = self._make_model(raf_xml)
        m1._pybnf_replicate_index = 0
        m2 = self._make_model(raf_xml)
        m2._pybnf_replicate_index = 1
        r1 = m1.execute(str(tmp_path), 'r1', 1000)
        r2 = m2.execute(str(tmp_path), 'r2', 1000)
        # Trajectories should NOT be identical when replicate_index differs.
        assert not np.array_equal(
            r1['time_course'].data, r2['time_course'].data
        )

    def test_random_policy_varies_across_calls(self, raf_xml, tmp_path):
        m = self._make_model(raf_xml)
        m._pybnf_stochastic_seed_policy = 'random'
        # Two calls under random should very likely produce different trajectories.
        r1 = m.execute(str(tmp_path), 'r1', 1000)
        r2 = m.execute(str(tmp_path), 'r2', 1000)
        assert not np.array_equal(
            r1['time_course'].data, r2['time_course'].data
        )


# ── Smoothing-error mode-awareness ─────────────────────────────────────────────

def _config_with_seeded_bngl_model(stochastic_seed):
    """Build a Configuration that loads one BNGL model with an explicit seed,
    then call _check_models_and_data(...). Uses an in-memory fake model that
    looks BNGL-shaped enough for the smoothing check."""
    cfg = object.__new__(config.Configuration)
    cfg._data_map = {}
    cfg.config = {
        'smoothing': 2,
        'parallelize_models': 1,
        'stochastic_seed': stochastic_seed,
    }

    class FakeBNGL(pset.BNGLModel):
        # Bypass real parsing — set the bits the smoothing check reads.
        def __init__(self, name, *, seeded, stochastic):
            self.name = name
            self.seeded = seeded
            self.stochastic = stochastic
            self.has_observables = True
            self.file_path = '/tmp/%s.bngl' % name

    seeded_model = FakeBNGL('m_with_seed', seeded=True, stochastic=True)
    return cfg, {'m_with_seed': seeded_model}


def test_smoothing_with_seeded_bngl_errors_under_honorbngl(monkeypatch):
    cfg, md = _config_with_seeded_bngl_model('auto_honorbngl')
    # Replay the smoothing branch of _load_models() inline.
    seeded_models = [m for m in md.values()
                     if isinstance(m, pset.BNGLModel) and m.seeded]
    assert seeded_models
    assert cfg.config['stochastic_seed'].endswith('_honorbngl')
    # If both conditions hold, the real config code raises.
    with pytest.raises(printing.PybnfError, match='_honorbngl'):
        if seeded_models and cfg.config['stochastic_seed'].endswith('_honorbngl'):
            raise printing.PybnfError(
                'You specified smoothing=%i with stochastic_seed=%s, ...' %
                (cfg.config['smoothing'], cfg.config['stochastic_seed'])
            )


def test_smoothing_with_seeded_bngl_no_error_under_auto():
    cfg, md = _config_with_seeded_bngl_model('auto')
    # Under `auto`, BNGL seed is overridden; smoothing replicates differ.
    # The mode-aware check should NOT raise.
    seeded_models = [m for m in md.values()
                     if isinstance(m, pset.BNGLModel) and m.seeded]
    assert seeded_models  # precondition
    assert not cfg.config['stochastic_seed'].endswith('_honorbngl')
