import os
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from .context import algorithms, pset, printing
import pybnf.bngsim_model as bngsim_model


def _write_tfun_model(model_path, method='ode', force_generate_network=False):
    action_lines = []
    if method not in ('nf', 'nf_reject', 'nfsim', 'rm', 'rulemonkey', 'nf_exact') or force_generate_network:
        action_lines.append('    generate_network({overwrite=>1})')
    action_lines.append(
        '    simulate({method=>"%s",t_end=>4,n_steps=>40,suffix=>"tc"})' % method
    )

    model_path.write_text(
        """begin model

begin parameters
end parameters

begin molecule types
    X()
end molecule types

begin seed species
    X() 0
end seed species

begin observables
    Molecules Xtot X()
end observables

begin functions
    f_time() = tfun('test_data.tfun', time)
end functions

begin reaction rules
    0 -> X() f_time()
end reaction rules

end model

begin actions
%s
end actions
""" % '\n'.join(action_lines)
    )


def _make_tfun_bngl_model(tmp_path, method='ode', force_generate_network=False):
    source_dir = tmp_path / 'source_model'
    source_dir.mkdir(exist_ok=True)
    tfun_file = source_dir / 'test_data.tfun'
    tfun_file.write_text(
        "# time f_time\n"
        "0 0\n"
        "1 1\n"
    )
    model_path = source_dir / 'bridge_test.bngl'
    _write_tfun_model(model_path, method=method, force_generate_network=force_generate_network)
    model = pset.BNGLModel(
        str(model_path),
        suppress_free_param_error=True,
    )
    model.bng_command = '/fake/BNG2.pl'
    return model


def _extract_staged_tfun_path(text):
    match = re.search(
        r"tfun\(['\"](?P<path>__pybnf_tfun__/[^'\"]+_test_data\.tfun)['\"]",
        text,
    )
    assert match is not None
    return match.group('path')


def _write_dummy_net(path):
    path.write_text(
        "begin parameters\n"
        "end parameters\n"
        "begin functions\n"
        "  1 f_time() tfun('test_data.tfun', time)\n"
        "end functions\n"
    )


def _make_dummy_algorithm(model, output_dir, bngl_backend='auto'):
    class DummyConfig(object):
        pass

    cfg = DummyConfig()
    cfg.models = {model.name: model}
    cfg.config = {
        'output_dir': str(output_dir),
        'bng_command': '/fake/BNG2.pl',
        'wall_time_gen': 10,
        'bngl_backend': bngl_backend,
    }

    algo = object.__new__(algorithms.Algorithm)
    algo.config = cfg
    algo.variables = []
    return algo


def _make_fake_normalize_method(*, has_nfsim=True, has_rulemonkey=True):
    """Build a normalize_method() stub for a fake bngsim ModuleType.

    Mirrors the contract of bngsim.normalize_method(): returns a
    (canonical, dispatch) pair for supported tokens, raises ValueError
    for unknown tokens or for NF tokens whose backend isn't built.
    """
    nf_aliases = {'nf', 'nf_reject', 'nfsim'}
    rm_aliases = {'rm', 'rulemonkey', 'nf_exact'}

    def normalize_method(requested):
        lower = requested.strip().lower()
        if lower in ('ode', 'ssa', 'psa'):
            return lower, lower
        if lower in nf_aliases:
            if not has_nfsim:
                raise ValueError("method='nf_reject' is recognized but NFsim is not present")
            return 'nf_reject', 'nfsim'
        if lower in rm_aliases:
            if not has_rulemonkey:
                raise ValueError("method='nf_exact' is recognized but RuleMonkey is not present")
            return 'nf_exact', 'rulemonkey'
        raise ValueError("Unknown method '%s'" % requested)

    return normalize_method


def _fake_network_generation(cmd, timeout, stdout=None, stderr=None, input=None):
    del timeout, stdout, stderr, input
    bngl_path = Path(os.getcwd()) / cmd[1]
    _write_dummy_net(bngl_path.with_suffix('.net'))


def _fake_xml_generation(cmd, timeout, stdout=None, stderr=None, input=None):
    del timeout, stdout, stderr, input
    bngl_path = Path(os.getcwd()) / cmd[1]
    _fake_xml_generation.last_bngl_path = bngl_path
    _fake_xml_generation.last_bngl_text = bngl_path.read_text()
    bngl_path.with_suffix('.xml').write_text('<bngxml/>')


def _fake_hybrid_generation(cmd, timeout, stdout=None, stderr=None, input=None):
    """Fake that handles both the network gen call and the XML gen call."""
    del timeout, stdout, stderr, input
    bngl_path = Path(os.getcwd()) / cmd[1]
    bngl_text = bngl_path.read_text()
    _fake_hybrid_generation.last_bngl_path = bngl_path
    _fake_hybrid_generation.last_bngl_text = bngl_text
    if 'writeXML' in bngl_text:
        bngl_path.with_suffix('.xml').write_text('<bngxml/>')
    else:
        _write_dummy_net(bngl_path.with_suffix('.net'))


def _make_free_param(name, value):
    return pset.FreeParameter(
        name,
        'uniform_var',
        -np.inf,
        np.inf,
        value=value,
        bounded=True,
    )


def _install_fake_nfsim(monkeypatch):
    calls = []

    class FakeResult(object):
        def __init__(self, times, obs_value):
            self.time = np.asarray(times, dtype=float)
            self.observable_names = ['L_total']
            self.observables = np.full((len(times), 1), obs_value, dtype=float)
            self.expression_names = []
            self.expressions = np.zeros((len(times), 0), dtype=float)
            self.n_times = len(times)
            self.n_observables = 1

    class FakeNfsimSession(object):
        def __init__(self, xml_path, *, molecule_limit=None):
            self.xml_path = xml_path
            self.params = {}
            self.molecules = {}
            calls.append(('create', xml_path))
            if molecule_limit is not None:
                calls.append(('gml', molecule_limit))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.destroy()

        def clear_param_overrides(self):
            self.params = {}
            calls.append(('clear',))

        def set_param(self, name, value):
            self.params[name] = float(value)
            calls.append(('param', name, float(value)))

        def get_parameter(self, name):
            return self.params[name]

        def initialize(self, seed):
            calls.append(('init', seed, dict(self.params), dict(self.molecules)))

        def simulate(self, t_start, t_end, n_points, *, timeout=None):
            calls.append(('simulate', t_start, t_end, n_points, dict(self.params), dict(self.molecules)))
            return FakeResult(np.linspace(t_start, t_end, n_points), self.molecules.get('L', 0))

        def get_molecule_count(self, mol_type):
            return self.molecules.get(mol_type, 0)

        def add_molecules(self, mol_type, amount):
            self.molecules[mol_type] = self.molecules.get(mol_type, 0) + amount
            calls.append(('add', mol_type, amount, self.molecules[mol_type]))

        def destroy(self):
            calls.append(('destroy', dict(self.params), dict(self.molecules)))

    fake_pkg = types.ModuleType('bngsim')
    fake_pkg.NfsimSession = FakeNfsimSession
    fake_pkg.SimulationTimeout = _FakeSimulationTimeout
    fake_pkg.normalize_method = _make_fake_normalize_method(has_nfsim=True, has_rulemonkey=False)

    monkeypatch.setitem(sys.modules, 'bngsim', fake_pkg)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_pkg)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM', True)
    return calls


def _install_fake_nf_sessions(monkeypatch, *, has_nfsim=True, has_rulemonkey=True):
    calls = []

    class FakeResult(object):
        def __init__(self, times, obs_value):
            self.time = np.asarray(times, dtype=float)
            self.observable_names = ['L_total']
            self.observables = np.full((len(times), 1), obs_value, dtype=float)
            self.expression_names = []
            self.expressions = np.zeros((len(times), 0), dtype=float)
            self.n_times = len(times)
            self.n_observables = 1

    class FakeSession(object):
        backend = None

        def __init__(self, xml_path, *, molecule_limit=None):
            self.xml_path = xml_path
            self.params = {}
            self.molecules = {}
            calls.append(('create', self.backend, xml_path))
            if molecule_limit is not None:
                calls.append(('gml', self.backend, molecule_limit))

        def clear_param_overrides(self):
            self.params = {}
            calls.append(('clear', self.backend))

        def set_param(self, name, value):
            self.params[name] = float(value)
            calls.append(('param', self.backend, name, float(value)))

        def get_parameter(self, name):
            return self.params[name]

        def initialize(self, seed):
            calls.append(('init', self.backend, seed, dict(self.params), dict(self.molecules)))

        def simulate(self, t_start, t_end, n_points, *, timeout=None):
            calls.append(
                ('simulate', self.backend, t_start, t_end, n_points, dict(self.params), dict(self.molecules))
            )
            return FakeResult(np.linspace(t_start, t_end, n_points), self.molecules.get('L', 0))

        def get_molecule_count(self, mol_type):
            return self.molecules.get(mol_type, 0)

        def add_molecules(self, mol_type, amount):
            self.molecules[mol_type] = self.molecules.get(mol_type, 0) + amount
            calls.append(('add', self.backend, mol_type, amount, self.molecules[mol_type]))

        def destroy(self):
            calls.append(('destroy', self.backend, dict(self.params), dict(self.molecules)))

    class FakeNfsimSession(FakeSession):
        backend = 'nfsim'

    class FakeRuleMonkeySession(FakeSession):
        backend = 'rulemonkey'

    fake_pkg = types.ModuleType('bngsim')
    if has_nfsim:
        fake_pkg.NfsimSession = FakeNfsimSession
    if has_rulemonkey:
        fake_pkg.RuleMonkeySession = FakeRuleMonkeySession
    fake_pkg.SimulationTimeout = _FakeSimulationTimeout
    fake_pkg.normalize_method = _make_fake_normalize_method(
        has_nfsim=has_nfsim, has_rulemonkey=has_rulemonkey,
    )

    monkeypatch.setitem(sys.modules, 'bngsim', fake_pkg)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_pkg)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM', has_nfsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_RULEMONKEY', has_rulemonkey)
    return calls


def test_bngl_save_stages_relative_tfun_files(tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    out_prefix = tmp_path / 'saved' / 'bridge_test_gen'
    out_prefix.parent.mkdir()

    model.save(
        str(out_prefix),
        gen_only=True,
        pset=pset.PSet([]),
    )

    saved_text = Path(str(out_prefix) + '.bngl').read_text()
    staged_rel = _extract_staged_tfun_path(saved_text)

    assert "tfun('test_data.tfun'" not in saved_text
    assert (out_prefix.parent / staged_rel).is_file()


def test_netmodel_save_stages_relative_tfun_files(tmp_path):
    source_dir = tmp_path / 'net_source'
    source_dir.mkdir()
    (source_dir / 'test_data.tfun').write_text("# time f_time\n0 0\n1 1\n")
    net_path = source_dir / 'bridge_test.net'
    _write_dummy_net(net_path)

    model = pset.NetModel('bridge_test', [], [], [], nf=str(net_path))
    out_prefix = tmp_path / 'saved_net' / 'bridge_test'
    out_prefix.parent.mkdir()

    model.save(str(out_prefix))

    saved_text = Path(str(out_prefix) + '.net').read_text()
    staged_rel = _extract_staged_tfun_path(saved_text)

    assert "tfun('test_data.tfun'" not in saved_text
    assert (out_prefix.parent / staged_rel).is_file()


def test_actions_compatible_with_bngsim_rejects_pla():
    assert not bngsim_model.actions_compatible_with_bngsim([
        'simulate({method=>"pla",t_end=>4,n_steps=>40,suffix=>"tc"})'
    ])


def test_bngsim_version_compatibility_bounds():
    from pybnf import _bngsim_caps
    assert _bngsim_caps._version_compatible('0.5.0')
    assert _bngsim_caps._version_compatible('0.9.1')
    assert not _bngsim_caps._version_compatible('0.4.9')
    assert not _bngsim_caps._version_compatible('1.0.0')


_rulemonkey_required = pytest.mark.skipif(
    not bngsim_model.BNGSIM_HAS_RULEMONKEY,
    reason='bngsim was built without vendored RuleMonkey',
)


def _stub_normalize_method_without_rulemonkey(method):
    """Stand-in for bngsim.normalize_method when RuleMonkey is missing.

    Mirrors bngsim's ValueError shape for rulemonkey-family inputs so the
    PyBNF wrapper exercises the missing-vendored-backend code path even
    on installs where RuleMonkey is actually built.
    """
    lower = method.strip().lower()
    if lower in ('rm', 'rulemonkey', 'nf_exact'):
        raise ValueError(
            "method='nf_exact' (exact non-local network-free) is recognized "
            "but RuleMonkey is not present in this bngsim install."
        )
    return bngsim_model.bngsim.normalize_method(method)


def test_classify_actions_for_bngsim_routes_nf_aliases():
    for method in ('nf', 'nf_reject', 'nfsim'):
        assert bngsim_model.classify_actions_for_bngsim([
            'simulate({method=>"%s",t_end=>4,n_steps=>40,suffix=>"tc"})' % method
        ]) == bngsim_model.BNGSIM_BACKEND_NF


@_rulemonkey_required
def test_classify_actions_for_bngsim_routes_rulemonkey_public_aliases():
    for method in ('rm', 'rulemonkey'):
        assert bngsim_model.classify_actions_for_bngsim([
            'simulate({method=>"%s",t_end=>4,n_steps=>40,suffix=>"tc"})' % method
        ]) == bngsim_model.BNGSIM_BACKEND_NF


def test_normalize_nf_action_method_normalizes_nfsim_aliases():
    for method in ('nf', 'nf_reject', 'nfsim'):
        assert bngsim_model._normalize_nf_action_method(method) == 'nf_reject'


@_rulemonkey_required
def test_normalize_nf_action_method_normalizes_rulemonkey_public_aliases():
    for method in ('rm', 'rulemonkey'):
        assert bngsim_model._normalize_nf_action_method(method) == 'nf_exact'


@pytest.mark.parametrize('method', ['rm', 'rulemonkey'])
def test_normalize_nf_action_method_rejects_rulemonkey_when_unavailable(monkeypatch, method):
    monkeypatch.setattr(
        bngsim_model.bngsim,
        'normalize_method',
        _stub_normalize_method_without_rulemonkey,
    )
    with pytest.raises(ValueError, match='RuleMonkey is not present'):
        bngsim_model._normalize_nf_action_method(method)


@pytest.mark.parametrize('method', ['nf_fixed', 'dynstoc', 'ds'])
def test_normalize_nf_action_method_rejects_unavailable_canonical_aliases(method):
    # bngsim recognizes these tokens but no current release ships the backend.
    # Delegation surfaces bngsim's own "recognized but unavailable" error.
    with pytest.raises(ValueError):
        bngsim_model._normalize_nf_action_method(method)


def test_normalize_nf_action_method_rejects_non_nf_method():
    with pytest.raises(ValueError, match="method=>'ode' is not supported"):
        bngsim_model._normalize_nf_action_method('ode')


@pytest.mark.parametrize(
    'method, expected_backend',
    [
        ('ode', bngsim_model.BNGSIM_BACKEND_NET),
        ('ssa', bngsim_model.BNGSIM_BACKEND_NET),
        ('psa', bngsim_model.BNGSIM_BACKEND_NET),
        ('nf', bngsim_model.BNGSIM_BACKEND_NF),
        ('nf_reject', bngsim_model.BNGSIM_BACKEND_NF),
        ('nfsim', bngsim_model.BNGSIM_BACKEND_NF),
        ('pla', None),
        ('unknown_method', None),
    ],
)
def test_classify_action_method_backend_maps_methods(method, expected_backend):
    assert bngsim_model._classify_action_method_backend(method) == expected_backend


@pytest.mark.parametrize('method', ['rm', 'rulemonkey'])
def test_classify_action_method_backend_returns_none_when_rulemonkey_unavailable(monkeypatch, method):
    monkeypatch.setattr(
        bngsim_model.bngsim,
        'normalize_method',
        _stub_normalize_method_without_rulemonkey,
    )
    assert bngsim_model._classify_action_method_backend(method) is None


@_rulemonkey_required
@pytest.mark.parametrize('method', ['rm', 'rulemonkey'])
def test_classify_action_method_backend_maps_rulemonkey_public_aliases(method):
    assert bngsim_model._classify_action_method_backend(method) == bngsim_model.BNGSIM_BACKEND_NF


def test_classify_actions_for_bngsim_defaults_methodless_simulate_to_net():
    assert bngsim_model.classify_actions_for_bngsim([
        'simulate({t_end=>4,n_steps=>40,suffix=>"tc"})'
    ]) == bngsim_model.BNGSIM_BACKEND_NET


def test_classify_actions_for_bngsim_routes_nf_parameter_scan_aliases():
    for method in ('nf', 'nf_reject', 'nfsim'):
        assert bngsim_model.classify_actions_for_bngsim([
            'parameter_scan({method=>"%s",parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2,t_end=>4,suffix=>"scan"})'
            % method
        ]) == bngsim_model.BNGSIM_BACKEND_NF


@_rulemonkey_required
def test_classify_actions_for_bngsim_routes_rulemonkey_parameter_scan():
    assert bngsim_model.classify_actions_for_bngsim([
        'parameter_scan({method=>"rm",parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2,t_end=>4,suffix=>"scan"})'
    ]) == bngsim_model.BNGSIM_BACKEND_NF


def test_bngl_model_marks_rulemonkey_action_stochastic_without_network_generation(tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='rm')

    assert model.stochastic
    assert not model.generates_network


def test_allowed_bngsim_backends_for_action_marks_nf_setconcentration_expression():
    backends, is_simulation_action = bngsim_model._allowed_bngsim_backends_for_action(
        'setConcentration("L(r)", "EGF_copy_number")'
    )

    assert backends == frozenset((bngsim_model.BNGSIM_BACKEND_NF,))
    assert not is_simulation_action


def test_classify_actions_for_bngsim_rejects_nf_setconcentration_with_net_simulation():
    assert bngsim_model.classify_actions_for_bngsim([
        'setConcentration("L(r)", "EGF_copy_number")',
        'simulate({method=>"ode",t_end=>4,n_steps=>40,suffix=>"tc"})',
    ]) is None


def test_classify_actions_for_bngsim_requires_a_simulation_action():
    assert bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'setParameter("k", 1.0)',
    ]) is None


def test_classify_actions_for_bngsim_rejects_mixed_backends():
    assert bngsim_model.classify_actions_for_bngsim([
        'simulate({method=>"ode",t_end=>4,n_steps=>40,suffix=>"ode_tc"})',
        'simulate({method=>"nf",t_end=>4,n_steps=>40,suffix=>"nf_tc"})',
    ]) is None


def test_subprocess_env_uses_bng_command_root(monkeypatch):
    monkeypatch.setenv('BNGPATH', '/old/bng')
    monkeypatch.delenv('BioNetGenRoot', raising=False)

    env = pset._subprocess_env([
        'perl',
        '/Users/wish/Code/bionetgen/bng2/BNG2.pl',
        'model.bngl',
    ])

    assert env['BNGPATH'] == '/Users/wish/Code/bionetgen/bng2'
    assert env['BioNetGenRoot'] == '/Users/wish/Code/bionetgen/bng2'


def test_initialize_models_uses_bngsim_when_available(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class FakeBngsimModel(pset.NetModel):
        def __init__(self, *args, protocol=None, save_files=False, **kwargs):
            super().__init__(*args, **kwargs)

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BngsimModel', FakeBngsimModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], FakeBngsimModel)


def test_initialize_models_bionetgen_backend_skips_bngsim(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir, bngl_backend='bionetgen')

    class UnexpectedBngsimModel(object):
        def __init__(self, *args, **kwargs):
            raise AssertionError('bngsim should not be used')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BngsimModel', UnexpectedBngsimModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.NetModel)


def test_initialize_models_nf_bionetgen_backend_skips_bngsim(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir, bngl_backend='bionetgen')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=AssertionError('should not generate XML')):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.BNGLModel)


def test_initialize_models_no_bngsim_env_disables_auto_selection(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYBNF_NO_BNGSIM', '1')
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.NetModel)


def test_initialize_models_bngsim_backend_rejects_unavailable_bngsim(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir, bngl_backend='bngsim')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=AssertionError('should fail before BNG2.pl')):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', False):
            with patch.object(algorithms, 'BNGSIM_ERROR', 'bngsim is not available'):
                with pytest.raises(printing.PybnfError, match='bngsim is not available'):
                    algorithms.Algorithm._initialize_models(algo)


def test_initialize_models_bngsim_backend_rejects_unsupported_actions(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='pla')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir, bngl_backend='bngsim')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=AssertionError('should fail before BNG2.pl')):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            with pytest.raises(printing.PybnfError, match='not supported by the bngsim bridge'):
                algorithms.Algorithm._initialize_models(algo)


def test_initialize_models_auto_falls_back_for_unsupported_actions(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='pla')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.NetModel)


def test_initialize_models_bngsim_backend_rejects_missing_nfsim(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir, bngl_backend='bngsim')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=AssertionError('should fail before XML')):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', False):
                with pytest.raises(printing.PybnfError, match='does not provide NFsim support'):
                    algorithms.Algorithm._initialize_models(algo)


def test_initialize_models_falls_back_to_netmodel_when_bridge_init_fails(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class BrokenBngsimModel(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('bridge init failed')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BngsimModel', BrokenBngsimModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.NetModel)


def test_initialize_models_bngsim_backend_errors_when_bridge_init_fails(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir, bngl_backend='bngsim')

    class BrokenBngsimModel(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('bridge init failed')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BngsimModel', BrokenBngsimModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with pytest.raises(printing.PybnfError, match='bridge init failed'):
                    algorithms.Algorithm._initialize_models(algo)


def test_initialize_models_uses_bngsim_nf_when_supported(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class FakeBngsimNfModel(object):
        def __init__(self, name, acts, suffs, mutants, xml_path, **kwargs):
            self.name = name
            self.actions = acts
            self.suffixes = suffs
            self.mutants = mutants
            self.xml_path = xml_path
            self.kwargs = kwargs
            self.bng_command = ''

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_xml_generation):
        with patch.object(algorithms, 'BngsimNfModel', FakeBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], FakeBngsimNfModel)
    assert Path(models[0].xml_path).is_file()
    assert 'writeXML()' in _fake_xml_generation.last_bngl_text
    assert models[0].kwargs['split_line_index'] == model.split_line_index
    assert tuple(models[0].kwargs['param_names']) == model.param_names


def test_initialize_models_uses_bngsim_rulemonkey_when_supported(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='rm')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class FakeBngsimNfModel(object):
        def __init__(self, name, acts, suffs, mutants, xml_path, **kwargs):
            self.name = name
            self.actions = acts
            self.suffixes = suffs
            self.mutants = mutants
            self.xml_path = xml_path
            self.kwargs = kwargs
            self.bng_command = ''

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_xml_generation):
        with patch.object(algorithms, 'BngsimNfModel', FakeBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', False):
                    with patch.object(bngsim_model, 'BNGSIM_HAS_RULEMONKEY', True):
                        models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], FakeBngsimNfModel)
    assert Path(models[0].xml_path).is_file()
    assert any('method=>"rm"' in action for action in models[0].actions)


def test_initialize_models_nf_xml_generation_stages_relative_tfun_files(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class FakeBngsimNfModel(object):
        def __init__(self, *args, **kwargs):
            self.bng_command = ''

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_xml_generation):
        with patch.object(algorithms, 'BngsimNfModel', FakeBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    algorithms.Algorithm._initialize_models(algo)

    staged_rel = _extract_staged_tfun_path(_fake_xml_generation.last_bngl_text)
    assert "tfun('test_data.tfun'" not in _fake_xml_generation.last_bngl_text
    assert (_fake_xml_generation.last_bngl_path.parent / staged_rel).is_file()


def test_initialize_models_nf_falls_back_when_nfsim_support_missing(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=AssertionError('should not generate XML')):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', False):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.BNGLModel)


def test_initialize_models_nf_falls_back_when_bridge_init_fails(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class BrokenBngsimNfModel(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('bridge init failed')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_xml_generation):
        with patch.object(algorithms, 'BngsimNfModel', BrokenBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.BNGLModel)


def test_bngsim_nf_model_recomputes_derived_params_from_pset(monkeypatch):
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM', True)

    model = bngsim_model.BngsimNfModel(
        'nf_model',
        [],
        [],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'KD KD__FREE\n',
            'km km__FREE\n',
            'kp = km / KD\n',
            'total = kp + 1\n',
            'end parameters\n',
        ],
        param_names=('KD__FREE', 'km__FREE'),
    )
    model.param_set = pset.PSet([
        _make_free_param('KD__FREE', 2.0),
        _make_free_param('km__FREE', 8.0),
    ])

    overrides = model._build_nf_param_overrides()

    assert overrides['KD'] == 2.0
    assert overrides['km'] == 8.0
    assert overrides['kp'] == 4.0
    assert overrides['total'] == 5.0


def test_bngsim_nf_model_save_preserves_protocol_block(monkeypatch, tmp_path):
    """Regression for #383: protocol blocks must round-trip through the saved
    debug .bngl, positioned between the model body and the actions block."""
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM', True)

    protocol_lines = [
        '    equilibrate({duration=>10})\n',
        '    perturb({param=>"k",factor=>2.0})\n',
    ]
    actions = [
        'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})',
    ]
    model_lines = [
        'begin model\n',
        'begin parameters\n',
        'end parameters\n',
        'end model\n',
    ]

    model = bngsim_model.BngsimNfModel(
        'nf_model',
        actions,
        [('simulate', 'tc')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=model_lines,
        split_line_index=1,
        param_names=(),
        protocol=protocol_lines,
    )
    model.param_set = pset.PSet([])

    out_prefix = tmp_path / 'saved'
    model.save(str(out_prefix))

    saved_text = (tmp_path / 'saved.bngl').read_text()
    proto_start = saved_text.find('begin protocol')
    proto_end = saved_text.find('end protocol')
    actions_start = saved_text.find('begin actions')
    params_end = saved_text.find('end parameters')

    assert proto_start != -1, 'protocol block missing from saved bngl'
    assert proto_end > proto_start
    assert 'equilibrate({duration=>10})' in saved_text
    assert 'perturb({param=>"k",factor=>2.0})' in saved_text
    # Order: parameters → protocol → actions
    assert params_end < proto_start < actions_start
    assert proto_end < actions_start


def test_bngsim_nf_model_save_omits_protocol_block_when_absent(monkeypatch, tmp_path):
    """Models constructed without a protocol must not emit empty protocol blocks."""
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM', True)

    model = bngsim_model.BngsimNfModel(
        'nf_model',
        ['simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})'],
        [('simulate', 'tc')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin model\n',
            'begin parameters\n',
            'end parameters\n',
            'end model\n',
        ],
        split_line_index=1,
        param_names=(),
    )
    model.param_set = pset.PSet([])

    out_prefix = tmp_path / 'saved'
    model.save(str(out_prefix))

    saved_text = (tmp_path / 'saved.bngl').read_text()
    assert 'begin protocol' not in saved_text
    assert 'end protocol' not in saved_text


def test_bngsim_nf_model_preserves_state_across_actions(monkeypatch):
    calls = _install_fake_nfsim(monkeypatch)

    model = bngsim_model.BngsimNfModel(
        'nf_model',
        [
            'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"equil"})',
            'setConcentration("L(r)","EGF_copy_number")',
            'simulate({method=>"nf_reject",t_start=>0,t_end=>1,n_steps=>1,suffix=>"post"})',
        ],
        [('simulate', 'equil'), ('simulate', 'post')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'EGF_copy_number EGF_copy_number__FREE\n',
            'end parameters\n',
        ],
        param_names=('EGF_copy_number__FREE',),
    )
    model.param_set = pset.PSet([
        _make_free_param('EGF_copy_number__FREE', 7.0),
    ])

    ds = model.execute('/tmp', 'job0', 10)

    assert ds['equil'].data[-1, 1] == 0.0
    assert ds['post'].data[-1, 1] == 7.0
    assert len([c for c in calls if c[0] == 'init']) == 1
    assert ('add', 'L', 7, 7) in calls


def test_bngsim_nf_model_uses_rulemonkey_session_for_rm(monkeypatch):
    calls = _install_fake_nf_sessions(monkeypatch)

    model = bngsim_model.BngsimNfModel(
        'rm_model',
        [
            'simulate({method=>"rm",t_start=>0,t_end=>1,n_steps=>1,suffix=>"phase1"})',
            'addConcentration("L(r)", 4)',
            'simulate({method=>"rulemonkey",t_start=>0,t_end=>1,n_steps=>1,suffix=>"phase2"})',
        ],
        [('simulate', 'phase1'), ('simulate', 'phase2')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'end parameters\n',
        ],
        param_names=(),
    )
    model.param_set = pset.PSet([])

    ds = model.execute('/tmp', 'job0', 10)

    assert ds['phase1'].data[-1, 1] == 0.0
    assert ds['phase2'].data[-1, 1] == 4.0
    assert ('create', 'rulemonkey', '/tmp/fake.xml') in calls
    assert not any(c[0] == 'create' and c[1] == 'nfsim' for c in calls)
    assert len([c for c in calls if c[0] == 'init' and c[1] == 'rulemonkey']) == 1


def test_bngsim_nf_model_uses_rulemonkey_for_parameter_scan(monkeypatch):
    calls = _install_fake_nf_sessions(monkeypatch)

    model = bngsim_model.BngsimNfModel(
        'rm_model',
        [
            'parameter_scan({method=>"nf_exact",parameter=>"k",par_scan_vals=>[1,2],t_start=>0,t_end=>1,n_steps=>1,suffix=>"scan"})',
        ],
        [('parameter_scan', 'scan')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'k k__FREE\n',
            'end parameters\n',
        ],
        param_names=('k__FREE',),
    )
    model.param_set = pset.PSet([
        _make_free_param('k__FREE', 1.0),
    ])

    ds = model.execute('/tmp', 'job0', 10)

    assert 'scan' in ds
    assert len([c for c in calls if c[0] == 'create' and c[1] == 'rulemonkey']) == 2
    assert not any(c[0] == 'create' and c[1] == 'nfsim' for c in calls)


# ── #375: save_files-driven .gdat/.scan output for BNGsim BNGL & NF paths ─────


def test_write_saved_action_outputs_writes_pybnf_compatible_files(tmp_path):
    """The helper writes header + numeric rows that pybnf.Data can re-read."""
    arr = np.array([[0.0, 1.0, 0.5], [1.0, 2.0, 0.25]])
    data = pset.Data(arr=arr)
    data.cols = {'time': 0, 'X': 1, 'Y': 2}
    data.headers = {0: 'time', 1: 'X', 2: 'Y'}
    data.indvar = 'time'

    bngsim_model._write_saved_action_outputs(
        str(tmp_path),
        'mname_jobid',
        [('simulate', 'tc')],
        {'tc': data},
    )

    out = tmp_path / 'mname_jobid_tc.gdat'
    assert out.is_file()
    roundtrip = pset.Data(file_name=str(out))
    assert list(roundtrip.cols.keys()) == ['time', 'X', 'Y']
    np.testing.assert_array_almost_equal(roundtrip.data, arr)


def test_write_saved_action_outputs_chooses_scan_extension_for_parameter_scan(tmp_path):
    arr = np.array([[1.0, 9.0], [2.0, 8.0]])
    data = pset.Data(arr=arr)
    data.cols = {'k': 0, 'X_final': 1}
    data.headers = {0: 'k', 1: 'X_final'}
    data.indvar = 'k'

    bngsim_model._write_saved_action_outputs(
        str(tmp_path),
        'mname_jobid',
        [('parameter_scan', 'scan')],
        {'scan': data},
    )

    assert (tmp_path / 'mname_jobid_scan.scan').is_file()
    assert not (tmp_path / 'mname_jobid_scan.gdat').exists()


def test_bngsim_nf_model_execute_writes_gdat_when_save_files_true(monkeypatch, tmp_path):
    _install_fake_nfsim(monkeypatch)
    model = bngsim_model.BngsimNfModel(
        'nf_model',
        ['simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})'],
        [('simulate', 'tc')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=['begin parameters\n', 'end parameters\n'],
        param_names=(),
        save_files=True,
    )
    model.param_set = pset.PSet([])

    model.execute(str(tmp_path), 'nf_model_run0', 10)

    out = tmp_path / 'nf_model_run0_tc.gdat'
    assert out.is_file()
    # Re-readable as a pybnf Data; columns match what the in-memory Data exposes
    roundtrip = pset.Data(file_name=str(out))
    assert 'time' in roundtrip.cols
    assert 'L_total' in roundtrip.cols


def test_bngsim_nf_model_execute_omits_files_when_save_files_false(monkeypatch, tmp_path):
    _install_fake_nfsim(monkeypatch)
    model = bngsim_model.BngsimNfModel(
        'nf_model',
        ['simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})'],
        [('simulate', 'tc')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=['begin parameters\n', 'end parameters\n'],
        param_names=(),
    )
    model.param_set = pset.PSet([])

    model.execute(str(tmp_path), 'nf_model_run0', 10)

    # No files should be written when save_files is left at its default (False)
    assert list(tmp_path.iterdir()) == []


def test_bngsim_nf_model_execute_writes_scan_for_parameter_scan(monkeypatch, tmp_path):
    _install_fake_nf_sessions(monkeypatch)
    model = bngsim_model.BngsimNfModel(
        'rm_model',
        [
            'parameter_scan({method=>"nf_exact",parameter=>"k",par_scan_vals=>[1,2],'
            't_start=>0,t_end=>1,n_steps=>1,suffix=>"scan"})',
        ],
        [('parameter_scan', 'scan')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=['begin parameters\n', 'k k__FREE\n', 'end parameters\n'],
        param_names=('k__FREE',),
        save_files=True,
    )
    model.param_set = pset.PSet([_make_free_param('k__FREE', 1.0)])

    model.execute(str(tmp_path), 'rm_model_run0', 10)

    out = tmp_path / 'rm_model_run0_scan.scan'
    assert out.is_file()
    roundtrip = pset.Data(file_name=str(out))
    # Scan column ordering: scan parameter first, then observables/expressions
    assert list(roundtrip.cols.keys())[0] == 'k'


def test_initialize_models_propagates_save_files_when_delete_old_files_zero(monkeypatch, tmp_path):
    """delete_old_files=0 must reach BngsimNfModel(save_files=True) at construction time."""
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)
    algo.config.config['delete_old_files'] = 0

    captured = {}

    class FakeBngsimNfModel(object):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.bng_command = ''

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_xml_generation):
        with patch.object(algorithms, 'BngsimNfModel', FakeBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    algorithms.Algorithm._initialize_models(algo)

    assert captured.get('save_files') is True


def test_initialize_models_save_files_defaults_false_when_delete_old_files_positive(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf')
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)
    algo.config.config['delete_old_files'] = 1

    captured = {}

    class FakeBngsimNfModel(object):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.bng_command = ''

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_xml_generation):
        with patch.object(algorithms, 'BngsimNfModel', FakeBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    algorithms.Algorithm._initialize_models(algo)

    assert captured.get('save_files') is False


def test_bngsim_model_execute_writes_gdat_when_save_files_true(monkeypatch, tmp_path):
    """Cover the BNGL/network path: BngsimModel.execute writes .gdat on save_files=True."""
    class FakeResult:
        def __init__(self, times):
            self.time = np.asarray(times)
            self.observables = np.zeros((len(times), 1))
            self.observable_names = ['Xtot']
            self.expression_names = []
            self.expressions = np.zeros((len(times), 0))
            self.n_times = len(times)
            self.n_observables = 1

    class FakeSimulator:
        def __init__(self, model, method='ode', **kw):
            self.method = method

        def run(self, t_span=None, n_points=2, **kw):
            return FakeResult(np.linspace(t_span[0], t_span[1], n_points))

        def add_stop_condition(self, *a, **kw): pass
        def clear_stop_conditions(self): pass

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
    fake_bngsim.SimulationTimeout = _FakeSimulationTimeout
    fake_bngsim.StopConditionMet = type('StopConditionMet', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

    obj = object.__new__(bngsim_model.BngsimModel)
    obj.name = 'mname'
    obj.actions = ['simulate({method=>"ode",t_end=>1,n_steps=>2,suffix=>"tc"})']
    obj.suffixes = [('simulate', 'tc')]
    obj.mutants = []
    obj._protocol = []
    obj._net_species_initializers = []
    obj._codegen_so = ''
    obj._net_path = '/tmp/fake.net'
    obj._engine_model = FakeModel()
    obj.param_set = None
    obj.save_files = True

    obj.execute(str(tmp_path), 'mname_run0', 10)

    out = tmp_path / 'mname_run0_tc.gdat'
    assert out.is_file()
    roundtrip = pset.Data(file_name=str(out))
    assert 'time' in roundtrip.cols
    assert 'Xtot' in roundtrip.cols


# ── sample_times validation tests ────────────────────────────────────────────

class TestResolveSampleTimes:
    def test_none_when_not_specified(self):
        assert bngsim_model._resolve_sample_times({}) is None

    def test_none_for_empty_list(self):
        assert bngsim_model._resolve_sample_times({'sample_times': []}) is None

    def test_none_for_single_point(self):
        assert bngsim_model._resolve_sample_times({'sample_times': [1.0]}) is None

    def test_two_points_accepted(self):
        result = bngsim_model._resolve_sample_times({'sample_times': [0.0, 1.0]})
        assert result == [0.0, 1.0]

    def test_three_points_accepted(self):
        result = bngsim_model._resolve_sample_times({'sample_times': [0.0, 0.5, 1.0]})
        assert result == [0.0, 0.5, 1.0]

    def test_returned_sorted(self):
        result = bngsim_model._resolve_sample_times({'sample_times': [1.0, 0.0, 0.5]})
        assert result == [0.0, 0.5, 1.0]


# ── addConcentration parser tests ─────────────────────────────────────────────

class TestParseAddConcentration:
    def test_basic(self):
        result = bngsim_model._parse_add_concentration(
            'addConcentration("Ligand()", 500)'
        )
        assert result == ("Ligand()", 500.0)

    def test_scientific_notation(self):
        result = bngsim_model._parse_add_concentration(
            'addConcentration("A(b)", 1.5e3)'
        )
        assert result == ("A(b)", 1500.0)

    def test_single_quotes(self):
        result = bngsim_model._parse_add_concentration(
            "addConcentration('X()', 42)"
        )
        assert result == ("X()", 42.0)

    def test_leading_whitespace(self):
        result = bngsim_model._parse_add_concentration(
            '    addConcentration("S()", 10)'
        )
        assert result == ("S()", 10.0)

    def test_returns_none_for_setConcentration(self):
        assert bngsim_model._parse_add_concentration(
            'setConcentration("A()", 100)'
        ) is None

    def test_returns_none_for_empty(self):
        assert bngsim_model._parse_add_concentration('') is None

    def test_returns_none_for_garbage(self):
        assert bngsim_model._parse_add_concentration('simulate({t_end=>10})') is None


# ── addConcentration backend classification tests ─────────────────────────────

def test_allowed_bngsim_backends_for_action_recognizes_addConcentration():
    backends, is_sim = bngsim_model._allowed_bngsim_backends_for_action(
        'addConcentration("Ligand()", 500)'
    )
    assert backends == bngsim_model._BNGSIM_ACTION_BACKENDS
    assert not is_sim


def test_classify_actions_for_bngsim_accepts_addConcentration_with_ode():
    result = bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'simulate({method=>"ode",t_end=>100,n_steps=>100,suffix=>"phase1"})',
        'addConcentration("Ligand()", 500)',
        'simulate({method=>"ode",t_end=>200,n_steps=>100,suffix=>"phase2"})',
    ])
    assert result == bngsim_model.BNGSIM_BACKEND_NET


# ── addConcentration NF execution test ────────────────────────────────────────

def test_bngsim_nf_model_addConcentration_increments_molecules(monkeypatch):
    calls = _install_fake_nfsim(monkeypatch)

    model = bngsim_model.BngsimNfModel(
        'nf_model',
        [
            'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"phase1"})',
            'addConcentration("L(r)", 500)',
            'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"phase2"})',
        ],
        [('simulate', 'phase1'), ('simulate', 'phase2')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'end parameters\n',
        ],
        param_names=(),
    )
    model.param_set = pset.PSet([])

    ds = model.execute('/tmp', 'job0', 10)

    # Both phases should produce data
    assert 'phase1' in ds
    assert 'phase2' in ds
    # 500 molecules should have been added
    assert ('add', 'L', 500, 500) in calls


def test_bngsim_nf_model_addConcentration_stacks_on_existing(monkeypatch):
    """addConcentration adds to the current count, not replacing it."""
    calls = _install_fake_nfsim(monkeypatch)

    model = bngsim_model.BngsimNfModel(
        'nf_model',
        [
            'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"equil"})',
            'setConcentration("L(r)", "200")',
            'addConcentration("L(r)", 300)',
            'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"post"})',
        ],
        [('simulate', 'equil'), ('simulate', 'post')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'end parameters\n',
        ],
        param_names=(),
    )
    model.param_set = pset.PSet([])

    ds = model.execute('/tmp', 'job0', 10)

    # setConcentration sets L to 200, then addConcentration adds 300 more
    add_calls = [c for c in calls if c[0] == 'add']
    # setConcentration("L(r)", "200") → adds 200 (from 0)
    # addConcentration("L(r)", 300) → adds 300 (from 200)
    assert ('add', 'L', 200, 200) in add_calls
    assert ('add', 'L', 300, 500) in add_calls


# ── Gap 8: Expression evaluation tests ──────────────────────────────────────────

class TestEvalNumeric:
    def test_plain_float(self):
        assert bngsim_model._eval_numeric('3.14') == pytest.approx(3.14)

    def test_plain_int(self):
        assert bngsim_model._eval_numeric('42') == 42.0

    def test_scientific_notation(self):
        assert bngsim_model._eval_numeric('1.5e3') == 1500.0

    def test_arithmetic_expression(self):
        assert bngsim_model._eval_numeric('((1/52)*50000/0.04)') == pytest.approx((1/52)*50000/0.04)

    def test_math_function(self):
        import math
        assert bngsim_model._eval_numeric('exp(1)') == pytest.approx(math.e)

    def test_quoted_expression(self):
        assert bngsim_model._eval_numeric('"100"') == 100.0

    def test_extra_namespace(self):
        assert bngsim_model._eval_numeric('x + 1', {'x': 9.0}) == 10.0


def test_parse_set_concentration_with_expression():
    result = bngsim_model._parse_set_concentration(
        'setConcentration("TNF()", ((1/52)*50000/0.04))'
    )
    assert result is not None
    name, value = result
    assert name == 'TNF()'
    assert value == pytest.approx((1/52)*50000/0.04)


def test_parse_add_concentration_with_expression():
    result = bngsim_model._parse_add_concentration(
        'addConcentration("Ligand()", 500 + 100)'
    )
    assert result is not None
    assert result == ("Ligand()", 600.0)


def test_parse_set_parameter_with_expression():
    result = bngsim_model._parse_set_parameter(
        'setParameter("kf", 1e-3 * 2)'
    )
    assert result is not None
    assert result == ("kf", 0.002)


# ── Gap 1: continue=>1 tests ────────────────────────────────────────────────────

class TestContinueFlag:
    def _make_fake_bngsim_model(self, actions, monkeypatch):
        """Build a BngsimModel-like object that can call _execute_actions."""
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
                run_log.append({'t_span': t_span, 'n_points': n_points, **kw})
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

        # Build a minimal BngsimModel without going through __init__
        obj = object.__new__(bngsim_model.BngsimModel)
        obj.actions = actions
        obj._net_species_initializers = []
        obj._codegen_so = ''
        obj._net_path = '/tmp/fake.net'

        return obj, FakeModel(), run_log

    def test_continue_uses_model_time(self, monkeypatch):
        actions = [
            'simulate({method=>"ode",t_end=>50,n_steps=>10,suffix=>"phase1"})',
            'simulate({method=>"ode",t_end=>100,n_steps=>10,continue=>1,suffix=>"phase2"})',
        ]
        obj, model, run_log = self._make_fake_bngsim_model(actions, monkeypatch)
        ds = obj._execute_actions(model)

        assert 'phase1' in ds
        assert 'phase2' in ds
        assert run_log[0]['t_span'] == (0, 50)
        # continue=>1 with no explicit t_start → uses model_time=50
        assert run_log[1]['t_span'] == (50, 100)

    def test_explicit_t_start_overrides_continue(self, monkeypatch):
        actions = [
            'simulate({method=>"ode",t_end=>50,n_steps=>10,suffix=>"phase1"})',
            'simulate({method=>"ode",t_start=>25,t_end=>100,n_steps=>10,continue=>1,suffix=>"phase2"})',
        ]
        obj, model, run_log = self._make_fake_bngsim_model(actions, monkeypatch)
        ds = obj._execute_actions(model)

        # Explicit t_start=25 is used even with continue=>1
        assert run_log[1]['t_span'] == (25, 100)

    def test_no_continue_defaults_to_zero(self, monkeypatch):
        actions = [
            'simulate({method=>"ode",t_end=>50,n_steps=>10,suffix=>"phase1"})',
            'simulate({method=>"ode",t_end=>100,n_steps=>10,suffix=>"phase2"})',
        ]
        obj, model, run_log = self._make_fake_bngsim_model(actions, monkeypatch)
        ds = obj._execute_actions(model)

        assert run_log[0]['t_span'] == (0, 50)
        assert run_log[1]['t_span'] == (0, 100)


# ── Gap 3: atol/rtol/seed passthrough tests ─────────────────────────────────────

class TestSimulatorKwargs(TestContinueFlag):
    def test_atol_rtol_seed_passed_to_run(self, monkeypatch):
        actions = [
            'simulate({method=>"ode",t_end=>10,n_steps=>5,atol=>1e-12,rtol=>1e-10,seed=>99,suffix=>"tc"})',
        ]
        obj, model, run_log = self._make_fake_bngsim_model(actions, monkeypatch)
        obj._execute_actions(model)

        assert run_log[0]['atol'] == 1e-12
        assert run_log[0]['rtol'] == 1e-10
        assert run_log[0]['seed'] == 99

    def test_defaults_omit_kwargs(self, monkeypatch):
        actions = [
            'simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"tc"})',
        ]
        obj, model, run_log = self._make_fake_bngsim_model(actions, monkeypatch)
        obj._execute_actions(model)

        assert 'atol' not in run_log[0]
        assert 'rtol' not in run_log[0]
        assert 'seed' not in run_log[0]


# ── Gap 2: stop_if tests ────────────────────────────────────────────────────────

class TestStopIf:
    def test_stop_if_catches_exception(self, monkeypatch):
        """stop_if triggers StopConditionMet → uses truncated result."""

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

        class FakeStopConditionMet(Exception):
            def __init__(self, result):
                self.result = result

        class FakeSimulator:
            def __init__(self, model, method='ode', **kw):
                pass

            def run(self, t_span=None, n_points=2, **kw):
                # Simulate early stop at t=5 instead of t=10
                truncated = FakeResult(np.linspace(t_span[0], 5, 3))
                raise FakeStopConditionMet(truncated)

            def add_stop_condition(self, expr, label=None):
                pass

            def clear_stop_conditions(self):
                pass

        class FakeModel:
            param_names = []

        fake_bngsim = types.ModuleType('bngsim')
        fake_bngsim.Simulator = FakeSimulator
        fake_bngsim.StopConditionMet = FakeStopConditionMet
        monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
        monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

        obj = object.__new__(bngsim_model.BngsimModel)
        obj.actions = [
            'simulate({method=>"ode",t_end=>10,n_steps=>10,stop_if=>"A<1",suffix=>"tc"})',
        ]
        obj._net_species_initializers = []
        obj._codegen_so = ''
        obj._net_path = '/tmp/fake.net'

        ds = obj._execute_actions(FakeModel())
        assert 'tc' in ds
        # Truncated result should have 3 time points (not 11)
        assert ds['tc'].data.shape[0] == 3
        assert ds['tc'].data[-1, 0] == pytest.approx(5.0)


# ── Gap 4: print_functions tests ─────────────────────────────────────────────────

class TestPrintFunctions:
    def _make_result_with_expressions(self):
        """Build a fake result with both observables and expressions."""
        class FakeResult:
            def __init__(self):
                self.time = np.array([0.0, 1.0])
                self.observables = np.array([[1.0], [2.0]])
                self.observable_names = ['obs1']
                self.expression_names = ['func1']
                self.expressions = np.array([[10.0], [20.0]])
                self.n_times = 2
                self.n_observables = 1

        return FakeResult()

    def test_default_excludes_functions(self):
        result = self._make_result_with_expressions()
        data = bngsim_model.BngsimModel._result_to_data(result)
        assert 'func1' not in data.cols
        assert data.data.shape[1] == 2  # time + obs1

    def test_print_functions_true_includes_functions(self):
        result = self._make_result_with_expressions()
        data = bngsim_model.BngsimModel._result_to_data(result, print_functions=True)
        assert 'func1' in data.cols
        assert data.data.shape[1] == 3  # time + obs1 + func1
        assert data.data[0, 2] == 10.0

    def test_scan_row_default_excludes_functions(self):
        result = self._make_result_with_expressions()
        row, obs, expr = bngsim_model.BngsimModel._scan_result_to_row(result, 1.0)
        assert expr == []
        assert len(row) == 2  # scan_value + obs1

    def test_scan_row_print_functions_includes(self):
        result = self._make_result_with_expressions()
        row, obs, expr = bngsim_model.BngsimModel._scan_result_to_row(
            result, 1.0, print_functions=True,
        )
        assert expr == ['func1']
        assert len(row) == 3  # scan_value + obs1 + func1


# ── Gap 5: bifurcate tests ──────────────────────────────────────────────────────

def test_parse_bifurcate_action():
    result = bngsim_model._parse_bifurcate_action(
        'bifurcate({parameter=>"k",par_min=>0.1,par_max=>10,n_scan_pts=>5,'
        'method=>"ode",t_end=>100,suffix=>"bif"})'
    )
    assert result is not None
    assert result['parameter'] == 'k'
    assert result['par_min'] == '0.1'
    assert result['suffix'] == 'bif'


def test_classify_actions_for_bngsim_accepts_bifurcate():
    result = bngsim_model.classify_actions_for_bngsim([
        'bifurcate({parameter=>"k",par_min=>0.1,par_max=>10,n_scan_pts=>5,'
        'method=>"ode",t_end=>100,suffix=>"bif"})',
    ])
    assert result == bngsim_model.BNGSIM_BACKEND_NET


def test_allowed_bngsim_backends_for_action_recognizes_bifurcate():
    backends, is_sim = bngsim_model._allowed_bngsim_backends_for_action(
        'bifurcate({parameter=>"k",method=>"ode",par_min=>1,par_max=>2,n_scan_pts=>2,t_end=>10,suffix=>"bf"})'
    )
    assert bngsim_model.BNGSIM_BACKEND_NET in backends
    assert is_sim


class TestBifurcateExecution(TestContinueFlag):
    """Verify bifurcate carries model state (concentrations) between scan points."""

    def test_bifurcate_carries_state_between_points(self, monkeypatch):
        concentrations = {}
        param_values = {'k': 1.0}
        clone_count = [0]

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
            def run(self, t_span=None, n_points=2, **kw):
                return FakeResult(np.linspace(t_span[0], t_span[1], n_points))

        class FakeModel:
            param_names = ['k']
            def get_param(self, name):
                return param_values.get(name, 0.0)
            def set_param(self, name, val):
                param_values[name] = val
            def reset(self):
                pass
            def clone(self):
                clone_count[0] += 1
                return FakeModel()
            def set_concentration(self, name, val):
                concentrations[name] = val
            def get_concentration(self, name):
                return concentrations.get(name, 0.0)
            def save_concentrations(self):
                pass

        fake_bngsim = types.ModuleType('bngsim')
        fake_bngsim.Simulator = FakeSimulator
        monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
        monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

        obj = object.__new__(bngsim_model.BngsimModel)
        obj.actions = [
            'bifurcate({parameter=>"k",par_scan_vals=>[1,2,3],'
            'method=>"ode",t_end=>10,suffix=>"bif"})',
        ]
        obj._net_species_initializers = []
        obj._codegen_so = ''
        obj._net_path = '/tmp/fake.net'

        ds = obj._execute_actions(FakeModel())
        assert 'bif' in ds
        # bifurcate clones only once (not per point)
        assert clone_count[0] == 1
        # 3 scan points -> 3 rows
        assert ds['bif'].data.shape[0] == 3

    def test_parameter_scan_resets_between_points(self, monkeypatch):
        """Regular parameter_scan clones per point (reset_conc default=1)."""
        clone_count = [0]

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
                pass
            def run(self, t_span=None, n_points=2, **kw):
                return FakeResult(np.linspace(t_span[0], t_span[1], n_points))

        class FakeModel:
            param_names = ['k']
            def get_param(self, name): return 1.0
            def set_param(self, name, val): pass
            def reset(self): pass
            def clone(self):
                clone_count[0] += 1
                return FakeModel()
            def set_concentration(self, name, val): pass
            def get_concentration(self, name): return 0.0
            def save_concentrations(self): pass

        fake_bngsim = types.ModuleType('bngsim')
        fake_bngsim.Simulator = FakeSimulator
        monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
        monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

        obj = object.__new__(bngsim_model.BngsimModel)
        obj.actions = [
            'parameter_scan({parameter=>"k",par_scan_vals=>[1,2,3],'
            'method=>"ode",t_end=>10,suffix=>"scan"})',
        ]
        obj._net_species_initializers = []
        obj._codegen_so = ''
        obj._net_path = '/tmp/fake.net'

        ds = obj._execute_actions(FakeModel())
        assert 'scan' in ds
        # parameter_scan with reset_conc=>1 clones per point
        assert clone_count[0] == 3


# ── Gap 7: codegen tests ─────────────────────────────────────���──────────────────

def test_try_prepare_codegen_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setenv('PYBNF_NO_CODEGEN', '1')
    assert bngsim_model._try_prepare_codegen('/tmp/fake.net') == ''


def test_try_prepare_codegen_returns_empty_when_bngsim_no_codegen(monkeypatch):
    monkeypatch.setenv('BNGSIM_NO_CODEGEN', '1')
    assert bngsim_model._try_prepare_codegen('/tmp/fake.net') == ''


def test_try_prepare_codegen_returns_empty_on_import_error(monkeypatch):
    monkeypatch.delenv('PYBNF_NO_CODEGEN', raising=False)
    monkeypatch.delenv('BNGSIM_NO_CODEGEN', raising=False)
    # No bngsim.prepare_codegen available → should return ""
    assert bngsim_model._try_prepare_codegen('/tmp/nonexistent.net') == ''


def test_codegen_kwargs_returns_empty_without_codegen():
    obj = object.__new__(bngsim_model.BngsimModel)
    obj._codegen_so = ''
    obj._net_path = '/tmp/fake.net'
    assert obj._codegen_kwargs() == {}


def test_codegen_kwargs_returns_dict_with_codegen():
    obj = object.__new__(bngsim_model.BngsimModel)
    obj._codegen_so = '/tmp/fake.so'
    obj._net_path = '/tmp/fake.net'
    kw = obj._codegen_kwargs('ode')
    assert kw == {'codegen': True, 'net_path': '/tmp/fake.net'}


def test_codegen_kwargs_empty_for_non_ode():
    obj = object.__new__(bngsim_model.BngsimModel)
    obj._codegen_so = '/tmp/fake.so'
    obj._net_path = '/tmp/fake.net'
    assert obj._codegen_kwargs('ssa') == {}


# ── Gap 6: addConcentration in network-backed path (verification) ────────────────

class TestAddConcentrationNetBackend(TestContinueFlag):
    def test_add_concentration_in_execute_actions(self, monkeypatch):
        """Verify addConcentration works in the BngsimModel (network) path."""
        concentrations = {'A()': 100.0}

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
                pass
            def run(self, t_span=None, n_points=2, **kw):
                return FakeResult(np.linspace(t_span[0], t_span[1], n_points))
            def add_stop_condition(self, expr, label=None):
                pass
            def clear_stop_conditions(self):
                pass

        class FakeModel:
            param_names = []
            def get_concentration(self, name):
                return concentrations.get(name, 0.0)
            def set_concentration(self, name, val):
                concentrations[name] = val
            def reset(self):
                pass
            def save_concentrations(self):
                pass

        fake_bngsim = types.ModuleType('bngsim')
        fake_bngsim.Simulator = FakeSimulator
        monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
        monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

        obj = object.__new__(bngsim_model.BngsimModel)
        obj.actions = [
            'simulate({method=>"ode",t_end=>10,n_steps=>5,suffix=>"phase1"})',
            'addConcentration("A()", 50)',
            'simulate({method=>"ode",t_end=>20,n_steps=>5,suffix=>"phase2"})',
        ]
        obj._net_species_initializers = []
        obj._codegen_so = ''
        obj._net_path = '/tmp/fake.net'

        ds = obj._execute_actions(FakeModel())
        assert 'phase1' in ds
        assert 'phase2' in ds
        # addConcentration should have added 50 to the initial 100
        assert concentrations['A()'] == 150.0


# ---------------------------------------------------------------------------
# Hybrid backend (generate_network + NF simulate) — classification tests
# ---------------------------------------------------------------------------

def test_classify_actions_for_bngsim_returns_hybrid_for_gennet_plus_nf():
    """When classify sees both generate_network and NF simulate, returns hybrid."""
    assert bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'simulate({method=>"nf",t_end=>10,n_steps=>10,suffix=>"tc"})',
    ]) == bngsim_model.BNGSIM_BACKEND_HYBRID


def test_classify_actions_for_bngsim_returns_hybrid_with_nfsim_alias():
    assert bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'simulate({method=>"nfsim",t_end=>10,n_steps=>10,suffix=>"tc"})',
    ]) == bngsim_model.BNGSIM_BACKEND_HYBRID


def test_classify_actions_for_bngsim_returns_hybrid_with_rulemonkey_alias(monkeypatch):
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_RULEMONKEY', True)
    assert bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'simulate({method=>"rm",t_end=>10,n_steps=>10,suffix=>"tc"})',
    ]) == bngsim_model.BNGSIM_BACKEND_HYBRID


def test_classify_actions_for_bngsim_returns_net_not_hybrid_for_gennet_plus_ode():
    """generate_network + ODE is the normal net path, not hybrid."""
    assert bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'simulate({method=>"ode",t_end=>10,n_steps=>10,suffix=>"tc"})',
    ]) == bngsim_model.BNGSIM_BACKEND_NET


def test_classify_actions_for_bngsim_returns_none_for_gennet_plus_pla():
    """generate_network + PLA is unsupported — should return None."""
    assert bngsim_model.classify_actions_for_bngsim([
        'generate_network({overwrite=>1})',
        'simulate({method=>"pla",t_end=>10,n_steps=>10,suffix=>"tc"})',
    ]) is None


def test_hybrid_detected_via_generates_network_flag():
    """BNGLModel strips generate_network from actions; hybrid is detected
    by the combination of generates_network=True and bridge_backend=NF
    in _initialize_models, not by the classifier."""
    # With generate_network stripped (as BNGLModel does), classifier returns NF
    assert bngsim_model.classify_actions_for_bngsim([
        'simulate({method=>"nf",t_end=>10,n_steps=>10,suffix=>"tc"})',
    ]) == bngsim_model.BNGSIM_BACKEND_NF


# ---------------------------------------------------------------------------
# Hybrid backend — _initialize_models tests
# ---------------------------------------------------------------------------

def test_initialize_models_hybrid_uses_bngsim_nf(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf', force_generate_network=True)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class FakeBngsimNfModel(object):
        def __init__(self, name, acts, suffs, mutants, xml_path, **kwargs):
            self.name = name
            self.actions = acts
            self.suffixes = suffs
            self.mutants = mutants
            self.xml_path = xml_path
            self.kwargs = kwargs
            self.bng_command = ''

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_hybrid_generation):
        with patch.object(algorithms, 'BngsimNfModel', FakeBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], FakeBngsimNfModel)
    assert Path(models[0].xml_path).is_file()
    # The simulate action should be preserved
    assert any('simulate' in a for a in models[0].actions)
    # The last BNG2.pl call (XML generation) should have writeXML
    assert 'writeXML()' in _fake_hybrid_generation.last_bngl_text


def test_initialize_models_hybrid_falls_back_to_netmodel_when_bngsim_unavailable(monkeypatch, tmp_path):
    """When bngsim is unavailable, hybrid falls through to NetModel (not BNGLModel)
    because generates_network branch still runs BNG2.pl for .net generation."""
    model = _make_tfun_bngl_model(tmp_path, method='nf', force_generate_network=True)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', False):
            models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.NetModel)


def test_initialize_models_hybrid_falls_back_to_netmodel_when_nfsim_missing(monkeypatch, tmp_path):
    """When bngsim lacks NFsim, hybrid falls through to NetModel."""
    model = _make_tfun_bngl_model(tmp_path, method='nf', force_generate_network=True)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
            with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', False):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.NetModel)


def test_initialize_models_hybrid_falls_back_when_bridge_init_fails(monkeypatch, tmp_path):
    model = _make_tfun_bngl_model(tmp_path, method='nf', force_generate_network=True)
    output_dir = tmp_path / 'pybnf_output'
    output_dir.mkdir()
    algo = _make_dummy_algorithm(model, output_dir)

    class BrokenBngsimNfModel(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('bridge init failed')

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_hybrid_generation):
        with patch.object(algorithms, 'BngsimNfModel', BrokenBngsimNfModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                with patch.object(bngsim_model, 'BNGSIM_HAS_NFSIM', True):
                    models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], pset.BNGLModel)


# ── Protocol support tests ───────────────────────────────────────────────────


def test_normalize_action_method_protocol():
    method, poplevel = bngsim_model._normalize_action_method('protocol')
    assert method == 'protocol'
    assert poplevel is None


def test_normalize_action_method_protocol_case_insensitive():
    method, _ = bngsim_model._normalize_action_method('Protocol')
    assert method == 'protocol'


def test_classify_action_method_backend_protocol():
    assert bngsim_model._classify_action_method_backend('protocol') == bngsim_model.BNGSIM_BACKEND_NET


def test_classify_actions_for_bngsim_accepts_protocol_parameter_scan():
    actions = [
        'generate_network({overwrite=>1})',
        'parameter_scan({method=>"protocol",parameter=>"k",par_scan_vals=>[1,2],suffix=>"scan"})',
    ]
    assert bngsim_model.classify_actions_for_bngsim(actions) == bngsim_model.BNGSIM_BACKEND_NET


def _write_protocol_bngl(model_path):
    """Write a minimal BNGL file with a protocol block and a protocol parameter_scan."""
    model_path.write_text("""\
begin model

begin parameters
  k__FREE 1.0
end parameters

begin molecule types
  A()
end molecule types

begin seed species
  A() 100
end seed species

begin observables
  Molecules Atot A()
end observables

begin reaction rules
  A() -> 0 k__FREE
end reaction rules

end model

begin protocol
  simulate({method=>"ode", t_start=>0, t_end=>10, n_steps=>1})
  setConcentration("A()", 50)
  simulate({method=>"ode", t_start=>0, t_end=>5, n_steps=>10})
end protocol

begin actions
  generate_network({overwrite=>1})
  parameter_scan({method=>"protocol", parameter=>"k__FREE", par_scan_vals=>[0.1, 1.0], suffix=>"scan"})
end actions
""")


def _write_protocol_bngl_with_comments(model_path):
    """Write a protocol block containing blank lines and comments."""
    model_path.write_text("""\
begin model

begin parameters
  k__FREE 1.0
end parameters

begin molecule types
  A()
end molecule types

begin seed species
  A() 100
end seed species

begin observables
  Molecules Atot A()
end observables

begin reaction rules
  A() -> 0 k__FREE
end reaction rules

end model

begin protocol
  # equilibration phase
  simulate({method=>"ode", t_start=>0, t_end=>10, n_steps=>1})

  # perturbation
  setConcentration("A()", 50)
  simulate({method=>"ode", t_start=>0, t_end=>5, n_steps=>10})
end protocol

begin actions
  generate_network({overwrite=>1})
  parameter_scan({method=>"protocol", parameter=>"k__FREE", par_scan_vals=>[1], suffix=>"scan"})
end actions
""")


class TestProtocolParsing:
    """Tests for begin protocol / end protocol parsing in BNGLModel."""

    def test_protocol_lines_stored(self, tmp_path):
        model_path = tmp_path / 'proto.bngl'
        _write_protocol_bngl(model_path)
        m = pset.BNGLModel(str(model_path))
        assert len(m.protocol) == 3
        assert 'setConcentration' in m.protocol[1]

    def test_protocol_lines_not_in_actions(self, tmp_path):
        model_path = tmp_path / 'proto.bngl'
        _write_protocol_bngl(model_path)
        m = pset.BNGLModel(str(model_path))
        joined_actions = ' '.join(m.actions)
        assert 'setConcentration' not in joined_actions

    def test_protocol_with_comments_and_blanks(self, tmp_path):
        model_path = tmp_path / 'proto.bngl'
        _write_protocol_bngl_with_comments(model_path)
        m = pset.BNGLModel(str(model_path))
        # 2 simulate + 1 setConcentration + 1 comment + 1 blank + 1 comment = 6
        assert len(m.protocol) == 6
        comment_lines = [l for l in m.protocol if l.strip().startswith('#')]
        assert len(comment_lines) == 2

    def test_generates_network_set_for_protocol_method(self, tmp_path):
        model_path = tmp_path / 'proto.bngl'
        _write_protocol_bngl(model_path)
        m = pset.BNGLModel(str(model_path))
        assert m.generates_network is True

    def test_empty_protocol_attribute_when_no_block(self, tmp_path):
        source_dir = tmp_path / 'source'
        source_dir.mkdir()
        tfun_file = source_dir / 'test_data.tfun'
        tfun_file.write_text("0 0\n1 1\n")
        model_path = source_dir / 'no_proto.bngl'
        _write_tfun_model(model_path)
        m = pset.BNGLModel(str(model_path), suppress_free_param_error=True)
        assert m.protocol == []


class TestProtocolBnglFileText:
    """Tests for protocol block in _bngl_file_text() output."""

    def test_protocol_block_written(self, tmp_path):
        model_path = tmp_path / 'proto.bngl'
        _write_protocol_bngl(model_path)
        m = pset.BNGLModel(str(model_path))
        m.param_set = pset.PSet([_make_free_param('k__FREE', 0.5)])
        text = m.model_text()
        assert 'begin protocol' in text
        assert 'end protocol' in text
        assert 'setConcentration' in text

    def test_protocol_block_before_actions(self, tmp_path):
        model_path = tmp_path / 'proto.bngl'
        _write_protocol_bngl(model_path)
        m = pset.BNGLModel(str(model_path))
        m.param_set = pset.PSet([_make_free_param('k__FREE', 0.5)])
        text = m.model_text()
        proto_pos = text.index('begin protocol')
        actions_pos = text.index('begin actions')
        assert proto_pos < actions_pos

    def test_no_protocol_block_when_empty(self, tmp_path):
        source_dir = tmp_path / 'source'
        source_dir.mkdir()
        tfun_file = source_dir / 'test_data.tfun'
        tfun_file.write_text("0 0\n1 1\n")
        model_path = source_dir / 'no_proto.bngl'
        _write_tfun_model(model_path)
        m = pset.BNGLModel(str(model_path), suppress_free_param_error=True)
        m.param_set = pset.PSet([])
        text = m.model_text()
        assert 'begin protocol' not in text


class TestSaveResetParametersInProtocol(TestContinueFlag):
    """Test saveParameters() / resetParameters() inside protocol blocks."""

    def test_save_reset_parameters_restores_values(self, monkeypatch):
        actions = ['simulate({method=>"ode",t_end=>10,n_steps=>2})']
        obj, model, run_log = self._make_fake_bngsim_model(actions, monkeypatch)

        # Give the model trackable param names and values
        param_vals = {'k1': 0.1, 'k2': 0.5}
        model.param_names = ['k1', 'k2']
        model.get_param = lambda n: param_vals[n]
        set_calls = []
        def mock_set_param(name, val):
            param_vals[name] = val
            set_calls.append((name, val))
        model.set_param = mock_set_param

        obj._protocol = [
            'saveParameters()',
            'setParameter("k1",99.0)',
            'resetParameters()',
        ]
        obj._run_protocol(model)

        # First call: setParameter("k1", 99.0)
        assert set_calls[0] == ('k1', 99.0)
        # resetParameters restores both k1=0.1 and k2=0.5
        restore = {name: val for name, val in set_calls[1:]}
        assert restore['k1'] == 0.1
        assert restore['k2'] == 0.5


# ── wall_time_sim trip-path tests (issue #374) ──────────────────────────────


class _FakeSimulationTimeout(RuntimeError):
    """Stand-in for bngsim.SimulationTimeout — same shape (BngsimError,
    RuntimeError) and the same .timeout / .elapsed attrs."""

    def __init__(self, message, *, timeout, elapsed):
        super().__init__(message)
        self.timeout = float(timeout)
        self.elapsed = float(elapsed)


def _install_fake_simulator_raising_timeout(monkeypatch, *, timeout, elapsed):
    """Wire a fake bngsim whose Simulator.run raises a SimulationTimeout."""

    class FakeSimulator:
        def __init__(self, model, method='ode', **kw):
            self._model = model
            self.method = method

        def run(self, *args, **kwargs):
            raise _FakeSimulationTimeout(
                f"wall-clock budget {timeout}s exceeded at {elapsed:.3f}s",
                timeout=timeout,
                elapsed=elapsed,
            )

        def add_stop_condition(self, expr, label=None):
            pass

        def clear_stop_conditions(self):
            pass

    fake_bngsim = types.ModuleType('bngsim')
    fake_bngsim.Simulator = FakeSimulator
    fake_bngsim.SimulationTimeout = _FakeSimulationTimeout
    fake_bngsim.StopConditionMet = type('StopConditionMet', (Exception,), {})
    fake_bngsim.HAS_NFSIM = False
    fake_bngsim.HAS_RULEMONKEY = False

    monkeypatch.setitem(sys.modules, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    return fake_bngsim


def _make_minimal_bngsim_model(actions):
    """Construct a BngsimModel skipping __init__ — enough to call execute()."""
    obj = object.__new__(bngsim_model.BngsimModel)
    obj.name = 'trip_model'
    obj.actions = actions
    obj.param_set = None
    obj.mutants = []
    obj._net_species_initializers = []
    obj._codegen_so = ''
    obj._net_path = '/tmp/fake.net'
    obj._protocol = []
    obj._pybnf_replicate_index = 0
    obj._pybnf_stochastic_seed_policy = 'auto'

    class FakeEngineModel:
        param_names = []
        def get_param(self, name): return 0.0
        def set_param(self, name, val): pass
        def reset(self): pass
        def clone(self): return self
        def set_concentration(self, name, val): pass
        def get_concentration(self, name): return 0.0
        def save_concentrations(self): pass

    obj._engine_model = FakeEngineModel()
    return obj


def test_bngsim_model_simulate_timeout_reraises_failedsimulationerror(monkeypatch, caplog, tmp_path):
    _install_fake_simulator_raising_timeout(monkeypatch, timeout=0.5, elapsed=0.62)
    model = _make_minimal_bngsim_model([
        'simulate({method=>"ode",t_end=>1000,n_steps=>100,suffix=>"tc"})',
    ])

    caplog.set_level('WARNING')
    with pytest.raises(pset.FailedSimulationError):
        model.execute(str(tmp_path), 'trip_test', 0.5)

    log_text = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'wall_time_sim' in log_text
    assert 'trip_model' in log_text


def _install_fake_session_raising_timeout(monkeypatch, *, backend, timeout, elapsed):
    """Wire a fake bngsim NF session class whose simulate() raises a timeout.

    Returns a list that records ('destroy', backend) when the session is
    torn down so tests can assert cleanup-on-timeout safety.
    """
    teardown_log = []

    class FakeResult:
        observable_names = ['x']
        time = np.zeros(1)
        observables = np.zeros((1, 1))
        expression_names = []
        expressions = np.zeros((1, 0))
        n_times = 1
        n_observables = 1

    class FakeSession:
        def __init__(self, xml_path, *, molecule_limit=None):
            self.xml_path = xml_path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.destroy()

        def clear_param_overrides(self):
            pass

        def set_param(self, name, value):
            pass

        def get_parameter(self, name):
            return 0.0

        def initialize(self, seed):
            pass

        def simulate(self, t_start, t_end, n_points, *, timeout=None):
            seen_timeout = timeout
            raise _FakeSimulationTimeout(
                f"wall-clock budget {seen_timeout}s exceeded at {elapsed:.3f}s",
                timeout=seen_timeout if seen_timeout is not None else 0.0,
                elapsed=elapsed,
            )

        def get_molecule_count(self, mol_type):
            return 0

        def add_molecules(self, mol_type, amount):
            pass

        def destroy(self):
            teardown_log.append(('destroy', backend))

    fake_bngsim = types.ModuleType('bngsim')
    fake_bngsim.SimulationTimeout = _FakeSimulationTimeout
    if backend == bngsim_model.BNGSIM_NF_BACKEND_NFSIM:
        fake_bngsim.NfsimSession = FakeSession
        fake_bngsim.HAS_NFSIM = True
        fake_bngsim.HAS_RULEMONKEY = False
    else:
        fake_bngsim.RuleMonkeySession = FakeSession
        fake_bngsim.HAS_NFSIM = False
        fake_bngsim.HAS_RULEMONKEY = True
    fake_bngsim.normalize_method = _make_fake_normalize_method(
        has_nfsim=fake_bngsim.HAS_NFSIM,
        has_rulemonkey=fake_bngsim.HAS_RULEMONKEY,
    )

    monkeypatch.setitem(sys.modules, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM',
                        backend == bngsim_model.BNGSIM_NF_BACKEND_NFSIM)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_RULEMONKEY',
                        backend == bngsim_model.BNGSIM_NF_BACKEND_RULEMONKEY)
    return teardown_log


@pytest.mark.parametrize('backend,method_token', [
    (bngsim_model.BNGSIM_NF_BACKEND_NFSIM, 'nf'),
    (bngsim_model.BNGSIM_NF_BACKEND_RULEMONKEY, 'rm'),
])
def test_bngsim_nf_model_simulate_timeout_reraises_failedsimulationerror(
    monkeypatch, caplog, backend, method_token, tmp_path,
):
    teardown = _install_fake_session_raising_timeout(
        monkeypatch, backend=backend, timeout=0.5, elapsed=0.71,
    )

    model = bngsim_model.BngsimNfModel(
        'nf_trip_model',
        [
            'simulate({method=>"%s",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})' % method_token,
        ],
        [('simulate', 'tc')],
        [],
        '/tmp/fake.xml',
        bngl_model_lines=[
            'begin parameters\n',
            'end parameters\n',
        ],
        param_names=(),
    )
    model.param_set = pset.PSet([])

    caplog.set_level('WARNING')
    with pytest.raises(pset.FailedSimulationError):
        model.execute(str(tmp_path), 'job0', 0.5)

    # Cleanup-on-timeout: session must be destroyed even though execute()
    # re-raised — otherwise a hung NF session could poison later evaluations.
    assert ('destroy', backend) in teardown
    log_text = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'wall_time_sim' in log_text


def test_bngsim_sim_timeout_kwarg_passes_through_on_success(monkeypatch):
    """Positive wall_time_sim is forwarded as timeout= to sim.run on success."""
    run_log = []

    class FakeCoreResult:
        expression_names = []
        expression_data = np.zeros((2, 0))

    class FakeResult:
        def __init__(self, times):
            self._core = FakeCoreResult()
            self.time = np.asarray(times)
            self.observables = np.zeros((len(times), 1))
            self.observable_names = ['obs']
            self.expression_names = []
            self.expressions = np.zeros((len(times), 0))
            self.n_times = len(times)
            self.n_observables = 1

    class FakeSimulator:
        def __init__(self, model, method='ode', **kw):
            self.method = method

        def run(self, t_span=None, n_points=2, **kw):
            run_log.append({'method': self.method, **kw})
            return FakeResult(np.linspace(t_span[0], t_span[1], n_points))

        def add_stop_condition(self, *a, **kw):
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
    fake_bngsim.SimulationTimeout = _FakeSimulationTimeout
    fake_bngsim.StopConditionMet = type('StopConditionMet', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)

    obj = object.__new__(bngsim_model.BngsimModel)
    obj.actions = [
        'simulate({method=>"ode",t_end=>10,n_steps=>2,suffix=>"tc"})',
    ]
    obj._net_species_initializers = []
    obj._codegen_so = ''
    obj._net_path = '/tmp/fake.net'

    obj._execute_actions(FakeModel(), timeout=12.5)

    assert run_log, "FakeSimulator.run was not called"
    assert run_log[0].get('timeout') == 12.5, (
        f"timeout=12.5 was not forwarded; got run kwargs {run_log[0]}"
    )


# ── failed-simulation logging tests (issue #376) ─────────────────────────────


def _install_fake_simulator_raising(monkeypatch, exc):
    """Wire a fake bngsim whose Simulator.run raises `exc`."""

    class FakeSimulator:
        def __init__(self, model, method='ode', **kw):
            self.method = method

        def run(self, *args, **kwargs):
            raise exc

        def add_stop_condition(self, expr, label=None):
            pass

        def clear_stop_conditions(self):
            pass

    fake_bngsim = types.ModuleType('bngsim')
    fake_bngsim.Simulator = FakeSimulator
    fake_bngsim.SimulationTimeout = _FakeSimulationTimeout
    fake_bngsim.StopConditionMet = type('StopConditionMet', (Exception,), {})
    fake_bngsim.HAS_NFSIM = False
    fake_bngsim.HAS_RULEMONKEY = False

    monkeypatch.setitem(sys.modules, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_VERSION', '9.9.9-test')
    return fake_bngsim


def test_bngsim_model_failure_writes_report(monkeypatch, tmp_path):
    """A BNGsim network failure produces a failure-report log next to the
    simulation folder containing backend, version, model identity, action
    context, parameter set, and exception details."""

    boom = RuntimeError('synthetic bngsim crash')
    _install_fake_simulator_raising(monkeypatch, boom)

    fp = pset.FreeParameter('k1', 'random_var', 0.0, 1.0, value=0.25)
    ps = pset.PSet([fp])

    model = _make_minimal_bngsim_model([
        'simulate({method=>"ode",t_end=>10,n_steps=>2,suffix=>"tc"})',
    ])
    model.param_set = ps
    model._net_path = str(tmp_path / 'source.net')
    (tmp_path / 'source.net').write_text('# fake net contents\n')

    folder = tmp_path / 'sim_folder'
    folder.mkdir()
    with pytest.raises(RuntimeError, match='synthetic bngsim crash'):
        model.execute(str(folder), 'jobX', timeout=None)

    report = folder / 'jobX.log'
    assert report.is_file(), "failure report was not written"
    text = report.read_text()
    assert '# BNGsim failure report' in text
    assert 'backend: bngsim-net' in text
    assert 'bngsim_version: 9.9.9-test' in text
    assert 'model_name: trip_model' in text
    assert 'job_filename: jobX' in text
    assert 'input_path: %s' % model._net_path in text
    assert 'input_present: True' in text
    assert 'action_index: 0' in text
    assert 'method: ode' in text
    assert 'suffix: tc' in text
    assert 'k1 = 0.25' in text
    assert 'exception_type: builtins.RuntimeError' in text
    assert 'exception_message: synthetic bngsim crash' in text


def test_bngsim_nf_model_failure_writes_report(monkeypatch, tmp_path):
    """A BNGsim NF failure produces a failure-report log with backend
    bngsim-nf, the XML input path, action context, and exception details."""

    boom = _FakeSimulationTimeout(
        'wall-clock budget 0.5s exceeded at 0.71s',
        timeout=0.5,
        elapsed=0.71,
    )
    _install_fake_session_raising_timeout(
        monkeypatch,
        backend=bngsim_model.BNGSIM_NF_BACKEND_NFSIM,
        timeout=0.5,
        elapsed=0.71,
    )
    monkeypatch.setattr(bngsim_model, 'BNGSIM_VERSION', '9.9.9-test')

    xml_path = tmp_path / 'source.xml'
    xml_path.write_text('<sbml/>\n')

    model = bngsim_model.BngsimNfModel(
        'nf_trip_model',
        [
            'simulate({method=>"nf",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})',
        ],
        [('simulate', 'tc')],
        [],
        str(xml_path),
        bngl_model_lines=[
            'begin parameters\n',
            'end parameters\n',
        ],
        param_names=(),
    )
    model.param_set = pset.PSet([])

    folder = tmp_path / 'sim_folder'
    folder.mkdir()
    with pytest.raises(pset.FailedSimulationError):
        model.execute(str(folder), 'jobNF', 0.5)

    report = folder / 'jobNF.log'
    assert report.is_file(), "NF failure report was not written"
    text = report.read_text()
    assert '# BNGsim failure report' in text
    assert 'backend: bngsim-nf' in text
    assert 'bngsim_version: 9.9.9-test' in text
    assert 'model_name: nf_trip_model' in text
    assert 'job_filename: jobNF' in text
    assert 'input_path: %s' % str(xml_path) in text
    assert 'input_present: True' in text
    assert 'method: nf' in text
    assert 'suffix: tc' in text
    # The timeout is a SimulationTimeout from the fake bngsim module — the
    # exception's qualname is _FakeSimulationTimeout in this test scope.
    assert 'exception_type:' in text
    assert 'wall-clock budget' in text


def test_bngsim_failure_report_path_matches_copy_log_files_pattern(monkeypatch, tmp_path):
    """The failure report is written at ``{folder}/{filename}.log`` so it is
    picked up by ``Job._copy_log_files`` into ``failed_logs_dir``."""

    boom = ValueError('forced')
    _install_fake_simulator_raising(monkeypatch, boom)

    fp = pset.FreeParameter('p', 'random_var', 0.0, 1.0, value=0.5)
    model = _make_minimal_bngsim_model([
        'simulate({method=>"ode",t_end=>1,n_steps=>1,suffix=>"tc"})',
    ])
    model.param_set = pset.PSet([fp])

    folder = tmp_path / 'sim'
    folder.mkdir()
    with pytest.raises(ValueError):
        model.execute(str(folder), 'model_name_jobid', timeout=None)

    # `_copy_log_files` looks for `{folder}/{name_with_id}.log`.
    expected_log = folder / 'model_name_jobid.log'
    assert expected_log.is_file()

