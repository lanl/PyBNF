import os
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .context import algorithms, pset
import pybnf.bngsim_model as bngsim_model


def _write_tfun_model(model_path, method='ode'):
    action_lines = []
    if method not in ('nf', 'nf_reject', 'nfsim'):
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


def _make_tfun_bngl_model(tmp_path, method='ode'):
    source_dir = tmp_path / 'source_model'
    source_dir.mkdir()
    tfun_file = source_dir / 'test_data.tfun'
    tfun_file.write_text(
        "# time f_time\n"
        "0 0\n"
        "1 1\n"
    )
    model_path = source_dir / 'bridge_test.bngl'
    _write_tfun_model(model_path, method=method)
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


def _make_dummy_algorithm(model, output_dir):
    class DummyConfig(object):
        pass

    cfg = DummyConfig()
    cfg.models = {model.name: model}
    cfg.config = {
        'output_dir': str(output_dir),
        'bng_command': '/fake/BNG2.pl',
        'wall_time_gen': 10,
    }

    algo = object.__new__(algorithms.Algorithm)
    algo.config = cfg
    algo.variables = []
    return algo


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

    class FakeCoreResult(object):
        def __init__(self, times, obs_value):
            self.time = np.asarray(times, dtype=float)
            self.observable_names = ['L_total']
            self.observable_data = np.full((len(times), 1), obs_value, dtype=float)
            self.expression_names = []
            self.expression_data = np.zeros((len(times), 0), dtype=float)

    class FakeResult(object):
        def __init__(self, core_result):
            self._core = core_result
            self.time = core_result.time
            self.observables = core_result.observable_data
            self.observable_names = list(core_result.observable_names)
            self.n_times = len(core_result.time)
            self.n_observables = len(self.observable_names)

    class FakeNfsimSimulator(object):
        def __init__(self, xml_path):
            self.xml_path = xml_path
            self.params = {}
            self.molecules = {}
            calls.append(('create', xml_path))

        def clear_param_overrides(self):
            self.params = {}
            calls.append(('clear',))

        def set_molecule_limit(self, limit):
            calls.append(('gml', limit))

        def set_param(self, name, value):
            self.params[name] = float(value)
            calls.append(('param', name, float(value)))

        def get_parameter(self, name):
            return self.params[name]

        def initialize(self, seed):
            calls.append(('init', seed, dict(self.params), dict(self.molecules)))

        def simulate(self, t_start, t_end, n_points):
            calls.append(('simulate', t_start, t_end, n_points, dict(self.params), dict(self.molecules)))
            return FakeCoreResult(np.linspace(t_start, t_end, n_points), self.molecules.get('L', 0))

        def get_molecule_count(self, mol_type):
            return self.molecules.get(mol_type, 0)

        def add_molecules(self, mol_type, amount):
            self.molecules[mol_type] = self.molecules.get(mol_type, 0) + amount
            calls.append(('add', mol_type, amount, self.molecules[mol_type]))

        def destroy_session(self):
            calls.append(('destroy', dict(self.params), dict(self.molecules)))

    fake_pkg = types.ModuleType('bngsim')
    fake_pkg.Result = FakeResult
    fake_core = types.ModuleType('bngsim._bngsim_core')
    fake_core.NfsimSimulator = FakeNfsimSimulator

    monkeypatch.setitem(sys.modules, 'bngsim', fake_pkg)
    monkeypatch.setitem(sys.modules, 'bngsim._bngsim_core', fake_core)
    monkeypatch.setattr(bngsim_model, 'bngsim', fake_pkg)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model, 'BNGSIM_HAS_NFSIM', True)
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


def test_classify_actions_for_bngsim_routes_nf_aliases():
    for method in ('nf', 'nf_reject', 'nfsim'):
        assert bngsim_model.classify_actions_for_bngsim([
            'simulate({method=>"%s",t_end=>4,n_steps=>40,suffix=>"tc"})' % method
        ]) == bngsim_model.BNGSIM_BACKEND_NF


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
        pass

    monkeypatch.chdir(tmp_path)
    with patch.object(algorithms, 'run_subprocess', side_effect=_fake_network_generation):
        with patch.object(algorithms, 'BngsimModel', FakeBngsimModel):
            with patch.object(algorithms, 'BNGSIM_AVAILABLE', True):
                models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], FakeBngsimModel)


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
                with patch.object(algorithms, 'BNGSIM_HAS_NFSIM', True):
                    models = algorithms.Algorithm._initialize_models(algo)

    assert len(models) == 1
    assert isinstance(models[0], FakeBngsimNfModel)
    assert Path(models[0].xml_path).is_file()
    assert 'writeXML()' in _fake_xml_generation.last_bngl_text
    assert models[0].kwargs['split_line_index'] == model.split_line_index
    assert tuple(models[0].kwargs['param_names']) == model.param_names


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
                with patch.object(algorithms, 'BNGSIM_HAS_NFSIM', True):
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
            with patch.object(algorithms, 'BNGSIM_HAS_NFSIM', False):
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
                with patch.object(algorithms, 'BNGSIM_HAS_NFSIM', True):
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
