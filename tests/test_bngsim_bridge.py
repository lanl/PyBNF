import os
import re
from pathlib import Path
from unittest.mock import patch

from .context import algorithms, pset
import pybnf.bngsim_model as bngsim_model


def _write_tfun_model(model_path, method='ode'):
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
    generate_network({overwrite=>1})
    simulate({method=>"%s",t_end=>4,n_steps=>40,suffix=>"tc"})
end actions
""" % method
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
