from pathlib import Path
import types

import numpy.testing as npt
import pytest

from .context import config, parse, printing, pset
import pybnf.bngsim_sbml_model as bngsim_sbml_model


def _raf_xml_path():
    return str(Path(__file__).resolve().parent / 'bngl_files' / 'raf.xml')


def _raf_params():
    return [
        pset.FreeParameter('K3', 'uniform_var', 2000., 10000., 8000.),
        pset.FreeParameter('K5', 'uniform_var', 0.1, 1., 0.3),
    ]


def _raf_params_mutant():
    return [
        pset.FreeParameter('K3', 'uniform_var', 2000., 10000., 2000.),
        pset.FreeParameter('K5', 'uniform_var', 0.1, 1., 0.3),
    ]


def _model_loader_config(sbml_backend='roadrunner'):
    xml_path = _raf_xml_path()
    cfg = object.__new__(config.Configuration)
    cfg._data_map = {}
    cfg.config = {
        'models': {xml_path},
        xml_path: [],
        'delete_old_files': 1,
        'wall_time_sim': 0,
        'sbml_integrator': 'cvode',
        'sbml_backend': sbml_backend,
        'smoothing': 1,
        'parallelize_models': 1,
        'fit_type': 'check',
    }
    return cfg


def test_parse_accepts_sbml_backend():
    assert parse.parse('sbml_backend = bngsim') == ['sbml_backend', 'bngsim']


@pytest.mark.parametrize(
    'bngsim_available, has_loader, libsbml_available, expected',
    [
        (False, False, False, (False, 'bngsim is not available')),
        (True, False, True, (False, 'installed bngsim does not expose SBML loading')),
        (True, True, False, (False, 'python-libsbml is not installed')),
        (True, True, True, (True, '')),
    ],
)
def test_detect_bngsim_sbml_support(monkeypatch, bngsim_available, has_loader, libsbml_available, expected):
    if has_loader:
        model_cls = type('FakeSbmlModel', (), {'from_sbml': staticmethod(lambda path: path)})
    else:
        model_cls = type('FakeSbmlModel', (), {})

    fake_bngsim = types.SimpleNamespace(Model=model_cls)
    monkeypatch.setattr(bngsim_sbml_model, 'BNGSIM_AVAILABLE', bngsim_available)
    monkeypatch.setattr(bngsim_sbml_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_sbml_model, 'LIBSBML_AVAILABLE', libsbml_available)

    assert bngsim_sbml_model._detect_bngsim_sbml_support() == expected


def test_config_routes_xml_to_roadrunner_by_default():
    cfg = _model_loader_config()

    models = cfg._load_models()

    assert isinstance(models['raf'], pset.SbmlModelNoTimeout)
    assert not isinstance(models['raf'], bngsim_sbml_model.BngsimSbmlModelNoTimeout)


@pytest.mark.skipif(
    not bngsim_sbml_model.BNGSIM_HAS_SBML,
    reason='bngsim SBML backend is not available in this environment',
)
def test_config_routes_xml_to_bngsim_when_requested():
    cfg = _model_loader_config(sbml_backend='bngsim')

    models = cfg._load_models()

    assert isinstance(models['raf'], bngsim_sbml_model.BngsimSbmlModelNoTimeout)


def test_config_rejects_bngsim_backend_without_support(monkeypatch):
    cfg = _model_loader_config(sbml_backend='bngsim')
    monkeypatch.setattr(config, 'BNGSIM_HAS_SBML', False)
    monkeypatch.setattr(config, 'BNGSIM_SBML_ERROR', 'python-libsbml is not installed')

    with pytest.raises(printing.PybnfError, match='python-libsbml is not installed'):
        cfg._load_models()


def test_config_rejects_non_cvode_integrator_for_bngsim_backend():
    cfg = object.__new__(config.Configuration)
    cfg.models = {'raf': object.__new__(bngsim_sbml_model.BngsimSbmlModelNoTimeout)}
    cfg.config = {
        'bng_command': '',
        'sbml_backend': 'bngsim',
        'sbml_integrator': 'rk4',
    }

    with pytest.raises(printing.PybnfError, match='sbml_integrator = cvode'):
        cfg._load_simulators()


@pytest.mark.skipif(
    not bngsim_sbml_model.BNGSIM_HAS_SBML,
    reason='bngsim SBML backend is not available in this environment',
)
def test_bngsim_sbml_timecourse_matches_existing_expectations(tmp_path):
    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.TimeCourse({'time': '1000', 'step': '10'})
    model = bngsim_sbml_model.BngsimSbmlModel(
        xml_path,
        xml_path,
        pset=ps,
        actions=(action,),
    )

    result = model.execute(str(tmp_path), 'raf_test_exec', 1000)
    dat = result['time_course']

    assert abs(dat['RIRI'][-1] - 2.94514) < 0.01
    assert abs(dat['R'][-1] - 0.358949) < 0.01
    assert dat.cols['time'] == 0


@pytest.mark.skipif(
    not bngsim_sbml_model.BNGSIM_HAS_SBML,
    reason='bngsim SBML backend is not available in this environment',
)
def test_bngsim_sbml_param_scan_matches_existing_expectations(tmp_path):
    xml_path = _raf_xml_path()
    ps = pset.PSet(_raf_params())
    action = pset.ParamScan({'param': 'K3', 'min': '500', 'max': '10000', 'step': '500', 'time': '1000'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path,
        xml_path,
        pset=ps,
        actions=(action,),
    )

    result = model.execute(str(tmp_path), 'raf_test_scan', 1000)
    dat = result['param_scan']

    assert dat.indvar == 'K3'
    assert abs(dat['I'][0] - 0.236666) < 0.01
    assert abs(dat['R'][-1] - 0.315964) < 0.01


@pytest.mark.skipif(
    not bngsim_sbml_model.BNGSIM_HAS_SBML,
    reason='bngsim SBML backend is not available in this environment',
)
def test_bngsim_sbml_mutant_matches_existing_expectations(tmp_path):
    xml_path = _raf_xml_path()
    mut = pset.Mutation('K3', '*', 4)
    mutset = pset.MutationSet((mut,), suffix='k3x4')
    ps = pset.PSet(_raf_params_mutant())
    action = pset.TimeCourse({'time': '1000', 'step': '10'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path,
        xml_path,
        pset=ps,
        actions=(action,),
    )
    model.add_mutant(mutset)

    result = model.execute(str(tmp_path), 'raf_test_mut', 1000)
    dat = result['time_coursek3x4']

    assert abs(dat['RIRI'][-1] - 2.94514) < 0.01
    assert abs(dat['R'][-1] - 0.358949) < 0.01
    assert dat.cols['time'] == 0


@pytest.mark.skipif(
    not bngsim_sbml_model.BNGSIM_HAS_SBML,
    reason='bngsim SBML backend is not available in this environment',
)
def test_bngsim_sbml_species_ic_scan_matches_roadrunner(tmp_path):
    xml_path = _raf_xml_path()
    action = pset.ParamScan({'param': 'I', 'min': '10', 'max': '50', 'step': '10', 'time': '1000'})
    empty_pset = pset.PSet([])
    rr_model = pset.SbmlModelNoTimeout(xml_path, xml_path, pset=empty_pset, actions=(action,))
    bngsim_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml_path,
        xml_path,
        pset=empty_pset,
        actions=(action,),
    )

    rr_data = rr_model.execute(str(tmp_path), 'raf_rr_ic_scan', 1000)['param_scan']
    bngsim_data = bngsim_model.execute(str(tmp_path), 'raf_bngsim_ic_scan', 1000)['param_scan']

    assert rr_data.indvar == 'I_0'
    assert bngsim_data.indvar == 'I_0'
    npt.assert_allclose(bngsim_data.data[:, 0], rr_data.data[:, 0], rtol=0, atol=1e-12)
    npt.assert_allclose(bngsim_data['R'], rr_data['R'], rtol=1e-3, atol=1e-4)
    npt.assert_allclose(bngsim_data['RIRI'], rr_data['RIRI'], rtol=1e-3, atol=1e-4)
