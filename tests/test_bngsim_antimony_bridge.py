import math
import types

import pytest

from .context import config, parse, printing, pset
import pybnf.bngsim_antimony_model as bngsim_antimony_model


_ANTIMONY_MODEL_TEXT = """model test_decay()
  S = 10.0
  k = 0.1
  S' = -k*S
end
"""


def _write_antimony_model(tmp_path):
    model_path = tmp_path / 'test_decay.ant'
    model_path.write_text(_ANTIMONY_MODEL_TEXT)
    return model_path


def _make_model_loader_config(model_path, sbml_integrator='cvode'):
    cfg = object.__new__(config.Configuration)
    cfg._data_map = {}
    cfg.config = {
        'models': {str(model_path)},
        str(model_path): [],
        'delete_old_files': 1,
        'wall_time_sim': 0,
        'sbml_integrator': sbml_integrator,
        'sbml_backend': 'roadrunner',
        'smoothing': 1,
        'parallelize_models': 1,
        'fit_type': 'check',
    }
    return cfg


def _decay_pset(rate):
    return pset.PSet([
        pset.FreeParameter('k', 'uniform_var', 0.01, 10.0, rate),
    ])


def test_parse_accepts_antimony_model_extension():
    assert parse.parse('model = thing.ant: data.exp') == ['model', 'thing.ant', 'data.exp']


@pytest.mark.parametrize(
    'bngsim_available, has_loader, antimony_available, libsbml_available, expected',
    [
        (False, False, False, False, (False, 'bngsim is not available')),
        (True, False, True, True, (False, 'installed bngsim does not expose Antimony loading')),
        (
            True,
            True,
            False,
            True,
            (False, 'antimony is not installed. Install with: pip install antimony python-libsbml'),
        ),
        (
            True,
            True,
            True,
            False,
            (False, 'python-libsbml is not installed. Install with: pip install antimony python-libsbml'),
        ),
        (True, True, True, True, (True, '')),
    ],
)
def test_detect_bngsim_antimony_support(
    monkeypatch,
    bngsim_available,
    has_loader,
    antimony_available,
    libsbml_available,
    expected,
):
    if has_loader:
        model_cls = type('FakeAntimonyModel', (), {'from_antimony': staticmethod(lambda path: path)})
    else:
        model_cls = type('FakeAntimonyModel', (), {})

    fake_bngsim = types.SimpleNamespace(Model=model_cls)
    monkeypatch.setattr(bngsim_antimony_model, 'BNGSIM_AVAILABLE', bngsim_available)
    monkeypatch.setattr(bngsim_antimony_model, 'bngsim', fake_bngsim)
    monkeypatch.setattr(bngsim_antimony_model, 'ANTIMONY_AVAILABLE', antimony_available)
    monkeypatch.setattr(bngsim_antimony_model, 'LIBSBML_AVAILABLE', libsbml_available)

    assert bngsim_antimony_model._detect_bngsim_antimony_support() == expected


@pytest.mark.skipif(
    not bngsim_antimony_model.BNGSIM_HAS_ANTIMONY,
    reason='bngsim Antimony backend is not available in this environment',
)
def test_config_routes_antimony_models_to_bngsim(tmp_path):
    model_path = _write_antimony_model(tmp_path)
    cfg = _make_model_loader_config(model_path)

    models = cfg._load_models()

    assert isinstance(
        models['test_decay'],
        bngsim_antimony_model.BngsimAntimonyModelNoTimeout,
    )


def test_config_rejects_antimony_models_without_support(tmp_path, monkeypatch):
    model_path = _write_antimony_model(tmp_path)
    cfg = _make_model_loader_config(model_path)
    monkeypatch.setattr(config, 'BNGSIM_HAS_ANTIMONY', False)
    monkeypatch.setattr(
        config,
        'BNGSIM_ANTIMONY_ERROR',
        'antimony is not installed. Install with: pip install antimony python-libsbml',
    )

    with pytest.raises(printing.PybnfError, match='antimony is not installed'):
        cfg._load_models()


@pytest.mark.skipif(
    not bngsim_antimony_model.BNGSIM_HAS_ANTIMONY,
    reason='bngsim Antimony backend is not available in this environment',
)
def test_config_rejects_non_cvode_integrator_for_antimony_models(tmp_path):
    model_path = _write_antimony_model(tmp_path)
    cfg = _make_model_loader_config(model_path, sbml_integrator='rk4')

    with pytest.raises(printing.PybnfError, match='cvode'):
        cfg._load_models()


@pytest.mark.skipif(
    not bngsim_antimony_model.BNGSIM_HAS_ANTIMONY,
    reason='bngsim Antimony backend is not available in this environment',
)
def test_bngsim_antimony_timecourse_uses_pset_value(tmp_path):
    model_path = _write_antimony_model(tmp_path)
    action = pset.TimeCourse({'time': '20', 'step': '1'})
    model = bngsim_antimony_model.BngsimAntimonyModel(
        str(model_path),
        str(model_path),
        pset=_decay_pset(0.3),
        actions=(action,),
    )

    result = model.execute(str(tmp_path), 'ant_decay_tc', 1000)
    dat = result['time_course']

    expected = 10.0 * math.exp(-0.3 * 20.0)
    assert abs(dat['S'][-1] - expected) < 1e-4
    assert dat.cols['time'] == 0


@pytest.mark.skipif(
    not bngsim_antimony_model.BNGSIM_HAS_ANTIMONY,
    reason='bngsim Antimony backend is not available in this environment',
)
def test_bngsim_antimony_param_scan_matches_analytic_decay(tmp_path):
    model_path = _write_antimony_model(tmp_path)
    action = pset.ParamScan({'param': 'k', 'min': '0.1', 'max': '0.5', 'step': '0.1', 'time': '10'})
    model = bngsim_antimony_model.BngsimAntimonyModelNoTimeout(
        str(model_path),
        str(model_path),
        pset=_decay_pset(0.3),
        actions=(action,),
    )

    result = model.execute(str(tmp_path), 'ant_decay_scan', 1000)
    dat = result['param_scan']

    assert dat.indvar == 'k'
    assert abs(dat['S'][0] - 10.0 * math.exp(-0.1 * 10.0)) < 1e-4
    assert abs(dat['S'][-1] - 10.0 * math.exp(-0.5 * 10.0)) < 1e-4


@pytest.mark.skipif(
    not bngsim_antimony_model.BNGSIM_HAS_ANTIMONY,
    reason='bngsim Antimony backend is not available in this environment',
)
def test_bngsim_antimony_mutants_modify_parameters(tmp_path):
    model_path = _write_antimony_model(tmp_path)
    action = pset.TimeCourse({'time': '10', 'step': '1'})
    model = bngsim_antimony_model.BngsimAntimonyModelNoTimeout(
        str(model_path),
        str(model_path),
        pset=_decay_pset(0.3),
        actions=(action,),
    )
    model.add_mutant(pset.MutationSet((pset.Mutation('k', '*', 2),), suffix='k2x'))

    result = model.execute(str(tmp_path), 'ant_decay_mut', 1000)
    dat = result['time_coursek2x']

    expected = 10.0 * math.exp(-0.6 * 10.0)
    assert abs(dat['S'][-1] - expected) < 1e-4
