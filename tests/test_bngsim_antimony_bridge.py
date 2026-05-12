import math
import os
import subprocess
import sys
import textwrap

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


def test_subprocess_antimony_unavailable_surfaces_actionable_error(tmp_path):
    """Realistic shape: bngsim installed but the antimony Python package missing.

    Inject a fake bngsim into sys.modules before pybnf imports anything so
    `_bngsim_caps` consumes the simulated 'antimony_import: False' shape.
    Verify (a) BNGSIM_ANTIMONY_ERROR carries an actionable install hint and
    (b) that hint is surfaced verbatim by the config's PybnfError."""
    model_path = _write_antimony_model(tmp_path)
    script = textwrap.dedent('''
        import sys
        import types
        # Pre-inject a fake bngsim so PyBNF's `_bngsim_caps` reads it instead
        # of the real installed package on the next `import bngsim`.
        fake = types.ModuleType('bngsim')
        fake.__version__ = '0.5.0'
        fake.capabilities = lambda: {
            'version': '0.5.0',
            'features': {
                'nfsim': True, 'rulemonkey': True, 'libsbml': True,
                'antimony': False, 'sbml_import': True, 'sbml_ssa': True,
                'sbml_psa': True, 'antimony_import': False, 'codegen': True,
            },
            'missing': {
                'antimony': "optional dependency 'antimony' not installed",
                'antimony_import': (
                    "requires optional dependency 'antimony'; "
                    "install with: pip install antimony"
                ),
            },
        }
        sys.modules['bngsim'] = fake

        from pybnf import config, printing
        from pybnf._bngsim_caps import BNGSIM_HAS_ANTIMONY, BNGSIM_ANTIMONY_ERROR
        assert BNGSIM_HAS_ANTIMONY is False, BNGSIM_HAS_ANTIMONY
        # Actionable error must name antimony AND give an install hint.
        msg_lc = BNGSIM_ANTIMONY_ERROR.lower()
        assert 'antimony' in msg_lc, BNGSIM_ANTIMONY_ERROR
        assert 'install' in msg_lc or 'pip' in msg_lc, BNGSIM_ANTIMONY_ERROR

        model_path = __MODEL_PATH__
        cfg = object.__new__(config.Configuration)
        cfg._data_map = {}
        cfg.config = {
            'models': {model_path},
            model_path: [],
            'delete_old_files': 1,
            'wall_time_sim': 0,
            'sbml_integrator': 'cvode',
            'sbml_backend': 'roadrunner',
            'smoothing': 1,
            'parallelize_models': 1,
            'fit_type': 'check',
        }
        try:
            cfg._load_models()
        except printing.PybnfError as exc:
            raised = str(exc)
            assert 'Antimony model support' in raised, raised
            # The actionable substring must propagate from BNGSIM_ANTIMONY_ERROR.
            assert 'pip install antimony' in raised, raised
            print('OK')
            sys.exit(0)
        sys.exit('expected PybnfError but none was raised')
    ''').replace('__MODEL_PATH__', repr(str(model_path)))
    env = os.environ.copy()
    env.pop('PYBNF_NO_BNGSIM', None)
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        'stdout=%r stderr=%r' % (result.stdout, result.stderr)
    )
    assert 'OK' in result.stdout


def test_subprocess_pybnf_no_bngsim_rejects_antimony_models(tmp_path):
    """`.ant` model loading must fail at config load when PYBNF_NO_BNGSIM=1.

    Antimony has no roadrunner alternative — when bngsim is disabled there
    is no fallback path. Subprocess pattern is required because
    PYBNF_NO_BNGSIM is read at import time by `_bngsim_caps`.
    """
    model_path = _write_antimony_model(tmp_path)
    script = textwrap.dedent('''
        import sys
        from pybnf import config, printing
        from pybnf._bngsim_caps import BNGSIM_AVAILABLE, BNGSIM_ANTIMONY_ERROR
        assert BNGSIM_AVAILABLE is False, 'env var did not disable bngsim'
        assert 'PYBNF_NO_BNGSIM' in BNGSIM_ANTIMONY_ERROR, (
            'BNGSIM_ANTIMONY_ERROR did not surface env-var reason: '
            + repr(BNGSIM_ANTIMONY_ERROR)
        )
        model_path = __MODEL_PATH__
        cfg = object.__new__(config.Configuration)
        cfg._data_map = {}
        cfg.config = {
            'models': {model_path},
            model_path: [],
            'delete_old_files': 1,
            'wall_time_sim': 0,
            'sbml_integrator': 'cvode',
            'sbml_backend': 'roadrunner',
            'smoothing': 1,
            'parallelize_models': 1,
            'fit_type': 'check',
        }
        try:
            cfg._load_models()
        except printing.PybnfError as exc:
            msg = str(exc)
            assert 'Antimony model support' in msg, msg
            assert 'PYBNF_NO_BNGSIM' in msg, msg
            print('OK')
            sys.exit(0)
        sys.exit('expected PybnfError but none was raised')
    ''').replace('__MODEL_PATH__', repr(str(model_path)))
    env = os.environ.copy()
    env['PYBNF_NO_BNGSIM'] = '1'
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        'stdout=%r stderr=%r' % (result.stdout, result.stderr)
    )
    assert 'OK' in result.stdout


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
