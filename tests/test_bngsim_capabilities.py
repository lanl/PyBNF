"""Tests for the centralized BNGsim capability detection layer.

Exercises ``pybnf._bngsim_caps`` (the module that owns version enforcement
and feature gating) against the issue #378 acceptance criteria:
too-old BNGsim, PYBNF_NO_BNGSIM-disabled, and missing-capability cases.
"""

import importlib
import logging
import os
import subprocess
import sys
import textwrap
import types

import pytest

from pybnf import _bngsim_caps


def _reload_caps_with(monkeypatch, fake_module, env=None):
    """Replace sys.modules['bngsim'] and reload _bngsim_caps; return the module."""
    monkeypatch.setitem(sys.modules, 'bngsim', fake_module)
    if env is None:
        monkeypatch.delenv('PYBNF_NO_BNGSIM', raising=False)
    else:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    return importlib.reload(_bngsim_caps)


def _restore_caps():
    """Reload _bngsim_caps against the real bngsim package."""
    sys.modules.pop('bngsim', None)
    importlib.reload(_bngsim_caps)


def _make_fake_bngsim(*, version='0.5.0', features=None, missing=None,
                     include_capabilities=True):
    fake = types.ModuleType('bngsim')
    fake.__version__ = version
    if features is None:
        features = {
            'nfsim': True, 'rulemonkey': True, 'libsbml': True,
            'antimony': True, 'sbml_import': True, 'sbml_ssa': True,
            'sbml_psa': True, 'antimony_import': True, 'codegen': True,
        }
    if missing is None:
        missing = {}
    if include_capabilities:
        fake.capabilities = lambda: {
            'version': version,
            'features': dict(features),
            'missing': dict(missing),
        }
    return fake


def test_too_old_bngsim_version_is_rejected(monkeypatch):
    fake = _make_fake_bngsim(version='0.4.9')
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.BNGSIM_AVAILABLE is False
        assert '0.4.9' in caps.BNGSIM_ERROR
        assert 'bngsim>=0.5.0' in caps.BNGSIM_ERROR
    finally:
        _restore_caps()


def test_bngsim_without_capabilities_api_is_rejected(monkeypatch):
    """A bngsim install lacking capabilities() looks too old to PyBNF."""
    fake = _make_fake_bngsim(version='0.5.0', include_capabilities=False)
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.BNGSIM_AVAILABLE is False
        assert 'capabilities' in caps.BNGSIM_ERROR
    finally:
        _restore_caps()


def test_pybnf_no_bngsim_env_disables_backend(monkeypatch):
    fake = _make_fake_bngsim()
    try:
        caps = _reload_caps_with(monkeypatch, fake, env={'PYBNF_NO_BNGSIM': '1'})
        assert caps.BNGSIM_AVAILABLE is False
        assert 'PYBNF_NO_BNGSIM' in caps.BNGSIM_ERROR
        assert caps.BNGSIM_HAS_NFSIM is False
        assert caps.BNGSIM_HAS_SBML is False
    finally:
        _restore_caps()


def test_features_propagate_from_bngsim_capabilities(monkeypatch):
    fake = _make_fake_bngsim(
        features={
            'nfsim': True, 'rulemonkey': False,
            'libsbml': True, 'antimony': False,
            'sbml_import': True, 'sbml_ssa': True, 'sbml_psa': True,
            'antimony_import': False, 'codegen': True,
        },
        missing={
            'rulemonkey': 'RuleMonkey backend not present in this install',
            'antimony': "optional dependency 'antimony' not installed",
            'antimony_import': "requires optional dependency 'antimony'",
        },
    )
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.BNGSIM_AVAILABLE is True
        assert caps.BNGSIM_HAS_NFSIM is True
        assert caps.BNGSIM_HAS_RULEMONKEY is False
        assert caps.BNGSIM_HAS_SBML is True
        assert caps.BNGSIM_HAS_ANTIMONY is False
        assert 'RuleMonkey' in caps.feature_missing_reason('rulemonkey')
        assert "'antimony'" in caps.feature_missing_reason('antimony_import')
        assert caps.feature_missing_reason('nfsim') == ''
    finally:
        _restore_caps()


def test_output_sensitivities_flag_tracks_feature(monkeypatch):
    """BNGSIM_HAS_OUTPUT_SENS mirrors the backend feature, defaulting False (#447)."""
    # Feature present -> flag True.
    present = _make_fake_bngsim(features={'output_sensitivities': True})
    try:
        caps = _reload_caps_with(monkeypatch, present)
        assert caps.BNGSIM_HAS_OUTPUT_SENS is True
    finally:
        _restore_caps()

    # Feature absent from a build (older bngsim that still passes the version
    # floor): flag defaults False without raising the floor.
    absent = _make_fake_bngsim(features={'nfsim': True})
    try:
        caps = _reload_caps_with(monkeypatch, absent)
        assert caps.BNGSIM_AVAILABLE is True
        assert caps.BNGSIM_HAS_OUTPUT_SENS is False
    finally:
        _restore_caps()


def test_missing_libsbml_produces_actionable_sbml_error(monkeypatch):
    fake = _make_fake_bngsim(
        features={
            'nfsim': True, 'rulemonkey': True,
            'libsbml': False, 'antimony': False,
            'sbml_import': False, 'sbml_ssa': False, 'sbml_psa': False,
            'antimony_import': False, 'codegen': True,
        },
        missing={
            'libsbml': "optional dependency 'python-libsbml' not installed",
            'sbml_import': "optional dependency 'python-libsbml' not installed",
            'sbml_ssa': "optional dependency 'python-libsbml' not installed",
            'sbml_psa': "optional dependency 'python-libsbml' not installed",
            'antimony': "optional dependency 'antimony' not installed",
            'antimony_import': (
                "requires optional dependencies 'antimony' and 'python-libsbml'"
            ),
        },
    )
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.BNGSIM_HAS_SBML is False
        assert 'python-libsbml' in caps.BNGSIM_SBML_ERROR
        assert caps.BNGSIM_HAS_ANTIMONY is False
        assert 'python-libsbml' in caps.BNGSIM_ANTIMONY_ERROR
    finally:
        _restore_caps()


def test_missing_nfsim_backend_distinguishes_from_python_dependency(monkeypatch):
    """A compiled-backend gap reads differently from a missing pip dep."""
    fake = _make_fake_bngsim(
        features={
            'nfsim': False, 'rulemonkey': True,
            'libsbml': True, 'antimony': True,
            'sbml_import': True, 'sbml_ssa': True, 'sbml_psa': True,
            'antimony_import': True, 'codegen': True,
        },
        missing={
            'nfsim': (
                'NFsim backend not present in this install '
                '(vendored at third_party/nfsim/ and built by default; this '
                'install was either configured -DBNGSIM_BUILD_NFSIM=OFF or '
                'installed from a wheel that excludes NFsim)'
            ),
        },
    )
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        reason = caps.feature_missing_reason('nfsim')
        assert 'not present' in reason
        assert 'BNGSIM_BUILD_NFSIM' in reason
    finally:
        _restore_caps()


def test_capabilities_pins_module_state_at_import(monkeypatch):
    """Subsequent mutations to the bngsim module don't change PyBNF's view.

    PyBNF reads capabilities() once at import; this guards against a future
    refactor that calls capabilities() on every check (which would create a
    surprise observability dependency on bngsim's internal state).
    """
    fake = _make_fake_bngsim()
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.BNGSIM_HAS_NFSIM is True
        # Mutate after import — flags must stay pinned.
        fake.capabilities = lambda: {
            'version': '0.5.0',
            'features': dict(caps.BNGSIM_FEATURES, nfsim=False),
            'missing': {},
        }
        assert caps.BNGSIM_HAS_NFSIM is True
    finally:
        _restore_caps()


@pytest.mark.parametrize('feature', [
    'nfsim', 'rulemonkey', 'libsbml', 'antimony',
    'sbml_import', 'antimony_import', 'codegen',
])
def test_feature_missing_reason_returns_empty_when_available(monkeypatch, feature):
    fake = _make_fake_bngsim()
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.feature_missing_reason(feature) == ''
    finally:
        _restore_caps()


def test_feature_missing_reason_when_bngsim_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'bngsim', None)
    monkeypatch.setenv('PYBNF_NO_BNGSIM', '1')
    try:
        caps = importlib.reload(_bngsim_caps)
        assert caps.feature_missing_reason('nfsim') == caps.BNGSIM_ERROR
        assert 'PYBNF_NO_BNGSIM' in caps.feature_missing_reason('nfsim')
    finally:
        _restore_caps()


def test_subprocess_pybnf_no_bngsim_disables(tmp_path):
    """Belt-and-suspenders subprocess check: PYBNF_NO_BNGSIM survives
    a fresh interpreter, with no reload tricks involved."""
    script = textwrap.dedent('''
        from pybnf import _bngsim_caps
        assert _bngsim_caps.BNGSIM_AVAILABLE is False
        assert 'PYBNF_NO_BNGSIM' in _bngsim_caps.BNGSIM_ERROR
        print('OK')
    ''')
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


# --------------------------------------------------------------------------- #
# _version_compatible — the pure compatibility gate (incl. the unparseable case)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('version, expected', [
    ('0.5.0', True),    # exactly the floor
    ('0.9.7', True),    # in range
    ('0.5.0.dev3+gabc', True),   # leading semver parses; suffix ignored
    ('0.4.9', False),   # below floor
    ('1.0.0', False),   # major >= max major
    ('2.3.1', False),   # well above
])
def test_version_compatible_parseable(version, expected):
    assert _bngsim_caps._version_compatible(version) is expected


@pytest.mark.parametrize('version', [None, '', 'unknown', '1.0', '2024.03'])
def test_unparseable_version_warns_and_accepts(version, caplog):
    """An unparseable version string is a packaging quirk, not proof of
    incompatibility: warn-and-continue (return True) rather than fail-closed,
    so a working install isn't bricked over a format the regex can't read."""
    with caplog.at_level(logging.WARNING, logger='pybnf._bngsim_caps'):
        assert _bngsim_caps._version_compatible(version) is True
    assert any('Could not parse bngsim version' in r.message for r in caplog.records)
