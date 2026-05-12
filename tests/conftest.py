import importlib.util
import shutil
from pathlib import Path

import pytest

from pybnf import _bngsim_caps


_ROOT_DIR = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
_BNGL_LINK = _ROOT_DIR / 'bngl_files'
_created_bngl_link = False


def pytest_sessionstart(session):
    """Expose tests/bngl_files at the repo root for legacy relative-path tests."""
    del session
    global _created_bngl_link

    if not _BNGL_LINK.exists():
        _BNGL_LINK.symlink_to(_TESTS_DIR / 'bngl_files', target_is_directory=True)
        _created_bngl_link = True


def pytest_sessionfinish(session, exitstatus):
    del session, exitstatus
    if _created_bngl_link and _BNGL_LINK.is_symlink():
        _BNGL_LINK.unlink()


def _marker_skip_reasons():
    """Map marker name → skip reason string (None means dependency present)."""
    reasons = {}

    if not _bngsim_caps.BNGSIM_AVAILABLE:
        reasons['bngsim'] = _bngsim_caps.BNGSIM_ERROR or 'bngsim is not available'
    if not _bngsim_caps.BNGSIM_HAS_NFSIM:
        reasons['bngsim_nfsim'] = 'bngsim NFsim backend is not available'
    if not _bngsim_caps.BNGSIM_HAS_RULEMONKEY:
        reasons['bngsim_rulemonkey'] = 'bngsim RuleMonkey backend is not available'
    if not _bngsim_caps.BNGSIM_HAS_SBML:
        reasons['bngsim_sbml'] = (
            _bngsim_caps.BNGSIM_SBML_ERROR
            or 'bngsim SBML backend is not available'
        )
    if not _bngsim_caps.BNGSIM_HAS_ANTIMONY:
        reasons['bngsim_antimony'] = (
            _bngsim_caps.BNGSIM_ANTIMONY_ERROR
            or 'bngsim Antimony backend is not available'
        )
    if importlib.util.find_spec('roadrunner') is None:
        reasons['roadrunner'] = 'libroadrunner is not installed'
    if shutil.which('BNG2.pl') is None:
        reasons['bionetgen'] = 'BNG2.pl is not on PATH'

    return reasons


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose marker's dependency isn't satisfied.

    Centralises the BNGSIM_HAS_X / roadrunner / BNG2.pl checks so individual
    tests can just declare ``@pytest.mark.bngsim_sbml`` etc. without each
    re-deriving the skip condition. The dispatch table lives here (and
    imports the capability flags from ``pybnf._bngsim_caps``) so adding a
    new dependency-conditional marker is a one-line change.
    """
    del config
    reasons = _marker_skip_reasons()
    if not reasons:
        return
    for item in items:
        for marker in item.iter_markers():
            reason = reasons.get(marker.name)
            if reason is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
