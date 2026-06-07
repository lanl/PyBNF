"""Guard tests for the ``pybnf.bngsim_model`` capability-flag seam (ADR-0018).

The ``bngsim_model`` package routes every read of the bngsim capability flags
(``bngsim``, ``BNGSIM_AVAILABLE``, ``BNGSIM_HAS_NFSIM``/``RULEMONKEY``, …) through
``pybnf.bngsim_model._runtime`` so that a single ``monkeypatch.setattr`` on that
module bites every reader, no matter which submodule it lives in. ``monkeypatch``
only reaches a reader that resolves the name in the *patched* namespace, so if a
future split relocates a flag-reader to a submodule that reads a bare name
(``BNGSIM_HAS_NFSIM``) instead of ``_runtime.BNGSIM_HAS_NFSIM``, the production
patches in ``test_bngsim_bridge.py`` would silently stop biting — a green suite
testing nothing (the exact ADR-0001 hazard).

These tests patch the seam and assert the production path actually used the fake.
They are deliberately *not* marked ``bngsim``: they fake the package entirely and
must run on the bngsim-less CI tier, where they are the tripwire.
"""

from types import SimpleNamespace

import pytest

import pybnf.bngsim_model as bngsim_model


def test_runtime_seam_bngsim_package_read_bites(monkeypatch):
    """Patching ``_runtime.bngsim`` must reach the normalize-method reader."""
    sentinel = ('sentinel_canonical', object())
    fake = SimpleNamespace(normalize_method=lambda method: sentinel)
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim', fake)
    assert bngsim_model._bngsim_normalize_method('whatever') == sentinel


def test_runtime_seam_available_flag_gate_bites(monkeypatch):
    """Patching ``_runtime.BNGSIM_AVAILABLE`` must reach the availability gate."""
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_AVAILABLE', False)
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_ERROR', 'seam-test-error')
    with pytest.raises(ValueError, match='seam-test-error'):
        bngsim_model._bngsim_normalize_method('ode')


@pytest.mark.parametrize(
    'backend, flag',
    [
        (bngsim_model.BNGSIM_NF_BACKEND_NFSIM, 'BNGSIM_HAS_NFSIM'),
        (bngsim_model.BNGSIM_NF_BACKEND_RULEMONKEY, 'BNGSIM_HAS_RULEMONKEY'),
    ],
)
def test_runtime_seam_nf_capability_flags_bite(monkeypatch, backend, flag):
    """Patching an NF capability flag on ``_runtime`` must reach its reader."""
    monkeypatch.setattr(bngsim_model._runtime, flag, False)
    assert bngsim_model._bngsim_has_nf_session_backend(backend) is False
    monkeypatch.setattr(bngsim_model._runtime, flag, True)
    assert bngsim_model._bngsim_has_nf_session_backend(backend) is True
