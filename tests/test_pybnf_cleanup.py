"""ROB-4: main()'s post-run cleanup/exit must not mask a propagating exception.

``pybnf.pybnf._finalize`` runs inside ``main()``'s outer ``finally``. A failure
inside ``alg.cleanup()`` is logged and swallowed -- but only if it is an
ordinary ``Exception``. A ``KeyboardInterrupt``/``SystemExit`` raised during
cleanup must propagate, not be masked by the ``exit()`` call (which previously
sat in a ``finally`` block that swallowed any in-flight exception).
"""

import pytest

from pybnf.pybnf import _finalize


class _StubAlg:
    """Stands in for the fitting algorithm; records whether cleanup ran and
    optionally raises a chosen exception from cleanup()."""

    def __init__(self, exc=None):
        self._exc = exc
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True
        if self._exc is not None:
            raise self._exc


def test_finalize_exits_zero_on_success():
    # On success, cleanup is skipped and the process exits 0.
    alg = _StubAlg(exc=RuntimeError("should never run"))
    with pytest.raises(SystemExit) as exc_info:
        _finalize(success=True, alg=alg, start_time=0.0)
    assert exc_info.value.code == 0
    assert alg.cleaned is False


def test_finalize_exits_one_on_failure():
    alg = _StubAlg()
    with pytest.raises(SystemExit) as exc_info:
        _finalize(success=False, alg=alg, start_time=0.0)
    assert exc_info.value.code == 1
    assert alg.cleaned is True


def test_finalize_swallows_ordinary_cleanup_exception():
    """An ordinary Exception during cleanup is logged, not propagated, so the
    process still exits with the failure code."""
    alg = _StubAlg(exc=RuntimeError("cleanup boom"))
    with pytest.raises(SystemExit) as exc_info:
        _finalize(success=False, alg=alg, start_time=0.0)
    assert exc_info.value.code == 1


def test_finalize_does_not_mask_keyboard_interrupt():
    """ROB-4 core: a KeyboardInterrupt during cleanup must propagate rather than
    being caught by a bare ``except:`` and then masked into ``SystemExit`` by an
    ``exit()`` sitting in a ``finally``. Mutation-distinguishing test: the old
    inline form (bare except + exit-in-finally) raised SystemExit here."""
    alg = _StubAlg(exc=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _finalize(success=False, alg=alg, start_time=0.0)
