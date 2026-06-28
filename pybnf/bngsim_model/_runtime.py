"""Mutable mirror of the bngsim capability flags for the ``bngsim_model`` package.

The flags originate in :mod:`pybnf._bngsim_caps` (detected once at import). Every
flag-reader in this package resolves them as ``_runtime.<name>`` at call time, so
a single ``monkeypatch.setattr(pybnf.bngsim_model._runtime, "bngsim", fake)`` (or
``BNGSIM_AVAILABLE`` / ``BNGSIM_HAS_NFSIM`` / …) bites *every* reader, no matter
which submodule it lives in. This is the package's seam target (ADR-0018): the
test patch follows the name to where it is read, instead of the readers reaching
back to a package-facade binding (ADR-0001 rejects that magic indirection).

Keep this module a thin re-export of ``_bngsim_caps`` with no logic of its own.
"""


from .. import _bngsim_caps


bngsim = _bngsim_caps.bngsim
BNGSIM_AVAILABLE = _bngsim_caps.BNGSIM_AVAILABLE
BNGSIM_ERROR = _bngsim_caps.BNGSIM_ERROR
BNGSIM_HAS_NFSIM = _bngsim_caps.BNGSIM_HAS_NFSIM
BNGSIM_HAS_RULEMONKEY = _bngsim_caps.BNGSIM_HAS_RULEMONKEY
BNGSIM_HAS_OUTPUT_SENS = _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS
BNGSIM_VERSION = _bngsim_caps.BNGSIM_VERSION
feature_missing_reason = _bngsim_caps.feature_missing_reason
