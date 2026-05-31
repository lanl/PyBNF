"""Centralized BNGsim capability detection.

PyBNF imports ``bngsim`` once and routes every backend/feature check through
``bngsim.capabilities()`` (the stable public API introduced in bngsim 0.5.0).
Other PyBNF modules should consume the module-level flags exported here
rather than reach for ``getattr(bngsim, ...)`` or ``hasattr(...)`` probes.
"""


import importlib.metadata
import logging
import os
import re


logger = logging.getLogger(__name__)

# Required floor: 0.5.0 is the release that exposes bngsim.capabilities(),
# bngsim.SimulationTimeout, bngsim.StopConditionMet, Simulator.run_batch,
# {Nfsim,RuleMonkey}Session(..., molecule_limit=...), and the
# timeout= kwarg on every Simulator/Session entry point.
_BNGSIM_MIN_VERSION = (0, 5, 0)
_BNGSIM_MAX_MAJOR = 1


def _parse_version(version):
    if not version:
        return None
    match = re.match(r'^\s*(\d+)\.(\d+)\.(\d+)', version)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _version_compatible(version):
    parsed = _parse_version(version)
    if parsed is None:
        # Couldn't parse a MAJOR.MINOR.PATCH out of the reported version. Warn
        # and accept rather than reject: an unparseable string is far more
        # likely a packaging-format quirk than a genuinely incompatible build,
        # and fail-closed here would brick an otherwise-working install.
        logger.warning(
            'Could not parse bngsim version %r; proceeding without a '
            'compatibility check (PyBNF expects bngsim>=%s,<%d).',
            version, _min_version_str(), _BNGSIM_MAX_MAJOR,
        )
        return True
    return _BNGSIM_MIN_VERSION <= parsed and parsed[0] < _BNGSIM_MAX_MAJOR


def _detect_version(module):
    version = getattr(module, '__version__', None)
    if version:
        return version
    try:
        return importlib.metadata.version('bngsim')
    except importlib.metadata.PackageNotFoundError:
        return None


def _min_version_str():
    return '%d.%d.%d' % _BNGSIM_MIN_VERSION


def _empty_capabilities():
    return {
        'version': None,
        'features': {
            'nfsim': False,
            'rulemonkey': False,
            'libsbml': False,
            'antimony': False,
            'sbml_import': False,
            'sbml_ssa': False,
            'sbml_psa': False,
            'antimony_import': False,
            'codegen': False,
        },
        'missing': {},
    }


bngsim = None
BNGSIM_AVAILABLE = False
BNGSIM_VERSION = None
BNGSIM_ERROR = ''
_capabilities = _empty_capabilities()

try:
    if os.environ.get('PYBNF_NO_BNGSIM'):
        raise ImportError('PYBNF_NO_BNGSIM set')
    import bngsim as _bngsim_module
    BNGSIM_VERSION = _detect_version(_bngsim_module)
    if not _version_compatible(BNGSIM_VERSION):
        raise ImportError(
            'installed bngsim version %s is incompatible; '
            'PyBNF requires bngsim>=%s,<%d'
            % (BNGSIM_VERSION, _min_version_str(), _BNGSIM_MAX_MAJOR)
        )
    if not hasattr(_bngsim_module, 'capabilities'):
        raise ImportError(
            'installed bngsim %s does not expose bngsim.capabilities(); '
            'PyBNF requires bngsim>=%s,<%d'
            % (BNGSIM_VERSION, _min_version_str(), _BNGSIM_MAX_MAJOR)
        )
    bngsim = _bngsim_module
    _capabilities = bngsim.capabilities()
    BNGSIM_AVAILABLE = True
except ImportError as exc:
    BNGSIM_ERROR = str(exc) or 'bngsim is not available'


BNGSIM_FEATURES = dict(_capabilities.get('features', {}))
BNGSIM_MISSING = dict(_capabilities.get('missing', {}))

# Convenience flags. Names match BNGsim's feature keys where they overlap;
# BNGSIM_HAS_SBML / BNGSIM_HAS_ANTIMONY track end-to-end import readiness
# (compiled support + Python dependency installed), since that is the
# question PyBNF callers actually care about.
BNGSIM_HAS_NFSIM = bool(BNGSIM_FEATURES.get('nfsim', False))
BNGSIM_HAS_RULEMONKEY = bool(BNGSIM_FEATURES.get('rulemonkey', False))
BNGSIM_HAS_LIBSBML = bool(BNGSIM_FEATURES.get('libsbml', False))
BNGSIM_HAS_ANTIMONY_PY = bool(BNGSIM_FEATURES.get('antimony', False))
BNGSIM_HAS_SBML = bool(BNGSIM_FEATURES.get('sbml_import', False))
BNGSIM_HAS_ANTIMONY = bool(BNGSIM_FEATURES.get('antimony_import', False))
BNGSIM_HAS_CODEGEN = bool(BNGSIM_FEATURES.get('codegen', False))


def feature_missing_reason(name):
    """Return BNGsim's explanation for a missing feature, or '' if available.

    Falls back to a generic message when bngsim itself isn't importable.
    """
    if not BNGSIM_AVAILABLE:
        return BNGSIM_ERROR or 'bngsim is not available'
    if BNGSIM_FEATURES.get(name, False):
        return ''
    return BNGSIM_MISSING.get(
        name,
        'bngsim feature %r is unavailable in this install' % name,
    )


BNGSIM_SBML_ERROR = feature_missing_reason('sbml_import') if BNGSIM_AVAILABLE else BNGSIM_ERROR
BNGSIM_ANTIMONY_ERROR = (
    feature_missing_reason('antimony_import') if BNGSIM_AVAILABLE else BNGSIM_ERROR
)
