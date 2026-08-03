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
            'output_sensitivities': False,
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
# Forward output sensitivities (∂g/∂θ) from the ODE backend, consumed by the
# gradient-plumbing path (#385/#447). Gates the gradient *path*, not the install:
# the version floor stays 0.5.0, so a build without this feature still runs every
# scalar (metaheuristic) fit unchanged -- only gradient-based fitting refuses.
BNGSIM_HAS_OUTPUT_SENS = bool(BNGSIM_FEATURES.get('output_sensitivities', False))
# Steady-state (KINSOL/Newton) forward sensitivities exposed at the
# observable/expression level on ``SteadyStateResult.output_sensitivities``
# (bngsim>=0.11.35, lanl/bngsim#12). ``capabilities()`` has no dedicated feature
# key for it, so probe the type directly here -- the one place PyBNF centralizes
# the getattr/hasattr shim the module docstring reserves for this module -- and
# export a clean flag. Gates only the *gradient* Newton dose-response scan path
# (#478): a scalar Newton scan, and every non-scan gradient fit, are unaffected.
BNGSIM_HAS_SS_OUTPUT_SENS = bool(
    BNGSIM_AVAILABLE
    and hasattr(getattr(bngsim, 'SteadyStateResult', None), 'output_sensitivities')
)
# ``Simulator.parameter_scan`` / ``bifurcate`` carry the state each point starts from
# TOGETHER WITH its ``dx/dθ`` (bngsim>=0.12.0, lanl/bngsim#81), and resolve the
# ``on_point`` hook's own ``∂x(0)/∂θ`` row by row (lanl/bngsim#111). Both landed before
# the 0.12.0 release, and ``Model.declare_ic_sensitivity`` -- the public API #111 added --
# is present exactly when they are, so it is the probe. ``capabilities()`` has no feature
# key for it, so this is the second (and last) direct type probe this module owns. Gates
# only the *gradient* carried-state (pre-equilibrated) dose-response scan (#532): a scalar
# carried-state scan, and every fresh-from-seed gradient scan, are unaffected.
BNGSIM_HAS_SCAN_SENS_CARRY = bool(
    BNGSIM_AVAILABLE
    and hasattr(getattr(bngsim, 'Model', None), 'declare_ic_sensitivity')
)
# Forward sensitivities that survive a DISCRETE EVENT -- the jump
#
#     s+ = dh/dx . (s- + f-.dt*/dp) + dh/dp - f+.dt*/dp
#
# bngsim applies at each fire, for the event subclasses it can classify (#536).
# The floor is set by the last build that could return a wrong jump *without
# saying so*, because that -- not a missing feature -- is what this flag guards
# against. Three separate silent derivatives had to go first:
#
# * a trigger that reads the state through an SBML document (``S < 30``) was
#   neither refused nor differentiated, returning a finite tensor missing the
#   event's contribution (lanl/bngsim#52, fixed in 0.12.0);
# * an event assignment that *reads* the state (``A := A + dose`` -- the
#   repeat-dosing idiom) lost the carried term dh/dx.s-, so the assigned row
#   restarted from zero: measured on 0.12.1 as -10.96 where the model's own
#   central difference says -311.20, while the same model built through
#   ``ModelBuilder.add_event`` was right to 2e-6. Fixed as a side effect of
#   lanl/bngsim#144's jump-handler rework, after 0.12.1;
# * a CVODE root that fires *nothing* -- a discontinuity root, or an event root
#   that crossed without rising -- rewound the state but not the sensitivity
#   history, injecting a spurious step into every column (lanl/bngsim#146),
#   also after 0.12.1.
#
# So the floor is "newer than 0.12.1", which every later release satisfies
# whether the next one bumps the minor or the major. What the qualifying build
# then *supports* is a separate, wider question -- a fixed trigger time
# (bngsim GH #212), a threshold that is a fitted constant (lanl/bngsim#49), a
# state-dependent trigger differentiated in flight (lanl/bngsim#144) -- and it
# refuses the rest per simulation rather than guessing, which is exactly why
# PyBNF can stop classifying events itself.
#
# ``capabilities()`` has no feature key to read here -- it reports compiled
# backends and build options, and what separates these builds is a set of bug
# fixes -- so this is a version floor rather than a probe. An unparseable
# version reads as *absent* (fail closed): elsewhere in this module an
# unparseable version is accepted, because rejecting it would brick an
# otherwise-working install, but here the cost of guessing wrong is a wrong
# gradient rather than a refusal.
_BNGSIM_EVENT_SENS_VERSION = (0, 12, 2)
BNGSIM_HAS_EVENT_SENS = bool(
    BNGSIM_AVAILABLE
    and (_parse_version(BNGSIM_VERSION) or ()) >= _BNGSIM_EVENT_SENS_VERSION
)


def event_sens_min_version():
    """The bngsim version whose forward sensitivities survive a discrete event (#536)."""
    return '%d.%d.%d' % _BNGSIM_EVENT_SENS_VERSION


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
        f'bngsim feature {name!r} is unavailable in this install',
    )


BNGSIM_SBML_ERROR = feature_missing_reason('sbml_import') if BNGSIM_AVAILABLE else BNGSIM_ERROR
BNGSIM_ANTIMONY_ERROR = (
    feature_missing_reason('antimony_import') if BNGSIM_AVAILABLE else BNGSIM_ERROR
)
