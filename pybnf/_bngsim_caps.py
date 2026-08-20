"""Centralized BNGsim capability detection.

PyBNF imports ``bngsim`` once and routes every backend/feature check through
``bngsim.capabilities()`` (the stable public API introduced in bngsim 0.5.0).
Other PyBNF modules should consume the module-level flags exported here
rather than reach for ``getattr(bngsim, ...)`` or ``hasattr(...)`` probes.
"""


import collections
import contextlib
import ctypes
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
# The line this flag draws is set by the last build that could return a wrong
# jump *without saying so*, because that -- not a missing feature -- is what it
# guards against. Three separate silent derivatives had to go first:
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
# What a qualifying build then *supports* is a separate, wider question -- a
# fixed trigger time (bngsim GH #212), a threshold that is a fitted constant
# (lanl/bngsim#49), a state-dependent trigger differentiated in flight
# (lanl/bngsim#144) -- and it refuses the rest per simulation rather than
# guessing, which is exactly why PyBNF can stop classifying events itself.
#
# HOW THE LINE IS DRAWN (#558). It used to be drawn by a version floor at
# exactly 0.12.2, and a version floor is the wrong instrument for it: the three
# fixes above are behaviour, and a build's version string is a claim about
# behaviour that a from-source build does not have to honour. bngsim bumps
# ``__version__`` at the start of a release cycle, so every dev build between
# that bump and the fixes *also* declares 0.12.2 -- and a floor reports the
# capability PRESENT on one, which is the expensive direction: a gradient that
# is wrong rather than a refusal. The neighbouring ``BNGSIM_HAS_PER_SPECIES_ATOL``
# comment states the same hazard for the same version string, and it cost a real
# fit (#558). "Probe capability, never version."
#
# So the flag resolves through three routes, first match wins, and each is
# reported by :func:`event_sens_probe` so a refusal can say how it decided:
#
# 1. ``features['event_sensitivities']`` -- a dedicated key. bngsim publishes no
#    such key today; naming it here costs nothing and means the flag starts
#    reading the real answer, in BOTH directions, on the first build that grows
#    one. That is #558's ask 1, and it is why this route is checked first even
#    though it never fires yet.
#
# 2. ``features['effective_ic_sensitivity']`` -- a WITNESS. This key is not the
#    capability; it reports ``Model.effective_ic_sensitivity``, the dx(0)/dtheta
#    reader ADR-0100 consumes. It is usable here because of WHERE it landed:
#    lanl/bngsim#155 added it three commits after #146 and seven after #144, inside
#    the same 0.12.1 -> 0.12.2 window, so a build that publishes it necessarily
#    carries every fix above. The implication runs one way only, and that is the
#    safe way -- a build from the handful of commits between #146 and #155 has
#    the fixes and lacks the witness, and is refused. A refusal on a build that
#    would have been right is a metaheuristic fit; the converse is a wrong
#    gradient nobody sees. Being a published ``capabilities()`` key rather than a
#    ``hasattr`` also means bngsim's own "existing names will not be renamed or
#    removed" contract covers it.
#
# 3. Neither key published -- absent. The version floor survives ONLY as a
#    conjunct on route 2, where it vetoes an incoherent build; it can no longer
#    report *present* on its own. The witness shipped IN 0.12.2, so a build
#    claiming 0.12.2-or-newer that does not publish it is precisely the lying
#    pre-release build, and a build below 0.12.2 was refused before this change
#    too. Every released bngsim at or above the floor publishes the witness, so
#    no install that works today is refused by this change.
#
# An unparseable version reads as *absent* (fail closed): elsewhere in this
# module an unparseable version is accepted, because rejecting it would brick an
# otherwise-working install, but here the cost of guessing wrong is a wrong
# gradient rather than a refusal.
_BNGSIM_EVENT_SENS_VERSION = (0, 12, 2)
_BNGSIM_EVENT_SENS_FEATURE = 'event_sensitivities'
_BNGSIM_EVENT_SENS_WITNESS = 'effective_ic_sensitivity'


def _feature_key(name):
    """Tri-state read of a bngsim feature key: the flag, or ``None`` if unpublished.

    ``BNGSIM_FEATURES.get(name, False)`` cannot tell a build that says "no" from a
    build too old to have been asked, and that distinction is the whole of #558:
    an unpublished key must fall through to the next route, while a published
    ``False`` is an answer and must be honoured.
    """
    if not BNGSIM_AVAILABLE or name not in BNGSIM_FEATURES:
        return None
    return bool(BNGSIM_FEATURES[name])


def _resolve_event_sens():
    """``(present, route)`` for :data:`BNGSIM_HAS_EVENT_SENS` -- see the block above."""
    if not BNGSIM_AVAILABLE:
        return False, 'bngsim is not available'
    declared = _feature_key(_BNGSIM_EVENT_SENS_FEATURE)
    if declared is not None:
        return declared, "bngsim feature key %r" % _BNGSIM_EVENT_SENS_FEATURE
    floor_ok = (_parse_version(BNGSIM_VERSION) or ()) >= _BNGSIM_EVENT_SENS_VERSION
    witness = _feature_key(_BNGSIM_EVENT_SENS_WITNESS)
    if witness is not None:
        if witness and not floor_ok:
            # The witness says yes and the version cannot corroborate it. Not a
            # coherent build; the veto stands, and the message has to name the
            # veto rather than the witness or it describes the wrong evidence.
            return False, (
                "bngsim feature key %r is published, but the reported version (%s) "
                "is below %s, so the build contradicts itself"
                % (_BNGSIM_EVENT_SENS_WITNESS, BNGSIM_VERSION or 'unparseable',
                   '%d.%d.%d' % _BNGSIM_EVENT_SENS_VERSION)
            )
        return bool(witness), (
            "bngsim feature key %r, which is published only by a build that also "
            "carries the event-sensitivity fixes" % _BNGSIM_EVENT_SENS_WITNESS
        )
    return False, (
        "no bngsim feature key to read (%r is unpublished by this build), and a "
        "version floor alone is not evidence of a behavioural fix"
        % _BNGSIM_EVENT_SENS_WITNESS
    )


BNGSIM_HAS_EVENT_SENS, _BNGSIM_EVENT_SENS_ROUTE = _resolve_event_sens()


# A **per-species** absolute tolerance: ``Simulator.run(atol=...)`` accepts a vector and
# routes it to ``CVodeSVtolerances`` (lanl/bngsim#196), so a model spanning ten decades no
# longer has to pick one number for both ends. ADR-0103 derived a scalar because that was
# the whole of what the backend offered; ADR-0105 derives the vector.
#
# The probe is a name, not a version. The build that first carried #196 still declares
# 0.12.2 -- the same string as the released wheel 25 commits behind it -- so a version
# floor here would report *present* on an install that does not have it, and that is the
# expensive direction: the vector would be handed to a ``run`` that takes only a scalar.
# ``AUTO`` and ``normalize_atol_vector`` are exported from the package namespace by
# lanl/bngsim#212 and both are listed in ``__all__``; probing the two names PyBNF
# actually calls keeps the flag honest if either ever moves. Absent, the SBML backend
# keeps ADR-0103's scalar bit-for-bit, so an older bngsim runs every fit it runs today.
BNGSIM_HAS_PER_SPECIES_ATOL = bool(
    BNGSIM_AVAILABLE
    and hasattr(bngsim, 'AUTO')
    and hasattr(bngsim, 'normalize_atol_vector')
)


# An absolute tolerance that follows the **trajectory** rather than the initial state:
# ``Simulator.run(atol=TrackingAtol(...))`` installs a ``CVodeWFtolerances`` error-weight
# function computing ``clamp(rtol*|y_i|, ceiling_i * 10**-decades, ceiling_i)`` at the
# state actually being integrated (lanl/bngsim#213). A vector read off initial values
# cannot see a species that starts at order one and decays to nothing; this can, and it
# is what ADR-0105 named as the half it could not reach.
#
# A name probe again, and for the same reason as ``BNGSIM_HAS_PER_SPECIES_ATOL``: the
# capability arrived without a version bump that identifies it, and the cost of guessing
# wrong is a ``TrackingAtol`` handed to a ``run`` that does not know the type. ``#557``
# refuses ``sbml_atol = tracking`` outright when this is False rather than quietly
# integrating at something else -- a tolerance mode that silently did not apply is the
# failure that looks like a modelling result.
BNGSIM_HAS_TRACKING_ATOL = bool(
    BNGSIM_AVAILABLE and hasattr(bngsim, 'TrackingAtol')
)


def event_sens_min_version():
    """The first bngsim RELEASE whose forward sensitivities survive a discrete event (#536).

    A release number is the actionable half of the refusal -- it is what a reader
    installs -- but it is no longer what :data:`BNGSIM_HAS_EVENT_SENS` decides on,
    because a from-source build can declare it without carrying it (#558). Pair it
    with :func:`event_sens_probe` when a message has to explain a *refusal*, so a
    reader who already has this version learns why it did not count.
    """
    return '%d.%d.%d' % _BNGSIM_EVENT_SENS_VERSION


def event_sens_probe():
    """How :data:`BNGSIM_HAS_EVENT_SENS` was decided, as a phrase for a message (#558).

    Names the route that actually answered -- a dedicated feature key, the
    ``effective_ic_sensitivity`` witness, or neither -- so a refusal on a build
    whose version *looks* new enough says which evidence it was missing instead of
    reading as a version complaint the reader has already satisfied.
    """
    return _BNGSIM_EVENT_SENS_ROUTE


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


# --- compiled-core provenance (#558) ---------------------------------------- #
#
# A version floor is blind to two different things, and the section above deals
# with only the first. Which COMMITS the Python layer carries is one; whether the
# COMPILED layer matches its own C++ sources is the other, and it is worse,
# because nothing in the Python layer moves at all. An editable bngsim serves live
# Python from the source tree but loads ``_bngsim_core*.so`` from a separately
# built artifact with auto-rebuild deliberately off (lanl/bngsim#23), so the two
# halves drift: an install reporting a perfectly good version has been observed
# with a core binary three days older than the ``.cpp`` next to it. Every
# version-, metadata- and feature-key-based check passes there.
#
# bngsim already detects this and warns at import. The problem is WHEN: PyBNF
# imports bngsim while loading its own package, which is before ``init_logging``
# has run and long before a fit is a commitment, so the warning lands in import
# noise on a terminal nobody is reading yet. A fit is hours; the warning is worth
# one line at job start, where it is still cheap to abort.
#
# ``bngsim._build_provenance`` is a private module, so every read here is guarded
# and every failure is silent -- an install that cannot answer is reported as "no
# opinion", never as an error. This is the module that owns such shims. The reads
# are also LAZY and memoized: ``gather()`` walks the whole C++ source tree
# (src/, include/, third_party/, cmake/), which is not a cost to pay at import on
# behalf of callers who never ask.

_provenance = None


def _bngsim_provenance():
    """bngsim's own build-provenance snapshot, or ``None`` when it cannot say.

    Memoized, including the negative answer: a wheel install has no ``src/`` to
    walk, and re-deciding that per call would still pay for the import attempt.
    """
    global _provenance
    if _provenance is None:
        _provenance = (False, None, None)
        if BNGSIM_AVAILABLE:
            try:
                from bngsim import _build_provenance
                _provenance = (True, _build_provenance,
                               _build_provenance.gather(include_head=False))
            except Exception:
                # Private module, absent on some installs, and a filesystem walk:
                # three independent ways to fail, none of them worth a traceback
                # in front of a user who asked for a fit.
                logger.debug('bngsim build provenance is unavailable', exc_info=True)
    ok, module, prov = _provenance
    return (module, prov) if ok else (None, None)


def bngsim_build_id():
    """The commit the loaded ``_bngsim_core`` was BUILT from, or ``''``.

    The identifier #558 asks for: two installs can declare the same version and
    be different builds, and this is the only thing on hand that tells them apart.
    Baked into the extension by bngsim's CMake, so it describes the binary rather
    than the package metadata -- which is exactly the distinction a version string
    cannot make. ``''`` for a wheel built without it.
    """
    _, prov = _bngsim_provenance()
    return getattr(prov, 'build_commit', None) or ''


def bngsim_identity_line():
    """One line naming the loaded compiled core -- path, build commit, mtime, state.

    For the job-start log. ``''`` when bngsim cannot say, so a caller can simply
    skip the line rather than print a placeholder.
    """
    module, prov = _bngsim_provenance()
    if module is None:
        return ''
    try:
        return module.identity_line(prov)
    except Exception:                                  # pragma: no cover - defensive
        return ''


def bngsim_stale_core_report():
    """bngsim's multi-line staleness report, or ``''`` when the core is not stale.

    Truthy exactly when the loaded binary predates its own C++ sources -- i.e.
    when every capability answer this module gives, and every number the fit is
    about to produce, describes code that is no longer in the tree. Honours
    bngsim's own ``BNGSIM_NO_BUILD_CHECK`` opt-out, since that is the user saying
    the heuristic misfires here.
    """
    module, prov = _bngsim_provenance()
    if module is None or prov is None:
        return ''
    try:
        if module._checks_disabled() or not prov.is_stale:
            return ''
        return module.format_report(prov)
    except Exception:                                  # pragma: no cover - defensive
        return ''


# --- the analytic ∂f/∂p, per (build, MODEL) (#606) --------------------------- #
#
# Everything above this line is a property of the INSTALL. This is not: whether a
# gradient runs on bngsim's analytic sensitivity RHS or on CVODES' internal
# difference quotient is decided per model, at codegen, and two models on one
# bngsim get different answers.
#
# ``CVodeSensInit1`` takes ONE sensitivity-RHS callback for every column, so a
# single rate law bngsim cannot differentiate declines the analytic ∂f/∂p for the
# WHOLE model -- there is no per-reaction fallback to mix in. The difference
# quotient that replaces it costs an extra RHS evaluation per column per step, so
# an N-parameter fit pays roughly N times the sensitivity cost. That is not an
# annoyance on a fit measured in hours: on ``Smith_BMCSystBiol2013`` all 25 columns
# fell back, every start timed out to ``inf``, and thirteen hours produced nothing
# (#558). The only signal was a bngsim log line nobody had a reason to look for.
#
# WHY THE LOG LINE IS NOT THE INSTRUMENT. bngsim reports every decline on the
# ``bngsim`` logger at codegen time, and PyBNF could listen for it -- that needs no
# new bngsim API and works across the whole supported range. It is not enough on
# its own, because the codegen cache short-circuits the step that emits it: since
# lanl/bngsim#174 the cache key is STRUCTURAL, so a warm cache resolves the ``.so``
# without generating any source, and source generation is where the decline is
# derived and logged. Measured against an ``abs()`` rate law on 0.14.0 and 0.13.0
# alike: first construction reports the decline, a second construction of the same
# model in the same process reports nothing and is on the same fallback. The cache is on disk
# and persists across runs, so the run that hears nothing is typically the SECOND
# run of a fit -- exactly the one made after the first came back empty.
#
# Silence on that logger therefore means "declined, or served from cache". It can
# support a statement that a model IS on the fallback and never one that it is not.
#
# WHAT IS. The compiled artifact either exports ``bngsim_codegen_sens_rhs`` or it
# does not, and that is the exact symbol bngsim's C++ resolves with ``try_symbol``
# to choose the analytic RHS over the difference quotient. Reading it back off the
# artifact answers the question for the run that is actually about to happen,
# cache hit or not, and it is available on every build in the pin.
#
# So the two channels are used for different halves of the report, and
# :func:`analytic_sens_rhs_probe` / :func:`capture_sens_rhs_declines` are that
# split: the artifact carries the VERDICT (stable, so a policy may key off it), and
# the logger carries the REASON (best-effort, so only the prose may key off it).
#
# The probe resolves through four routes, first match wins, and reports which one
# answered -- the same ladder, and the same "prefer what the build publishes"
# rule, as ``_resolve_event_sens`` above.
_SENS_RHS_SYMBOL = 'bngsim_codegen_sens_rhs'
# Route 1. A public per-run answer, if bngsim ever publishes one. No build exposes
# this attribute today, so naming it costs nothing and means PyBNF reads the real
# answer -- in BOTH directions -- on the first build that grows one, with no PyBNF
# release in between. lanl/bngsim#431 is the standing ask it would arrive under.
_SENS_RHS_PUBLIC_ATTR = 'has_analytic_sens_rhs'
# Route 2. bngsim's own ground truth, private and >= 0.14.0. Preferred over route 3
# wherever it exists, because it is upstream's answer to upstream's question: if the
# artifact ever stops being where the symbol lives, a build carrying this method
# keeps answering correctly while route 3 would not.
_SENS_RHS_OWNED_METHOD = '_codegen_provides_sens_rhs'

# bngsim phrases every decline as "Forward sensitivity: <reason>, so the analytic
# sensitivity RHS is declined ..." (or "... <reason>. The analytic sensitivity RHS
# is declined ..." for the variant that also says the fallback is wrong). Match the
# clause that is common to both and keep the reason; a message that does not match
# is kept whole rather than dropped, since an unrecognized decline is still a
# decline and the wording is not PyBNF's to depend on.
_SENS_RHS_DECLINE_MARK = 'analytic sensitivity rhs is declined'
_SENS_RHS_DECLINE_RE = re.compile(
    r'Forward sensitivity:\s*(?P<reason>.*?)[.,]\s*(?:so\s+)?[Tt]he analytic '
    r'sensitivity RHS is declined',
    re.DOTALL,
)
# The half of the decline space where "correct, but slower" is FALSE: the model
# branches at a crossing whose time moves, the difference quotient integrates the
# variational equation straight through it, and every column is wrong at and after
# the crossing by the dropped saltation term. From 0.14.0 bngsim REFUSES such a run
# (``SensitivityUnsupportedError``, lanl/bngsim#414/#416) rather than return a
# gradient it has flagged as wrong, so there it reaches PyBNF as an ordinary error
# and needs nothing here; on 0.13.0, which PyBNF's floor still admits, the same model
# only warns and returns the wrong gradient. This phrase is how the variant
# identifies itself on the builds where it is still only a warning.
_SENS_RHS_UNCOMPENSATED_MARK = 'does not recover'


class SensRhsStatus(collections.namedtuple(
        'SensRhsStatus', 'analytic route reasons columns')):
    """One model's answer to "is this fit's gradient on the analytic ``∂f/∂p``?" (#606).

    ``analytic`` is the tri-state verdict :func:`analytic_sens_rhs_probe` reached
    (``True`` / ``False`` / ``None`` for no opinion) and ``route`` names the evidence
    that reached it. ``reasons`` is whatever bngsim said on its own logger while the
    Simulator was being built -- ``(reason, fallback_is_wrong)`` pairs, and **empty
    is not an answer**: a warm codegen cache emits nothing for a model that is on the
    fallback all the same. Only ``analytic`` may be acted on; ``reasons`` only ever
    adds detail to a verdict already reached.

    ``columns`` is how many sensitivity columns this model was asked for, which is
    what turns the verdict into a cost: the difference quotient spends one extra RHS
    evaluation per column per step, so it is the multiplier a reader needs in order
    to decide whether to wait.
    """

    __slots__ = ()

    @property
    def declined(self):
        """True only for a model KNOWN to be on the difference-quotient fallback."""
        return self.analytic is False

    @property
    def fallback_is_wrong(self):
        """True when a captured reason says the difference quotient answers a
        different question -- a branch crossing nobody compensates, where every
        column is wrong at and after it (lanl/bngsim#150/#232).

        Best-effort, like every read of :attr:`reasons`: ``False`` here means "not
        heard", not "not so". bngsim >= 0.14.0 raises on this case rather than
        warning, so it is only reachable on a sub-0.14.0 build with a cold cache.
        """
        return any(wrong for _, wrong in self.reasons)


def probe_sens_rhs(make_simulator, columns=0):
    """Build one sensitivity-bearing Simulator and report what it will run on (#606).

    ``make_simulator`` is a thunk each backend supplies -- the same construction its
    own ODE actions make, so the artifact read is about the artifact the fit will
    actually install; ``columns`` is that model's requested sensitivity width.

    Building the Simulator is the cost of the answer: it is a codegen, which
    on a cold cache is a compile. That compile is one every worker was going to pay
    anyway, content-addressed and shared, so paying it once here mostly moves work
    rather than adding it -- and it buys the answer BEFORE the fit commits hours to
    a gradient it cannot afford.

    Never raises. A model that cannot be prepared for a sensitivity run reports no
    opinion: this is a diagnostic, and a diagnostic that can end a fit is worse than
    no diagnostic. Whatever is genuinely wrong surfaces at the first simulation with
    its own message.
    """
    try:
        with capture_sens_rhs_declines() as reasons:
            sim = make_simulator()
        state, route = analytic_sens_rhs_probe(sim)
        return SensRhsStatus(state, route, list(reasons), int(columns))
    except Exception:
        logger.debug('could not probe the analytic sensitivity RHS', exc_info=True)
        return SensRhsStatus(
            None, 'the model could not be prepared for a sensitivity run', [],
            int(columns))


def analytic_sens_rhs_probe(sim):
    """``(state, route)`` -- is ``sim`` about to run on the analytic ``∂f/∂p``? (#606)

    ``state`` is tri-state on purpose:

    * ``True``  -- the artifact this run installs carries the analytic sensitivity RHS;
    * ``False`` -- it does not, so CVODES' internal difference quotient is used;
    * ``None``  -- **no opinion**. There is no artifact to read (an interpreted RHS,
      ``codegen=False``), or the one there is could not be opened. A caller must
      report nothing rather than guess, in either direction: a false *present* hides
      the cost this whole probe exists to surface, and a false *absent* warns about
      a fit that is fine.

    ``route`` names the evidence that answered, for a message that has to say how it
    decided. Never raises -- a probe that cannot answer says so.
    """
    if sim is None:
        return None, 'no simulator to read'
    published = getattr(sim, _SENS_RHS_PUBLIC_ATTR, None)
    if isinstance(published, bool):
        return published, 'the bngsim Simulator attribute %r' % _SENS_RHS_PUBLIC_ATTR
    owned = getattr(sim, _SENS_RHS_OWNED_METHOD, None)
    if callable(owned):
        try:
            return bool(owned()), "bngsim's own Simulator.%s()" % _SENS_RHS_OWNED_METHOD
        except Exception:                              # pragma: no cover - defensive
            logger.debug('bngsim Simulator.%s() failed', _SENS_RHS_OWNED_METHOD,
                         exc_info=True)
    return _read_sens_rhs_artifact(sim)


def _read_sens_rhs_artifact(sim):
    """Route 3: does the codegen artifact this run installs export the symbol?

    The JIT backends keep the generated C source on the Simulator and compile
    nothing, so the source is checked first; it names the symbol only where it also
    defines the function, which is what makes the substring test equivalent to the
    ``.so`` symbol test. ``ctypes.CDLL`` on the ``.so`` is cheap -- bngsim's core has
    already ``dlopen``\\ ed it, so this resolves from the loader's own table.
    """
    source = getattr(sim, '_codegen_c_source', '') or ''
    if source:
        return (_SENS_RHS_SYMBOL in source), 'the generated C source this run installs'
    path = getattr(sim, '_codegen_so_path', '') or ''
    if path:
        try:
            return (hasattr(ctypes.CDLL(path), _SENS_RHS_SYMBOL),
                    'the compiled codegen artifact this run installs')
        except Exception:
            # Broad on purpose: the usual failure is an OSError from ``dlopen``, but
            # this is a diagnostic, and no way of failing to read a path is worth
            # raising out of one. An unreadable artifact is an unknown, not a verdict.
            logger.debug('could not open the bngsim codegen artifact %s', path,
                         exc_info=True)
            return None, 'the compiled codegen artifact could not be opened'
    return None, 'this run installs no codegen artifact to read'


class _SensRhsDeclineHandler(logging.Handler):
    """Collects bngsim's own decline reasons, and nothing else, off its logger."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.reasons = []

    def emit(self, record):
        try:
            message = record.getMessage()
        except Exception:                              # pragma: no cover - defensive
            return
        if _SENS_RHS_DECLINE_MARK not in message.lower():
            return
        match = _SENS_RHS_DECLINE_RE.search(message)
        reason = match.group('reason').strip() if match else message.strip()
        uncompensated = _SENS_RHS_UNCOMPENSATED_MARK in message.lower()
        if (reason, uncompensated) not in self.reasons:
            self.reasons.append((reason, uncompensated))


@contextlib.contextmanager
def capture_sens_rhs_declines():
    """Collect bngsim's decline reasons emitted inside the block (#606).

    Yields the list the handler fills: ``(reason, fallback_is_wrong)`` pairs, in the
    order bngsim reported them and de-duplicated, where ``fallback_is_wrong`` marks
    the variant whose difference quotient does NOT answer the same question (a
    branch crossing nobody compensates -- see :data:`_SENS_RHS_UNCOMPENSATED_MARK`).

    Best-effort **by construction**: the block has to contain the codegen that
    derives the decline, and a warm structural cache skips that step entirely, so an
    empty list means "declined, or served from cache" and never "not declined". Use
    it to explain a verdict :func:`analytic_sens_rhs_probe` has already reached,
    never to reach one.

    The handler is attached to the ``bngsim`` logger itself rather than to root:
    PyBNF's own root handlers stay untouched, so a decline still lands in the run's
    log exactly as it does today.
    """
    handler = _SensRhsDeclineHandler()
    bngsim_logger = logging.getLogger('bngsim')
    # No level is set here, deliberately. The ``bngsim`` logger is left at NOTSET and
    # inherits root's, and ``init_logging`` sets root to DEBUG and does its level
    # filtering on the FileHandler -- so the decline record reaches this handler even
    # under ``--log_level error``, where it would not reach the log file. Lowering a
    # level here would instead be a global side effect, and would push records into
    # PyBNF's own handlers that the user asked not to see.
    bngsim_logger.addHandler(handler)
    try:
        yield handler.reasons
    finally:
        bngsim_logger.removeHandler(handler)
