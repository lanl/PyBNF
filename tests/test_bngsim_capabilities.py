"""Tests for the centralized BNGsim capability detection layer.

Exercises ``pybnf._bngsim_caps`` (the module that owns version enforcement
and feature gating) against the issue #378 acceptance criteria:
too-old BNGsim, PYBNF_NO_BNGSIM-disabled, and missing-capability cases.
"""

import contextlib
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


# The session's real PYBNF_NO_BNGSIM, captured before any test can monkeypatch it. The
# bngsim-less CI leg sets it for the whole run, so "restore" cannot mean "unset".
_REAL_NO_BNGSIM = os.environ.get('PYBNF_NO_BNGSIM')


def _restore_caps():
    """Reload _bngsim_caps against the real bngsim package and the real environment.

    Both halves matter. These tests reload the capability module against a fake bngsim, and
    some of them also set ``PYBNF_NO_BNGSIM``; this runs from a ``finally``, which is *before*
    monkeypatch undoes either. Popping the fake module but leaving the environment variable set
    reloaded the module as though bngsim were absent, and monkeypatch then restored the variable
    without recomputing -- leaving every capability constant false for the rest of the session.

    That went unnoticed because the constant it broke, ``BNGSIM_HAS_EVENT_SENS``, was already
    false on every release below its 0.12.2 floor, so the poisoned value matched the real one.
    On 0.12.2 it is true, and ``test_event_model_reaches_gradient_setup_through_the_real_config``
    (skipped below the floor, so it had never run in the same session) started failing with
    "the installed bngsim (version unknown)" -- a discrete-event refusal that reads as a product
    bug and is entirely an artefact of test ordering.
    """
    sys.modules.pop('bngsim', None)
    if _REAL_NO_BNGSIM is None:
        os.environ.pop('PYBNF_NO_BNGSIM', None)
    else:
        os.environ['PYBNF_NO_BNGSIM'] = _REAL_NO_BNGSIM
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


# --------------------------------------------------------------------------- #
# BNGSIM_HAS_EVENT_SENS — a capability probe, not a version compare (#558)
# --------------------------------------------------------------------------- #
#
# The flag guards against a bngsim that answers a discrete event's forward
# sensitivity *wrongly and quietly*. It used to be a version floor at exactly
# 0.12.2, and the hazard with that is not hypothetical: bngsim bumps
# ``__version__`` at the start of a release cycle, so every from-source build
# between that bump and the fixes declares the same string as the release that
# carries them. A floor reports the capability PRESENT on such a build, and the
# symptom is not a refusal but a fit that runs to completion on a wrong gradient.
#
# The resolution order under test: a dedicated feature key first (bngsim
# publishes none yet -- this is the hook that starts working on the first build
# that grows one), then the ``effective_ic_sensitivity`` witness, and the version
# floor only as a conjunct that can veto but never carry.

_WITNESS = 'effective_ic_sensitivity'
_DEDICATED = 'event_sensitivities'


def _event_sens(monkeypatch, *, version, features):
    """Reload the capability module against a fake bngsim; return (flag, route)."""
    fake = _make_fake_bngsim(version=version, features=features)
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        return caps.BNGSIM_HAS_EVENT_SENS, caps.event_sens_probe()
    finally:
        _restore_caps()


def test_event_sens_reads_a_dedicated_feature_key_when_one_exists(monkeypatch):
    """Ask 1 of #558: the moment bngsim publishes a key for this, the flag reads it.

    In BOTH directions, and ahead of everything else -- a published ``False`` on a
    version that clears the floor must report absent, or the key is decoration.
    """
    present, route = _event_sens(
        monkeypatch, version='0.9.0',
        features={_DEDICATED: True})
    assert present is True                     # ... even below the old floor
    assert _DEDICATED in route

    absent, route = _event_sens(
        monkeypatch, version='0.13.0',
        features={_DEDICATED: False, _WITNESS: True})
    assert absent is False                     # ... and it outranks the witness
    assert _DEDICATED in route


def test_event_sens_reads_the_witness_key_when_there_is_no_dedicated_one(monkeypatch):
    """``effective_ic_sensitivity`` stands in for a key bngsim does not publish.

    It is not the capability. It is usable as evidence because of where it landed:
    lanl/bngsim#155 added it inside the same 0.12.1 -> 0.12.2 window as the fixes
    (#144, #146) and after both, so a build that publishes it necessarily carries
    them.
    """
    present, route = _event_sens(
        monkeypatch, version='0.12.2', features={_WITNESS: True})
    assert present is True
    assert _WITNESS in route


def test_event_sens_refuses_a_prerelease_build_that_only_declares_the_version(monkeypatch):
    """The #558 defect itself: version says 0.12.2, capabilities say otherwise.

    A from-source bngsim built after the release cycle's version bump but before
    the fixes declares ``0.12.2`` -- the same string as the release -- and the old
    floor reported the capability present on it. It publishes no witness key,
    because the key shipped IN 0.12.2, so reading capabilities instead of the
    version tells the two apart and fails toward a refusal.
    """
    present, route = _event_sens(
        monkeypatch, version='0.12.2', features={'codegen': True})
    assert present is False
    assert 'version floor alone is not evidence' in route


def test_event_sens_does_not_regress_any_released_bngsim(monkeypatch):
    """Every released bngsim at or above the floor publishes the witness, so this
    change refuses nothing that worked before -- and still refuses everything below."""
    for version in ('0.12.2', '0.13.0', '0.14.0'):
        present, _ = _event_sens(
            monkeypatch, version=version, features={_WITNESS: True})
        assert present is True, version
    for version in ('0.11.35', '0.12.0', '0.12.1'):
        present, _ = _event_sens(
            monkeypatch, version=version, features={'codegen': True})
        assert present is False, version


def test_event_sens_vetoes_a_witness_that_contradicts_the_version_floor(monkeypatch):
    """The floor survives as a conjunct: it can veto, it can no longer carry.

    A build below the floor that somehow publishes the witness is incoherent, and
    an unparseable version is no evidence at all. Both read absent -- the
    deliberate asymmetry with ``_version_compatible``, which accepts an unparseable
    version rather than brick an install, because the cost of guessing wrong here
    is a wrong gradient rather than a refusal.
    """
    below, route = _event_sens(monkeypatch, version='0.11.0', features={_WITNESS: True})
    assert below is False
    assert 'contradicts itself' in route
    unparseable, _ = _event_sens(monkeypatch, version='unknown', features={_WITNESS: True})
    assert unparseable is False


def test_event_sens_resolves_on_a_first_import_not_only_on_a_reload():
    """The resolution runs at module scope, and the tests above reach it by RELOAD --
    which reuses the module dict, so a name the resolver reads before its own
    definition still resolves to the previous load's copy. Only a fresh interpreter
    sees the import-time ordering. Every route is exercised there, since the bug
    class is per-branch."""
    script = textwrap.dedent('''
        import sys, types
        version, features = sys.argv[1], eval(sys.argv[2])
        fake = types.ModuleType('bngsim')
        fake.__version__ = version
        fake.capabilities = lambda: {
            'version': version, 'features': features, 'missing': {}}
        sys.modules['bngsim'] = fake
        from pybnf import _bngsim_caps as caps
        print('%s|%s' % (caps.BNGSIM_HAS_EVENT_SENS, caps.event_sens_probe()))
    ''')
    cases = [
        ('0.13.0', "{'event_sensitivities': True}", 'True'),
        ('0.13.0', "{'effective_ic_sensitivity': True}", 'True'),
        ('0.12.2', "{'codegen': True}", 'False'),
        ('0.11.0', "{'effective_ic_sensitivity': True}", 'False'),
    ]
    env = os.environ.copy()
    env.pop('PYBNF_NO_BNGSIM', None)
    for version, features, expected in cases:
        result = subprocess.run(
            [sys.executable, '-c', script, version, features],
            env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, (
            '%s %s: stdout=%r stderr=%r' % (version, features, result.stdout, result.stderr))
        assert result.stdout.startswith(expected + '|'), (version, features, result.stdout)


def test_event_sens_is_absent_and_says_so_without_bngsim(monkeypatch):
    monkeypatch.setitem(sys.modules, 'bngsim', None)
    monkeypatch.setenv('PYBNF_NO_BNGSIM', '1')
    try:
        caps = importlib.reload(_bngsim_caps)
        assert caps.BNGSIM_HAS_EVENT_SENS is False
        assert 'not available' in caps.event_sens_probe()
    finally:
        _restore_caps()


def test_event_sens_refusal_names_the_probe_rather_than_the_version(monkeypatch):
    """The refusal a user reads must not be a version complaint they have answered.

    On a build that declares a new enough version but publishes no capability, the
    old message said only "upgrade to >= 0.12.2" to someone who already had it.
    """
    import types

    from pybnf.algorithms.optimizers.trf import TRFAlgorithm
    from pybnf.printing import PybnfError

    fake = _make_fake_bngsim(version='0.12.2', features={'codegen': True})
    try:
        caps = _reload_caps_with(monkeypatch, fake)
        assert caps.BNGSIM_HAS_EVENT_SENS is False
        alg = object.__new__(TRFAlgorithm)
        alg.fit_type = 'trf'
        alg.model_list = [types.SimpleNamespace(name='m', has_discrete_events=True)]
        with pytest.raises(PybnfError) as exc:
            alg._require_differentiable_dynamics()
        message = exc.value.message
        assert 'capability, not a version' in message
        assert _WITNESS in message
    finally:
        _restore_caps()


# --------------------------------------------------------------------------- #
# Compiled-core provenance — the staleness a version cannot see (#558)
# --------------------------------------------------------------------------- #


def _fake_provenance_module(*, stale, build_commit='deadbeef1234', disabled=False):
    """A stand-in for bngsim's private ``_build_provenance`` module."""
    prov = types.SimpleNamespace(is_stale=stale, build_commit=build_commit)
    module = types.ModuleType('bngsim._build_provenance')
    module.gather = lambda **kwargs: prov
    module.identity_line = lambda p=None: (
        '[bngsim] _bngsim_core: /x/_bngsim_core.so | built=%s | mtime=? | %s'
        % (build_commit, 'STALE' if stale else 'installed'))
    module.format_report = lambda p=None: (
        module.identity_line() + '\n[bngsim]   STALE: src/x.cpp is newer than the '
        'loaded binary.')
    module._checks_disabled = lambda: disabled
    return module


@contextlib.contextmanager
def _caps_with_provenance(monkeypatch, provenance_module):
    """Reload the capability module against a fake bngsim carrying ``provenance_module``.

    A context manager rather than a plain call because of how the absent case has
    to be spelled. ``from bngsim import _build_provenance`` falls back to importing
    the *submodule* when the package object has no such attribute, and the real one
    is already in ``sys.modules`` from this session's own import, so modelling an
    install without it means putting ``None`` there -- the import system's spelling
    of "unavailable". That stub must come back out **before** ``_restore_caps()``
    re-imports the real bngsim, whose own ``__init__`` imports that submodule;
    leaving it to monkeypatch's teardown (which runs after the ``finally``) would
    make the restore reload find bngsim unimportable and poison every capability
    constant for the rest of the session -- the ordering failure ``_restore_caps``
    documents.
    """
    fake = _make_fake_bngsim(version='0.13.0', features={_WITNESS: True})
    stub = 'bngsim._build_provenance'
    stubbed = provenance_module is None
    original = sys.modules.get(stub)
    if stubbed:
        sys.modules[stub] = None
    else:
        fake._build_provenance = provenance_module
    try:
        yield _reload_caps_with(monkeypatch, fake)
    finally:
        if stubbed:
            if original is None:
                sys.modules.pop(stub, None)
            else:
                sys.modules[stub] = original
        _restore_caps()


def test_build_id_distinguishes_two_installs_declaring_one_version(monkeypatch):
    """The commit baked into the compiled core, which package metadata cannot give.

    Two bngsim installs can report the same ``__version__`` and be different
    builds; this is the identifier that tells them apart, and it describes the
    *binary* rather than the package.
    """
    with _caps_with_provenance(monkeypatch, _fake_provenance_module(stale=False)) as caps:
        assert caps.bngsim_build_id() == 'deadbeef1234'
        assert 'built=deadbeef1234' in caps.bngsim_identity_line()
        assert caps.bngsim_stale_core_report() == ''


def test_stale_compiled_core_is_reported(monkeypatch):
    """A core older than its own C++ passes every version and feature check.

    Nothing in the Python layer moves, so ``capabilities()`` is as wrong as a
    version string here -- bngsim's mtime comparison is the only thing that sees
    it, and PyBNF has to ask.
    """
    with _caps_with_provenance(monkeypatch, _fake_provenance_module(stale=True)) as caps:
        assert 'STALE' in caps.bngsim_stale_core_report()
        assert caps.BNGSIM_HAS_EVENT_SENS is True   # every other check still passes


def test_stale_report_honors_bngsims_own_opt_out(monkeypatch):
    """``BNGSIM_NO_BUILD_CHECK`` is the user saying the heuristic misfires here;
    PyBNF does not get a second vote."""
    stubbed = _fake_provenance_module(stale=True, disabled=True)
    with _caps_with_provenance(monkeypatch, stubbed) as caps:
        assert caps.bngsim_stale_core_report() == ''


def test_provenance_is_silent_when_bngsim_cannot_answer(monkeypatch):
    """``_build_provenance`` is private to bngsim: an install without it (or one
    whose reads raise) reports no opinion rather than taking the run down."""
    with _caps_with_provenance(monkeypatch, None) as caps:
        assert caps.bngsim_build_id() == ''
        assert caps.bngsim_identity_line() == ''
        assert caps.bngsim_stale_core_report() == ''

    exploding = _fake_provenance_module(stale=True)
    exploding.gather = lambda **kwargs: (_ for _ in ()).throw(RuntimeError('boom'))
    with _caps_with_provenance(monkeypatch, exploding) as caps:
        assert caps.bngsim_identity_line() == ''
        assert caps.bngsim_stale_core_report() == ''


@contextlib.contextmanager
def _collect_warnings(logger_name):
    """Collect warning records from one logger.

    Not ``caplog``: these tests reload ``pybnf._bngsim_caps`` against fake modules,
    and keeping the capture explicit keeps that dance out of the assertion.
    """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect(level=logging.WARNING)
    log = logging.getLogger(logger_name)
    log.addHandler(handler)
    try:
        yield records
    finally:
        log.removeHandler(handler)


def test_job_start_surfaces_a_stale_core_where_the_user_will_see_it(monkeypatch, capsys):
    """Ask 3 of #558. bngsim warns at *import* -- for PyBNF that is while the
    ``pybnf`` package loads, before logging is configured and before the user has
    committed to anything, so the warning scrolls past in import noise. A fit is
    hours; this repeats it at job start, on the console and in the run's own log."""
    from pybnf import pybnf as pybnf_main

    monkeypatch.setattr(pybnf_main._bngsim_caps, 'bngsim_identity_line',
                        lambda: '[bngsim] _bngsim_core: ... | built=abc123 | STALE')
    monkeypatch.setattr(pybnf_main._bngsim_caps, 'bngsim_stale_core_report',
                        lambda: '[bngsim]   STALE: src/x.cpp is newer than the binary.')
    with _collect_warnings('pybnf.pybnf') as records:
        pybnf_main._report_bngsim_build()
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'older than its own C++' in out
    assert any('older than its own' in r.getMessage() for r in records)

    monkeypatch.setattr(pybnf_main._bngsim_caps, 'bngsim_stale_core_report', lambda: '')
    pybnf_main._report_bngsim_build()
    assert 'WARNING' not in capsys.readouterr().out

