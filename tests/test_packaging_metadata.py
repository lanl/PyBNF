import fnmatch
import importlib.metadata
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_ACTION = REPO_ROOT / '.github' / 'actions' / 'setup-pybnf' / 'action.yml'


def test_project_metadata_declares_python_floor_and_bngsim_dependency():
    pyproject_path = REPO_ROOT / 'pyproject.toml'
    metadata = tomllib.loads(pyproject_path.read_text())
    project = metadata['project']

    assert project['requires-python'] == '>=3.12'
    # bngsim >= 0.15.0 is bought by a CONTRACT, not by a feature PyBNF wants. lanl/bngsim#431
    # publishes `event_sensitivities` as a real capabilities() feature key. Until it existed
    # PyBNF read `effective_ic_sensitivity` as a WITNESS for the same thing (ADR-0119), which
    # was sound only because lanl/bngsim#155 happened to land a few commits after the event
    # fixes -- a fact about commit ordering that stops being evidence the moment the two are
    # decoupled, silently. Guessing wrong here is not symmetric: a build without those fixes
    # does not refuse, it returns a finite gradient with a term missing, so the fit converges
    # and reports a plausible number. 0.15.0 is the first release where PyBNF asks the real
    # question. It also carries the codegen-cache decline reason PyBNF reads (#647), and, on
    # the refusal side, a RuleMonkey re-vendor that declines a TotalRate rule where RuleMonkey
    # and NFsim genuinely disagree rather than picking a reading.
    # (0.12.2/0.12.0 bought the carried-state parameter_scan/bifurcate sensitivities of
    # lanl/bngsim#81 and #111 for #532; 0.11.35 the steady-state
    # SteadyStateResult.output_sensitivities of lanl/bngsim#12 for #478; 0.11.34 the native
    # carried-state parameter_scan + named saved states of lanl/bngsim#11 for #474.)
    assert 'bngsim>=0.15.0,<1' in project['dependencies']


def _canonical_name(name):
    """PEP 503 normalization, so `pytest-xdist` and `pytest_xdist` are one name."""
    return re.sub(r'[-_.]+', '-', name).lower()


_REQUIREMENT = re.compile(
    r'^([A-Za-z][A-Za-z0-9._-]*)'   # distribution name
    r'(?:\[[^\]]*\])?'             # extras, which the two files are allowed to differ on
    r'\s*(.*)$'                     # the version specifier, or nothing at all
)


def _split_requirement(text):
    """``'bngsim[antimony]>=0.15.0,<1'`` -> ``('bngsim', '>=0.15.0,<1')``.

    Extras are dropped on purpose: the CI action installs ``bngsim[antimony]`` where
    pyproject's runtime list declares bare ``bngsim``, and that difference is deliberate
    (the extra is what unskips the bngsim_antimony-marked tests). The version range is
    the part that has to agree. Returns None for a string that is not a requirement.
    """
    match = _REQUIREMENT.match(text.strip())
    if match is None:
        return None
    name, specifier = match.groups()
    return _canonical_name(name), specifier.replace(' ', '')


def _requirements_quoted_in(text):
    """Every single-quoted requirement in a YAML file, keyed by distribution name.

    The CI action is a shell script embedded in YAML, so its requirements are ordinary
    single-quoted shell words -- in the `uv pip install` argument list, in the BNGSIM_SPEC
    and JAX_SPEC arrays. Quotes are matched
    within a single line and comment lines are dropped, because the surrounding YAML prose
    is full of apostrophes ("pyproject.toml's", "the action's") that otherwise pair up with
    each other and swallow the real strings.
    """
    found = {}
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        for quoted in re.findall(r"'([^'\n]*)'", line):
            parsed = _split_requirement(quoted)
            if parsed is not None:
                found.setdefault(parsed[0], parsed[1])
    return found


def test_ci_action_installs_every_dependency_pyproject_declares():
    """The CI action and pyproject.toml must name the same version range for every package.

    The action does not install PyBNF's dependencies by resolving pyproject.toml. It
    installs a hand-written list that mirrors it, because a `bngsim: false` leg has to be
    able to leave bngsim out and a re-resolve would pull it back in. Two hand-maintained
    copies of one list drift, and this one did: the action pinned `bngsim>=0.11.35` while
    pyproject declared `bngsim>=0.12.2`. The two were set to the same number on 2026-07-24;
    pyproject moved twice over the following fortnight and the action never followed, so for
    three weeks continuous integration tested against a bngsim older than the project said
    it required. Nothing caught it. Both files carried a comment asking a human to keep them
    in sync, which is what a comment can do.

    The failure is quiet in both directions. Too low a floor means CI passes on a build the
    published wheel will not accept. A package declared in pyproject's [tests] extra but
    absent from the action does not fail a job either -- the suite that needs it
    `importorskip`s and reports as skipped.
    """
    metadata = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    declared = {}
    for requirement in (
        metadata['project']['dependencies']
        + metadata['project']['optional-dependencies']['tests']
    ):
        name, specifier = _split_requirement(requirement)
        declared[name] = specifier

    installed = _requirements_quoted_in(CI_ACTION.read_text())

    missing = sorted(name for name in declared if name not in installed)
    assert not missing, (
        'pyproject.toml declares these but %s never installs them, so the tests that '
        'need them skip in CI instead of failing: %s' % (CI_ACTION.name, ', '.join(missing))
    )

    drifted = {
        name: (declared[name], installed[name])
        for name in declared
        if declared[name] != installed[name]
    }
    assert not drifted, (
        'the CI action and pyproject.toml disagree about a version range '
        '(name: pyproject wants, action installs): %s' % drifted
    )


def test_the_version_is_the_same_everywhere_it_is_written_down():
    """One release version, written by hand into four files, must agree in all of them.

    pyproject.toml does not carry the number: it declares `dynamic = ["version"]` and reads
    `pybnf.__version__`, which makes `pybnf/__init__.py` the source of truth and the built
    wheel correct by construction. The other three are separate copies that a release has to
    remember to bump, and nothing has been checking them. A stale `CITATION.cff` tells anyone
    citing PyBNF the wrong version, and a stale `docs/conf.py` labels the published
    documentation with the previous release.

    The changelog is checked in the same breath because it is the same ritual and the same
    mistake. The release commit assembles the staged `changelog.d/` fragments into a dated
    heading (`tools/changelog.py build`), so the newest versioned heading is the released
    version at every commit -- both before a release, when the assembly has not happened and
    neither has the bump, and after it, when both have. Bumping one without the other is what
    this catches.
    """
    init_text = (REPO_ROOT / 'pybnf' / '__init__.py').read_text()
    match = re.search(r'^__version__ = [\'"]([^\'"]+)[\'"]', init_text, re.MULTILINE)
    assert match is not None, 'pybnf/__init__.py does not define __version__'
    version = match.group(1)

    # The mechanism that makes __init__.py authoritative. If this ever changes, the rest of
    # this test is checking copies against something that is no longer the original.
    metadata = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    assert 'version' in metadata['project']['dynamic']
    assert metadata['tool']['setuptools']['dynamic']['version'] == {
        'attr': 'pybnf.__version__'
    }

    citation = (REPO_ROOT / 'CITATION.cff').read_text()
    assert re.search(r'^version: %s$' % re.escape(version), citation, re.MULTILINE), (
        "CITATION.cff does not say version %s -- anyone citing PyBNF gets the wrong "
        'release' % version
    )

    conf = (REPO_ROOT / 'docs' / 'conf.py').read_text()
    assert re.search(r"^version = '%s'$" % re.escape(version), conf, re.MULTILINE), (
        "docs/conf.py's `version` is not %s" % version
    )
    # `release` carries a leading v; `version` does not. Both are bumped by hand.
    assert re.search(r"^release = 'v%s'$" % re.escape(version), conf, re.MULTILINE), (
        "docs/conf.py's `release` is not v%s" % version
    )

    changelog = (REPO_ROOT / 'CHANGELOG.md').read_text()
    headings = re.findall(r'^## \[v?([^\]]+)\]', changelog, re.MULTILINE)
    released = [h for h in headings if h != 'Unreleased']
    assert released, 'CHANGELOG.md has no released version heading'
    assert released[0] == version, (
        "CHANGELOG.md's newest released heading is %s but __version__ is %s -- one of the "
        'two was bumped without the other' % (released[0], version)
    )


def test_every_pybnf_subpackage_is_shipped():
    """Every importable subpackage under ``pybnf/`` must be matched by the
    setuptools package-discovery config, so a newly added subpackage cannot
    silently vanish from the built wheel.

    Regression guard: a hand-maintained ``[tool.setuptools] packages`` list
    omitted ``pybnf.priors`` and ``pybnf.noise`` (added in later refactors).
    The source tree imports both at module load, so the gap broke an installed
    wheel while a source checkout kept working. We now use glob discovery
    (``[tool.setuptools.packages.find] include = ["pybnf*"]``); this test
    re-derives the on-disk packages and asserts the include patterns cover
    them, using ``fnmatch`` to mirror setuptools' own glob semantics (and so
    avoid importing setuptools, which the test venv may not provide).
    """
    repo_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((repo_root / 'pyproject.toml').read_text())

    # On-disk subpackages: every directory under pybnf/ holding an __init__.py.
    pkg_root = repo_root / 'pybnf'
    ondisk = sorted(
        '.'.join(init.parent.relative_to(repo_root).parts)
        for init in pkg_root.rglob('__init__.py')
    )
    # The two subpackages the old hand-maintained list dropped -- anchor the
    # regression so the test is meaningful even if discovery is reworked again.
    assert 'pybnf.priors' in ondisk
    assert 'pybnf.noise' in ondisk

    find_cfg = metadata['tool']['setuptools']['packages']['find']
    includes = find_cfg.get('include', ['*'])
    excludes = find_cfg.get('exclude', [])
    for pkg in ondisk:
        assert any(fnmatch.fnmatch(pkg, pat) for pat in includes), (
            f'{pkg} is on disk but not matched by packages.find include={includes}'
        )
        assert not any(fnmatch.fnmatch(pkg, pat) for pat in excludes), (
            f'{pkg} is on disk but excluded by packages.find exclude={excludes}'
        )


def test_subprocess_no_jax_import_outside_hmc():
    """No non-HMC path may import ``jax``: importing PyBNF, registering every job type,
    and running a **log-scaled gradient assembly** must leave ``sys.modules`` jax-free.

    ``jax``/``jaxlib``/``blackjax`` are the optional ``pybnf[jax]`` extra, adopted for
    exactly one thing (ADR-0059): autodiff of the analytical model's log-density for the
    ``hmc`` reference sampler. That surface -- each prior family's ``logpdf_jax``, the
    bijector's ``*_jax`` peers, ``Scale.inverse_jax``, the PEtab formula's ``jax``
    lambdify backend -- hangs off general-purpose objects, so any layer can reach for a
    jax peer and quietly make a second subsystem depend on the extra. That is exactly
    what happened (#524): gradient assembly obtained a log-scaled parameter's
    ``d theta/d u`` with ``jax.grad(inverse_jax)``, so most gradient fits -- anything
    declaring a rate constant ``loguniform_var`` -- hard-required the extra. ADR-0087
    replaced it with the scale's analytic ``d_inverse``; this test keeps the next such
    reuse from going unnoticed.

    Runs in a subprocess because the pytest session itself imports jax (``test_hmc.py``,
    and the ADR-0087 parity tests), so the parent's ``sys.modules`` says nothing.

    Both halves matter, and which one fires depends on the environment. **Without** the
    extra installed, a module-scope ``import jax`` anywhere in the imported surface makes
    the child die with ``ImportError`` (a nonzero exit, asserted below). **With** it
    installed -- the ``pytest-jax`` CI leg, and a dev machine -- the ``sys.modules`` check
    additionally catches a leak inside a function body, which is the shape #524 took.
    """
    script = textwrap.dedent('''
        import sys

        import numpy as np

        # Every job type registers here, including hmc: its module must import no jax at
        # import time (ADR-0059), or `pip install pybnf` could not even list the samplers.
        import pybnf.algorithms
        import pybnf.config
        import pybnf.objective
        import pybnf.pset
        import pybnf.priors          # 18 families, each with a logpdf_jax peer
        import pybnf.gradient
        try:
            import pybnf.petab       # its formula module has a 'jax' lambdify backend
        except ImportError:
            pass                     # the optional pybnf[petab] extra is absent

        from pybnf.data import Data, OutputSensitivities
        from pybnf.gradient import (
            assemble_gaussian_gradient, ExperimentRouting, ParamRoute, PARAM)
        from pybnf.objective import ChiSquareObjective
        from pybnf.pset import FreeParameter

        # A log-scaled free parameter through the prior/scale surface (sampling-space
        # transform, prior density, a draw) -- the priors package's own jax peers.
        k = FreeParameter('k', 'loguniform_var', 0.01, 100.0, value=2.0)
        u = k.to_sampling_space(k.value)
        assert np.isclose(k.from_sampling_space(u), k.value)
        k.prior_logpdf(k.value)
        k.sample_value(np.random.default_rng(0))

        # ...and through a full gradient assembly, the #524 regression path: the
        # native->sampling Jacobian factor of a log-scaled parameter.
        times = np.array([0.0, 1.0, 2.0])
        sim = Data.from_columns(
            np.column_stack([times, [100.0, 74.0, 55.0]]), ['time', 'Stot'])
        sim.output_sensitivities = OutputSensitivities(
            selectors=['observable:Stot'], param_names=['k'], ic_species=[],
            d_param=np.array([0.0, -80.0, -106.0]).reshape(3, 1, 1))
        exp = Data.from_columns(
            np.column_stack([times, [100.0, 70.0, 60.0], [5.0] * 3]),
            ['time', 'Stot', 'Stot_SD'])
        routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})
        result = assemble_gaussian_gradient(
            ChiSquareObjective(), [(sim, exp, routing)], [k])
        # d theta/du = ln(10)*theta scales the one column (ADR-0087), analytically.
        assert np.isclose(result.jacobian[1, 0], (-80.0 / 5.0) * np.log(10.0) * 2.0)

        # Report the top-level roots only -- importing jax pulls in ~250 submodules,
        # and the name that matters is which extra leaked, not its module list.
        roots = {m.split('.')[0] for m in sys.modules}
        leaked = sorted(roots & {'jax', 'jaxlib', 'blackjax'})
        assert not leaked, (
            'imported outside the hmc path: %r -- the pybnf[jax] extra is for '
            'ADR-0059 autodiff of the analytical log-density only (see #524).' % (leaked,))
        print('OK')
    ''')
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        'stdout=%r stderr=%r' % (result.stdout, result.stderr)
    )
    assert 'OK' in result.stdout


def test_subprocess_pybnf_no_bngsim_package_root_import_smoke():
    """Importing pybnf and its bngsim-aware submodules must not raise under
    PYBNF_NO_BNGSIM=1.

    Belt-and-suspenders for
    ``test_bngsim_capabilities.py::test_subprocess_pybnf_no_bngsim_disables``,
    which only checks ``pybnf._bngsim_caps``. This test exercises the full
    set of submodules that lazily import bngsim at top level, so a future
    accidental ``import bngsim`` at module scope would surface here.
    """
    script = textwrap.dedent('''
        import pybnf
        import pybnf.algorithms
        import pybnf.bngsim_model
        import pybnf.bngsim_sbml_model
        import pybnf.bngsim_antimony_model
        from pybnf._bngsim_caps import BNGSIM_AVAILABLE
        assert BNGSIM_AVAILABLE is False, BNGSIM_AVAILABLE
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


def _requirement_floor(specifier):
    """The lower bound a requirement declares, or ``None`` when it declares none.

    ``'>=0.9,<1'`` -> ``'0.9'``. Only the ``>=`` clause matters here: an upper bound cannot be
    violated by a stale environment, because resolving an old lock can only leave a package
    behind, never ahead of a ceiling that was lowered.
    """
    for clause in specifier.split(','):
        clause = clause.strip()
        if clause.startswith('>='):
            return clause[2:]
    return None


def _release_tuple(version_text):
    """The numeric release segment of a version as a tuple: ``'2026.3.0'`` -> ``(2026, 3, 0)``.

    A pre-release, post-release or local suffix is dropped rather than ordered. That is the safe
    direction for this check: dropping it can only let a violation through, never invent one, and
    no floor this project declares is a pre-release. Comparing tuples also keeps the test free of
    a `packaging` import, which PyBNF does not declare and only has transitively.
    """
    release = re.match(r'\d+(?:\.\d+)*', version_text.strip())
    return tuple(int(part) for part in release.group(0).split('.')) if release else ()


def _declared_floors():
    """Every distribution pyproject declares a lower bound for, name -> floor.

    Runtime dependencies and every optional extra, since a developer environment installs the
    extras too. A distribution named in more than one list is held to the highest floor any of
    them declares, so the check cannot be weakened by an out-of-date copy of the same entry.
    """
    metadata = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    groups = [metadata['project']['dependencies']]
    groups += list(metadata['project'].get('optional-dependencies', {}).values())
    floors = {}
    for group in groups:
        for requirement in group:
            # An environment marker is not part of the version range.
            parsed = _split_requirement(requirement.split(';')[0])
            if parsed is None:
                continue
            name, specifier = parsed
            floor = _requirement_floor(specifier)
            if floor is None:
                continue
            if name not in floors or _release_tuple(floor) > _release_tuple(floors[name]):
                floors[name] = floor
    return floors


def test_the_installed_environment_meets_every_floor_pyproject_declares():
    """An environment below a declared floor must say so, rather than failing somewhere else.

    `uv.lock` is not tracked, so the lock in a working copy and the declarations in
    pyproject.toml drift apart with nothing to notice. When they do, the tests fail somewhere
    far away from the cause. A `.venv` holding petab 0.8.2 against a declared `petab>=0.9,<1`
    made six tests in `TestNativeBnglModel` fail with "Unknown model format: bngl", because
    petab gained its BioNetGen loader in 0.9.0. Nothing in that output points at the
    environment, and a run of the petab modules in that state reports nothing trustworthy in
    either direction (#803).

    A package that is not installed at all is skipped rather than failed. Every one of these
    lives in an optional extra, the suites that need them `importorskip`, and an environment
    without them is a supported way to work on the rest of PyBNF. Present but below the floor
    never is.
    """
    too_old = {}
    for name, floor in sorted(_declared_floors().items()):
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if _release_tuple(installed) < _release_tuple(floor):
            too_old[name] = (installed, floor)

    detail = ', '.join(f'{name} {installed} is below the declared {floor}'
                       for name, (installed, floor) in sorted(too_old.items()))
    assert not too_old, (
        f'this environment does not meet what pyproject.toml declares ({detail}). The lock '
        f'file is not tracked, so it can predate a floor that moved. Run '
        f'`uv sync --extra petab --extra tests` (adding whichever other extras you work with) '
        f'and run the suite again. Until then a failure anywhere else proves nothing.')


def test_the_floor_check_reads_a_requirement_and_orders_versions():
    """The parts of the check above, on the case that prompted it and on the shapes it meets.

    The guard itself passes trivially in a healthy environment, so the logic that decides what
    "too old" means is pinned here rather than left to the day it matters.
    """
    assert _requirement_floor('>=0.9,<1') == '0.9'
    assert _requirement_floor('>=0.15.0,<1') == '0.15.0'
    assert _requirement_floor('') is None                      # `h5py`, no bound at all
    assert _requirement_floor('<2') is None                    # a ceiling is not a floor

    # The real case: petab 0.8.2 installed against a declared floor of 0.9.
    assert _release_tuple('0.8.2') < _release_tuple('0.9')
    assert _release_tuple('0.9.0') >= _release_tuple('0.9')    # the floor itself passes
    assert _release_tuple('2026.3.0') > _release_tuple('2024.1.0')
    assert _release_tuple('1.24.4') > _release_tuple('1.24')
    assert _release_tuple('0.9.0rc1') == (0, 9, 0)             # a suffix is dropped, not ordered
    assert _release_tuple('') == ()

    floors = _declared_floors()
    assert floors['petab'] == '0.9'                            # read out of pyproject, not typed
    assert floors['bngsim'] == '0.15.0'                        # extras stripped
    assert 'h5py' not in floors                                # declared with no lower bound
