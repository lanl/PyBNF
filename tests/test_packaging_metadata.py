import fnmatch
import importlib.metadata
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest


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
    # 0.16.0 is the floor since lanl/PyBNF#991: the gradient router asks bngsim which derived
    # parameters it holds live and what each fresh run is seeded with, and only from 0.16.0 on
    # do the answers it gets from a released build and from bngsim main describe the same
    # columns closely enough that one set of refusals and witnesses covers both; 0.15.x is no
    # longer a build PyBNF has to keep a gradient path right on.
    # 0.17.0 is the floor since 1.8.2, the release cut together with it: every fix the
    # tests above credit to bngsim ``main`` rather than to 0.16.0 is in a released build
    # from 0.17.0 on, so an install resolved from PyPI gets the answers they assert.
    # The gradient fit's own run-time gate still names 0.16.0, the oldest build that
    # code is right on.
    # (0.12.2/0.12.0 bought the carried-state parameter_scan/bifurcate sensitivities of
    # lanl/bngsim#81 and #111 for #532; 0.11.35 the steady-state
    # SteadyStateResult.output_sensitivities of lanl/bngsim#12 for #478; 0.11.34 the native
    # carried-state parameter_scan + named saved states of lanl/bngsim#11 for #474.)
    assert 'bngsim>=0.17.0,<1' in project['dependencies']


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
    assert floors['bngsim'] == '0.17.0'                        # extras stripped
    assert 'h5py' not in floors                                # declared with no lower bound


# --------------------------------------------------------------------------- #
# Dependency floors (#1023)
# --------------------------------------------------------------------------- #
DEPENDENCY_FLOORS = REPO_ROOT / 'tools' / 'dependency_floors.py'
DOCKERFILE = REPO_ROOT / 'Dockerfile'


def _load_dependency_floors():
    import importlib.util
    spec = importlib.util.spec_from_file_location('pybnf_tools_dependency_floors',
                                                  DEPENDENCY_FLOORS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_floors_pins_each_requirement_at_its_lower_bound():
    """The constraints the floors leg installs with, on requirements of every shape it meets.

    Written out by hand, not read back from pyproject, so a parser that misreads a shape
    fails here instead of producing a constraints file that pins the wrong version.
    """
    floors = _load_dependency_floors()
    assert floors.floor_pins([
        'bngsim[antimony]>=0.17.0,<1',       # extras dropped, ceiling ignored
        'numpy>=1.26',
        'pydantic >= 2.0.2, < 3',            # spaces
        'exact==1.2.3',
        'compatible~=4.5',
        'tomli>=2; python_version < "3.11"',  # an environment marker is carried through
    ]) == [
        'bngsim==0.17.0',
        'numpy==1.26',
        'pydantic==2.0.2',
        'exact==1.2.3',
        'compatible==4.5',
        'tomli==2; python_version < "3.11"',
    ]


def test_dependency_floors_refuses_a_runtime_dependency_with_no_lower_bound():
    """A dependency with no floor cannot be tested at its floor, so the tool must not pass it
    over. Left out of the constraints, it would resolve to the newest release and the floors
    leg would report on a version nobody declared."""
    floors = _load_dependency_floors()
    with pytest.raises(ValueError, match=r'no lower bound.*h5py.*scipy<2'):
        floors.floor_pins(['numpy>=1.26', 'h5py', 'scipy<2'])


@pytest.mark.parametrize('requirement, cause', [
    ('numpy>1.25', 'excludes the version it names'),        # no single lowest version
    ('numpy===1.26.0', 'arbitrary equality'),               # not a version
    ('numpy>=1.26,!=1.26.0', 'excludes the floor 1.26'),    # the pin would be unsatisfiable
])
def test_dependency_floors_refuses_a_lower_bound_that_is_not_one_version(requirement, cause):
    """Each of these has a lower bound, but pinning it as ``name==floor`` would pin a version
    the requirement excludes or one that is not a version at all. Refuse, naming the cause,
    rather than write a constraints file that tests something nobody declared."""
    floors = _load_dependency_floors()
    with pytest.raises(ValueError, match=re.escape(cause)) as raised:
        floors.floor_pins(['scipy>=1.11.2', requirement])
    assert requirement in str(raised.value)


def test_dependency_floors_check_names_each_package_off_its_floor():
    """``--check`` is what stops a floors leg whose pins never took from passing on the newest
    releases, so its comparison is pinned on the cases it has to tell apart."""
    floors = _load_dependency_floors()
    installed = {'numpy': '1.26.0', 'scipy': '1.14.1', 'tornado': '6.2'}

    def lookup(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    assert floors.mismatches(
        ['numpy==1.26', 'scipy==1.11.2', 'tornado==6.2', 'msgpack==1.0.6'], lookup
    ) == [('scipy', '1.11.2', '1.14.1'), ('msgpack', '1.0.6', None)]
    # '1.26' and '1.26.0' are one release, and a marker does not stop the match.
    assert floors.mismatches(['numpy==1.26.0; python_version >= "3"'], lookup) == []


def test_dependency_floors_runs_on_the_real_pyproject_with_the_standard_library_alone():
    """The floors leg runs the tool with the runner's bare interpreter, before anything is
    installed. Run it the same way here, and check it pins every runtime dependency once."""
    result = subprocess.run([sys.executable, '-I', str(DEPENDENCY_FLOORS)],
                            capture_output=True, text=True, check=True)
    pins = result.stdout.split()
    declared = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())['project']['dependencies']
    assert len(pins) == len(declared)
    assert sorted(pin.split('==')[0] for pin in pins) == sorted(
        _split_requirement(requirement)[0] for requirement in declared)


def test_the_dockerfile_installs_every_runtime_dependency_at_pyproject_s_range():
    """The Dockerfile is a third hand-written copy of the runtime dependency list, beside
    pyproject.toml and the CI action, and nothing compared it with the other two. When #1023
    raised seven floors that were below anything installable, the Dockerfile still held all
    seven old values.

    It leaves bngsim out on purpose and installs PyBNF with --no-deps (#414), so bngsim is the
    one runtime dependency it is allowed to omit.
    """
    metadata = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    declared = dict(_split_requirement(r) for r in metadata['project']['dependencies'])
    declared.pop('bngsim')

    in_dockerfile = _requirements_quoted_in(DOCKERFILE.read_text())

    missing = sorted(name for name in declared if name not in in_dockerfile)
    assert not missing, f'the Dockerfile does not install these runtime dependencies: {missing}'
    drifted = {name: (declared[name], in_dockerfile[name])
               for name in declared if declared[name] != in_dockerfile[name]}
    assert not drifted, (
        'the Dockerfile and pyproject.toml disagree about a version range '
        f'(name: pyproject wants, Dockerfile installs): {drifted}')


# --------------------------------------------------------------------------- #
# Dependency floors: the CI step and the --check command, run rather than read
# (added in independent review of #1023)
# --------------------------------------------------------------------------- #
TESTS_WORKFLOW = REPO_ROOT / '.github' / 'workflows' / 'tests.yml'

_STUB_UV = r'''#!/bin/sh
# Stands in for uv: record the call, and on `uv venv` create a .venv/bin/python that
# records its own calls and exits with $STUB_CHECK_EXIT.
{ echo "CALL uv"; for arg in "$@"; do echo "$arg"; done; } >> "$STUB_LOG"
if [ "$1" = venv ]; then
  mkdir -p .venv/bin
  cat > .venv/bin/python <<'EOF'
#!/bin/sh
{ echo "CALL venv-python"; for arg in "$@"; do echo "$arg"; done; } >> "$STUB_LOG"
exit "${STUB_CHECK_EXIT:-0}"
EOF
  chmod +x .venv/bin/python
fi
'''


def _ci_legs():
    """Every job/leg in tests.yml that runs the setup-pybnf action, with the inputs it
    passes, evaluated the way the workflow expressions evaluate them."""
    import yaml
    workflow = yaml.safe_load(TESTS_WORKFLOW.read_text())
    legs = []
    for job_name, job in workflow['jobs'].items():
        step = next((s for s in job['steps'] if s.get('uses') == './.github/actions/setup-pybnf'),
                    None)
        if step is None:
            continue
        matrix_rows = job.get('strategy', {}).get('matrix', {}).get('include', [{}])
        for row in matrix_rows:
            def resolve(value, row=row):
                text = re.sub(r"matrix\.([\w-]+) == '1'",
                              lambda m: str(str(row.get(m.group(1), '')) == '1'), str(value))
                match = re.fullmatch(r"\$\{\{ (True|False) && '(\w+)' \|\| '(\w+)' \}\}", text)
                if match:
                    return match.group(2) if match.group(1) == 'True' else match.group(3)
                return re.sub(r'\$\{\{ matrix\.([\w-]+) \}\}',
                              lambda m: str(row.get(m.group(1), '')), text)
            inputs = {key: resolve(value) for key, value in step['with'].items()}
            legs.append(pytest.param(inputs, id='%s-%s' % (job_name, '-'.join(
                '%s=%s' % kv for kv in sorted(inputs.items()) if kv[0] != 'cache-suffix'))))
    return legs


def _run_install_step(tmp_path, inputs, check_exit=0, pyproject_text=None):
    """Run the action's 'Create venv and install pybnf' script under GitHub's own bash flags,
    with a stub uv, in a scratch copy holding the real tools/dependency_floors.py."""
    import shutil
    import yaml
    if shutil.which('bash') is None:
        pytest.skip('needs bash')
    action = yaml.safe_load(CI_ACTION.read_text())
    step = next(s for s in action['runs']['steps']
                if s.get('name') == 'Create venv and install pybnf')
    defaults = {name: spec.get('default') for name, spec in action['inputs'].items()}
    merged = {**defaults, **inputs}
    script = step['run'].replace('${{ steps.python.outputs.python-path }}', sys.executable)

    work = tmp_path / 'checkout'
    (work / 'tools').mkdir(parents=True)
    shutil.copy(DEPENDENCY_FLOORS, work / 'tools' / 'dependency_floors.py')
    (work / 'pyproject.toml').write_text(
        pyproject_text if pyproject_text is not None
        else (REPO_ROOT / 'pyproject.toml').read_text())
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'uv').write_text(_STUB_UV)
    (bindir / 'uv').chmod(0o755)
    (tmp_path / 'step.sh').write_text(script)
    log = tmp_path / 'calls.log'
    runner_temp = tmp_path / 'runner_temp'
    runner_temp.mkdir()
    env = dict(os.environ, PATH='%s%s%s' % (bindir, os.pathsep, os.environ['PATH']),
               STUB_LOG=str(log), STUB_CHECK_EXIT=str(check_exit), RUNNER_TEMP=str(runner_temp),
               BNGSIM=merged['bngsim'], JAX=merged['jax'], FLOORS=merged['dependency-floors'])
    result = subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail',
                             str(tmp_path / 'step.sh')],
                            cwd=work, env=env, capture_output=True, text=True)
    calls = []
    for line in (log.read_text().splitlines() if log.exists() else []):
        if line.startswith('CALL '):
            calls.append([line[5:]])
        else:
            calls[-1].append(line)
    return result, calls, runner_temp


@pytest.mark.parametrize('inputs', _ci_legs())
def test_each_ci_leg_installs_the_floors_exactly_when_it_asks_for_them(tmp_path, inputs):
    """Run the install step for every leg tests.yml defines, and check what it asks uv for.

    The floors leg must constrain every runtime dependency to the pins the tool derives from
    pyproject (an independent read here, through ``packaging``), refuse sdists, leave petab
    out, and run ``--check`` after the last install. Every other leg must do none of that and
    must still install petab, or the petab-marked tests would skip there too.
    """
    from packaging.requirements import Requirement
    result, calls, runner_temp = _run_install_step(tmp_path, inputs)
    assert result.returncode == 0, result.stderr

    installs = [c for c in calls if c[:3] == ['uv', 'pip', 'install'] and '--no-deps' not in c]
    assert len(installs) == 1, calls
    install = installs[0]
    names = {Requirement(arg).name for arg in install[3:]
             if not arg.startswith('-') and re.match(r'[A-Za-z]', arg)
             and not arg.startswith(str(runner_temp)) and arg != ':all:'}
    checks = [i for i, c in enumerate(calls)
              if c[0] == 'venv-python' and c[1:] == ['tools/dependency_floors.py', '--check']]
    last_install = max(i for i, c in enumerate(calls) if c[:3] == ['uv', 'pip', 'install'])

    floors = inputs.get('dependency-floors') == 'true'
    assert ('bngsim' in names) == (inputs.get('bngsim', 'true') == 'true')
    assert ('jax' in names) == (inputs.get('jax') == 'true')
    if not floors:
        assert '--constraint' not in install and '--only-binary' not in install
        assert 'petab' in names, 'a newest-release leg must install petab, or its tests skip'
        assert not checks
        return

    assert 'petab' not in names
    assert install[install.index('--only-binary') + 1] == ':all:'
    constraints = Path(install[install.index('--constraint') + 1]).read_text().split()
    declared = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())['project']['dependencies']
    expected = []
    for text in declared:
        requirement = Requirement(text)
        lower = [s.version for s in requirement.specifier if s.operator in ('>=', '==', '~=')]
        assert len(lower) == 1, text
        assert requirement.specifier.contains(lower[0]), text
        expected.append('%s==%s' % (requirement.name, lower[0]))
    assert sorted(constraints) == sorted(expected)
    # Every runtime dependency is named on the command line, so every constraint binds.
    assert {Requirement(text).name for text in declared} <= names
    assert checks and checks[-1] > last_install, (
        'the floors check must run after the last install, or a later install can undo it')


def test_the_floors_step_fails_when_the_installed_environment_is_not_the_floors(tmp_path):
    """The leg's guard against pins that never took is the exit status of ``--check``. Under
    the bash flags GitHub uses for a composite step, a failing check must fail the step."""
    result, calls, _ = _run_install_step(tmp_path, {'dependency-floors': 'true'}, check_exit=1)
    assert result.returncode != 0
    assert calls[-1] == ['venv-python', 'tools/dependency_floors.py', '--check']


def test_the_floors_step_stops_before_installing_when_a_dependency_has_no_floor(tmp_path):
    """A runtime dependency with no lower bound must stop the leg before anything installs,
    not write a partial constraints file and test whatever the resolver picks for it."""
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text().replace(
        '"msgpack>=1.0.6",', '"msgpack",')
    assert '"msgpack",' in pyproject
    result, calls, _ = _run_install_step(tmp_path, {'dependency-floors': 'true'},
                                         pyproject_text=pyproject)
    assert result.returncode != 0
    assert 'msgpack' in result.stderr and 'no lower bound' in result.stderr
    assert not [c for c in calls if c[:3] == ['uv', 'pip', 'install']]


def test_dependency_floors_check_command_exits_on_the_environment_it_runs_in(tmp_path):
    """``--check`` end to end, as the leg runs it: a subprocess, its exit status and its
    message, against a pyproject naming packages that are installed here. The unit test above
    drives ``mismatches`` with a fake lookup; this one goes through ``main`` and the real
    ``importlib.metadata``."""
    import shutil
    pytest_version = importlib.metadata.version('pytest')
    tools = tmp_path / 'tools'
    tools.mkdir()
    shutil.copy(DEPENDENCY_FLOORS, tools / 'dependency_floors.py')

    def check(dependencies):
        (tmp_path / 'pyproject.toml').write_text(
            '[project]\ndependencies = [%s]\n' % ', '.join('"%s"' % d for d in dependencies))
        return subprocess.run([sys.executable, str(tools / 'dependency_floors.py'), '--check'],
                              capture_output=True, text=True)

    at_floor = check(['pytest>=%s' % pytest_version])
    assert at_floor.returncode == 0, at_floor.stderr
    assert 'all 1 runtime dependencies are at their declared floors' in at_floor.stdout

    below = check(['pytest>=1.0', 'not-a-real-distribution-xyz>=1.0'])
    assert below.returncode == 1
    assert 'pytest is %s, but the declared floor is 1.0' % pytest_version in below.stderr
    assert 'not-a-real-distribution-xyz is not installed' in below.stderr


def test_every_installed_runtime_dependency_imports():
    """A runtime dependency that is installed must import. The suite can pass without ever
    importing one -- paramiko is reached only through ``dask ssh`` on a real cluster -- so on
    the dependency-floors leg a floor that installs but cannot run would go unnoticed.

    Absent is skipped, as the floor guard above skips it (the bngsim-less leg has no bngsim).
    Present but failing to import is a failure, whatever the reason.
    """
    import json
    declared = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())['project']['dependencies']
    by_distribution = {}
    for top, dists in importlib.metadata.packages_distributions().items():
        if top.startswith('_'):
            continue
        for dist in dists:
            by_distribution.setdefault(_canonical_name(dist), set()).add(top)
    modules = {}
    for requirement in declared:
        name = _split_requirement(requirement)[0]
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        modules[name] = sorted(by_distribution.get(name, ()))
        assert modules[name], f'{name} is installed but provides no importable top-level module'
    script = textwrap.dedent('''
        import importlib, json, sys, traceback
        failed = {}
        for dist, tops in json.loads(sys.argv[1]).items():
            for top in tops:
                try:
                    importlib.import_module(top)
                except BaseException as error:
                    failed[f'{dist} ({top})'] = f'{type(error).__name__}: {error}'
        print(json.dumps(failed))
    ''')
    result = subprocess.run([sys.executable, '-c', script, json.dumps(modules)],
                            capture_output=True, text=True, check=True)
    failed = json.loads(result.stdout.strip().splitlines()[-1])
    versions = {name: importlib.metadata.version(name) for name in modules}
    assert not failed, (
        'installed runtime dependencies that do not import (installed versions %s): %s'
        % (versions, failed))
