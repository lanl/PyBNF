import fnmatch
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


def test_project_metadata_declares_python_floor_and_bngsim_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    metadata = tomllib.loads(pyproject_path.read_text())
    project = metadata['project']

    assert project['requires-python'] == '>=3.11'
    assert 'bngsim>=0.5.0,<1' in project['dependencies']


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
