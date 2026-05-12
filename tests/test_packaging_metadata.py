import os
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def test_project_metadata_declares_python_floor_and_bngsim_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    metadata = tomllib.loads(pyproject_path.read_text())
    project = metadata['project']

    assert project['requires-python'] == '>=3.10'
    assert 'bngsim>=0.5.0,<1' in project['dependencies']


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
