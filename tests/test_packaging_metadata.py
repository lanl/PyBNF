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
    assert 'bngsim>=0.3.0,<1' in project['dependencies']
