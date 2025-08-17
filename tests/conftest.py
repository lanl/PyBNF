# tests/conftest.py
import os
from pathlib import Path
import pytest

@pytest.fixture(scope="session", autouse=True)
def _chdir_to_tests_dir():
    os.chdir(Path(__file__).parent.resolve())
