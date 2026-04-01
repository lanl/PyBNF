from pathlib import Path


_ROOT_DIR = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
_BNGL_LINK = _ROOT_DIR / 'bngl_files'
_created_bngl_link = False


def pytest_sessionstart(session):
    """Expose tests/bngl_files at the repo root for legacy relative-path tests."""
    del session
    global _created_bngl_link

    if not _BNGL_LINK.exists():
        _BNGL_LINK.symlink_to(_TESTS_DIR / 'bngl_files', target_is_directory=True)
        _created_bngl_link = True


def pytest_sessionfinish(session, exitstatus):
    del session, exitstatus
    if _created_bngl_link and _BNGL_LINK.is_symlink():
        _BNGL_LINK.unlink()
