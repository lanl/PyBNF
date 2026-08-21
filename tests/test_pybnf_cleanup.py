"""ROB-4: main()'s post-run cleanup/exit must not mask a propagating exception.

``pybnf.pybnf._finalize`` runs inside ``main()``'s outer ``finally``. A failure
inside ``alg.cleanup()`` is logged and swallowed -- but only if it is an
ordinary ``Exception``. A ``KeyboardInterrupt``/``SystemExit`` raised during
cleanup must propagate, not be masked by the ``exit()`` call (which previously
sat in a ``finally`` block that swallowed any in-flight exception).
"""

import os

import pytest

from pybnf.pybnf import _cleanup_dask_workspace, _finalize


class _StubAlg:
    """Stands in for the fitting algorithm; records whether cleanup ran and
    optionally raises a chosen exception from cleanup()."""

    def __init__(self, exc=None):
        self._exc = exc
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True
        if self._exc is not None:
            raise self._exc


def test_finalize_exits_zero_on_success():
    # On success, cleanup is skipped and the process exits 0.
    alg = _StubAlg(exc=RuntimeError("should never run"))
    with pytest.raises(SystemExit) as exc_info:
        _finalize(success=True, alg=alg, start_time=0.0)
    assert exc_info.value.code == 0
    assert alg.cleaned is False


def test_finalize_exits_one_on_failure():
    alg = _StubAlg()
    with pytest.raises(SystemExit) as exc_info:
        _finalize(success=False, alg=alg, start_time=0.0)
    assert exc_info.value.code == 1
    assert alg.cleaned is True


def test_finalize_swallows_ordinary_cleanup_exception():
    """An ordinary Exception during cleanup is logged, not propagated, so the
    process still exits with the failure code."""
    alg = _StubAlg(exc=RuntimeError("cleanup boom"))
    with pytest.raises(SystemExit) as exc_info:
        _finalize(success=False, alg=alg, start_time=0.0)
    assert exc_info.value.code == 1


def test_finalize_does_not_mask_keyboard_interrupt():
    """ROB-4 core: a KeyboardInterrupt during cleanup must propagate rather than
    being caught by a bare ``except:`` and then masked into ``SystemExit`` by an
    ``exit()`` sitting in a ``finally``. Mutation-distinguishing test: the old
    inline form (bare except + exit-in-finally) raised SystemExit here."""
    alg = _StubAlg(exc=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _finalize(success=False, alg=alg, start_time=0.0)


# Tests for _cleanup_dask_workspace (issue #620)


def test_cleanup_dask_scratch_space_in_cwd(tmp_path):
    """Cleanup removes dask-scratch-space (current dask directory name) from cwd."""
    # Create the directory that modern dask creates
    scratch_dir = tmp_path / 'dask-scratch-space'
    scratch_dir.mkdir()
    (scratch_dir / 'test_file.txt').write_text('test')
    
    # Change to the temp directory and run cleanup
    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        _cleanup_dask_workspace()
        # The directory should be removed
        assert not scratch_dir.exists()
    finally:
        os.chdir(original_dir)


def test_cleanup_dask_worker_space_in_cwd(tmp_path):
    """Cleanup removes dask-worker-space (legacy dask directory name) from cwd."""
    # Create the directory that old dask created
    worker_dir = tmp_path / 'dask-worker-space'
    worker_dir.mkdir()
    (worker_dir / 'test_file.txt').write_text('test')
    
    # Change to the temp directory and run cleanup
    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        _cleanup_dask_workspace()
        # The directory should be removed
        assert not worker_dir.exists()
    finally:
        os.chdir(original_dir)


def test_cleanup_both_dask_directories(tmp_path):
    """Cleanup removes both old and new dask directory names when both exist."""
    scratch_dir = tmp_path / 'dask-scratch-space'
    worker_dir = tmp_path / 'dask-worker-space'
    scratch_dir.mkdir()
    worker_dir.mkdir()
    (scratch_dir / 'file1.txt').write_text('test')
    (worker_dir / 'file2.txt').write_text('test')
    
    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        _cleanup_dask_workspace()
        # Both should be removed
        assert not scratch_dir.exists()
        assert not worker_dir.exists()
    finally:
        os.chdir(original_dir)


def test_cleanup_when_no_dask_directories_exist(tmp_path):
    """Cleanup runs without error when no dask directories exist."""
    # No directories created - just run cleanup
    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Should not raise any errors
        _cleanup_dask_workspace()
    finally:
        os.chdir(original_dir)


def test_cleanup_dask_scratch_space_in_home(tmp_path, monkeypatch):
    """Cleanup removes dask-scratch-space from the home directory.

    Home and cwd are deliberately different directories. If they were the
    same, the cwd branch of the cleanup would remove the directory and this
    test would pass even with the home branch broken.
    """
    home = tmp_path / 'home'
    home.mkdir()
    workdir = tmp_path / 'work'
    workdir.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))  # Windows
    monkeypatch.chdir(workdir)

    # Create the directory in "home"
    scratch_dir = home / 'dask-scratch-space'
    scratch_dir.mkdir()
    (scratch_dir / 'test_file.txt').write_text('test')

    _cleanup_dask_workspace()
    # The directory should be removed
    assert not scratch_dir.exists()


def test_cleanup_dask_worker_space_in_home(tmp_path, monkeypatch):
    """Cleanup removes dask-worker-space (legacy name) from the home directory.

    Home and cwd are deliberately different directories, for the same reason
    as the test above.
    """
    home = tmp_path / 'home'
    home.mkdir()
    workdir = tmp_path / 'work'
    workdir.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))  # Windows
    monkeypatch.chdir(workdir)

    # Create the legacy directory in "home"
    worker_dir = home / 'dask-worker-space'
    worker_dir.mkdir()
    (worker_dir / 'test_file.txt').write_text('test')

    _cleanup_dask_workspace()
    # The directory should be removed
    assert not worker_dir.exists()
