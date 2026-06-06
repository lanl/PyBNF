"""
Tests for the node-local live-simulation registry and reaper in ``pybnf.pset``.

This is process-orchestration glue, so the oracles are real OS state, not mocks:
a real detached process actually dies (or doesn't), and the registry directory
actually contains (or drops) the right PGID-named files. The registry's whole job
is to let an aborting fit ``killpg`` the detached sim process groups that would
otherwise orphan ("Ctrl-C does nothing; kill -9 needed"), so the tests exercise
that against genuine subprocesses.

Isolation: ``PYBNF_SIM_REGISTRY`` is set via ``monkeypatch.setenv`` (auto-restored,
so it can't leak the registry into sibling tests) and ``tempfile.gettempdir`` is
redirected to a per-test ``tmp_path`` so nothing touches the real tempdir.
"""
import os
import signal
import subprocess
import time

import pytest

from .context import pset

# The registry exists to killpg detached process groups, which is POSIX-only
# (run_subprocess already no-ops the process-group path on Windows).
pytestmark = pytest.mark.skipif(os.name == 'nt', reason='POSIX process-group registry/reaper')


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _alive(pid):
    """True if a process with this PID currently exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _spawn_detached():
    """Spawn a real long-lived child in its own session (PGID == PID), mirroring
    how run_subprocess launches a sim. Returns the Popen handle."""
    return subprocess.Popen(['sleep', '300'], start_new_session=True)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Enable the registry under an isolated tempdir; yields the registry dir."""
    monkeypatch.setattr(pset.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setenv('PYBNF_SIM_REGISTRY', 'unit')
    return pset._registry_dir()


# --------------------------------------------------------------------------- #
# _registry_dir / set_sim_registry — resolution from the env var
# --------------------------------------------------------------------------- #
class TestRegistryResolution:

    def test_disabled_when_env_unset(self, tmp_path, monkeypatch):
        """With PYBNF_SIM_REGISTRY unset the registry is off: _registry_dir is
        None and every operation is an inert no-op (this is exactly today's
        behavior for a process that never enabled the registry)."""
        monkeypatch.setattr(pset.tempfile, 'gettempdir', lambda: str(tmp_path))
        monkeypatch.delenv('PYBNF_SIM_REGISTRY', raising=False)
        assert pset._registry_dir() is None
        assert pset.reap_active_sims() == []
        pset._register_sim(4242)          # no-op, must not raise or create files
        pset._deregister_sim(4242)
        assert list(tmp_path.iterdir()) == []

    def test_set_sim_registry_creates_node_local_dir(self, tmp_path, monkeypatch):
        """set_sim_registry sets the env var and creates a directory under *this
        host's* tempdir named for the run -- the node-locality that makes its
        PGIDs safe to killpg only from a process on the same host."""
        monkeypatch.setattr(pset.tempfile, 'gettempdir', lambda: str(tmp_path))
        monkeypatch.delenv('PYBNF_SIM_REGISTRY', raising=False)
        pset.set_sim_registry('myrun')
        assert os.environ['PYBNF_SIM_REGISTRY'] == 'myrun'
        d = pset._registry_dir()
        assert d == str(tmp_path / 'pybnf_sims_myrun')
        assert os.path.isdir(d)

    def test_run_id_sanitised_to_one_path_component(self, tmp_path, monkeypatch):
        """A run_id with path separators / odd characters can't escape the tempdir
        or create nested dirs -- it is sanitised to a single safe component."""
        monkeypatch.setattr(pset.tempfile, 'gettempdir', lambda: str(tmp_path))
        monkeypatch.setenv('PYBNF_SIM_REGISTRY', 'a/b ../c:d')
        d = pset._registry_dir()
        assert os.path.dirname(d) == str(tmp_path)                 # stayed in tempdir
        assert os.path.basename(d) == 'pybnf_sims_a_b_.._c_d'      # one component


# --------------------------------------------------------------------------- #
# register / deregister — the PGID-named file mechanics
# --------------------------------------------------------------------------- #
class TestRegisterDeregister:

    def test_register_then_deregister_roundtrip(self, registry):
        """_register_sim drops an empty file named by the PGID; _deregister_sim
        removes it. The file is empty (the PGID *is* the name) -- the property
        that bounds storage to the live-sim count, not the evaluation count."""
        pset._register_sim(13579)
        path = os.path.join(registry, '13579')
        assert os.path.isfile(path)
        assert os.path.getsize(path) == 0
        pset._deregister_sim(13579)
        assert not os.path.exists(path)

    def test_deregister_missing_is_silent(self, registry):
        """Deregistering an unknown PGID (e.g. double-deregister) is a harmless
        no-op, never an error."""
        pset._deregister_sim(99999)  # must not raise


# --------------------------------------------------------------------------- #
# reap_active_sims — the actual kill, against real processes
# --------------------------------------------------------------------------- #
class TestReap:

    def test_reap_kills_real_detached_process(self, registry):
        """The core contract: a registered, *running* detached process group is
        SIGKILLed and dropped from the registry, and its PGID is reported."""
        proc = _spawn_detached()
        pgid = os.getpgid(proc.pid)
        try:
            pset._register_sim(pgid)
            assert _alive(proc.pid)

            reaped = pset.reap_active_sims()

            assert pgid in reaped
            proc.wait(timeout=5)                       # killpg took it down
            assert not _alive(proc.pid)
            assert os.listdir(registry) == []          # entry removed
        finally:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_reap_skips_dead_and_non_pgid_entries(self, registry):
        """Stale entries (a group that already exited) and non-PGID filenames are
        skipped without error, and both are cleaned up. Oracle: a genuinely dead
        PGID (a child we spawned and reaped) plus a junk file."""
        proc = _spawn_detached()
        dead_pgid = os.getpgid(proc.pid)
        os.killpg(dead_pgid, signal.SIGKILL)
        proc.wait(timeout=5)                           # PID/PGID now freed

        pset._register_sim(dead_pgid)
        open(os.path.join(registry, 'not-a-pgid'), 'w').close()

        reaped = pset.reap_active_sims()

        assert reaped == []                            # dead group not "killed"
        assert sorted(os.listdir(registry)) == ['not-a-pgid']  # PGID entry cleared, junk kept

    def test_reap_noop_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pset.tempfile, 'gettempdir', lambda: str(tmp_path))
        monkeypatch.delenv('PYBNF_SIM_REGISTRY', raising=False)
        assert pset.reap_active_sims() == []


# --------------------------------------------------------------------------- #
# clear_sim_registry / stale sweep — housekeeping that bounds storage
# --------------------------------------------------------------------------- #
class TestHousekeeping:

    def test_clear_removes_dir_and_disables(self, registry, tmp_path):
        """At run end the whole per-run directory is removed and the registry is
        disabled, so it doesn't linger in tempdir."""
        pset._register_sim(111)
        assert os.path.isdir(registry)
        pset.clear_sim_registry()
        assert not os.path.exists(registry)
        assert 'PYBNF_SIM_REGISTRY' not in os.environ

    def test_stale_sweep_removes_old_keeps_recent(self, tmp_path, monkeypatch):
        """The startup sweep removes pybnf_sims_* dirs orphaned by a prior kill -9
        (older than the age cutoff) but never touches a concurrent run's recent
        dir -- so accumulation across hard-killed runs can't build up."""
        monkeypatch.setattr(pset.tempfile, 'gettempdir', lambda: str(tmp_path))
        old = tmp_path / 'pybnf_sims_old'
        recent = tmp_path / 'pybnf_sims_recent'
        unrelated = tmp_path / 'something_else'
        for d in (old, recent, unrelated):
            d.mkdir()
        two_days = time.time() - 2 * 86400
        os.utime(old, (two_days, two_days))

        pset._sweep_stale_sim_registries()

        assert not old.exists()        # stale -> removed
        assert recent.exists()         # recent -> kept (could be a live run)
        assert unrelated.exists()      # non-registry dir -> untouched


# --------------------------------------------------------------------------- #
# run_subprocess — registration lifecycle + kill-on-interruption hardening
# --------------------------------------------------------------------------- #
class TestRunSubprocessIntegration:

    def test_clean_run_leaves_registry_empty(self, registry):
        """A sim that completes registers then deregisters: no leaked entry, no
        orphaned process. Oracle: the registry dir is empty afterwards."""
        pset.run_subprocess(['sleep', '0'], timeout=10)
        assert os.listdir(registry) == []

    def test_kill_on_arbitrary_interruption_and_reraise(self, monkeypatch, registry):
        """Hardening: ANY interruption of communicate (not just a timeout) takes
        the detached sim group down via killpg(SIGKILL) and re-raises. Here a
        KeyboardInterrupt stands in for the general case; we assert the exact
        killpg call and that it propagates."""
        class _FakeProc:
            pid = 4242
            returncode = None

            def communicate(self, input=None, timeout=None):
                raise KeyboardInterrupt()

            def wait(self):
                self.waited = True

            def kill(self):
                pass

        fake = _FakeProc()
        monkeypatch.setattr(pset, 'Popen', lambda *a, **k: fake)
        monkeypatch.setattr(pset.os, 'getpgid', lambda pid: 4242)
        killed = []
        monkeypatch.setattr(pset.os, 'killpg', lambda pgid, sig: killed.append((pgid, sig)))

        with pytest.raises(KeyboardInterrupt):
            pset.run_subprocess(['x'], timeout=None)

        assert killed == [(4242, signal.SIGKILL)]
        assert getattr(fake, 'waited', False)
        assert os.listdir(registry) == []                  # deregistered in finally

    def test_timeout_still_kills_real_group_and_propagates(self, registry):
        """Regression: the original timeout-cleanup path still works -- a real
        sim exceeding its timeout raises TimeoutExpired and its process group is
        killed (the registry ends empty, proving deregistration ran)."""
        with pytest.raises(subprocess.TimeoutExpired):
            pset.run_subprocess(['sleep', '30'], timeout=0.3)
        assert os.listdir(registry) == []


# --------------------------------------------------------------------------- #
# End-to-end: a real dask LocalCluster worker spawns a sim; main reaps it.
# --------------------------------------------------------------------------- #
def _spawn_and_register_on_worker():
    """Run on a dask worker: spawn a detached sim and register it exactly as
    run_subprocess would, reading PYBNF_SIM_REGISTRY from the worker's inherited
    environment. Returns the sim PID."""
    import os as _os
    import subprocess as _sp
    import pybnf.pset as _pset
    p = _sp.Popen(['sleep', '300'], start_new_session=True)
    _pset._register_sim(_os.getpgid(p.pid))
    return p.pid


@pytest.mark.slow
def test_end_to_end_main_reaps_worker_spawned_sims(monkeypatch):
    """The real scenario: workers (separate processes) inherit PYBNF_SIM_REGISTRY,
    register the sims they spawn into the shared node-local directory, and the
    main process reaps them all on abort. Proves env-var inheritance + the
    file-based registry + reap work together against live processes."""
    from distributed import Client, LocalCluster

    # Real env var (inherited by worker subprocesses), real tempdir (workers can't
    # see a monkeypatched gettempdir); auto-restored by monkeypatch afterwards.
    monkeypatch.setenv('PYBNF_SIM_REGISTRY', f'pytest_e2e_{os.getpid()}')
    pids = []
    try:
        with LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                          dashboard_address=':0') as lc, Client(lc) as client:
            pids = list(client.run(_spawn_and_register_on_worker).values())
            assert pids and all(_alive(p) for p in pids)

            reaped = pset.reap_active_sims()          # main reaps this node

        assert reaped, 'expected the reaper to find worker-registered sims'
        time.sleep(0.5)
        assert not any(_alive(p) for p in pids), 'worker-spawned sims should be dead'
    finally:
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        pset.clear_sim_registry()
