"""
Orchestration tests for ``pybnf.cluster.Cluster`` — the SSH/HPC dask bring-up.

This is glue code, not numerical math: the only deterministic contracts are
*which external call PyBNF constructs, with what arguments, and which branch
fires for which config*. So the oracle for each test is the constructed
``scontrol``/``dask-ssh`` command string, the SLURM stdout parse, the
subprocess-error → PybnfError mapping, the ``ceil(parallel_count/num_nodes)``
per-node arithmetic, or the ``Client(...)``/``LocalCluster(...)`` call the
config selects. (For glue with no math, "the right command/Client call was
made" *is* the oracle — not the mock-the-world anti-pattern.)

Substitution strategy (per dependency):
  * ``cluster.run`` (subprocess) — **mock**: a recorder returning a fake proc
    with canned ``.stdout`` bytes, or raising ``TimeoutExpired`` /
    ``CalledProcessError``.
  * ``cluster.Popen`` / ``cluster.time.sleep`` / ``cluster.cpu_count`` —
    **mock**: capture the command; stub the 10s sleep so the test is instant.
  * ``cluster.Client`` / ``cluster.LocalCluster`` / ``cluster.init_logging`` /
    ``cluster.reinit_logging`` and ``Cluster.read_node_names`` /
    ``Cluster.setup_cluster`` — **fakes** recording their call args, so the
    ``__init__`` branch dispatch is asserted without a real dask cluster.

#393 note: these assert PyBNF's *own* command-string / branch logic, never
dask/distributed internals or a pinned dask version (the version-specific
``reinit_logging`` workaround is asserted *to be called*, not pinned), so they
remain a valid safety net across the dask-unpinning upgrade.
"""
import os
import types

import pytest

from subprocess import TimeoutExpired, CalledProcessError

from .context import cluster, printing


# --------------------------------------------------------------------------- #
# Lightweight config stub: cluster only ever reads ``config.config[<key>]``.
# --------------------------------------------------------------------------- #
def _cfg(**overrides):
    base = {'scheduler_file': None, 'scheduler_node': None, 'worker_nodes': None,
            'parallel_count': None, 'cluster_type': None}
    base.update(overrides)
    return types.SimpleNamespace(config=base)


# --------------------------------------------------------------------------- #
# read_node_names — SLURM parse, command string, error mapping
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, stdout):
        self.stdout = stdout


class _FakeDaskProc:
    """Stand-in for the dask-ssh Popen object: setup_cluster only calls poll()."""
    def __init__(self, returncode=None):
        self._returncode = returncode

    def poll(self):
        return self._returncode


class TestReadNodeNames:

    def test_no_cluster_type_is_local(self, monkeypatch):
        """No cluster_type ⇒ a local run: (None, None) and the host-detection
        subprocess is never invoked."""
        called = []
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: called.append((a, k)))
        assert cluster.Cluster.read_node_names(_cfg()) == (None, None)
        assert called == []

    def test_slurm_parse_and_command_string(self, monkeypatch):
        """SLURM: runs ``scontrol show hostname <$SLURM_JOB_NODELIST>`` as an
        argument list with NO shell (ROB-3), 10s timeout, check=True, capturing
        stdout, then parses the newline-separated hostnames. Oracle: scheduler ==
        nodes[0], node_string == ' '.join(nodes). The trailing newline must be
        stripped, not parsed into an empty fourth node. The nodelist comes from
        os.environ and is passed as a single literal arg (so a compressed nodelist
        like ``node[17-19]`` reaches scontrol intact, not shell-globbed)."""
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append((cmd, kwargs))
            return _FakeProc(b'node17\nnode18\nnode19\n')

        monkeypatch.setenv('SLURM_JOB_NODELIST', 'node[17-19]')
        monkeypatch.setattr(cluster, 'run', fake_run)
        scheduler, node_string = cluster.Cluster.read_node_names(_cfg(cluster_type='slurm'))

        assert scheduler == 'node17'                       # nodes[0], not nodes[-1]
        assert node_string == 'node17 node18 node19'       # ' '.join(nodes), all three
        cmd, kwargs = captured[0]
        assert cmd == ['scontrol', 'show', 'hostname', 'node[17-19]']  # arg list, nodelist intact
        assert kwargs.get('shell', False) is False         # no shell -> no injection / globbing
        assert kwargs['timeout'] == 10
        assert kwargs['check'] is True
        assert kwargs['stdout'] is cluster.PIPE

    def test_slurm_nodelist_from_env_passed_as_one_literal_arg(self, monkeypatch):
        """ROB-3: a $SLURM_JOB_NODELIST carrying shell metacharacters is handed to
        scontrol as a single literal argv entry with shell off -- it is never
        interpreted by a shell."""
        captured = []
        monkeypatch.setenv('SLURM_JOB_NODELIST', 'n1; touch pwned')
        monkeypatch.setattr(cluster, 'run',
                            lambda cmd, **k: captured.append((cmd, k)) or _FakeProc(b'n1\n'))
        cluster.Cluster.read_node_names(_cfg(cluster_type='slurm'))
        cmd, kwargs = captured[0]
        assert cmd == ['scontrol', 'show', 'hostname', 'n1; touch pwned']
        assert kwargs.get('shell', False) is False

    def test_slurm_unset_nodelist_omits_arg(self, monkeypatch):
        """When $SLURM_JOB_NODELIST is unset, the nodelist arg is omitted (matching
        the old empty shell expansion), so scontrol falls back to its own default."""
        captured = []
        monkeypatch.delenv('SLURM_JOB_NODELIST', raising=False)
        monkeypatch.setattr(cluster, 'run',
                            lambda cmd, **k: captured.append((cmd, k)) or _FakeProc(b'n1\n'))
        cluster.Cluster.read_node_names(_cfg(cluster_type='slurm'))
        assert captured[0][0] == ['scontrol', 'show', 'hostname']

    def test_slurm_single_node(self, monkeypatch):
        """A one-node allocation: scheduler and node_string are the same single
        host (the scheduler also acts as the only worker)."""
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: _FakeProc(b'solo01\n'))
        assert cluster.Cluster.read_node_names(_cfg(cluster_type='slurm')) == ('solo01', 'solo01')

    def test_slurm_strips_surrounding_whitespace(self, monkeypatch):
        """Whitespace around the scontrol output is stripped before splitting, so
        leading/trailing blanks don't become phantom empty node names."""
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: _FakeProc(b'  n1\nn2  \n\n'))
        assert cluster.Cluster.read_node_names(_cfg(cluster_type='slurm')) == ('n1', 'n1\nn2'.replace('\n', ' '))

    @pytest.mark.parametrize('ctype', ['slurm', 'SLURM', 'Slurm', 'sLuRm'])
    def test_slurm_detection_is_case_insensitive(self, monkeypatch, ctype):
        """The cluster_type regex matches SLURM case-insensitively, so any
        capitalization takes the SLURM branch (and runs scontrol)."""
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: _FakeProc(b'a\nb\n'))
        assert cluster.Cluster.read_node_names(_cfg(cluster_type=ctype)) == ('a', 'a b')

    def test_timeout_maps_to_pybnf_error(self, monkeypatch):
        """scontrol hanging past the 10s timeout (TimeoutExpired) is translated to
        a PybnfError about not finding nodes in a reasonable time — not allowed to
        propagate as a raw subprocess exception."""
        def boom(*a, **k):
            raise TimeoutExpired(cmd='scontrol', timeout=10)
        monkeypatch.setattr(cluster, 'run', boom)
        with pytest.raises(printing.PybnfError, match='reasonable time'):
            cluster.Cluster.read_node_names(_cfg(cluster_type='slurm'))

    def test_called_process_error_maps_to_pybnf_error(self, monkeypatch):
        """A non-zero scontrol exit (CalledProcessError, raised by check=True) maps
        to a distinct PybnfError telling the user to confirm they really are on
        SLURM."""
        def boom(*a, **k):
            raise CalledProcessError(returncode=1, cmd='scontrol')
        monkeypatch.setattr(cluster, 'run', boom)
        with pytest.raises(printing.PybnfError, match='Command to find node names failed'):
            cluster.Cluster.read_node_names(_cfg(cluster_type='slurm'))

    @pytest.mark.parametrize('ctype', ['torque', 'TORQUE', 'pbs', 'PBS', 'Torque'])
    def test_torque_pbs_not_implemented(self, monkeypatch, ctype):
        """TORQUE/PBS (case-insensitive) is recognized but explicitly unsupported:
        a PybnfError saying so, rather than silently falling through to the
        unknown-type branch."""
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: _FakeProc(b''))
        with pytest.raises(printing.PybnfError, match='not yet implemented'):
            cluster.Cluster.read_node_names(_cfg(cluster_type=ctype))

    def test_unknown_cluster_type_raises(self, monkeypatch):
        """A cluster_type matching neither SLURM nor TORQUE/PBS is a config error:
        PybnfError naming the unknown type."""
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: _FakeProc(b''))
        with pytest.raises(printing.PybnfError, match='Unknown cluster type'):
            cluster.Cluster.read_node_names(_cfg(cluster_type='mesos'))


# --------------------------------------------------------------------------- #
# setup_cluster — the dask-ssh command string + per-node arithmetic
# --------------------------------------------------------------------------- #
class TestSetupCluster:

    def _patch(self, monkeypatch, cpu=4, returncode=None, stderr_bytes=b''):
        """Patch the three externals setup_cluster touches: Popen (capture the
        command), time.sleep (don't actually wait 10s), cpu_count (deterministic).
        The fake proc's ``poll()`` returns ``returncode`` (None = still running,
        the healthy default); if ``stderr_bytes`` is given the fake writes it to
        the stderr file setup_cluster handed to Popen, so the early-exit error
        path can read it back. Returns the recorder list of Popen (args, kwargs)."""
        popen_calls = []

        def fake_popen(*args, **kwargs):
            popen_calls.append((args, kwargs))
            if stderr_bytes:
                kwargs['stderr'].write(stderr_bytes)
            return _FakeDaskProc(returncode)

        monkeypatch.setattr(cluster, 'Popen', fake_popen)
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        monkeypatch.setattr(cluster, 'cpu_count', lambda: cpu)
        return popen_calls

    def test_default_parallel_count_uses_cpu_count(self, monkeypatch):
        """parallel_count=None ⇒ dask-ssh's own default of one worker per CPU:
        ``--nthreads 1 --nworkers {cpu_count()}`` (note this branch's flag order is
        --nthreads then --nworkers). Oracle: the exact argument list (ROB-3: an argv
        list launched with no shell, each node its own entry) with cpu_count()=7."""
        popen_calls = self._patch(monkeypatch, cpu=7)
        proc = cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=None)

        assert proc.poll() is None
        (args, kwargs), = popen_calls
        assert args[0] == ['dask-ssh', 'n1', 'n2',
                           '--log-directory', '/out', '--nthreads', '1', '--nworkers', '7']
        assert kwargs.get('shell', False) is False         # no shell -> no injection
        assert kwargs['stdout'] is cluster.DEVNULL
        # stderr is captured to a readable file (not discarded), so an early
        # bring-up failure can be surfaced — see test_failed_bringup_*.
        assert kwargs['stderr'] is not cluster.DEVNULL
        assert hasattr(kwargs['stderr'], 'read')

    def test_running_proc_is_returned_without_raising(self, monkeypatch):
        """The happy path: dask-ssh is still running after the startup wait
        (poll() is None), so setup_cluster returns the proc rather than raising."""
        self._patch(monkeypatch)
        proc = cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)
        assert proc.poll() is None

    def test_failed_bringup_raises_with_stderr(self, monkeypatch):
        """If dask-ssh has already exited after the startup wait, the cluster
        never came up. setup_cluster must raise PybnfError (not return a dead
        proc that later surfaces as an opaque Client connection error), and the
        captured stderr is included for diagnosis."""
        self._patch(monkeypatch, returncode=1,
                    stderr_bytes=b'ssh: connect to host node9 port 22: Connection refused')
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.setup_cluster('node9', '/log', parallel_count=1)
        msg = str(exc.value)
        assert 'code 1' in msg
        assert 'Connection refused' in msg

    def test_parallel_count_divides_per_node_with_ceil(self, monkeypatch):
        """With an explicit parallel_count, workers are spread over nodes:
        n_per_node = ceil(parallel_count / num_nodes). 5 threads over 3 nodes ⇒
        ceil(5/3) = 2 per node (floor would give 1; multiplication 15). Branch
        flag order here is --nworkers then --nthreads."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('a b c', '/log', parallel_count=5)

        (args, _), = popen_calls
        assert args[0] == ['dask-ssh', 'a', 'b', 'c',
                           '--log-directory', '/log', '--nworkers', '2', '--nthreads', '1']

    def test_parallel_count_exact_division(self, monkeypatch):
        """4 threads over 2 nodes ⇒ exactly 2 per node (ceil of an integer is
        itself)."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('h1 h2', '/log', parallel_count=4)
        (args, _), = popen_calls
        assert args[0] == ['dask-ssh', 'h1', 'h2',
                           '--log-directory', '/log', '--nworkers', '2', '--nthreads', '1']

    def test_single_node_gets_all_workers(self, monkeypatch):
        """One node ⇒ all parallel_count workers land on it (ceil(6/1) = 6); this
        pins the divisor as the *node count*, not a constant."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('only', '/log', parallel_count=6)
        (args, _), = popen_calls
        assert args[0] == ['dask-ssh', 'only',
                           '--log-directory', '/log', '--nworkers', '6', '--nthreads', '1']

    def test_node_names_passed_as_literal_argv_no_shell(self, monkeypatch):
        """ROB-3: node names reach dask-ssh as their own literal argv entries with
        shell off, so a metacharacter-bearing node name can't be interpreted by a
        shell."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('n1$(whoami) n2', '/log', parallel_count=2)
        (args, kwargs), = popen_calls
        assert args[0][:3] == ['dask-ssh', 'n1$(whoami)', 'n2']  # literal, unexpanded
        assert kwargs.get('shell', False) is False

    def test_sleeps_ten_seconds_for_startup(self, monkeypatch):
        """After launching dask-ssh, setup_cluster waits 10s for workers to come
        up before returning the proc. Oracle: time.sleep called once with 10."""
        self._patch(monkeypatch)
        sleeps = []
        monkeypatch.setattr(cluster.time, 'sleep', lambda s: sleeps.append(s))
        cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)
        assert sleeps == [10]


# --------------------------------------------------------------------------- #
# __init__ — node-detection dispatch + Client-construction dispatch
# --------------------------------------------------------------------------- #
class _ClientStub:
    def __init__(self):
        self.run_calls = []
        self.closed = False

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))

    def close(self):
        self.closed = True


class _Recorder:
    """Records calls to the patched Client / LocalCluster factories and the
    logging hooks, so __init__'s branch choice can be read off the call args."""

    def __init__(self):
        self.client_calls = []   # (args, kwargs) for each Client(...)
        self.last_client = None
        self.lc_calls = []       # (args, kwargs) for each LocalCluster(...)
        self.last_lc = None
        self.init_logging_calls = []
        self.reinit_logging_calls = []
        self.read_calls = []
        self.setup_calls = []

    def Client(self, *args, **kwargs):
        self.client_calls.append((args, kwargs))
        self.last_client = _ClientStub()
        return self.last_client

    def LocalCluster(self, *args, **kwargs):
        self.lc_calls.append((args, kwargs))
        self.last_lc = object()
        return self.last_lc


def _patch_init(monkeypatch, read_returns=(None, None)):
    rec = _Recorder()
    monkeypatch.setattr(cluster, 'Client', rec.Client)
    monkeypatch.setattr(cluster, 'LocalCluster', rec.LocalCluster)
    monkeypatch.setattr(cluster, 'init_logging',
                        lambda *a, **k: rec.init_logging_calls.append((a, k)))
    monkeypatch.setattr(cluster, 'reinit_logging',
                        lambda *a, **k: rec.reinit_logging_calls.append((a, k)))

    def fake_read(config):
        rec.read_calls.append(config)
        return read_returns

    def fake_setup(node_string, out_dir, parallel_count):
        rec.setup_calls.append((node_string, out_dir, parallel_count))
        return 'PROC'

    monkeypatch.setattr(cluster.Cluster, 'read_node_names', staticmethod(fake_read))
    monkeypatch.setattr(cluster.Cluster, 'setup_cluster', staticmethod(fake_setup))
    return rec


def _build(cfg):
    return cluster.Cluster(cfg, log_prefix='pf', debug=False, log_level_name='INFO')


class TestInitNodeDispatch:

    def test_scheduler_file_skips_setup_and_uses_scheduler_file_client(self, monkeypatch):
        """scheduler_file set ⇒ the scheduler is read from the shared-FS file:
        no dask-ssh bring-up (_dask_proc is None, setup_cluster never called) and
        the client is built via Client(scheduler_file=...). read_node_names is
        bypassed entirely."""
        rec = _patch_init(monkeypatch)
        c = _build(_cfg(scheduler_file='/shared/sched.json'))

        assert rec.setup_calls == []
        assert c._dask_proc is None
        assert rec.read_calls == []
        assert rec.client_calls == [((), {'scheduler_file': '/shared/sched.json'})]
        assert c.local is False

    def test_scheduler_node_plus_worker_nodes_joins_worker_list(self, monkeypatch):
        """scheduler_node + explicit worker_nodes ⇒ node_string is the
        space-joined worker list (read_node_names is NOT consulted), dask-ssh is
        brought up on that list, and the client connects to scheduler_node:8786."""
        rec = _patch_init(monkeypatch)
        c = _build(_cfg(scheduler_node='head', worker_nodes=['w1', 'w2', 'w3'],
                        parallel_count=12))

        assert rec.read_calls == []
        assert rec.setup_calls == [('w1 w2 w3', os.getcwd(), 12)]
        assert c._dask_proc == 'PROC'
        assert rec.client_calls == [(('head:8786',), {})]
        assert c.local is False

    def test_scheduler_node_alone_detects_workers_via_read_node_names(self, monkeypatch):
        """scheduler_node set but no worker_nodes ⇒ the worker list comes from
        read_node_names (e.g. SLURM detection), while the scheduler stays the
        configured node. Oracle: setup_cluster gets read_node_names' node_string,
        and the client connects to the *configured* scheduler_node, not the
        detected one."""
        rec = _patch_init(monkeypatch, read_returns=('detected_head', 'd1 d2'))
        c = _build(_cfg(scheduler_node='head', parallel_count=8))

        assert len(rec.read_calls) == 1
        assert rec.setup_calls == [('d1 d2', os.getcwd(), 8)]
        assert rec.client_calls == [(('head:8786',), {})]
        assert c.local is False

    def test_detected_cluster_uses_both_outputs_of_read_node_names(self, monkeypatch):
        """Neither scheduler_file nor scheduler_node ⇒ both the scheduler and the
        worker list come from read_node_names. With a non-empty node_string,
        dask-ssh is set up and the client connects to the *detected* scheduler."""
        rec = _patch_init(monkeypatch, read_returns=('sched9', 'sched9 c1 c2'))
        c = _build(_cfg(parallel_count=4))

        assert len(rec.read_calls) == 1
        assert rec.setup_calls == [('sched9 c1 c2', os.getcwd(), 4)]
        assert rec.client_calls == [(('sched9:8786',), {})]
        assert c.local is False


class TestInitClientDispatch:

    def test_local_default_when_no_nodes_and_no_parallel_count(self, monkeypatch):
        """No node config and parallel_count=None ⇒ a default local client:
        Client() with no args, _dask_proc None, local True, and init_logging
        pushed to workers via client.run."""
        rec = _patch_init(monkeypatch, read_returns=(None, None))
        c = _build(_cfg())

        assert c._dask_proc is None
        assert rec.lc_calls == []
        assert rec.client_calls == [((), {})]
        assert c.local is True
        # init_logging is broadcast to workers through client.run(...).
        assert len(rec.last_client.run_calls) == 1
        (run_args, _), = rec.last_client.run_calls
        assert run_args == (cluster.init_logging, 'pf', False, 'INFO')

    def test_local_manual_parallel_count_builds_localcluster(self, monkeypatch):
        """No node config but parallel_count set ⇒ a manually-sized LocalCluster
        (n_workers=parallel_count, threads_per_worker=1) wrapped in a Client, and
        init_logging broadcast. Oracle: the LocalCluster kwargs and that the
        Client is built from that LocalCluster object."""
        rec = _patch_init(monkeypatch, read_returns=(None, None))
        c = _build(_cfg(parallel_count=5))

        assert rec.lc_calls == [((), {'n_workers': 5, 'threads_per_worker': 1})]
        assert rec.client_calls == [((rec.last_lc,), {})]
        assert c.local is True
        assert len(rec.last_client.run_calls) == 1

    def test_remote_clients_do_not_broadcast_init_logging(self, monkeypatch):
        """The scheduler_file / scheduler_node clients connect to an already-
        configured cluster, so they must NOT call client.run(init_logging) (that
        is only for locally-spawned workers)."""
        rec = _patch_init(monkeypatch)
        _build(_cfg(scheduler_file='/s.json'))
        assert rec.last_client.run_calls == []

    def test_reinit_logging_always_called(self, monkeypatch):
        """The distributed-version workaround: after every Client construction,
        reinit_logging is called once with (log_prefix, debug, log_level_name) —
        regardless of which branch built the client. Asserted across all four
        branches so dropping it fails here, but not pinned to a dask version."""
        for cfg in (_cfg(scheduler_file='/s.json'),
                    _cfg(scheduler_node='h', worker_nodes=['w1']),
                    _cfg(parallel_count=2),
                    _cfg()):
            rec = _patch_init(monkeypatch)
            _build(cfg)
            assert rec.reinit_logging_calls == [(('pf', False, 'INFO'), {})]


# --------------------------------------------------------------------------- #
# teardown — close the client, terminate the dask-ssh proc only if it exists
# --------------------------------------------------------------------------- #
class _ProcStub:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class TestTeardown:

    def test_closes_client_and_terminates_proc(self):
        """With a live dask-ssh proc, teardown closes the client and terminates
        the proc."""
        c = object.__new__(cluster.Cluster)
        c.client = _ClientStub()
        c._dask_proc = _ProcStub()

        c.teardown()

        assert c.client.closed is True
        assert c._dask_proc.terminated is True

    def test_no_proc_only_closes_client(self):
        """When _dask_proc is None (a local client with no dask-ssh subprocess),
        teardown closes the client and must NOT attempt to terminate None — an
        unconditional terminate would raise AttributeError here."""
        c = object.__new__(cluster.Cluster)
        c.client = _ClientStub()
        c._dask_proc = None

        c.teardown()  # must not raise

        assert c.client.closed is True
