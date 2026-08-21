"""
Orchestration tests for ``pybnf.cluster.Cluster`` — the SSH/HPC dask bring-up.

This is glue code, not numerical math: the only deterministic contracts are
*which external call PyBNF constructs, with what arguments, and which branch
fires for which config*. So the oracle for each test is the constructed
``scontrol``/``dask ssh`` command string, the SLURM stdout parse, the
subprocess-error → PybnfError mapping, the ``ceil(parallel_count/num_nodes)``
per-node arithmetic, or the ``Client(...)``/``LocalCluster(...)`` call the
config selects. (For glue with no math, "the right command/Client call was
made" *is* the oracle — not the mock-the-world anti-pattern.)

The srun launcher (#614, ADR-0122) is tested the same way: its oracle is the
``dask scheduler`` / ``srun ... dask worker`` argument lists PyBNF constructs,
the readiness polls it makes on the scheduler file and the worker count, and
the branch dispatch that keeps it away from ``dask ssh``. No SLURM and no dask
is involved -- ``Popen``, ``time.sleep`` and the ``Client`` are substituted, and
the scheduler file is a real file in ``tmp_path``.

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
import json
import os
import sys
import types

import pytest

from subprocess import TimeoutExpired, CalledProcessError

from .context import cluster, printing


# --------------------------------------------------------------------------- #
# Lightweight config stub: cluster only ever reads ``config.config[<key>]``.
# --------------------------------------------------------------------------- #
def _cfg(**overrides):
    base = {'scheduler_file': None, 'scheduler_node': None, 'worker_nodes': None,
            'parallel_count': None, 'cluster_type': None, 'output_dir': 'pybnf_output'}
    base.update(overrides)
    return types.SimpleNamespace(config=base)


# --------------------------------------------------------------------------- #
# read_node_names — SLURM parse, command string, error mapping
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, stdout):
        self.stdout = stdout


class _FakeDaskProc:
    """Stand-in for a launched Popen object: the bring-up paths only poll it, and
    terminate it when they have to abandon a partly-built cluster."""
    def __init__(self, returncode=None):
        self._returncode = returncode
        self.terminated = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True


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

    @pytest.mark.parametrize('ctype', ['slurm-srun', 'slurm_srun', 'SLURM-SRUN', 'srun'])
    def test_srun_cluster_types_read_the_same_slurm_node_list(self, monkeypatch, ctype):
        """#614: the srun launcher is a SLURM cluster too -- it reads the node list
        the same way and only starts the workers differently -- so every srun
        spelling takes the SLURM branch and returns the same names. This also pins
        the ordering hazard: ``re.match('slurm', 'slurm-srun')`` succeeds, so a
        prefix test placed first would have swallowed ``slurm-srun`` silently."""
        monkeypatch.setattr(cluster, 'run', lambda *a, **k: _FakeProc(b'n1\nn2\n'))
        assert cluster.Cluster.read_node_names(_cfg(cluster_type=ctype)) == ('n1', 'n1 n2')

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
# check_dask_subcommand — the pre-flight on the dask CLI (#615)
# --------------------------------------------------------------------------- #
class _FakeEntryPoint:
    def __init__(self, name):
        self.name = name


class TestCheckDaskSubcommand:

    @pytest.mark.parametrize('subcommand', ['ssh', 'scheduler', 'worker'])
    def test_the_installed_dask_really_provides_what_pybnf_runs(self, subcommand):
        """Deliberately NOT mocked: this asks the installed environment, through the
        same ``dask_cli`` entry point group dask's own CLI builds its command set
        from. It is the one assertion in this file that would go red if dask renamed
        or dropped a command PyBNF runs -- which is exactly what happened in #615,
        where every other test here stayed green against a ``dask-ssh`` that no
        longer existed on any current install."""
        cluster.check_dask_subcommand(subcommand)   # must not raise

    def test_missing_subcommand_names_what_is_available(self, monkeypatch):
        """The refusal has to be actionable: it says which subcommand was wanted,
        what the installation does offer, and which package supplies it."""
        monkeypatch.setattr(cluster, 'entry_points',
                            lambda group: [_FakeEntryPoint('scheduler'), _FakeEntryPoint('worker')])
        with pytest.raises(printing.PybnfError) as exc:
            cluster.check_dask_subcommand('ssh')
        assert "no 'ssh' subcommand" in str(exc.value)
        assert 'scheduler, worker' in str(exc.value)          # what is there instead
        assert 'distributed' in exc.value.message             # the hint names the package

    def test_no_dask_cli_at_all_is_a_distinct_error(self, monkeypatch):
        """A dask too old to have a command line interface is a different problem
        from a dask whose CLI lacks one command, and says so."""
        monkeypatch.setattr(cluster, 'find_spec', lambda name: None)
        with pytest.raises(printing.PybnfError, match='no command line interface'):
            cluster.check_dask_subcommand('ssh')

    def test_the_group_queried_is_the_one_dask_itself_reads(self, monkeypatch):
        """The check is only as good as the question it asks: dask assembles its
        subcommands from the ``dask_cli`` entry point group, so anything this cannot
        see, ``dask`` cannot run either."""
        groups = []
        monkeypatch.setattr(cluster, 'entry_points',
                            lambda group: groups.append(group) or [_FakeEntryPoint('ssh')])
        cluster.check_dask_subcommand('ssh')
        assert groups == ['dask_cli']


# --------------------------------------------------------------------------- #
# setup_cluster — the dask ssh command string + per-node arithmetic
# --------------------------------------------------------------------------- #
# What PyBNF prepends to every worker-launch command (#615). Spelled out here
# rather than imported from cluster.DASK_CLI so that the tests below pin the
# actual argv, not merely agree with whatever the module happens to build.
DASK_SSH = [sys.executable, '-m', 'dask', 'ssh']


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
        """parallel_count=None ⇒ dask ssh's own default of one worker per CPU:
        ``--nthreads 1 --nworkers {cpu_count()}`` (note this branch's flag order is
        --nthreads then --nworkers). Oracle: the exact argument list (ROB-3: an argv
        list launched with no shell, each node its own entry) with cpu_count()=7."""
        popen_calls = self._patch(monkeypatch, cpu=7)
        proc = cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=None)

        assert proc.poll() is None
        (args, kwargs), = popen_calls
        assert args[0] == [*DASK_SSH, 'n1', 'n2',
                           '--log-directory', '/out', '--nthreads', '1', '--nworkers', '7']
        assert kwargs.get('shell', False) is False         # no shell -> no injection
        assert kwargs['stdout'] is cluster.DEVNULL
        # stderr is captured to a readable file (not discarded), so an early
        # bring-up failure can be surfaced — see test_failed_bringup_*.
        assert kwargs['stderr'] is not cluster.DEVNULL
        assert hasattr(kwargs['stderr'], 'read')

    def test_the_launcher_is_dask_ssh_through_this_interpreter(self, monkeypatch):
        """#615: the command is the ``dask ssh`` *subcommand*, run through the
        interpreter running PyBNF -- not the standalone ``dask-ssh`` script, which
        distributed stopped installing in 2026.6.0 and whose absence killed every
        multi-machine run on FileNotFoundError, and not a bare ``dask`` from PATH,
        which could belong to a different environment than this fit (dask ssh
        passes its own sys.executable on to the remote workers)."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)

        (args, _), = popen_calls
        assert args[0][:4] == [sys.executable, '-m', 'dask', 'ssh']
        assert 'dask-ssh' not in args[0]
        assert args[0][0] != 'dask'

    def test_a_missing_subcommand_is_refused_before_launching(self, monkeypatch):
        """The pre-flight runs *before* Popen, so a dask that cannot do this reads
        as a configuration error naming what is installed, rather than as a
        FileNotFoundError traceback ("an unknown error ... please report this bug")
        from a process PyBNF already tried to start."""
        popen_calls = self._patch(monkeypatch)
        monkeypatch.setattr(cluster, 'entry_points', lambda group: [])
        with pytest.raises(printing.PybnfError, match="no 'ssh' subcommand"):
            cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)
        assert popen_calls == []

    def test_running_proc_is_returned_without_raising(self, monkeypatch):
        """The happy path: dask ssh is still running after the startup wait
        (poll() is None), so setup_cluster returns the proc rather than raising."""
        self._patch(monkeypatch)
        proc = cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)
        assert proc.poll() is None

    def test_failed_bringup_raises_with_stderr(self, monkeypatch):
        """If dask ssh has already exited after the startup wait, the cluster
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
        assert args[0] == [*DASK_SSH, 'a', 'b', 'c',
                           '--log-directory', '/log', '--nworkers', '2', '--nthreads', '1']

    def test_parallel_count_exact_division(self, monkeypatch):
        """4 threads over 2 nodes ⇒ exactly 2 per node (ceil of an integer is
        itself)."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('h1 h2', '/log', parallel_count=4)
        (args, _), = popen_calls
        assert args[0] == [*DASK_SSH, 'h1', 'h2',
                           '--log-directory', '/log', '--nworkers', '2', '--nthreads', '1']

    def test_single_node_gets_all_workers(self, monkeypatch):
        """One node ⇒ all parallel_count workers land on it (ceil(6/1) = 6); this
        pins the divisor as the *node count*, not a constant."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('only', '/log', parallel_count=6)
        (args, _), = popen_calls
        assert args[0] == [*DASK_SSH, 'only',
                           '--log-directory', '/log', '--nworkers', '6', '--nthreads', '1']

    def test_node_names_passed_as_literal_argv_no_shell(self, monkeypatch):
        """ROB-3: node names reach dask ssh as their own literal argv entries with
        shell off, so a metacharacter-bearing node name can't be interpreted by a
        shell."""
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('n1$(whoami) n2', '/log', parallel_count=2)
        (args, kwargs), = popen_calls
        assert args[0][:len(DASK_SSH) + 2] == [*DASK_SSH, 'n1$(whoami)', 'n2']  # literal, unexpanded
        assert kwargs.get('shell', False) is False

    def test_sleeps_ten_seconds_for_startup(self, monkeypatch):
        """After launching dask ssh, setup_cluster waits 10s for workers to come
        up before returning the proc. Oracle: time.sleep called once with 10."""
        self._patch(monkeypatch)
        sleeps = []
        monkeypatch.setattr(cluster.time, 'sleep', lambda s: sleeps.append(s))
        cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)
        assert sleeps == [10]


# --------------------------------------------------------------------------- #
# __init__ — node-detection dispatch + Client-construction dispatch
# --------------------------------------------------------------------------- #
class _ProcStub:
    """Stand-in for a process PyBNF started and later terminates."""
    def __init__(self, returncode=None):
        self.terminated = False
        self._returncode = returncode

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True


class _ClientStub:
    def __init__(self, workers=('tcp://n1:1',)):
        self.run_calls = []
        self.closed = False
        self._workers = dict.fromkeys(workers, {})

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))

    def scheduler_info(self):
        return {'workers': self._workers}

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
        self.srun_setup_calls = []   # (scheduler_file, out_dir, node_count, parallel_count)
        self.srun_wait_calls = []    # (client, srun_proc, worker_log)

    def Client(self, *args, **kwargs):
        self.client_calls.append((args, kwargs))
        self.last_client = _ClientStub()
        return self.last_client

    def LocalCluster(self, *args, **kwargs):
        self.lc_calls.append((args, kwargs))
        self.last_lc = object()
        return self.last_lc


def _patch_init(monkeypatch, read_returns=(None, None), srun_raises=None):
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

    def fake_setup_srun(scheduler_file, out_dir, node_count, parallel_count):
        rec.srun_setup_calls.append((scheduler_file, out_dir, node_count, parallel_count))
        return _ProcStub(), _ProcStub()

    def fake_wait(client, srun_proc, worker_log, **kwargs):
        rec.srun_wait_calls.append((client, srun_proc, worker_log))
        if srun_raises:
            raise srun_raises
        return 1

    monkeypatch.setattr(cluster.Cluster, 'read_node_names', staticmethod(fake_read))
    monkeypatch.setattr(cluster.Cluster, 'setup_cluster', staticmethod(fake_setup))
    monkeypatch.setattr(cluster.Cluster, 'setup_srun_cluster', staticmethod(fake_setup_srun))
    monkeypatch.setattr(cluster.Cluster, 'wait_for_srun_workers', staticmethod(fake_wait))
    monkeypatch.setattr(cluster.Cluster, 'require_slurm_allocation', staticmethod(lambda: None))
    return rec


def _build(cfg):
    return cluster.Cluster(cfg, log_prefix='pf', debug=False, log_level_name='INFO')


class TestInitNodeDispatch:

    def test_scheduler_file_skips_setup_and_uses_scheduler_file_client(self, monkeypatch):
        """scheduler_file set ⇒ the scheduler is read from the shared-FS file:
        no dask ssh bring-up (_dask_proc is None, setup_cluster never called) and
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
        space-joined worker list (read_node_names is NOT consulted), dask ssh is
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
        dask ssh is set up and the client connects to the *detected* scheduler."""
        rec = _patch_init(monkeypatch, read_returns=('sched9', 'sched9 c1 c2'))
        c = _build(_cfg(parallel_count=4))

        assert len(rec.read_calls) == 1
        assert rec.setup_calls == [('sched9 c1 c2', os.getcwd(), 4)]
        assert rec.client_calls == [(('sched9:8786',), {})]
        assert c.local is False


class TestInitClientDispatch:

    def test_local_default_when_no_nodes_and_no_parallel_count(self, monkeypatch):
        """No node config and parallel_count=None ⇒ a LocalCluster that pins
        threads_per_worker=1 and leaves n_workers to dask (#526), wrapped in a
        Client; _dask_proc None, local True, and init_logging pushed to workers
        via client.run. Omitting n_workers is deliberate: given one thread per
        worker, dask sizes the pool at one worker per available core, so total
        concurrency matches the old bare Client() default."""
        rec = _patch_init(monkeypatch, read_returns=(None, None))
        c = _build(_cfg())

        assert c._dask_proc is None
        assert rec.lc_calls == [((), {'threads_per_worker': 1})]
        assert rec.client_calls == [((rec.last_lc,), {})]
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

    @pytest.mark.parametrize('parallel_count', [None, 1, 4, 36])
    def test_every_local_client_is_single_threaded_per_worker(self, monkeypatch, parallel_count):
        """#526: whether a locally-spawned worker runs one thread or several must
        not depend on an unrelated key. Setting parallel_count chooses the number
        of worker *processes*; every local client pins threads_per_worker=1,
        because the simulation backends carry process-wide state that is not
        thread-safe (#525's sympy->C printer race is one instance).

        This is the oracle the old code failed: the parallel_count branch pinned
        1, the default branch let dask pick (several threads per worker on any
        machine with >4 cores)."""
        rec = _patch_init(monkeypatch, read_returns=(None, None))
        _build(_cfg(parallel_count=parallel_count))

        (_, lc_kwargs), = rec.lc_calls
        assert lc_kwargs['threads_per_worker'] == 1

    def test_local_and_dask_ssh_defaults_agree_on_one_thread(self, monkeypatch):
        """The two default paths -- local and dask ssh -- must request the same
        thread-per-worker policy, since the same non-thread-safe backends run on
        both. Oracle: with nothing configured, dask ssh asks for --nthreads 1 and
        the local LocalCluster asks for threads_per_worker=1.

        (The dask ssh half runs first: _patch_init replaces setup_cluster with a
        fake, so the real one has to be exercised before that.)"""
        popen_calls = []
        monkeypatch.setattr(cluster, 'Popen',
                            lambda *a, **k: popen_calls.append((a, k)) or _FakeDaskProc())
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        monkeypatch.setattr(cluster, 'cpu_count', lambda: 7)
        cluster.Cluster.setup_cluster('n1', '/out', parallel_count=None)
        (ssh_args, _), = popen_calls
        cmd = ssh_args[0]

        rec = _patch_init(monkeypatch, read_returns=(None, None))
        _build(_cfg())
        (_, lc_kwargs), = rec.lc_calls

        assert cmd[cmd.index('--nthreads') + 1] == '1'
        assert lc_kwargs['threads_per_worker'] == 1


class TestLocalClusterKwargs:
    """The single place the local thread/worker policy is decided (#526)."""

    def test_default_omits_n_workers_so_dask_sizes_by_core_count(self):
        """parallel_count=None ⇒ only threads_per_worker is specified. n_workers
        is deliberately left out rather than computed here: dask derives it from
        dask.system.CPU_COUNT, which honors CPU affinity and cgroup quotas, so a
        run inside a 2-core container gets 2 workers rather than the host's core
        count."""
        assert cluster.Cluster.local_cluster_kwargs(None) == {'threads_per_worker': 1}

    @pytest.mark.parametrize('parallel_count', [1, 3, 40])
    def test_explicit_count_becomes_n_workers_not_threads(self, parallel_count):
        """parallel_count is a *process* count: it lands in n_workers and never
        raises threads_per_worker above 1."""
        assert cluster.Cluster.local_cluster_kwargs(parallel_count) == {
            'n_workers': parallel_count, 'threads_per_worker': 1}

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
# teardown — close the client, terminate the launcher proc only if it exists
# --------------------------------------------------------------------------- #
def _torn_down(client=None, dask_proc=None, scheduler_proc=None, scheduler_file=None):
    """A Cluster built by hand, carrying only the attributes teardown reads."""
    c = object.__new__(cluster.Cluster)
    c.client = client if client is not None else _ClientStub()
    c._dask_proc = dask_proc
    c._scheduler_proc = scheduler_proc
    c._own_scheduler_file = scheduler_file
    return c


class TestTeardown:

    def test_closes_client_and_terminates_proc(self):
        """With a live dask ssh proc, teardown closes the client and terminates
        the proc."""
        proc = _ProcStub()
        c = _torn_down(dask_proc=proc)

        c.teardown()

        assert c.client.closed is True
        assert proc.terminated is True

    def test_no_proc_only_closes_client(self):
        """When _dask_proc is None (a local client with no dask ssh subprocess),
        teardown closes the client and must NOT attempt to terminate None — an
        unconditional terminate would raise AttributeError here."""
        c = _torn_down()

        c.teardown()  # must not raise

        assert c.client.closed is True

    def test_srun_teardown_stops_workers_then_scheduler_and_removes_the_file(self, tmp_path):
        """#614: under the srun launcher PyBNF owns both processes and the
        scheduler file, so teardown terminates both and deletes the file. Order
        matters -- srun (the workers) is signalled before the scheduler they talk
        to -- and the file must go, since a connection file naming a scheduler
        that is shutting down is exactly what the next run would mistake for a
        live cluster."""
        order = []
        sched_file = tmp_path / 'dask_scheduler.json'
        sched_file.write_text('{"address": "tcp://n1:8786"}')
        srun_proc, scheduler_proc = _ProcStub(), _ProcStub()
        srun_proc.terminate = lambda: order.append('workers')
        scheduler_proc.terminate = lambda: order.append('scheduler')
        c = _torn_down(dask_proc=srun_proc, scheduler_proc=scheduler_proc,
                       scheduler_file=str(sched_file))

        c.teardown()

        assert c.client.closed is True
        assert order == ['workers', 'scheduler']
        assert not sched_file.exists()

    def test_teardown_leaves_a_scheduler_file_pybnf_did_not_write(self, tmp_path):
        """A ``scheduler_file`` run attaches to a cluster someone else brought up:
        PyBNF neither started those processes nor wrote that file, so teardown
        must not delete it (_own_scheduler_file is None on that path)."""
        sched_file = tmp_path / 'their_cluster.json'
        sched_file.write_text('{"address": "tcp://n1:8786"}')
        c = _torn_down()

        c.teardown()

        assert sched_file.exists()

    def test_stop_own_processes_is_idempotent(self, tmp_path):
        """stop_own_processes runs both from a failed bring-up and from teardown,
        so calling it twice must not terminate an already-terminated process or
        fail on the file it just deleted."""
        sched_file = tmp_path / 'dask_scheduler.json'
        sched_file.write_text('{}')
        srun_proc, scheduler_proc = _ProcStub(), _ProcStub()
        c = _torn_down(dask_proc=srun_proc, scheduler_proc=scheduler_proc,
                       scheduler_file=str(sched_file))

        c.stop_own_processes()
        c.stop_own_processes()  # must not raise

        assert (srun_proc.terminated, scheduler_proc.terminated) == (True, True)
        assert (c._dask_proc, c._scheduler_proc, c._own_scheduler_file) == (None, None, None)


# --------------------------------------------------------------------------- #
# The srun launcher (#614, ADR-0122)
# --------------------------------------------------------------------------- #
class TestUsesSrun:
    """Which cluster_type values select the srun launcher."""

    @pytest.mark.parametrize('ctype', ['slurm-srun', 'slurm_srun', 'slurmsrun', 'srun',
                                       'SLURM-SRUN', 'Slurm_Srun', '  slurm-srun  '])
    def test_accepted_spellings(self, ctype):
        """The documented spelling is ``slurm-srun``; the underscore, run-together
        and bare-``srun`` forms are accepted too, case-insensitively and with
        surrounding whitespace stripped, so a reasonable guess is not answered
        with "Unknown cluster type"."""
        assert cluster.uses_srun(ctype) is True

    @pytest.mark.parametrize('ctype', [None, '', 'slurm', 'SLURM', 'torque', 'pbs',
                                       'srunny', 'slurm srun', 'srun-slurm'])
    def test_rejected_spellings(self, ctype):
        """Everything else is not the srun launcher. ``slurm`` in particular must
        stay on the SSH launcher (matched here by fullmatch, so a value that
        merely *starts* with a recognized word does not select it)."""
        assert cluster.uses_srun(ctype) is False


class TestRequireSlurmAllocation:

    def test_allocation_present_is_accepted(self, monkeypatch):
        """Inside an allocation, the check passes silently."""
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        cluster.Cluster.require_slurm_allocation()  # must not raise

    def test_legacy_variable_is_accepted(self, monkeypatch):
        """Older SLURM exports the allocation as $SLURM_JOBID; either name counts."""
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        monkeypatch.setenv('SLURM_JOBID', '12345')
        cluster.Cluster.require_slurm_allocation()  # must not raise

    def test_no_allocation_is_refused_with_a_remedy(self, monkeypatch):
        """Outside an allocation, srun does not *place* a task -- it submits a job
        and waits for one, which would read as PyBNF hanging with no output. That
        is refused up front, and the message says where to start PyBNF instead."""
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        monkeypatch.delenv('SLURM_JOBID', raising=False)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.require_slurm_allocation()
        assert 'SLURM_JOB_ID' in str(exc.value)
        # The remedy is a hint, so it reaches the user's message without displacing
        # the diagnosis (#527): both are present in what gets printed.
        assert 'SLURM_JOB_ID' in exc.value.message
        assert 'salloc' in exc.value.message


class TestSrunSchedulerFile:

    def test_defaults_into_the_output_directory(self):
        """With no scheduler_file set, PyBNF writes the connection file into the
        output directory -- which a cluster run already requires to be on the
        shared filesystem the workers read. Absolute, so it means the same thing
        in the srun command as it does here."""
        path = cluster.Cluster.srun_scheduler_file(_cfg(output_dir='out'))
        assert path == os.path.abspath(os.path.join('out', 'dask_scheduler.json'))

    def test_scheduler_file_chooses_where_it_is_written(self):
        """Under this launcher the scheduler file is an *output* (PyBNF starts the
        scheduler that writes it), so scheduler_file selects the path rather than
        naming a cluster to attach to."""
        path = cluster.Cluster.srun_scheduler_file(
            _cfg(scheduler_file='/shared/mine.json', output_dir='out'))
        assert path == '/shared/mine.json'


class TestCpusPerNode:

    def test_reads_what_slurm_granted(self, monkeypatch):
        """$SLURM_CPUS_ON_NODE is what the *allocation* granted, which is the
        number the launcher can actually ask SLURM for."""
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '12')
        monkeypatch.setattr(cluster, 'cpu_count', lambda: 64)
        assert cluster.Cluster.cpus_per_node() == 12

    @pytest.mark.parametrize('value', [None, '', 'many', '0', '-4'])
    def test_falls_back_to_the_machine_core_count(self, monkeypatch, value):
        """With nothing usable in the environment, fall back to the machine's own
        core count (the number the SSH launcher uses unconditionally)."""
        monkeypatch.delenv('SLURM_CPUS_ON_NODE', raising=False)
        if value is not None:
            monkeypatch.setenv('SLURM_CPUS_ON_NODE', value)
        monkeypatch.setattr(cluster, 'cpu_count', lambda: 64)
        assert cluster.Cluster.cpus_per_node() == 64


class TestSrunWorkerCommand:

    def _patch(self, monkeypatch, granted=8):
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', str(granted))

    def test_default_is_one_worker_per_granted_cpu(self, monkeypatch):
        """parallel_count unset ⇒ one single-threaded worker per CPU the job holds
        on a node, one srun task per node, and that task given all the CPUs its
        workers need. Oracle: the exact argument list (a literal argv list run
        with no shell, ROB-3), including the interpreter running the workers --
        this process's own, not whatever ``dask`` the remote PATH resolves to."""
        self._patch(monkeypatch, granted=8)
        cmd = cluster.Cluster.srun_worker_command('/shared/s.json', 3, parallel_count=None)
        assert cmd == ['srun', '--nodes', '3', '--ntasks', '3', '--ntasks-per-node', '1',
                       '--cpus-per-task', '8', '--label',
                       sys.executable, '-m', 'dask', 'worker',
                       '--scheduler-file', '/shared/s.json',
                       '--nworkers', '8', '--nthreads', '1']

    def test_parallel_count_divides_per_node_with_ceil(self, monkeypatch):
        """parallel_count is a total over all nodes, divided the same way the SSH
        launcher divides it: ceil(5/3) = 2 workers per node (floor would give 1)."""
        self._patch(monkeypatch, granted=8)
        cmd = cluster.Cluster.srun_worker_command('/s.json', 3, parallel_count=5)
        assert cmd[cmd.index('--nworkers') + 1] == '2'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '2'

    def test_cpus_requested_match_the_workers_started(self, monkeypatch):
        """The CPU request is not decoration: with task/cgroup binding, a task that
        took the default single CPU would confine every worker it forks to that one
        CPU and quietly serialize the node. So --cpus-per-task tracks --nworkers."""
        self._patch(monkeypatch, granted=16)
        cmd = cluster.Cluster.srun_worker_command('/s.json', 2, parallel_count=8)
        assert cmd[cmd.index('--nworkers') + 1] == '4'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '4'

    def test_oversubscription_caps_the_cpu_request_not_the_worker_count(self, monkeypatch):
        """A parallel_count above what the job holds is the user deliberately
        oversubscribing, which the SSH launcher has always allowed. SLURM refuses a
        request for more CPUs than the job holds, so the *request* is capped at the
        allocation while the requested number of workers still starts."""
        self._patch(monkeypatch, granted=4)
        cmd = cluster.Cluster.srun_worker_command('/s.json', 1, parallel_count=16)
        assert cmd[cmd.index('--nworkers') + 1] == '16'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '4'

    def test_never_asks_for_zero_workers(self, monkeypatch):
        """parallel_count = 0 is not validated anywhere upstream; ``--nworkers 0``
        would start a cluster that can never run a job, so the floor is one."""
        self._patch(monkeypatch, granted=4)
        cmd = cluster.Cluster.srun_worker_command('/s.json', 2, parallel_count=0)
        assert cmd[cmd.index('--nworkers') + 1] == '1'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '1'

    def test_workers_are_single_threaded(self, monkeypatch):
        """Same policy as every other worker PyBNF starts (#526, ADR-0089): the
        simulation backends hold process-wide state that is not thread-safe, so a
        worker process runs one job at a time."""
        self._patch(monkeypatch)
        cmd = cluster.Cluster.srun_worker_command('/s.json', 2)
        assert cmd[cmd.index('--nthreads') + 1] == '1'

    def test_scheduler_file_is_one_literal_argument(self, monkeypatch):
        """ROB-3: the path reaches srun as a single literal argv entry, so a path
        carrying shell metacharacters is never interpreted by a shell."""
        self._patch(monkeypatch)
        cmd = cluster.Cluster.srun_worker_command('/tmp/a b$(whoami).json', 1)
        assert cmd[cmd.index('--scheduler-file') + 1] == '/tmp/a b$(whoami).json'


class _SchedulerSpy:
    """A fake ``Popen`` for the srun launcher: records each command, and optionally
    writes the scheduler file the way ``dask scheduler`` would."""

    def __init__(self, scheduler_file=None, address='tcp://n1:8786', returncode=None,
                 write_after=0, log_text=b''):
        self.calls = []                       # (cmd, kwargs) per launch
        self.procs = []
        self._scheduler_file = scheduler_file
        self._address = address
        self._returncode = returncode
        self._write_after = write_after       # polls to wait before the file appears
        self._log_text = log_text

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self._log_text:
            kwargs['stdout'].write(self._log_text)
            kwargs['stdout'].flush()
        proc = _FakeDaskProc(self._returncode if len(self.calls) == 1 else None)
        self.procs.append(proc)
        if self._scheduler_file is not None and len(self.calls) == 1:
            self._sleeps = 0
        return proc

    def sleep(self, _seconds):
        """Stands in for time.sleep: the scheduler file appears after N polls."""
        self._sleeps = getattr(self, '_sleeps', 0) + 1
        if self._scheduler_file is not None and self._sleeps >= self._write_after:
            with open(self._scheduler_file, 'w') as f:
                json.dump({'type': 'Scheduler', 'address': self._address}, f)


class TestWaitForSchedulerFile:

    def test_returns_the_address_once_the_file_is_complete(self, monkeypatch, tmp_path):
        """The scheduler's readiness signal is its connection file: the wait returns
        the address the file names, so the caller can log where the cluster is."""
        sched_file = tmp_path / 's.json'
        spy = _SchedulerSpy(str(sched_file), address='tcp://10.0.0.1:8786', write_after=2)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        address = cluster.Cluster.wait_for_scheduler_file(
            str(sched_file), _FakeDaskProc(), str(tmp_path / 'sched.log'))
        assert address == 'tcp://10.0.0.1:8786'

    def test_a_half_written_file_is_not_treated_as_ready(self, monkeypatch, tmp_path):
        """dask writes the file in place rather than renaming it into place, so a
        reader can catch it half-written. Requiring it to parse as JSON carrying an
        address is what makes its appearance a readiness signal and not a race:
        here the first poll sees a truncated file and the wait continues."""
        sched_file = tmp_path / 's.json'
        sched_file.write_text('{"type": "Sched')          # torn mid-write
        polls = []

        def fake_sleep(_seconds):
            polls.append(1)
            if len(polls) == 2:
                sched_file.write_text('{"address": "tcp://n2:8786"}')

        monkeypatch.setattr(cluster.time, 'sleep', fake_sleep)
        address = cluster.Cluster.wait_for_scheduler_file(
            str(sched_file), _FakeDaskProc(), str(tmp_path / 'sched.log'))
        assert address == 'tcp://n2:8786'
        assert len(polls) == 2                            # it kept waiting rather than failing

    def test_a_dead_scheduler_is_reported_immediately_with_its_log(self, monkeypatch, tmp_path):
        """A scheduler that exits (an occupied port, a bad interpreter) is reported
        as soon as it exits rather than after the full timeout, and its log is
        quoted -- that text is the only place the reason exists."""
        log = tmp_path / 'sched.log'
        log.write_text('OSError: [Errno 48] Address already in use')
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_scheduler_file(
                str(tmp_path / 'absent.json'), _FakeDaskProc(returncode=1), str(log))
        assert 'code 1' in str(exc.value)
        assert 'Address already in use' in str(exc.value)

    def test_a_file_that_never_appears_times_out_naming_the_log(self, monkeypatch, tmp_path):
        """A scheduler that stays alive but never writes the file cannot be waited
        on forever; the error names the file and the log to read."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_scheduler_file(
                str(tmp_path / 'absent.json'), _FakeDaskProc(), str(tmp_path / 'sched.log'),
                timeout=1.)
        assert 'absent.json' in str(exc.value)
        assert 'sched.log' in str(exc.value)


class TestWaitForSrunWorkers:

    def test_returns_once_a_worker_registers(self, monkeypatch):
        """The readiness signal for the workers is a worker registering with the
        scheduler -- not srun having been launched, which says nothing."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        n = cluster.Cluster.wait_for_srun_workers(
            _ClientStub(workers=('tcp://n1:1', 'tcp://n2:1')), _FakeDaskProc(), '/log')
        assert n == 2

    def test_waits_while_the_cluster_is_still_empty(self, monkeypatch):
        """A scheduler with no workers yet is not an error: the poll continues
        until the workers arrive."""
        client = _ClientStub(workers=())
        polls = []

        def fake_sleep(_seconds):
            polls.append(1)
            if len(polls) == 3:
                client._workers = {'tcp://n1:1': {}}

        monkeypatch.setattr(cluster.time, 'sleep', fake_sleep)
        assert cluster.Cluster.wait_for_srun_workers(client, _FakeDaskProc(), '/log') == 1
        assert len(polls) == 3

    def test_a_transient_scheduler_error_is_not_fatal(self, monkeypatch):
        """A failed round-trip to the scheduler during bring-up is a hiccup, not a
        verdict: it counts as "no workers yet" and the poll continues."""
        class _FlakyClient(_ClientStub):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def scheduler_info(self):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError('connection reset')
                return {'workers': {'tcp://n1:1': {}}}

        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        assert cluster.Cluster.wait_for_srun_workers(_FlakyClient(), _FakeDaskProc(), '/log') == 1

    def test_srun_exiting_early_is_reported_with_its_log(self, monkeypatch, tmp_path):
        """srun exiting before any worker registered means the placement failed --
        a bad flag, a request larger than the allocation. Reported at once, quoting
        srun's own message rather than waiting out the timeout."""
        log = tmp_path / 'workers.log'
        log.write_text('srun: error: Unable to allocate resources')
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_srun_workers(
                _ClientStub(workers=()), _FakeDaskProc(returncode=1), str(log))
        assert 'code 1' in str(exc.value)
        assert 'Unable to allocate resources' in str(exc.value)

    def test_no_worker_in_time_names_the_log_and_the_step_hazard(self, monkeypatch, tmp_path):
        """srun still running with no worker registered is the shape of a queued
        job step: connecting to our own scheduler succeeded, so nothing else would
        report it, and the fit would submit jobs no one takes. The message points
        at srun's own log and at the likely cause."""
        log = tmp_path / 'workers.log'
        log.write_text('srun: Job step creation temporarily disabled, retrying')
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_srun_workers(
                _ClientStub(workers=()), _FakeDaskProc(), str(log), timeout=1.)
        assert 'srun: Job step creation' in str(exc.value)   # srun's own words, in the log
        assert 'workers.log' in exc.value.message            # ... and where to read more
        assert 'job step' in exc.value.message.lower()


class TestSetupSrunCluster:

    def test_starts_the_scheduler_here_then_the_workers_with_srun(self, monkeypatch, tmp_path):
        """The whole point of the launcher: two commands, neither of which logs in
        anywhere. The scheduler runs on this node as an ordinary subprocess and is
        told to write the connection file; the workers are one srun task per node
        reading that same file. Both are argv lists run with no shell, and both
        write to a named log in the output directory (an undrained pipe would
        deadlock a process that outlives this call)."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '4')

        scheduler_proc, srun_proc = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), 2, parallel_count=None)

        (sched_cmd, sched_kwargs), (srun_cmd, srun_kwargs) = spy.calls
        assert sched_cmd == [sys.executable, '-m', 'dask', 'scheduler',
                             '--scheduler-file', str(sched_file)]
        assert srun_cmd[0] == 'srun'
        assert srun_cmd[srun_cmd.index('--scheduler-file') + 1] == str(sched_file)
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '4'
        for kwargs in (sched_kwargs, srun_kwargs):
            assert kwargs.get('shell', False) is False
            assert kwargs['stderr'] is cluster.STDOUT
            assert hasattr(kwargs['stdout'], 'write')
        assert (tmp_path / 'dask_scheduler.log').exists()
        assert (tmp_path / 'dask_workers.log').exists()
        assert (scheduler_proc, srun_proc) == (spy.procs[0], spy.procs[1])

    def test_the_workers_are_started_only_after_the_scheduler_is_ready(self, monkeypatch, tmp_path):
        """Ordering is load-bearing: a worker started before the scheduler file
        exists has nothing to read. srun is launched only after the wait returns."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=3)
        launched_at = []

        def fake_sleep(seconds):
            launched_at.append(len(spy.calls))
            spy.sleep(seconds)

        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', fake_sleep)
        cluster.Cluster.setup_srun_cluster(str(sched_file), str(tmp_path), 1)

        assert launched_at == [1, 1, 1]      # only the scheduler was running while waiting
        assert len(spy.calls) == 2

    def test_a_stale_scheduler_file_is_removed_before_the_scheduler_starts(self, monkeypatch, tmp_path):
        """A file left by an earlier run names a scheduler that is no longer
        listening. It is removed first, so the file's reappearance is proof that
        *this* scheduler started -- otherwise the wait would return instantly and
        the client would connect to nothing."""
        sched_file = tmp_path / 'dask_scheduler.json'
        sched_file.write_text('{"address": "tcp://dead:8786"}')
        spy = _SchedulerSpy(str(sched_file), address='tcp://live:8786', write_after=1)
        seen = []
        monkeypatch.setattr(cluster, 'Popen',
                            lambda cmd, **k: seen.append(sched_file.exists()) or spy(cmd, **k))
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)

        cluster.Cluster.setup_srun_cluster(str(sched_file), str(tmp_path), 1)

        assert seen[0] is False                             # gone before the scheduler started
        assert json.loads(sched_file.read_text())['address'] == 'tcp://live:8786'

    def test_a_scheduler_that_dies_takes_no_srun_with_it(self, monkeypatch, tmp_path):
        """If the scheduler never comes up, the workers are not started at all --
        and the dead scheduler process is terminated rather than left behind by a
        constructor that raised before there was a Cluster to tear down."""
        spy = _SchedulerSpy(returncode=1, log_text=b'Address already in use')
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)

        with pytest.raises(printing.PybnfError, match='scheduler exited'):
            cluster.Cluster.setup_srun_cluster(
                str(tmp_path / 'dask_scheduler.json'), str(tmp_path), 2)

        assert len(spy.calls) == 1                          # srun was never launched
        assert spy.procs[0].terminated is True

    def test_a_missing_scheduler_file_directory_is_refused_up_front(self, tmp_path):
        """A scheduler file in a directory that does not exist would fail inside
        dask, as a traceback in a log file nobody is watching. Checked here, where
        it can be a configuration error naming the path."""
        with pytest.raises(printing.PybnfError, match='scheduler file .* does not exist'):
            cluster.Cluster.setup_srun_cluster(
                str(tmp_path / 'no_such_dir' / 's.json'), str(tmp_path), 1)

    def test_a_missing_log_directory_is_refused_up_front(self, tmp_path):
        """Likewise for the directory the logs go in: opening that file is the first
        thing the bring-up does, and a bare OSError from it would reach the user as
        "an unknown error ... please report this bug"."""
        with pytest.raises(printing.PybnfError, match='logs .* does not exist'):
            cluster.Cluster.setup_srun_cluster(
                str(tmp_path / 's.json'), str(tmp_path / 'no_such_dir'), 1)


class TestInitSrunDispatch:

    def _cfg_srun(self, **overrides):
        return _cfg(cluster_type='slurm-srun', **overrides)

    def test_srun_type_never_reaches_dask_ssh(self, monkeypatch):
        """#614's whole point: with the srun launcher selected, the SSH bring-up
        must not run. (``re.match('slurm', 'slurm-srun')`` succeeds, so a dispatch
        that tested the SSH branch first would have started dask ssh here and
        failed the login this issue is about.)"""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1 n2'))
        c = _build(self._cfg_srun())

        assert rec.setup_calls == []                        # no dask ssh
        assert len(rec.srun_setup_calls) == 1
        assert c.local is False

    def test_brings_up_srun_with_the_node_count_and_connects_by_file(self, monkeypatch):
        """The srun bring-up gets the scheduler-file path, the output directory,
        the *number* of nodes (all srun needs -- it places the workers itself) and
        parallel_count; the client then connects through that file."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1 n2 n3'))
        c = _build(self._cfg_srun(output_dir='out', parallel_count=12))

        expected_file = os.path.abspath(os.path.join('out', 'dask_scheduler.json'))
        assert rec.srun_setup_calls == [(expected_file, 'out', 3, 12)]
        assert rec.client_calls == [((), {'scheduler_file': expected_file})]
        assert c._own_scheduler_file == expected_file

    def test_scheduler_file_chooses_the_path_rather_than_an_existing_cluster(self, monkeypatch):
        """With the srun launcher, scheduler_file says *where PyBNF writes*; the
        cluster is still brought up. (Without the launcher the same key means the
        opposite -- attach to a cluster someone else started -- and that path must
        keep starting nothing.)"""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1'))
        _build(self._cfg_srun(scheduler_file='/shared/mine.json'))
        assert rec.srun_setup_calls[0][0] == '/shared/mine.json'
        assert rec.client_calls == [((), {'scheduler_file': '/shared/mine.json'})]

        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1'))
        c = _build(_cfg(scheduler_file='/shared/theirs.json'))
        assert rec.srun_setup_calls == []
        assert c._scheduler_proc is None

    def test_the_allocation_is_checked_before_anything_is_started(self, monkeypatch):
        """The refusal outside an allocation is a precondition, not a diagnosis
        after the fact: nothing is launched and no client is built."""
        real_check = cluster.Cluster.require_slurm_allocation
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1'))
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        monkeypatch.delenv('SLURM_JOBID', raising=False)
        monkeypatch.setattr(cluster.Cluster, 'require_slurm_allocation', staticmethod(real_check))
        with pytest.raises(printing.PybnfError, match='SLURM_JOB_ID'):
            _build(self._cfg_srun())
        assert rec.srun_setup_calls == []
        assert rec.client_calls == []

    def test_waits_for_the_workers_before_returning(self, monkeypatch):
        """Handing back a client whose cluster has no workers would turn a failed
        placement into a fit that submits jobs and never gets one back, so the
        constructor does not return until a worker has registered."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1 n2'))
        c = _build(self._cfg_srun(output_dir='out'))

        assert len(rec.srun_wait_calls) == 1
        client, srun_proc, worker_log = rec.srun_wait_calls[0]
        assert client is rec.last_client
        assert srun_proc is c._dask_proc
        assert worker_log == os.path.join('out', 'dask_workers.log')

    def test_a_failed_worker_wait_stops_what_it_started(self, monkeypatch):
        """A constructor that raises never becomes a Cluster, so no one else can
        tear it down: the scheduler and srun processes it started are stopped on
        the way out rather than left running in the allocation."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1'),
                          srun_raises=printing.PybnfError('no workers'))
        started = []
        faked_setup = cluster.Cluster.setup_srun_cluster   # the recorder _patch_init installed

        def spy_setup(*args):
            procs = faked_setup(*args)
            started.extend(procs)
            return procs

        monkeypatch.setattr(cluster.Cluster, 'setup_srun_cluster', staticmethod(spy_setup))
        with pytest.raises(printing.PybnfError, match='no workers'):
            _build(self._cfg_srun())

        assert [p.terminated for p in started] == [True, True]

    def test_node_keys_are_ignored_with_a_warning(self, monkeypatch, caplog):
        """scheduler_node / worker_nodes name machines to log in to, which this
        launcher never does. They are ignored -- but loudly, since a user who set
        them is expecting them to decide something."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1 n2'))
        with caplog.at_level('WARNING'):
            _build(self._cfg_srun(scheduler_node='head', worker_nodes=['w1', 'w2']))

        assert rec.setup_calls == []                       # no dask ssh to those nodes
        assert rec.client_calls[0][1].get('scheduler_file')  # connected by file, not to head:8786
        assert any('ignored' in r.message for r in caplog.records)

    def test_no_logging_broadcast_to_srun_workers(self, monkeypatch):
        """Like every other remote path, the srun workers are not local processes
        sharing this run's log file; init_logging is broadcast only to workers a
        LocalCluster spawned here."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1'))
        _build(self._cfg_srun())
        assert rec.last_client.run_calls == []
        assert rec.reinit_logging_calls == [(('pf', False, 'INFO'), {})]
