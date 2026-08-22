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
  * ``cluster.Popen`` / ``cluster.time.sleep`` — **mock**: capture the command;
    stub the 10s sleep so the test is instant.
  * the worker-count sources — ``$SLURM_CPUS_ON_NODE``, ``cluster.DASK_CPU_COUNT``
    and ``cluster.cpu_count`` — **stubbed to three different numbers**, so a test
    pins which source PyBNF consulted and not merely a plausible count (#616).
  * ``cluster.Client`` / ``cluster.LocalCluster`` / ``cluster.init_logging`` /
    ``cluster.reinit_logging`` and ``Cluster.read_node_names`` /
    ``Cluster.setup_cluster`` — **fakes** recording their call args, so the
    ``__init__`` branch dispatch is asserted without a real dask cluster.

Deliberately *not* substituted (#619): one section asks the programs that are
actually installed whether they still exist and still take the options PyBNF
passes them. Substituting the outside world is right for everything else here,
but it is also why nothing here could notice #615 -- an outside program renamed
out from under PyBNF, every multi-machine run dead, and every test still green.
Those checks build their commands by calling the real builders, so no argv is
written down twice, and they skip where the program is not installed.

#393 note: these assert PyBNF's *own* command-string / branch logic, never
dask/distributed internals or a pinned dask version (the version-specific
``reinit_logging`` workaround is asserted *to be called*, not pinned), so they
remain a valid safety net across the dask-unpinning upgrade.
"""
import io
import json
import os
import re
import sys
import types

import pytest

from functools import lru_cache
from importlib.util import find_spec
from shutil import which
from subprocess import run, PIPE, STDOUT, TimeoutExpired, CalledProcessError

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
    """Stand-in for a launched Popen object: the bring-up paths poll it, and
    terminate then wait on it when they have to abandon a partly-built cluster."""
    def __init__(self, returncode=None):
        self._returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        if self._returncode is None:
            self._returncode = -15

    def wait(self, timeout=None):
        return self._returncode

    def kill(self):
        self.killed = True
        self._returncode = -9


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
# The programs PyBNF runs, against the programs that are actually installed
# (#619)
# --------------------------------------------------------------------------- #
# Every other test in this file substitutes the outside world. That is right --
# they are about PyBNF's own branch and command-building logic -- but it is also
# why none of them could notice #615, where distributed stopped installing
# ``dask-ssh``, every multi-machine run died on FileNotFoundError before a single
# simulation, and every test here kept passing: the name they compared against was
# a copy of the wrong name, written into this file.
#
# These checks close that gap by asking the installed programs themselves. Each
# one takes a command from the code that builds it for a real fit -- no argv is
# written down a second time here -- and asks the program named in it whether it
# runs, whether it still has that subcommand, and whether its ``--help`` still
# declares every option PyBNF passes. A renamed command or a renamed option fails
# here, loudly, while the mocked tests below go on pinning the argv PyBNF is
# supposed to build.


# What PyBNF prepends to every command it hands to dask (#615). Written out once,
# rather than imported from cluster.DASK_CLI, so that the mocked tests below pin
# the actual argv instead of agreeing with the module by construction -- and
# written out only *once*, because #619 is in part about a name that had to be
# corrected in seven places. The copy and the original are compared in
# test_the_invocation_these_tests_pin_is_the_modules_own, so the copy cannot drift
# away unnoticed either.
DASK = [sys.executable, '-m', 'dask']
DASK_SSH = [*DASK, 'ssh']


@lru_cache(maxsize=None)
def _help_text(argv):
    """Run a program's help screen and return what it printed.

    ``argv`` is a tuple so that it can be a cache key: several checks read the
    same screen, and each reading costs a process.
    """
    proc = run([*argv, '--help'], stdout=PIPE, stderr=STDOUT, timeout=120)
    output = proc.stdout.decode('UTF-8', errors='replace')
    assert proc.returncode == 0, ('`%s --help` failed (exit %s), so PyBNF cannot run it '
                                  'either:\n%s' % (' '.join(argv), proc.returncode, output))
    return output


_DECLARED_OPTION_RE = re.compile(r'^ {1,4}(-[^\s,]+(?:,\s+-[^\s,]+)*)')


def _declared_options(help_text):
    """The option names a click ``--help`` screen declares.

    Read from the option column alone -- a declaration begins within the first
    few columns of its line, while the description beside it wraps far deeper in.
    The distinction earns its keep: ``dask worker --help`` names ``--nworkers``
    inside the prose describing three *other* options, so a search of the whole
    screen would go on passing after the option itself was gone.
    """
    declared = set()
    for line in help_text.splitlines():
        match = _DECLARED_OPTION_RE.match(line)
        if match:
            declared.update(word.strip() for word in match.group(1).split(','))
    return declared


def _help_mentions(help_text, option):
    """Whether a help screen names an option anywhere, as a whole word.

    Weaker than :func:`_declared_options`, and used only for a program whose help
    is not laid out by click and whose layout cannot be checked from a developer
    machine. It still catches the failure that matters: an option that no longer
    exists, whose name has left the screen entirely.
    """
    return re.search(r'(?<![\w-])%s(?![\w-])' % re.escape(option), help_text) is not None


def _split_at_dask_cli(argv):
    """Split a command PyBNF builds into (what comes before dask, dask's own argv).

    The dask CLI is *located* in the argv rather than assumed to start it, since
    the srun launcher puts srun's own arguments in front. Failing to find it is an
    error, which makes this the check that every worker-launch command goes
    through ``cluster.DASK_CLI`` -- the one place that decides how dask is invoked
    -- rather than through a spelling of its own.
    """
    width = len(cluster.DASK_CLI)
    for i in range(len(argv) - width + 1):
        if argv[i:i + width] == cluster.DASK_CLI:
            return argv[:i], argv[i + width:]
    raise AssertionError('%r does not invoke the dask CLI (%r)' % (argv, cluster.DASK_CLI))


def _options_in(argv):
    """The options a command passes, deduplicated and in a stable order."""
    return sorted({a for a in argv if a.startswith('-')})


def _ssh_command(monkeypatch):
    """The argv ``setup_cluster`` hands to Popen, built by the real code path."""
    launched = []
    monkeypatch.setattr(cluster, 'Popen',
                        lambda cmd, **kwargs: launched.append(cmd) or _FakeDaskProc())
    monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '4')
    cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=None)
    return launched[0]


def _scheduler_command(monkeypatch):
    """The argv the srun launcher starts its scheduler with. Takes ``monkeypatch``
    it does not need, so that all three builders can be called the same way."""
    return cluster.Cluster.dask_scheduler_command('/shared/dask_scheduler.json')


def _worker_command(monkeypatch):
    """The argv the srun launcher starts its workers with -- srun's own arguments
    in front, then the dask CLI."""
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '4')
    return cluster.Cluster.srun_worker_command('/shared/dask_scheduler.json', 2)


# Every dask command PyBNF builds, keyed by the subcommand it runs and produced
# by calling the code that produces it for a real fit.
DASK_COMMAND_BUILDERS = {'ssh': _ssh_command,
                         'scheduler': _scheduler_command,
                         'worker': _worker_command}


class TestTheInstalledDaskIsTheOnePyBNFRuns:

    # dask itself is a hard dependency -- this module cannot even import without
    # it -- but its command line interface is a separate module that an unusually
    # trimmed install can lack, and PyBNF already treats that as its own error.
    pytestmark = pytest.mark.skipif(find_spec('dask.__main__') is None,
                                    reason='the installed dask has no command line interface')

    def test_the_dask_cli_pybnf_invokes_can_actually_be_run(self):
        """The plainest form of what #615 broke: the program does not exist. Asked
        of ``cluster.DASK_CLI`` itself, so what is proved runnable is the
        invocation PyBNF uses rather than a spelling of it kept here."""
        assert 'Usage' in _help_text(tuple(cluster.DASK_CLI))

    @pytest.mark.parametrize('subcommand', sorted(DASK_COMMAND_BUILDERS))
    def test_each_command_runs_a_dask_subcommand_that_still_exists(self, subcommand, monkeypatch):
        """Every command PyBNF builds goes through the dask CLI and names a
        subcommand the installed dask really has -- established by running it, not
        by reading an entry point (which is what ``check_dask_subcommand`` does,
        one step further from what happens at bring-up)."""
        _, dask_argv = _split_at_dask_cli(DASK_COMMAND_BUILDERS[subcommand](monkeypatch))
        assert dask_argv[0] == subcommand
        assert 'Usage' in _help_text((*cluster.DASK_CLI, subcommand))

    @pytest.mark.parametrize('subcommand', sorted(DASK_COMMAND_BUILDERS))
    def test_every_option_pybnf_passes_is_still_declared_by_dask(self, subcommand, monkeypatch):
        """The half of #619 the subcommand check cannot reach: a command that still
        exists but no longer takes the option PyBNF hands it. ``--nworkers`` is
        itself a survivor of that -- distributed renamed it from ``--nprocs`` --
        and the next rename would otherwise reach a user as a bring-up that fails,
        with dask's complaint in a log nobody is reading."""
        _, dask_argv = _split_at_dask_cli(DASK_COMMAND_BUILDERS[subcommand](monkeypatch))
        declared = _declared_options(_help_text((*cluster.DASK_CLI, subcommand)))
        missing = [opt for opt in _options_in(dask_argv) if opt not in declared]
        assert not missing, ('`dask %s` no longer declares %s, which PyBNF passes it'
                             % (subcommand, ', '.join(missing)))

    def test_dask_ssh_accepts_the_whole_command_pybnf_builds(self, monkeypatch):
        """Stronger than reading the help screen, because dask does the reading:
        the real command is handed to the real program with ``--help`` appended,
        which parses every option and its value and then stops before an SSH login
        is attempted. An unknown option is refused there with a non-zero exit.
        Only ``dask ssh`` can be asked this way -- ``scheduler`` and ``worker``
        forward unrecognized arguments to preload modules rather than refusing
        them, so for those two the help screen is the only witness."""
        assert 'Usage' in _help_text(tuple(_ssh_command(monkeypatch)))

    def test_the_invocation_these_tests_pin_is_the_modules_own(self):
        """The mocked tests below spell the invocation out, so that they pin the
        real argv rather than agreeing with the module by construction. This is
        what keeps that spelling honest: the copy and ``cluster.DASK_CLI`` are
        compared here, once, so a change made to one and not the other is a
        failure rather than a silent divergence."""
        assert DASK == cluster.DASK_CLI


class TestTheInstalledSlurmIsTheOnePyBNFRuns:
    """The same question asked of SLURM's own programs.

    Skipped wherever SLURM is not installed, which is every developer machine and
    every CI runner. That does not make it dead weight: PyBNF's tests are run on
    clusters, and a cluster is the one place where a renamed ``srun`` option can be
    caught before a fit walks into it. #619 is about the whole class of outside
    programs, not about dask alone.
    """

    # ``srun``'s presence is what says SLURM is installed here at all; the checks
    # below then hold the individual programs to what PyBNF names.
    pytestmark = pytest.mark.skipif(which('srun') is None, reason='SLURM is not installed here')

    def test_srun_still_takes_every_option_the_launcher_passes(self, monkeypatch):
        """The srun half of the worker-launch command, held to the srun that is
        installed. ``--cpus-per-task`` in particular is load-bearing rather than
        decorative -- without it a task confines every worker it forks to one CPU
        -- so a rename there would quietly serialize a node rather than fail."""
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '4')
        srun_argv, _ = _split_at_dask_cli(
            cluster.Cluster.srun_worker_command('/shared/dask_scheduler.json', 2))
        help_text = _help_text((srun_argv[0],))
        missing = [opt for opt in _options_in(srun_argv) if not _help_mentions(help_text, opt)]
        assert not missing, ('`%s` no longer mentions %s, which PyBNF passes it'
                             % (srun_argv[0], ', '.join(missing)))

    def test_srun_still_takes_every_option_the_group_builder_passes(self, monkeypatch):
        """The heterogeneous path (#617) hands srun a few more options than the single
        step does: ``--nodelist``, ``--nodes``, ``--ntasks`` and ``--ntasks-per-node``,
        which name exactly the machines one size group runs on. Held to the installed
        srun the same way, so a rename in any of them is caught on a cluster rather than
        in a fit."""
        srun_argv, _ = _split_at_dask_cli(
            cluster.Cluster.srun_worker_command_for_group(
                '/shared/dask_scheduler.json', ['n1', 'n2'], 40))
        help_text = _help_text((srun_argv[0],))
        missing = [opt for opt in _options_in(srun_argv) if not _help_mentions(help_text, opt)]
        assert not missing, ('`%s` no longer mentions %s, which PyBNF passes it'
                             % (srun_argv[0], ', '.join(missing)))

    def test_the_node_list_is_read_with_a_program_that_is_installed(self, monkeypatch):
        """``scontrol`` is the other SLURM program PyBNF runs, and the one whose
        absence would stop a cluster fit first: the node list is read before
        anything is launched. Its name is taken from the command PyBNF builds, so
        this fails if PyBNF starts naming something this SLURM does not ship."""
        named = []
        monkeypatch.setattr(cluster, 'run',
                            lambda cmd, **kwargs: named.append(cmd) or _FakeProc(b'n1\n'))
        cluster.Cluster.read_node_names(_cfg(cluster_type='slurm'))
        program = named[0][0]
        assert which(program) is not None, ('PyBNF reads the node list with %r, which is not '
                                            'installed on this SLURM system' % program)


# --------------------------------------------------------------------------- #
# setup_cluster — the dask ssh command string + per-node arithmetic
# --------------------------------------------------------------------------- #
class TestSetupCluster:

    def _patch(self, monkeypatch, granted=None, affinity=4, cpu=64,
               returncode=None, output_bytes=b''):
        """Patch what setup_cluster touches: Popen (capture the command),
        time.sleep (don't actually wait 10s), and every source the default worker
        count can come from, so the count is deterministic *and* it is visible which
        source produced it -- ``$SLURM_CPUS_ON_NODE`` (what the job was granted;
        removed from the environment unless ``granted`` is passed), the
        affinity/cgroup count dask derives, and the whole machine's ``cpu_count()``.
        The three defaults are deliberately three different numbers.
        The fake proc's ``poll()`` returns ``returncode`` (None = still running,
        the healthy default); if ``output_bytes`` is given the fake writes it to
        the capture file setup_cluster handed to Popen, so the early-exit error
        path can read it back. It is written to the **stdout** handle because that
        is the stream dask explains an SSH failure on (#618), and because stderr is
        merged into it. Returns the recorder list of Popen (args, kwargs)."""
        popen_calls = []

        def fake_popen(*args, **kwargs):
            popen_calls.append((args, kwargs))
            if output_bytes:
                kwargs['stdout'].write(output_bytes)
            return _FakeDaskProc(returncode)

        monkeypatch.setattr(cluster, 'Popen', fake_popen)
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        monkeypatch.delenv('SLURM_CPUS_ON_NODE', raising=False)
        if granted is not None:
            monkeypatch.setenv('SLURM_CPUS_ON_NODE', str(granted))
        monkeypatch.setattr(cluster, 'DASK_CPU_COUNT', affinity)
        monkeypatch.setattr(cluster, 'cpu_count', lambda: cpu)
        return popen_calls

    @staticmethod
    def _bringup_then_wait(node_string='node9', out_dir='/log', parallel_count=1):
        """Launch dask ssh and run the readiness wait that watches it (#398).

        setup_cluster no longer waits or reports a failure by itself; it launches dask ssh
        and hands back the process, the worker count, and the captured-output file. The
        readiness wait is what then watches the process and reports a bring-up that has
        already died, reading the output setup_cluster captured. These tests drive the two
        together, with a client reporting no workers so the only thing that can happen is the
        dead-process branch."""
        proc, expected, out_file = cluster.Cluster.setup_cluster(node_string, out_dir,
                                                                 parallel_count=parallel_count)
        return cluster.Cluster.wait_for_ssh_workers(_ClientStub(workers=()), proc, expected,
                                                    out_file, out_dir)

    def test_default_worker_count_is_what_the_job_was_granted(self, monkeypatch):
        """parallel_count=None ⇒ one single-threaded worker per CPU **the job holds
        on a node**: ``--nthreads 1 --nworkers {cpus_per_node()}`` (note this
        branch's flag order is --nthreads then --nworkers). Oracle: the exact
        argument list (ROB-3: an argv list launched with no shell, each node its own
        entry) with SLURM having granted 7, over an affinity count of 4 and a
        machine of 64."""
        popen_calls = self._patch(monkeypatch, granted=7, affinity=4, cpu=64)
        proc, expected, out_file = cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=None)

        assert proc.poll() is None
        assert expected == 14           # 7 workers on each of 2 nodes
        (args, kwargs), = popen_calls
        assert args[0] == [*DASK_SSH, 'n1', 'n2',
                           '--log-directory', '/out', '--nthreads', '1', '--nworkers', '7']
        assert kwargs.get('shell', False) is False         # no shell -> no injection
        # Both streams are captured to one readable file (nothing is discarded), so an
        # early bring-up failure can be surfaced — see test_failed_bringup_*.
        assert hasattr(kwargs['stdout'], 'read')
        assert kwargs['stderr'] is cluster.STDOUT

    def test_default_worker_count_is_not_the_size_of_the_machine(self, monkeypatch):
        """#616, stated as the reported case: a job granted 4 CPUs of a
        128-processor node must start 4 workers per node, not 128. This is the
        assertion the old code failed -- it passed ``multiprocessing.cpu_count()``,
        which reports every processor the machine has whatever the scheduler
        granted, so the pool overshot the job by 32x. The two numbers agree only
        when whole nodes were allocated, which is why the defect could hide."""
        popen_calls = self._patch(monkeypatch, granted=4, affinity=4, cpu=128)
        cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=None)

        (args, _), = popen_calls
        assert args[0][args[0].index('--nworkers') + 1] == '4'
        assert '128' not in args[0]

    def test_default_worker_count_falls_back_to_affinity_not_the_machine(self, monkeypatch):
        """With no scheduler count published, the next-best number is what the
        operating system will let this process run on -- CPU affinity narrowed by
        any cgroup quota, which is what dask derives and what a local PyBNF run
        already sizes itself by -- rather than the whole machine."""
        popen_calls = self._patch(monkeypatch, granted=None, affinity=6, cpu=64)
        cluster.Cluster.setup_cluster('n1', '/out', parallel_count=None)

        (args, _), = popen_calls
        assert args[0][args[0].index('--nworkers') + 1] == '6'

    def test_default_worker_count_and_its_source_are_logged(self, monkeypatch, caplog):
        """A user who sees an unexpected number of workers has to be able to find
        out which number PyBNF believed and where it read it, so the count, the node
        count and the source are all logged (#616)."""
        self._patch(monkeypatch, granted=7)
        with caplog.at_level('INFO', logger='pybnf.cluster'):
            cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=None)

        line, = [r.message for r in caplog.records if 'worker process' in r.message]
        assert '7' in line and '2 node' in line
        assert 'SLURM_CPUS_ON_NODE' in line

    def test_explicit_parallel_count_is_logged_as_its_own_source(self, monkeypatch, caplog):
        """When parallel_count decides the count, the log says so: the number did
        not come from the job's CPUs, and a user comparing the two needs to know
        which one is in force."""
        self._patch(monkeypatch, granted=7)
        with caplog.at_level('INFO', logger='pybnf.cluster'):
            cluster.Cluster.setup_cluster('n1 n2', '/out', parallel_count=6)

        line, = [r.message for r in caplog.records if 'worker process' in r.message]
        assert 'parallel_count' in line
        assert 'SLURM_CPUS_ON_NODE' not in line

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
        assert args[0][:4] == DASK_SSH
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

    def test_setup_returns_the_proc_expected_count_and_output_file(self, monkeypatch):
        """setup_cluster launches dask ssh and hands back what the readiness wait needs
        (#398): the running process, the number of workers it should bring up (one per node
        times the per-node count), and the open file its output was captured to. It no longer
        waits or decides success itself."""
        self._patch(monkeypatch)
        proc, expected, out_file = cluster.Cluster.setup_cluster('n1 n2', '/log', parallel_count=6)
        assert proc.poll() is None
        assert expected == 6            # ceil(6/2)=3 per node, over 2 nodes
        assert hasattr(out_file, 'read')

    def test_failed_bringup_raises_with_the_captured_output(self, monkeypatch):
        """If dask ssh has already exited after the startup wait, the cluster
        never came up. setup_cluster must raise PybnfError (not return a dead
        proc that later surfaces as an opaque Client connection error), and
        everything dask ssh said is included for diagnosis."""
        self._patch(monkeypatch, returncode=1,
                    output_bytes=b'ssh: connect to host node9 port 22: Connection refused')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        msg = str(exc.value)
        assert 'code 1' in msg
        assert 'Connection refused' in msg

    def test_bringup_captures_the_stream_dask_explains_itself_on(self, monkeypatch):
        """#618: dask's own account of a failed login -- the node it was connecting
        to and the exception paramiko raised -- is ``print``ed, i.e. written to
        **stdout**; only the traceback falls to stderr. Sending stdout to DEVNULL
        discarded the half of the output that names the cause, which is how a
        refused login reached the user as a bare exit code. Oracle: what dask writes
        to stdout comes back in the message."""
        self._patch(monkeypatch, returncode=1,
                    output_bytes=b'[ dask ssh ] : SSH connection error when connecting to '
                                 b'node9:22\n               SSH reported this exception: '
                                 b'Authentication failed.\n')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        assert 'SSH reported this exception: Authentication failed.' in str(exc.value)

    def test_dask_is_run_unbuffered_so_its_own_account_survives(self, monkeypatch):
        """#618, and the reason capturing stdout is worth anything: dask ends a failed
        bring-up with ``os._exit(1)``, which does not flush Python's buffers. Its
        stdout, writing to a file, is block-buffered, and its few hundred bytes of
        explanation never reach the 8 KB that would force a write -- so the whole of
        it is discarded at exit. Measured against dask 2026.7.1 on a login that
        fails: 0 of dask's own lines survive without ``PYTHONUNBUFFERED``, all 15
        with it. The rest of the environment is passed through, since the workers dask
        starts inherit it (PATH, a loaded module's variables, BNGPATH)."""
        monkeypatch.setenv('BNGPATH', '/opt/bng')
        popen_calls = self._patch(monkeypatch)
        cluster.Cluster.setup_cluster('n1', '/log', parallel_count=1)

        (_, kwargs), = popen_calls
        assert kwargs['env']['PYTHONUNBUFFERED'] == '1'
        assert kwargs['env']['BNGPATH'] == '/opt/bng'

    def test_traceback_frames_are_folded_out_of_the_message(self, monkeypatch):
        """What dask ssh writes on a failed login is mostly traceback -- one per node
        per retry, a dozen frames of dask's and paramiko's own source each -- and the
        sentences that say what happened are buried in it, or pushed out of the
        message by them. The frames come out of the message; the exception line each
        traceback ends with, and dask's own lines, stay."""
        self._patch(monkeypatch, returncode=1, output_bytes=(
            b'[ dask ssh ] : SSH connection error when connecting to node9:22\n'
            b'Traceback (most recent call last):\n'
            b'  File "/x/distributed/deploy/old_ssh.py", line 47, in async_ssh\n'
            b'    ssh.connect(\n'
            b'    ^^^^^^^^^^^^\n'
            b'paramiko.ssh_exception.AuthenticationException: Authentication failed.\n'
            b'               Retrying... (attempt 1/3)\n'))
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        message = exc.value.message
        assert 'SSH connection error when connecting to node9:22' in message
        assert 'AuthenticationException: Authentication failed.' in message
        assert 'Retrying... (attempt 1/3)' in message      # indented, but not a frame
        assert 'old_ssh.py' not in message                 # the frames themselves
        assert 'Traceback' not in message

    def test_the_log_keeps_the_frames_the_message_folds(self, monkeypatch, caplog):
        """The folding is a choice about the *message*, which a user reads once and
        has to act on. Nothing is lost: the log keeps the output as it was written,
        for whoever ends up reading the traceback."""
        self._patch(monkeypatch, returncode=1, output_bytes=(
            b'Traceback (most recent call last):\n'
            b'  File "/x/distributed/deploy/old_ssh.py", line 47, in async_ssh\n'
            b'paramiko.ssh_exception.AuthenticationException: Authentication failed.\n'))
        with caplog.at_level('ERROR', logger='pybnf.cluster'):
            with pytest.raises(printing.PybnfError):
                self._bringup_then_wait('node9', '/log')
        line, = [r.message for r in caplog.records if 'dask ssh exited' in r.message]
        assert 'old_ssh.py' in line

    def test_a_login_failure_is_named_as_one(self, monkeypatch):
        """#618: the reported case was a failed login, and nothing in the message
        said so. When the output carries the vocabulary of a refused credential, the
        message says the login is the likely cause and says what PyBNF logs in with
        -- a library that can offer only a public key or a password -- since that is
        what makes the failure survive ``ssh-keygen``, and makes plain ``ssh``
        succeeding from the same shell no evidence at all."""
        self._patch(monkeypatch, returncode=1,
                    output_bytes=b'paramiko.ssh_exception.AuthenticationException: '
                                 b'Authentication failed.')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        message = exc.value.message
        assert 'login' in message
        assert 'paramiko' in message
        assert 'public key' in message and 'password' in message
        assert 'host-based' in message and 'GSSAPI' in message

    def test_a_network_failure_is_not_blamed_on_the_login(self, monkeypatch):
        """The converse, and the reason the test above is not satisfied by saying
        "login" every time: a machine that could not be reached at all is a
        different problem, and answering it with advice about keys and passwords
        would send the user off to fix something that is not broken."""
        self._patch(monkeypatch, returncode=1,
                    output_bytes=b'paramiko.ssh_exception.NoValidConnectionsError: '
                                 b'[Errno None] Unable to connect to port 22 on 10.0.0.9')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        message = exc.value.message
        assert 'failed login' not in message                    # no diagnosis is offered
        assert 'public key' not in message
        assert 'Unable to connect to port 22' in message        # ... but the output is quoted

    def test_failure_names_both_ways_of_running_without_a_login(self, monkeypatch):
        """#618: the failure ends the run, so the message is the whole of what the
        user gets, and both ways of using several machines that never log in
        anywhere are named -- srun inside the allocation (#614) and a scheduler file
        naming a cluster that is already up. Named whatever the cause, since
        anything that stops dask ssh leaves them as the ways forward: this is the
        no-login-vocabulary case."""
        self._patch(monkeypatch, returncode=1, output_bytes=b'exit status 127')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        message = exc.value.message
        assert 'slurm-srun' in message
        assert 'scheduler_file' in message

    def test_output_is_reported_even_when_there_is_none(self, monkeypatch):
        """#618: the old message quoted the captured output only when it happened to
        be non-empty, and otherwise said "Check the cluster log directory" without
        naming a directory -- so a user could not tell a silent failure from one
        whose explanation had been thrown away. Silence is now reported as such, and
        the directory the nodes write to is named."""
        self._patch(monkeypatch, returncode=1, output_bytes=b'')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/logdir')
        message = exc.value.message
        assert 'no output' in message
        assert '/logdir' in message

    def test_captured_output_is_logged_as_well_as_raised(self, monkeypatch, caplog):
        """The message goes to a user who may not have kept the terminal; the log is
        the copy that survives. Both carry what dask ssh said."""
        self._patch(monkeypatch, returncode=1, output_bytes=b'Authentication failed.')
        with caplog.at_level('ERROR', logger='pybnf.cluster'):
            with pytest.raises(printing.PybnfError):
                self._bringup_then_wait('node9', '/log')
        line, = [r.message for r in caplog.records if 'dask ssh exited' in r.message]
        assert 'Authentication failed.' in line

    def test_colour_codes_are_stripped_from_the_quoted_output(self, monkeypatch):
        """dask wraps its failure lines in terminal colour escapes. Quoted as they
        are, they reach the log file and the message as literal characters."""
        self._patch(monkeypatch, returncode=1,
                    output_bytes=b'\x1b[91mSSH connection failed after 3 retries.\x1b[0m')
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        assert 'SSH connection failed after 3 retries.' in str(exc.value)
        assert '\x1b' not in exc.value.message

    def test_only_the_tail_of_a_long_output_is_quoted(self, monkeypatch):
        """One failure per node, each retried three times, would otherwise put
        hundreds of lines in front of the advice at the end of the message."""
        self._patch(monkeypatch, returncode=1,
                    output_bytes=b'\n'.join(b'line %i' % i for i in range(200)))
        with pytest.raises(printing.PybnfError) as exc:
            self._bringup_then_wait('node9', '/log')
        quoted = str(exc.value)
        assert 'line 199' in quoted
        assert 'line 0\n' not in quoted

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

# --------------------------------------------------------------------------- #
# __init__ — node-detection dispatch + Client-construction dispatch
# --------------------------------------------------------------------------- #
class _ProcStub:
    """Stand-in for a process PyBNF started and later terminates. Teardown asks it to
    stop and then waits on it, so it records both and reports itself as exited."""
    def __init__(self, returncode=None):
        self.terminated = False
        self.killed = False
        self._returncode = returncode

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        if self._returncode is None:
            self._returncode = -15

    def wait(self, timeout=None):
        return self._returncode

    def kill(self):
        self.killed = True
        self._returncode = -9


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
        self.setup_proc = None       # the dask ssh process fake_setup handed back
        self.ssh_wait_calls = []     # (client, dask_proc, expected, output_file, out_dir)
        self.srun_setup_calls = []   # (scheduler_file, out_dir, node_names, parallel_count)
        self.srun_worker_procs = None  # the srun worker procs fake_setup_srun handed back
        self.srun_wait_calls = []    # (client, worker_procs, expected, worker_logs)

    def Client(self, *args, **kwargs):
        self.client_calls.append((args, kwargs))
        self.last_client = _ClientStub()
        return self.last_client

    def LocalCluster(self, *args, **kwargs):
        self.lc_calls.append((args, kwargs))
        self.last_lc = object()
        return self.last_lc


def _patch_init(monkeypatch, read_returns=(None, None), srun_raises=None, ssh_raises=None):
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
        # setup_cluster returns the process, the worker count it should bring up, and the
        # file its output was captured to (#398). The count and file are placeholders here;
        # wait_for_ssh_workers, which consumes them, is faked below.
        rec.setup_proc = _ProcStub()
        return rec.setup_proc, 2, None

    def fake_setup_srun(scheduler_file, out_dir, node_names, parallel_count):
        rec.srun_setup_calls.append((scheduler_file, out_dir, node_names, parallel_count))
        # scheduler proc, the list of srun worker procs, the total worker count to wait for,
        # and the log each srun step writes. wait_for_srun_workers, which consumes them, is
        # faked below, so the count and logs here are placeholders.
        rec.srun_worker_procs = [_ProcStub()]
        return _ProcStub(), rec.srun_worker_procs, 1, [os.path.join(out_dir, 'dask_workers.log')]

    def fake_ssh_wait(client, dask_proc, expected, output_file, out_dir, **kwargs):
        rec.ssh_wait_calls.append((client, dask_proc, expected, output_file, out_dir))
        if ssh_raises:
            raise ssh_raises
        return expected

    def fake_wait(client, worker_procs, expected, worker_logs, **kwargs):
        rec.srun_wait_calls.append((client, worker_procs, expected, worker_logs))
        if srun_raises:
            raise srun_raises
        return expected

    monkeypatch.setattr(cluster.Cluster, 'read_node_names', staticmethod(fake_read))
    monkeypatch.setattr(cluster.Cluster, 'setup_cluster', staticmethod(fake_setup))
    monkeypatch.setattr(cluster.Cluster, 'wait_for_ssh_workers', staticmethod(fake_ssh_wait))
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
        assert c._dask_proc is rec.setup_proc
        assert rec.client_calls == [(('head:8786',), {})]
        assert c.local is False
        # Having started dask ssh itself, PyBNF waits for those workers to register (#398).
        assert len(rec.ssh_wait_calls) == 1
        assert rec.ssh_wait_calls[0][1] is rec.setup_proc

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
        assert c._dask_proc is rec.setup_proc
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
        assert c._dask_proc is rec.setup_proc
        assert rec.client_calls == [(('sched9:8786',), {})]
        assert c.local is False

    def test_a_failed_worker_wait_stops_the_dask_ssh_it_started(self, monkeypatch):
        """If the workers never register, the readiness wait raises and the constructor
        never becomes a Cluster, so no one else can tear it down. The dask ssh process it
        started is stopped on the way out rather than left running (#398)."""
        rec = _patch_init(monkeypatch, read_returns=('sched9', 'sched9 c1 c2'),
                          ssh_raises=printing.PybnfError('workers never came up'))
        with pytest.raises(printing.PybnfError, match='workers never came up'):
            _build(_cfg(parallel_count=4))

        assert len(rec.ssh_wait_calls) == 1
        assert rec.setup_proc.terminated is True


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
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '7')
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
    c._ssh_output_file = None
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
    """The one place either launcher decides how many workers a node gets (#616).

    The four sources are given four different numbers throughout, so each test
    pins *which* one was consulted rather than merely a plausible count.
    """

    def _sources(self, monkeypatch, granted=None, allocation=None, affinity=6, cpu=64):
        monkeypatch.delenv('SLURM_CPUS_ON_NODE', raising=False)
        monkeypatch.delenv('SLURM_JOB_CPUS_PER_NODE', raising=False)
        if granted is not None:
            monkeypatch.setenv('SLURM_CPUS_ON_NODE', str(granted))
        if allocation is not None:
            monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', str(allocation))
        monkeypatch.setattr(cluster, 'DASK_CPU_COUNT', affinity)
        monkeypatch.setattr(cluster, 'cpu_count', lambda: cpu)

    def test_reads_what_slurm_granted(self, monkeypatch):
        """$SLURM_CPUS_ON_NODE is what the *allocation* granted. It is preferred
        over both local numbers because it describes the allocation rather than the
        process asking, so it is still right for a worker started on another
        machine -- and because it is the number the srun launcher can actually ask
        SLURM for."""
        self._sources(monkeypatch, granted=12, affinity=6, cpu=64)
        count, source = cluster.Cluster.cpus_per_node()
        assert count == 12
        assert 'SLURM_CPUS_ON_NODE' in source

    @pytest.mark.parametrize('value', [None, '', 'many', '0', '-4'])
    def test_an_unusable_slurm_value_falls_through_to_the_affinity_count(self, monkeypatch, value):
        """With no usable scheduler count, the next-best number is what the OS will
        let this process run on -- CPU affinity narrowed by any cgroup quota, the
        number dask derives and a local PyBNF run already uses -- not the machine."""
        self._sources(monkeypatch, granted=value, affinity=6, cpu=64)
        count, source = cluster.Cluster.cpus_per_node()
        assert count == 6
        assert 'affinity' in source

    def test_the_whole_machine_is_the_last_resort(self, monkeypatch):
        """The machine's own processor count is right only when nothing is limiting
        the job at all, so it is reached only when neither better number exists."""
        self._sources(monkeypatch, granted=None, affinity=0, cpu=64)
        count, source = cluster.Cluster.cpus_per_node()
        assert count == 64
        assert 'machine' in source

    def test_a_granted_count_never_reports_the_machine_count(self, monkeypatch):
        """The #616 oracle, stated once for both launchers: a job granted a small
        share of a large node is sized by the share. The old SSH-launcher code
        returned 128 here, oversubscribing the job 32-fold."""
        self._sources(monkeypatch, granted=4, affinity=4, cpu=128)
        assert cluster.Cluster.cpus_per_node()[0] == 4

    def test_the_jobs_own_per_node_list_is_read_when_the_step_variable_is_empty(self, monkeypatch):
        """#642: SLURM sets $SLURM_CPUS_ON_NODE only inside a step running on an
        allocated node, so it is empty in the shell ``salloc`` opens on a login node --
        while $SLURM_JOB_CPUS_PER_NODE, which answers the same question for the job as
        a whole, is set correctly there. Reading it keeps the count describing the
        allocation (20) instead of falling through to the machine-level numbers, which
        describe a login node that is not in the allocation at all (128)."""
        self._sources(monkeypatch, granted=None, allocation='20', affinity=128, cpu=96)
        count, source = cluster.Cluster.cpus_per_node()
        assert count == 20
        assert 'SLURM_JOB_CPUS_PER_NODE' in source

    def test_a_mixed_allocation_reduces_to_its_smallest_machine(self, monkeypatch):
        """One number has to serve every machine here: it sizes a pool started on all of
        them, and an srun step asks SLURM for it on all of them. Asking for fewer CPUs
        than a machine holds costs speed, while asking for more than the smallest machine
        holds is refused outright, so the safe direction is the smallest grant. (Sizing
        each machine on its own is per_node_cpus, which the default srun path uses.)"""
        self._sources(monkeypatch, granted=None, allocation='40(x2),96', affinity=128, cpu=96)
        assert cluster.Cluster.cpus_per_node()[0] == 40

    def test_the_per_step_count_still_wins_when_both_are_set(self, monkeypatch):
        """Inside the allocation both are set, and the per-step count is the one that
        describes *this* node. The job-level list is a fallback for a launching process
        that is not on an allocated node, and does not change what a launch from inside
        the allocation reads."""
        self._sources(monkeypatch, granted=12, allocation='40(x2),96', affinity=6, cpu=64)
        count, source = cluster.Cluster.cpus_per_node()
        assert count == 12
        assert 'SLURM_CPUS_ON_NODE' in source

    @pytest.mark.parametrize('value', ['', 'nonsense', '0'])
    def test_an_unusable_per_node_list_falls_through_to_the_local_numbers(self, monkeypatch, value):
        """A list this code cannot read, or one that says no CPUs at all, is treated as
        absent rather than guessed at -- the local numbers are still better than nothing."""
        self._sources(monkeypatch, granted=None, allocation=value, affinity=6, cpu=64)
        count, source = cluster.Cluster.cpus_per_node()
        assert count == 6
        assert 'affinity' in source


class TestExpandCpusPerNode:
    """Reading $SLURM_JOB_CPUS_PER_NODE into one count per machine (#617). SLURM
    writes the counts run-length encoded, in the same order as the node list."""

    @pytest.mark.parametrize('spec, expected', [
        ('40(x2),96', [40, 40, 96]),      # the mixed-size case this issue is about
        ('8', [8]),                       # a single machine
        ('8(x3)', [8, 8, 8]),             # one size, several machines
        ('4(x2),8,16(x2)', [4, 4, 8, 16, 16]),
        (' 40(x2) , 96 ', [40, 40, 96]),  # whitespace around the groups is tolerated
    ])
    def test_expands_the_run_length_form(self, spec, expected):
        assert cluster.expand_cpus_per_node(spec) == expected

    @pytest.mark.parametrize('spec', [None, '', '   ', 'many', '4(x)', '(x2)', '4,', 'a,b'])
    def test_unusable_text_is_none(self, spec):
        """Anything that is not the run-length form -- unset, empty, or a value this
        code does not recognize -- returns None, so the caller falls back to sizing
        every machine the same rather than acting on a misread."""
        assert cluster.expand_cpus_per_node(spec) is None


class TestPerNodeCpus:
    """How many CPUs each machine was granted, one number per node (#617)."""

    def test_reads_the_per_machine_counts_in_node_order(self, monkeypatch):
        """The preferred source is $SLURM_JOB_CPUS_PER_NODE, which lines up with the
        node list; each machine is sized by what it was granted."""
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '40(x2),96')
        counts, source = cluster.Cluster.per_node_cpus(['n1', 'n2', 'n3'])
        assert counts == [40, 40, 96]
        assert 'SLURM_JOB_CPUS_PER_NODE' in source

    def test_unset_variable_sizes_every_machine_the_same(self, monkeypatch):
        """With the per-machine variable unset, there is nothing to size each machine
        by, so every machine gets the single cpus_per_node count -- the behaviour
        before this change. Here that count comes from $SLURM_CPUS_ON_NODE."""
        monkeypatch.delenv('SLURM_JOB_CPUS_PER_NODE', raising=False)
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '8')
        counts, source = cluster.Cluster.per_node_cpus(['n1', 'n2', 'n3'])
        assert counts == [8, 8, 8]
        assert 'SLURM_CPUS_ON_NODE' in source            # the fallback source, named

    def test_a_length_mismatch_falls_back_rather_than_misaligning(self, monkeypatch):
        """A per-machine list that does not have one entry per node cannot be trusted
        to line up, so it is not used: every machine is sized the same instead."""
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '40,96')   # two, but three nodes
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '8')
        counts, _ = cluster.Cluster.per_node_cpus(['n1', 'n2', 'n3'])
        assert counts == [8, 8, 8]

    def test_unparseable_variable_falls_back(self, monkeypatch):
        """A value this code cannot read is treated as unavailable, not guessed at."""
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', 'nonsense')
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '8')
        counts, _ = cluster.Cluster.per_node_cpus(['n1', 'n2'])
        assert counts == [8, 8]

    def test_the_fallback_off_an_allocated_node_still_reads_the_allocation(self, monkeypatch):
        """The two fixes compose. A list that cannot be lined up with the machines is not
        used to size them individually, but its values still describe machines in *this*
        job, so the single count they fall back to is the smallest of them -- not the size
        of the login node the launcher happens to be running on (#642)."""
        monkeypatch.delenv('SLURM_CPUS_ON_NODE', raising=False)      # not on an allocated node
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '20,40')       # two, but three nodes
        monkeypatch.setattr(cluster, 'DASK_CPU_COUNT', 128)
        counts, source = cluster.Cluster.per_node_cpus(['n1', 'n2', 'n3'])
        assert counts == [20, 20, 20]
        assert 'SLURM_JOB_CPUS_PER_NODE' in source

    def test_the_fallback_is_logged_so_a_mixed_cluster_user_is_told(self, monkeypatch, caplog):
        """A user on a mixed cluster is expecting each machine to be sized on its own,
        so falling back to one size for all is worth a warning that says why."""
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '40,96')
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '8')
        with caplog.at_level('WARNING'):
            cluster.Cluster.per_node_cpus(['n1', 'n2', 'n3'])
        assert any('SLURM_JOB_CPUS_PER_NODE' in r.message for r in caplog.records)


class TestSrunWorkerCommandForGroup:
    """The srun command for one group of same-sized machines (#617). One worker per
    granted CPU on each machine, the machines named so concurrent groups stay
    disjoint, and the CPU request tracking the worker count for the same cgroup
    reason srun_worker_command has."""

    def test_names_the_machines_and_sizes_them_by_their_grant(self):
        cmd = cluster.Cluster.srun_worker_command_for_group('/s.json', ['n1', 'n2'], 40)
        assert cmd == ['srun', '--nodelist', 'n1,n2',
                       '--nodes', '2', '--ntasks', '2', '--ntasks-per-node', '1',
                       '--cpus-per-task', '40', '--label',
                       *DASK, 'worker', '--scheduler-file', '/s.json',
                       '--nworkers', '40', '--nthreads', '1']

    def test_a_single_machine_group(self):
        cmd = cluster.Cluster.srun_worker_command_for_group('/s.json', ['big'], 96)
        assert cmd[cmd.index('--nodelist') + 1] == 'big'
        assert cmd[cmd.index('--nodes') + 1] == '1'
        assert cmd[cmd.index('--nworkers') + 1] == '96'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '96'

    def test_workers_are_single_threaded(self):
        cmd = cluster.Cluster.srun_worker_command_for_group('/s.json', ['n1'], 4)
        assert cmd[cmd.index('--nthreads') + 1] == '1'

    def test_scheduler_file_is_one_literal_argument(self):
        cmd = cluster.Cluster.srun_worker_command_for_group('/tmp/a b$(whoami).json', ['n1'], 2)
        assert cmd[cmd.index('--scheduler-file') + 1] == '/tmp/a b$(whoami).json'


class TestSrunWorkerCommand:

    def _patch(self, monkeypatch, granted=8):
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', str(granted))
        monkeypatch.delenv('SLURM_JOB_CPUS_PER_NODE', raising=False)

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
                       *DASK, 'worker',
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

    def test_a_caller_supplied_count_is_used_instead_of_the_environment(self, monkeypatch, caplog):
        """#642: a caller that already knows what the allocation granted passes it in, and
        that number is what sizes the pool and what SLURM is asked for -- the environment
        is not consulted at all. srun_worker_layout is that caller: it reads the
        allocation's own per-node list, which is right wherever this process is running,
        before it decides which command to build. The phrase naming where the count came
        from is the caller's too, since the log is how an unexpected count is traced."""
        self._patch(monkeypatch, granted=128)          # must not be consulted
        with caplog.at_level('INFO'):
            cmd = cluster.Cluster.srun_worker_command(
                '/s.json', 2, None, granted=20,
                source='what SLURM granted each machine ($SLURM_JOB_CPUS_PER_NODE)')
        assert cmd[cmd.index('--nworkers') + 1] == '20'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '20'
        assert any('SLURM_JOB_CPUS_PER_NODE' in r.message for r in caplog.records)

    def test_a_caller_supplied_count_still_caps_an_oversubscribed_request(self, monkeypatch):
        """The cap is the same rule as before, measured against the caller's count: the
        workers a user explicitly asked for all start, while the CPU request stays inside
        what the job holds so SLURM does not refuse the step."""
        self._patch(monkeypatch, granted=128)          # must not be consulted
        cmd = cluster.Cluster.srun_worker_command('/s.json', 1, parallel_count=64, granted=20)
        assert cmd[cmd.index('--nworkers') + 1] == '64'
        assert cmd[cmd.index('--cpus-per-task') + 1] == '20'


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

    def test_returns_once_all_the_workers_register(self, monkeypatch):
        """The readiness signal for the workers is all of them registering with the
        scheduler -- not srun having been launched, which says nothing. Two expected,
        two connected, so the wait returns two."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        n = cluster.Cluster.wait_for_srun_workers(
            _ClientStub(workers=('tcp://n1:1', 'tcp://n2:1')), [_FakeDaskProc()],
            expected=2, worker_logs=['/log'])
        assert n == 2

    def test_waits_while_the_cluster_is_still_filling_up(self, monkeypatch):
        """A scheduler short of the expected count is not ready and not an error: the
        poll continues until the last worker arrives (#200, #617). Here one of two is
        up, then the second arrives, and only then does the wait return."""
        client = _ClientStub(workers=('tcp://n1:1',))
        polls = []

        def fake_sleep(_seconds):
            polls.append(1)
            if len(polls) == 3:
                client._workers = dict.fromkeys(('tcp://n1:1', 'tcp://n2:1'), {})

        monkeypatch.setattr(cluster.time, 'sleep', fake_sleep)
        assert cluster.Cluster.wait_for_srun_workers(
            client, [_FakeDaskProc()], expected=2, worker_logs=['/log']) == 2
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
        assert cluster.Cluster.wait_for_srun_workers(
            _FlakyClient(), [_FakeDaskProc()], expected=1, worker_logs=['/log']) == 1

    def test_any_srun_step_exiting_early_is_reported_with_its_log(self, monkeypatch, tmp_path):
        """An srun step exiting before its workers registered means the placement
        failed -- a bad flag, a request larger than that part of the allocation. With
        machines of different sizes there is one step per size, so any of them exiting
        is caught, reported at once and quoting srun's own message."""
        log = tmp_path / 'workers.log'
        log.write_text('srun: error: Unable to allocate resources')
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_srun_workers(
                _ClientStub(workers=()), [_FakeDaskProc(returncode=1)],
                expected=1, worker_logs=[str(log)])
        assert 'code 1' in str(exc.value)
        assert 'Unable to allocate resources' in str(exc.value)

    def test_too_few_in_time_names_the_logs_and_the_step_hazard(self, monkeypatch, tmp_path):
        """srun still running with the workers short of the count is the shape of a
        queued job step: connecting to our own scheduler succeeded, so nothing else
        would report it, and the fit would run on fewer machines than reserved. The
        message points at srun's own log(s) and at the likely cause."""
        log = tmp_path / 'workers.log'
        log.write_text('srun: Job step creation temporarily disabled, retrying')
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_srun_workers(
                _ClientStub(workers=()), [_FakeDaskProc()], expected=2,
                worker_logs=[str(log)], timeout=1.)
        assert 'srun: Job step creation' in str(exc.value)   # srun's own words, in the log
        assert 'workers.log' in exc.value.message            # ... and where to read more
        assert 'job step' in exc.value.message.lower()


class TestPollForWorkers:
    """The readiness loop both launchers share (#398). The srun and SSH waits are thin wrappers
    that add their own worker count and their own failure vocabulary on top of this; the loop
    itself is what a real srun run exercises, so pin its three outcomes here. It reports the
    outcome rather than raising, leaving each launcher to phrase the error its own way."""

    def test_ready_when_the_expected_count_is_reached(self, monkeypatch):
        """Enough workers registered, process still running: 'ready', with the count."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        outcome, n, rc = cluster.Cluster._poll_for_workers(
            _ClientStub(workers=('tcp://n1:1', 'tcp://n2:1')), _FakeDaskProc(),
            expected=2, timeout=5., poll=0.25)
        assert (outcome, n, rc) == ('ready', 2, None)

    def test_exited_when_the_process_is_gone(self, monkeypatch):
        """The bring-up process exited before the workers arrived: 'exited', with its code,
        so the caller can quote whatever that launcher logged."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        outcome, n, rc = cluster.Cluster._poll_for_workers(
            _ClientStub(workers=()), _FakeDaskProc(returncode=1),
            expected=2, timeout=5., poll=0.25)
        assert (outcome, rc) == ('exited', 1)

    def test_timeout_when_too_few_arrive_in_time(self, monkeypatch):
        """Process still running but short of the count when time runs out: 'timeout', with
        how many did register."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        outcome, n, rc = cluster.Cluster._poll_for_workers(
            _ClientStub(workers=('tcp://n1:1',)), _FakeDaskProc(),
            expected=3, timeout=1., poll=0.25)
        assert (outcome, n, rc) == ('timeout', 1, None)

    def test_exited_when_any_of_several_processes_is_gone(self, monkeypatch):
        """The srun launcher passes a list of processes, one per machine size. If any one of
        them exits the placement failed, so the loop reports 'exited' with that process's
        code even though the others are still running."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        alive, dead = _FakeDaskProc(), _FakeDaskProc(returncode=2)
        outcome, n, rc = cluster.Cluster._poll_for_workers(
            _ClientStub(workers=()), [alive, dead],
            expected=4, timeout=5., poll=0.25)
        assert (outcome, rc) == ('exited', 2)


class TestWaitForSSHWorkers:
    """The SSH launcher's startup readiness check (#398), driven directly. The dead-process
    branch is covered through setup_cluster in TestSetupCluster; these pin the worker-count
    branch: dask ssh still running, the scheduler filling up over time."""

    def test_returns_once_the_full_expected_count_registers(self, monkeypatch):
        """The readiness signal is all of the expected workers registering, not dask ssh
        having been launched. Two asked for, two connected, so the wait returns two."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        n = cluster.Cluster.wait_for_ssh_workers(
            _ClientStub(workers=('tcp://n1:1', 'tcp://n2:1')), _FakeDaskProc(),
            expected=2, output_file=io.BytesIO(), out_dir='/log')
        assert n == 2

    def test_a_partial_cluster_is_not_ready_yet(self, monkeypatch):
        """Fewer workers than were asked for is the silent-degradation case of #200: the poll
        keeps waiting rather than returning a cluster smaller than reserved. Here one of two
        is up, then the second arrives, and only then does the wait return."""
        client = _ClientStub(workers=('tcp://n1:1',))
        polls = []

        def fake_sleep(_seconds):
            polls.append(1)
            if len(polls) == 2:
                client._workers = dict.fromkeys(('tcp://n1:1', 'tcp://n2:1'), {})

        monkeypatch.setattr(cluster.time, 'sleep', fake_sleep)
        n = cluster.Cluster.wait_for_ssh_workers(
            client, _FakeDaskProc(), expected=2, output_file=io.BytesIO(), out_dir='/log')
        assert n == 2
        assert len(polls) == 2

    def test_too_few_in_time_names_the_expected_and_connected_counts(self, monkeypatch):
        """dask ssh still running but short of the count means some workers never came up --
        a fit would quietly use less than was reserved. The message names how many of how
        many arrived and where to read what the missing ones wrote."""
        monkeypatch.setattr(cluster.time, 'sleep', lambda *_: None)
        with pytest.raises(printing.PybnfError) as exc:
            cluster.Cluster.wait_for_ssh_workers(
                _ClientStub(workers=('tcp://n1:1',)), _FakeDaskProc(),
                expected=3, output_file=io.BytesIO(), out_dir='/logdir', timeout=1.)
        message = exc.value.message
        assert '1 of the 3' in message
        assert '/logdir' in ' '.join(exc.value.hints)


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
        monkeypatch.delenv('SLURM_JOB_CPUS_PER_NODE', raising=False)   # same size everywhere

        scheduler_proc, worker_procs, expected, worker_logs = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), ['n1', 'n2'], parallel_count=None)

        (sched_cmd, sched_kwargs), (srun_cmd, srun_kwargs) = spy.calls
        assert sched_cmd == [*DASK, 'scheduler', '--scheduler-file', str(sched_file)]
        assert srun_cmd[0] == 'srun'
        assert srun_cmd[srun_cmd.index('--scheduler-file') + 1] == str(sched_file)
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '4'
        for kwargs in (sched_kwargs, srun_kwargs):
            assert kwargs.get('shell', False) is False
            assert kwargs['stderr'] is cluster.STDOUT
            assert hasattr(kwargs['stdout'], 'write')
        assert (tmp_path / 'dask_scheduler.log').exists()
        assert (tmp_path / 'dask_workers.log').exists()
        # One srun step (same size everywhere), and the worker total is what the readiness
        # wait needs: four workers on each of two machines.
        assert (scheduler_proc, worker_procs) == (spy.procs[0], [spy.procs[1]])
        assert expected == 8
        assert worker_logs == [str(tmp_path / 'dask_workers.log')]

    def test_machines_of_different_sizes_each_get_their_own_srun_step(self, monkeypatch, tmp_path):
        """#617: a mixed-size allocation cannot be placed by one srun step, because
        --cpus-per-task is a single number for the whole step and under cgroup binding a
        task that under-requests is confined to what it asked for. So each distinct size
        gets its own step: two 40-processor machines in one step at 40 workers each, the
        96-processor machine in a second step at 96. The steps run on disjoint machines,
        which SLURM allows concurrently, and each writes its own log so their output does
        not interleave."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '40(x2),96')

        scheduler_proc, worker_procs, expected, worker_logs = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), ['n1', 'n2', 'n3'], parallel_count=None)

        (sched_cmd, _), (first_cmd, _), (second_cmd, _) = spy.calls
        assert sched_cmd == [*DASK, 'scheduler', '--scheduler-file', str(sched_file)]
        # First step: the two same-size machines, 40 workers each.
        assert first_cmd[first_cmd.index('--nodelist') + 1] == 'n1,n2'
        assert first_cmd[first_cmd.index('--cpus-per-task') + 1] == '40'
        assert first_cmd[first_cmd.index('--nworkers') + 1] == '40'
        # Second step: the larger machine on its own, 96 workers.
        assert second_cmd[second_cmd.index('--nodelist') + 1] == 'n3'
        assert second_cmd[second_cmd.index('--cpus-per-task') + 1] == '96'
        assert second_cmd[second_cmd.index('--nworkers') + 1] == '96'
        assert worker_procs == [spy.procs[1], spy.procs[2]]
        assert expected == 40 * 2 + 96                       # every worker across both steps
        assert worker_logs == [str(tmp_path / 'dask_workers.log'),
                               str(tmp_path / 'dask_workers_2.log')]
        assert (tmp_path / 'dask_workers.log').exists()
        assert (tmp_path / 'dask_workers_2.log').exists()

    def test_the_per_machine_arrangement_is_logged(self, monkeypatch, tmp_path, caplog):
        """#617 asks for the arrangement to be recorded, so a user can see which machines
        got how many workers. One line per distinct size, naming the machines and their
        count."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '40(x2),96')

        with caplog.at_level('INFO'):
            cluster.Cluster.setup_srun_cluster(
                str(sched_file), str(tmp_path), ['n1', 'n2', 'n3'], parallel_count=None)

        text = '\n'.join(r.message for r in caplog.records)
        assert 'n1' in text and 'n2' in text and '40' in text
        assert 'n3' in text and '96' in text

    def test_an_explicit_parallel_count_is_still_one_step_and_an_even_split(self, monkeypatch, tmp_path):
        """A user who sets parallel_count is asking for a specific number of workers, split
        evenly across the machines the way the SSH launcher has always done. That path is
        left alone by #617: one srun step, the count taken from parallel_count rather than
        from what each machine was granted, even on a mixed allocation."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '40(x2),96')   # ignored: the user chose

        scheduler_proc, worker_procs, expected, worker_logs = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), ['n1', 'n2', 'n3'], parallel_count=12)

        (_, _), (srun_cmd, _) = spy.calls                    # scheduler + exactly one srun
        assert worker_procs == [spy.procs[1]]
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '4'   # 12 across three machines
        assert expected == 12
        assert worker_logs == [str(tmp_path / 'dask_workers.log')]

    def _login_node(self, monkeypatch, allocation, cores=128):
        """The environment of the shell ``salloc`` opens on a login node (#642): SLURM
        publishes the job's per-node CPU list, but not the per-step count, which it sets
        only inside a step running on an allocated node. Both machine-level numbers
        describe the login node, which is large and is not in the allocation."""
        monkeypatch.delenv('SLURM_CPUS_ON_NODE', raising=False)
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', allocation)
        monkeypatch.setattr(cluster, 'DASK_CPU_COUNT', cores)
        monkeypatch.setattr(cluster, 'cpu_count', lambda: cores)

    def test_a_launch_from_the_login_node_asks_for_what_the_job_holds(self, monkeypatch, tmp_path):
        """#642, the reported failure: on many clusters the shell ``salloc`` opens runs on
        the login node while the allocation is held on a compute node. The launcher used to
        fall through to the login node's own processor count and ask SLURM for that many
        CPUs per task, which SLURM refuses outright -- "Unable to create step ...: More
        processors requested than permitted" -- so no worker ever started. The oracle is the
        argv: 20 workers and 20 CPUs, from the allocation, not 128 from the login node."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        self._login_node(monkeypatch, '20', cores=128)

        _, worker_procs, expected, _ = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), ['n1'], parallel_count=None)

        (_, _), (srun_cmd, _) = spy.calls                    # scheduler + exactly one srun
        assert worker_procs == [spy.procs[1]]
        assert srun_cmd[srun_cmd.index('--cpus-per-task') + 1] == '20'
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '20'
        assert expected == 20

    def test_an_explicit_parallel_count_is_capped_by_the_job_not_by_the_login_node(self, monkeypatch, tmp_path):
        """The CPU request is capped so that a deliberately oversubscribed parallel_count
        still runs: every worker the user asked for starts, while the request stays inside
        what the job holds, because SLURM refuses a larger one. From the login node that cap
        was measured against the login node's processors and so did nothing -- 64 workers
        asked for 64 CPUs of a 20-CPU allocation and the step was refused."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        self._login_node(monkeypatch, '20', cores=128)

        _, _, expected, _ = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), ['n1'], parallel_count=64)

        (_, _), (srun_cmd, _) = spy.calls
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '64'     # what the user asked for
        assert srun_cmd[srun_cmd.index('--cpus-per-task') + 1] == '20'
        assert expected == 64

    def test_the_count_the_layout_read_is_the_count_the_command_uses(self, monkeypatch, tmp_path):
        """The per-machine list PyBNF already read is handed to the command rather than the
        environment being read a second time. So a $SLURM_CPUS_ON_NODE that does not describe
        this job -- exported by hand as the #642 workaround, or left over from an earlier
        allocation -- no longer sizes the run: the list SLURM published for this job does."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '128')              # stale, not this job
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '20(x2)')      # this job, two machines

        _, _, expected, _ = cluster.Cluster.setup_srun_cluster(
            str(sched_file), str(tmp_path), ['n1', 'n2'], parallel_count=None)

        (_, _), (srun_cmd, _) = spy.calls
        assert srun_cmd[srun_cmd.index('--cpus-per-task') + 1] == '20'
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '20'
        assert expected == 40

    def test_an_unusable_cpus_per_node_falls_back_to_one_uniform_step(self, monkeypatch, tmp_path, caplog):
        """If SLURM did not publish a per-machine list PyBNF can line up with the node names,
        it cannot size each machine, so it does the safe thing the launcher did before #617:
        one step, the same count everywhere, with a warning that per-machine sizing was not
        available."""
        sched_file = tmp_path / 'dask_scheduler.json'
        spy = _SchedulerSpy(str(sched_file), write_after=1)
        monkeypatch.setattr(cluster, 'Popen', spy)
        monkeypatch.setattr(cluster.time, 'sleep', spy.sleep)
        monkeypatch.setenv('SLURM_CPUS_ON_NODE', '4')
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', 'garbage')     # will not parse

        with caplog.at_level('WARNING'):
            scheduler_proc, worker_procs, expected, worker_logs = cluster.Cluster.setup_srun_cluster(
                str(sched_file), str(tmp_path), ['n1', 'n2'], parallel_count=None)

        (_, _), (srun_cmd, _) = spy.calls                    # scheduler + exactly one srun
        assert worker_procs == [spy.procs[1]]
        assert srun_cmd[srun_cmd.index('--nworkers') + 1] == '4'
        assert expected == 8
        assert worker_logs == [str(tmp_path / 'dask_workers.log')]
        assert any('every machine the same' in r.message.lower() and 'SLURM_JOB_CPUS_PER_NODE' in r.message
                   for r in caplog.records)

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
        cluster.Cluster.setup_srun_cluster(str(sched_file), str(tmp_path), ['n1'])

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

        cluster.Cluster.setup_srun_cluster(str(sched_file), str(tmp_path), ['n1'])

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
                str(tmp_path / 'dask_scheduler.json'), str(tmp_path), ['n1', 'n2'])

        assert len(spy.calls) == 1                          # srun was never launched
        assert spy.procs[0].terminated is True

    def test_a_missing_scheduler_file_directory_is_refused_up_front(self, tmp_path):
        """A scheduler file in a directory that does not exist would fail inside
        dask, as a traceback in a log file nobody is watching. Checked here, where
        it can be a configuration error naming the path."""
        with pytest.raises(printing.PybnfError, match='scheduler file .* does not exist'):
            cluster.Cluster.setup_srun_cluster(
                str(tmp_path / 'no_such_dir' / 's.json'), str(tmp_path), ['n1'])

    def test_a_missing_log_directory_is_refused_up_front(self, tmp_path):
        """Likewise for the directory the logs go in: opening that file is the first
        thing the bring-up does, and a bare OSError from it would reach the user as
        "an unknown error ... please report this bug"."""
        with pytest.raises(printing.PybnfError, match='logs .* does not exist'):
            cluster.Cluster.setup_srun_cluster(
                str(tmp_path / 's.json'), str(tmp_path / 'no_such_dir'), ['n1'])


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

    def test_brings_up_srun_with_the_node_names_and_connects_by_file(self, monkeypatch):
        """The srun bring-up gets the scheduler-file path, the output directory, the
        *names* of the machines (needed to size each one, #617) and parallel_count;
        the client then connects through that file."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1 n2 n3'))
        c = _build(self._cfg_srun(output_dir='out', parallel_count=12))

        expected_file = os.path.abspath(os.path.join('out', 'dask_scheduler.json'))
        assert rec.srun_setup_calls == [(expected_file, 'out', ['n1', 'n2', 'n3'], 12)]
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
        client, worker_procs, expected, worker_logs = rec.srun_wait_calls[0]
        assert client is rec.last_client
        assert worker_procs is c._srun_worker_procs
        assert worker_logs == [os.path.join('out', 'dask_workers.log')]

    def test_a_failed_worker_wait_stops_what_it_started(self, monkeypatch):
        """A constructor that raises never becomes a Cluster, so no one else can
        tear it down: the scheduler and srun processes it started are stopped on
        the way out rather than left running in the allocation."""
        rec = _patch_init(monkeypatch, read_returns=('n1', 'n1'),
                          srun_raises=printing.PybnfError('no workers'))
        started = []
        faked_setup = cluster.Cluster.setup_srun_cluster   # the recorder _patch_init installed

        def spy_setup(*args):
            scheduler_proc, worker_procs, expected, logs = faked_setup(*args)
            started.append(scheduler_proc)
            started.extend(worker_procs)
            return scheduler_proc, worker_procs, expected, logs

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
