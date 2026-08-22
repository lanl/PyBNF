"""Functions for managing dask cluster setup and teardown on distributed computing systems

Two launchers bring up a multi-machine cluster, chosen by ``cluster_type``:

* the **SSH launcher** (``cluster_type = slurm``, or ``scheduler_node`` / ``worker_nodes``
  set by hand), which runs ``dask ssh`` and therefore logs in to every node; and
* the **srun launcher** (``cluster_type = slurm-srun``, #614), which never logs in
  anywhere. It starts the scheduler here, has SLURM place one ``dask worker`` per node
  inside the allocation the scheduler already granted, and connects through the scheduler
  file the scheduler writes.

The srun launcher exists because the SSH one cannot work at all on a cluster whose nodes
authenticate to each other by host-based or Kerberos (GSSAPI) SSH: ``dask ssh`` logs in
with paramiko, which offers only public-key and password authentication -- it has no
host-based support and dask never enables its GSSAPI support -- so on such a cluster the
login fails no matter what the user configures, and no amount of ``ssh-keygen`` helps. See
docs/adr/0122 for the full argument. When that login is what fails, the SSH launcher says
so, quotes what ``dask ssh`` said, and names both ways of running that need no login
(#618) -- the failure ends the run, so the message is the whole of what the user gets.

Both launchers size their default worker pool from what the *job* was granted rather than
from how big the machine is, and record which number they used and where it came from
(#616); ``Cluster.cpus_per_node`` is the one place that decides it.
"""


from .printing import PybnfError

from importlib.metadata import entry_points
from importlib.util import find_spec
from subprocess import run, TimeoutExpired, Popen, PIPE, CalledProcessError, STDOUT
from tempfile import TemporaryFile

import json
import logging
import re
import sys
import time
import numpy as np
import os
from multiprocessing import cpu_count
from distributed import Client, LocalCluster
from dask import __version__ as daskv
# What the operating system will actually let this process run on: the machine's
# processors narrowed by CPU affinity and by any cgroup CPU quota. Bound to a module
# global (rather than read through ``dask.system``) both because dask computes it once at
# import time and because that makes it substitutable in tests, the way ``cpu_count`` is.
from dask.system import CPU_COUNT as DASK_CPU_COUNT
from distributed import __version__ as distributedv
from .config import init_logging, reinit_logging

logger = logging.getLogger(__name__)


# The cluster_type values that select the srun launcher (#614): ``slurm-srun`` is the
# documented spelling; ``slurm_srun``, ``slurmsrun`` and a bare ``srun`` are accepted so a
# reasonable guess is not answered with "Unknown cluster type". Matched with fullmatch, so
# this never swallows a plain ``slurm`` (and, conversely, the SLURM branch's prefix match
# must be tested *after* this one, since ``re.match('slurm', 'slurm-srun')`` succeeds).
SRUN_CLUSTER_TYPE_RE = re.compile(r'(slurm[-_]?)?srun', flags=re.IGNORECASE)

# What the srun launcher writes into the output directory. The scheduler file is how the
# workers and this process find the scheduler, so it has to live on the shared filesystem
# PyBNF already requires for a cluster run; the two logs are the only record of what the
# scheduler and the srun-launched workers said, since neither has a terminal.
SRUN_SCHEDULER_FILENAME = 'dask_scheduler.json'
SRUN_SCHEDULER_LOG = 'dask_scheduler.log'
SRUN_WORKER_LOG = 'dask_workers.log'

# Readiness limits for both launchers. Every wait is a poll on a real signal -- the
# scheduler file appearing, a worker registering with the scheduler -- rather than a fixed
# sleep, and every one also watches the launched process, so a bring-up that fails outright
# is reported in the time it takes to fail rather than after the whole timeout.
#
# SSH_WORKER_TIMEOUT is the SSH launcher's share of this (#398). It replaces a fixed 10 s
# sleep in setup_cluster that assumed the cluster was up after ten seconds whatever the
# truth: on a real SLURM cluster the workers took 26-59 s to register (measured, see #398),
# so ten seconds connected before they were ready; on a fast cluster it was longer than
# needed. The wait now polls for the workers dask ssh brings up and reports a login that
# fails as soon as dask ssh exits. Sized like SRUN_WORKER_TIMEOUT, with headroom over the
# slowest bring-up observed.
SCHEDULER_FILE_TIMEOUT = 60.
SRUN_WORKER_TIMEOUT = 120.
SSH_WORKER_TIMEOUT = 120.
READINESS_POLL_INTERVAL = 0.25

# How long teardown waits for a process PyBNF started (the worker launcher, the scheduler) to
# exit after being asked to stop, before killing it outright (#398). This replaces a fixed
# 10 s sleep that followed every cluster teardown: teardown now returns as soon as the
# processes are really gone, and still bounds the wait for one that will not stop.
TEARDOWN_TIMEOUT = 30.


# How PyBNF invokes dask's command line interface (#615). Two things are decided here.
#
# **The subcommand form.** PyBNF used to run ``dask-ssh``, one of three standalone scripts
# (with ``dask-scheduler`` and ``dask-worker``) that distributed stopped installing in
# 2026.6.0 -- so on any current install every multi-machine run died immediately on
# ``FileNotFoundError: dask-ssh``, before a single simulation. The same features are
# subcommands of the unified ``dask`` program, and have been since well before 2024.1.0,
# the oldest dask/distributed pyproject.toml allows: that release already registers
# ``ssh``, ``scheduler`` and ``worker`` in the ``dask_cli`` entry point group the CLI
# builds itself from. So the new form works across the whole supported range, and no
# version floor has to move.
#
# **Through this interpreter, not through PATH.** ``dask ssh`` propagates its own
# ``sys.executable`` into the commands it starts on the remote nodes, so a ``dask``
# resolved from PATH -- a system-wide one, say, when PyBNF is run from a virtualenv by
# absolute path without activating it -- would start the remote workers under a different
# Python than the one running the fit. ``-m`` makes the environment that runs the workers
# the same one that runs PyBNF, by construction.
DASK_CLI = [sys.executable, '-m', 'dask']


# What a failed SSH bring-up says when the *login* is what failed (#618). The words can
# come from either half of what PyBNF captures: dask prints its own account of the failure
# ("SSH reported this exception: <the paramiko exception>") and lets paramiko's traceback
# fall to stderr, and both now land in one file. This is the vocabulary of a refused
# credential only -- "Authentication failed", "No authentication methods available", an
# encrypted key (PasswordRequiredException), a host key that did not match. Deliberately
# not dask's own "SSH connection error" heading, which it prints for a machine that could
# not be reached at all just as readily: a network failure is a different problem, and
# answering it with advice about keys and passwords would send the user the wrong way.
SSH_LOGIN_FAILURE_RE = re.compile(
    r'authentication|SSHException|permission denied|publickey|password|host ?key',
    flags=re.IGNORECASE)

# Terminal colour codes, which dask wraps its failure lines in. They have to come out
# before that text is quoted into a log file or an error message, where they would
# otherwise appear as literal escape characters.
ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def check_dask_subcommand(subcommand):
    """
    Confirm that ``dask <subcommand>`` can be run by this interpreter, before running it.

    A missing command surfaces as ``FileNotFoundError`` or an unrecognized-command exit
    from a subprocess PyBNF launched -- either an "unknown error ... please report this
    bug" traceback or, worse, a bring-up that fails ten seconds later for a reason the
    user has to go digging for (#615). This is the same question dask's own CLI asks when
    it assembles its command group: subcommands come from the ``dask_cli`` entry point
    group, so anything this check cannot see, ``dask`` cannot run either.

    :param subcommand: The dask subcommand PyBNF is about to run, e.g. 'ssh'
    :type subcommand: str
    :raises PybnfError: if this interpreter has no dask CLI, or no such subcommand
    """
    if find_spec('dask.__main__') is None:
        logger.error('The installed dask has no command line interface (dask.__main__)')
        raise PybnfError('The installed dask (v%s) has no command line interface, which PyBNF '
                         'needs to start workers on other machines.' % daskv,
                         hint=['Install dask 2024.1.0 or newer in the environment running '
                               'PyBNF (%s).' % sys.executable])
    available = sorted(ep.name for ep in entry_points(group='dask_cli'))
    if subcommand not in available:
        logger.error("The dask CLI has no '%s' subcommand; it offers: %s"
                     % (subcommand, ', '.join(available) or 'nothing'))
        raise PybnfError("The installed dask has no '%s' subcommand, which PyBNF needs to start "
                         "workers on other machines. Available subcommands: %s."
                         % (subcommand, ', '.join(available) or 'none'),
                         hint=["'%s' is provided by the distributed package (v%s here); install "
                               'distributed 2024.1.0 or newer alongside dask in the environment '
                               'running PyBNF (%s).' % (subcommand, distributedv, sys.executable)])


def uses_srun(cluster_type):
    """
    Whether this ``cluster_type`` selects the srun launcher (#614).

    :param cluster_type: The configured cluster_type, or None for a local run
    :type cluster_type: str or None
    :return: True if workers should be started with srun rather than over SSH
    :rtype: bool
    """
    if not cluster_type:
        return False
    return SRUN_CLUSTER_TYPE_RE.fullmatch(cluster_type.strip()) is not None


# One compressed CPU-count group as SLURM writes it in $SLURM_JOB_CPUS_PER_NODE: a number,
# optionally followed by "(xN)" meaning N nodes in a row were granted that many CPUs. So
# "40(x2),96" is three nodes granted 40, 40 and 96. The whole variable, once expanded, lines
# up one-to-one with the node list `scontrol show hostname` returns, which is how the srun
# launcher learns how many workers each machine should run (#617).
CPUS_PER_NODE_GROUP_RE = re.compile(r'\s*(\d+)(?:\(x(\d+)\))?\s*')


def expand_cpus_per_node(spec):
    """
    Expand a ``$SLURM_JOB_CPUS_PER_NODE`` string into one CPU count per node (#617).

    SLURM records the per-node counts in a run-length form, e.g. ``40(x2),96`` for three
    nodes granted 40, 40 and 96 CPUs. This turns that into ``[40, 40, 96]``, in the same
    order as the node list, so each machine's granted count can be matched to its name.

    :param spec: The value of ``$SLURM_JOB_CPUS_PER_NODE``, or None
    :type spec: str or None
    :return: one CPU count per node, or None if the text is empty or does not parse
    :rtype: list or None
    """
    if not spec or not spec.strip():
        return None
    counts = []
    for group in spec.strip().split(','):
        match = CPUS_PER_NODE_GROUP_RE.fullmatch(group)
        if not match:
            return None
        count = int(match.group(1))
        repeats = int(match.group(2)) if match.group(2) else 1
        counts.extend([count] * repeats)
    return counts


class Cluster:
    """
    Class handling the setup and teardown of the dask Client used to submit simulation jobs
    The client is accessible
    """

    def __init__(self, config, log_prefix, debug, log_level_name):
        """
        Create the dask client using the given configuration

        :param config: Configuration object
        :type config: Configuration
        :param log_prefix:
        :type log_prefix: str
        :param debug: Whether debug mode is active
        :type debug: bool
        :param log_level_name: The logging level for the application
        :type log_level_name: str
        """
        logger.info('Initializing the Cluster')

        # The process that starts the workers: dask ssh on the SSH launcher, srun on the
        # srun launcher. None for a local run or when attaching to a cluster someone else
        # brought up.
        self._dask_proc = None
        # A dask scheduler PyBNF started itself, and the scheduler file it was told to
        # write. Both are None unless the srun launcher is in use -- every other path
        # either attaches to an existing scheduler or lets dask ssh start one (#614).
        self._scheduler_proc = None
        self._own_scheduler_file = None
        # srun launcher readiness and teardown (#614, #617). The workers are one or more srun
        # processes -- more than one only when the allocation holds machines of different
        # sizes, which take one job step each -- so this is a list. The expected count is the
        # total number of workers over all of them, and the logs are what each step wrote.
        self._srun_worker_procs = []
        self._srun_expected_workers = None
        self._srun_worker_logs = []
        # SSH launcher readiness (#398). When PyBNF starts the cluster with dask ssh, these
        # hold what the readiness wait needs: the file dask ssh's output was captured to (so a
        # failed login can be quoted), how many workers it should bring up, and the directory
        # it was told to log each node to.
        self._ssh_output_file = None
        self._ssh_expected_workers = None
        self._ssh_out_dir = None

        # Where the client should look for the scheduler. This is the user's
        # ``scheduler_file`` when they are attaching to a cluster of their own, and the
        # file PyBNF asks its own scheduler to write under the srun launcher.
        scheduler_file = config.config['scheduler_file']

        # Find the name of the scheduler node, and a list of all available nodes (node_string), depending on what
        # cluster options are set
        if uses_srun(config.config['cluster_type']):
            # srun launcher (#614): SLURM places the workers itself, so the node names are not
            # needed to *reach* the nodes -- but they are needed to size each machine, because
            # the default worker count on a machine is one per CPU that machine was granted,
            # and machines in one allocation can differ in size (#617).
            scheduler_node = None
            node_string = None
            if config.config['scheduler_node'] or config.config['worker_nodes']:
                logger.warning('cluster_type = %s starts workers with srun, so the '
                               'scheduler_node / worker_nodes keys are ignored.'
                               % config.config['cluster_type'])
            self.require_slurm_allocation()
            scheduler_file = self.srun_scheduler_file(config)
            self._own_scheduler_file = scheduler_file
            out_dir = config.config['output_dir']
            dummy, srun_nodes = self.read_node_names(config)
            (self._scheduler_proc, self._srun_worker_procs, self._srun_expected_workers,
             self._srun_worker_logs) = self.setup_srun_cluster(
                scheduler_file, out_dir, srun_nodes.split(), config.config['parallel_count'])
        elif config.config['scheduler_file']:
            # Scheduler node will be read in from scheduler file stored on shared file system
            node_string = None
            scheduler_node = None
        elif config.config['scheduler_node'] and config.config['worker_nodes']:
            scheduler_node = config.config['scheduler_node']
            node_string = ' '.join(config.config['worker_nodes'])
        elif config.config['scheduler_node']:
            dummy, node_string = self.read_node_names(config)
            scheduler_node = config.config['scheduler_node']
        else:
            scheduler_node, node_string = self.read_node_names(config)

        if node_string:
            ssh_out_dir = os.getcwd()
            self._ssh_out_dir = ssh_out_dir
            (self._dask_proc, self._ssh_expected_workers,
             self._ssh_output_file) = self.setup_cluster(node_string, ssh_out_dir,
                                                          config.config['parallel_count'])

        logger.info(f'Initializing dask Client with dask v{daskv}, distributed v{distributedv}')

        if self._scheduler_proc is not None:
            # srun launcher: PyBNF owns the scheduler and the workers, so it is also the
            # one path that can tell whether the workers arrived. Connecting to our own
            # scheduler always succeeds -- a scheduler with no workers is a perfectly good
            # scheduler -- so without the readiness check a failed placement would surface
            # as a fit that submits jobs and never gets one back. Anything that fails here
            # leaves processes we started running, so stop them before propagating.
            logger.info('Creating a client using the scheduler file PyBNF wrote')
            try:
                self.client = Client(scheduler_file=scheduler_file)
                self.local = False
                self.wait_for_srun_workers(self.client, self._srun_worker_procs,
                                           self._srun_expected_workers, self._srun_worker_logs)
            except Exception:
                self.stop_own_processes()
                raise
        elif scheduler_file:
            # Scheduler node read in from scheduler file stored on shared file system
            logger.info('Creating a client using the scheduler file')
            self.client = Client(scheduler_file=scheduler_file)
            self.local = False
        elif scheduler_node:
            logger.info(f'Creating a client by connecting to the scheduler node {scheduler_node}:8786')
            self.client = Client(f'{scheduler_node}:8786')
            self.local = False
            if self._dask_proc is not None:
                # PyBNF started this cluster with dask ssh, so it is the one that can tell
                # whether the workers arrived. Connecting to the scheduler says nothing about
                # that -- a scheduler with no workers connects fine -- so without this wait a
                # failed or slow bring-up would surface later as a fit that submits jobs and
                # never gets one back (#398, and the silent-degradation risk of #200). Anything
                # that fails here leaves the dask ssh process running, so stop it first.
                try:
                    self.wait_for_ssh_workers(self.client, self._dask_proc,
                                              self._ssh_expected_workers, self._ssh_output_file,
                                              self._ssh_out_dir)
                except Exception:
                    self.stop_own_processes()
                    raise
        else:
            # One local branch, one thread policy (#526). `parallel_count` chooses how many
            # worker *processes* there are; it never decides how many threads run inside one.
            lc_kwargs = self.local_cluster_kwargs(config.config['parallel_count'])
            if 'n_workers' in lc_kwargs:
                logger.info('Creating a local client manually set to %i single-threaded workers'
                            % lc_kwargs['n_workers'])
            else:
                logger.info('Creating a local client with one single-threaded worker per available core')
            lc = LocalCluster(**lc_kwargs)
            self.client = Client(lc)
            self.client.run(init_logging, log_prefix, debug, log_level_name)
            self.local = True

        # Required because with distributed v1.22.0, logger breaks after calling Client()
        reinit_logging(log_prefix, debug, log_level_name)

    @staticmethod
    def local_cluster_kwargs(parallel_count):
        """
        Build the ``LocalCluster`` keyword arguments for a local (non-cluster) run.

        ``threads_per_worker`` is 1 unconditionally (#526). PyBNF's simulation backends hold
        process-wide state that is not advertised as thread-safe -- a C++ engine plus code
        generation with module-level caches -- so two worker threads in one process can race
        (issue #525 caught exactly that: concurrent emissions through bngsim's cached
        sympy->C printer intermittently reported ordinary quotients as non-differentiable,
        which killed a `trf` fit). Every other client PyBNF builds is already single-threaded
        per worker: both ``dask ssh`` branches pass ``--nthreads 1``, and the manual-setup
        documentation recommends the same. Only the local *default* used to let dask pick,
        so a user who set nothing got the less safe configuration.

        ``n_workers`` is left to dask when ``parallel_count`` is None: given one thread per
        worker, dask sizes the pool at one worker per available core (``dask.system.CPU_COUNT``,
        which honors CPU affinity and cgroup quotas), matching the ``dask ssh`` default of
        ``--nworkers <cores> --nthreads 1``. Total concurrency is therefore unchanged from the
        old default -- the same number of jobs run at once, each in its own process.

        :param parallel_count: Number of parallel jobs requested, or None for one per core
        :type parallel_count: int or None
        :return: kwargs for ``distributed.LocalCluster``
        :rtype: dict
        """
        kwargs = {'threads_per_worker': 1}
        if parallel_count is not None:
            kwargs['n_workers'] = parallel_count
        return kwargs

    @staticmethod
    def read_node_names(config):
        """
        Reads the available node names, if running on a cluster.
        If not running on a cluster, returns None for both.

        :param config: PyBNF configuration
        :type config: pybnf.config.Configuration

        :return: scheduler node, string composed of all available nodes
        """
        scheduler_node, node_string = None, None  # Local run (Default if nothing set)
        # Set up cluster if necessary
        if config.config['cluster_type']:
            ctype = config.config['cluster_type']
            # The srun launcher is a SLURM cluster too -- it reads the same node list, and
            # only the way the workers are *started* differs -- so it shares this branch.
            # It has to be tested first: re.match('slurm', 'slurm-srun') succeeds, so the
            # prefix test below cannot distinguish them (#614).
            if uses_srun(ctype) or re.match('slurm', ctype, flags=re.IGNORECASE):
                logger.debug('Detected selection of SLURM cluster')
                # Build the command as an argument list and run it WITHOUT a
                # shell (ROB-3). The node list comes from $SLURM_JOB_NODELIST,
                # which we resolve via os.environ rather than shell expansion --
                # this both closes the shell-injection surface and avoids the
                # shell glob-expanding a compressed nodelist like "node[01-04]".
                # An unset/empty variable is omitted (matching the old empty
                # shell expansion), letting scontrol fall back to its own
                # default.
                get_hosts_cmd = ['scontrol', 'show', 'hostname']
                nodelist = os.environ.get('SLURM_JOB_NODELIST', '')
                if nodelist:
                    get_hosts_cmd.append(nodelist)
                try:
                    proc = run(get_hosts_cmd, stdout=PIPE, timeout=10, check=True)
                except TimeoutExpired:
                    logger.error('Could not retrieve host names in 10s')
                    raise PybnfError('Failed to find node names in a reasonable time.  Exiting')
                except CalledProcessError:
                    logger.error('User specified SLURM cluster, but command "{}" failed'.format(' '.join(get_hosts_cmd)))
                    raise PybnfError('Command to find node names failed.  Confirm use of SLURM cluster.  Exiting')
                nodes = re.split('\n', proc.stdout.decode('UTF-8').strip())
                scheduler_node = nodes[0]
                logger.info(f'Node {scheduler_node} is being used as the scheduler node')
                logger.info('Node(s) {} is/are being used as compute nodes'.format(','.join(nodes)))
                node_string = ' '.join(nodes)
            elif re.match('((torque)|(pbs))', ctype, flags=re.IGNORECASE):
                raise PybnfError("TORQUE cluster support is not yet implemented")
            else:
                logger.error("Unknown cluster type: {}".format(config.config['cluster_type']))
                raise PybnfError("Unknown cluster type: {}".format(config.config['cluster_type']))
        return scheduler_node, node_string

    @staticmethod
    def setup_cluster(node_string, out_dir, parallel_count=None):
        """
        Sets up a Dask cluster using the `dask ssh` command

        :param node_string: A string composed of a list of compute nodes
        :param out_dir: A directory for cluster logging output
        :param parallel_count: Total number of single-threaded worker processes over all
            nodes, divided evenly among them. If None, one worker per CPU the job was
            granted on a node (``cpus_per_node``)
        :return: a tuple of the ``dask ssh`` process, the number of worker processes it
            should bring up (one per node times the per-node count), and the open file its
            output was captured to. The caller waits on that worker count and quotes the
            captured file if the bring-up fails (:meth:`wait_for_ssh_workers`, #398)
        :rtype: tuple
        """
        # Ask before launching, so a dask that cannot do this reads as a configuration
        # error rather than as a FileNotFoundError traceback from Popen (#615).
        check_dask_subcommand('ssh')
        logger.info(f'Starting dask ssh subprocess using nodes {node_string}')
        # Build the dask ssh invocation as an argument list and launch it WITHOUT
        # a shell (ROB-3): each node name becomes its own literal argv entry, so
        # node names from config/SLURM can't be interpreted by a shell.
        # The command is `dask ssh`, not the `dask-ssh` script PyBNF used to run:
        # distributed stopped installing that script in 2026.6.0 (#615). See DASK_CLI
        # for why it is invoked through this interpreter rather than found on PATH.
        # The per-host worker-count flag is --nworkers. distributed renamed it
        # from --nprocs (the old name was deprecated ~2022.10 and removed by
        # 2023.x), so --nprocs no longer parses on any supported dask version
        # (pyproject pins dask/distributed >=2024.1.0).
        nodes = node_string.split()
        if parallel_count is None:
            # One worker per CPU the *job* holds on a node, not per processor the machine
            # has (#616). The two differ by more than an order of magnitude on a job that
            # asked for a small share of a large node, and the machine's count is the one
            # that oversubscribes it. The source is logged because a user who sees an
            # unexpected worker count needs to know which number PyBNF believed.
            n_per_node, source = Cluster.cpus_per_node()
            logger.info('Starting %i worker process(es) on each of %i node(s), one per CPU, '
                        'from %s' % (n_per_node, len(nodes), source))
            dask_ssh_cmd = [*DASK_CLI, 'ssh', *nodes,
                            '--log-directory', out_dir, '--nthreads', '1', '--nworkers', str(n_per_node)]
        else:
            n_per_node = int(np.ceil(parallel_count/len(nodes)))
            logger.info('Manually setting %i worker process(es) on each of %i node(s), from the '
                        'parallel_count key (%i over all nodes)'
                        % (n_per_node, len(nodes), parallel_count))
            dask_ssh_cmd = [*DASK_CLI, 'ssh', *nodes,
                            '--log-directory', out_dir, '--nworkers', str(n_per_node), '--nthreads', '1']
        # Capture what dask ssh says to a temp file rather than to a PIPE: dask ssh stays
        # running for the whole fit, and an undrained PIPE would deadlock once its buffer
        # fills. A regular file lets us surface an early bring-up failure below without
        # that risk.
        #
        # stdout is captured too rather than discarded (#618). When a login fails, dask
        # prints its own account of it -- which node it was connecting to, and the
        # exception paramiko raised -- with ``print``, and only the traceback falls to
        # stderr. Sending stdout to DEVNULL therefore threw away the half of the output
        # that names the cause, which is how a refused login could reach the user as a
        # bare exit code with nothing else attached. Keeping the stream costs nothing for
        # the rest of a healthy run: ``--log-directory`` above makes dask redirect each
        # remote command's output into a file on its own node, so the SSH channels carry
        # almost nothing back.
        #
        # ``PYTHONUNBUFFERED`` is what makes capturing stdout worth anything. dask ends a
        # failed bring-up with ``os._exit(1)``, which does not flush Python's buffers, and
        # stdout writing to a file is block-buffered -- so its account of the failure, a
        # few hundred bytes short of the buffer's 8 KB, is discarded at exit and never
        # reaches the file at all. Measured against dask 2026.7.1 on a login that fails:
        # 0 of dask's own lines survive without this, all 15 with it.
        dask_ssh_out = TemporaryFile()
        dask_ssh_proc = Popen(dask_ssh_cmd, stdout=dask_ssh_out, stderr=STDOUT,
                              env=dict(os.environ, PYTHONUNBUFFERED='1'))
        # No fixed wait here any more (#398). The caller connects a Client and then calls
        # wait_for_ssh_workers, which polls for these workers to register and watches this
        # process, so a failed login is reported as soon as dask ssh exits and a healthy
        # bring-up proceeds the moment the workers are up rather than after a fixed guess.
        expected_workers = n_per_node * len(nodes)
        return dask_ssh_proc, expected_workers, dask_ssh_out

    @staticmethod
    def captured_text(handle):
        """
        Read back everything a launched process wrote to its capture file, as plain text.

        Colour escapes are removed: dask colours its own failure lines, and those bytes
        would otherwise reach a log file and an error message as literal characters.

        :param handle: The open binary file the process was given as its output
        :return: the captured text, or '' if nothing was captured or it cannot be read
        :rtype: str
        """
        try:
            handle.seek(0)
            text = handle.read().decode('UTF-8', errors='replace')
        except (OSError, ValueError):
            return ''
        return ANSI_ESCAPE_RE.sub('', text).strip()

    @staticmethod
    def fold_traceback_frames(text, lines=40):
        """
        The same output with Python traceback frames folded away, for an error message.

        What ``dask ssh`` writes when a login fails is mostly traceback: one per node per
        retry, three retries each, every one of them a dozen frames of dask's and
        paramiko's own source. Quoted whole, the sentences that say what happened -- dask's
        "SSH reported this exception: ...", and the exception line each traceback ends with
        -- are buried in code the user did not write and cannot act on, and are the first
        thing a length limit throws away. Folding the frames left 32 lines of a measured
        137, and lost none of the sentences.

        Each traceback is recognized by its header and ends at the first line that is not
        indented, which is the exception itself; that line is kept, the frames between are
        dropped. The full text still goes to the log.

        :param text: The captured output
        :type text: str
        :param lines: Number of trailing lines to keep after folding
        :type lines: int
        :return: the output without traceback frames, at most ``lines`` lines
        :rtype: str
        """
        kept = []
        in_frames = False
        for line in text.splitlines():
            if line.startswith('Traceback (most recent call last)'):
                in_frames = True
                continue
            if in_frames:
                if not line.strip() or line[:1].isspace():
                    continue
                in_frames = False
            kept.append(line)
        return '\n'.join(kept[-lines:])

    @staticmethod
    def ssh_bringup_hints(output):
        """
        What to suggest to a user whose SSH bring-up failed (#618).

        Two things a bare exit code does not tell them. First, whether the login is the
        problem: on the cluster this was reported from it was, and no part of the message
        said so. A login failure is worth naming outright because the obvious remedy --
        creating SSH keys -- fixes only one of its causes, and because ``ssh`` succeeding
        from the same shell makes the failure look impossible (``dask ssh`` does not run
        ``ssh``; it logs in with paramiko, which offers a public key or a password and
        nothing else).

        Second, that a failed login is not the end of the run: two of the ways PyBNF can
        use several machines never log in anywhere, and both remain open. They are named
        whatever the cause, since anything that stops ``dask ssh`` leaves them as the ways
        forward.

        :param output: What dask ssh wrote before exiting
        :type output: str
        :return: suggested remedies, most specific first
        :rtype: list
        """
        hints = []
        if SSH_LOGIN_FAILURE_RE.search(output or ''):
            hints.append(
                'This looks like a failed login. PyBNF starts the workers with `dask ssh`, '
                'which does not run your `ssh` command: it logs in with the paramiko '
                'library, which can offer a public key or a typed password and nothing '
                'else. A cluster whose nodes authenticate to each other by host-based or '
                'Kerberos (GSSAPI) SSH refuses that login however you configure it, and '
                'creating SSH keys does not help -- the cluster is not asking for a key. '
                '`ssh OTHERNODE hostname` succeeding proves nothing here; the "Running on '
                'a cluster" documentation gives a one-line test of the login PyBNF makes.')
        hints.append(
            'Two ways of running on several machines need no login at all. On a SLURM '
            'cluster, cluster_type = slurm-srun (or pybnf -t slurm-srun) starts the workers '
            'with srun, inside the allocation SLURM already granted.')
        hints.append(
            'The other: start a dask scheduler and workers yourself, by whatever means your '
            'cluster supports, and give PyBNF the scheduler file with -s or the '
            'scheduler_file key. PyBNF then only connects to a cluster that is already up.')
        return hints

    # ----------------------------------------------------------------------- #
    # The srun launcher (#614, ADR-0122): bring the cluster up without an SSH
    # login, by asking SLURM to place the workers inside the allocation it has
    # already granted.
    # ----------------------------------------------------------------------- #

    @staticmethod
    def require_slurm_allocation():
        """
        Refuse the srun launcher when this process is not inside a SLURM allocation.

        Outside an allocation ``srun`` does not place a task, it *submits a job* and then
        waits for the scheduler to grant it -- so the failure this catches would otherwise
        look like PyBNF hanging with no output, possibly for hours. The allocation is what
        makes the launcher credential-free, so its absence is a configuration error.

        :raises PybnfError: if no SLURM allocation is visible in the environment
        """
        if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')):
            logger.error('cluster_type selects the srun launcher, but $SLURM_JOB_ID is not set')
            raise PybnfError('This cluster_type starts workers with srun, which must run inside a '
                             'SLURM allocation, but no allocation was found ($SLURM_JOB_ID is not '
                             'set).',
                             hint=['Start PyBNF from the shell that holds the allocation: the one '
                                   'salloc opened, or your sbatch script. A separate login to one '
                                   'of the allocated nodes does not inherit it.',
                                   'To run on nodes PyBNF reaches over SSH instead, set '
                                   'cluster_type = slurm.'])

    @staticmethod
    def srun_scheduler_file(config):
        """
        The scheduler file the srun launcher writes, as an absolute path.

        Under this launcher the scheduler file is an *output*: PyBNF starts the scheduler
        that writes it. ``scheduler_file`` therefore chooses where it goes (useful when the
        output directory is not the filesystem you want the workers to read connection
        information from), and defaults to the output directory, which a cluster run
        already requires to be shared.

        :param config: PyBNF configuration
        :type config: pybnf.config.Configuration
        :return: absolute path of the scheduler file
        :rtype: str
        """
        named = config.config['scheduler_file']
        if named:
            return os.path.abspath(named)
        return os.path.abspath(os.path.join(config.config['output_dir'], SRUN_SCHEDULER_FILENAME))

    @staticmethod
    def cpus_per_node():
        """
        The number of CPUs the running job was granted on a node, and where that came from.

        Both launchers size their default worker pool with this, because the number that
        decides how many processes to start has to describe what the *job* holds, not what
        the machine has (#616). ``multiprocessing.cpu_count()`` answers the second question:
        it reports every processor on the machine whatever the scheduler granted, so a job
        given 4 CPUs of a 128-processor node is told 128, and one worker process per
        processor oversubscribes it 32-fold -- 32 times the memory, and workers competing
        for time rather than a fit that runs faster. The two numbers agree only when whole
        nodes were allocated, which is why the defect stayed hidden.

        Three sources are consulted, best first:

        * ``$SLURM_CPUS_ON_NODE`` -- what the allocation granted on a node. Preferred
          because it is the only one that describes the *allocation* rather than the process
          doing the asking, so it is still the right number for a worker the SSH launcher
          starts on some other machine. (When nodes differ in size it describes this node;
          per-node counts are issue #617.)
        * ``dask.system.CPU_COUNT`` -- what the operating system will let this process run
          on: the machine's processors narrowed by CPU affinity and by any cgroup CPU quota.
          This is the number a local run already sizes itself by, and it is the right one
          whenever the job is confined on the machine PyBNF is running on but no scheduler
          published a count.
        * ``multiprocessing.cpu_count()`` -- the whole machine, correct only when nothing is
          limiting the job at all, and reached only if neither number above is usable.

        The srun launcher does not merely count workers with this: it also asks SLURM for
        that many CPUs per task, and a request larger than the allocation is refused
        outright.

        :return: CPUs granted per node, and a phrase naming where that number came from
        :rtype: tuple
        """
        granted = os.environ.get('SLURM_CPUS_ON_NODE', '').strip()
        if granted.isdigit() and int(granted) > 0:
            return int(granted), 'what SLURM granted the job ($SLURM_CPUS_ON_NODE)'
        if DASK_CPU_COUNT > 0:
            return DASK_CPU_COUNT, ("this process's CPU affinity and cgroup limits "
                                    '(dask.system.CPU_COUNT)')
        return cpu_count(), "this machine's whole processor count (multiprocessing.cpu_count)"

    @staticmethod
    def per_node_cpus(node_names):
        """
        How many CPUs the job was granted on each machine, one number per node (#617).

        The srun launcher's default is one worker per granted CPU, and machines in one
        allocation can differ in size, so this returns a count for each machine rather than
        the single number :meth:`cpus_per_node` gives. It reads ``$SLURM_JOB_CPUS_PER_NODE``,
        which lists the per-node counts in the same order as ``node_names``. If that variable
        is missing, does not parse, or does not have one entry per node, per-machine sizing is
        not available, so this falls back to the single :meth:`cpus_per_node` count for every
        machine -- the behaviour before this change, where every machine was sized the same --
        and says so, since a user on a mixed cluster is expecting each machine to be sized on
        its own.

        :param node_names: The machines in the allocation, in the order SLURM lists them
        :type node_names: list
        :return: a CPU count for each machine, and a phrase naming where the counts came from
        :rtype: tuple
        """
        spec = os.environ.get('SLURM_JOB_CPUS_PER_NODE', '')
        counts = expand_cpus_per_node(spec)
        if counts is not None and len(counts) == len(node_names):
            return counts, 'what SLURM granted each machine ($SLURM_JOB_CPUS_PER_NODE)'
        granted, source = Cluster.cpus_per_node()
        if not spec.strip():
            reason = '$SLURM_JOB_CPUS_PER_NODE is not set'
        elif counts is None:
            reason = 'could not read $SLURM_JOB_CPUS_PER_NODE (%r)' % spec
        else:
            reason = ('$SLURM_JOB_CPUS_PER_NODE lists %i machine(s) but the allocation has %i'
                      % (len(counts), len(node_names)))
        logger.warning('Sizing every machine the same because %s; using %s.' % (reason, source))
        return [granted] * len(node_names), source

    @staticmethod
    def dask_scheduler_command(scheduler_file):
        """
        Build the ``dask scheduler`` invocation that starts the scheduler on this node.

        A one-line command, named anyway so that it sits beside
        :meth:`srun_worker_command` and can be read -- and checked against the dask that
        is actually installed (#619) -- without starting a cluster to see it.

        :param scheduler_file: Path the scheduler should write its connection information to
        :type scheduler_file: str
        :return: the dask scheduler argument list
        :rtype: list
        """
        return [*DASK_CLI, 'scheduler', '--scheduler-file', scheduler_file]

    @staticmethod
    def srun_worker_command(scheduler_file, node_count, parallel_count=None):
        """
        Build the srun invocation that starts one ``dask worker`` process group per node.

        :param scheduler_file: Path of the scheduler file the workers should read
        :type scheduler_file: str
        :param node_count: Number of nodes in the allocation
        :type node_count: int
        :param parallel_count: Total number of worker processes over all nodes, or None for
            one per granted CPU
        :type parallel_count: int or None
        :return: the srun argument list
        :rtype: list
        """
        granted, source = Cluster.cpus_per_node()
        if parallel_count is None:
            n_per_node = granted
        else:
            # Same arithmetic as the SSH launcher: parallel_count is a total over all nodes.
            # Per-node counts on nodes of different sizes are issue #617, which needs a
            # per-node command either launcher can express -- ``dask ssh`` cannot.
            n_per_node = max(1, int(np.ceil(parallel_count / node_count)))
        # One task per node, each of which forks n_per_node single-threaded workers. The
        # task has to be given CPUs for all of them: with task/cgroup binding, a task that
        # asked for the default single CPU confines every process it forks to that one CPU,
        # which would quietly serialize the whole node. Capped at what was granted so that
        # a deliberately oversubscribed parallel_count still runs (SLURM refuses a request
        # for more CPUs than the job holds) rather than failing the run.
        cpus_per_task = max(1, min(n_per_node, granted))
        logger.info('Starting %i worker process(es) per node on %i node(s), asking SLURM for %i '
                    'CPU(s) per node; %i available per node, from %s'
                    % (n_per_node, node_count, cpus_per_task, granted, source))
        return ['srun',
                '--nodes', str(node_count), '--ntasks', str(node_count),
                '--ntasks-per-node', '1', '--cpus-per-task', str(cpus_per_task),
                # Prefix each output line with its task number, so one log file records
                # which node said what.
                '--label',
                # Run the worker with *this* interpreter rather than whatever ``dask`` the
                # remote PATH resolves to: the launcher exists to remove ambiguity about
                # which environment starts the workers, and PyBNF already requires the
                # shared filesystem that makes this path valid on every node (DASK_CLI).
                *DASK_CLI, 'worker',
                '--scheduler-file', scheduler_file,
                '--nworkers', str(n_per_node),
                # One thread per worker, for the same reason every other PyBNF-built worker
                # is single-threaded (#526, ADR-0089): the simulation backends hold
                # process-wide state that is not thread-safe.
                '--nthreads', '1']

    @staticmethod
    def srun_worker_command_for_group(scheduler_file, nodes, cpus):
        """
        Build the srun invocation for one group of machines that were all granted the same
        number of CPUs, starting one worker per granted CPU on each (#617).

        This is what the default path uses when the allocation holds machines of different
        sizes. A single srun step cannot start different numbers of workers on different
        machines -- ``--cpus-per-task`` is one value for the whole step, and under task/cgroup
        binding a task that under-asked for CPUs is confined to them -- so each distinct size
        is its own step, named by ``--nodelist``. The homogeneous case does not come here; it
        stays the single :meth:`srun_worker_command`.

        :param scheduler_file: Path of the scheduler file the workers should read
        :type scheduler_file: str
        :param nodes: The machines in this group, all granted the same CPU count
        :type nodes: list
        :param cpus: CPUs granted on each machine in the group, and so workers to start there
        :type cpus: int
        :return: the srun argument list for this group
        :rtype: list
        """
        return ['srun',
                # Name the exact machines this step runs on, so the steps for the different
                # sizes land on disjoint machines and can run at the same time.
                '--nodelist', ','.join(nodes),
                '--nodes', str(len(nodes)), '--ntasks', str(len(nodes)),
                '--ntasks-per-node', '1', '--cpus-per-task', str(cpus),
                '--label',
                *DASK_CLI, 'worker',
                '--scheduler-file', scheduler_file,
                '--nworkers', str(cpus),
                '--nthreads', '1']

    @staticmethod
    def setup_srun_cluster(scheduler_file, out_dir, node_names, parallel_count=None):
        """
        Start a dask scheduler here and a set of dask workers with srun, with no SSH login.

        The scheduler runs as an ordinary subprocess of this process, on this node, and is
        told to write ``scheduler_file``; the workers are started with srun, each reading that
        file. Nothing authenticates anywhere: SLURM already granted the allocation, which is
        the whole point of the launcher (#614). With no ``parallel_count`` set, each machine
        runs one worker per CPU it was granted, and an allocation of different-sized machines
        is brought up as one srun step per distinct size (#617).

        :param scheduler_file: Path the scheduler should write its connection information to
        :type scheduler_file: str
        :param out_dir: Directory for the scheduler and worker logs
        :type out_dir: str
        :param node_names: The machines in the allocation, in the order SLURM lists them
        :type node_names: list
        :param parallel_count: Total number of worker processes over all nodes, or None
        :type parallel_count: int or None
        :return: the scheduler process, the list of srun worker processes, the total number of
            workers to wait for, and the log file each srun step is writing
        :rtype: tuple
        """
        # Both of these would otherwise fail inside dask, or as a bare OSError from opening
        # a log file, rather than as a configuration error naming the path at fault.
        scheduler_dir = os.path.dirname(scheduler_file) or '.'
        if not os.path.isdir(scheduler_dir):
            raise PybnfError('Cannot write the dask scheduler file to %s: the directory does not '
                             'exist.' % scheduler_file)
        if not os.path.isdir(out_dir):
            raise PybnfError('Cannot write the dask cluster logs to %s: the directory does not '
                             'exist.' % out_dir)
        # A scheduler file left behind by an earlier run names a scheduler that is no longer
        # listening. Remove it up front, so that its reappearance is proof that *this*
        # scheduler started rather than something we have to distinguish from history.
        if os.path.exists(scheduler_file):
            logger.info('Removing a scheduler file left over from an earlier run: %s' % scheduler_file)
            os.remove(scheduler_file)

        check_dask_subcommand('scheduler')
        check_dask_subcommand('worker')
        scheduler_log = os.path.join(out_dir, SRUN_SCHEDULER_LOG)
        scheduler_cmd = Cluster.dask_scheduler_command(scheduler_file)
        logger.info('Starting the dask scheduler on this node, logging to %s' % scheduler_log)
        scheduler_proc = Cluster.popen_logged(scheduler_cmd, scheduler_log)
        try:
            address = Cluster.wait_for_scheduler_file(scheduler_file, scheduler_proc, scheduler_log)
        except Exception:
            scheduler_proc.terminate()
            raise
        logger.info('The dask scheduler is listening at %s' % address)

        commands, worker_logs, expected_total = Cluster.srun_worker_layout(
            scheduler_file, out_dir, node_names, parallel_count)
        worker_procs = []
        for srun_cmd, worker_log in zip(commands, worker_logs):
            logger.info('Starting dask workers with srun, logging to %s' % worker_log)
            worker_procs.append(Cluster.popen_logged(srun_cmd, worker_log))
        return scheduler_proc, worker_procs, expected_total, worker_logs

    @staticmethod
    def srun_worker_layout(scheduler_file, out_dir, node_names, parallel_count):
        """
        Work out the srun command(s) that start the workers, the log each writes, and how many
        workers to expect in total (#617).

        There is one command in every case except the one this issue is about: a default
        (auto-sized) run on machines of different sizes, which becomes one command per distinct
        size so each machine can be given a worker per CPU it holds. Everything else -- an
        explicit ``parallel_count``, and a default run where every machine is the same size --
        stays the single :meth:`srun_worker_command` it was before, so the common case is
        unchanged.

        :param scheduler_file: Path of the scheduler file the workers should read
        :type scheduler_file: str
        :param out_dir: Directory the worker logs are written in
        :type out_dir: str
        :param node_names: The machines in the allocation, in the order SLURM lists them
        :type node_names: list
        :param parallel_count: Total number of worker processes over all nodes, or None
        :type parallel_count: int or None
        :return: the srun command list(s), the matching log path(s), and the total worker count
        :rtype: tuple
        """
        node_count = len(node_names)
        main_log = os.path.join(out_dir, SRUN_WORKER_LOG)
        if parallel_count is not None:
            # The explicit override is left exactly as it was: one srun, an even split over all
            # nodes. Making it per-machine is deliberately out of scope (#617).
            cmd = Cluster.srun_worker_command(scheduler_file, node_count, parallel_count)
            per_node = int(cmd[cmd.index('--nworkers') + 1])
            return [cmd], [main_log], per_node * node_count
        counts, source = Cluster.per_node_cpus(node_names)
        if len(set(counts)) <= 1:
            # Every machine the same size (the norm): the current single command, unchanged.
            cmd = Cluster.srun_worker_command(scheduler_file, node_count, None)
            per_node = int(cmd[cmd.index('--nworkers') + 1])
            return [cmd], [main_log], per_node * node_count
        # Machines of different sizes: one srun step per distinct size, each machine in the
        # step given one worker per CPU it was granted. Group by size, keeping first-seen order.
        groups = {}
        for name, cpus in zip(node_names, counts):
            groups.setdefault(cpus, []).append(name)
        logger.info('Machines of different sizes in this allocation; sizing each by %s. '
                    'Starting one srun step per size:' % source)
        commands, logs, expected_total = [], [], 0
        for index, (cpus, nodes) in enumerate(groups.items()):
            commands.append(Cluster.srun_worker_command_for_group(scheduler_file, nodes, cpus))
            # The first step keeps the usual log name; the rest are numbered, so the concurrent
            # steps do not truncate and interleave one another's output.
            logs.append(main_log if index == 0
                        else os.path.join(out_dir, 'dask_workers_%i.log' % (index + 1)))
            expected_total += cpus * len(nodes)
            logger.info('  %i worker(s) on %i machine(s) (%s)'
                        % (cpus, len(nodes), ', '.join(nodes)))
        return commands, logs, expected_total

    @staticmethod
    def popen_logged(cmd, log_path):
        """
        Launch ``cmd`` with no shell, sending its output to ``log_path``.

        Both srun-launcher processes outlive this call and neither has a terminal, so their
        output has to go somewhere a user can read it afterwards -- and somewhere that
        cannot fill up and deadlock the writer, as an undrained pipe would. stderr is
        merged into stdout so one file reads in order.

        :param cmd: The argument list to run
        :type cmd: list
        :param log_path: File to write the process output to
        :type log_path: str
        :return: subprocess.Popen
        """
        logger.debug('Running: %s' % ' '.join(cmd))
        log_file = open(log_path, 'wb')
        try:
            return Popen(cmd, stdout=log_file, stderr=STDOUT)
        finally:
            # The child holds its own duplicate of the descriptor, so this handle is not
            # needed once the process exists; the log is read back from disk when a failure
            # has to be reported.
            log_file.close()

    @staticmethod
    def log_tail(log_path, lines=20):
        """
        The last few lines of a log file, for inclusion in an error message.

        :param log_path: File to read
        :type log_path: str
        :param lines: Number of trailing lines to keep
        :type lines: int
        :return: the trailing text, or '' if the file cannot be read
        :rtype: str
        """
        try:
            with open(log_path, 'rb') as f:
                text = f.read().decode('UTF-8', errors='replace')
        except OSError:
            return ''
        return '\n'.join(text.strip().splitlines()[-lines:])

    @staticmethod
    def wait_for_scheduler_file(scheduler_file, scheduler_proc, scheduler_log,
                                timeout=SCHEDULER_FILE_TIMEOUT, poll=READINESS_POLL_INTERVAL):
        """
        Wait until the scheduler has written a usable scheduler file, and return its address.

        The file is written without being renamed into place, so a reader can catch it
        half-written; requiring it to parse as JSON carrying an address is what makes its
        appearance a readiness signal rather than a race. The scheduler process is checked
        on every pass, so a scheduler that dies (an occupied port, a bad interpreter) is
        reported immediately instead of after the timeout.

        :param scheduler_file: Path the scheduler was told to write
        :type scheduler_file: str
        :param scheduler_proc: The running scheduler process
        :param scheduler_log: Path of the scheduler's log, quoted if it failed
        :type scheduler_log: str
        :param timeout: Seconds to wait before giving up
        :type timeout: float
        :param poll: Seconds between checks
        :type poll: float
        :return: the scheduler address recorded in the file
        :rtype: str
        :raises PybnfError: if the scheduler exits, or the file does not appear in time
        """
        for _ in range(max(1, int(timeout / poll))):
            returncode = scheduler_proc.poll()
            if returncode is not None:
                details = Cluster.log_tail(scheduler_log)
                logger.error('The dask scheduler exited with code %s during cluster bring-up. '
                             'Log:\n%s' % (returncode, details))
                raise PybnfError('The dask scheduler exited with code %s during cluster bring-up. %s'
                                 % (returncode, ('Details:\n%s' % details) if details
                                    else 'See %s for details.' % scheduler_log))
            try:
                with open(scheduler_file) as f:
                    address = json.load(f).get('address')
            except (OSError, ValueError):
                address = None
            if address:
                return address
            time.sleep(poll)
        logger.error('The dask scheduler did not write %s within %s s' % (scheduler_file, timeout))
        raise PybnfError('The dask scheduler did not write its connection file %s within %s s. '
                         'See %s for details.' % (scheduler_file, timeout, scheduler_log))

    @staticmethod
    def _poll_for_workers(client, worker_procs, expected, timeout, poll):
        """
        Poll the scheduler until enough workers have registered, watching the process (#398).

        This is the readiness mechanism both launchers share. Whether PyBNF started the workers
        with srun or with dask ssh, the question is the same: have the workers registered with
        the scheduler yet, and is the process that was meant to bring them up still running?
        Both are checked on every pass, so a bring-up that dies is caught in the time it takes
        to die rather than after the whole timeout. What differs between the launchers -- how
        many workers to wait for, and what to say when it goes wrong -- is left to the caller,
        which is why this returns an outcome rather than raising: the srun path quotes its log,
        the SSH path quotes what dask ssh said and names the login, and neither vocabulary
        belongs here.

        The srun launcher is the one path that runs on a cluster whose nodes need no login, so
        this shared loop is what a real multi-machine run actually exercises (#398's own reason
        for staying open was that the SSH login could not be tested on the cluster available).

        :param client: The connected dask Client
        :param worker_procs: The process bringing the workers up (dask ssh), or the list of
            them (the srun launcher runs more than one when machines differ in size, #617).
            Any one of them exiting is reported as ``'exited'``, since it means a group of
            workers failed to start.
        :param expected: Number of registered workers that counts as ready
        :type expected: int
        :param timeout: Seconds to wait before giving up
        :type timeout: float
        :param poll: Seconds between checks
        :type poll: float
        :return: a tuple of the outcome (``'ready'``, ``'exited'`` or ``'timeout'``), the number
            of workers that had registered, and the process return code (only set on ``'exited'``)
        :rtype: tuple
        """
        procs = worker_procs if isinstance(worker_procs, (list, tuple)) else [worker_procs]
        n_workers = 0
        for _ in range(max(1, int(timeout / poll))):
            for worker_proc in procs:
                returncode = worker_proc.poll()
                if returncode is not None:
                    return 'exited', n_workers, returncode
            try:
                n_workers = len(client.scheduler_info()['workers'])
            except Exception:
                n_workers = 0
            if n_workers >= expected:
                return 'ready', n_workers, None
            time.sleep(poll)
        return 'timeout', n_workers, None

    @staticmethod
    def wait_for_srun_workers(client, worker_procs, expected, worker_logs,
                              timeout=SRUN_WORKER_TIMEOUT, poll=READINESS_POLL_INTERVAL):
        """
        Wait until all of the srun-launched workers have registered with the scheduler.

        The waiting itself is :meth:`_poll_for_workers`, the loop both launchers share; this
        adds what is specific to srun -- the count to wait for, and what to say, quoting srun's
        own log(s), when an srun step dies or the workers never all arrive.

        The full ``expected`` count is required rather than just one worker (#200, #617). On a
        default run over machines of different sizes there is one srun step per size, and a
        step that never places its workers -- a queued job step, a request larger than that
        part of the allocation -- would otherwise be masked by another step that did place its
        own. Connecting to our own scheduler always succeeds, so nothing else would report it,
        and the fit would quietly run on fewer machines than were reserved.

        :param client: The connected dask Client
        :param worker_procs: The running srun process(es), one per machine size
        :type worker_procs: list
        :param expected: Total number of workers that should register across all the steps
        :type expected: int
        :param worker_logs: The srun output log(s), quoted if the workers never all arrive
        :type worker_logs: list
        :param timeout: Seconds to wait before giving up
        :type timeout: float
        :param poll: Seconds between checks
        :type poll: float
        :return: the number of workers that had registered
        :rtype: int
        :raises PybnfError: if an srun step exits, or the workers do not all register in time
        """
        outcome, n_workers, returncode = Cluster._poll_for_workers(
            client, worker_procs, expected=expected, timeout=timeout, poll=poll)
        if outcome == 'ready':
            logger.info('%i of %i dask worker process(es) registered with the scheduler'
                        % (n_workers, expected))
            return n_workers
        details = '\n'.join(t for t in (Cluster.log_tail(log) for log in worker_logs) if t)
        where = ', '.join(worker_logs)
        if outcome == 'exited':
            logger.error('An srun step exited with code %s before all workers started. Log:\n%s'
                         % (returncode, details))
            raise PybnfError('srun exited with code %s before all the dask workers started. %s'
                             % (returncode, ('Details:\n%s' % details) if details
                                else 'See %s for details.' % where))
        logger.error('Only %i of %i dask worker(s) registered within %s s. Log:\n%s'
                     % (n_workers, expected, timeout, details))
        raise PybnfError('Only %i of the %i expected dask worker process(es) started by srun '
                         'registered with the scheduler within %s s.%s'
                         % (n_workers, expected, timeout,
                            ('\nLog:\n%s' % details) if details else ''),
                         hint=['Read the srun log(s) (%s): srun reports there what it is waiting '
                               'for.' % where,
                               'A message about job step creation means another step already holds '
                               'the allocation; run PyBNF as the only job step.'])

    @staticmethod
    def wait_for_ssh_workers(client, dask_proc, expected, output_file, out_dir,
                             timeout=SSH_WORKER_TIMEOUT, poll=READINESS_POLL_INTERVAL):
        """
        Wait until the workers dask ssh is bringing up have registered with the scheduler (#398).

        This is what replaces the fixed 10 s sleep the SSH launcher used to take after starting
        dask ssh. The waiting itself is :meth:`_poll_for_workers`, the loop the srun launcher
        also uses, so the behaviour this adds is exercised on a real cluster through the srun
        path even though the SSH login cannot be (#398). What this adds is specific to dask ssh.
        If dask ssh exits, the login or launch failed, and its captured output is quoted the way
        an immediate failure was before -- naming the login as the likely cause when it looks
        like one, and naming the ways of running that need no login (#618).

        The full ``expected`` count is required rather than just one worker. A run that connected
        with fewer workers than were asked for is the silent-degradation problem of #200: it does
        not fail, it just quietly uses less than was reserved. Waiting for all of them turns that
        into a clear, bounded error instead.

        :param client: The connected dask Client
        :param dask_proc: The running dask ssh process
        :param expected: Number of worker processes dask ssh should bring up
        :type expected: int
        :param output_file: Open file dask ssh's output was captured to, quoted on failure
        :param out_dir: Directory dask ssh logged each node to, named when it said nothing here
        :type out_dir: str
        :param timeout: Seconds to wait before giving up
        :type timeout: float
        :param poll: Seconds between checks
        :type poll: float
        :return: the number of workers that had registered
        :rtype: int
        :raises PybnfError: if dask ssh exits, or fewer than ``expected`` workers register in time
        """
        outcome, n_workers, returncode = Cluster._poll_for_workers(
            client, dask_proc, expected=expected, timeout=timeout, poll=poll)
        if outcome == 'ready':
            logger.info('%i of %i dask worker process(es) registered with the scheduler'
                        % (n_workers, expected))
            return n_workers
        if outcome == 'exited':
            # dask ssh exited during bring-up, so the cluster never came up -- almost always
            # a refused login (see ssh_bringup_hints). The log keeps every line, including
            # the traceback frames; the message keeps what a user can act on.
            output = Cluster.captured_text(output_file)
            logger.error('dask ssh exited with code %s during cluster bring-up. Output:\n%s'
                         % (returncode, output if output else '(it produced none)'))
            raise PybnfError(
                'Could not start the workers on the other machines: dask ssh exited with code '
                '%s during cluster bring-up.\n%s'
                % (returncode,
                   ('This is what it said:\n%s' % Cluster.fold_traceback_frames(output))
                   if output else
                   ('It produced no output at all. Anything the nodes themselves wrote is '
                    'in %s on each of them.' % out_dir)),
                hint=Cluster.ssh_bringup_hints(output))
        output = Cluster.captured_text(output_file)
        logger.error('Only %i of the %i expected dask worker process(es) registered within %s s. '
                     'dask ssh output:\n%s' % (n_workers, expected, timeout,
                                               output if output else '(it produced none)'))
        raise PybnfError(
            'Only %i of the %i expected worker process(es) started over SSH registered with the '
            'scheduler within %s s.%s'
            % (n_workers, expected, timeout,
               ('\nThis is what dask ssh said:\n%s' % Cluster.fold_traceback_frames(output))
               if output else ''),
            hint=['Anything the workers themselves wrote is in the log directory %s on each '
                  'node.' % out_dir,
                  'A slow or busy cluster may need longer; the wait is SSH_WORKER_TIMEOUT in '
                  'pybnf/cluster.py.'])

    def teardown(self):
        """
        Terminates the processes PyBNF started for this run, after the fitting run completes

        The worker launcher goes first and the scheduler second: terminating srun signals
        the workers, which are the processes that would otherwise be left talking to a
        scheduler that is already gone. Nothing here touches a cluster PyBNF did not start
        -- ``scheduler_file`` and ``scheduler_node`` runs have no processes of their own.
        """
        logger.info('Closing client')
        self.client.close()
        self.stop_own_processes()

    def stop_own_processes(self):
        """
        Terminate the cluster processes PyBNF started, and remove the scheduler file it wrote

        Separate from :meth:`teardown` because bring-up can fail before there is a client to
        close, and the processes started up to that point still have to be stopped.
        """
        if self._dask_proc:
            self.stop_process(self._dask_proc, 'worker launcher subprocess')
            self._dask_proc = None
        # The srun launcher's workers, one process per machine size (#617). Stopped before the
        # scheduler, for the same reason the SSH worker launcher is: terminating them signals
        # workers that would otherwise be left talking to a scheduler that is already gone.
        for srun_proc in getattr(self, '_srun_worker_procs', None) or []:
            self.stop_process(srun_proc, 'srun worker launcher subprocess')
        self._srun_worker_procs = []
        if self._scheduler_proc:
            self.stop_process(self._scheduler_proc, 'dask scheduler subprocess')
            self._scheduler_proc = None
        if self._ssh_output_file is not None:
            # The captured dask ssh output is only needed while waiting for the workers; once
            # the run is over, close it so the temporary file is released.
            try:
                self._ssh_output_file.close()
            except OSError:
                pass
            self._ssh_output_file = None
        if self._own_scheduler_file and os.path.exists(self._own_scheduler_file):
            # The file names a scheduler that is being shut down, so leaving it in place
            # would leave a live-looking connection file behind for the next run to find.
            try:
                os.remove(self._own_scheduler_file)
            except OSError:
                logger.debug('Could not remove the scheduler file %s' % self._own_scheduler_file)
            self._own_scheduler_file = None

    @staticmethod
    def stop_process(proc, description, timeout=TEARDOWN_TIMEOUT):
        """
        Ask a process PyBNF started to stop, and wait until it actually has (#398).

        This is what replaces the fixed 10 s sleep that used to follow every cluster teardown.
        The process is asked to terminate and then waited on, so teardown returns as soon as it
        is really gone rather than after a fixed guess. A process that ignores the request is
        killed after ``timeout`` seconds, so a stuck one cannot hold the run open forever.

        :param proc: The process to stop
        :param description: What to call it in the log
        :type description: str
        :param timeout: Seconds to wait for it to exit before killing it
        :type timeout: float
        """
        logger.info('Closing the %s' % description)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except TimeoutExpired:
            logger.warning('The %s did not exit within %s s of being asked to; killing it.'
                           % (description, timeout))
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except TimeoutExpired:
                logger.error('The %s did not exit even after being killed.' % description)
