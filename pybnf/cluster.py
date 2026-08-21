"""Functions for managing dask cluster setup and teardown on distributed computing systems

Two launchers bring up a multi-machine cluster, chosen by ``cluster_type``:

* the **SSH launcher** (``cluster_type = slurm``, or ``scheduler_node`` / ``worker_nodes``
  set by hand), which runs ``dask-ssh`` and therefore logs in to every node; and
* the **srun launcher** (``cluster_type = slurm-srun``, #614), which never logs in
  anywhere. It starts the scheduler here, has SLURM place one ``dask worker`` per node
  inside the allocation the scheduler already granted, and connects through the scheduler
  file the scheduler writes.

The srun launcher exists because the SSH one cannot work at all on a cluster whose nodes
authenticate to each other by host-based or Kerberos (GSSAPI) SSH: ``dask-ssh`` logs in
with paramiko, which offers only public-key and password authentication -- it has no
host-based support and dask never enables its GSSAPI support -- so on such a cluster the
login fails no matter what the user configures, and no amount of ``ssh-keygen`` helps. See
docs/adr/0122 for the full argument.
"""


from .printing import PybnfError

from subprocess import run, TimeoutExpired, Popen, PIPE, CalledProcessError, DEVNULL, STDOUT
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

# Readiness limits for the srun launcher. Both waits are polls on a real signal -- the
# scheduler file appearing, and a worker registering with the scheduler -- rather than a
# fixed sleep, and both also watch the launched process, so a bring-up that fails outright
# is reported in the time it takes to fail rather than after the whole timeout (#398 asks
# for the same treatment of the SSH path's two fixed 10 s sleeps, which this does not
# touch).
SCHEDULER_FILE_TIMEOUT = 60.
SRUN_WORKER_TIMEOUT = 120.
READINESS_POLL_INTERVAL = 0.25


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

        # The process that starts the workers: dask-ssh on the SSH launcher, srun on the
        # srun launcher. None for a local run or when attaching to a cluster someone else
        # brought up.
        self._dask_proc = None
        # A dask scheduler PyBNF started itself, and the scheduler file it was told to
        # write. Both are None unless the srun launcher is in use -- every other path
        # either attaches to an existing scheduler or lets dask-ssh start one (#614).
        self._scheduler_proc = None
        self._own_scheduler_file = None
        self._srun_worker_log = None

        # Where the client should look for the scheduler. This is the user's
        # ``scheduler_file`` when they are attaching to a cluster of their own, and the
        # file PyBNF asks its own scheduler to write under the srun launcher.
        scheduler_file = config.config['scheduler_file']

        # Find the name of the scheduler node, and a list of all available nodes (node_string), depending on what
        # cluster options are set
        if uses_srun(config.config['cluster_type']):
            # srun launcher (#614): no node names are needed to *reach* the nodes -- SLURM
            # places the workers itself -- but the node count is, because it is what the
            # per-node worker arithmetic divides by.
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
            self._srun_worker_log = os.path.join(out_dir, SRUN_WORKER_LOG)
            dummy, srun_nodes = self.read_node_names(config)
            self._scheduler_proc, self._dask_proc = self.setup_srun_cluster(
                scheduler_file, out_dir, len(srun_nodes.split()), config.config['parallel_count'])
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
            self._dask_proc = self.setup_cluster(node_string, os.getcwd(), config.config['parallel_count'])

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
                self.wait_for_srun_workers(self.client, self._dask_proc, self._srun_worker_log)
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
        per worker: both ``dask-ssh`` branches pass ``--nthreads 1``, and the manual-setup
        documentation recommends the same. Only the local *default* used to let dask pick,
        so a user who set nothing got the less safe configuration.

        ``n_workers`` is left to dask when ``parallel_count`` is None: given one thread per
        worker, dask sizes the pool at one worker per available core (``dask.system.CPU_COUNT``,
        which honors CPU affinity and cgroup quotas), matching the ``dask-ssh`` default of
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
        Sets up a Dask cluster using the `dask-ssh` convenience script

        :param node_string: A string composed of a list of compute nodes
        :param out_dir: A directory for cluster logging output
        :param parallel_count: Total number of parallel threads to use over all nodes. If None, use all available threads
            (the dask-ssh default)
        :return: subprocess.Popen
        """
        logger.info(f'Starting dask-ssh subprocess using nodes {node_string}')
        # Build the dask-ssh invocation as an argument list and launch it WITHOUT
        # a shell (ROB-3): each node name becomes its own literal argv entry, so
        # node names from config/SLURM can't be interpreted by a shell.
        # The per-host worker-count flag is --nworkers. distributed renamed it
        # from --nprocs (the old name was deprecated ~2022.10 and removed by
        # 2023.x), so --nprocs no longer parses on any supported dask version
        # (pyproject pins dask/distributed >=2024.1.0).
        nodes = node_string.split()
        if parallel_count is None:
            dask_ssh_cmd = ['dask-ssh', *nodes,
                            '--log-directory', out_dir, '--nthreads', '1', '--nworkers', str(cpu_count())]
        else:
            n_per_node = int(np.ceil(parallel_count/len(nodes)))
            logger.info('Manually setting %i workers per node' % n_per_node)
            dask_ssh_cmd = ['dask-ssh', *nodes,
                            '--log-directory', out_dir, '--nworkers', str(n_per_node), '--nthreads', '1']
        # Capture stderr to a temp file rather than a PIPE: dask-ssh stays
        # running for the whole fit, and an undrained PIPE would deadlock once
        # its buffer fills. A regular file lets us surface an early bring-up
        # failure below without that risk.
        dask_ssh_err = TemporaryFile()
        dask_ssh_proc = Popen(dask_ssh_cmd, stdout=DEVNULL, stderr=dask_ssh_err)
        time.sleep(10)
        # If dask-ssh has already exited, the cluster never came up. Surface the
        # failure here instead of letting it resurface later as an opaque dask
        # Client connection error.
        returncode = dask_ssh_proc.poll()
        if returncode is not None:
            dask_ssh_err.seek(0)
            err_text = dask_ssh_err.read().decode('UTF-8', errors='replace').strip()
            dask_ssh_err.close()
            logger.error(f'dask-ssh exited with code {returncode} during cluster bring-up. stderr:\n{err_text}')
            raise PybnfError('Failed to start the dask-ssh cluster (dask-ssh exited with code {}). {}'.format(returncode, (f'Details:\n{err_text}') if err_text
                                else 'Check the cluster log directory for details.'))
        return dask_ssh_proc

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
        The number of CPUs the running job was granted on a node.

        ``$SLURM_CPUS_ON_NODE`` is what the allocation actually granted; ``cpu_count()`` is
        the size of the whole machine, which is only the same number when whole nodes were
        allocated. The srun launcher reads the former because it does not merely count
        workers with it -- it also asks SLURM for that many CPUs per task, and a request
        larger than the allocation is refused outright. (The SSH launcher still uses
        ``cpu_count()``; correcting that is issue #616, and it has to be corrected there
        too rather than here.)

        :return: CPUs granted per node, falling back to the machine's core count
        :rtype: int
        """
        granted = os.environ.get('SLURM_CPUS_ON_NODE', '').strip()
        if granted.isdigit() and int(granted) > 0:
            return int(granted)
        return cpu_count()

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
        granted = Cluster.cpus_per_node()
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
        logger.info('Starting %i worker process(es) per node on %i node(s), %i CPU(s) per node'
                    % (n_per_node, node_count, cpus_per_task))
        return ['srun',
                '--nodes', str(node_count), '--ntasks', str(node_count),
                '--ntasks-per-node', '1', '--cpus-per-task', str(cpus_per_task),
                # Prefix each output line with its task number, so one log file records
                # which node said what.
                '--label',
                # Run the worker with *this* interpreter rather than whatever ``dask`` the
                # remote PATH resolves to: the launcher exists to remove ambiguity about
                # which environment starts the workers, and PyBNF already requires the
                # shared filesystem that makes this path valid on every node.
                sys.executable, '-m', 'dask', 'worker',
                '--scheduler-file', scheduler_file,
                '--nworkers', str(n_per_node),
                # One thread per worker, for the same reason every other PyBNF-built worker
                # is single-threaded (#526, ADR-0089): the simulation backends hold
                # process-wide state that is not thread-safe.
                '--nthreads', '1']

    @staticmethod
    def setup_srun_cluster(scheduler_file, out_dir, node_count, parallel_count=None):
        """
        Start a dask scheduler here and a set of dask workers with srun, with no SSH login.

        The scheduler runs as an ordinary subprocess of this process, on this node, and is
        told to write ``scheduler_file``; the workers are one srun task per node, each
        reading that file. Nothing authenticates anywhere: SLURM already granted the
        allocation, which is the whole point of the launcher (#614).

        :param scheduler_file: Path the scheduler should write its connection information to
        :type scheduler_file: str
        :param out_dir: Directory for the scheduler and worker logs
        :type out_dir: str
        :param node_count: Number of nodes in the allocation
        :type node_count: int
        :param parallel_count: Total number of worker processes over all nodes, or None
        :type parallel_count: int or None
        :return: the scheduler process and the srun process
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

        scheduler_log = os.path.join(out_dir, SRUN_SCHEDULER_LOG)
        scheduler_cmd = [sys.executable, '-m', 'dask', 'scheduler', '--scheduler-file', scheduler_file]
        logger.info('Starting the dask scheduler on this node, logging to %s' % scheduler_log)
        scheduler_proc = Cluster.popen_logged(scheduler_cmd, scheduler_log)
        try:
            address = Cluster.wait_for_scheduler_file(scheduler_file, scheduler_proc, scheduler_log)
        except Exception:
            scheduler_proc.terminate()
            raise
        logger.info('The dask scheduler is listening at %s' % address)

        worker_log = os.path.join(out_dir, SRUN_WORKER_LOG)
        srun_cmd = Cluster.srun_worker_command(scheduler_file, node_count, parallel_count)
        logger.info('Starting dask workers with srun, logging to %s' % worker_log)
        srun_proc = Cluster.popen_logged(srun_cmd, worker_log)
        return scheduler_proc, srun_proc

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
    def wait_for_srun_workers(client, srun_proc, worker_log,
                              timeout=SRUN_WORKER_TIMEOUT, poll=READINESS_POLL_INTERVAL):
        """
        Wait until at least one srun-launched worker has registered with the scheduler.

        :param client: The connected dask Client
        :param srun_proc: The running srun process
        :param worker_log: Path of the srun output log, quoted if the workers never arrive
        :type worker_log: str
        :param timeout: Seconds to wait before giving up
        :type timeout: float
        :param poll: Seconds between checks
        :type poll: float
        :return: the number of workers that had registered
        :rtype: int
        :raises PybnfError: if srun exits, or no worker registers in time
        """
        for _ in range(max(1, int(timeout / poll))):
            returncode = srun_proc.poll()
            if returncode is not None:
                details = Cluster.log_tail(worker_log)
                logger.error('srun exited with code %s before any worker started. Log:\n%s'
                             % (returncode, details))
                raise PybnfError('srun exited with code %s before any dask worker started. %s'
                                 % (returncode, ('Details:\n%s' % details) if details
                                    else 'See %s for details.' % worker_log))
            try:
                n_workers = len(client.scheduler_info()['workers'])
            except Exception:
                n_workers = 0
            if n_workers:
                logger.info('%i dask worker process(es) registered with the scheduler' % n_workers)
                return n_workers
            time.sleep(poll)
        details = Cluster.log_tail(worker_log)
        logger.error('No dask worker registered within %s s. Log:\n%s' % (timeout, details))
        raise PybnfError('No dask worker started by srun registered with the scheduler within '
                         '%s s.%s' % (timeout, ('\nLog:\n%s' % details) if details else ''),
                         hint=['Read %s: srun reports there what it is waiting for.' % worker_log,
                               'A message about job step creation means another step already holds '
                               'the allocation; run PyBNF as the only job step.'])

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
            logger.info('Closing the worker launcher subprocess')
            self._dask_proc.terminate()
            self._dask_proc = None
        if self._scheduler_proc:
            logger.info('Closing the dask scheduler subprocess')
            self._scheduler_proc.terminate()
            self._scheduler_proc = None
        if self._own_scheduler_file and os.path.exists(self._own_scheduler_file):
            # The file names a scheduler that is being shut down, so leaving it in place
            # would leave a live-looking connection file behind for the next run to find.
            try:
                os.remove(self._own_scheduler_file)
            except OSError:
                logger.debug('Could not remove the scheduler file %s' % self._own_scheduler_file)
            self._own_scheduler_file = None
