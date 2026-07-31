"""Functions for managing dask cluster setup and teardown on distributed computing systems"""


from .printing import PybnfError

from subprocess import run, TimeoutExpired, Popen, PIPE, CalledProcessError, DEVNULL
from tempfile import TemporaryFile

import logging
import re
import time
import numpy as np
import os
from multiprocessing import cpu_count
from distributed import Client, LocalCluster
from dask import __version__ as daskv
from distributed import __version__ as distributedv
from .config import init_logging, reinit_logging

logger = logging.getLogger(__name__)


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

        # Find the name of the scheduler node, and a list of all available nodes (node_string), depending on what
        # cluster options are set
        if config.config['scheduler_file']:
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
        else:
            self._dask_proc = None

        logger.info(f'Initializing dask Client with dask v{daskv}, distributed v{distributedv}')

        if config.config['scheduler_file']:
            # Scheduler node read in from scheduler file stored on shared file system
            logger.info('Creating a client using the scheduler file')
            self.client = Client(scheduler_file=config.config['scheduler_file'])
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
            if re.match('slurm', ctype, flags=re.IGNORECASE):
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

    def teardown(self):
        """
        Terminates the process running the `dask-ssh` script after completion of fitting run

        """
        logger.info('Closing client')
        self.client.close()
        if self._dask_proc:
            logger.info('Closing dask-ssh subprocess')
            self._dask_proc.terminate()
