"""Functions for managing dask cluster setup and teardown on distributed computing systems"""


from .printing import PybnfError

from subprocess import run, TimeoutExpired, Popen, PIPE, STDOUT, CalledProcessError, DEVNULL

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

        logger.info('Initializing dask Client with dask v%s, distributed v%s' % (daskv, distributedv))

        if config.config['scheduler_file']:
            # Scheduler node read in from scheduler file stored on shared file system
            logger.info('Creating a client using the scheduler file')
            self.client = Client(scheduler_file=config.config['scheduler_file'])
            self.local = False
        elif scheduler_node:
            logger.info('Creating a client by connecting to the scheduler node %s:8786' % scheduler_node)
            self.client = Client('%s:8786' % scheduler_node)
            self.local = False
        elif config.config['parallel_count'] is not None:
            logger.info('Creating a local client manually set to %i workers' % config.config['parallel_count'])
            lc = LocalCluster(n_workers=config.config['parallel_count'], threads_per_worker=1)
            self.client = Client(lc)
            self.client.run(init_logging, log_prefix, debug, log_level_name)
            self.local = True
        else:
            logger.info('Creating a local client with default parallel count')
            self.client = Client()
            self.client.run(init_logging, log_prefix, debug, log_level_name)
            self.local = True

        # Required because with distributed v1.22.0, logger breaks after calling Client()
        reinit_logging(log_prefix, debug, log_level_name)

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
                get_hosts_cmd = ['scontrol', 'show', 'hostname', '$SLURM_JOB_NODELIST']
                try:
                    proc = run(' '.join(get_hosts_cmd), shell=True, stdout=PIPE, timeout=10, check=True)
                except TimeoutExpired:
                    logger.error('Could not retrieve host names in 10s')
                    raise PybnfError('Failed to find node names in a reasonable time.  Exiting')
                except CalledProcessError:
                    logger.error('User specified SLURM cluster, but command "%s" failed' % ' '.join(get_hosts_cmd))
                    raise PybnfError('Command to find node names failed.  Confirm use of SLURM cluster.  Exiting')
                nodes = re.split('\n', proc.stdout.decode('UTF-8').strip())
                scheduler_node = nodes[0]
                logger.info('Node %s is being used as the scheduler node' % scheduler_node)
                logger.info('Node(s) %s is/are being used as compute nodes' % ','.join(nodes))
                node_string = ' '.join(nodes)
            elif re.match('((torque)|(pbs))', ctype, flags=re.IGNORECASE):
                raise PybnfError("TORQUE cluster support is not yet implemented")
            else:
                logger.error("Unknown cluster type: %s" % config.config['cluster_type'])
                raise PybnfError("Unknown cluster type: %s" % config.config['cluster_type'])
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
        logger.info('Starting dask-ssh subprocess using nodes %s' % node_string)
        if parallel_count is None:
            dask_ssh_cmd = 'dask-ssh %s --log-directory %s --nthreads 1 --nprocs %i' % (node_string, out_dir, cpu_count())
        else:
            n_per_node = int(np.ceil(parallel_count/len(node_string.split())))
            logger.info('Manually setting %i workers per node' % n_per_node)
            dask_ssh_cmd = 'dask-ssh %s --log-directory %s --nprocs %i --nthreads 1' % (node_string, out_dir, n_per_node)
        dask_ssh_proc = Popen(dask_ssh_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        time.sleep(10)
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
