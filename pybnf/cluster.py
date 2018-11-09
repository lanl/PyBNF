"""Functions for managing dask cluster setup and teardown on distributed computing systems"""


from .printing import PybnfError

from subprocess import run, TimeoutExpired, Popen, PIPE, STDOUT, CalledProcessError, DEVNULL

import logging
import re
import time
import numpy as np
from multiprocessing import cpu_count


logger = logging.getLogger(__name__)


def get_scheduler(config):
    """
    :param config: PyBNF configuration
    :type config: pybnf.config.Configuration

    :return: scheduler node, string composed of all available nodes
    """
    scheduler_node, node_string = None, None
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


def setup_cluster(node_string, out_dir, parallel_count=None):
    """
    Sets up a Dask cluster using the `dask-ssh` convenience script

    :param node_string: A string composed of a list of compute nodes
    :param out_dir: A directory for
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


def teardown_cluster(dsp):
    """
    Terminates the process running the `dask-ssh` script after completion of fitting run

    :param dsp: subprocess.Popen
    :return:
    """
    logger.info('Closing dask-ssh subprocess')
    dsp.terminate()
