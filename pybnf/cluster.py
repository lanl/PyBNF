"""pybnf.cluster: functions for managing dask cluster setup and teardown"""


from .printing import PybnfError

from subprocess import run, TimeoutExpired, Popen, PIPE, DEVNULL, STDOUT

import logging
import re
import time


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
            logging.debug('Detected selection of SLURM cluster')
            get_hosts_cmd = ['scontrol', 'show', 'hostname', '$SLURM_JOB_NODELIST']
            try:
                proc = run(' '.join(get_hosts_cmd), shell=True, stdout=PIPE, timeout=10)
            except TimeoutExpired:
                logging.debug('Could not retrieve host names in 10s')
                raise PybnfError('Failed to find node names.  Exiting')
            nodes = re.split('\n', proc.stdout.decode('UTF-8').strip())
            scheduler_node = nodes[0]
            logging.info('Node %s is being used as the scheduler node' % scheduler_node)
            logging.info('Node(s) %s is/are being used as compute nodes' % ','.join(nodes))
            node_string = ' '.join(nodes)
        elif re.match('((torque)|(pbs))', ctype, flags=re.IGNORECASE):
            raise PybnfError("TORQUE cluster support is not yet implemented")
        else:
            logging.error("Unknown cluster type: %s" % config.config['cluster_type'])
            raise PybnfError("Unknown cluster type: %s" % config.config['cluster_type'])
    return scheduler_node, node_string


def setup_cluster(node_string):
    logging.info('Starting dask-ssh subprocess using nodes %s' % node_string)
    dask_ssh_proc = Popen('dask-ssh %s' % node_string, shell=True, stdout=DEVNULL, stderr=STDOUT)
    time.sleep(10)
    return dask_ssh_proc


def teardown_cluster(dsp):
    logging.info('Closing dask-ssh subprocess')
    dsp.terminate()
