"""Functions for managing dask cluster setup and teardown on distributed computing systems"""


from .printing import PybnfError

from subprocess import Popen,  STDOUT, DEVNULL

import logging
import re
import time
from itertools import product
import copy
import os

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
            try:
                nodeecho = os.environ['SLURM_JOB_NODELIST']
            except KeyError:
                raise PybnfError('SLURM cluster setup failed: Enviornment variable $SLURM_JOB_NODELIST was not found')
            nodes = parse_nodes(nodeecho)
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

def parse_nodes(slurm_string):
    """
    Converts a SLURM_JOB_NODELIST into the list of nodes represented
    :param slurm_string:
    :return:
    """

    # Replace all bracketed things with placeholders *1*, *2*, etc
    index = 1
    brackets = dict()
    while True:
        m = re.search('(?<=\[)[^\[\]]*(?=\])', slurm_string)
        if m is None:
            break
        slurm_string = re.sub('\[[^\[\]]*\]', '*%s*' % index, slurm_string, count=1)
        brackets[index] = m.group(0)
        index += 1

    # Expand the bracketed substrings into all the numbers they represent
    bracket_nums = dict()
    for k in brackets:
        s = brackets[k]
        numlist = []
        nums = s.split(',')
        for ni in nums:
            if '-' in ni:
                nrange = ni.split('-')
                numlist += list(range(int(nrange[0]), int(nrange[1])+1))
            else:
                numlist.append(int(ni))
        bracket_nums[k] = numlist

    # Split the outside string by commas, then generate the node names for each part
    final_list = []
    parts = slurm_string.split(',')
    for p in parts:
        tosub = re.findall('\*[0-9]+\*', p)
        tosub = [int(i[1:-1]) for i in tosub]
        sublists = [bracket_nums[i] for i in tosub]
        for subs in product(*sublists):
            thisp = copy.copy(p)
            for i, si in enumerate(subs):
                thisp = re.sub('\*%i\*' % tosub[i], str(si), thisp)
            final_list.append(thisp)
    return final_list

def setup_cluster(node_string, out_dir):
    """
    Sets up a Dask cluster using the `dask-ssh` convenience script

    :param node_string: A string composed of a list of compute nodes
    :param out_dir: A directory for
    :return: subprocess.Popen
    """
    logger.info('Starting dask-ssh subprocess using nodes %s' % node_string)
    dask_ssh_proc = Popen('dask-ssh %s --log-directory %s' % (node_string, out_dir), shell=True, stdout=DEVNULL, stderr=STDOUT)
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
