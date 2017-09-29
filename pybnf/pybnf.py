"""pybnf.pybnf: defines the entry point for the PyBNF application"""


import logging
import argparse
from .parse import load_config
from .config import Configuration
import pybnf.algorithms as algs

__version__ = "0.1"


def main():
    log_format = "%(levelname)s\t%(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
            
    logging.info("PyBNF v%s" % __version__)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', action='store', dest='conf_file',
    help='Path to the BioNetFit configuration file', metavar='config.txt')

    # Load the conf file and create the algorithm
    results = parser.parse_args()
    conf_dict = load_config(results.conf_file)
    config = Configuration(conf_dict)
    if conf_dict['fit_type'] == 'pso':
        alg = algs.ParticleSwarm(config)
    elif conf_dict['fit_type'] == 'de':
        raise NotImplementedError('Differential evolution is not written yet')
    else:
        raise ValueError('Invalid fit_type %s. Options are: pso, de' % conf_dict['fit_type'])

    # Run the algorithm!
    alg.run()