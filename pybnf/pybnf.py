"""pybnf.pybnf: defines the entry point for the PyBNF application"""


import logging
import argparse
from .parse import load_config
from .config import Configuration
import pybnf.algorithms as algs
import os
import shutil
__version__ = "0.1"


def main():
    log_format = "%(levelname)s\t%(message)s"
    logging.basicConfig(format=log_format, level=logging.DEBUG)
            
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
        alg = algs.DifferentialEvolution(config)
    elif conf_dict['fit_type'] == 'ss':
        alg = algs.ScatterSearch(config)
    elif conf_dict['fit_type'] == 'bmc':
        alg = algs.BayesAlgorithm(config)
    else:
        raise ValueError('Invalid fit_type %s. Options are: pso, de, ss, bmc' % conf_dict['fit_type'])

    # Create output folders, checking for overwrites.
    if os.path.exists(config.config['output_dir']):
        if os.path.isdir(config.config['output_dir']):
            if os.path.exists(config.config['output_dir']+'/Results') or os.path.exists(
                            config.config['output_dir']+'/Simulations'):

                ans = 'x'
                while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                    ans = input('It looks like your output_dir already contains Results/ and/or Simulations/ folders '
                                'from a previous run. Overwrite them with the current run? [y/n] (n) ')
                if ans.lower() == 'y' or ans.lower == 'yes':
                    if os.path.exists(config.config['output_dir'] + '/Results'):
                        shutil.rmtree(config.config['output_dir'] + '/Results')
                    if os.path.exists(config.config['output_dir'] + '/Simulations'):
                        shutil.rmtree(config.config['output_dir'] + '/Simulations')
                else:
                    print('Quitting')
                    return

        else:
            logging.error('Your specified output_dir already exists as an ordinary file. Please choose a different '
                          'name.')
            return
    os.makedirs(config.config['output_dir']+'/Results')
    os.mkdir(config.config['output_dir']+'/Simulations')
    shutil.copy(results.conf_file, config.config['output_dir']+'/Results')

    # Run the algorithm!
    alg.run()