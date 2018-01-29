"""pybnf.pybnf: defines the entry point for the PyBNF application"""

import logging
import argparse
from .parse import load_config
from .config import Configuration
import pybnf.printing as printing
from .printing import print0, print1, print2, PybnfError
import pybnf.algorithms as algs
import os
import shutil
import traceback

__version__ = "0.1"


def main():
    try:
        log_format = "%(asctime)-15s\t%(levelname)s\t%(message)s"
        logging.basicConfig(format=log_format, level=logging.DEBUG, filename='bnf.log', filemode='w')

        print0("PyBNF v%s" % __version__)
        logging.info('Running PyBNF v%s' % __version__)

        parser = argparse.ArgumentParser(description='Performs parameter fitting on models defined in BNGL')

        parser.add_argument('-c', action='store', dest='conf_file',
                            help='Path to the BioNetFit configuration file', metavar='config.conf')

        # Load the conf file and create the algorithm
        results = parser.parse_args()
        if results.conf_file is None:
            print0('No configuration file given, so I won''t do anything.\nFor more information, try pybnf --help')
            exit(0)

        logging.info('Loading configuration file: %s' % results.conf_file)
        conf_dict = load_config(results.conf_file)
        if 'verbosity' in conf_dict:
            printing.verbosity = conf_dict['verbosity']

        config = Configuration(conf_dict)
        if conf_dict['fit_type'] == 'pso':
            alg = algs.ParticleSwarm(config)
        elif conf_dict['fit_type'] == 'de':
            alg = algs.DifferentialEvolution(config)
        elif conf_dict['fit_type'] == 'ss':
            alg = algs.ScatterSearch(config)
        elif conf_dict['fit_type'] == 'bmc':
            alg = algs.BayesAlgorithm(config)
        elif conf_dict['fit_type'] == 'sim':
            alg = algs.SimplexAlgorithm(config)
        else:
            raise PybnfError('Invalid fit_type %s. Options are: pso, de, ss, bmc, sim' % conf_dict['fit_type'])

        # Create output folders, checking for overwrites.
        if os.path.exists(config.config['output_dir']):
            if os.path.isdir(config.config['output_dir']):
                if os.path.exists(config.config['output_dir'] + '/Results') or os.path.exists(
                                config.config['output_dir'] + '/Simulations'):
                    logging.info("Output directory has subdirectories... querying user for overwrite permission")
                    ans = 'x'
                    while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                        ans = input('It looks like your output_dir already contains Results/ and/or Simulations/ folders '
                                    'from a previous run. Overwrite them with the current run? [y/n] (n) ')
                    if ans.lower() == 'y' or ans.lower == 'yes':
                        logging.info("Overwriting existing output directory")
                        if os.path.exists(config.config['output_dir'] + '/Results'):
                            shutil.rmtree(config.config['output_dir'] + '/Results')
                        if os.path.exists(config.config['output_dir'] + '/Simulations'):
                            shutil.rmtree(config.config['output_dir'] + '/Simulations')
                    else:
                        logging.info("Overwrite rejected... exiting")
                        print0('Quitting')
                        exit()
            else:
                logging.info("Requested output directory exists as an ordinary file... exiting")
                print0('Your specified output_dir already exists as an ordinary file. Please choose a '
                                'different name.')
                exit()

        os.makedirs(config.config['output_dir'] + '/Results')
        os.mkdir(config.config['output_dir'] + '/Simulations')
        shutil.copy(results.conf_file, config.config['output_dir'] + '/Results')

        # Run the algorithm!
        logging.debug('Algorithm initialization')
        alg.run()

        if config.config['refine'] == 1:
            logging.debug('Refinement requested for best fit parameter set')
            if config.config['fit_type'] == 'sim':
                logging.debug('Cannot refine further if Simplex algorithm was used for original fit')
                print1("You specified refine=1, but refine uses the Simplex algorithm, which you already just ran."
                      "\nSkipping refine.")
            else:
                logging.debug('Refining further using the Simplex algorithm')
                print1("Refining the best fit by the Simplex algorithm")
                config.config['simplex_start_point'] = alg.trajectory.best_fit()
                simplex = algs.SimplexAlgorithm(config)
                simplex.trajectory = alg.trajectory  # Reuse existing trajectory; don't start a new one.
                simplex.run()
        print0('Fitting complete')

    except PybnfError as e:
        # Exceptions generated by problems such as bad user input should be caught here and print a useful message
        # before quitting
        logging.error('Terminating due to a PybnfError:')
        logging.error(e.log_message)
        print0('Error: %s' % e.message)
        exit(1)
    except Exception:
        # Sends any unhandled errors to log instead of to user output
        logging.exception('Internal error')
        exceptiondata = traceback.format_exc().splitlines()
        print0('Sorry, an unknown error occurred: %s\n'
               'Details have been saved to bnf.log.\n'
               'Please report this bug to help us improve PyBNF.' % exceptiondata[-1])
        exit(1)
