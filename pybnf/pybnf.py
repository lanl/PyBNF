"""pybnf.pybnf: defines the entry point for the PyBNF application"""


from .parse import load_config
from .config import init_logging
from .printing import print0, print1, PybnfError
from .cluster import get_scheduler, setup_cluster, teardown_cluster
from .pset import Trajectory
import pybnf.algorithms as algs
import pybnf.printing as printing

from subprocess import run

import logging
import argparse
import os
import shutil
import time
import traceback
import pickle


__version__ = "0.1"


def main():

    success = False
    node_string = None
    alg = None

    parser = argparse.ArgumentParser(description='Performs parameter fitting on models defined in BNGL')

    parser.add_argument('-c', action='store', dest='conf_file',
                        help='Path to the BioNetFit configuration file', metavar='config.conf')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='automatically overwrites existing folders if necessary')
    parser.add_argument('-t', '--cluster_type', action='store',
                        help='optional string denoting the type of cluster')
    parser.add_argument('-r', '--resume', action='store', nargs='?', const=0, default=None, type=int,
                        metavar='iterations',
                        help='automatically resume the previously stopped fitting run; '
                             'if a number is passed with this flag, add that many iterations to the fitting run.')
    parser.add_argument('-d', '--debug_logging', action='store_true',
                        help='outputs debugging log (file could be very large)')
    parser.add_argument('-l', '--log_prefix', action='store',
                        help='specifies a custom prefix for the log files (will overwrite if files already exist')
    cmdline_args = parser.parse_args()

    if cmdline_args.log_prefix:
        log_prefix = cmdline_args.log_prefix
    else:
        log_prefix = 'bnf_%s' % time.strftime('%Y%m%d-%H%M%S')

    # Overwrite log file if it exists
    if os.path.isfile('%s_debug.log' % log_prefix):
        os.remove('%s_debug.log' % log_prefix)
    if os.path.isfile('%s.log' % log_prefix):
        os.remove('%s.log' % log_prefix)

    init_logging(log_prefix, cmdline_args.debug_logging)
    logger = logging.getLogger(__name__)

    print0("PyBNF v%s" % __version__)
    logger.info('Running PyBNF v%s' % __version__)

    try:
        # Load the conf file and create the algorithm
        if cmdline_args.conf_file is None:
            print0('No configuration file given, so I won''t do anything.\nFor more information, try pybnf --help')
            exit(0)
        logger.info('Loading configuration file: %s' % cmdline_args.conf_file)

        config = load_config(cmdline_args.conf_file)
        if 'verbosity' in config.config:
            printing.verbosity = config.config['verbosity']

        if cmdline_args.resume is not None and cmdline_args.overwrite:
            raise PybnfError("Options --overwrite and --resume are contradictory. Use --resume to continue a previous "
                             "run, or --overwrite to overwrite the previous run with a new one.")

        continue_file = None
        if cmdline_args.resume is not None:
            if os.path.exists(config.config['output_dir'] + '/Simulations/alg_backup.bp'):
                continue_file = config.config['output_dir'] + '/Simulations/alg_backup.bp'
            elif os.path.exists(config.config['output_dir'] + '/Results/alg_finished.bp'):
                if cmdline_args.resume <= 0:
                    raise PybnfError('The fitting run saved in %s already finished. If you want to continue the '
                                     'fitting with more iterations, pass a number of iterations with the '
                                     '--resume flag.' % config.config['output_dir'])
                continue_file = config.config['output_dir'] + '/Results/alg_finished.bp'
            else:
                raise PybnfError('No algorithm found to resume in %s' % (config.config['output_dir']))
        elif os.path.exists(config.config['output_dir'] + '/Simulations/alg_backup.bp') and not cmdline_args.overwrite:
            ans = 'x'
            while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                ans = input('Your output_dir contains an in-progress run.\nContinue that run? [y/n] (y) ')
            if ans.lower() in ('y', 'yes', ''):
                logger.info('Resuming a previous run')
                continue_file = config.config['output_dir'] + '/Simulations/alg_backup.bp'
                cmdline_args.resume = 0

        if continue_file:
            # TODO check if resuming bootstrapping
            # Restart the loaded algorithm
            logger.info('Reloading algorithm')
            f = open(continue_file, 'rb')
            alg, pending = pickle.load(f)
            config = alg.config
            alg.add_iterations(cmdline_args.resume)
            f.close()
            if isinstance(alg, algs.SimplexAlgorithm):
                # The continuing alg is already on the Simplex stage, so don't restart simplex after completion
                alg.config.config['refine'] = 0
        else:
            # Create output folders, checking for overwrites.
            if os.path.exists(config.config['output_dir']):
                if cmdline_args.overwrite:
                    logger.info('Overwriting existing output directory')
                    shutil.rmtree(config.config['output_dir'])
                else:
                    logger.info("Output directory already exists... querying user for overwrite permission")
                    ans = 'x'
                    while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                        ans = input(
                            'It looks like you may already have results in the specified output directory.  '
                            'Overwrite? [y/n] (n) ')
                    if ans.lower() == 'y' or ans.lower() == 'yes':
                        logger.info('Overwriting existing output directory')
                        shutil.rmtree(config.config['output_dir'])
                    else:
                        logger.info("Overwrite rejected... exiting")
                        print('Quitting')
                        exit(0)

            os.makedirs(config.config['output_dir'] + '/Results')
            os.mkdir(config.config['output_dir'] + '/Simulations')
            shutil.copy(cmdline_args.conf_file, config.config['output_dir'] + '/Results')
            pending = None
    
            if config.config['fit_type'] == 'pso':
                alg = algs.ParticleSwarm(config)
            elif config.config['fit_type'] == 'de':
                alg = algs.DifferentialEvolution(config)
            elif config.config['fit_type'] == 'ss':
                alg = algs.ScatterSearch(config)
            elif config.config['fit_type'] == 'bmc' or config.config['fit_type'] == 'pt':
                # Note: bmc vs pt difference is handled in Config by setting or not setting the exchange_every key.
                alg = algs.BayesAlgorithm(config)
            elif config.config['fit_type'] == 'sa':
                alg = algs.BayesAlgorithm(config, sa=True)
            elif config.config['fit_type'] == 'sim':
                alg = algs.SimplexAlgorithm(config)
            else:
                raise PybnfError('Invalid fit_type %s. Options are: pso, de, ss, bmc, pt, sa, sim' % config.config['fit_type'])

        # override cluster type value in configuration file if specified with cmdline args
        if cmdline_args.cluster_type:
            config.config['cluster_type'] = cmdline_args.cluster_type

        # Set up cluster
        if config.config['scheduler_node'] and config.config['worker_nodes']:
            scheduler_node = config.config['scheduler_node']
            node_string = ' '.join(config.config['worker_nodes'])
        elif config.config['scheduler_node']:
            dummy, node_string = get_scheduler(config)
            scheduler_node = config.config['scheduler_node']
        else:
            scheduler_node, node_string = get_scheduler(config)

        if node_string:
            dask_ssh_proc = setup_cluster(node_string, os.getcwd())

        # Run the algorithm!
        logger.debug('Algorithm initialization')
        alg.run(log_prefix, scheduler_node, resume=pending, debug=cmdline_args.debug_logging)

        if config.config['refine'] == 1:
            logger.debug('Refinement requested for best fit parameter set')
            if config.config['fit_type'] == 'sim':
                logger.debug('Cannot refine further if Simplex algorithm was used for original fit')
                print1("You specified refine=1, but refine uses the Simplex algorithm, which you already just ran."
                      "\nSkipping refine.")
            else:
                logger.debug('Refining further using the Simplex algorithm')
                print1("Refining the best fit by the Simplex algorithm")
                config.config['simplex_start_point'] = alg.trajectory.best_fit()
                simplex = algs.SimplexAlgorithm(config)
                simplex.trajectory = alg.trajectory  # Reuse existing trajectory; don't start a new one.
                simplex.run(log_prefix, scheduler_node)

        print0('Fitting complete')

        # Bootstrapping (optional)
        if config.config['bootstrap'] > 0:

            if config.config['bootstrap_max_obj']:
                bootstrap_max_obj = config.config['bootstrap_max_obj']
            else:
                bootstrap_max_obj = alg.trajectory.trajectory[alg.trajectory.best_fit()]

            num_to_bootstrap = config.config['bootstrap']
            bootstrapped_psets = Trajectory(num_to_bootstrap)
            completed_bootstrap_runs = 0
            consec_failed_bootstrap_runs = 0
            while completed_bootstrap_runs < num_to_bootstrap:
                alg.reset(bootstrap=completed_bootstrap_runs)

                for name, data in alg.exp_data.items():
                    data.gen_bootstrap_weights()
                    data.weights_to_file('%s/%s_weights_%s.txt' % (alg.res_dir, name, completed_bootstrap_runs))

                logger.info('Beginning bootstrap run %s' % completed_bootstrap_runs)
                print0("Beginning bootstrap run %s" % completed_bootstrap_runs)
                alg.run(log_prefix, scheduler_node, resume=pending, debug=cmdline_args.debug_logging)

                if config.config['refine'] == 1:
                    logger.debug('Refinement requested for best fit parameter set')
                    if config.config['fit_type'] == 'sim':
                        logger.debug('Cannot refine further if Simplex algorithm was used for original fit')
                        print1("You specified refine=1, but refine uses the Simplex algorithm, which you already just ran."
                              "\nSkipping refine.")
                    else:
                        logger.debug('Refining further using the Simplex algorithm')
                        print1("Refining the best fit by the Simplex algorithm")
                        config.config['simplex_start_point'] = alg.trajectory.best_fit()
                        simplex = algs.SimplexAlgorithm(config)
                        simplex.trajectory = alg.trajectory  # Reuse existing trajectory; don't start a new one.
                        simplex.run(log_prefix, scheduler_node)

                best_fit_pset = alg.trajectory.best_fit()
                best_fit_obj = alg.trajectory.trajectory[best_fit_pset]

                if best_fit_obj <= bootstrap_max_obj:
                    logger.info('Bootstrap run %s complete' % completed_bootstrap_runs)
                    bootstrapped_psets.add(best_fit_pset, best_fit_obj, 'bootstrap_run_%s' % completed_bootstrap_runs,
                                           config.config['output_dir'] + '/Results/bootstrapped_parameter_sets.txt',
                                           completed_bootstrap_runs == 0)
                    completed_bootstrap_runs += 1
                    consec_failed_bootstrap_runs = 0
                else:
                    consec_failed_bootstrap_runs += 1
                    print0("Bootstrap run did not achieve maximum allowable objective function value.  Retrying")
                    logger.warning("Bootstrap run did not achieve maximum allowable objective function value.")
                    shutil.rmtree(alg.res_dir)
                    if os.path.exists(alg.sim_dir):
                        shutil.rmtree(alg.sim_dir)
                    if consec_failed_bootstrap_runs > 20:  # Arbitrary...  should we make this configurable or smaller?
                        raise PybnfError("20 consecutive bootstrap runs failed to achieve maximum allowable objective "
                                         "function values.  Check 'bootstrap_max_obj' configuration key")

            # bootstrapped_psets.write_to_file(config.config['output_dir'] + "/Results/bootstrapped_parameter_sets.txt")
            print0('Bootstrapping complete')

        success = True

    except PybnfError as e:
        # Exceptions generated by problems such as bad user input should be caught here and print a useful message
        # before quitting
        logger.error('Terminating due to a PybnfError:')
        logger.error(e.log_message)
        print0('Error: %s' % e.message)
    except KeyboardInterrupt:
        print0('Fitting aborted.')
        logger.info('Terminating due to keyboard interrupt')
        logger.exception('Keyboard interrupt')
    except Exception:
        # Sends any unhandled errors to log instead of to user output
        logger.exception('Internal error')
        exceptiondata = traceback.format_exc().splitlines()
        print0('Sorry, an unknown error occurred: %s\n'
               'Logs have been saved to %s.log.\n'
               'Please report this bug to help us improve PyBNF.' % (exceptiondata[-1], log_prefix))
    finally:
        # Stop dask-ssh regardless of success
        if node_string:
            teardown_cluster(dask_ssh_proc)
            time.sleep(10)  # wait for teardown before continuing

        # Attempt to remove dask-worker-space directory if necessary
        # (exists in directory where workers were instantiated)
        # Tries current and home directories
        if os.path.isdir('dask-worker-space'):
            run(['rm', '-rf', 'dask-worker-space'])
        if os.path.isdir(os.environ['HOME'] + '/dask-worker-space'):
            run(['rm', '-rf', os.environ['HOME'] + '/dask-worker-space'])

        # After any error, try to clean up.
        try:
            if not success:
                logger.info('Fitting unsuccessful.  Attempting cleanup')
                if alg:
                    alg.cleanup()
                    logger.info('Completed cleanup after exception')
        except:
            logger.exception('During cleanup, another exception occurred')
        finally:
            exit(0 if success else 1)
