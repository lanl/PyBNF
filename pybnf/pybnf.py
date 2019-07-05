"""The entry point for the PyBNF application containing the main function and version"""


from .parse import load_config
from .config import init_logging
from .printing import print0, print1, print2, PybnfError
from .cluster import Cluster
from .pset import Trajectory
import pybnf.algorithms as algs
import pybnf.printing as printing

from subprocess import run
from numpy import inf

import logging
import argparse
import os
import shutil
import time
import traceback
import pickle


__version__ = "1.0.1"


def main():
    """The main function for running a fitting job"""
    start_time = time.time()

    success = False
    alg = None
    cluster = None

    parser = argparse.ArgumentParser(description='Performs parameter fitting on systems biology models defined in '
                                                 'BNGL or SBML. For documentation, examples, and source code, go to '
                                                 'https://github.com/lanl/PyBNF')

    parser.add_argument('-c', action='store', dest='conf_file',
                        help='Path to the BioNetFit configuration file', metavar='config.conf')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='automatically overwrites existing folders if necessary')
    parser.add_argument('-t', '--cluster_type', action='store',
                        help='optional string denoting the type of cluster')
    parser.add_argument('-s', '--scheduler_file', action='store',
                        help='optional file on shared filesystem to get scheduler location, should be same as passed to dask-scheduler and dask-worker.')
    parser.add_argument('-r', '--resume', action='store', nargs='?', const=0, default=None, type=int,
                        metavar='iterations',
                        help='automatically resume the previously stopped fitting run; '
                             'if a number is passed with this flag, add that many iterations to the fitting run.')
    parser.add_argument('-d', '--debug_logging', action='store_true',
                        help='outputs a separate debugging log (file could be very large)')
    parser.add_argument('-l', '--log_prefix', action='store',
                        help='specifies a custom prefix for the log files (will overwrite if files already exist')
    parser.add_argument('-L', '--log_level', type=str.lower, default='i',
                        choices=['debug', 'info', 'warning', 'error', 'critical', 'none', 'd', 'i', 'w', 'e', 'c', 'n'],
                        help='set the level of output to the log file. Options in decreasing order of verbosity are: '
                             'debug, info, warning, error, critical, none.')
    cmdline_args = parser.parse_args()

    if cmdline_args.log_prefix:
        log_prefix = cmdline_args.log_prefix
    else:
        log_prefix = 'bnf_%s' % time.strftime('%Y%m%d-%H%M%S')
    debug = cmdline_args.debug_logging

    # Overwrite log file if it exists
    if os.path.isfile('%s_debug.log' % log_prefix):
        os.remove('%s_debug.log' % log_prefix)
    if os.path.isfile('%s.log' % log_prefix):
        os.remove('%s.log' % log_prefix)

    init_logging(log_prefix, debug, cmdline_args.log_level)
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
            if os.path.exists(config.config['output_dir'] + '/alg_backup.bp'):
                continue_file = config.config['output_dir'] + '/alg_backup.bp'
            elif os.path.exists(config.config['output_dir'] + '/alg_finished.bp'):
                if cmdline_args.resume <= 0:
                    raise PybnfError('The fitting run saved in %s already finished. If you want to continue the '
                                     'fitting with more iterations, pass a number of iterations with the '
                                     '--resume flag.' % config.config['output_dir'])
                continue_file = config.config['output_dir'] + '/alg_finished.bp'
            else:
                raise PybnfError('No algorithm found to resume in %s' % (config.config['output_dir']))
        elif os.path.exists(config.config['output_dir'] + '/alg_backup.bp') and not cmdline_args.overwrite:
            ans = 'x'
            while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                ans = input('Your output_dir contains an in-progress run.\nContinue that run? [y/n] (y) ')
            if ans.lower() in ('y', 'yes', ''):
                logger.info('Resuming a previous run')
                continue_file = config.config['output_dir'] + '/alg_backup.bp'
                cmdline_args.resume = 0

        if continue_file:
            # Restart the loaded algorithm
            logger.info('Reloading algorithm')
            f = open(continue_file, 'rb')
            alg, pending = pickle.load(f)
            logger.debug('Loaded algorithm is the %s algorithm' % ('refinement' if alg.refine else 'configured'))
            config = alg.config

            logger.debug('Checking for Simulations directory')
            if not os.path.exists(alg.sim_dir):
                os.mkdir(alg.sim_dir)

            if alg.bootstrap_number is not None:
                print0('Resuming a bootstrapping run')
                logger.info('Resuming a bootstrapping run')
                if cmdline_args.resume > 0 and cmdline_args.resume is not None:
                    raise PybnfError("Cannot increase the number of iterations in a boostrapping run")
            else:
                print0('Resuming a fitting run')

            alg.add_iterations(cmdline_args.resume)
            f.close()
            if isinstance(alg, algs.SimplexAlgorithm):
                # The continuing alg is already on the Simplex stage, so don't restart simplex after completion
                alg.config.config['refine'] = 0
        else:
            # Create output folders, checking for overwrites.
            subdirs = ('Simulations', 'Results', 'Initialize', 'FailedSimLogs')
            subfiles = ('alg_backup.bp', 'alg_finished.bp', 'alg_refine_finished.bp')
            will_overwrite = [subdir for subdir in subdirs + subfiles
                              if os.path.exists(config.config['output_dir'] + '/' + subdir)]
            if config.config['simulation_dir']:
                simdir = config.config['simulation_dir'] + '/Simulations'
                if os.path.exists(simdir):
                    will_overwrite.append(simdir)
            if len(will_overwrite) > 0:
                if not cmdline_args.overwrite:
                    logger.info("Output directory already exists... querying user for overwrite permission")
                    ans = 'x'
                    while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                        print0('Your output directory contains files from a previous run: %s.' % ', '.join(will_overwrite))
                        ans = input(
                            'Overwrite them with the current run? [y/n] (n) ')
                    if not(ans.lower() == 'y' or ans.lower() == 'yes'):
                        logger.info("Overwrite rejected... exiting")
                        print0('Quitting')
                        exit(0)
                # If we get here, safe to overwrite files
                for subdir in subdirs:
                    try:
                        shutil.rmtree(config.config['output_dir'] + '/' + subdir)
                        logger.info('Deleted old directory %s' % config.config['output_dir'] + '/' + subdir)
                    except OSError:
                        logger.debug('Directory %s does not already exist' % config.config['output_dir'] + '/' + subdir)
                for subfile in subfiles:
                    try:
                        os.remove(config.config['output_dir'] + '/' + subfile)
                        logger.info('Deleted old file %s' % config.config['output_dir'] + '/' + subfile)
                    except OSError:
                        logger.debug('File %s does not already exist' % config.config['output_dir'] + '/' + subfile)
                if config.config['simulation_dir']:
                    try:
                        shutil.rmtree(config.config['simulation_dir']+'/Simulations')
                        logger.info('Deleted old simulation directory %s' %
                                    config.config['simulation_dir']+'/Simulations')
                    except OSError:
                        logger.debug('Simulation directory %s does not already exist' %
                                     config.config['simulation_dir']+'/Simulations')


            # Create new directories for the current run.
            os.makedirs(config.config['output_dir'] + '/Results')
            if config.config['simulation_dir']:
                os.makedirs(config.config['simulation_dir'] + '/Simulations')
            else:
                os.mkdir(config.config['output_dir'] + '/Simulations')
            shutil.copy(cmdline_args.conf_file, config.config['output_dir'] + '/Results')
            pending = None
    
            if config.config['fit_type'] == 'pso':
                alg = algs.ParticleSwarm(config)
            elif config.config['fit_type'] == 'de':
                alg = algs.DifferentialEvolution(config)
            elif config.config['fit_type'] == 'ss':
                alg = algs.ScatterSearch(config)
            elif config.config['fit_type'] == 'mh' or config.config['fit_type'] == 'pt':
                # Note: mh vs pt difference is handled in Config by setting or not setting the exchange_every key.
                alg = algs.BasicBayesMCMCAlgorithm(config)
            elif config.config['fit_type'] == 'sa':
                alg = algs.BasicBayesMCMCAlgorithm(config, sa=True)
            elif config.config['fit_type'] == 'sim':
                alg = algs.SimplexAlgorithm(config)
            elif config.config['fit_type'] == 'ade':
                alg = algs.AsynchronousDifferentialEvolution(config)
            elif config.config['fit_type'] == 'dream':
                alg = algs.DreamAlgorithm(config)
            elif config.config['fit_type'] == 'check':
                alg = algs.ModelCheck(config)
            else:
                raise PybnfError('Invalid fit_type %s. Options are: pso, de, ade, ss, mh, pt, sa, sim, check' % config.config['fit_type'])

        # Override configuration values if provided on command line
        if cmdline_args.cluster_type:
            config.config['cluster_type'] = cmdline_args.cluster_type
        if cmdline_args.scheduler_file:
            config.config['scheduler_file'] = cmdline_args.scheduler_file

        if config.config['fit_type'] != 'check':
            # Set up cluster
            cluster = Cluster(config, log_prefix, debug, cmdline_args.log_level)
            # Run the algorithm!
            logger.debug('Algorithm initialization')
            alg.run(cluster.client, resume=pending, debug=debug)
        else:
            # Run model checking
            logger.debug('Model checking initialization')
            alg.run_check(debug=debug)

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
                simplex = algs.SimplexAlgorithm(config, refine=True)
                simplex.trajectory = alg.trajectory  # Reuse existing trajectory; don't start a new one.
                simplex.run(cluster.client, debug=debug)

        if alg.bootstrap_number is None:
            print0('Fitting complete')

        # Bootstrapping (optional)
        if config.config['bootstrap'] > 0:

            # Bootstrapping setup
            if config.config['bootstrap_max_obj']:
                bootstrap_max_obj = config.config['bootstrap_max_obj']
            else:
                bootstrap_max_obj = inf
                logger.info('No bootstrap_max_obj specified; set to infinity')
                print1('No bootstrap_max_obj specified. All bootstrap replicates will be accepted regardless of '
                       'objective value.')

            num_to_bootstrap = config.config['bootstrap']
            completed_bootstrap_runs = 0
            if alg.bootstrap_number is None:
                bootstrapped_psets = Trajectory(num_to_bootstrap)
            else:  # Check if finished a resumed bootstrap fitting run
                completed_bootstrap_runs += alg.bootstrap_number
                if completed_bootstrap_runs == 0:
                    bootstrapped_psets = Trajectory(num_to_bootstrap)
                else:
                    if completed_bootstrap_runs > 0:
                        bootstrapped_psets = Trajectory.load_trajectory(config.config['output_dir'] +
                                                                        '/Results/bootstrapped_parameter_sets.txt',
                                                                        config.variables,
                                                                        config.config['num_to_output'])

                if alg.best_fit_obj <= bootstrap_max_obj:
                    logger.info('Bootstrap run %s complete' % completed_bootstrap_runs)
                    bootstrapped_psets.add(alg.trajectory.best_fit(), alg.best_fit_obj,
                                           'bootstrap_run_%s' % completed_bootstrap_runs,
                                           config.config['output_dir'] + '/Results/bootstrapped_parameter_sets.txt',
                                           completed_bootstrap_runs == 0)
                    logger.info('Succesfully completed resumed bootstrapping run %s' % completed_bootstrap_runs)
                    completed_bootstrap_runs += 1
                else:
                    shutil.rmtree(alg.res_dir)
                    if os.path.exists(alg.sim_dir):
                        shutil.rmtree(alg.sim_dir)
                    print0("Bootstrap run did not achieve maximum allowable objective function value.  Retrying")
                    logger.info('Resumed bootstrapping run %s did not achieve maximum allowable objective function '
                                'value.  Retrying' % completed_bootstrap_runs)

            # Run bootstrapping
            consec_failed_bootstrap_runs = 0
            while completed_bootstrap_runs < num_to_bootstrap:
                alg.reset(bootstrap=completed_bootstrap_runs)

                for model in alg.exp_data:
                    for name, data in alg.exp_data[model].items():
                        data.gen_bootstrap_weights()
                        data.weights_to_file('%s/%s_weights_%s.txt' % (alg.res_dir, name, completed_bootstrap_runs))

                logger.info('Beginning bootstrap run %s' % completed_bootstrap_runs)
                print0("Beginning bootstrap run %s" % completed_bootstrap_runs)
                alg.run(cluster.client, debug=debug)

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
                        simplex = algs.SimplexAlgorithm(config, refine=True)
                        simplex.trajectory = alg.trajectory  # Reuse existing trajectory; don't start a new one.
                        simplex.run(cluster.client, debug=debug)

                best_fit_pset = alg.trajectory.best_fit()

                if alg.best_fit_obj <= bootstrap_max_obj:
                    logger.info('Bootstrap run %s complete' % completed_bootstrap_runs)
                    bootstrapped_psets.add(best_fit_pset, alg.best_fit_obj, 'bootstrap_run_%s' % completed_bootstrap_runs,
                                           config.config['output_dir'] + '/Results/bootstrapped_parameter_sets.txt',
                                           completed_bootstrap_runs == 0)
                    completed_bootstrap_runs += 1
                    consec_failed_bootstrap_runs = 0
                else:
                    consec_failed_bootstrap_runs += 1
                    print0("Bootstrap run did not achieve maximum allowable objective function value.  Retrying")
                    logger.warning("Bootstrap run did not achieve maximum allowable objective function value.")
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
        if cluster:
            try:
                cluster.teardown()
                if not cluster.local:
                    time.sleep(10)  # wait for teardown before continuing
            except Exception:
                logging.exception('Failed to tear down cluster')
        else:
            logging.info('No cluster to tear down')

        # Attempt to remove dask-worker-space directory if necessary
        # (exists in directory where workers were instantiated)
        # Tries current and home directories
        if os.path.isdir('dask-worker-space'):
            if os.name == 'nt':  # Windows
                shutil.rmtree('dask-worker-space', ignore_errors=True)
            else:
                run(['rm', '-rf', 'dask-worker-space'])  # More likely to succeed than rmtree()
        home_dask_dir = os.path.expanduser(os.path.join('~', 'dask-worker-space'))
        if os.path.isdir(home_dask_dir):
            if os.name == 'nt':  # Windows
                shutil.rmtree(home_dask_dir, ignore_errors=True)
            else:
                run(['rm', '-rf', home_dask_dir])

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
            end_time = time.time()
            secs = end_time - start_time
            mins, secs = divmod(secs, 60)
            hrs, mins = divmod(mins, 60)
            print2('Total fitting time: %d:%02d:%02d' % (hrs, mins, secs))
            logger.info('Total fitting time: %d:%02d:%02d' % (hrs, mins, secs))
            exit(0 if success else 1)
