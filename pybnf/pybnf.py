"""The entry point for the PyBNF application containing the main function and version"""


from . import __version__
from . import _bngsim_caps
from . import method_chain
from .budget import FitBudget, format_duration, spend_reserve
from .parse import load_config
from .config import init_logging
from .printing import print0, print1, print2, PybnfError
from .cluster import Cluster
from .pset import Trajectory, set_sim_registry, reap_active_sims, clear_sim_registry
from .registry import FIT_TYPE_REGISTRY
import pybnf.algorithms as algs
import pybnf.printing as printing

from subprocess import run
from numpy import inf
import numpy as np

import logging
import argparse
import os
import shutil
import time
import traceback
import pickle
from pathlib import Path


def _report_bngsim_build():
    """Name the compiled bngsim core this run loaded, and warn if it is stale (#558).

    An editable/from-source bngsim serves live Python from the source tree while
    loading its compiled core from a separately-built ``.so``, with auto-rebuild
    deliberately off, so the two halves drift. bngsim detects that and warns -- but
    it warns at *import*, which for PyBNF is while the ``pybnf`` package is loading:
    before ``init_logging``, before the config is read, and long before the user has
    decided anything. The warning scrolls past in import noise and the fit proceeds
    on numbers produced by C++ that is no longer in the tree.

    Repeating it here puts it at the top of the run's own log and console output,
    where a reader still has the option to stop. The identity line goes to the log
    unconditionally -- it carries the build commit, which is the only thing that
    tells two installs declaring the same version apart -- and the staleness report
    is promoted to a warning at verbosity 0, because it invalidates every capability
    answer and every number that follows it.

    Never raises: every read is guarded inside ``_bngsim_caps``, so an install
    that cannot answer reports no opinion rather than taking the fit down.
    """
    logger = logging.getLogger(__name__)
    identity = _bngsim_caps.bngsim_identity_line()
    if identity:
        logger.info(identity)
    stale = _bngsim_caps.bngsim_stale_core_report()
    if stale:
        logger.warning('The installed bngsim\'s compiled core is older than its own '
                       'C++ sources; every number this fit produces describes that '
                       'older code.\n%s', stale)
        print0("WARNING: the installed bngsim's compiled core is older than its own C++ "
               "sources, so every number this fit produces describes that older code. "
               "Rebuild bngsim before trusting this run.")
        print1(stale)


def _initialize_random_seed(config):
    """
    Resolve and log the run's random seed.

    If the config does not specify a seed, one is chosen from system entropy so it
    can be reused later for reproducibility. The resolved seed is written back into
    the config; each algorithm builds its own ``np.random.Generator`` (default_rng)
    from it in ``Algorithm.__init__``. NumPy's legacy global RNG is no longer seeded
    or used.
    """
    seed = config.config['random_seed']
    if seed is None:
        # Draw a fresh seed from OS entropy (kept within the config's allowed
        # [0, 2**32) range) without touching the legacy global RNG.
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    config.config['random_seed'] = seed
    logger = logging.getLogger(__name__)
    logger.info('Random seed: %d', seed)
    return seed


def _finalize(success, alg, start_time):
    """Run best-effort post-fit cleanup, report the total run time, then exit.

    Called from ``main()``'s outer ``finally``. A failure inside
    ``alg.cleanup()`` is logged and swallowed so it can't hide the real run
    outcome -- but only if it is an ordinary ``Exception``. A
    ``KeyboardInterrupt``/``SystemExit`` raised during cleanup is allowed to
    propagate instead of being masked: ``exit()`` lives here as a plain
    statement, not inside a ``finally``, so an in-flight exception is no longer
    swallowed by it.
    """
    logger = logging.getLogger(__name__)
    try:
        if not success:
            logger.info('Fitting unsuccessful.  Attempting cleanup')
            if alg:
                alg.cleanup()
                logger.info('Completed cleanup after exception')
    except Exception:
        logger.exception('During cleanup, another exception occurred')

    # Remove this run's (now-empty) live-sim registry directory.
    clear_sim_registry()

    secs = time.time() - start_time
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    print2('Total fitting time: %d:%02d:%02d' % (hrs, mins, secs))
    logger.info('Total fitting time: %d:%02d:%02d' % (hrs, mins, secs))
    exit(0 if success else 1)


def _build_arg_parser():
    """Construct the command-line argument parser for the fitting application."""
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
    return parser


def _setup_logging(cmdline_args):
    """Resolve the log prefix, clear any stale log files, and initialize logging.

    Returns the resolved ``(log_prefix, debug)`` pair.
    """
    if cmdline_args.log_prefix:
        log_prefix = cmdline_args.log_prefix
    else:
        log_prefix = 'bnf_{}'.format(time.strftime('%Y%m%d-%H%M%S'))
    debug = cmdline_args.debug_logging

    # Overwrite log file if it exists
    if os.path.isfile(f'{log_prefix}_debug.log'):
        os.remove(f'{log_prefix}_debug.log')
    if os.path.isfile(f'{log_prefix}.log'):
        os.remove(f'{log_prefix}.log')

    init_logging(log_prefix, debug, cmdline_args.log_level)
    return log_prefix, debug


def _resolve_continue_file(config, cmdline_args):
    """Determine which saved-algorithm file (if any) to resume from.

    Returns the path to the backup/finished file to reload, or ``None`` for a
    fresh run. When the user opts (at the prompt) to continue an in-progress run
    found in ``output_dir``, ``cmdline_args.resume`` is set to 0 as a side effect.
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(config.config['output_dir'])
    backup_file = output_dir / 'alg_backup.bp'
    finished_file = output_dir / 'alg_finished.bp'
    continue_file = None
    if cmdline_args.resume is not None:
        if backup_file.exists():
            continue_file = backup_file
        elif finished_file.exists():
            if cmdline_args.resume <= 0:
                raise PybnfError(f'The fitting run saved in {output_dir} already finished. If you want to continue '
                                 'the fitting with more iterations, pass a number of iterations with the '
                                 '--resume flag.')
            continue_file = finished_file
        else:
            raise PybnfError(f'No algorithm found to resume in {output_dir}')
    elif backup_file.exists() and not cmdline_args.overwrite:
        ans = 'x'
        while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
            ans = input('Your output_dir contains an in-progress run.\nContinue that run? [y/n] (y) ')
        if ans.lower() in ('y', 'yes', ''):
            logger.info('Resuming a previous run')
            continue_file = backup_file
            cmdline_args.resume = 0
    return continue_file


def _load_resumed_algorithm(continue_file, cmdline_args):
    """Reload a pickled algorithm (and its pending jobs) to resume a run.

    Returns ``(alg, pending, config)``; ``config`` is taken from the reloaded
    algorithm.
    """
    logger = logging.getLogger(__name__)
    logger.info('Reloading algorithm')
    with open(continue_file, 'rb') as f:
        alg, pending = pickle.load(f)
    logger.debug('Loaded algorithm is the %s algorithm' % ('refinement' if alg.refine else 'configured'))
    config = alg.config

    logger.debug('Checking for Simulations directory')
    if not os.path.exists(alg.sim_dir):
        os.makedirs(alg.sim_dir)

    if alg.bootstrap_number is not None:
        print0('Resuming a bootstrapping run')
        logger.info('Resuming a bootstrapping run')
        if cmdline_args.resume > 0 and cmdline_args.resume is not None:
            raise PybnfError("Cannot increase the number of iterations in a boostrapping run")
    else:
        print0('Resuming a fitting run')

    alg.add_iterations(cmdline_args.resume)
    if isinstance(alg, algs.SimplexAlgorithm):
        # The continuing alg is already on the Simplex stage, so don't restart simplex after completion
        alg.config.config['refine'] = 0
    return alg, pending, config


def _prepare_run_directories(config, cmdline_args):
    """Create the output/results/simulation directories for a fresh run.

    Prompts for (or auto-applies, with ``--overwrite``) deletion of any leftover
    files from a previous run, then creates the fresh directory tree and copies
    the config file into Results. May ``exit(0)`` if the user declines to overwrite.
    """
    logger = logging.getLogger(__name__)
    # Create output folders, checking for overwrites.
    output_dir = Path(config.config['output_dir'])
    sim_dir_cfg = config.config['simulation_dir']
    subdirs = ('Simulations', 'Results', 'Initialize', 'FailedSimLogs')
    subfiles = ('alg_backup.bp', 'alg_finished.bp', 'alg_refine_finished.bp')
    will_overwrite = [subdir for subdir in subdirs + subfiles
                      if (output_dir / subdir).exists()]
    if sim_dir_cfg:
        simdir = Path(sim_dir_cfg) / 'Simulations'
        if simdir.exists():
            will_overwrite.append(str(simdir))
    if len(will_overwrite) > 0:
        if not cmdline_args.overwrite:
            logger.info("Output directory already exists... querying user for overwrite permission")
            ans = 'x'
            while ans.lower() not in ['y', 'yes', 'n', 'no', '']:
                print0('Your output directory contains files from a previous run: {}.'.format(', '.join(will_overwrite)))
                ans = input(
                    'Overwrite them with the current run? [y/n] (n) ')
            if not(ans.lower() == 'y' or ans.lower() == 'yes'):
                logger.info("Overwrite rejected... exiting")
                print0('Quitting')
                exit(0)
        # If we get here, safe to overwrite files
        for subdir in subdirs:
            target = output_dir / subdir
            try:
                shutil.rmtree(target)
                logger.info('Deleted old directory %s' % target)
            except OSError:
                logger.debug('Directory %s does not already exist' % target)
        for subfile in subfiles:
            target = output_dir / subfile
            try:
                target.unlink()
                logger.info('Deleted old file %s' % target)
            except OSError:
                logger.debug('File %s does not already exist' % target)
        if sim_dir_cfg:
            sim_simulations = Path(sim_dir_cfg) / 'Simulations'
            try:
                shutil.rmtree(sim_simulations)
                logger.info('Deleted old simulation directory %s' % sim_simulations)
            except OSError:
                logger.debug('Simulation directory %s does not already exist' % sim_simulations)

    # Create new directories for the current run.
    (output_dir / 'Results').mkdir(parents=True)
    if sim_dir_cfg:
        (Path(sim_dir_cfg) / 'Simulations').mkdir(parents=True)
    else:
        (output_dir / 'Simulations').mkdir()
    shutil.copy(cmdline_args.conf_file, output_dir / 'Results')


def _create_algorithm(config):
    """Instantiate the fitting algorithm selected by ``config['fit_type']``.

    Dispatch goes through the self-registering FIT_TYPE_REGISTRY, populated when
    ``pybnf.algorithms`` is imported above (each leaf's ``@register_fit_type``
    fires on import); see ADR-0005. Deprecated fit types warn but still run.
    The mh-vs-pt difference is handled in Config by setting or not setting the
    exchange_every key; sa runs the same class in simulated-annealing mode.
    """
    logger = logging.getLogger(__name__)
    fit_type = config.config['fit_type']
    entry = FIT_TYPE_REGISTRY.get(fit_type)
    if entry is None:
        raise PybnfError('Invalid job_type {}. Options are: {}'.format(fit_type, ', '.join(FIT_TYPE_REGISTRY)))
    if entry.deprecated:
        msg = f'job_type {fit_type} is deprecated and may be removed in a future release.'
        logger.warning(msg)
        print1(f'Warning: {msg}')
    return entry.cls(config, **entry.kwargs)


def _record_phase(alg, phase, method, status, reason=None, simulations=None,
                  best_objective=None, extra=None):
    """Append one phase to the run's executed-method record, if it is keeping one.

    A no-op when ``alg`` carries no :class:`~pybnf.method_chain.MethodChain` -- an
    algorithm driven directly rather than through ``main()``.
    """
    chain = getattr(alg, 'method_chain', None)
    if chain is None:
        return
    chain.record(phase, method, status, reason=reason, simulations=simulations,
                 best_objective=best_objective,
                 bootstrap_replicate=getattr(alg, 'bootstrap_number', None), extra=extra)


def _phase_status(alg):
    """Whether a finished phase ended on its own terms or on the wall-time budget.

    ``Algorithm.stop_reason`` is set by exactly one thing -- the budget (#529); every
    method's own stop criteria live on its runner objects -- so its presence is the
    distinction, and its text is the explanation.
    """
    if alg.stop_reason:
        return method_chain.WALL_TIME_EXPIRED, alg.stop_reason
    return method_chain.COMPLETED, None


def _refine_best_fit(config, alg, cluster, debug):
    """Refine the best-fit parameter set with the chosen local optimizer, if
    requested.

    A no-op unless ``refine == 1``; the refiner is the start-point optimizer named
    by ``refine_method`` (Simplex / Powell / CMA-ES / a gradient optimizer,
    #403/ADR-0015), dispatched through the registry's ``refiner`` flag. Skipped (with
    a message) when the original fit already used that same algorithm. Reuses the
    algorithm's generated networks and trajectory so refinement continues from the
    existing best fit.

    ``refine = 1`` is a request for a *method* -- search globally, then polish
    locally -- so under a wall-clock budget the refine runs on the slice of it the
    search was forbidden to spend (``wall_time_refine_frac``, #564/ADR-0107), released
    here by :func:`~pybnf.budget.spend_reserve`. Whether it ran, and on what terms, is
    recorded in ``Results/method_chain.json`` either way.
    """
    logger = logging.getLogger(__name__)
    if config.config['refine'] != 1:
        return
    logger.debug('Refinement requested for best fit parameter set')
    method = config.config['refine_method']
    entry = FIT_TYPE_REGISTRY[method]   # validated by Configuration._check_refine_method
    refiner_cls = entry.cls
    name = entry.display_name
    # Checked before the budget: this is not a refine the clock cost anyone, and it is
    # the more useful thing to be told (no reserve was taken for it either).
    if config.config['fit_type'] == method:
        skipped = (f"You specified refine=1 with refine_method={method}, but that is the algorithm you just ran."
                   "\nSkipping refine.")
        logger.debug(f'Cannot refine further if the {name} algorithm was used for the original fit')
        print1(skipped)
        _record_phase(alg, 'refine', method, method_chain.SKIPPED,
                      reason='refine_method %s is the algorithm the fit itself ran' % method)
        return
    reserve = alg.budget.reserve if alg.budget is not None else 0.0
    # Everything from here runs with the reserve released: this is the phase it was
    # held back FOR, so "is the budget spent?" must be asked of the refine's own share
    # of it, not of the search's -- otherwise the reserve buys the refine nothing and
    # #564 stands unfixed. Inside the block the budget is the run's whole remaining
    # time: the reserve, plus whatever a search that converged early left behind.
    with spend_reserve(alg.budget):
        if alg.budget is not None and alg.budget.expired():
            # Reachable only when the run reserved nothing for this phase
            # (wall_time_refine_frac = 0) or the search overran the reserve anyway --
            # one in-flight simulation may outlive the deadline by up to wall_time_sim
            # (ADR-0093). Either way the requested method chain did NOT run, so say so
            # at every verbosity: a silent downgrade to a different method is the defect
            # the reserve exists to prevent.
            reason = ('the wall-time budget was spent before the refine could start'
                      if reserve <= 0 else
                      'the search overran the %d s reserved for the refine'
                      % int(round(reserve)))
            logger.info('Wall-time budget spent; skipping the refine of the best fit')
            print0('Warning: %s, so the best fit was NOT refined by the %s algorithm -- this run '
                   'executed %s alone.\nRaise wall_time_refine_frac (the share of wall_time_fit held '
                   'back for the refine) to make room for it.'
                   % (reason, name, config.config['fit_type']))
            _record_phase(alg, 'refine', method, method_chain.SKIPPED, reason=reason)
            return
        logger.debug(f'Refining further using the {name} algorithm')
        print1(f"Refining the best fit by the {name} algorithm")
        config.config[refiner_cls.START_POINT_KEY] = alg.trajectory.best_fit()
        refiner = refiner_cls(config, refine=True)
        refiner.model_list = alg.model_list  # Reuse already-generated networks
        refiner.trajectory = alg.trajectory  # Reuse existing trajectory; don't start a new one.
        refiner.budget = alg.budget  # One deadline for the whole run, not one per phase
        # The refine's outputs belong beside the fit's, in the directory that fit wrote to.
        # For the main fit that is the default the refiner already builds; for a bootstrap
        # replicate it is Results-boot{N}, and taking the default there would overwrite the
        # MAIN run's sorted_params_refine_final.txt (and stop_reason.txt) once per replicate.
        refiner.res_dir = alg.res_dir
        if alg.bootstrap_number is not None:
            # During bootstrapping the replicate's fit wrote to per-replicate working dirs
            # (Simulations-boot{N} etc.) that alg.reset(bootstrap=N) (re)creates for every
            # replicate and retry, so point the refiner at those same, existing dirs.
            # Otherwise it defaults to the shared output_dir/Simulations, which the *main*
            # fit's own refinement already deleted at teardown (it is a leaf Simplex run).
            # Every replicate refine then tries os.mkdir under that missing parent; the
            # FileNotFoundError reads as OSError, which the job's mkdir-retry loop treats as
            # a name collision and walks to _rerun999 before giving up -> FailedSimulation.
            # The whole refine then scores inf and never polishes the replicate's fit.
            refiner.sim_dir = alg.sim_dir
            refiner.failed_logs_dir = alg.failed_logs_dir
        refiner.run(cluster.client, debug=debug)
    status, reason = _phase_status(refiner)
    _record_phase(alg, 'refine', method, status, reason=reason,
                  simulations=refiner.completed_simulations,
                  best_objective=refiner.trajectory.best_score() if len(refiner.trajectory) else None)


def _run_bootstrapping(config, alg, cluster, debug):
    """Run the configured number of bootstrap replicate fits.

    Resamples the experimental-data weights for each replicate, re-runs the fit
    (optionally refining), and records accepted best-fit parameter sets. Replicates
    whose objective exceeds ``bootstrap_max_obj`` are retried.
    """
    logger = logging.getLogger(__name__)
    # Path the accepted-replicate parameter sets are accumulated into (str at the
    # boundary: Trajectory.load_trajectory / Trajectory.add take a filename).
    bootstrap_file = str(Path(config.config['output_dir']) / 'Results' / 'bootstrapped_parameter_sets.txt')
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
                bootstrapped_psets = Trajectory.load_trajectory(bootstrap_file,
                                                                config.variables,
                                                                config.config['num_to_output'])

        # Gate on the trajectory's best score, exactly as the main loop below. main()
        # ran _refine_best_fit on this resumed replicate before calling us, and the
        # refiner shares alg.trajectory -- but alg.best_fit_obj is the *pre*-refine value
        # the loaded algorithm stopped at (a replicate resumed mid-fit finishes its fit
        # and is then polished), so gating on it would reject a replicate the polish
        # already brought under the threshold and pair the refined pset with a stale score.
        resumed_best_obj = alg.trajectory.best_score()
        if resumed_best_obj <= bootstrap_max_obj:
            logger.info(f'Bootstrap run {completed_bootstrap_runs} complete')
            bootstrapped_psets.add(alg.trajectory.best_fit(), resumed_best_obj,
                                   f'bootstrap_run_{completed_bootstrap_runs}',
                                   bootstrap_file,
                                   completed_bootstrap_runs == 0)
            logger.info(f'Succesfully completed resumed bootstrapping run {completed_bootstrap_runs}')
            completed_bootstrap_runs += 1
        else:
            shutil.rmtree(alg.res_dir)
            if os.path.exists(alg.sim_dir):
                shutil.rmtree(alg.sim_dir)
            print0("Bootstrap run did not achieve maximum allowable objective function value.  Retrying")
            logger.info(f'Resumed bootstrapping run {completed_bootstrap_runs} did not achieve maximum allowable objective function '
                        'value.  Retrying')

    # Run bootstrapping
    consec_failed_bootstrap_runs = 0
    while completed_bootstrap_runs < num_to_bootstrap:
        if alg.budget is not None and alg.budget.expired():
            # A replicate is a whole fit's worth of new work; the budget bounds the run,
            # not each replicate (#529). Stop with the replicates already accepted --
            # bootstrapped_parameter_sets.txt was appended to as each one finished, so
            # what is on disk is a complete record of them.
            logger.info('Wall-time budget spent after %d of %d bootstrap replicates',
                        completed_bootstrap_runs, num_to_bootstrap)
            print0('Wall-time budget spent; stopping after %d of %d bootstrap replicates.'
                   % (completed_bootstrap_runs, num_to_bootstrap))
            _record_bootstrap_total(config, alg, completed_bootstrap_runs, num_to_bootstrap,
                                    method_chain.WALL_TIME_EXPIRED)
            return
        # A rejected replicate is retried with a *fresh* resample: bumping the retry
        # count advances the RNG sub-stream reset() derives. Without it, reset() keys
        # the seed only on the (unchanged) replicate number, so every retry regenerates
        # the identical resample and identical fit -- the 20 "retries" are byte-for-byte
        # the same failing run, and the loop can never make progress. The first attempt
        # (the common case) is byte-identical to the historical seeding.
        alg.bootstrap_attempt = consec_failed_bootstrap_runs
        alg.reset(bootstrap=completed_bootstrap_runs)

        for model in alg.exp_data:
            for name, data in alg.exp_data[model].items():
                # alg.reset(bootstrap=...) just reseeded alg.rng to this replicate's
                # deterministic sub-stream, so the resampled weights are reproducible.
                data.gen_bootstrap_weights(alg.rng)
                data.weights_to_file(f'{alg.res_dir}/{name}_weights_{completed_bootstrap_runs}.txt')

        logger.info(f'Beginning bootstrap run {completed_bootstrap_runs}')
        print0(f"Beginning bootstrap run {completed_bootstrap_runs}")
        alg.run(cluster.client, debug=debug)
        status, reason = _phase_status(alg)
        _record_phase(alg, 'fit', config.config['fit_type'], status, reason=reason,
                      simulations=alg.completed_simulations,
                      best_objective=alg.trajectory.best_score() if len(alg.trajectory) else None)

        _refine_best_fit(config, alg, cluster, debug)

        best_fit_pset = alg.trajectory.best_fit()
        # Gate (and record) on the trajectory's best score, which reflects the refine
        # polish above (the refiner shares alg.trajectory). alg.best_fit_obj is the
        # *pre*-refine value the fit algorithm stopped at, so using it here would reject
        # a replicate the polish already brought under the threshold -- and pair the
        # recorded (refined) pset with a stale, larger objective.
        best_fit_obj = alg.trajectory.best_score()

        if best_fit_obj <= bootstrap_max_obj:
            logger.info(f'Bootstrap run {completed_bootstrap_runs} complete')
            bootstrapped_psets.add(best_fit_pset, best_fit_obj, f'bootstrap_run_{completed_bootstrap_runs}',
                                   bootstrap_file,
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
    _record_bootstrap_total(config, alg, completed_bootstrap_runs, num_to_bootstrap,
                            method_chain.COMPLETED)
    print0('Bootstrapping complete')


def _record_bootstrap_total(config, alg, completed, requested, status):
    """Record how many bootstrap replicates the run actually produced.

    The replicates themselves are recorded one by one as they run; this is the single
    field a consumer can assert on -- ``bootstrap = 30`` in the conf is worth nothing
    if the budget stopped the run at 11 (#564).
    """
    chain = getattr(alg, 'method_chain', None)
    if chain is None:
        return
    chain.record('bootstrap', config.config['fit_type'], status,
                 reason=None if completed >= requested else
                 'the wall-time budget was spent after %d of %d replicates' % (completed, requested),
                 extra={'replicates_requested': requested, 'replicates_completed': completed})


def _reap_running_sims():
    """SIGKILL any simulation subprocesses still running on this node.

    Called from ``main()``'s finally on every exit. On a clean finish it is a
    no-op (finished sims have deregistered); on a Ctrl-C or crash it kills the
    in-flight, detached sim process groups that would otherwise orphan (the
    "kill -9 needed" problem). Local node only: sims on remote dask-ssh worker
    nodes are cleaned by SLURM's cgroup tracking at step/job teardown.
    """
    try:
        reaped = reap_active_sims()
        if reaped:
            logging.info('Killed %d orphaned simulation process group(s) on abort', len(reaped))
    except Exception:
        logging.exception('Failed while reaping simulation subprocesses')


def _teardown_cluster(cluster):
    """Tear down the dask cluster after a run, logging (not raising on) any failure."""
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


def _cleanup_dask_workspace():
    """Remove any leftover ``dask-worker-space`` directories (cwd and home)."""
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


def main():
    """The main function for running a fitting job"""
    start_time = time.time()

    success = False
    alg = None
    cluster = None

    cmdline_args = _build_arg_parser().parse_args()
    log_prefix, debug = _setup_logging(cmdline_args)
    logger = logging.getLogger(__name__)

    # Enable the node-local live-sim registry so a Ctrl-C or crash can reap any
    # in-flight simulation subprocesses. Set before the cluster is created so
    # locally-spawned dask workers inherit PYBNF_SIM_REGISTRY (see pset).
    set_sim_registry(f'{log_prefix}_{os.getpid()}')

    print0(f"PyBNF v{__version__}")
    logger.info(f'Running PyBNF v{__version__}')
    _report_bngsim_build()

    try:
        # Load the conf file and create the algorithm
        if cmdline_args.conf_file is None:
            print0('No configuration file given, so I won''t do anything.\nFor more information, try pybnf --help')
            exit(0)
        logger.info(f'Loading configuration file: {cmdline_args.conf_file}')

        config = load_config(cmdline_args.conf_file)
        if 'verbosity' in config.config:
            printing.verbosity = config.config['verbosity']
        _initialize_random_seed(config)

        if cmdline_args.resume is not None and cmdline_args.overwrite:
            raise PybnfError("Options --overwrite and --resume are contradictory. Use --resume to continue a previous "
                             "run, or --overwrite to overwrite the previous run with a new one.")

        continue_file = _resolve_continue_file(config, cmdline_args)
        if continue_file:
            # Restart the loaded algorithm
            alg, pending, config = _load_resumed_algorithm(continue_file, cmdline_args)
        else:
            # Fresh run: prepare the output directory tree and build the algorithm.
            _prepare_run_directories(config, cmdline_args)
            pending = None
            alg = _create_algorithm(config)

        # The fit's total wall-clock budget (wall_time_fit, #529/ADR-0093), or None when
        # unbounded (the default). Built from the process start time so configuration
        # loading and network generation are charged to it -- the budget bounds the run
        # the way an external `timeout` around this process would have -- and attached to
        # the algorithm, which stops its run loop and finalizes when the budget is spent.
        # The same object is handed to the refiner and reused across bootstrap replicates,
        # so one deadline bounds the whole run rather than each phase separately.
        alg.budget = FitBudget.from_config(config, started_at=start_time)
        if alg.budget is not None:
            logger.info('Total wall-time budget for this fit: %d s (%s remaining)',
                        alg.budget.limit, format_duration(alg.budget.remaining()))
            print1('Wall-time budget for this fit: %s (wall_time_fit = %d s).'
                   % (format_duration(alg.budget.limit), alg.budget.limit))
            if alg.budget.reserve > 0:
                # State the phase split up front. It is the difference between the
                # method the conf requests and the method a budgeted run executes, so
                # it belongs on the console before the search starts, not in a
                # post-mortem (#564).
                logger.info('Reserving %d s of the budget for the refine', alg.budget.reserve)
                print1('  of which %s is reserved for the refine (wall_time_refine_frac = %g), '
                       'leaving %s to search.'
                       % (format_duration(alg.budget.reserve),
                          config.config['wall_time_refine_frac'],
                          format_duration(alg.budget.limit - alg.budget.reserve)))

        # Override configuration values if provided on command line
        if cmdline_args.cluster_type:
            config.config['cluster_type'] = cmdline_args.cluster_type
        if cmdline_args.scheduler_file:
            config.config['scheduler_file'] = cmdline_args.scheduler_file

        # Everything inside this branch is Algorithm surface. ``ModelCheck``
        # (``job_type = check``) deliberately does not subclass Algorithm -- it has no
        # res_dir, no trajectory, no completed_simulations and no stop_reason -- so a
        # line that reads any of those belongs here and not above the branch, where it
        # would run on both paths and take the checker down before it ever evaluated
        # the model (#569).
        if config.config['fit_type'] != 'check':
            # The record of which methods this run actually executes (#564/ADR-0107). Kept
            # on the algorithm beside the budget, and written after every phase, so it is
            # complete on disk even if a later phase raises.
            alg.method_chain = method_chain.chain_for_run(alg.res_dir, config.config,
                                                          budget=alg.budget, version=__version__)
            # Set up cluster
            cluster = Cluster(config, log_prefix, debug, cmdline_args.log_level)
            # Run the algorithm!
            logger.debug('Algorithm initialization')
            alg.run(cluster.client, resume=pending, debug=debug)

            fit_status, fit_reason = _phase_status(alg)
            _record_phase(alg, 'fit', config.config['fit_type'], fit_status, reason=fit_reason,
                          simulations=alg.completed_simulations,
                          best_objective=alg.trajectory.best_score() if len(alg.trajectory) else None)
        else:
            # Run model checking. No method chain: a check is one evaluation of the
            # parameters as given, not a chain of search phases, and ModelCheck holds no
            # res_dir to write the record into.
            logger.debug('Model checking initialization')
            alg.run_check(debug=debug)

        _refine_best_fit(config, alg, cluster, debug)

        if alg.bootstrap_number is None:
            print0('Fitting complete')

        # Bootstrapping (optional)
        if config.config['bootstrap'] > 0:
            _run_bootstrapping(config, alg, cluster, debug)

        success = True

    except PybnfError as e:
        # Exceptions generated by problems such as bad user input should be caught here and print a useful message
        # before quitting
        logger.error('Terminating due to a PybnfError:')
        logger.error(e.log_message)
        print0(f'Error: {e.message}')
    except KeyboardInterrupt:
        print0('Fitting aborted.')
        logger.info('Terminating due to keyboard interrupt')
        logger.exception('Keyboard interrupt')
    except Exception:
        # Sends any unhandled errors to log instead of to user output
        logger.exception('Internal error')
        exceptiondata = traceback.format_exc().splitlines()
        print0(f'Sorry, an unknown error occurred: {exceptiondata[-1]}\n'
               f'Logs have been saved to {log_prefix}.log.\n'
               'Please report this bug to help us improve PyBNF.')
    finally:
        # Kill any in-flight sims first (while their PGIDs are still live), then
        # tear down the cluster and report timing.
        _reap_running_sims()
        _teardown_cluster(cluster)
        _cleanup_dask_workspace()
        # After any error, try to clean up; then report timing and exit.
        _finalize(success, alg, start_time)


# PyBNF is launched via the ``pybnf`` console script (``pybnf -c fit.conf``) or
# ``python -m pybnf`` (see pybnf/__main__.py) -- both call main() behind a proper
# __main__ guard, which is what keeps spawned dask workers from re-running the fit.
# Executing this *submodule* directly (``python -m pybnf.pybnf``) would instead
# import-and-exit silently, reading as a confusing no-op. Fail loudly and point at
# the supported entry points rather than adding a second, unguarded run path.
if __name__ == '__main__':
    raise SystemExit(
        'Run PyBNF via the `pybnf` console script (e.g. `pybnf -c fit.conf`) or '
        '`python -m pybnf`, not `python -m pybnf.pybnf`.')
