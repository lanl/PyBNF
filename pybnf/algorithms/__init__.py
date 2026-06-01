
"""Contains the Algorithm class and subclasses as well as support classes and functions for running simulations"""


from . import core
# Re-export the core execution primitives so the package facade
# (algorithms.Result, algorithms.run_job, ...) keeps resolving for external
# callers and tests. The run loop itself resolves the patched names as
# core.run_job / core.as_completed / core.Job (ADR-0001), so it does not rely
# on these bare names. `X as X` marks them as intentional re-exports for ruff.
from .core import (
    Result as Result,
    FailedSimulation as FailedSimulation,
    Job as Job,
    JobGroup as JobGroup,
    MultimodelJobGroup as MultimodelJobGroup,
    HybridJobGroup as HybridJobGroup,
    run_job as run_job,
    result_from_completed as result_from_completed,
)
# The Algorithm base class and its module-level helpers live in base.py. Every
# leaf class below subclasses Algorithm, so it must be in this namespace before
# those definitions execute; exp10/latin_hypercube are re-exported so leaves
# (SimplexAlgorithm) and the facade (algorithms.exp10) keep resolving them.
from .base import (
    Algorithm as Algorithm,
    latin_hypercube as latin_hypercube,
    exp10 as exp10,
)
# BayesianAlgorithm (sampler base + R-hat/ESS diagnostics) lives in
# samplers/base.py. The Bayesian leaf classes below subclass it, so it must be
# in this namespace before their definitions execute; the facade re-export also
# keeps algorithms.BayesianAlgorithm resolving for tests.
from .samplers.base import BayesianAlgorithm as BayesianAlgorithm
# Leaf fit types, re-exported so pybnf.py's algs.* dispatch and the test facade
# (algorithms.ParticleSwarm, ...) keep resolving. Peeled into optimizers/ and
# samplers/ one file per commit (M1 Step 4).
from .optimizers.particle_swarm import ParticleSwarm as ParticleSwarm
from .optimizers.differential_evolution import (
    DifferentialEvolutionBase as DifferentialEvolutionBase,
    DifferentialEvolution as DifferentialEvolution,
    AsynchronousDifferentialEvolution as AsynchronousDifferentialEvolution,
)
from .optimizers.scatter_search import ScatterSearch as ScatterSearch
from .optimizers.simplex import SimplexAlgorithm as SimplexAlgorithm
from .samplers.dream import DreamAlgorithm as DreamAlgorithm
from .samplers.pdream import PDreamAlgorithm as PDreamAlgorithm
from ..pset import PSet
from ..pset import OutOfBoundsException
from ..printing import print0, print1, print2, PybnfError
from ..objective import ConstraintCounter

import logging
import numpy as np
import os
import re
import shutil
import copy
import traceback
from scipy import stats
# Facade re-export: the run loop (now in base.py) catches CancelledError, and
# tests construct algorithms.CancelledError to drive that path.
from concurrent.futures import CancelledError as CancelledError



logger = logging.getLogger(__name__)


class BasicBayesMCMCAlgorithm(BayesianAlgorithm):

    """
    Implements a Bayesian Markov chain Monte Carlo simulation.
    This is essentially a non-parallel algorithm, but here, we run n instances in parallel, and pool all results.
    This will give you a best fit (which is maybe not great), but more importantly, generates an extra result file
    that gives the probability distribution of each variable.
    This distribution depends on the prior, which is specified according to the variable initialization rules.
    With sa=True, this instead acts as a simulated annealing algorithm with n indepdendent chains.
    """

    def __init__(self, config, sa=False):  # expdata, objective, priorfile, gamma=0.1):
        super(BasicBayesMCMCAlgorithm, self).__init__(config)
        self.sa = sa

        if sa:
            self.cooling = config.config['cooling']
            self.beta_max = config.config['beta_max']

        self.exchange_every = config.config['exchange_every']
        self.pt = self.exchange_every != np.inf
        self.reps_per_beta = self.config.config['reps_per_beta']
        self.betas_per_group = self.num_parallel // self.reps_per_beta  # Number of unique betas considered (in PT)

        # The temperature of each replicate
        # For MCMC, probably n copies of the same number, unless the user set it up strangely
        # For SA, starts all the same (unless set up strangely), and independently decrease during the run
        # For PT, contains reps_per_beta copies of the same ascending sequence of betas, e.g.
        # [0.6, 0.8, 1., 0.6, 0.8, 1.]. Indices congruent to -1 mod (population_size/reps_per_beta) have the max beta
        # (probably 1), and only these replicas are sampled.
        self.betas = config.config['beta_list']

        self.wait_for_sync = [False] * self.num_parallel

        self.prior = None
        self.load_priors()

        self.attempts = 0
        self.accepted = 0
        self.exchange_attempts = 0
        self.exchange_accepted = 0

        self.staged = []  # Used only when resuming a run and adding iterations
        self.converged = False  # Set by try_to_choose_new_pset on R-hat convergence

    def reset(self, bootstrap=None):
        super(BasicBayesMCMCAlgorithm, self).reset(bootstrap)

        self.current_pset = None
        self.ln_current_P = None
        self.iteration = [0] * self.num_parallel

        self.wait_for_sync = [False] * self.num_parallel
        self.samples_file = None

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.
        :return: list of PSets
        """
        if self.sa:
            print2('Running simulated annealing on %i independent replicates in parallel, for %i iterations each or '
                   'until 1/T reaches %s' % (self.num_parallel, self.max_iterations, self.beta_max))
        else:
            if not self.pt:
                print2('Running Markov Chain Monte Carlo on %i independent replicates in parallel, for %i iterations each.'
                       % (self.num_parallel, self.max_iterations))
            else:
                print2('Running parallel tempering on %i replicates for %i iterations, with replica exchanges performed '
                       'every %i iterations' % (self.num_parallel, self.max_iterations, self.exchange_every))

            print2('Statistical samples will be recorded every %i iterations, after an initial %i-iteration burn-in period'
                   % (self.sample_every, self.burn_in))
            if self.max_iterations <= self.burn_in:
                raise PybnfError(
                    'max_iterations (%i) must be greater than burn_in (%i), '
                    'otherwise no samples will be collected.'
                    % (self.max_iterations, self.burn_in))

        setup_samples = not self.sa
        return super(BasicBayesMCMCAlgorithm, self).start_run(setup_samples=setup_samples)

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data
        :param res: PSet that was run in this simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """

        pset = res.pset
        score = res.score
        self.total_evaluations += 1

        # Figure out which parallel run this is from based on the .name field.
        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        # Calculate the acceptance probability
        lnprior = self.ln_prior(pset) # Need something clever for box constraints
        lnlikelihood = -score

        # Because the P's are so small to start, we express posterior, p_accept, and current_P in ln space
        lnposterior = lnprior + lnlikelihood

        ln_p_accept = min(0., lnposterior - self.ln_current_P[index])

        # Decide whether to accept move.
        self.attempts += 1
        if np.random.rand() < np.exp(ln_p_accept*self.betas[index]) or np.isnan(self.ln_current_P[index]):
            # Accept the move, so update our current PSet and P
            self.accepted += 1
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.evaluate_constraints(res.simdata, index)
            # For simulated annealing, reduce the temperature if this was an unfavorable move.
            if self.sa and ln_p_accept < 0.:
                self.betas[index] += self.cooling
                if self.betas[index] >= self.beta_max:
                    print2('Finished replicate %i because beta_max was reached.' % index)
                    logger.info('Finished replicate %i because beta_max was reached.' % index)
                    if min(self.betas) >= self.beta_max:
                        logger.info('All annealing replicates have reached the maximum beta value')
                        return 'STOP'
                    else:
                        return []

        # Store chain history (after accept/reject, so it reflects the kept state)
        if self.current_pset[index] is not None:
            self.chain_history[index].append(self._param_vec(self.current_pset[index]))
            self.ln_posterior_history[index].append(self.ln_current_P[index])

        # Record the current PSet (clarification: what if failed? Sample old again?)
        # Using either the newly accepted PSet or the old PSet, propose the next PSet.
        proposed_pset = self.try_to_choose_new_pset(index)

        if proposed_pset is None:
            if self.converged:
                print0('Overall move accept rate: %f' % (self.accepted/self.attempts))
                return 'STOP'
            elif np.all(self.wait_for_sync):
                # Do the replica exchange, then propose n new psets so all chains resume
                self.wait_for_sync = [False] * self.num_parallel
                return self.replica_exchange()
            elif min(self.iteration) >= self.max_iterations:
                print0('Overall move accept rate: %f' % (self.accepted/self.attempts))
                if not self.sa:
                    self.update_histograms('_final')
                    self.report_constraint_satisfaction('_final')
                return 'STOP'
            else:
                return []

        proposed_pset.name = 'iter%irun%i' % (self.iteration[index], index)
        # Note self.staged is empty unless we just resumed a run with added iterations and need to restart chains.
        if len(self.staged) != 0:
            toreturn = [proposed_pset] + self.staged
            self.staged = []
            return toreturn
        return [proposed_pset]

    def try_to_choose_new_pset(self, index):
        """
        Helper function
        Advances the iteration number, and tries to choose a new parameter set for chain index i
        If that fails (e.g. due to a box constraint), keeps advancing iteration number and trying again.
        If it hits an iteration where it has to stop and wait (a replica exchange iteration or the end), returns None
        Otherwise returns the new PSet.
        :param index:
        :return:
        """
        proposed_pset = None
        # This part is a loop in case a box constraint makes a move automatically rejected.
        loop_count = 0
        while proposed_pset is None:
            loop_count += 1
            if loop_count == 20:
                logger.warning('Instance %i spent 20 iterations at the same point' % index)
                print1('One of your samples is stuck at the same point for 20+ iterations because it keeps '
                       'hitting box constraints. Consider using looser box constraints or a smaller '
                       'step_size.')
            if loop_count == 1000:
                logger.warning('Instance %i terminated after 1000 iterations at the same point' % index)
                print1('Instance %i was terminated after it spent 1000 iterations stuck at the same point '
                       'because it kept hitting box constraints. Consider using looser box constraints or a '
                       'smaller step_size.' % index)
                self.iteration[index] = self.max_iterations

            self.iteration[index] += 1
            # Check if it's time to do various things
            if not self.sa:
                if self.iteration[index] > self.burn_in and self.iteration[index] % self.sample_every == 0 \
                        and self.should_sample(index):
                    self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
                if (self.iteration[index] > self.burn_in
                   and self.iteration[index] % (self.output_hist_every * self.sample_every) == 0
                   and self.iteration[index] == min(self.iteration)):
                    self.update_histograms('_%i' % self.iteration[index])

            if self.iteration[index] == min(self.iteration):
                if self.iteration[index] % self.config.config['output_every'] == 0:
                    self.output_results()
                if self.iteration[index] % 10 == 0:
                    print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                    print2('Current move accept rate: %f' % (self.accepted/self.attempts))
                    if self.exchange_attempts > 0:
                        print2('Current replica exchange rate: %f' % (self.exchange_accepted / self.exchange_attempts))
                else:
                    print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
                if not self.sa and self.iteration[index] % self.diagnostics_every == 0:
                    max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                    if self.check_convergence(self.iteration[index], max_rhat):
                        self.converged = True
                        return None
                logger.debug('Completed %i iterations' % self.iteration[index])
                logger.debug('Current move accept rate: %f' % (self.accepted/self.attempts))
                if self.exchange_attempts > 0:
                    logger.debug('Current replica exchange rate: %f' % (self.exchange_accepted / self.exchange_attempts))
                if self.sa:
                    logger.debug('Current betas: ' + str(self.betas))
                print2('Current -Ln Likelihoods: ' + str(self.ln_current_P))
            if self.iteration[index] >= self.max_iterations:
                logger.info('Finished replicate number %i' % index)
                print2('Finished replicate number %i' % index)
                return None
            if self.iteration[index] % self.exchange_every == 0:
                # Need to wait for the rest of the chains to catch up to do replica exchange
                self.wait_for_sync[index] = True
                return None
            proposed_pset = self.choose_new_pset(self.current_pset[index])
        return proposed_pset

    def should_sample(self, index):
        """
        Checks whether this replica index is one that gets sampled.
        For mcmc, always True. For pt, must be a replica at the max beta
        """
        return (index + 1) % self.betas_per_group == 0 if self.pt else True

    def choose_new_pset(self, oldpset):
        """
        Helper function to perturb the old PSet, generating a new proposed PSet.
        The step is a fixed-magnitude (step_size) random-walk move; any component
        that would leave the box is reflected back inside (see FreeParameter._reflect).
        :param oldpset: The PSet to be changed
        :type oldpset: PSet
        :return: the new PSet
        """

        delta_vector = {k: np.random.normal() for k in oldpset.keys()}
        delta_vector_magnitude = np.sqrt(sum([x ** 2 for x in delta_vector.values()]))
        delta_vector_normalized = {k: self.step_size * delta_vector[k] / delta_vector_magnitude for k in oldpset.keys()}
        new_vars = []
        for v in oldpset:
            # Box constraints are handled by reflection: FreeParameter.add defaults to
            # reflect=True, so a step that would leave the box is folded back inside
            # rather than rejected. The fold is symmetric, so the plain Metropolis
            # ratio in got_result still targets the correct bound-restricted posterior.
            new_var = v.add(delta_vector_normalized[v.name])
            new_vars.append(new_var)

        return PSet(new_vars)

    def replica_exchange(self):
        """
        Performs replica exchange for parallel tempering.
        Then proposes n new parameter sets to resume all chains after the exchange.
        :return: List of n PSets to run
        """
        logger.debug('Performing replica exchange on iteration %i' % self.iteration[0])
        # Who exchanges with whom is a little complicated. Each replica tries one exchange with a replica at the next
        # beta. But if we have multiple reps per beta, then the exchanges aren't necessarily within the same group of
        # reps. We use this random permutation to determine which groups exchange.
        for i in range(self.betas_per_group - 1):
            permutation = np.random.permutation(range(self.reps_per_beta))
            for group in range(self.reps_per_beta):
                # Determine the 2 indices we're exchanging, ind_hi and ind_lo
                ind_hi = self.betas_per_group * group + i
                other_group = permutation[group]
                ind_lo = self.betas_per_group * other_group + i + 1
                # Consider exchanging index ind_hi (higher T) with ind_lo (lower T)
                ln_p_exchange = min(0., -(self.betas[ind_lo]-self.betas[ind_hi]) * (self.ln_current_P[ind_lo]-self.ln_current_P[ind_hi]))
                # Scratch work: Should there be a - sign in front? You want to always accept if moving the better answer
                # to the lower temperature. ind_lo has lower T so higher beta, so the first term is positive. The second
                # term is positive if ind_lo is better. But you want a positive final answer when ind_hi, currently at
                # higher T, is better. So you need a - sign.
                self.exchange_attempts += 1
                if np.random.random() < np.exp(ln_p_exchange):
                    # Do the exchange
                    logger.debug('Exchanging individuals %i and %i' % (ind_hi, ind_lo))
                    self.exchange_accepted += 1
                    hold_pset = self.current_pset[ind_hi]
                    hold_p = self.ln_current_P[ind_hi]
                    self.current_pset[ind_hi] = self.current_pset[ind_lo]
                    self.ln_current_P[ind_hi] = self.ln_current_P[ind_lo]
                    self.current_pset[ind_lo] = hold_pset
                    self.ln_current_P[ind_lo] = hold_p
        # Propose new psets - it's more complicated because of going out of box, and other counters.
        proposed = []
        for j in range(self.num_parallel):
            proposed_pset = self.try_to_choose_new_pset(j)
            if proposed_pset is None:
                if np.all(self.wait_for_sync):
                    logger.error('Aborting because no changes were made between one replica exchange and the next.')
                    print0("I seem to have gone from one replica exchange to the next replica exchange without "
                           "proposing a single valid move. Something is probably wrong for this to happen, so I'm "
                           "going to stop.")
                    return 'STOP'
                elif min(self.iteration) >= self.max_iterations:
                    return 'STOP'
            else:
                # Iteration number got off by 1 because try_to_choose_new_pset() was called twice: once a while ago
                # when it reached the exchange point and returned None, and a second time just now.
                # Need to correct for that here.
                self.iteration[j] -= 1
                proposed_pset.name = 'iter%irun%i' % (self.iteration[j], j)
                proposed.append(proposed_pset)
        return proposed

    def cleanup(self):
        """Called when quitting due to error.
        Save the histograms in addition to the usual algorithm cleanup"""
        super().cleanup()
        self.update_histograms('_end')
        self.report_constraint_satisfaction('_end')

    def add_iterations(self, n):
        oldmax = self.max_iterations
        self.max_iterations += n
        # Any chains that already completed need to be restarted with a new proposed parameter set
        for index in range(self.num_parallel):
            if self.iteration[index] >= oldmax:
                ps = self.try_to_choose_new_pset(index)
                if ps:
                    # Add to a list of new psets to run that will be submitted when the first result comes back.
                    ps.name = 'iter%irun%i' % (self.iteration[index], index)
                    logger.debug('Added PSet %s to BayesAlgorithm.staged to resume a chain' % (ps.name))
                    self.staged.append(ps)

class Adaptive_MCMC(BayesianAlgorithm):
    def __init__(self, config):  # expdata, objective, priorfile, gamma=0.1):
        super(Adaptive_MCMC, self).__init__(config)
        # set the params decleared in the configuaration file
        if self.config.config['normalization']:
            self.norm = self.config.config['normalization']
        else:
            self.norm = None
           
        self.time = self.config.config['time_length'] 
       
        self.adaptive = self.config.config['adaptive']
        # The iteration number that the adaptive starts at
        self.valid_range = self.burn_in + self.adaptive
        # The length of the ouput arrays and the number of iterations before they are written out
        self.arr_length = 1
        # set recorders
        self.acceptances = 0
        self.acceptance_rates = 0
        self.attempts = 0
        self.factor = [0] * self.num_parallel
        self.staged = []
        self.alpha = [0] * self.num_parallel
        # start lists
        self.current_param_set = [0] * self.num_parallel
        self.current_param_set_diff = [0] * self.num_parallel
        self.scores = np.zeros((self.num_parallel, self.arr_length))
        # set arrays for features and graphs
        self.parameter_index = np.zeros((self.num_parallel, self.arr_length, len(self.variables)))
        self.mu = np.zeros((self.num_parallel, 1, len(self.variables))) 
        # warm start features
        
        os.makedirs(self.config.config['output_dir'] + '/adaptive_files', exist_ok=True)
        os.makedirs(self.config.config['output_dir'] + '/Results/A_MCMC/Runs', exist_ok=True)
        os.makedirs(self.config.config['output_dir'] + '/Results/Histograms/', exist_ok=True)
        
        if self.config.config['output_trajectory']:
            self.output_columns = []
            for i in self.config.config['output_trajectory']:
                new = i.replace(',', '')
                self.output_columns.append(new)
            self.output_run_current = {}
            self.output_run_all = {}
            for i in self.output_columns:
                for k in self.time.keys():
                    if '_Cum' in i:
                        self.output_run_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                    else:
                        self.output_run_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                     
        
        if self.config.config['output_noise_trajectory']:
            self.output_noise_columns = []
            for i in self.config.config['output_noise_trajectory']:
                new = i.replace(',', '')
                self.output_noise_columns.append(new)
            self.output_run_noise_current = {}
            self.output_run_noise_all = {}
            for i in self.output_noise_columns:
                for k in self.time.keys():
                    if '_Cum' in i:
                        self.output_run_noise_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_noise_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                    else:
                        self.output_run_noise_current[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
                        self.output_run_noise_all[k + i] = np.zeros((self.num_parallel, 1, self.time[k]+1))
        if self.config.config['continue_run'] == 1:
            adaptive_dir = self.config.config['output_dir'] + '/adaptive_files'
            required = ['diff.txt', 'MLE_params.txt', 'diffMatrix.txt']
            missing = [f for f in required if not os.path.exists(os.path.join(adaptive_dir, f))]
            if missing:
                raise PybnfError(
                    'continue_run = 1 requires adaptive files from a completed prior run, '
                    'but the following files are missing from %s: %s. '
                    'Run the model first without continue_run, or check that output_dir '
                    'points to a previous run\'s output.' % (adaptive_dir, ', '.join(missing)))
            self.diff = [self.step_size] * self.num_parallel
            self.diff_best = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diff.txt')
            self.diffMatrix = np.zeros((self.num_parallel, len(self.variables), len(self.variables))) 
            self.diffMatrix_log = np.zeros((self.num_parallel, len(self.variables), len(self.variables)))
            if self.adaptive != 1:
                self.mle_best = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/MLE_params.txt')
                self.diffMatrix_best = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diffMatrix.txt')
                for i in range(self.num_parallel):  
                    self.diffMatrix[i] = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diffMatrix.txt')
                    self.diff[i] = np.loadtxt(self.config.config['output_dir'] + '/adaptive_files/diff.txt')
                    
        else:
            self.mle_best = np.zeros((self.arr_length, len(self.variables)))
            self.diff = [self.step_size] * self.num_parallel
            self.diff_best = self.step_size
            self.diffMatrix = np.zeros((self.num_parallel, len(self.variables), len(self.variables)))  
           

        # make sure that the adaptive and burn in iterations are less then the max iterations
        if self.adaptive + self.burn_in >= self.max_iterations - 1:
            raise PybnfError('The max iterations must be at least 2 more then the sum of the adaptive and burn-in iterations.')    
    ''' Used for resuming runs and adding iterations'''
    def reset(self, bootstrap=None):
        super(Adaptive_MCMC, self).reset(bootstrap)

        self.current_pset = None
        self.ln_current_P = None
        self.iteration = [0] * self.num_parallel

        self.wait_for_sync = [False] * self.num_parallel
        self.samples_file = None
    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.
        :return: list of PSets
        """

        print2(
                'Running Adaptive Markov Chain Monte Carlo on %i independent replicates in parallel, for %i iterations each.'
                % (self.num_parallel, self.max_iterations))


        return super(Adaptive_MCMC, self).start_run(setup_samples=True)

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data
        :param res: PSet that was run in this simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """
        pset = res.pset
        score = res.score
        self.total_evaluations += 1

        # Figure out which parallel run this is from based on the .name field.
        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        lnprior = self.ln_prior(pset)
        lnlikelihood = -score
        lnposterior = lnlikelihood + lnprior

        self.accept = False
        self.attempts += 1
        # Decide whether to accept move
        if lnposterior > self.ln_current_P[index] or np.isnan(self.ln_current_P[index]):
            self.accept = True
            self.alpha[index] = 1
        else:
            self.alpha[index] = np.exp((lnposterior-self.ln_current_P[index]))
            if np.random.random() < self.alpha[index]:
                self.accept = True
        # if accept then update the lists
        if self.accept == True:
            self.current_pset[index] = pset
            self.acceptances += 1
            self.evaluate_constraints(res.simdata, index)
            self.list_trajactory = []      
            self.cp = []
            for i in self.current_pset[index]:
                self.cp.append(i.value)
            self.current_param_set[index] = self.cp 
            # Keep track of the overall best chain and its adaptive features
            if lnposterior > max(self.ln_current_P):
                self.mle_best = self.current_param_set[index]
                self.diffMatrix_best = self.diffMatrix[index]
                self.diff_best = self.diff[index]
            if self.iteration[index] == 0:
                self.mle_best = self.current_param_set[index]
                self.diffMatrix_best = np.eye(len(self.variables))
                self.diff_best = self.diff[index]

            # The order of varible reassignment is very important here    
            self.ln_current_P[index] = lnposterior    
            if self.config.config['parallelize_models'] != 1:
                res.out = res.simdata
            if isinstance(res.out, FailedSimulation):
                pass
            else:
                if self.config.config['output_trajectory']:
                    for l in self.output_columns:     
                        for i in res.out:
                            for j in res.out[i]:
                                if l in res.out[i][j].cols:
                                    if self.norm:
                                        res.out[i][j].normalize(self.norm)
                                    column = res.out[i][j].cols[l]
                                    self.list_trajactory = []
                                    for z in res.out[i][j].data:
                                        self.list_trajactory.append(z.data[column])      
                                    if '_Cum' in l:
                                        getFirstValue = np.concatenate((self.list_trajactory[0],np.diff(self.list_trajactory)))
                                        self.output_run_current[j+l][index]= getFirstValue
                                    else:
                                        self.output_run_current[j+l][index]= self.list_trajactory
                                    self.list_trajactory = []
                if self.config.config['output_noise_trajectory']:
                    for la in self.output_noise_columns:     
                        for ib in res.out:
                            for js in res.out[ib]:
                                if la in res.out[ib][js].cols:
                                    if self.norm:
                                        res.out[ib][js].normalize(self.norm)
                                    column = res.out[ib][js].cols[la]
                                    self.list_trajactory = []
                                    for z in res.out[ib][js].data:
                                        self.list_trajactory.append(z.data[column])      
                                    if '_Cum' in la:
                                        getFirstValue = np.concatenate(([self.list_trajactory[0]],np.diff(self.list_trajactory)))
                                        self.output_run_noise_current[js+la][index]= getFirstValue
                                    else:
                                        self.output_run_noise_current[js+la][index]= self.list_trajactory
                                    self.list_trajactory = []
                                              
        # After the burn in period start to record the accepted params for the adaptive feature.
        if self.iteration[index] >= self.burn_in:
            self.parameter_index[index][self.factor[index]] = self.current_param_set[index]
        
        # record the trajactorys for the graphs
        if self.iteration[index] >= self.valid_range and self.iteration[index] % self.config.config['sample_every'] == 0:
            # if the objective function is negbin then add the negbin noise to the traj output else record accepted sim vals as is
            if (self.config.config['objfunc'] == 'neg_bin' and self.config.config['output_noise_trajectory']) or (self.config.config['objfunc'] == 'neg_bin_dynamic' and self.config.config['output_noise_trajectory']):
                for l in self.output_noise_columns:     
                    for i in self.output_run_noise_current.keys():
                        if l in i:
                            self.output_run_noise_all[i][index][self.factor[index]] =  self.generateBinomialNoise(self.output_run_noise_current[i][index][0], self.current_pset[index])
            if self.config.config['output_trajectory']:
                for l in self.output_columns:
                    for i in self.output_run_current.keys():
                        if l in i:
                            self.output_run_all[i][index][self.factor[index]] = self.output_run_current[i][index][0]

        # Record that this individual is complete
        self.scores[index][self.factor[index]] = self.ln_current_P[index]

        # Track chain history for convergence diagnostics (R-hat, ESS)
        if self.current_pset[index] is not None:
            self.chain_history[index].append(self._param_vec(self.current_pset[index]))
            self.ln_posterior_history[index].append(self.ln_current_P[index])

        self.iteration[index] += 1

        # Standard BayesianAlgorithm sampling
        if (self.iteration[index] > self.burn_in
                and self.iteration[index] % self.sample_every == 0):
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        if (self.iteration[index] > self.burn_in
                and self.iteration[index] % (self.sample_every * self.output_hist_every) == 0):
            self.update_histograms('_%i' % self.iteration[index])

        self.wait_for_sync[index] = True
        # Wait for entire generation to finish
        if np.all(self.wait_for_sync):
            self.acceptance_rates = self.acceptances / self.attempts
            #self.wait_for_sync = [False] * self.num_parallel
            # Increase or reset the factor number and see if it's time to write things out
            for i in range(self.num_parallel):
                # self.factor[i] +=1
                # if self.factor[i] == self.arr_length:
                #     self.factor[i] = 0
                if self.iteration[i] % self.arr_length == 0 :
                    self.write_out_scores(i)
                if self.iteration[i] >= (self.burn_in -1) and self.iteration[i] <= (self.burn_in + self.adaptive):
                    if self.iteration[i] % self.arr_length == 0:
                        self.write_out_params(i)
                if self.iteration[i] > (self.burn_in + self.adaptive) and self.iteration[i] % self.config.config['sample_every'] == 0:
                    if self.iteration[i] % self.arr_length == 0:
                        self.write_out_params(i)
                if self.config.config['output_trajectory']:
                    if self.iteration[i] >= self.valid_range and self.iteration[i] % self.config.config['sample_every'] == 0:
                        if self.iteration[i] % self.arr_length == 0:
                            self.write_out_trajactorys(i)
                if self.config.config['output_noise_trajectory']:
                    if self.iteration[i] >= self.valid_range and self.iteration[i] % self.config.config['sample_every'] == 0:
                        if self.iteration[i] % self.arr_length == 0:
                            self.write_out_trajactorys_noise(i)

            # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
            if self.iteration[index] % self.diagnostics_every == 0:
                max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                if self.check_convergence(self.iteration[index], max_rhat):
                    self.combine_chains_params()
                    self.combine_chains_traj()
                    self.samples_file = self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt'
                    return 'STOP'

            # Set here because I don't want these commands to exacute more then once.
            if min(self.iteration) >= self.max_iterations:
                # Save the current postion of the MCMC run
                self.diff_best = [self.diff_best]
                np.savetxt(self.config.config['output_dir'] + '/adaptive_files/MLE_params.txt', self.mle_best)
                np.savetxt(self.config.config['output_dir'] + '/adaptive_files/diffMatrix.txt', self.diffMatrix_best)
                np.savetxt(self.config.config['output_dir'] + '/adaptive_files/diff.txt', self.diff_best)
                self.combine_chains_params()
                self.combine_chains_traj()
                self.samples_file = self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt'
                self.report_constraint_satisfaction('_final')
                return 'STOP'
            # Check if it's time to report stuff
            if self.iteration[index] % 10 == 0:
                print2('Acceptance rates: %s\n' % str(self.acceptance_rates))
                print2('Current -Ln Posteriors: %s' % str(self.ln_current_P))
            print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))

            
            # Propose next Pset
            next_generation = []
            for i, p in enumerate(self.current_pset):
                new_pset = self.pick_new_pset(i)
                if new_pset:
                    new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                    next_generation.append(new_pset)
                self.wait_for_sync[i] = False        
            return next_generation
        return []

    def generateBinomialNoise(self, timeseries, pset):
        # Generate the binomial noise for the results
        self.output = np.copy(timeseries)
        self.pset = pset
        if self.config.config['objfunc'] == 'neg_bin_dynamic':
            for p in self.pset:
                if p.name == 'r__FREE':
                    self.r = p.value
        else:
            self.r = self.config.config['neg_bin_r']
        for i in range(len(timeseries)):
            self.prob = np.clip( self.r/(self.r+timeseries[i]), 1e-10, 1-1e-10)
            self.output[i] = stats.nbinom.rvs(n=self.r, p=self.prob, size=1)

        return self.output

    def write_out_scores(self, idx):
        # Write out the scores. Need more practical method
        self.write_out_score = self.scores[idx]
        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/scores_' + str(idx) + '.txt', 'a') as f:
            np.savetxt(f, self.write_out_score)

    def write_out_params(self, idx):
        # WRite out the param. Need more practical method
        if self.iteration[idx] == self.burn_in - 1:
            self.write_out_p = self.parameter_index[idx][~(self.parameter_index[idx]==0).all(1)]
            varibles = []
            for v in self.variables:
                varibles.append(v.name)
            varNames = '\t'.join(varibles)
            with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt', 'a') as f:
                f.write(varNames+'\n')
        else:
            self.write_out_p = self.parameter_index[idx][~(self.parameter_index[idx]==0).all(1)]
            with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt', 'a') as f:
                np.savetxt(f, self.write_out_p)

    def write_out_trajactorys(self, idx):
        # write out trajectories need more practical method
        for l in self.output_columns:     
            for i in self.output_run_current.keys():
                if l in i:
                    self.write_out_t = self.output_run_all[i][idx][~(self.output_run_all[i][idx]==0).all(1)]
                    if len(self.write_out_t) != 0:
                        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_' + i + '_chain_' + str(idx) + '.txt', 'a') as f:
                            np.savetxt(f, self.write_out_t)
    def write_out_trajactorys_noise(self, idx):
        # Basically this IO on every iter is to expensice timewise
        for l in self.output_noise_columns:     
            for i in self.output_run_noise_current.keys():
                if l in i: 
                    self.write_out_t = self.output_run_noise_all[i][idx][~(self.output_run_noise_all[i][idx]==0).all(1)]
                    if len(self.write_out_t) != 0:
                        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_noise_' + i + '_chain_' + str(idx) + '.txt', 'a') as f:
                            np.savetxt(f, self.write_out_t)
    def combine_chains_params(self):
        #combine the chains for the final output file
        # if self.num_parallel != 1:
        with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt', 'w') as f:
            varsnNames = []
            for v in self.variables:
                varsnNames.append(v.name)
            varsNames = '\t'.join(varsnNames)    
            f.write(varsNames+'\n')
            for i in range(self.num_parallel):
                file_append = np.loadtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(i) + '.txt', skiprows=1)
                file_append = file_append[self.adaptive:]
                np.savetxt(f, file_append)   
        shutil.copyfile(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_params.txt', self.config.config['output_dir'] + '/adaptive_files/combined_params.txt')      
    def combine_chains_traj(self):
        # combine the trains for the file output file
        if self.num_parallel != 1:
            if self.config.config['output_trajectory']:
                for j in range(self.num_parallel):
                    for l in self.output_columns:     
                        for i in self.output_run_current.keys():
                            if l in i:
                                with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_traj_' + i + '.txt', 'a') as f:
                                    file_append = np.loadtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_' + i + '_chain_' + str(j) + '.txt')
                                    np.savetxt(f, file_append)
            if self.config.config['output_noise_trajectory']:
                for j in range(self.num_parallel):
                    for l in self.output_noise_columns:     
                        for i in self.output_run_noise_current.keys():
                            if l in i:
                                with open(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/combined_traj_noise_' + i + '.txt', 'a') as f:
                                    file_append = np.loadtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/traj_noise_' + i + '_chain_' + str(j) + '.txt')
                                    np.savetxt(f, file_append)                                     

    def pick_new_pset(self, idx):
        """
        :param idx: Index of PSet to update
        :return: A mew
        """
                   
        params = []
        for var in self.variables:
            if 'log' in var.type:
                # Work in base-10 log, consistent with how the proposal is applied
                # (FreeParameter.add -> 10**(log10(value)+summand)) and with the
                # rest of the codebase (loguniform_var dist, prior_logpdf,
                # _param_vec R-hat history, FreeParameter.diff all use log10).
                params.append(np.log10(self.current_pset[idx].get_param(var.name).value))
            else:
                params.append(self.current_pset[idx].get_param(var.name).value)
        len_params = len(params) 
        self.stablizingCov = self.config.config['stablizingCov']*np.eye(len_params)
        if self.iteration[idx] >= self.burn_in + self.adaptive:
            if self.iteration[idx] == self.burn_in + self.adaptive:
                self.parameter_index_file_input = np.genfromtxt(self.config.config['output_dir'] + '/Results/A_MCMC/Runs/params_' + str(idx) + '.txt', names = True)
                for v in self.variables:
                    if 'log' in v.type:
                        self.parameter_index_file_input[v.name] = np.log10(self.parameter_index_file_input[v.name])
                self.parameter_index_file = self.parameter_index_file_input.view((np.float64, len(self.parameter_index_file_input.dtype.names)))
                self.mu[idx] = np.reshape(np.mean(self.parameter_index_file,axis=0), [1, len_params])  # compute the mean parameters along the past chain 
                self.diffMatrix[idx] = np.matmul(self.parameter_index_file.T, self.parameter_index_file)/(self.iteration[idx] - self.burn_in)-np.matmul(self.mu[idx].T, self.mu[idx])+self.stablizingCov
                self.diff[idx] = 2.38**2/len_params
            # Weight each new sample by 1/(samples folded so far + 1). The seed
            # (above) is built from the `adaptive` post-burn-in history rows
            # (divisor iteration - burn_in == adaptive), so the running count is
            # (iteration - burn_in), NOT the global iteration. Using the global
            # counter under-weights new samples by ~(1+iteration)/(1+adaptive)
            # at the seeding step, freezing the proposal near the seed (AM-2).
            self.mu[idx] = self.mu[idx] + (1./(1+self.iteration[idx]-self.burn_in))*(params - self.mu[idx])
            self.diffVector = np.reshape(params - self.mu[idx], [1, len_params])
            self.diffMatrix[idx] = self.diffMatrix[idx] + (1./(1 + self.iteration[idx]-self.burn_in))*(np.matmul(self.diffVector.T, self.diffVector)+self.stablizingCov-self.diffMatrix[idx])
            self.diff[idx] = np.exp( np.log(self.diff[idx]) + (1./(1 + self.iteration[idx]- self.adaptive - self.burn_in))*(self.alpha[idx]-0.234))
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = np.random.multivariate_normal(mean=np.zeros((len_params,)), cov=self.diffMatrix[idx])
                delta_vector_add = {k: self.diff[idx]*delta_vector[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        if num < 10000:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                        else:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True) 
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass       
        elif self.config.config['continue_run'] == 1:
            if self.config.config['calculate_covari']:
                start_end = self.config.config['calculate_covari']
                start = int(start_end[0])
                end = int(start_end[1])
                if self.iteration[idx] == 1:
                    self.parameter_index_file_input = np.genfromtxt(self.config.config['output_dir'] + '/adaptive_files/combined_params.txt', names = True)
                    for v in self.variables:
                        if 'log' in v.type:
                            self.parameter_index_file_input[v.name] = np.log10(self.parameter_index_file_input[v.name])
                    self.parameter_index_file_range = self.parameter_index_file_input.view((np.float64, len(self.parameter_index_file_input.dtype.names)))
                    self.parameter_index_file = self.parameter_index_file_range[start:end]
                    self.mu[idx] = np.reshape(np.mean(self.parameter_index_file,axis=0), [1, len_params])  # compute the mean parameters along the past chain 
                    self.diffMatrix[idx] = (np.matmul(self.parameter_index_file.T, self.parameter_index_file)-np.matmul(self.mu[idx].T, self.mu[idx]))/(len(self.parameter_index_file_input)*0.75)
                    self.diff[idx] = self.config.config['step_size']
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = np.random.multivariate_normal(mean=np.zeros((len_params,)), cov=self.diffMatrix[idx])
                delta_vector_add = {k: self.diff[idx] * delta_vector[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        if num < 10000:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                        else:
                            new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True)      
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass       
        else:
            diffMatrix = np.eye(len_params)
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = np.random.multivariate_normal(mean=np.zeros((len_params,)), cov=diffMatrix)
                delta_vector_add = {k: self.step_size * delta_vector[i] for i,k in enumerate(oldpset.keys())}
                #delta_vector_multiply_log = {k: self.step_size*delta_vector_log[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        if num < 10000:
                            if 'log' in k.type:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                            else:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], False)
                        else:
                            if 'log' in k.type:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True) 
                            else:
                                new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], True)
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass        
    
    def update_histograms(self, file_ext):
            pass   


class ModelCheck(object):
    """
    An algorithm that just checks the fit quality for a job with no free parameters.

    Does not subclass Algorithm. To run, instead call run_check() with no Cluster.
    """

    def __init__(self, config):
        """
        Instantiates ModelCheck with a Configuration object.
        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        self.bootstrap_number = None

        logger.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        if self.config.config['simulation_dir']:
            self.sim_dir = self.config.config['simulation_dir'] + '/Simulations'
        else:
            self.sim_dir = self.config.config['output_dir'] + '/Simulations'

        # Store a list of all Model objects.
        self.model_list = copy.deepcopy(list(self.config.models.values()))

    def run_check(self, debug=False):
        """Main loop for executing the algorithm"""

        print1('Running model checking on the given model(s)')

        empty = PSet([])
        empty.name = 'check'
        job = core.Job(self.model_list, empty, 'check', self.sim_dir, self.config.config['wall_time_sim'], None,
                  None, dict(), delete_folder=False,
                  stochastic_seed_policy=self.config.config['stochastic_seed'])
        result = core.run_job(job, debug, self.sim_dir)

        if isinstance(result, FailedSimulation):
            print0('Simulation failed.')
            return

        result.normalize(self.config.config['normalization'])
        try:
            result.postprocess_data(self.config.postprocessing)
        except Exception:
            logger.exception('User-defined post-processing script failed')
            traceback.print_exc()
            print0('User-defined post-processing script failed. Exiting')
            return

        result.score = self.objective.evaluate_multiple(result.simdata, self.exp_data, result.pset,
                                                         self.config.constraints)
        if result.score is None:
            print0('Simulation contained NaN or Inf values. Cannot calculate objective value.')
            return
        print0('Objective value is %s' % result.score)
        if len(self.config.constraints) > 0:
            counter = ConstraintCounter()
            fail_count = counter.evaluate_multiple(result.simdata, self.exp_data, self.config.constraints)
            total = sum([len(cset.constraints) for cset in self.config.constraints])
            print('Satisfied %i out of %i constraints' % (total-fail_count, total))
            for cset in self.config.constraints:
                cset.output_itemized_eval(result.simdata, self.sim_dir)
