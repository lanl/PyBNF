"""BasicBayesMCMCAlgorithm — Metropolis-Hastings and Parallel Tempering (the
``mh`` and ``pt`` fit types).

mh is deprecated but still runs; pt is a working method sharing this class.
(Simulated annealing, formerly ``sa`` on this class, is now a standalone
optimizer — see optimizers/simulated_annealing.py, M2.2/ADR-0008.) Subclasses the
sampler base (BayesianAlgorithm) and inherits the run loop + execution seam from
Algorithm.
"""


from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...pset import PSet
from ...printing import print0, print1, print2, PybnfError
from ...registry import register_fit_type

from typing import Any

import logging
import numpy as np
import re


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class BasicMCMCConfig(MCMCFamilyConfig):
    """Config for the basic-MCMC codes mh/pt, co-located with the method
    (ADR-0006). Adds the one key those samplers read (``exchange_every``) on top
    of the shared family fields; the β-ladder ``postprocess`` hook is inherited.
    ``exchange_every`` is typed ``Any`` because the hook overwrites it with
    ``np.inf`` for the non-PT (mh) method (it holds both an int and an infinity).

    (``cooling``/``beta_max`` were sa-only; M2.2/ADR-0008 moved them to
    :class:`SimulatedAnnealingConfig` when sa became a standalone optimizer.)
    """

    exchange_every: Any = 20

    # reps_per_beta is a pt input the beta-ladder postprocess reads (and BasicMCMC's
    # __init__ reads after it defaults to 1); shared by mh/pt, who share this class and
    # so share its valid-key surface. Unioned onto the family's beta_range (#401).
    RUNTIME_KEYS = frozenset({'reps_per_beta'})


# Two codes share this class: pt is a working sampler; mh (= pt with
# exchange_every=inf) is deprecated but still runs. sa was historically a third
# code on this class (kwargs sa=True); M2.2 (ADR-0008) rewrote it as a true
# optimizer in optimizers/simulated_annealing.py, so it no longer registers here.
@register_fit_type('pt', family='sampler', display_name='Parallel Tempering MCMC',
                   schema=BasicMCMCConfig)
@register_fit_type('mh', family='sampler', display_name='Metropolis-Hastings MCMC',
                   deprecated=True, schema=BasicMCMCConfig)
class BasicBayesMCMCAlgorithm(BayesianAlgorithm):

    """
    Implements a Bayesian Markov chain Monte Carlo simulation.
    This is essentially a non-parallel algorithm, but here, we run n instances in parallel, and pool all results.
    This will give you a best fit (which is maybe not great), but more importantly, generates an extra result file
    that gives the probability distribution of each variable.
    This distribution depends on the prior, which is specified according to the variable initialization rules.
    """

    def __init__(self, config):  # expdata, objective, priorfile, gamma=0.1):
        super(BasicBayesMCMCAlgorithm, self).__init__(config)

        self.exchange_every = config.config['exchange_every']
        self.pt = self.exchange_every != np.inf
        self.reps_per_beta = self.config.config['reps_per_beta']
        self.betas_per_group = self.num_parallel // self.reps_per_beta  # Number of unique betas considered (in PT)

        # The temperature of each replicate
        # For MCMC, probably n copies of the same number, unless the user set it up strangely
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

        return super(BasicBayesMCMCAlgorithm, self).start_run(setup_samples=True)

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
                if self.iteration[index] % self.diagnostics_every == 0:
                    max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                    if self.check_convergence(self.iteration[index], max_rhat):
                        self.converged = True
                        return None
                logger.debug('Completed %i iterations' % self.iteration[index])
                logger.debug('Current move accept rate: %f' % (self.accepted/self.attempts))
                if self.exchange_attempts > 0:
                    logger.debug('Current replica exchange rate: %f' % (self.exchange_accepted / self.exchange_attempts))
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
