"""
Differential Evolution Adaptive Metropolis (DREAM) sampler
"""

# 1. Standard library
import logging
import re

# 2. Third-party
import numpy as np

# 3. PyBNF application/library
from ...base import BayesianAlgorithm
from ...printing import print0, print1, print2
from ...pset import PSet, OutOfBoundsException

logger = logging.getLogger(__name__)


class DreamAlgorithm(BayesianAlgorithm):
    """
    **This algorithm is a work in progress, and does not currently work correctly. In our most recent testing, it
    generates incorrect probability distributions**

    Implements a variant of the DREAM algorithm as described in Vrugt (2016) Environmental Modelling
    and Software.

    Adapts Bayesian MCMC to use methods from differential evolution for accelerated convergence and
    more efficient sampling of parameter space
    """

    def __init__(self, config):
        super(DreamAlgorithm, self).__init__(config)
        print0('You are running the DREAM algorithm. This is a work in progress, and is not officially supported! In '
               'our most recent testing, it generates incorrect probability distributions.')
        self.n_dim = len(self.variables)
        self.all_idcs = np.arange(self.n_dim)
        self.ncr = [(1+x)/self.config.config['crossover_number'] for x in range(self.config.config['crossover_number'])]
        self.g_prob = self.config.config['gamma_prob']
        self.acceptances = [0]*self.num_parallel
        self.acceptance_rates = [0.0]*self.num_parallel

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

        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        # Calculate posterior of finished job
        lnprior = self.ln_prior(pset)
        lnlikelihood = -score

        lnposterior = lnprior + lnlikelihood

        # Metropolis-Hastings criterion
        ln_p_accept = np.log10(np.random.uniform()) < min(0., lnposterior - self.ln_current_P[index])
        if ln_p_accept:  # accept update based on MH criterion
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.acceptances[index] += 1

        # Record that this individual is complete
        self.wait_for_sync[index] = True
        self.iteration[index] += 1
        self.acceptance_rates[index] = self.acceptances[index] / self.iteration[index]

        # Update histograms and trajectories if necessary
        if self.iteration[index] % self.sample_every == 0 and self.iteration[index] > self.burn_in:
            self.sample_pset(self.current_pset[index], self.ln_current_P[index])
        if (self.iteration[index] % (self.sample_every * self.output_hist_every) == 0
            and self.iteration[index] > self.burn_in):
            self.update_histograms('_%i' % self.iteration[index])

        # Wait for entire generation to finish
        if np.all(self.wait_for_sync):

            self.wait_for_sync = [False] * self.num_parallel

            if min(self.iteration) >= self.max_iterations:
                return 'STOP'

            if self.iteration[index] % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                print2('Acceptance rates: %s\n' % str(self.acceptance_rates))
            else:
                print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
            logger.info('Completed %i iterations' % self.iteration[index])
            print2('Current -Ln Posteriors: %s' % str(self.ln_current_P))

            next_gen = []
            for i, p in enumerate(self.current_pset):
                new_pset = self.calculate_new_pset(i)
                if new_pset:
                    new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                    next_gen.append(new_pset)
                else:
                    #  If new PSet is outside of variable bounds, keep current PSet and wait for next generation
                    logger.debug('Proposed PSet %s is invalid.  Rejecting and waiting until next iteration' % i)
                    self.wait_for_sync[i] = True
                    self.iteration[i] += 1

            return next_gen

        return []

    def calculate_new_pset(self, idx):
        """
        Uses differential evolution-like update to calculate new PSet

        :param idx: Index of PSet to update
        :return:
        """

        # Choose individuals (not individual to be updated) for mutation
        sel = np.random.choice(self.all_idcs[self.all_idcs != idx], 2, replace=False)
        x0 = self.current_pset[idx]
        x1 = self.current_pset[sel[0]]
        x2 = self.current_pset[sel[1]]

        # Sample the probability of modifying a parameter
        cr = np.random.choice(self.ncr)
        while True:
            ds = np.random.uniform(size=self.n_dim) <= cr  # sample parameter subspace
            if np.any(ds):
                break

        # Sample whether to jump to the mode (when gamma = 1)
        gamma = 1 if np.random.uniform() < self.g_prob else self.step_size

        new_vars = []
        for i, d in enumerate(np.random.permutation(ds)):
            k = self.variables[i]
            diff = x1.get_param(k.name).diff(x2.get_param(k.name)) if d else 0.0
            zeta = np.random.normal(0, self.config.config['zeta'])
            lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

            # Differential evolution calculation (while satisfying detailed balance)
            try:
                # Do not reflect the parameter (need to reject if outside bounds)
                new_var = x0.get_param(k.name).add(zeta + (1. + lamb) * gamma * diff, False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                logger.debug("Variable %s is outside of bounds")
                return None

        return PSet(new_vars)


