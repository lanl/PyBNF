"""Simulated Annealing optimizer (the ``sa`` fit type).

A clean rewrite-to-spec (M2.2, ADR-0008). ``sa`` was historically built on the
Bayesian sampler base and evaluated the *posterior* (prior + likelihood) in its
Metropolis accept -- a silent MAP estimator for normal/lognormal priors, unlike
every other PyBNF optimizer. It is now a true optimizer: it **minimizes the raw
objective**, using the prior only for the initial random draw (like de/pso/ss/sim).
Box constraints are enforced by proposal reflection (``FreeParameter.add``) plus
the parameter bounds, exactly as before.

The algorithm runs ``population_size`` independent annealing chains in parallel.
Each chain makes a fixed-magnitude random-walk proposal and a Metropolis accept at
its own inverse temperature ``beta``; an accepted *uphill* (objective-increasing)
move cools the chain (``beta += cooling``). A chain finishes when its ``beta``
reaches ``beta_max`` or it runs ``max_iterations``; the run stops once every chain
has finished. ``sa`` is deprecated (it warns at dispatch) but still runs.
"""


from ..base import Algorithm
from ...config_schema import PyBNFConfigModel
from ...pset import PSet
from ...printing import print0, print1, print2, PybnfError
from ...registry import register_fit_type

from pydantic import Field

import logging
import numpy as np
import re


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class SimulatedAnnealingConfig(PyBNFConfigModel):
    """Simulated-annealing config fields, co-located with the method (ADR-0002,
    ADR-0008). Standalone -- ``sa`` is an optimizer, no longer part of the MCMC
    family -- carrying only the four knobs the algorithm reads. The defaults are
    byte-identical to the values these keys held under the old MCMC-family schema;
    under narrowing (ADR-0013) they appear only in an ``sa`` fit's effective config.
    """

    step_size: float = 0.2
    beta: list = Field(default_factory=lambda: [1.0])
    cooling: float = 0.01
    beta_max: float = float('inf')


@register_fit_type('sa', family='optimizer', display_name='Simulated Annealing',
                   deprecated=True, schema=SimulatedAnnealingConfig)
class SimulatedAnnealing(Algorithm):
    """Simulated annealing as a true optimizer (ADR-0008): minimizes the raw
    objective across ``population_size`` independent annealing chains."""

    def __init__(self, config):
        super(SimulatedAnnealing, self).__init__(config)
        self.num_parallel = self.config.config['population_size']
        self.step_size = self.config.config['step_size']
        self.cooling = self.config.config['cooling']
        self.beta_max = self.config.config['beta_max']
        self.betas = self._initial_betas(self.config.config['beta'])

        # Per-chain state. current_score is the raw objective at the chain's
        # current point; np.inf until the chain's first result, so the first
        # result is always accepted.
        self.current_pset = [None] * self.num_parallel
        self.current_score = [np.inf] * self.num_parallel
        self.iteration = [0] * self.num_parallel
        self.finished = [False] * self.num_parallel

        self.attempts = 0
        self.accepted = 0
        self.staged = []  # PSets queued to resume chains after add_iterations

    def _initial_betas(self, beta):
        """One starting inverse-temperature per chain. A single value is
        broadcast to every chain; otherwise the list must give one value per
        chain (population_size)."""
        if len(beta) == 1:
            return [float(beta[0])] * self.num_parallel
        if len(beta) == self.num_parallel:
            return [float(b) for b in beta]
        raise PybnfError(
            'For simulated annealing, the beta key must be a single value or one '
            'value per replicate (population_size = %i, but got %i beta values).'
            % (self.num_parallel, len(beta)))

    def reset(self, bootstrap=None):
        super(SimulatedAnnealing, self).reset(bootstrap)
        self.current_pset = [None] * self.num_parallel
        self.current_score = [np.inf] * self.num_parallel
        self.iteration = [0] * self.num_parallel
        self.finished = [False] * self.num_parallel
        self.attempts = 0
        self.accepted = 0
        self.staged = []

    def start_run(self):
        print2('Running simulated annealing on %i independent replicates in '
               'parallel, for up to %i iterations each or until each replicate\'s '
               '1/T reaches %s.' % (self.num_parallel, self.max_iterations, self.beta_max))
        if self.config.config['initialization'] == 'lh':
            first_psets = self.random_latin_hypercube_psets(self.num_parallel)
        else:
            first_psets = [self.random_pset() for _ in range(self.num_parallel)]
        for i in range(self.num_parallel):
            first_psets[i].name = 'iter0run%i' % i
        return first_psets

    def got_result(self, res):
        index = int(re.search(r'(?<=run)\d+', res.pset.name).group(0))
        score = res.score

        self.attempts += 1
        first = self.current_pset[index] is None
        # Metropolis accept for MINIMIZING the raw objective at inverse temp beta:
        # ln_p_accept = min(0, current_score - score) is 0 (a downhill move, always
        # accepted) or negative (an uphill move, accepted w.p. exp(beta*ln_p_accept)).
        ln_p_accept = 0.0 if first else min(0.0, self.current_score[index] - score)
        if first or np.random.rand() < np.exp(ln_p_accept * self.betas[index]):
            self.accepted += 1
            self.current_pset[index] = res.pset
            self.current_score[index] = score
            if ln_p_accept < 0.0:
                # Accepted an uphill move -> cool this chain.
                self.betas[index] += self.cooling
                if self.betas[index] >= self.beta_max:
                    return self._finish_chain(index, 'beta_max was reached')

        proposed = self._propose(index)
        if proposed is None:
            return self._finish_chain(index, 'max_iterations was reached')
        proposed.name = 'iter%irun%i' % (self.iteration[index], index)
        if self.staged:
            out = [proposed] + self.staged
            self.staged = []
            return out
        return [proposed]

    def _finish_chain(self, index, reason):
        """Mark chain ``index`` done. Returns ``'STOP'`` once every chain is
        done, else ``[]`` (this chain idles while the others keep running)."""
        if not self.finished[index]:
            self.finished[index] = True
            logger.info('Finished replicate %i because %s.' % (index, reason))
            print2('Finished replicate %i because %s.' % (index, reason))
        if all(self.finished):
            if self.attempts:
                print0('Overall move accept rate: %f' % (self.accepted / self.attempts))
            return 'STOP'
        return []

    def _propose(self, index):
        """Advance chain ``index`` by one iteration and return its next PSet, or
        ``None`` if the chain has reached ``max_iterations``."""
        self.iteration[index] += 1
        if self.iteration[index] == min(self.iteration):
            if self.iteration[index] % self.config.config['output_every'] == 0:
                self.output_results()
            if self.iteration[index] % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                if self.attempts:
                    print2('Current move accept rate: %f' % (self.accepted / self.attempts))
            logger.debug('Completed %i iterations' % self.iteration[index])
        if self.iteration[index] >= self.max_iterations:
            logger.info('Finished replicate number %i' % index)
            print2('Finished replicate number %i' % index)
            return None
        return self.choose_new_pset(self.current_pset[index])

    def choose_new_pset(self, oldpset):
        """Perturb ``oldpset`` by a fixed-magnitude (``step_size``) random-walk
        move. The direction is isotropic; any component that would leave the box
        is reflected back inside (``FreeParameter.add`` defaults to reflect=True),
        so the symmetric proposal keeps the plain Metropolis ratio valid."""
        delta = {k: np.random.normal() for k in oldpset.keys()}
        magnitude = np.sqrt(sum(x ** 2 for x in delta.values()))
        normalized = {k: self.step_size * delta[k] / magnitude for k in oldpset.keys()}
        return PSet([v.add(normalized[v.name]) for v in oldpset])

    def add_iterations(self, n):
        """Extend the budget by ``n`` iterations, restarting any chain that had
        already finished at ``max_iterations`` (mirrors the resume behavior of the
        other iterative methods); chains that finished by reaching ``beta_max``
        stay finished."""
        oldmax = self.max_iterations
        self.max_iterations += n
        for index in range(self.num_parallel):
            if self.iteration[index] >= oldmax and self.current_pset[index] is not None \
                    and self.betas[index] < self.beta_max:
                self.finished[index] = False
                ps = self.choose_new_pset(self.current_pset[index])
                ps.name = 'iter%irun%i' % (self.iteration[index], index)
                self.staged.append(ps)
