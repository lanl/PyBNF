"""DreamAlgorithm — the DREAM(ZS) sampler (the ``dream`` fit type).

DiffeRential Evolution Adaptive Metropolis drawing proposal donors from a
growing ZS archive (Vrugt 2016). Extracted byte-identical (M1 Step 4).
Subclasses the sampler base (BayesianAlgorithm) and inherits the run loop +
execution seam from Algorithm; it makes no core.* call of its own.
"""


from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...pset import PSet, OutOfBoundsException
from ...printing import print1, print2
from ...registry import register_fit_type

from typing import Any, Optional

import logging
import numpy as np
import re
import copy
from pydantic import Field
from scipy import stats


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class DreamConfig(MCMCFamilyConfig):
    """Config for DREAM(ZS) (dream), co-located with the method (ADR-0006). Adds the
    DREAM-specific keys on top of the shared family fields; the β-ladder
    ``postprocess`` hook is inherited. ``PDreamConfig`` (p_dream) subclasses this and
    adds ``precondition_adapt``. ``adaptive_step_size`` is typed ``Any`` so a user
    int 0/1 stays an int (matching the old behavior). Values byte-identical to the
    old global defaults.
    """

    gamma_prob: float = 0.1
    zeta: float = 1e-6
    lambda_: float = Field(0.1, alias='lambda')
    crossover_number: int = 3
    adaptive_step_size: Any = True    # bool default; user may pass int 0/1
    archive_size: Optional[int] = None
    archive_thin_rate: int = 10
    snooker_prob: float = 0.1
    delta: int = 1
    outlier_method: str = 'iqr'

    @classmethod
    def postprocess(cls, conf_dict, fit_type):
        """The MCMC β-ladder (inherited from ``MCMCFamilyConfig``) plus the
        DREAM-family ``step_size`` coupling (ADR-0013): a user-set ``step_size``
        pins ``adaptive_step_size`` off. Both keys are owned only by DREAM/P-DREAM,
        so the coupling is intra-family -- it formerly lived as a global write in
        ``config.py`` that, post-narrowing, would have stamped an orphan
        ``adaptive_step_size`` onto any ``mh``/``am``/``sa`` fit that set
        ``step_size``. ``PDreamConfig`` inherits this (it owns both keys too)."""
        super().postprocess(conf_dict, fit_type)
        if 'step_size' in conf_dict:
            conf_dict['adaptive_step_size'] = False
        return conf_dict


@register_fit_type('dream', family='sampler', display_name='DREAM(ZS)',
                   schema=DreamConfig)
class DreamAlgorithm(BayesianAlgorithm):
    """
    Implements a variant of the DREAM algorithm as described in Vrugt (2016) Environmental Modelling
    and Software.

    Adapts Bayesian MCMC to use methods from differential evolution for accelerated convergence and
    more efficient sampling of parameter space
    """

    def __init__(self, config):
        super().__init__(config)
        self.ncr = [(1+x)/self.config.config['crossover_number'] for x in range(self.config.config['crossover_number'])]
        self.ncr_count = len(self.ncr)
        self.g_prob = self.config.config['gamma_prob']
        self.adaptive_step_size = config.config['adaptive_step_size']
        self.acceptances = [0]*self.num_parallel
        self.acceptance_rates = [0.0]*self.num_parallel

        # CR adaptation state
        self.cr_probs = np.ones(self.ncr_count) / self.ncr_count
        self.cr_total_distance = np.zeros(self.ncr_count)
        self.cr_usage_count = np.zeros(self.ncr_count)
        self.cr_adapt_end = self.burn_in // 2
        self.cr_frozen = False

        # Per-generation tracking for CR adaptation
        self.gen_cr_indices = [None] * self.num_parallel
        self.gen_x_old = [None] * self.num_parallel
        self.gen_x_std = np.ones(self.n_dim)

        # ZS archive: external archive of past states for proposal generation
        m0 = config.config['archive_size']
        self.archive_m0 = m0 if m0 is not None else 10 * self.n_dim
        self.archive_thin_rate = config.config['archive_thin_rate']
        self.archive = []  # list of PSet objects

        # Snooker update
        self.snooker_prob = config.config['snooker_prob']
        self.gen_log_snooker_correction = [0.0] * self.num_parallel

        # Multiple chain pairs
        self.delta = config.config['delta']

        # Outlier detection method
        self.outlier_method = config.config['outlier_method']

    def start_run(self, setup_samples=True):
        first_psets = super().start_run(setup_samples)
        # Initialize the ZS archive with m0 random draws from the prior
        self.archive = [self.random_pset() for _ in range(self.archive_m0)]
        logger.info('Initialized ZS archive with %d entries (d=%d)' % (self.archive_m0, self.n_dim))
        return first_psets

    def calculate_snooker_pset(self, idx):
        """
        Snooker update proposal (ter Braak & Vrugt, 2008).
        Projects archive points onto the line through the current state and a reference archive point,
        then jumps along that axis.

        Returns (PSet or None, log_correction) where log_correction is the log of the Hastings
        correction factor (d-1)*log(||Xp - Zc|| / ||X - Zc||).
        """
        x0 = self.current_pset[idx]
        x0_vec = self._param_vec(x0)

        # Draw three distinct archive indices: c (reference), a, b (for projection difference)
        sel = np.random.choice(len(self.archive), 3, replace=False)
        zc_vec = self._param_vec(self.archive[sel[0]])
        za_vec = self._param_vec(self.archive[sel[1]])
        zb_vec = self._param_vec(self.archive[sel[2]])

        # Snooker axis: line through x0 and zc
        axis = x0_vec - zc_vec
        axis_norm_sq = np.dot(axis, axis)
        if axis_norm_sq < 1e-20:
            return None, 0.0

        # Project za and zb onto the snooker axis
        za_proj = zc_vec + axis * (np.dot(za_vec - zc_vec, axis) / axis_norm_sq)
        zb_proj = zc_vec + axis * (np.dot(zb_vec - zc_vec, axis) / axis_norm_sq)

        # Jump vector along the axis
        diff_proj = za_proj - zb_proj

        # Gamma for snooker: U(1.2, 2.2) per Vrugt (2016)
        gamma_s = np.random.uniform(1.2, 2.2)

        # Small perturbations
        zeta_vec = np.random.normal(0, self.config.config['zeta'], size=self.n_dim)
        lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

        xp_vec = x0_vec + zeta_vec + (1.0 + lamb) * gamma_s * diff_proj

        # Build the proposed PSet
        new_vars = []
        for i, v in enumerate(self.variables):
            try:
                if v.log_space:
                    new_var = v.set_value(10**xp_vec[i], reflect=False)
                else:
                    new_var = v.set_value(xp_vec[i], reflect=False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                return None, 0.0

        # Hastings correction: (||Xp - Zc|| / ||X - Zc||)^(d-1).
        # dist_x0_zc = sqrt(axis_norm_sq) is already guaranteed nonzero by the
        # axis_norm_sq < 1e-20 check above, so no divide-by-zero guard is needed here.
        dist_xp_zc = np.linalg.norm(xp_vec - zc_vec)
        dist_x0_zc = np.linalg.norm(x0_vec - zc_vec)
        log_correction = (self.n_dim - 1) * np.log(dist_xp_zc / dist_x0_zc)

        return PSet(new_vars), log_correction

    def _detect_outliers_iqr(self, mean_ln_p):
        """IQR outlier detection: chains below Q25 - 2*IQR are outliers."""
        Q75 = np.percentile(mean_ln_p, 75)
        Q25 = np.percentile(mean_ln_p, 25)
        iqr = Q75 - Q25
        return np.where(mean_ln_p < Q25 - 2.0 * iqr)[0]

    def _detect_outliers_grubbs(self, mean_ln_p):
        """Grubbs test for a single minimum outlier at significance alpha=0.01."""
        N = len(mean_ln_p)
        if N < 3:
            return np.array([], dtype=int)
        mu = np.mean(mean_ln_p)
        sd = np.std(mean_ln_p, ddof=1)
        if sd < 1e-20:
            return np.array([], dtype=int)
        G = (mu - np.min(mean_ln_p)) / sd
        alpha = 0.01
        t_crit_sq = stats.t.ppf(alpha / (2 * N), N - 2) ** 2
        T_c = (N - 1) / np.sqrt(N) * np.sqrt(t_crit_sq / (N - 2 + t_crit_sq))
        if G > T_c:
            return np.array([np.argmin(mean_ln_p)])
        return np.array([], dtype=int)

    def detect_and_reset_outliers(self):
        """
        Detect outlier chains using the configured method on mean log-posteriors
        (last 50% of history). Reset outlier chains to copies of randomly selected
        non-outlier chains.

        Methods: 'iqr' (interquartile range), 'grubbs' (Grubbs test at alpha=0.01).
        """
        min_len = min(len(h) for h in self.ln_posterior_history)
        start = min_len // 2
        if min_len - start < 5:
            return

        mean_ln_p = np.array([
            np.mean(self.ln_posterior_history[j][start:min_len]) for j in range(self.num_parallel)
        ])

        if self.outlier_method == 'grubbs':
            outlier_indices = self._detect_outliers_grubbs(mean_ln_p)
        else:
            outlier_indices = self._detect_outliers_iqr(mean_ln_p)

        if len(outlier_indices) == 0:
            return

        good_indices = [i for i in range(self.num_parallel) if i not in outlier_indices]
        if len(good_indices) == 0:
            return

        for out_idx in outlier_indices:
            donor_idx = np.random.choice(good_indices)
            logger.warning('Outlier chain %d reset to chain %d at iteration %d (method=%s)'
                           % (out_idx, donor_idx, self.iteration[out_idx], self.outlier_method))
            self.current_pset[out_idx] = copy.deepcopy(self.current_pset[donor_idx])
            self.ln_current_P[out_idx] = self.ln_current_P[donor_idx]
            self.ln_posterior_history[out_idx][start:min_len] = self.ln_posterior_history[donor_idx][start:min_len]
            self.chain_history[out_idx][start:min_len] = self.chain_history[donor_idx][start:min_len]

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

        m = re.search(r'(?<=run)\d+', pset.name)
        index = int(m.group(0))

        # Calculate posterior of finished job
        lnprior = self.ln_prior(pset)
        lnlikelihood = -score

        lnposterior = lnprior + lnlikelihood

        # Metropolis-Hastings criterion (includes snooker Hastings correction when applicable)
        ln_p_accept = min(0., lnposterior - self.ln_current_P[index]
                          + self.gen_log_snooker_correction[index])
        if np.log(np.random.uniform()) < ln_p_accept:  # accept update based on MH criterion
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.acceptances[index] += 1
            self.evaluate_constraints(res.simdata, index)

        # Store chain history (after accept/reject, so it reflects the kept state)
        self.chain_history[index].append(self._param_vec(self.current_pset[index]))
        self.ln_posterior_history[index].append(self.ln_current_P[index])

        # CR adaptation: compute standardized distance traveled
        if not self.cr_frozen and self.gen_cr_indices[index] is not None:
            x_new_vec = self._param_vec(self.current_pset[index])
            if self.gen_x_old[index] is not None:
                diff_vec = x_new_vec - self.gen_x_old[index]
                sd_dist = np.sum((diff_vec / np.maximum(self.gen_x_std, 1e-10)) ** 2)
                self.cr_total_distance[self.gen_cr_indices[index]] += sd_dist
                self.cr_usage_count[self.gen_cr_indices[index]] += 1

        # Record that this individual is complete
        self.wait_for_sync[index] = True
        self.iteration[index] += 1
        self.acceptance_rates[index] = self.acceptances[index] / self.iteration[index]

        # Update histograms and trajectories if necessary
        if self.iteration[index] % self.sample_every == 0 and self.iteration[index] > self.burn_in:
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        if (self.iteration[index] % (self.sample_every * self.output_hist_every) == 0
            and self.iteration[index] > self.burn_in):
            self.update_histograms('_%i' % self.iteration[index])

        # Wait for entire generation to finish
        # Loop handles the case where all proposals are out of bounds: advance
        # the generation counter and try again instead of returning an empty
        # list (which would exhaust the job pool and silently end the run).
        while np.all(self.wait_for_sync):

            self.wait_for_sync = [False] * self.num_parallel

            if min(self.iteration) >= self.max_iterations:
                self.update_histograms('_final')
                self.report_constraint_satisfaction('_final')
                return 'STOP'

            if self.iteration[index] % 10 == 0:
                print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
                print2('Acceptance rates: %s\n' % str(self.acceptance_rates))
            else:
                print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
            # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
            if self.iteration[index] % self.diagnostics_every == 0:
                max_rhat = self.report_convergence_diagnostics(self.iteration[index])
                if self.check_convergence(self.iteration[index], max_rhat):
                    return 'STOP'
            logger.debug('Completed %i iterations' % self.iteration[index])
            print2('Current -Ln Posteriors: %s' % str(self.ln_current_P))

            # Outlier detection (every 10 iterations, only during burn-in)
            if self.iteration[index] % 10 == 0 and self.iteration[index] <= self.burn_in:
                self.detect_and_reset_outliers()

            # CR adaptation: update probabilities
            if (self.iteration[index] % 10 == 0
                    and self.iteration[index] <= self.cr_adapt_end
                    and not self.cr_frozen):
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_dist = self.cr_total_distance / np.maximum(self.cr_usage_count, 1)
                if np.sum(mean_dist) > 0:
                    self.cr_probs = mean_dist / np.sum(mean_dist)
                    logger.debug('Updated CR probabilities: %s' % str(self.cr_probs))
            elif self.iteration[index] > self.cr_adapt_end and not self.cr_frozen:
                self.cr_frozen = True
                logger.debug('CR probabilities frozen at iteration %d: %s'
                            % (self.iteration[index], str(self.cr_probs)))

            # Grow the ZS archive: every K generations, append current chain states
            if self.iteration[index] % self.archive_thin_rate == 0:
                for i in range(self.num_parallel):
                    self.archive.append(copy.deepcopy(self.current_pset[i]))
                logger.debug('Archive grown to %d entries at iteration %d'
                            % (len(self.archive), self.iteration[index]))

            # Save old states and compute population std for CR adaptation
            for i in range(self.num_parallel):
                self.gen_x_old[i] = self._param_vec(self.current_pset[i])
            all_vecs = np.array(self.gen_x_old)
            self.gen_x_std = np.std(all_vecs, axis=0)

            next_gen = []
            for i, p in enumerate(self.current_pset):
                if np.random.uniform() < self.snooker_prob:
                    # Snooker update
                    new_pset, log_corr = self.calculate_snooker_pset(i)
                    self.gen_log_snooker_correction[i] = log_corr
                    self.gen_cr_indices[i] = None  # no CR for snooker
                else:
                    # Parallel direction update
                    new_pset, cr_idx = self.calculate_new_pset(i)
                    self.gen_log_snooker_correction[i] = 0.0
                    self.gen_cr_indices[i] = cr_idx
                if new_pset:
                    new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                    next_gen.append(new_pset)
                else:
                    # Out-of-bounds proposal: treat as a Metropolis rejection.
                    # Record the current state in chain history (chain stays in place)
                    # so that diagnostics (R-hat, ESS) correctly reflect the non-movement.
                    logger.debug('Proposed PSet for chain %d is out of bounds. Treating as rejection.' % i)
                    self.chain_history[i].append(self._param_vec(self.current_pset[i]))
                    self.ln_posterior_history[i].append(self.ln_current_P[i])
                    self.wait_for_sync[i] = True
                    self.iteration[i] += 1
                    self.acceptance_rates[i] = self.acceptances[i] / self.iteration[i]
                    if self.iteration[i] % self.sample_every == 0 and self.iteration[i] > self.burn_in:
                        self.sample_pset(self.current_pset[i], self.ln_current_P[i], i)

            if not next_gen:
                logger.warning('All %d proposals were out of bounds at iteration %d. '
                               'Advancing to next generation.'
                               % (self.num_parallel, min(self.iteration)))
                continue

            return next_gen

        return []

    def calculate_new_pset(self, idx):
        """
        Uses differential evolution-like update to calculate new PSet.
        Returns (PSet, cr_idx) or (None, cr_idx) if the proposal is out of bounds.

        :param idx: Index of PSet to update
        :return: tuple of (PSet or None, int)
        """

        x0 = self.current_pset[idx]

        # Draw 2*delta donor states from the ZS archive (without replacement)
        sel = np.random.choice(len(self.archive), 2 * self.delta, replace=False)

        # Sample crossover value and mask
        cr_idx = np.random.choice(self.ncr_count, p=self.cr_probs)
        cr = self.ncr[cr_idx]
        while True:
            ds = np.random.uniform(size=self.n_dim) <= cr  # sample parameter subspace
            if np.any(ds):
                break

        # Gamma selection: mode jump (gamma=1) or adaptive/fixed step size
        if np.random.uniform() < self.g_prob:
            gamma = 1
            ds[:] = True  # mode jump updates all dimensions
        else:
            d_prime = int(np.sum(ds))
            if self.adaptive_step_size:
                gamma = 2.38 / np.sqrt(2.0 * self.delta * d_prime)
            else:
                gamma = self.step_size

        new_vars = []
        for i, d in enumerate(ds):
            k = self.variables[i]
            if d:
                # Sum of delta difference vectors: sum_{j=1}^{delta} (Z_a_j - Z_b_j)
                total_diff = 0.0
                for j in range(self.delta):
                    total_diff += self.archive[sel[j]].get_param(k.name).diff(
                        self.archive[sel[self.delta + j]].get_param(k.name))
            else:
                total_diff = 0.0
            zeta = np.random.normal(0, self.config.config['zeta'])
            lamb = np.random.uniform(-self.config.config['lambda'], self.config.config['lambda'])

            # Differential evolution calculation (while satisfying detailed balance)
            try:
                # Do not reflect the parameter (need to reject if outside bounds)
                new_var = x0.get_param(k.name).add(zeta + (1. + lamb) * gamma * total_diff, False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                logger.debug("Variable %s is outside of bounds")
                return None, cr_idx

        return PSet(new_vars), cr_idx
