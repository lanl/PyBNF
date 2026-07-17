"""DreamAlgorithm — the DREAM(ZS) sampler (the ``dream`` fit type).

DiffeRential Evolution Adaptive Metropolis drawing proposal donors from a
growing ZS archive (Vrugt 2016). Extracted byte-identical (M1 Step 4).
Subclasses the sampler base (BayesianAlgorithm) and inherits the run loop +
execution seam from Algorithm; it makes no core.* call of its own.
"""


from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...pset import PSet, OutOfBoundsException
from ...printing import print1, print2, PybnfError
from ...registry import register_fit_type

from typing import Any, Optional

import logging
import re
import numpy as np
import copy
from pydantic import Field
from scipy import stats
from scipy.special import logsumexp


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
    # Proposal operator (ADR-0067 axis 1). 'de' is the classic DREAM(ZS)
    # parallel-direction proposal; 'whitened' is the covariance-preconditioned
    # proposal (the p_dream default, pinned by ``PDreamConfig``). Additional
    # operators (e.g. 'kalman') arrive in later ADR-0067 stages.
    proposal: str = 'de'
    # Multi-try evaluation protocol (ADR-0067 axis 2, MT-DREAM(ZS); Laloy &
    # Vrugt 2012). n_try = 1 is the classic single-try engine (byte-identical to
    # before this key). n_try = k > 1 draws k candidate proposals per chain per
    # generation, selects one in proportion to its posterior importance weight,
    # and accepts via a reference-set (multiple-try) Metropolis ratio. Composes
    # with every ``proposal`` value.
    n_try: int = 1

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

        # Output-augmented archive (ADR-0067 Stage 3, axis 2b -- "implied"). When a
        # proposal declares it needs model outputs (the 'kalman' proposal; see
        # _archive_stores_outputs, set after the proposal is resolved below), each
        # archive entry also carries the model's output vector f(Z), stored in
        # archive_outputs in lockstep with archive. The accepted state's output is
        # cached per chain at accept time (current_output_vec, mirroring
        # current_pointwise_loglik) and appended at archive growth. Both stay dormant
        # for the 'de'/'whitened' proposals (the gate is False), so a plain run stores
        # only PSets exactly as before -- carrying output vectors for a proposal that
        # never reads them would spend memory/bandwidth for nothing.
        self.archive_outputs = []                        # f(Z) per archive entry (or None)
        self.current_output_vec = [None] * self.num_parallel  # accepted f(x) per chain
        self._archive_stores_outputs = False             # set True for 'kalman' below

        # Snooker update
        self.snooker_prob = config.config['snooker_prob']
        self.gen_log_snooker_correction = [0.0] * self.num_parallel

        # Multiple chain pairs
        self.delta = config.config['delta']

        # Outlier detection method
        self.outlier_method = config.config['outlier_method']

        # Proposal operator (ADR-0067 axis 1). 'whitened' folds in what used to be
        # the separate PDreamAlgorithm: DE proposals computed in a covariance-
        # whitened space. The preconditioning state below is dormant for 'de'
        # (guarded everywhere by ``self.proposal == 'whitened'``), so a plain
        # ``dream`` run is byte-identical to before this fold-in.
        self.proposal = config.config['proposal']
        if self.proposal not in ('de', 'whitened'):
            raise PybnfError("Invalid proposal '%s'" % self.proposal,
                             "Config key 'proposal' must be 'de' or 'whitened'.")

        # Multi-try count (ADR-0067 axis 2). n_try == 1 is the classic engine;
        # n_try > 1 activates the multiple-try (MT-DREAM(ZS)) barrier in
        # got_result. Everything guarded on ``self.n_try > 1`` is dormant at the
        # default, so a plain run is byte-identical to before this key.
        self.n_try = config.config['n_try']
        if not isinstance(self.n_try, int) or self.n_try < 1:
            raise PybnfError("Invalid n_try '%s'" % self.n_try,
                             "Config key 'n_try' must be an integer >= 1.")
        # Per-chain two-phase multi-try state (dormant unless n_try > 1). A
        # generation for chain i runs TRIALS (k candidates from x) -> select the
        # winner Y in proportion to its importance weight -> REFERENCE (k-1 draws
        # from Y; the k-th reference is x itself) -> multiple-try accept. See
        # _got_result_multitry.
        n = self.num_parallel
        self.mt_is_snooker = [False] * n      # this generation's proposal kind (per chain)
        self.mt_trials_expected = [0] * n     # # of emitted (in-bounds) trials awaited
        self.mt_trial_meta = [[] for _ in range(n)]   # per emitted trial: {cand_term, anchor, cr_idx}
        self.mt_trials = [[] for _ in range(n)]       # buffered arrived trials (dicts)
        self.mt_selected = [None] * n         # the chosen candidate Y (dict)
        self.mt_ref_terms = [[] for _ in range(n)]    # per emitted reference: snooker log-term
        self.mt_refs_expected = [0] * n       # # of emitted (in-bounds) references awaited
        self.mt_ref_logw = [[] for _ in range(n)]     # buffered arrived reference log-weights
        self.mt_cur_logw = [0.0] * n          # current-state (k-th) reference log-weight
        self._cov_L = None       # Cholesky factor of the covariance estimate
        self._cov_L_inv = None   # Inverse of the Cholesky factor (whitening matrix)
        self._preconditioned = False
        self.precondition_adapt = None
        if self.proposal == 'whitened':
            # precondition_adapt lives on PDreamConfig (p_dream's companion key);
            # read defensively so a base ``dream`` that opts into 'whitened' still
            # gets the documented burn_in//2 fallback.
            pa = config.config.get('precondition_adapt')
            self.precondition_adapt = pa if pa is not None else self.burn_in // 2

    def start_run(self, setup_samples=True):
        first_psets = super().start_run(setup_samples)
        # Initialize the ZS archive with m0 random draws from the prior
        self.archive = [self.random_pset() for _ in range(self.archive_m0)]
        # The initial archive is random prior draws with no evaluated model output, so
        # their output slots are None (ADR-0067 Stage 3). Kept index-aligned with archive;
        # the output-augmented proposals (kalman) draw their ensemble only from entries
        # whose output is not None (the grown, accepted states).
        self.archive_outputs = [None] * self.archive_m0
        logger.info('Initialized ZS archive with %d entries (d=%d)' % (self.archive_m0, self.n_dim))
        return first_psets

    def _proposal_pset(self, xp_vec):
        """Materialize a proposal vector (in sampling space ``u``) into a PSet
        *without reflection* — DREAM rejects an out-of-bounds proposal rather
        than folding it back into the box. Returns the PSet, or ``None`` if any
        coordinate falls outside its parameter's bounds.

        Shared by the snooker proposal and P-DREAM's whitened proposal, which
        differ only in how ``xp_vec`` is built and in their Hastings/return
        bookkeeping (kept in each method, per ADR-0009).
        """
        try:
            # The shared inverse bridge maps each coordinate back via its scale
            # (FreeParameter.from_sampling_space); reflect=False makes an
            # out-of-box coordinate raise, which we turn into a rejected proposal.
            return self._pset_from_u(xp_vec, reflect=False)
        except OutOfBoundsException:
            return None

    def record_accepted_output(self, res, index):
        """Cache the just-accepted state's model output vector f(x) for chain ``index``
        (ADR-0067 Stage 3), mirroring :meth:`record_pointwise_loglik`: computed here,
        where the accepted pset's simulation data is in hand, and consumed later at
        archive growth (:meth:`_advance_generation`) and by the Kalman proposal.

        A no-op unless the proposal declared it needs archived outputs
        (``_archive_stores_outputs``; the 'kalman' proposal). Reads the simdata through
        :meth:`_result_simdata` so it works whether the objective was scored worker-side
        or master-side (lanl/PyBNF#480), and extracts the aligned output vector via the
        objective's :meth:`~pybnf.objective.ObjectiveFunction.aligned_prediction_data`
        seam. Any failure is logged, never fatal -- the proposal falls back to ``de``
        when a chain lacks a cached output."""
        if not self._archive_stores_outputs:
            return
        try:
            aligned = self.objective.aligned_prediction_data(
                self._result_simdata(res), self.exp_data, res.pset)
        except Exception:
            logger.debug('Could not extract model output vector for chain %d',
                         index, exc_info=True)
            return
        if aligned is None:
            return
        prediction, _observation, _variance = aligned
        self.current_output_vec[index] = prediction

    def _snooker_propose(self, idx, x0_vec):
        """Core snooker geometry (ter Braak & Vrugt, 2008), proposing a jump
        from an arbitrary base point ``x0_vec`` (a parameter vector). Draws the
        anchor + two projection donors from the archive and jumps along the line
        through ``x0_vec`` and the anchor.

        Returns ``(PSet or None, cand_term, zc_vec)`` where

        - ``cand_term = (d-1)*log||Xp - Zc||`` is the destination-to-anchor
          snooker log-Jacobian, i.e. the term the multi-try importance weight
          adds to ``log pi`` (ADR-0067 Stage 2, "Variant A"; see got_result).
        - ``zc_vec`` is the anchor Zc, kept so the accept step can form the
          current-state reference weight ``(d-1)*log||x - Zc_selected||``.

        Draws RNG in exactly the order the classic single-try snooker used, so
        ``calculate_snooker_pset`` (which wraps this with ``x0`` = current state)
        stays byte-identical.
        """
        # Draw three distinct archive indices: c (reference), a, b (for projection difference)
        sel = self.chain_rngs[idx].choice(len(self.archive), 3, replace=False)
        zc_vec = self._param_vec(self.archive[sel[0]])
        za_vec = self._param_vec(self.archive[sel[1]])
        zb_vec = self._param_vec(self.archive[sel[2]])

        # Snooker axis: line through x0 and zc
        axis = x0_vec - zc_vec
        axis_norm_sq = np.dot(axis, axis)
        if axis_norm_sq < 1e-20:
            return None, 0.0, zc_vec

        # Project za and zb onto the snooker axis
        za_proj = zc_vec + axis * (np.dot(za_vec - zc_vec, axis) / axis_norm_sq)
        zb_proj = zc_vec + axis * (np.dot(zb_vec - zc_vec, axis) / axis_norm_sq)

        # Jump vector along the axis
        diff_proj = za_proj - zb_proj

        # Gamma for snooker: U(1.2, 2.2) per Vrugt (2016)
        gamma_s = self.chain_rngs[idx].uniform(1.2, 2.2)

        # Small perturbations
        zeta_vec = self.chain_rngs[idx].normal(0, self.config.config['zeta'], size=self.n_dim)
        lamb = self.chain_rngs[idx].uniform(-self.config.config['lambda'], self.config.config['lambda'])

        xp_vec = x0_vec + zeta_vec + (1.0 + lamb) * gamma_s * diff_proj

        # Build the proposed PSet (reject rather than reflect out-of-bounds)
        proposal = self._proposal_pset(xp_vec)
        if proposal is None:
            return None, 0.0, zc_vec

        # Destination-to-anchor snooker term (d-1)*log||Xp - Zc||.
        dist_xp_zc = np.linalg.norm(xp_vec - zc_vec)
        cand_term = (self.n_dim - 1) * np.log(dist_xp_zc)

        return proposal, cand_term, zc_vec

    def calculate_snooker_pset(self, idx):
        """
        Snooker update proposal (ter Braak & Vrugt, 2008).
        Projects archive points onto the line through the current state and a reference archive point,
        then jumps along that axis.

        Returns (PSet or None, log_correction) where log_correction is the log of the Hastings
        correction factor (d-1)*log(||Xp - Zc|| / ||X - Zc||).
        """
        x0_vec = self._param_vec(self.current_pset[idx])
        proposal, cand_term, zc_vec = self._snooker_propose(idx, x0_vec)
        if proposal is None:
            return None, 0.0

        # Hastings correction: (||Xp - Zc|| / ||X - Zc||)^(d-1).
        # dist_x0_zc = ||x0 - Zc|| is guaranteed nonzero by the axis_norm_sq
        # check inside _snooker_propose, so no divide-by-zero guard is needed.
        dist_x0_zc = np.linalg.norm(x0_vec - zc_vec)
        log_correction = cand_term - (self.n_dim - 1) * np.log(dist_x0_zc)

        return proposal, log_correction

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
            # Cross-chain reset (resolved at the generation barrier) -> root rng.
            donor_idx = self.rng.choice(good_indices)
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

        # Multi-try (ADR-0067 Stage 2) relocates the accept to a per-chain,
        # all-tries-in path. Guarded so a default (n_try == 1) run never enters
        # it and stays byte-identical to the classic single-try engine below.
        if self.n_try > 1:
            return self._got_result_multitry(res)

        pset = res.pset
        score = res.score
        self.total_evaluations += 1

        index = self._chain_index_from_name(pset.name)

        # Calculate posterior of finished job
        lnprior = self.ln_prior(pset)
        lnlikelihood = -score

        lnposterior = lnprior + lnlikelihood

        # Metropolis-Hastings criterion (includes snooker Hastings correction when applicable)
        ln_p_accept = min(0., lnposterior - self.ln_current_P[index]
                          + self.gen_log_snooker_correction[index])
        if np.log(self.chain_rngs[index].uniform()) < ln_p_accept:  # accept update based on MH criterion
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.acceptances[index] += 1
            self.evaluate_constraints(self._result_simdata(res), index)
            self.record_pointwise_loglik(res, index)
            self.record_accepted_output(res, index)

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

        # Wait for the entire generation to finish, then build + return the next
        # generation of proposals (one per chain). Shared with the multi-try
        # barrier via _run_barrier / _advance_generation.
        return self._run_barrier(index, self._build_generation_single)

    def _run_barrier(self, index, build_fn):
        """Generation barrier shared by the single-try and multi-try engines.

        Once every chain has synced (``np.all(wait_for_sync)``), advance the
        generation (diagnostics / outlier reset / CR adaptation / archive growth
        via :meth:`_advance_generation`), then build the next generation of
        proposals with ``build_fn`` (``_build_generation_single`` for the classic
        engine, ``_build_generation_multitry`` for MT-DREAM(ZS)). Returns the
        list of PSets to run next, ``'STOP'``, or ``[]`` while chains are still
        outstanding. The ``while`` handles an all-out-of-bounds generation by
        advancing and retrying rather than returning an empty list (which would
        exhaust the job pool and silently end the run).
        """
        while np.all(self.wait_for_sync):
            if self._advance_generation(index) == 'STOP':
                return 'STOP'

            next_gen = build_fn(index)

            if not next_gen:
                logger.warning('All %d proposals were out of bounds at iteration %d. '
                               'Advancing to next generation.'
                               % (self.num_parallel, min(self.iteration)))
                continue

            # Whitened proposal (ADR-0067): refresh the preconditioning covariance
            # once per synced generation, AFTER this generation's proposals were
            # built from the previous estimate. Byte-identical in timing to the old
            # PDreamAlgorithm.got_result hook, which ran this after super() returned
            # the (non-empty) next_gen list. Dormant for the 'de' proposal.
            if self.proposal == 'whitened' and min(self.iteration) >= self.precondition_adapt:
                self._update_covariance()

            return next_gen

        return []

    def _advance_generation(self, index):
        """Per-generation barrier housekeeping run once all chains are synced:
        reset ``wait_for_sync``, check the stopping conditions (max iterations,
        convergence), print progress, reset outlier chains, adapt CR, grow the
        ZS archive, and snapshot the population mean/std for CR distance. Returns
        ``'STOP'`` to end the run, else ``None``. Shared verbatim by both
        engines, so this is byte-identical to the classic barrier."""
        self.wait_for_sync = [False] * self.num_parallel

        if min(self.iteration) >= self.max_iterations:
            self.update_histograms('_final')
            self.report_constraint_satisfaction('_final')
            return 'STOP'

        if self.iteration[index] % 10 == 0:
            print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
            print2(f'Acceptance rates: {str(self.acceptance_rates)}\n')
        else:
            print2('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))
        # Convergence diagnostics (R-hat, ESS) on their own stride (PERF-1)
        if self.iteration[index] % self.diagnostics_every == 0:
            max_rhat = self.report_convergence_diagnostics(self.iteration[index])
            if self.check_convergence(self.iteration[index], max_rhat):
                return 'STOP'
        logger.debug('Completed %i iterations' % self.iteration[index])
        print2(f'Current -Ln Posteriors: {str(self.ln_current_P)}')

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
                logger.debug(f'Updated CR probabilities: {str(self.cr_probs)}')
        elif self.iteration[index] > self.cr_adapt_end and not self.cr_frozen:
            self.cr_frozen = True
            logger.debug('CR probabilities frozen at iteration %d: %s'
                        % (self.iteration[index], str(self.cr_probs)))

        # Grow the ZS archive: every K generations, append current chain states
        if self.iteration[index] % self.archive_thin_rate == 0:
            for i in range(self.num_parallel):
                self.archive.append(copy.deepcopy(self.current_pset[i]))
                # Output-augmented archive (ADR-0067 Stage 3): append the accepted
                # state's cached output vector in lockstep, so archive_outputs stays
                # index-aligned with archive. Dormant (no-op storage of None slots) for
                # the 'de'/'whitened' proposals that never read outputs.
                if self._archive_stores_outputs:
                    self.archive_outputs.append(self.current_output_vec[i])
            logger.debug('Archive grown to %d entries at iteration %d'
                        % (len(self.archive), self.iteration[index]))

        # Save old states and compute population std for CR adaptation
        for i in range(self.num_parallel):
            self.gen_x_old[i] = self._param_vec(self.current_pset[i])
        all_vecs = np.array(self.gen_x_old)
        self.gen_x_std = np.std(all_vecs, axis=0)
        return None

    def _build_generation_single(self, index):
        """Build the classic single-try next generation: one proposal per chain
        (snooker with probability ``snooker_prob``, else parallel-direction DE),
        named ``iter%irun%i``. Out-of-bounds proposals are treated as Metropolis
        rejections (chain stays in place, history records the non-movement).
        Returns the list of in-bounds proposal PSets. Byte-identical to the
        classic inline barrier body."""
        next_gen = []
        for i, p in enumerate(self.current_pset):
            if self.chain_rngs[i].uniform() < self.snooker_prob:
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
                logger.debug('Proposed PSet for chain %d is out of bounds. Treating as rejection.' % i)
                self._record_generation_rejection(i)
        return next_gen

    def _record_generation_rejection(self, i):
        """Record a whole-generation Metropolis rejection for chain ``i`` (its
        proposal, or in multi-try all of its trials, fell out of bounds): the
        chain stays put, history records the non-movement so R-hat/ESS reflect
        it, and the chain is marked complete. Shared by both engines."""
        self.chain_history[i].append(self._param_vec(self.current_pset[i]))
        self.ln_posterior_history[i].append(self.ln_current_P[i])
        self.wait_for_sync[i] = True
        self.iteration[i] += 1
        self.acceptance_rates[i] = self.acceptances[i] / self.iteration[i]
        if self.iteration[i] % self.sample_every == 0 and self.iteration[i] > self.burn_in:
            self.sample_pset(self.current_pset[i], self.ln_current_P[i], i)

    # ------------------------------------------------------------------ #
    # Multi-Try DREAM (MT-DREAM(ZS); Laloy & Vrugt 2012) — ADR-0067       #
    # Stage 2. Active only when n_try > 1. A generation for each chain    #
    # runs as a two-phase per-chain state machine:                       #
    #   TRIALS:    k candidates y_1..y_k drawn from the current state x   #
    #              (all snooker OR all parallel-direction). Each is       #
    #              evaluated; the winner Y is selected in proportion to   #
    #              its importance weight w(y) = pi(y) * g(y), where the   #
    #              snooker Jacobian g(y) = ||y - z_y||^(d-1) is 1 for the #
    #              symmetric proposals.                                   #
    #   REFERENCE: k-1 reference points drawn from T(Y, .); the k-th      #
    #              reference is the current state x itself. Y is accepted #
    #              over x with the multiple-try Metropolis ratio          #
    #              min(1, sum_j w(y_j) / sum_j w(x*_j)).                   #
    # The current-state reference weight carries the snooker Jacobian     #
    # ||x - z_Y||^(d-1) at the SELECTED candidate's anchor z_Y ("Variant  #
    # A"): the unique choice that reduces to the ter Braak & Vrugt (2008) #
    # single-try snooker ratio at k=1 (derived + verified for ADR-0067;   #
    # both the DREAM-Suite and PyDREAM reference codes depart from it).   #
    # 2k-1 evaluations per chain per generation.                         #
    # ------------------------------------------------------------------ #
    def _got_result_multitry(self, res):
        """MT-DREAM(ZS) result handler (n_try > 1). Routes each completed job by
        its name: the initial ``iter0run%i`` population (bootstrap, single forced
        accept), a ``try%i`` candidate, or a ``ref%i`` reference point."""
        self.total_evaluations += 1
        name = res.pset.name
        index = self._chain_index_from_name(name)
        m = re.search(r'(try|ref)(\d+)$', name)
        if m is None:
            # Initial generation: no MTM yet (ln_current_P is nan -> forced accept).
            return self._mt_bootstrap(res, index)
        phase, sub = m.group(1), int(m.group(2))
        if phase == 'try':
            return self._mt_handle_trial(res, index, sub)
        return self._mt_handle_reference(res, index, sub)

    def _mt_bootstrap(self, res, index):
        """Accept an initial-population evaluation (iter0). ``ln_current_P``
        starts NaN, forcing accept exactly as the single-try engine does; then
        hand off to the multi-try barrier to build the first k-candidate
        generation."""
        pset = res.pset
        lnposterior = self.ln_prior(pset) - res.score
        ln_p_accept = min(0., lnposterior - self.ln_current_P[index])
        if np.log(self.chain_rngs[index].uniform()) < ln_p_accept:
            self.current_pset[index] = pset
            self.ln_current_P[index] = lnposterior
            self.acceptances[index] += 1
            self.evaluate_constraints(self._result_simdata(res), index)
            self.record_pointwise_loglik(res, index)
            self.record_accepted_output(res, index)
        self.chain_history[index].append(self._param_vec(self.current_pset[index]))
        self.ln_posterior_history[index].append(self.ln_current_P[index])
        self.wait_for_sync[index] = True
        self.iteration[index] += 1
        self.acceptance_rates[index] = self.acceptances[index] / self.iteration[index]
        if self.iteration[index] % self.sample_every == 0 and self.iteration[index] > self.burn_in:
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        return self._run_barrier(index, self._build_generation_multitry)

    def _mt_handle_trial(self, res, index, sub):
        """Buffer a completed candidate (trial) for chain ``index``. When all of
        the chain's in-bounds trials are in, select the preferred candidate and
        emit the reference set."""
        meta = self.mt_trial_meta[index][sub]
        lnposterior = self.ln_prior(res.pset) - res.score
        self.mt_trials[index].append({
            'pset': res.pset,
            'lnpost': lnposterior,
            'snooker_term': meta['cand_term'],
            'anchor': meta['anchor'],
            'cr_idx': meta['cr_idx'],
            'res': res,
        })
        if len(self.mt_trials[index]) < self.mt_trials_expected[index]:
            return []
        return self._mt_select_and_emit_references(index)

    def _mt_select_and_emit_references(self, index):
        """All trials for chain ``index`` are in: select the winner Y with
        probability proportional to its importance weight, then draw the k-1
        reference points from ``T(Y, .)`` and emit them. The current-state slot
        weight (Variant A) is computed and stored for the accept step."""
        trials = self.mt_trials[index]
        logws = np.array([t['lnpost'] + t['snooker_term'] for t in trials])
        if not np.any(np.isfinite(logws)):
            # Every in-bounds candidate has zero posterior mass -> reject generation.
            self._record_generation_rejection(index)
            return self._run_barrier(index, self._build_generation_multitry)
        probs = np.exp(logws - logsumexp(logws))
        probs = probs / probs.sum()
        j_sel = int(self.chain_rngs[index].choice(len(trials), p=probs))
        sel = trials[j_sel]
        self.mt_selected[index] = sel
        y_vec = self._param_vec(sel['pset'])

        ref_psets = []
        ref_terms = []
        for _ in range(self.n_try - 1):
            if self.mt_is_snooker[index]:
                rpset, cand_term, _anchor = self._snooker_propose(index, y_vec)
            else:
                rpset, _cr = self.calculate_new_pset(index, base=sel['pset'])
                cand_term = 0.0
            if rpset is None:
                continue
            rpset.name = 'iter%irun%iref%i' % (self.iteration[index], index, len(ref_psets))
            ref_terms.append(cand_term)
            ref_psets.append(rpset)
        self.mt_ref_terms[index] = ref_terms
        self.mt_refs_expected[index] = len(ref_psets)
        self.mt_ref_logw[index] = []

        # k-th reference = the current state x. The Variant-A snooker term uses
        # x's distance to the SELECTED candidate's anchor z_Y (the anchor that
        # generated Y from x); zero for the symmetric proposals.
        if self.mt_is_snooker[index]:
            x_vec = self._param_vec(self.current_pset[index])
            cur_term = (self.n_dim - 1) * np.log(np.linalg.norm(x_vec - sel['anchor']) + 1e-300)
        else:
            cur_term = 0.0
        self.mt_cur_logw[index] = self.ln_current_P[index] + cur_term

        if not ref_psets:
            # All k-1 reference draws out of bounds: accept using only the
            # current-state slot as the reference weight.
            return self._mt_accept(index)
        return ref_psets

    def _mt_handle_reference(self, res, index, sub):
        """Buffer a completed reference point. When all in-bounds references are
        in, run the multiple-try accept."""
        lnposterior = self.ln_prior(res.pset) - res.score
        self.mt_ref_logw[index].append(lnposterior + self.mt_ref_terms[index][sub])
        if len(self.mt_ref_logw[index]) < self.mt_refs_expected[index]:
            return []
        return self._mt_accept(index)

    def _mt_accept(self, index):
        """Multiple-try Metropolis accept (Liu, Liang & Wong 2000; Laloy & Vrugt
        2012): accept the selected candidate Y over the current state x with
        probability min(1, sum_j w(y_j) / sum_j w(x*_j)) -- the k trial weights
        over the k reference weights (k-1 fresh draws from Y plus the current-
        state slot). Record the single accepted move, adapt CR from the winner's
        crossover value, then hand off to the barrier."""
        sel = self.mt_selected[index]
        trial_logws = np.array([t['lnpost'] + t['snooker_term'] for t in self.mt_trials[index]])
        ref_logws = np.array(self.mt_ref_logw[index] + [self.mt_cur_logw[index]])
        log_alpha = min(0.0, logsumexp(trial_logws) - logsumexp(ref_logws))
        if np.log(self.chain_rngs[index].uniform()) < log_alpha:
            self.current_pset[index] = sel['pset']
            self.ln_current_P[index] = sel['lnpost']
            self.acceptances[index] += 1
            self.evaluate_constraints(self._result_simdata(sel['res']), index)
            self.record_pointwise_loglik(sel['res'], index)
            self.record_accepted_output(sel['res'], index)

        # Store chain history (after accept/reject, so it reflects the kept state)
        self.chain_history[index].append(self._param_vec(self.current_pset[index]))
        self.ln_posterior_history[index].append(self.ln_current_P[index])

        # CR adaptation from the selected candidate's crossover value + realized move
        if not self.cr_frozen and sel['cr_idx'] is not None and self.gen_x_old[index] is not None:
            diff_vec = self._param_vec(self.current_pset[index]) - self.gen_x_old[index]
            sd_dist = np.sum((diff_vec / np.maximum(self.gen_x_std, 1e-10)) ** 2)
            self.cr_total_distance[sel['cr_idx']] += sd_dist
            self.cr_usage_count[sel['cr_idx']] += 1

        self.wait_for_sync[index] = True
        self.iteration[index] += 1
        self.acceptance_rates[index] = self.acceptances[index] / self.iteration[index]
        if self.iteration[index] % self.sample_every == 0 and self.iteration[index] > self.burn_in:
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        if (self.iteration[index] % (self.sample_every * self.output_hist_every) == 0
            and self.iteration[index] > self.burn_in):
            self.update_histograms('_%i' % self.iteration[index])

        return self._run_barrier(index, self._build_generation_multitry)

    def _build_generation_multitry(self, index):
        """Build the MT-DREAM(ZS) next generation: k candidate proposals per
        chain drawn from the current state (snooker for the whole chain-
        generation with probability ``snooker_prob``, else parallel-direction),
        named ``iter%irun%itry%i``. Resets the per-chain multi-try buffers. A
        chain whose k trials are all out of bounds is a generation rejection.
        Returns the flat list of all chains' trial PSets."""
        next_gen = []
        for i in range(self.num_parallel):
            is_snooker = self.chain_rngs[i].uniform() < self.snooker_prob
            self.mt_is_snooker[i] = is_snooker
            self.mt_trials[i] = []
            self.mt_trial_meta[i] = []
            self.mt_ref_logw[i] = []
            self.mt_selected[i] = None
            x_vec = self._param_vec(self.current_pset[i])
            trial_psets = []
            for _ in range(self.n_try):
                if is_snooker:
                    cpset, cand_term, anchor = self._snooker_propose(i, x_vec)
                    cr_idx = None
                else:
                    cpset, cr_idx = self.calculate_new_pset(i)
                    cand_term, anchor = 0.0, None
                if cpset is None:
                    continue
                cpset.name = 'iter%irun%itry%i' % (self.iteration[i], i, len(trial_psets))
                self.mt_trial_meta[i].append({'cand_term': cand_term, 'anchor': anchor, 'cr_idx': cr_idx})
                trial_psets.append(cpset)
            self.mt_trials_expected[i] = len(trial_psets)
            if not trial_psets:
                self._record_generation_rejection(i)
            else:
                next_gen.extend(trial_psets)
        return next_gen

    def calculate_new_pset(self, idx, base=None):
        """
        Uses differential evolution-like update to calculate new PSet.
        Returns (PSet, cr_idx) or (None, cr_idx) if the proposal is out of bounds.

        Dispatches on the configured proposal operator (ADR-0067 axis 1): the
        'whitened' proposal runs once its preconditioner has warmed up, otherwise
        (and always for 'de') the classic parallel-direction DE proposal below.

        ``base`` overrides the point the jump is applied to (default: the current
        chain state). Multi-try (ADR-0067 Stage 2) passes the selected candidate
        Y as ``base`` to draw the reference set from ``T(Y, .)``; with the default
        ``base=None`` this is byte-identical to the classic single-try proposal.

        :param idx: Index of PSet to update
        :return: tuple of (PSet or None, int)
        """
        if self.proposal == 'whitened' and self._preconditioned:
            return self._calculate_whitened_pset(idx, base)

        x0 = self.current_pset[idx] if base is None else base

        # Draw 2*delta donor states from the ZS archive (without replacement)
        sel = self.chain_rngs[idx].choice(len(self.archive), 2 * self.delta, replace=False)

        # Sample crossover value and mask
        cr_idx = self.chain_rngs[idx].choice(self.ncr_count, p=self.cr_probs)
        cr = self.ncr[cr_idx]
        while True:
            ds = self.chain_rngs[idx].uniform(size=self.n_dim) <= cr  # sample parameter subspace
            if np.any(ds):
                break

        # Gamma selection: mode jump (gamma=1) or adaptive/fixed step size
        if self.chain_rngs[idx].uniform() < self.g_prob:
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
            zeta = self.chain_rngs[idx].normal(0, self.config.config['zeta'])
            lamb = self.chain_rngs[idx].uniform(-self.config.config['lambda'], self.config.config['lambda'])

            # Differential evolution calculation (while satisfying detailed balance)
            try:
                # Do not reflect the parameter (need to reject if outside bounds)
                new_var = x0.get_param(k.name).add(zeta + (1. + lamb) * gamma * total_diff, False)
                new_vars.append(new_var)
            except OutOfBoundsException:
                logger.debug("Variable %s is outside of bounds")
                return None, cr_idx

        return PSet(new_vars), cr_idx

    # ------------------------------------------------------------------ #
    # Whitened ('whitened') proposal — folded in from PDreamAlgorithm    #
    # (ADR-0067). Active only when self.proposal == 'whitened'; the      #
    # state and hooks are dormant for the default 'de' proposal.         #
    # ------------------------------------------------------------------ #
    def _update_covariance(self):
        """
        Estimate the covariance from pooled chain history and compute Cholesky factors.
        Discards the first 50% of each chain as warmup (matching the convention used by
        diagnostics.split_chains for R-hat/ESS) so the early burn-in transient does not
        inflate the preconditioner. Pooling across chains is intentional: the whitened
        proposal's global preconditioner wants a proposal scale large enough for
        archive-based mode hopping.
        """
        # Pool the post-warmup half of each chain history into one matrix
        all_samples = []
        for chain in self.chain_history:
            if len(chain) > 1:
                all_samples.extend(chain[len(chain) // 2:])
        if len(all_samples) < 2 * self.n_dim:
            return  # Not enough samples yet

        X = np.array(all_samples)
        n = X.shape[0]
        d = X.shape[1]

        # Sample covariance with Haario-style regularization: C = Cov(X) + eps*I
        cov = np.cov(X, rowvar=False)
        eps = 1e-6 * np.trace(cov) / d if np.trace(cov) > 0 else 1e-6
        cov += eps * np.eye(d)

        try:
            L = np.linalg.cholesky(cov)
            self._cov_L = L
            self._cov_L_inv = np.linalg.solve(L, np.eye(d))
            if not self._preconditioned:
                self._preconditioned = True
                logger.info('P-DREAM: preconditioning activated at iteration %d '
                            'with %d pooled samples (d=%d)'
                            % (min(self.iteration), n, d))
            else:
                logger.debug('P-DREAM: covariance updated with %d samples' % n)
        except np.linalg.LinAlgError:
            logger.warning('P-DREAM: Cholesky decomposition failed, '
                           'skipping covariance update')

    def _whiten(self, x_vec):
        """Transform a parameter vector to whitened space: z = L_inv @ x."""
        return self._cov_L_inv @ x_vec

    def _unwhiten_diff(self, dz_vec):
        """Transform a difference vector from whitened space back: dx = L @ dz."""
        return self._cov_L @ dz_vec

    def _calculate_whitened_pset(self, idx, base=None):
        """
        DE proposal in whitened space (the 'whitened' proposal operator).

        1. Transform current state and archive donors to z = L_inv @ x
        2. Compute DE difference in z-space
        3. Apply crossover in z-space (dimensions are decorrelated)
        4. Scale and add perturbation in z-space
        5. Convert the total jump back: dx = L @ dz_total
        6. Propose x_new = x_current + dx

        Only reached from :meth:`calculate_new_pset` once ``self._preconditioned``
        is True; before that the classic DE proposal is used. ``base`` overrides
        the jump origin (multi-try reference draws pass the selected candidate);
        ``base=None`` is byte-identical to the single-try whitened proposal.
        """
        x0 = self.current_pset[idx] if base is None else base
        x0_vec = self._param_vec(x0)

        # Draw 2*delta donor states from the ZS archive (without replacement)
        sel = self.chain_rngs[idx].choice(len(self.archive), 2 * self.delta, replace=False)

        # Whiten the donor states
        z_donors = []
        for s in sel:
            z_donors.append(self._whiten(self._param_vec(self.archive[s])))

        # Sample crossover value and mask (in whitened space where dims are independent)
        cr_idx = self.chain_rngs[idx].choice(self.ncr_count, p=self.cr_probs)
        cr = self.ncr[cr_idx]
        while True:
            ds = self.chain_rngs[idx].uniform(size=self.n_dim) <= cr
            if np.any(ds):
                break

        # Gamma selection
        if self.chain_rngs[idx].uniform() < self.g_prob:
            gamma = 1
            ds[:] = True  # mode jump: update all dimensions
        else:
            d_prime = int(np.sum(ds))
            if self.adaptive_step_size:
                gamma = 2.38 / np.sqrt(2.0 * self.delta * d_prime)
            else:
                gamma = self.step_size

        # Compute DE difference in whitened space
        dz_total = np.zeros(self.n_dim)
        for j in range(self.delta):
            dz_total += z_donors[j] - z_donors[self.delta + j]

        # Apply crossover mask in whitened space
        dz_masked = np.where(ds, dz_total, 0.0)

        # Small perturbations in whitened space
        zeta_z = self.chain_rngs[idx].normal(0, self.config.config['zeta'], size=self.n_dim)
        lamb = self.chain_rngs[idx].uniform(-self.config.config['lambda'], self.config.config['lambda'])

        # Total jump in whitened space, then transform back to original space
        dz_jump = zeta_z + (1.0 + lamb) * gamma * dz_masked
        dx_jump = self._unwhiten_diff(dz_jump)

        # Build proposed PSet in original space (reject rather than reflect)
        xp_vec = x0_vec + dx_jump
        proposal = self._proposal_pset(xp_vec)
        if proposal is None:
            logger.debug("Proposed parameter outside of bounds")
            return None, cr_idx

        return proposal, cr_idx
