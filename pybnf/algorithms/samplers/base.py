"""The BayesianAlgorithm base class — shared by all Bayesian sampler fit types.

Extracted from the algorithms package (M1 Step 3). Holds posterior/prior
bookkeeping and the rank-normalized split-R-hat / ESS convergence diagnostics
(Vehtari et al. 2021). The run loop and execution seam live in the Algorithm
base (``..base``); BayesianAlgorithm inherits them and adds no core.* call of
its own, so this module does not import the ``core`` seam.
"""


from ..base import Algorithm
from ...config_schema import PyBNFConfigModel
from ...printing import print1, print2, PybnfError
from ... import diagnostics

import logging
import numpy as np
from pathlib import Path
from pydantic import Field


# Preserve the original module logger name (was getLogger(__name__) in
# algorithms.py) so log records keep the 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class MCMCFamilyConfig(PyBNFConfigModel):
    """Config shared by the whole Bayesian/MCMC family (mh, pt, am, dream,
    p_dream), co-located with the family base (ADR-0002, ADR-0006). (sa was once
    in this family; M2.2/ADR-0008 made it a standalone optimizer.)

    Holds the defaulted keys the shared ``BayesianAlgorithm`` base reads (so they
    are common to every MCMC method) plus the β-ladder ``postprocess`` hook. The
    per-method keys live on the leaf models that subclass this and co-locate with
    their algorithm class: :class:`BasicMCMCConfig` (mh/pt/sa, in basic_mcmc.py),
    :class:`AdaptiveMCMCConfig` (am, in adaptive_mcmc.py),
    :class:`DreamConfig` (dream, in dream.py) and
    :class:`PDreamConfig` (p_dream, in pdream.py). ``neg_bin_r`` stays in
    ``GlobalConfig`` -- an objfunc param read regardless of fit_type. Values are
    byte-identical to the old global defaults; under narrowing (ADR-0013) these
    keys appear only in each MCMC fit_type's own effective config. The per-fit_type
    validity of individual keys is reported by ``check_unused_keys`` in
    ``config.py``.
    """

    step_size: float = 0.2
    burn_in: int = 10000
    sample_every: int = 100
    output_hist_every: int = 100
    hist_bins: int = 10
    adaptive: int = 10000
    credible_intervals: list = Field(default_factory=lambda: [68.0, 95.0])
    beta: list = Field(default_factory=lambda: [1.0])
    continue_run: int = 0
    rhat_threshold: float = 0.0       # R-hat / ESS diagnostics live on the base
    diagnostics_every: int = 0

    # beta_range is a user input the beta-ladder postprocess consumes (geomspaces it
    # into beta_list) rather than a stored field; valid for every MCMC fit (#401).
    RUNTIME_KEYS = frozenset({'beta_range'})

    @classmethod
    def postprocess(cls, conf_dict, fit_type):
        """The β-ladder (ported from ``Configuration.postprocess_mcmc_keys``).

        Algorithms 'mh', 'pt', 'am', 'dream', 'p_dream' have similar but
        non-identical valid config keys; this builds ``beta_list`` / ``reps_per_beta``
        and reconciles ``exchange_every`` / ``population_size``. Mutates ``conf_dict``
        in place (operating on the RAW config dict, before defaults merge, so
        raw-presence checks like ``'beta' not in conf_dict`` mean "user did not set
        it"). ``config._build_config`` dispatches this uniformly; non-MCMC models
        inherit the no-op ``postprocess`` from ``PyBNFConfigModel`` (ADR-0006 #3).

        Only the config *transformations* live here now. The per-method unused-key
        *warnings* this hook used to also emit (``exchange_every``/``reps_per_beta``
        on non-pt, the sa-only ``cooling``/``beta_max``, the DREAM-only
        ``crossover_number``/``zeta``/``lambda``/``gamma_prob`` on mh/pt/am) moved to
        the unified, schema-derived ``Configuration.check_unused_keys`` (#401,
        ADR-0014), which now warns about every foreign key each MCMC fit does not own
        -- more precisely than this hook's hand-listed branches did.
        """
        # exchange_every/reps_per_beta are pt-only; pin them for the non-pt methods
        # (the warning for a user who set them on a non-pt fit comes from
        # check_unused_keys now).
        if conf_dict['fit_type'] != 'pt':
            conf_dict['exchange_every'] = np.inf
            conf_dict['reps_per_beta'] = 1
        elif 'reps_per_beta' not in conf_dict:
            conf_dict['reps_per_beta'] = 1  # Default value if using pt but didn't specify

        # Create the starting list of betas based on the various available options. Warn if tried to do something weird
        if 'beta' not in conf_dict and 'beta_range' not in conf_dict:
            conf_dict['beta'] = [1.]

        # Handle the Parallel Tempering case where reps_per_beta is specified.
        # First, check it's divisible by the population size
        if conf_dict['population_size'] % conf_dict['reps_per_beta'] != 0:
            conf_dict['population_size'] -= conf_dict['population_size'] % conf_dict['reps_per_beta']
            print1('Warning: Lowered your population_size to %i so that it is divisible by your setting for '
                   'reps_per_beta' % conf_dict['population_size'])
        # Then, we want the beta_list generated below to contain only one copy of the spread of betas to use
        # At the end, we make reps_per_beta copies of that list to arrive at the final beta list.
        subpop_size = conf_dict['population_size'] // conf_dict['reps_per_beta']

        if 'beta_range' in conf_dict:
            if len(conf_dict['beta_range']) != 2:
                raise PybnfError("Wrong number of entries in beta_range",
                                 "Config key 'beta_range' must have exactly 2 numbers: the min and the max.")
            if 'beta' in conf_dict:
                print1("Warning: Ignoring config key 'beta' because it is overridden by config key 'beta_range'")
            if conf_dict['fit_type'] != 'pt':
                print1("Warning: You used 'beta_range' with the method {}. This is an odd thing to do. Usually, you "
                       "would want all your replicates starting at the same beta value.".format(conf_dict['fit_type']))
            betalist = list(np.geomspace(conf_dict['beta_range'][0], conf_dict['beta_range'][1], subpop_size))
        elif len(conf_dict['beta']) > 1:
            betalist = conf_dict['beta']
            if conf_dict['fit_type'] != 'pt':
                print1("Warning: You specified multiple beta values with the method {}. This is an odd thing to do. "
                       "Usually, you would specify one beta value to use with all your replicates. ".format(conf_dict['fit_type']))
            if len(betalist) != subpop_size:
                print1("Warning: You specified %i beta values, so I will run %i replicates instead of using your "
                       "population_size setting" % (len(betalist), len(betalist)*conf_dict['reps_per_beta']))
                conf_dict['population_size'] = len(betalist)*conf_dict['reps_per_beta']
        else:
            betalist = conf_dict['beta'] * subpop_size  # n copies of the single beta value
            if conf_dict['fit_type'] == 'pt':
                print1("Warning: You specified a single beta value with the method pt. This makes the algorithm's "
                       "replica exchanges accomplish nothing. To make good use of this algorithm, set the key "
                       "'beta_range' or specify multiple values with the 'beta' key.")
        betalist.sort()
        betalist = betalist * conf_dict['reps_per_beta']
        conf_dict['beta_list'] = betalist

        if conf_dict['fit_type'] == 'pt' and betalist[-1] != 1:
            print1('Warning: You are about to calculate a distribution with beta=%i instead of 1. That means your '
                   'calculated distribution will be %s than the true probability distribution' %
                   (betalist[-1], 'narrower' if betalist[-1] > 1 else 'broader'))
        return conf_dict


class BayesianAlgorithm(Algorithm):
    """Superclass for Bayesian MCMC algorithms"""

    def __init__(self, config):
        super().__init__(config)
        self.num_parallel = config.config['population_size']
        self.max_iterations = config.config['max_iterations']
        self.step_size = config.config['step_size']
        self.n_dim = len(self.variables)

        # One independent Generator per parallel chain, indexed by chain index, so
        # each chain's proposal/accept draws come from its own stream regardless of
        # dask completion order (cross-chain barriers still use the root self.rng).
        self._rebuild_chain_rngs()

        self.iteration = [0] * self.num_parallel  # Iteration number that each PSet is on

        self.current_pset = None  # List of n PSets corresponding to the n independent runs
        self.ln_current_P = None  # List of n probabilities of those n PSets.

        self.burn_in = config.config['burn_in']  # todo: 'auto' option
        self.adaptive = config.config['adaptive']
        self.sample_every = config.config['sample_every']
        self.output_hist_every = config.config['output_hist_every']
        # A list of the % credible intervals to save, eg [68. 95]
        self.credible_intervals = config.config['credible_intervals']
        self.num_bins = config.config['hist_bins']

        self.wait_for_sync = [False] * self.num_parallel

        self.prior = None
        self.load_priors()

        self.samples_file = str(Path(self.config.config['output_dir']) / 'Results' / 'samples.txt')

        # Chain history for convergence diagnostics (R-hat, ESS)
        self.chain_history = [[] for _ in range(self.num_parallel)]
        self.ln_posterior_history = [[] for _ in range(self.num_parallel)]

        # Convergence threshold (0 = disabled)
        self.rhat_threshold = config.config['rhat_threshold']

        # How often (in iterations) to compute the R-hat/ESS convergence
        # diagnostics. Each computation rank-normalizes/autocorrelates the last
        # 50% of the chain history, whose length grows with the run, so a fixed
        # cadence makes total diagnostic cost ~O(max_iterations^2). A stride that
        # grows with the run instead caps the number of computations (~100),
        # keeping the cost ~O(max_iterations). The *value* reported at any given
        # iteration is unchanged — only how often it is computed. 0 = auto.
        self.diagnostics_every = config.config['diagnostics_every']
        if self.diagnostics_every <= 0:
            self.diagnostics_every = max(10, self.max_iterations // 100)

        # Total model evaluations for ESS/evaluation metric
        self.total_evaluations = 0

        # Constraint satisfaction tracking
        self.all_constraints = []
        for cset in self.config.constraints:
            self.all_constraints.extend(cset.constraints)
        self.current_constraint_satisfied = [None] * self.num_parallel
        self.constraint_samples_file = str(Path(self.config.config['output_dir']) / 'Results' / 'constraint_samples.txt')

        # Per-observation log-likelihood sidecar for LOO/WAIC (ADR-0056, #438 item 4).
        # Recorded only when output_inference_data is set AND the objective is a per-point
        # likelihood (a least-squares / distance / pass-through objective has no normalized
        # density to leave one out of -- warn once and stay off so the rest of the run is
        # untouched). Like constraint satisfaction, the accepted pset's pointwise vector is
        # cached per chain at accept time (where its simdata is in hand) and written for the
        # current pset at each sample iteration, so log_likelihood.txt stays row-aligned with
        # samples.txt (the bridge reads the two in lockstep).
        self.log_likelihood_file = str(Path(self.config.config['output_dir']) / 'Results' / 'log_likelihood.txt')
        self.current_pointwise_loglik = [None] * self.num_parallel
        self._loglik_ids = None
        self._loglik_header_written = False
        self._record_loglik = bool(self.config.config.get('output_inference_data')) \
            and getattr(self.objective, 'supports_pointwise_log_likelihood', False)
        if self.config.config.get('output_inference_data') and not self._record_loglik:
            print1("Note: output_inference_data is set, but objfunc '%s' is not a per-point "
                   "likelihood, so no pointwise log-likelihoods are recorded -- the "
                   "InferenceData will omit the log_likelihood group and az.loo/az.waic will "
                   "be unavailable. Use a likelihood objfunc (chi_sq, chi_sq_dynamic, "
                   "lognormal, laplace, neg_bin, neg_bin_dynamic) for LOO/WAIC."
                   % self.config.config.get('objfunc'))

        # Check that the iteration range is valid with respect to the burnin and or adaptive iterations
        

    def _rebuild_chain_rngs(self):
        """Spawn one independent Generator per parallel chain (overrides the base
        no-op). Called at construction and after each bootstrap-replicate reseed so
        every replicate's chains are reproducible from the run seed yet distinct."""
        self.chain_rngs = self.spawn_chain_rngs(self.num_parallel)

    def load_priors(self):
        """Builds the data structures for the priors, based on the variables specified in the config."""
        self.prior = dict()  # Maps each variable name to the FreeParameter containing its scipy.stats distribution.
        for var in self.variables:
            if var.has_prior:
                self.prior[var.name] = var

    def start_run(self, setup_samples=True):

        if self.config.config['initialization'] == 'lh':
            first_psets = self.random_latin_hypercube_psets(self.num_parallel)
        else:
            first_psets = [self.random_pset() for i in range(self.num_parallel)]

        # ADR-0043 Phase 2: seed exactly one chain's initial pset at the initial_value
        # point (a no-op unless a parameter: record declares one); the other chains stay
        # random. Placed before continue_run / starting_params below so those still take
        # precedence when set -- they override every chain's start uniformly.
        first_psets[0] = self._seed_initial_value_pset(first_psets[0])

        self.ln_current_P = [np.nan]*self.num_parallel  # Forces accept on the first run
        self.current_pset = [None]*self.num_parallel

        if self.config.config['continue_run'] == 1:
            self.mle_start = np.loadtxt(Path(self.config.config['output_dir']) / 'adaptive_files' / 'MLE_params.txt')
            for n in range(self.num_parallel):
                for i,p in enumerate(first_psets[n]):
                    p.value = self.mle_start[i]
        if self.config.config['starting_params'] and self.config.config['continue_run'] != 1:
            for n in range(self.num_parallel):
                for i,p in enumerate(first_psets[n]):
                    p.value = self.config.config['starting_params'][i]           
        for i in range(len(first_psets)):
            first_psets[i].name = 'iter0run%i' % i

        # Set up the output files
        # Cant do this in the constructor because that happens before the output folder is potentially overwritten.
        if setup_samples:
            with open(self.samples_file, 'w') as f:
                f.write('# Name\tLn_probability\t'+first_psets[0].keys_to_string()+'\n')
            if self._record_loglik:
                # Truncate any stale sidecar in lockstep with samples.txt above (a fresh
                # run starts both empty; a continue_run leaves both untouched, so the two
                # append in step and stay row-aligned). The id header is written lazily on
                # the first sample, once an accepted draw reveals the observation labels.
                open(self.log_likelihood_file, 'w').close()
                self._loglik_header_written = False
            if self.all_constraints:
                with open(self.constraint_samples_file, 'w') as f:
                    header = '\t'.join(c.source_line or 'constraint_%i' % i
                                       for i, c in enumerate(self.all_constraints))
                    f.write('# ' + header + '\n')
            (Path(self.config.config['output_dir']) / 'Results' / 'Histograms').mkdir(parents=True, exist_ok=True)



        return first_psets

    def got_result(self, res):
        NotImplementedError("got_result() must be implemented in BayesianAlgorithm subclass")

    def ln_prior(self, pset):
        """
        Returns the value of the prior distribution for the given parameter set

        :param pset:
        :type pset: PSet
        :return: float value of ln times the prior distribution
        """
        total = 0.
        for v, prior_var in self.prior.items():
            contribution = prior_var.prior_logpdf(pset[v])
            if not np.isfinite(contribution) and prior_var.has_bounded_support:
                logger.warning(f'Box-constrained parameter {v} reached a value outside the box.')
            total += contribution
        return total

    @staticmethod
    def _result_simdata(res):
        """The full multi-model simulation-data dict for a completed Result, wherever
        the objective was scored.

        The default worker-scoring path (``local_objective_eval=0`` with
        ``parallelize_models=1``) scores on the worker and then nulls
        ``res.simdata`` to save memory, moving the data to ``res.out``
        (:meth:`core.Job.run_simulation`). Master-side scoring
        (``parallelize_models>1``) leaves the data on ``res.simdata`` and never
        sets ``res.out``. The per-sample constraint-satisfaction and
        pointwise-loglik bookkeeping needs the whole dict either way, because
        cross-model ``.prop`` references resolve suffixes across models -- so
        reading only ``res.simdata`` handed constraint evaluation ``None`` on the
        default path and crashed (lanl/PyBNF#480)."""
        simdata = getattr(res, 'simdata', None)
        if simdata is not None:
            return simdata
        return getattr(res, 'out', None)

    def evaluate_constraints(self, simdata, chain_index):
        """
        Evaluate all constraints against simulation data and cache the pass/fail results for this chain.

        :param simdata: Simulation data dictionary
        :param chain_index: Index of the chain that was accepted
        """
        if not self.all_constraints:
            return
        if simdata is None:
            # Defensive: a genuinely missing simdata would make cross-model
            # (dotted-suffix) constraint resolution iterate ``None`` and crash
            # (lanl/PyBNF#480). Callers pass the resolved simdata via
            # _result_simdata, so this should not happen; if it ever does, skip
            # this sample's satisfaction bookkeeping rather than abort the run --
            # the prior cached vector (if any) stays in place.
            logger.debug('No simulation data available to evaluate constraints for chain %d; '
                         'skipping constraint-satisfaction bookkeeping for this sample.', chain_index)
            return
        satisfied = []
        for c in self.all_constraints:
            satisfied.append(1 if c.penalty(simdata) == 0 else 0)
        self.current_constraint_satisfied[chain_index] = satisfied

    def record_pointwise_loglik(self, res, chain_index):
        """Cache the just-accepted pset's per-observation log-likelihoods for this chain
        (ADR-0056), mirroring :meth:`evaluate_constraints`: computed here, where the
        accepted pset's simulation data is in hand, and written later by
        :meth:`sample_pset` for whichever pset is current at a sample iteration. A no-op
        unless ``output_inference_data`` is set and the objective is a likelihood; any
        failure is logged, never fatal (the run must not die for a diagnostics sidecar).

        Reads the simdata through :meth:`_result_simdata` so it works whether the
        objective was scored worker-side (data on ``res.out``) or master-side (data on
        ``res.simdata``); see lanl/PyBNF#480."""
        if not self._record_loglik:
            return
        try:
            result = self.objective.evaluate_pointwise(self._result_simdata(res), self.exp_data, res.pset)
        except Exception:
            logger.debug('Could not compute pointwise log-likelihood for chain %d',
                         chain_index, exc_info=True)
            return
        if result is None:
            return
        ids, values = result
        if self._loglik_ids is None:
            self._loglik_ids = ids
        self.current_pointwise_loglik[chain_index] = values

    def sample_pset(self, pset, ln_prob, chain_index=None):
        """
        Adds this pset to the set of sampled psets for the final distribution.
        :param pset:
        :type pset: PSet
        :param ln_prob - The probability of this PSet to record in the samples file.
        :type ln_prob: float
        :param chain_index: Index of the chain, used to look up cached constraint results.
        :type chain_index: int or None
        """
        with open(self.samples_file, 'a') as f:
            f.write(pset.name+'\t'+str(ln_prob)+'\t'+pset.values_to_string()+'\n')
        if self.all_constraints and chain_index is not None and self.current_constraint_satisfied[chain_index] is not None:
            with open(self.constraint_samples_file, 'a') as f:
                f.write('\t'.join(str(x) for x in self.current_constraint_satisfied[chain_index]) + '\n')
        if self._record_loglik:
            self._write_pointwise_loglik(chain_index)

    def _write_pointwise_loglik(self, chain_index):
        """Append the current chain's cached per-observation log-likelihood vector to
        log_likelihood.txt, one row per sample so it stays row-aligned with the
        samples.txt row :meth:`sample_pset` just wrote -- the bridge reads the two in
        lockstep (ADR-0056). The id header is written lazily on the first row, once the
        observation labels are known. If a vector is somehow missing but other chains
        have recorded (essentially impossible past burn-in, where every chain has
        accepted), a NaN row preserves the alignment rather than silently dropping a row."""
        if chain_index is None:
            return
        values = self.current_pointwise_loglik[chain_index]
        if values is None:
            if self._loglik_ids is None:
                return  # nothing recorded anywhere yet -- no row to align to
            values = np.full(len(self._loglik_ids), np.nan)
        with open(self.log_likelihood_file, 'a') as f:
            if not self._loglik_header_written:
                f.write('# ' + '\t'.join(self._loglik_ids) + '\n')
                self._loglik_header_written = True
            f.write('\t'.join('%.17g' % float(v) for v in values) + '\n')

    def report_constraint_satisfaction(self, file_ext):
        """
        Read the constraint samples file and write a summary of constraint satisfaction percentages.
        :param file_ext: String to append to the output file name
        """
        if not self.all_constraints:
            return
        try:
            dat = np.loadtxt(self.constraint_samples_file)
        except (OSError, ValueError):
            return
        if dat.ndim < 2 or dat.shape[0] == 0:
            return
        n_samples = dat.shape[0]
        filepath = self.config.config['output_dir'] + f'/Results/constraint_satisfaction{file_ext}.txt'
        with open(filepath, 'w') as f:
            f.write('# constraint\tpercent_satisfied\tn_satisfied\tn_total\n')
            for i, c in enumerate(self.all_constraints):
                n_satisfied = int(np.sum(dat[:, i]))
                pct = 100.0 * n_satisfied / n_samples
                label = c.source_line or 'constraint_%i' % i
                f.write('%s\t%.1f%%\t%i\t%i\n' % (label, pct, n_satisfied, n_samples))

    def update_histograms(self, file_ext):
        """
        Updates the files that contain histogram points for each variable
        :param file_ext: String to append to the save file names
        :type file_ext: str
        :return:
        """
        # Read the samples file into an array, ignoring the first row (header)
        # and first 2 columns (pset names, probabilities)
        dat_array = np.genfromtxt(self.samples_file, delimiter='\t', dtype=float,
                                  usecols=range(2, len(self.variables)+2))

        if dat_array.ndim < 2 or dat_array.shape[0] == 0:
            logger.warning('No samples collected — skipping histogram generation')
            return

        # Open the file(s) to save the credible intervals
        cred_files = []
        for i in self.credible_intervals:
            f = open(Path(self.config.config['output_dir']) / 'Results' / f'credible{i}{file_ext}.txt', 'w')
            f.write('# param\tlower_bound\tupper_bound\n')
            cred_files.append(f)

        for i in range(len(self.variables)):
            v = self.variables[i]
            fname = self.config.config['output_dir']+f'/Results/Histograms/{v.name}{file_ext}.txt'
            # Bin in the parameter's sampling space u -- log10 for a log variable,
            # identity otherwise; ask the parameter for the transform (#412).
            histdata = v.to_sampling_space(dat_array[:, i])
            # Label the bin edges with the variable's actual sampling space (log10 / ln /
            # linear) so a natural-log parameter is not mislabeled 'log10' (ADR-0043).
            edge_prefix = f'{v.scale_name}_' if v.log_space else ''
            header = f'{edge_prefix}lower_bound\t{edge_prefix}upper_bound\tcount'
            hist, bin_edges = np.histogram(histdata, bins=self.num_bins)
            result_array = np.stack((bin_edges[:-1], bin_edges[1:], hist), axis=-1)
            np.savetxt(fname, result_array, delimiter='\t', header=header)

            sorted_data = sorted(dat_array[:, i])
            for interval, file in zip(self.credible_intervals, cred_files):
                n = len(sorted_data)
                want = n * (interval/100)
                # Clamp to valid indices: an interval >= 100% (or rounding at small n)
                # would otherwise drive max_index to n (IndexError) or min_index < 0
                # (silently wraps to the wrong end via negative indexing).
                min_index = max(0, int(np.round(n/2 - want/2)))
                max_index = min(n - 1, int(np.round(n/2 + want/2 - 1)))
                file.write(f'{v.name}\t{sorted_data[min_index]}\t{sorted_data[max_index]}\n')

        for file in cred_files:
            file.close()

    def compute_rhat(self):
        """Rank-normalized split-R-hat per parameter (Vehtari et al. 2021).

        Thin glue over :func:`pybnf.diagnostics.rhat` (ADR-0009): reads this
        instance's chain history and delegates the pure math. Returns an
        ``(n_dim,)`` array, or ``None`` if there is insufficient history.
        """
        return diagnostics.rhat(self.chain_history, self.num_parallel)

    def compute_ess(self):
        """Bulk and tail effective sample size per parameter (Vehtari et al. 2021).

        Thin glue over :func:`pybnf.diagnostics.ess` (ADR-0009): reads this
        instance's chain history and delegates the pure math. Returns
        ``(bulk_ess, tail_ess)`` arrays of shape ``(n_dim,)``, or ``(None, None)``.
        """
        return diagnostics.ess(self.chain_history, self.num_parallel)

    def report_convergence_diagnostics(self, iteration):
        """
        Compute and report R-hat, ESS, and ESS/evaluation. Called every 10 iterations.
        Returns max_rhat for convergence checking, or None.
        """
        rhat = self.compute_rhat()
        max_rhat = None
        if rhat is not None:
            max_rhat = np.nanmax(rhat)
            print1(f'Max R-hat: {max_rhat:.4f}')
            print2(f'R-hat per parameter: {str(np.round(rhat, 4))}')
            logger.info(f'R-hat values: {str(rhat)}')

        bulk_ess, tail_ess = self.compute_ess()
        if bulk_ess is not None:
            print1(f'Min bulk ESS: {np.nanmin(bulk_ess):.1f}  Min tail ESS: {np.nanmin(tail_ess):.1f}')
            print2(f'Bulk ESS per parameter: {str(np.round(bulk_ess, 1))}')
            print2(f'Tail ESS per parameter: {str(np.round(tail_ess, 1))}')
            logger.info(f'Bulk ESS: {str(bulk_ess)}')
            logger.info(f'Tail ESS: {str(tail_ess)}')

            if self.total_evaluations > 0:
                ess_per_eval = bulk_ess / self.total_evaluations
                print2(f'Bulk ESS/evaluation: {str(np.round(ess_per_eval, 6))}')
                logger.info(f'Bulk ESS/evaluation: {str(ess_per_eval)}')

            # Write diagnostics to file
            self._write_diagnostics(iteration, rhat, bulk_ess, tail_ess)

        return max_rhat

    def check_convergence(self, iteration, max_rhat):
        """Check if R-hat has converged below threshold. Returns True if should stop."""
        if (self.rhat_threshold > 0
                and iteration > self.burn_in
                and max_rhat is not None
                and max_rhat <= self.rhat_threshold):
            print1(f'R-hat converged ({max_rhat:.4f} <= {self.rhat_threshold:.4f}). Stopping.')
            self.update_histograms('_final')
            self.report_constraint_satisfaction('_final')
            return True
        return False

    def _write_diagnostics(self, iteration, rhat, bulk_ess, tail_ess):
        """Append convergence diagnostics to the diagnostics output file."""
        diag_file = Path(self.config.config['output_dir']) / 'Results' / 'diagnostics.txt'
        write_header = not diag_file.exists()
        param_names = [v.name for v in self.variables]
        with open(diag_file, 'a') as f:
            if write_header:
                cols = ['iteration', 'total_evaluations']
                for name in param_names:
                    cols.extend([f'rhat_{name}', f'bulk_ess_{name}', f'tail_ess_{name}'])
                f.write('# ' + '\t'.join(cols) + '\n')
            vals = [str(iteration), str(self.total_evaluations)]
            for i in range(len(param_names)):
                rhat_val = f'{rhat[i]:.6f}' if rhat is not None else 'nan'
                bulk_val = f'{bulk_ess[i]:.2f}' if bulk_ess is not None else 'nan'
                tail_val = f'{tail_ess[i]:.2f}' if tail_ess is not None else 'nan'
                vals.extend([rhat_val, bulk_val, tail_val])
            f.write('\t'.join(vals) + '\n')

    def cleanup(self):
        """Called when quitting due to error.
        Save the histograms in addition to the usual algorithm cleanup"""
        super().cleanup()
        self.update_histograms('_end')
        self.report_constraint_satisfaction('_end')
