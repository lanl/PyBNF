"""Adaptive_MCMC — the Adaptive Metropolis sampler (the ``am`` fit type).

PyBNF's recommended Bayesian sampler. Extracted byte-identical (M1 Step 4).
Subclasses the sampler base (BayesianAlgorithm) and inherits the run loop +
execution seam from Algorithm.
"""


from ..core import FailedSimulation
from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...pset import PSet, OutOfBoundsException
from ...printing import print1, print2, PybnfError
from ...registry import register_fit_type

from typing import Any

import numpy as np
import shutil
from pathlib import Path
from scipy import stats


class AdaptiveMCMCConfig(MCMCFamilyConfig):
    """Config for adaptive MCMC (am), co-located with the method (ADR-0006). Adds
    the covariance-adaptation keys ``Adaptive_MCMC`` reads on top of the shared
    family fields; the β-ladder ``postprocess`` hook is inherited."""

    stablizingCov: float = 0.001
    calculate_covari: Any = None


@register_fit_type('am', family='sampler', display_name='Adaptive MCMC',
                   schema=AdaptiveMCMCConfig)
class Adaptive_MCMC(BayesianAlgorithm):
    #: How many proposals left the box and were rejected without a simulation (#709). Also
    #: a class attribute, so that a run resumed from a backup made before the counter
    #: existed reads it as 0 rather than failing at its first wall.
    boundary_rejections = 0

    def __init__(self, config):  # expdata, objective, priorfile, gamma=0.1):
        super().__init__(config)
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
        self.boundary_rejections = 0   # how many of those attempts left the box (#709)
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
        
        out = Path(self.config.config['output_dir'])
        adaptive_dir = out / 'adaptive_files'
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        (out / 'Results' / 'A_MCMC' / 'Runs').mkdir(parents=True, exist_ok=True)
        (out / 'Results' / 'Histograms').mkdir(parents=True, exist_ok=True)
        
        # Defined whether or not either key is set, so the names can be checked against a
        # completed simulation in one place (_check_trajectory_names, #755).
        self.output_columns = []
        self.output_noise_columns = []
        self._trajectory_names_checked = False

        # Which (buffer key, chain) slots a real result column has ever supplied. The write
        # step used to infer this from the values, treating an all-zero row as "nothing was
        # recorded here" -- which is also what a recorded trajectory of an observable that
        # sits at zero looks like, so those samples were dropped from the file (#758).
        self.output_run_recorded = set()
        self.output_run_noise_recorded = set()
        # Likewise for the accepted-parameter history, where the sentinel is a whole
        # parameter vector of zeros. Out of reach for a continuous proposal on more than one
        # parameter, but the fold can return a bound of exactly zero, so it is not a
        # guarantee -- and it costs one flag to stop relying on it.
        self.parameter_index_recorded = [False] * self.num_parallel

        if self.config.config['output_trajectory']:
            # The comma is a list separator that the parser removes (#751). This used to
            # strip one out of each name here, which fixed `A, B` and silently joined
            # `A,B` -- a single token to the parser -- into one name `AB`.
            self.output_columns = list(self.config.config['output_trajectory'])
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
            # Same as above: the separator is the parser's business now (#751).
            self.output_noise_columns = list(self.config.config['output_noise_trajectory'])
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
            required = ['diff.txt', 'MLE_params.txt', 'diffMatrix.txt']
            missing = [f for f in required if not (adaptive_dir / f).exists()]
            if missing:
                raise PybnfError(
                    'continue_run = 1 requires adaptive files from a completed prior run, '
                    'but the following files are missing from {}: {}. '
                    'Run the model first without continue_run, or check that output_dir '
                    'points to a previous run\'s output.'.format(adaptive_dir, ', '.join(missing)))
            self.diff = [self.step_size] * self.num_parallel
            self.diff_best = np.loadtxt(adaptive_dir / 'diff.txt')
            self.diffMatrix = np.zeros((self.num_parallel, len(self.variables), len(self.variables)))
            self.diffMatrix_log = np.zeros((self.num_parallel, len(self.variables), len(self.variables)))
            if self.adaptive != 1:
                self.mle_best = np.loadtxt(adaptive_dir / 'MLE_params.txt')
                self.diffMatrix_best = np.loadtxt(adaptive_dir / 'diffMatrix.txt')
                for i in range(self.num_parallel):
                    self.diffMatrix[i] = np.loadtxt(adaptive_dir / 'diffMatrix.txt')
                    self.diff[i] = np.loadtxt(adaptive_dir / 'diff.txt')
                    
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
        super().reset(bootstrap)

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


        return super().start_run(setup_samples=True)

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
        index = self._chain_index_from_name(pset.name)

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
            if self.chain_rngs[index].random() < self.alpha[index]:
                self.accept = True
        # if accept then update the lists
        if self.accept == True:
            self.current_pset[index] = pset
            self.acceptances += 1
            self.evaluate_constraints(self._result_simdata(res), index)
            self.record_pointwise_loglik(res, index)
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
                self._check_trajectory_names(res.out)
                if self.config.config['output_trajectory']:
                    for l in self.output_columns:
                        for i in res.out:
                            for j in res.out[i]:
                                if (j + l) not in self.output_run_current:
                                    # Off-diagonal <action><condition> cross-product suffix.
                                    # Now defensive: edition-2 pruning (#484, ADR-0069) no longer
                                    # produces off-diagonal suffixes -- res.out carries only the
                                    # scored diagonal (WT, KOko, ...), which is exactly what
                                    # output_run_current allocates. This guard remains so a stray
                                    # unallocated suffix skips rather than KeyError'ing, as it did
                                    # load-bearingly before pruning (lanl/PyBNF#483).
                                    continue
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
                                    # A real result column supplied this slot, which is what
                                    # the write step needs to know -- not whether the values
                                    # happen to be zero (#758).
                                    self.output_run_recorded.add((j + l, index))
                if self.config.config['output_noise_trajectory']:
                    for la in self.output_noise_columns:
                        for ib in res.out:
                            for js in res.out[ib]:
                                if (js + la) not in self.output_run_noise_current:
                                    # Defensive after edition-2 pruning (#484, ADR-0069): only the
                                    # scored diagonal keys are produced and allocated. Skip a stray
                                    # unallocated suffix rather than KeyError (lanl/PyBNF#483).
                                    continue
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
                                    self.output_run_noise_recorded.add((js + la, index))   # (#758)

        self._record_iteration(index)
        return self._run_barrier(index)

    def _record_iteration(self, index):
        """Record chain ``index``'s current point for the iteration that just ended, and
        advance the chain by one.

        This is everything that follows the accept/reject decision and is the same
        whichever way it went: the current point goes into the adaptive history, the
        trajectory and score buffers, the diagnostic history and -- on a sampling
        iteration -- the samples file. It is its own method because an iteration can end
        without a result to decide on: a proposal that leaves the box is rejected before
        it is simulated (:meth:`_reject_at_boundary`, #709), and that iteration is
        recorded exactly as any other rejection is, as the current point once more.
        """
        # After the burn in period start to record the accepted params for the adaptive feature.
        if self.iteration[index] >= self.burn_in:
            self.parameter_index[index][self.factor[index]] = self.current_param_set[index]
            self.parameter_index_recorded[index] = True                                # (#758)
        
        # record the trajactorys for the graphs
        if self.iteration[index] >= self.valid_range and self.iteration[index] % self.config.config['sample_every'] == 0:
            # if the objective function is negbin then add the negbin noise to the traj output else record accepted sim vals as is
            if (self.config.config['objfunc'] == 'neg_bin' and self.config.config['output_noise_trajectory']) or (self.config.config['objfunc'] == 'neg_bin_dynamic' and self.config.config['output_noise_trajectory']):
                for l in self.output_noise_columns:     
                    for i in self.output_run_noise_current.keys():
                        if l in i:
                            self.output_run_noise_all[i][index][self.factor[index]] =  self.generateBinomialNoise(self.output_run_noise_current[i][index][0], self.current_pset[index], self.chain_rngs[index])
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

        # Standard BayesianAlgorithm sampling. Note the window: the samples file, and
        # so the histograms and credible intervals built from it, starts after burn_in,
        # while the trajectory block above and combine_chains_params start at
        # valid_range (burn_in + adaptive). That is deliberate, not an oversight of
        # #771 -- during the adaptive window the chain runs a fixed-step Metropolis
        # kernel against the same posterior, so those draws are valid, just taken
        # before the covariance adapted. The two windows differing is still a wart
        # worth its own look (lanl/PyBNF#772).
        if (self.iteration[index] > self.burn_in
                and self.iteration[index] % self.sample_every == 0):
            self.sample_pset(self.current_pset[index], self.ln_current_P[index], index)
        if (self.iteration[index] > self.burn_in
                and self.iteration[index] % (self.sample_every * self.output_hist_every) == 0):
            self.update_histograms('_%i' % self.iteration[index])

        self.wait_for_sync[index] = True

    def _reject_at_boundary(self, index):
        """Spend chain ``index``'s iteration on a proposal that left the box (#709).

        The posterior is zero outside the box, so the Metropolis ratio of such a proposal
        is zero and it is rejected with certainty. The chain stays where it is, and no
        simulation is run, because nothing a simulation could return would change that.
        ``alpha`` is that zero, so the scale adaptation in :meth:`pick_new_pset`, which
        steers the acceptance rate to 0.234, counts this rejection like any other and
        shortens the step of a chain that keeps walking into a wall. ``total_evaluations``
        is left alone: it counts simulations, and ESS per evaluation is reported from it.
        """
        self.attempts += 1
        self.boundary_rejections += 1
        self.alpha[index] = 0
        self._record_iteration(index)

    def _run_barrier(self, index):
        """Once every chain has finished its iteration, write the generation out, check
        the stopping conditions and propose the next one. Returns the PSets to run,
        ``'STOP'``, or ``[]`` while chains are still outstanding.

        A chain whose proposal leaves the box finishes its next iteration here, without a
        simulation (:meth:`_reject_at_boundary`), so it is already waiting when the others
        report. When that is every chain there is nothing to submit and no result will
        arrive to reach this barrier again -- the scheduler would find its job pool empty
        and end the run -- so the loop goes around and handles the generation that just
        ended the same way. Each pass either returns proposals or advances every chain by
        one iteration, so ``max_iterations`` bounds it. DREAM's barrier has the same loop
        for the same reason (``DreamAlgorithm._run_barrier``).
        """
        # Wait for entire generation to finish
        while np.all(self.wait_for_sync):
            self.acceptance_rates = self.acceptances / self.attempts
            #self.wait_for_sync = [False] * self.num_parallel
            # Increase or reset the factor number and see if it's time to write things out
            for i in range(self.num_parallel):
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
                    out = Path(self.config.config['output_dir'])
                    self.samples_file = str(out / 'Results' / 'A_MCMC' / 'Runs' / 'combined_params.txt')
                    return 'STOP'

            # Set here because I don't want these commands to exacute more then once.
            if min(self.iteration) >= self.max_iterations:
                # Save the current postion of the MCMC run
                self.diff_best = [self.diff_best]
                out = Path(self.config.config['output_dir'])
                adaptive_dir = out / 'adaptive_files'
                np.savetxt(adaptive_dir / 'MLE_params.txt', self.mle_best)
                np.savetxt(adaptive_dir / 'diffMatrix.txt', self.diffMatrix_best)
                np.savetxt(adaptive_dir / 'diff.txt', self.diff_best)
                self.combine_chains_params()
                self.combine_chains_traj()
                # The final histograms and credible intervals, the pair every other
                # sampler writes at its stop point (dream's barrier, the shared
                # check_convergence). Both calls must come BEFORE samples_file is
                # repointed below: update_histograms reads that attribute, and
                # combined_params.txt is a different format -- variable columns only,
                # space separated, no '# Name  Ln_probability' prefix -- so reading it
                # with this method's usecols would not give the sampled values back.
                self.update_histograms('_final')
                self.report_constraint_satisfaction('_final')
                self.samples_file = str(out / 'Results' / 'A_MCMC' / 'Runs' / 'combined_params.txt')
                return 'STOP'
            # Check if it's time to report stuff
            if self.iteration[index] % 10 == 0:
                # The rate counts a proposal that left the box as the rejection it is, so
                # the share of attempts that went that way is reported next to it: those
                # cost no simulation, and a large share says the step is long for the box.
                print2(f'Acceptance rates: {str(self.acceptance_rates)} '
                       f'({self.boundary_rejections} of {self.attempts} proposals left the box)\n')
                print2(f'Current -Ln Posteriors: {str(self.ln_current_P)}')
            print1('Completed iteration %i of %i' % (self.iteration[index], self.max_iterations))

            # Propose next Pset
            next_generation = []
            for i in range(self.num_parallel):
                new_pset = self.pick_new_pset(i)
                if new_pset is None:
                    # Left the box: rejected here and now. The chain stays synced.
                    self._reject_at_boundary(i)
                    continue
                new_pset.name = 'iter%irun%i' % (self.iteration[i], i)
                next_generation.append(new_pset)
                self.wait_for_sync[i] = False
            if next_generation:
                return next_generation
        return []

    def _check_trajectory_names(self, out):
        """Warn once for an ``output_trajectory`` / ``output_noise_trajectory`` name that
        no simulation produced a column for (#755).

        Neither key was validated against anything. A name that matches no column is
        filled by nothing (the accumulate loop below runs only ``if l in cols``), so its
        array stays as allocated -- all zeros -- and the write step skips it, because it
        writes only the rows that are not all zero. The result was a `Runs/` directory
        with fewer ``traj_*.txt`` files than the conf asked for and nothing saying which
        name was dropped or why, so a typo looked exactly like a fit that never asked.

        Checked here rather than at config load, where the rest of the codebase refuses an
        undeclared name (``_resolve_profile_idxs``, ``design.greedy.resolve_targets``), for
        two reasons: no model class exposes its observable names -- ``BNGLModel`` records
        only the boolean ``has_observables`` -- and a column need not come from the model
        file at all, since the measurement layer contributes its own. A completed result
        carries the authoritative list, and carries it for every model in the fit at once
        (``res.out`` is the whole multi-model dict on both the worker-scoring and the
        master-scoring path), so a name valid for one model is not reported against
        another.

        A warning rather than an error: the fit itself is sound, only an output file is
        missing, and by the time this can be known simulations are already running.
        """
        if self._trajectory_names_checked:
            return
        self._trajectory_names_checked = True
        if not self.output_columns and not self.output_noise_columns:
            return
        available = set()
        for model in out:
            for suffix in out[model]:
                available.update(out[model][suffix].cols)
        for key, wanted in (('output_trajectory', self.output_columns),
                            ('output_noise_trajectory', self.output_noise_columns)):
            missing = [name for name in wanted if name not in available]
            if not missing:
                continue
            one = len(missing) == 1
            print1("Warning: %s names %s, which %s of any simulation in this fit, so no "
                   "trajectory will be written for %s. The simulations produce: %s."
                   % (key, ', '.join(missing),
                      'is not a column' if one else 'are not columns',
                      'it' if one else 'them', ', '.join(sorted(available))))

    def generateBinomialNoise(self, timeseries, pset, rng):
        # Generate the binomial noise for the results (rng = the chain's own Generator)
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
            self.output[i] = stats.nbinom.rvs(n=self.r, p=self.prob, size=1, random_state=rng)

        return self.output

    def write_out_scores(self, idx):
        # Write out the scores. Need more practical method
        self.write_out_score = self.scores[idx]
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        with open(runs_dir / f'scores_{idx}.txt', 'a') as f:
            np.savetxt(f, self.write_out_score)

    def write_out_params(self, idx):
        # Write out the accepted-parameter history that seeds the adaptive covariance.
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        params_file = runs_dir / f'params_{idx}.txt'
        # Written when the slot has been recorded, not when some parameter is non-zero
        # (#758): a vector of all zeros is a legal point of a box whose lower bound is
        # zero, which the reflecting fold can return exactly.
        self.write_out_p = (self.parameter_index[idx] if self.parameter_index_recorded[idx]
                            else self.parameter_index[idx][:0])
        # Emit the column-name header exactly once, when the seed file is first created.
        # Keying this on file creation rather than `iteration == burn_in - 1` keeps the
        # header present even when burn_in == 1 makes that iteration unreachable: iteration
        # is already incremented before this write block runs in got_result, so it is always
        # >= 1 here and `burn_in - 1 == 0` never matched -- leaving params_*.txt headerless
        # and breaking the `np.genfromtxt(..., names=True)` covariance seed at iteration
        # burn_in+adaptive ("no field of name <first parameter>"). Surfaced while verifying
        # the #480 fix on the MEK aMCMC example.
        if not params_file.exists() or params_file.stat().st_size == 0:
            varNames = '\t'.join(v.name for v in self.variables)
            with open(params_file, 'a') as f:
                f.write(varNames+'\n')
        # The burn_in-1 call only initializes the header; accepted-sample rows begin at
        # burn_in, preserving the row count the covariance normalization assumes.
        if self.iteration[idx] != self.burn_in - 1:
            with open(params_file, 'a') as f:
                np.savetxt(f, self.write_out_p)

    def write_out_trajactorys(self, idx):
        # write out trajectories need more practical method
        #
        # A slot is written when a result column has supplied it, not when its values are
        # non-zero. Reading an all-zero row as "nothing recorded here" also dropped a
        # recorded trajectory of an observable that sits at zero across the window -- an
        # absent species under a knockout, a _Cum counter over a quiet window -- so the
        # file silently carried fewer rows than the run sampled, with nothing marking the
        # gap, and a band or mean computed from it was over the samples where the
        # observable happened to be switched on (#758). `write_out_scores` never filtered.
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        for l in self.output_columns:
            for i in self.output_run_current.keys():
                if l in i and (i, idx) in self.output_run_recorded:
                    with open(runs_dir / f'traj_{i}_chain_{idx}.txt', 'a') as f:
                        np.savetxt(f, self.output_run_all[i][idx])
    def write_out_trajactorys_noise(self, idx):
        # Basically this IO on every iter is to expensice timewise
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        for l in self.output_noise_columns:
            for i in self.output_run_noise_current.keys():
                if l in i and (i, idx) in self.output_run_noise_recorded:   # (#758, as above)
                    with open(runs_dir / f'traj_noise_{i}_chain_{idx}.txt', 'a') as f:
                        np.savetxt(f, self.output_run_noise_all[i][idx])
    def combine_chains_params(self):
        #combine the chains for the final output file
        # if self.num_parallel != 1:
        out = Path(self.config.config['output_dir'])
        runs_dir = out / 'Results' / 'A_MCMC' / 'Runs'
        combined_file = runs_dir / 'combined_params.txt'
        with open(combined_file, 'w') as f:
            varsnNames = []
            for v in self.variables:
                varsnNames.append(v.name)
            varsNames = '\t'.join(varsnNames)    
            f.write(varsNames+'\n')
            for i in range(self.num_parallel):
                file_append = np.loadtxt(runs_dir / f'params_{i}.txt', skiprows=1)
                file_append = file_append[self.adaptive:]
                np.savetxt(f, file_append)   
        shutil.copyfile(combined_file, out / 'adaptive_files' / 'combined_params.txt')      
    def _combine_chain_files(self, runs_dir, columns, buffers, per_chain, combined, missing):
        """Concatenate each key's per-chain files into one, chain by chain.

        A source file that was never written is skipped and recorded in ``missing``
        rather than loaded. It used to be loaded unconditionally, so the absence of an
        output the run itself declined to write ended the run in ``FileNotFoundError``
        on the last step before ``'STOP'``, after all the sampling work was done (#760).
        The combined file is opened only once there is something to put in it, so a key
        with no source files leaves no empty file behind either."""
        for j in range(self.num_parallel):
            for l in columns:
                for i in buffers:
                    if l not in i:
                        continue
                    source = runs_dir / per_chain.format(key=i, chain=j)
                    if not source.is_file():
                        missing.append((i, j))
                        continue
                    # atleast_2d: loadtxt drops a one-row file to 1-D, and savetxt then
                    # writes that single sample down the combined file as one value per
                    # line instead of across one row -- a run with exactly one sampling
                    # iteration past valid_range wrote its trajectory transposed.
                    with open(runs_dir / combined.format(key=i), 'a') as f:
                        np.savetxt(f, np.atleast_2d(np.loadtxt(source)))

    def combine_chains_traj(self):
        # combine the trains for the file output file
        if self.num_parallel != 1:
            runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
            missing = []
            if self.config.config['output_trajectory']:
                self._combine_chain_files(
                    runs_dir, self.output_columns, self.output_run_current,
                    'traj_{key}_chain_{chain}.txt', 'combined_traj_{key}.txt', missing)
            if self.config.config['output_noise_trajectory']:
                self._combine_chain_files(
                    runs_dir, self.output_noise_columns, self.output_run_noise_current,
                    'traj_noise_{key}_chain_{chain}.txt', 'combined_traj_noise_{key}.txt',
                    missing)
            if missing:
                # Something the conf asked for is not in the combined output, and both ways
                # of getting here are worth naming: the run may have stopped inside the
                # adaptive window, before any trajectory is written, or a name may be one no
                # simulation ever produced a column for (#755). Chains can differ, since a
                # convergence stop fires on one chain's iteration count while the others are
                # still behind it.
                print1("Warning: the combined trajectory output is missing %d per-chain "
                       "file%s the run never wrote: %s. A trajectory is written only from "
                       "iteration %d (burn_in + adaptive), so a run that stopped before then "
                       "has none, and a name no simulation produced a column for has none "
                       "either."
                       % (len(missing), '' if len(missing) == 1 else 's',
                          ', '.join(f'{key} chain {chain}' for key, chain in missing),
                          self.valid_range))

    def pick_new_pset(self, idx):
        """
        :param idx: Index of PSet to update
        :return: A mew
        """
                   
        # Chain state in sampling space u (base-10 log for a log parameter),
        # consistent with how the proposal is applied (FreeParameter.add ->
        # 10**(log10(value)+summand)) and with the rest of the codebase (loguniform
        # prior, prior_logpdf, _param_vec R-hat history, FreeParameter.diff all use
        # log10). Ask the parameter for the transform rather than inlining it (#412).
        params = [var.to_sampling_space(self.current_pset[idx].get_param(var.name).value)
                  for var in self.variables]
        len_params = len(params)
        self.stablizingCov = self.config.config['stablizingCov']*np.eye(len_params)
        if self.iteration[idx] >= self.burn_in + self.adaptive:
            if self.iteration[idx] == self.burn_in + self.adaptive:
                runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
                self.parameter_index_file_input = np.genfromtxt(runs_dir / f'params_{idx}.txt', names = True)
                for v in self.variables:
                    # Read the seed history into sampling space u via the parameter's
                    # scale (log10 for a log variable, identity otherwise) (#412).
                    self.parameter_index_file_input[v.name] = v.to_sampling_space(
                        self.parameter_index_file_input[v.name])
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
            return self._propose(idx, self.diffMatrix[idx], self.diff[idx])
        elif self.config.config['continue_run'] == 1:
            if self.config.config['calculate_covari']:
                start_end = self.config.config['calculate_covari']
                start = int(start_end[0])
                end = int(start_end[1])
                if self.iteration[idx] == 1:
                    adaptive_dir = Path(self.config.config['output_dir']) / 'adaptive_files'
                    self.parameter_index_file_input = np.genfromtxt(adaptive_dir / 'combined_params.txt', names = True)
                    for v in self.variables:
                        # Read the seed history into sampling space u via the
                        # parameter's scale (#412).
                        self.parameter_index_file_input[v.name] = v.to_sampling_space(
                            self.parameter_index_file_input[v.name])
                    self.parameter_index_file_range = self.parameter_index_file_input.view((np.float64, len(self.parameter_index_file_input.dtype.names)))
                    self.parameter_index_file = self.parameter_index_file_range[start:end]
                    self.mu[idx] = np.reshape(np.mean(self.parameter_index_file,axis=0), [1, len_params])  # compute the mean parameters along the past chain 
                    self.diffMatrix[idx] = (np.matmul(self.parameter_index_file.T, self.parameter_index_file)-np.matmul(self.mu[idx].T, self.mu[idx]))/(len(self.parameter_index_file_input)*0.75)
                    self.diff[idx] = self.config.config['step_size']
            return self._propose(idx, self.diffMatrix[idx], self.diff[idx])
        else:
            return self._propose(idx, np.eye(len_params), self.step_size)

    def _propose(self, idx, cov, scale):
        """One Gaussian random-walk proposal from chain ``idx``'s current point:
        ``current + scale * N(0, cov)`` in sampling space. Returns the proposed PSet, or
        ``None`` if any component leaves its parameter's box.

        ``None`` is a rejection, which the caller spends the iteration on
        (:meth:`_reject_at_boundary`). The draw is made once. This used to redraw until a
        proposal landed in the box, and ``got_result`` then accepted it with the plain
        Metropolis ratio, which is only right for a symmetric proposal -- and that one is
        not. Redrawing makes the proposal density the Gaussian renormalized over the box,
        ``q(x -> y) = phi(y - x) / Z(x)`` with ``Z(x)`` the share of the Gaussian at ``x``
        that lies inside, and ``Z`` falls toward a wall. Detailed balance with the plain
        ratio then holds for ``pi(x) Z(x)``, not ``pi(x)``: the chain under-sampled a shell
        a few proposal widths deep along every wall, which is where the edge of a credible
        interval sits whenever a parameter presses against its bound (#709). A single draw
        from the untruncated Gaussian is symmetric, and rejecting the part of it that
        falls outside is the Metropolis rule applied to a posterior that is zero there.

        The fold that ``mh`` and ``pt`` use (``reflect=True``) is not an alternative here.
        A fold is applied to each coordinate separately, and the folded proposal is
        symmetric only if the Gaussian is unchanged by flipping the sign of one coordinate:
        true of their isotropic step, false of a covariance with off-diagonal terms, which
        is what this sampler adapts to. Folding a correlated proposal keeps every marginal
        right and gets the joint wrong, piling the chain into the corners the correlation
        points at.

        The component for ``self.variables[i]`` is ``delta[i]``, the order the adapted
        mean and covariance are kept in.
        """
        oldpset = self.current_pset[idx]
        delta = self.chain_rngs[idx].multivariate_normal(mean=np.zeros((len(self.variables),)), cov=cov)
        try:
            # FreeParameter.add applies the step in the parameter's own scale, so there is
            # no log/linear branch; reflect=False makes a step out of the box raise.
            return PSet([oldpset.get_param(v.name).add(scale * delta[i], False)
                         for i, v in enumerate(self.variables)])
        except OutOfBoundsException:
            return None
