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
            self.evaluate_constraints(res.simdata, index)
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
                self.samples_file = str(out / 'Results' / 'A_MCMC' / 'Runs' / 'combined_params.txt')
                self.report_constraint_satisfaction('_final')
                return 'STOP'
            # Check if it's time to report stuff
            if self.iteration[index] % 10 == 0:
                print2(f'Acceptance rates: {str(self.acceptance_rates)}\n')
                print2(f'Current -Ln Posteriors: {str(self.ln_current_P)}')
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
        # WRite out the param. Need more practical method
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        params_file = runs_dir / f'params_{idx}.txt'
        if self.iteration[idx] == self.burn_in - 1:
            self.write_out_p = self.parameter_index[idx][~(self.parameter_index[idx]==0).all(1)]
            varibles = []
            for v in self.variables:
                varibles.append(v.name)
            varNames = '\t'.join(varibles)
            with open(params_file, 'a') as f:
                f.write(varNames+'\n')
        else:
            self.write_out_p = self.parameter_index[idx][~(self.parameter_index[idx]==0).all(1)]
            with open(params_file, 'a') as f:
                np.savetxt(f, self.write_out_p)

    def write_out_trajactorys(self, idx):
        # write out trajectories need more practical method
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        for l in self.output_columns:     
            for i in self.output_run_current.keys():
                if l in i:
                    self.write_out_t = self.output_run_all[i][idx][~(self.output_run_all[i][idx]==0).all(1)]
                    if len(self.write_out_t) != 0:
                        with open(runs_dir / f'traj_{i}_chain_{idx}.txt', 'a') as f:
                            np.savetxt(f, self.write_out_t)
    def write_out_trajactorys_noise(self, idx):
        # Basically this IO on every iter is to expensice timewise
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        for l in self.output_noise_columns:     
            for i in self.output_run_noise_current.keys():
                if l in i: 
                    self.write_out_t = self.output_run_noise_all[i][idx][~(self.output_run_noise_all[i][idx]==0).all(1)]
                    if len(self.write_out_t) != 0:
                        with open(runs_dir / f'traj_noise_{i}_chain_{idx}.txt', 'a') as f:
                            np.savetxt(f, self.write_out_t)
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
    def combine_chains_traj(self):
        # combine the trains for the file output file
        if self.num_parallel != 1:
            runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
            if self.config.config['output_trajectory']:
                for j in range(self.num_parallel):
                    for l in self.output_columns:     
                        for i in self.output_run_current.keys():
                            if l in i:
                                with open(runs_dir / f'combined_traj_{i}.txt', 'a') as f:
                                    file_append = np.loadtxt(runs_dir / f'traj_{i}_chain_{j}.txt')
                                    np.savetxt(f, file_append)
            if self.config.config['output_noise_trajectory']:
                for j in range(self.num_parallel):
                    for l in self.output_noise_columns:     
                        for i in self.output_run_noise_current.keys():
                            if l in i:
                                with open(runs_dir / f'combined_traj_noise_{i}.txt', 'a') as f:
                                    file_append = np.loadtxt(runs_dir / f'traj_noise_{i}_chain_{j}.txt')
                                    np.savetxt(f, file_append)                                     

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
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = self.chain_rngs[idx].multivariate_normal(mean=np.zeros((len_params,)), cov=self.diffMatrix[idx])
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
            oldpset = self.current_pset[idx]
            num = 0
            while num != 10000*len_params:
                new_vars = []
                delta_vector = self.chain_rngs[idx].multivariate_normal(mean=np.zeros((len_params,)), cov=self.diffMatrix[idx])
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
                delta_vector = self.chain_rngs[idx].multivariate_normal(mean=np.zeros((len_params,)), cov=diffMatrix)
                delta_vector_add = {k: self.step_size * delta_vector[i] for i,k in enumerate(oldpset.keys())}
                #delta_vector_multiply_log = {k: self.step_size*delta_vector_log[i] for i,k in enumerate(oldpset.keys())}
                try:
                    for i, p in enumerate(oldpset):
                        k = self.variables[i]
                        # reflect=False during the first 10000 attempts, then True;
                        # FreeParameter.add already applies the proposal in the
                        # parameter's own scale, so there is no log/linear branch.
                        reflect = num >= 10000
                        new_var = oldpset.get_param(k.name).add(delta_vector_add[k.name], reflect)
                        new_vars.append(new_var)
                        if len(new_vars) == len_params:
                            return PSet(new_vars)
                except OutOfBoundsException:
                    num += 1
                    pass        
    
    def update_histograms(self, file_ext):
            pass   
