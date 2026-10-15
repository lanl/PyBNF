"""Adaptive_MCMC — the Adaptive Metropolis sampler (the ``am`` fit type).

PyBNF's recommended Bayesian sampler. Extracted byte-identical (M1 Step 4).
Subclasses the sampler base (BayesianAlgorithm) and inherits the run loop +
execution seam from Algorithm.
"""


from ..core import FailedSimulation
from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...pset import PSet, OutOfBoundsException, BNGLModel
from ...printing import print0, print1, print2, PybnfError
from ...registry import register_fit_type
from ...noise import ConstantSigma, FreeParameterSigma, NegBinomial
from ...noise.negative_binomial import draw_count
from ...noise.source import FormulaSigma
from ...objective import NegBinLikelihood, NegBinLikelihood_Dynamic
from types import SimpleNamespace

from typing import Any

import logging
import numpy as np
import re
import shutil
from pathlib import Path

# The samplers' module logger name (samplers/base.py), so these records keep that channel.
logger = logging.getLogger('pybnf.algorithms')


class AdaptiveMCMCConfig(MCMCFamilyConfig):
    """Config for adaptive MCMC (am), co-located with the method (ADR-0006). Adds
    the covariance-adaptation keys ``Adaptive_MCMC`` reads on top of the shared
    family fields; the β-ladder ``postprocess`` hook is inherited."""

    stablizingCov: float = 0.001
    calculate_covari: Any = None


def _noise_spec(alg, column):
    """``column``'s noise draw spec, resolved first if ``alg`` came from a backup made before
    the specs existed (#986: ``-r`` unpickles without running ``__init__``)."""
    if alg.output_noise_specs is None:
        alg.output_noise_specs = {c: alg._noise_draw_spec(c) for c in alg.output_noise_columns}
    return alg.output_noise_specs[column]


def _experiment_suffixes(config):
    """``{model name: [output suffix, ...]}``: the simulation outputs whose trajectories am
    records, which is every experiment of the fit, conditions and mutants included (#990).

    An output suffix is an action suffix followed by a condition's or a mutant's suffix (none
    for the model as written), and ``Model.get_suffixes`` lists every one a model produces.
    These used to be the keys of ``time_length`` (since removed, #998), which held the action
    suffixes alone, so an experiment under a condition (``epi2`` + ``hi``) or an edition-1
    mutant had no trajectory buffer, and its rows were dropped without a message.

    Under edition 2 a model runs each experiment's action under every condition the model
    declares, but an experiment is the one pair its ``experiment:`` line names -- its data key.
    The other pairs are nobody's experiment (nothing scores them, and the emit-set pruning of
    #484 does not simulate them where it can tell), so they are left out."""
    named = {}
    for name, (base, data_key) in config._experiment_data_keys.items():
        named.setdefault(base, {})[name] = data_key
    suffixes = {}
    for model_name, model in config.models.items():
        experiments = named.get(model_name, {})
        conditions = [''] + [m.suffix for m in getattr(model, 'mutants', ())]
        unpaired = ({action + c for action in experiments for c in conditions}
                    - set(experiments.values()))
        suffixes[model_name] = [s for s in dict.fromkeys(model.get_suffixes())
                                if s not in unpaired]
    return suffixes


def _prints_functions(action):
    """Whether a BNGL action's output carries the model's functions: ``True`` or ``False`` when
    the action says so, ``None`` when its text cannot be read.

    BNG2.pl and bngsim write a function column only under ``print_functions=>1``, and every
    action PyBNF synthesizes for an edition-2 experiment asks for it (``BNGLModel.add_action``).
    A hand-written action is read with the bngsim bridge's own action parser."""
    from ...bngsim_model.parsing import _parse_action_dict
    if not isinstance(action, str):
        return True
    action = action.strip().rstrip(';')       # BNG2.pl reads `simulate({...});` the same way
    name = re.match(r'\s*(\w+)\s*\(', action)
    params = _parse_action_dict(action, name.group(1)) if name else None
    if params is None:
        return None
    value = str(params.get('print_functions', '0')).strip()
    if value in ('0', ''):
        return False
    try:
        return float(value) != 0.0
    except ValueError:
        return None           # an expression: what it comes to is BNG2.pl's business


def _model_columns(config, suffixes):
    """``(listed, known, certain)``: the columns the fit's simulations can produce, read from the
    model files before anything runs, or ``None`` when they cannot be known in advance (#990).

    * ``listed`` are the names to offer a user who misspelled one: the observables and functions
      of a BNGL model, the species of an SBML or Antimony model, and the observables the
      measurement layer defines.
    * ``known`` is what a name is accepted against, and is wider on purpose: every identifier
      the model files contain, so that a column the parse does not list (the swept parameter of
      a scan, a species of a backend that writes them) is never refused here. A name this lets
      through but no simulation produces is refused at the first simulation instead
      (:meth:`Adaptive_MCMC._check_trajectory_names`).
    * ``certain`` is ``{model: {output suffix: columns}}``, the columns each experiment of
      ``suffixes`` is sure to have: a BNGL model's observables, its functions when the action
      prints them (:func:`_prints_functions`), an SBML model's species, and the measurement
      layer's observables, which it adds to every output. Two experiments that would write one
      file are refused from it when the configuration loads.

    ``None`` for a fit with a ``postprocess`` script, which may add or remove columns, and for a
    model that is neither BNGL nor SBML (an analytical target), whose outputs no file lists."""
    from ...petab._bngl import parse_model as parse_bngl
    from ...petab._sbml import parse_model as parse_sbml
    if config.postprocessing:
        return None
    measured = {k[1] for k in config.config if isinstance(k, tuple) and k[0] == 'measurement'}
    # Of those, the ones the measurement layer writes into every output as a column; a
    # row-varying one is bound per data point instead (ADR-0045) and is no column.
    layer = config.obj.measurement
    materialized = {m.observable_id for m in layer.models} if layer else set()
    listed, known, certain = set(measured), {'time'} | measured, {}
    for model_name, model in config.models.items():
        outputs = suffixes.get(model_name, ())
        if isinstance(model, BNGLModel):
            entities = parse_bngl(model.bngl_file_text)
            listed |= entities.observable_names | entities.function_names
            text = re.sub(r'#[^\n]*', '', model.bngl_file_text)   # a word in a comment is no name
            prints = {}
            for action in model.actions:
                tag = (action.suffix if not isinstance(action, str)
                       else (BNGLModel._get_action_suffix(action) or (None, None))[1])
                if tag is not None:
                    prints.setdefault(tag, set()).add(_prints_functions(action))
            conditions = [''] + [m.suffix for m in model.mutants]
            certain[model_name] = {}
            for output in outputs:
                flags = set().union(*(prints[a] for a in prints
                                      if any(output == a + c for c in conditions)))
                certain[model_name][output] = (
                    entities.observable_names | materialized
                    | (entities.function_names if flags == {True} else set()))
        else:
            # bngsim's SBML and Antimony models keep the SBML text they loaded (an Antimony
            # model's converted); RoadRunner's SBML model keeps only the path.
            text = getattr(model, '_base_sbml_text', None)
            path = getattr(model, 'abs_file_path', None)
            if text is None and path is not None and str(path).endswith('.xml'):
                text = Path(path).read_text(encoding='utf-8', errors='replace')
            if text is None:
                return None
            species = parse_sbml(text).species_names
            listed |= species
            certain[model_name] = {output: species | materialized for output in outputs}
        known |= set(re.findall(r'[A-Za-z_]\w*', text))
    return listed, known | listed, certain


def _shared_file(key, outputs, wanted):
    """Refuse two (model, experiment, column) sources of ``key`` whose files are one name.

    ``outputs`` is ``{model: {output suffix: columns}}``. A trajectory file is named by the
    output suffix followed by the column, with neither the model nor a separator in it, so two
    models that share an experiment suffix and a column, or two splits of one name (``epi`` +
    ``cases2`` and ``epic`` + ``ases2``), would write their rows into one file."""
    files = {}
    for model, experiments in sorted(outputs.items()):        # a stable message
        for suffix, columns in experiments.items():
            for column in wanted:
                if column not in columns:
                    continue
                source = (model, suffix, column)
                first = files.setdefault(suffix + column, source)
                if first != source:
                    raise PybnfError(
                        "%s would write two trajectories to one file, %s" % (key, suffix + column),
                        "%s names its files by experiment and column, and %s of model %s in "
                        "experiment %s and %s of model %s in experiment %s both come to %s. "
                        "Rename a column or an experiment so the two differ."
                        % (key, first[2], first[0], first[1], column, model, suffix,
                           suffix + column))


@register_fit_type('am', family='sampler', display_name='Adaptive MCMC',
                   schema=AdaptiveMCMCConfig)
class Adaptive_MCMC(BayesianAlgorithm):
    #: How many proposals left the box and were rejected without a simulation (#709). Also
    #: a class attribute, so that a run resumed from a backup made before the counter
    #: existed reads it as 0 rather than failing at its first wall.
    boundary_rejections = 0
    #: Each output_noise_trajectory column's draw (#986), resolved on first use. Also a class
    #: attribute, so a run resumed from a backup made before it existed resolves them then.
    output_noise_specs = None
    #: ``{model: [output suffix]}`` whose trajectories are recorded (:func:`_experiment_suffixes`,
    #: #990), resolved on first use for a run resumed from a backup made before it existed.
    trajectory_suffixes = None

    def __init__(self, config):  # expdata, objective, priorfile, gamma=0.1):
        super().__init__(config)
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
        
        # The comma is a list separator that the parser removes (#751). This used to strip
        # one out of each name here, which fixed `A, B` and silently joined `A,B` -- a single
        # token to the parser -- into one name `AB`.
        self.output_columns = list(self.config.config['output_trajectory'] or [])
        self.output_noise_columns = list(self.config.config['output_noise_trajectory'] or [])
        self._trajectory_names_checked = False
        # One buffer per (experiment, column), keyed by the experiment's output suffix followed
        # by the column, and allocated when a result first supplies it, at the length that
        # result has (#990). Sizing them in advance from `time_length` keyed them by action
        # suffix, which an experiment under a condition or a mutant does not have.
        self.output_run_current = {}
        self.output_run_all = {}
        self.output_run_noise_current = {}
        self.output_run_noise_all = {}

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

        if self.output_columns or self.output_noise_columns:
            self.trajectory_suffixes = _experiment_suffixes(self.config)
            self._check_trajectory_outputs()
        if self.output_noise_columns:
            # Each column's counts are drawn from the noise model that scores it (#986).
            self.output_noise_specs = {c: self._noise_draw_spec(c) for c in self.output_noise_columns}
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
            self._record_trajectories(res, index)

        self._record_iteration(index)
        return self._run_barrier(index)

    def _record_trajectories(self, res, index):
        """Put the accepted result's trajectories in chain ``index``'s slots.

        The simulation data is read wherever the objective was scored
        (:meth:`_result_simdata`): ``res.out`` on the worker-scoring path, ``res.simdata`` on
        the master-scoring one. This used to read ``res.out`` alone, which only the worker path
        and ``parallelize_models > 1`` set, so ``local_objective_eval = 1`` and ``smoothing > 1``
        stopped at the first accepted result, and so did a chain whose first simulation failed:
        a first result is always accepted, and a :class:`FailedSimulation` carries no ``out``
        (#990). A failed simulation has no trajectory, so nothing is recorded for it.

        Either way the data has been normalized and post-processed already, once, by whichever
        side scored it, and what is recorded is the series the objective read its predictions
        from. It used to be normalized a second time here, with the whole-fit normalization map
        passed where one experiment's transforms belong, which raised (``ValueError: too many
        values to unpack``) at the first accepted result of any fit with both a normalization and
        a trajectory output (#990).

        Only the experiments' outputs are recorded (:func:`_experiment_suffixes`). An edition-2
        action under a condition its experiment does not name is not one; when the emit-set
        pruning cannot skip it (#484, ADR-0069) it still reaches here, and is passed over
        (#483).

        Nor is anything recorded for a result without a finite score: a chain's first result is
        accepted whatever its score, and one the scoring penalized (a ``postprocess`` script that
        failed, an objective that could not be evaluated) may carry data it never finished with.
        The worker path drops that data, the master path keeps it, and the first-simulation name
        check read such data as the fit's columns, so the two paths ended differently (#990)."""
        simdata = self._result_simdata(res)
        if isinstance(res, FailedSimulation) or simdata is None or not np.isfinite(res.score):
            return
        if not self.output_columns and not self.output_noise_columns:
            return
        self._check_trajectory_names(simdata)
        for model, outputs in simdata.items():
            for suffix in self._trajectory_suffixes().get(model, ()):
                data = outputs.get(suffix)
                if data is None:
                    continue
                for column in self.output_columns:
                    if column in data.cols:
                        self._record_column(
                            self.output_run_current, self.output_run_all,
                            self.output_run_recorded, suffix, column, data, index,
                            differenced='_Cum' in column)
                for column in self.output_noise_columns:
                    if column in data.cols:
                        self._record_column(
                            self.output_run_noise_current, self.output_run_noise_all,
                            self.output_run_noise_recorded, suffix, column, data, index,
                            differenced=_noise_spec(self, column)[2])

    def _record_column(self, current, everything, recorded, suffix, column, data, index,
                       differenced):
        """Copy ``column`` of one experiment's output into chain ``index``'s slot of its buffer,
        allocating the buffer at that output's length the first time it is supplied."""
        values = np.asarray(data.data[:, data.cols[column]], dtype=float)
        if differenced:
            values = np.concatenate((values[:1], np.diff(values)))
        key = suffix + column
        if key not in current:
            current[key] = np.zeros((self.num_parallel, 1, len(values)))
            everything[key] = np.zeros((self.num_parallel, self.arr_length, len(values)))
        elif current[key].shape[-1] != len(values):
            raise PybnfError(
                "Experiment %s's simulation gave %d rows of %s, where an earlier one gave %d"
                % (suffix, len(values), column, current[key].shape[-1]),
                "A trajectory file holds one row per sample, each of one length, and the "
                "simulation of experiment %s has changed how many output times it has from one "
                "parameter set to the next." % suffix)
        current[key][index] = values
        # A real result column supplied this slot, which is what the write step needs to
        # know -- not whether the values happen to be zero (#758).
        recorded.add((key, index))

    def _trajectory_suffixes(self):
        """``{model: [output suffix]}`` whose trajectories are recorded, resolved from the
        configuration for a run resumed from a backup made before it was stored (#990)."""
        if self.trajectory_suffixes is None:
            self.trajectory_suffixes = _experiment_suffixes(self.config)
        return self.trajectory_suffixes

    def prepare_resume(self):
        """Report experiments whose trajectories a resumed run records for the first time (#990).

        A backup made before #990 recorded trajectories only for the action suffixes of
        ``time_length``, which it kept as ``self.time``: an experiment under a condition or an
        edition-1 mutant had none. Resumed, such a run records them from here on. The backup
        holds no simulation of a chain's current point for them, so a chain's files for them
        begin with the first move it accepts after the resume, while the other experiments'
        files already hold rows: they are shorter, and do not line up row for row. That is said
        here rather than left for a reader to find in the row counts."""
        super().prepare_resume()
        if self.trajectory_suffixes is not None or not (self.output_columns
                                                        or self.output_noise_columns):
            return
        before = vars(self).get('time')      # what a backup made before #990 recorded
        suffixes = self._trajectory_suffixes()
        if before is None:
            return
        new = [s for outputs in suffixes.values() for s in outputs if s not in before]
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'

        def rows(name):
            path = runs_dir / name
            return len(path.read_text().splitlines()) if path.is_file() else 0

        written = {chain: max((rows('%s%s%s_chain_%d.txt' % (prefix, s, column, chain))
                               for prefix, columns in (('traj_', self.output_columns),
                                                       ('traj_noise_', self.output_noise_columns))
                               for column in columns for s in before), default=0)
                   for chain in range(self.num_parallel)}
        if not new or not any(written.values()):
            return
        counts = set(written.values())
        already = ('%d rows in every chain' % counts.pop() if len(counts) == 1 else
                   ', '.join('%d rows in chain %d' % (n, c) for c, n in sorted(written.items())))
        message = ("Note: this run was backed up before PyBNF recorded the trajectories of "
                   "experiment%s %s. A chain's traj_* files for %s begin with the first move it "
                   "accepts after iteration %d, where the run resumes, so they are shorter than "
                   "the other experiments' files, which already hold %s, and do not line up "
                   "with them row for row."
                   % ('' if len(new) == 1 else 's', ', '.join(new),
                      'it' if len(new) == 1 else 'them', min(self.iteration), already))
        logger.warning(message)
        print0(message)

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
            # Draw each noise column's counts around the recorded prediction. _noise_draw_spec
            # refused, at load, a column whose noise model is not a negative binomial (#986).
            if self.config.config['output_noise_trajectory']:
                for l in self.output_noise_columns:
                    for i in self._keys_of(l, self.output_run_noise_current):
                        self.output_run_noise_all[i][index][self.factor[index]] =  self.generateBinomialNoise(self.output_run_noise_current[i][index][0], self.current_pset[index], self.chain_rngs[index], l)
            if self.config.config['output_trajectory']:
                for l in self.output_columns:
                    for i in self._keys_of(l, self.output_run_current):
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
                # check_convergence).
                #
                # samples_file stays on samples.txt after the stop. Both stop paths used
                # to repoint it at combined_params.txt, which nothing read except
                # update_histograms, and that cannot read it: combined_params.txt has
                # only the variable columns, space separated, under a header without
                # the '# Name  Ln_probability' prefix. So when a later step of the run
                # failed, the error-exit cleanup (cleanup -> update_histograms('_end'))
                # failed too and wrote no _end histograms or credible intervals
                # (lanl/PyBNF#856 follow-up).
                self.update_histograms('_final')
                self.report_constraint_satisfaction('_final')
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

    def _check_trajectory_outputs(self):
        """Refuse, when the configuration loads, an ``output_trajectory`` or
        ``output_noise_trajectory`` that would be missing or wrong (#990).

        * A name no model file or measurement formula defines. It used to be reported only at
          the first simulation, at ``verbosity >= 1``, and the run went on to exit 0 without
          the file (#755).
        * Two experiments whose trajectories would be written to one file
          (:func:`_shared_file`), from the columns each is sure to have.
        * A column ``normalization`` scales analytically (``scale``). The factor is profiled
          against the data inside the objective at each evaluation and is not part of the
          simulated series, so the file would hold the prediction off the data's scale by the
          factor the fit applied. The configuration now refuses ``scale`` for every sampler
          before ``am`` is built (#996), so a configuration does not reach this; it stays for
          the day a sampler takes ``scale`` with its marginal, when the file would still be
          wrong.
        * An ``output_noise_trajectory`` column an experiment normalizes. Its counts are drawn
          from the negative binomial whose mean is the prediction, and a normalized prediction
          (a ratio to the first or peak value, a z-score, a floored value) is not an expected
          count, so integer counts drawn around it describe nothing that was or could be
          measured. ``output_trajectory`` records the normalized series, which is what the
          objective compares with the data.

        Where the columns cannot be known before a simulation runs (:func:`_model_columns` says
        when), the first two are refused at the first simulation instead
        (:meth:`_check_trajectory_names`)."""
        model_columns = _model_columns(self.config, self.trajectory_suffixes)
        for key, wanted in (('output_trajectory', self.output_columns),
                            ('output_noise_trajectory', self.output_noise_columns)):
            if model_columns is not None:
                listed, known, certain = model_columns
                missing = [name for name in wanted if name not in known]
                if missing:
                    one = len(missing) == 1
                    raise PybnfError(
                        "%s names %s, which %s not a column any simulation in this fit produces"
                        % (key, ', '.join(missing), 'is' if one else 'are'),
                        "%s names %s, which %s not an observable or function of a BNGL model, "
                        "a species of an SBML model, or an observable a measurement formula "
                        "defines. The names it could match are: %s."
                        % (key, ', '.join(missing), 'is' if one else 'are',
                           ', '.join(sorted(listed))))
                _shared_file(key, certain, wanted)
            scaled = self.config.config.get('analytic_scale') or {}
            for column in wanted:
                where = sorted(dk for dk, scaled_columns in scaled.items()
                               if column in scaled_columns)
                if where:
                    raise PybnfError(
                        "%s names %s, which normalization scales analytically" % (key, column),
                        "normalization scales %s analytically (scale) in experiment %s. That "
                        "factor is profiled against the data inside the objective, so it is not "
                        "part of the simulated trajectory %s records, and the file would hold "
                        "the prediction off the data's scale by it. Remove scale from %s's "
                        "normalization, or %s from %s."
                        % (column, ', '.join(self._experiment_label(dk) for dk in where), key,
                           column, column, key))
        for column in self.output_noise_columns:
            where = self._normalizing_experiments(column)
            if where:
                raise PybnfError(
                    "output_noise_trajectory names %s, which normalization transforms"
                    % column,
                    "output_noise_trajectory draws counts from the negative binomial whose mean "
                    "is %s's prediction, and normalization makes that prediction a normalized "
                    "value in experiment %s, not an expected count, so counts drawn around it "
                    "mean nothing. Remove %s's normalization, or list it under output_trajectory "
                    "instead, which records the normalized trajectory the objective scores."
                    % (column, ', '.join(self._experiment_label(dk) for dk in where), column))

    def _experiment_label(self, data_key):
        """How a message names the experiment of ``data_key``: an edition-2 experiment by its
        name and condition (``epi2 (condition hi)`` for ``epi2hi``), any other by its data key,
        which in edition 1 is the data file's name without ``.exp``. The configuration's
        refusals name experiments the same way (``Configuration.experiment_label``)."""
        return self.config.experiment_label(data_key)

    def _normalizing_experiments(self, column):
        """The experiments, by output suffix, whose ``normalization`` transforms ``column``
        before it is scored -- read as ``Result.normalize`` applies it: each output suffix
        normalized by the transforms of the key it equals.

        The compiled setting is ``{data key: [(transform, [columns])]}``. The legacy per-file
        form also keeps each ``.exp`` file name it was written with as a key, holding the
        transform as written (``{'a.exp': 'init', 'a': [('init', ['cases'])]}``). No output
        suffix is a file name, so those keys normalize nothing; reading one as an experiment
        refused a column that file does not even measure (#990)."""
        settings = self.config.config['normalization']
        if not settings:
            return []
        recorded = sorted({s for outputs in self._trajectory_suffixes().values() for s in outputs})
        if isinstance(settings, str):
            return recorded          # one whole-fit transform of every column of every output
        return [s for s in recorded if s in settings and (
            isinstance(settings[s], str)
            or any(column in columns for _transform, columns in settings[s]))]

    def _check_trajectory_names(self, simdata):
        """Refuse, at the first completed simulation, an ``output_trajectory`` /
        ``output_noise_trajectory`` name that no experiment's output has a column for, and two
        experiments whose trajectories would be written to one file (#755, #990).

        Most of both are refused when the configuration loads (:meth:`_check_trajectory_outputs`).
        This catches the rest: a name some model file mentions without its being an output
        column; two experiments sharing a file through a column that is not sure to be written
        (a function whose action's ``print_functions`` cannot be read); and everything about a
        fit whose columns cannot be listed in advance, because a ``postprocess`` script may add
        or remove columns, or a model is an analytical target. A completed result carries the
        authoritative list, for every model in the fit at once, so a name valid for one model
        is not refused against another. It used to be a warning, printed at
        ``verbosity >= 1``, and the run went on to exit 0 with no file for the name."""
        if self._trajectory_names_checked:
            return
        produced = {model: {suffix: set(outputs[suffix].cols)
                            for suffix in self._trajectory_suffixes().get(model, ())
                            if suffix in outputs}
                    for model, outputs in simdata.items()}
        available = set().union(*(columns for experiments in produced.values()
                                  for columns in experiments.values()))
        for key, wanted in (('output_trajectory', self.output_columns),
                            ('output_noise_trajectory', self.output_noise_columns)):
            missing = [name for name in wanted if name not in available]
            if missing:
                one = len(missing) == 1
                raise PybnfError(
                    "%s names %s, which %s not a column of any experiment's simulation"
                    % (key, ', '.join(missing), 'is' if one else 'are'),
                    "%s names %s, which %s not a column of any experiment's simulation in this "
                    "fit, so no trajectory could be written for %s. The simulations produce: %s."
                    % (key, ', '.join(missing), 'is' if one else 'are', 'it' if one else 'them',
                       ', '.join(sorted(available))))
            _shared_file(key, produced, wanted)
        self._trajectory_names_checked = True

    def _keys_of(self, column, buffers):
        """The buffer keys of ``column``: one per experiment's output suffix, matched exactly.
        A test of ``column in key`` also matched a column whose name is inside another's key
        (``cases`` in ``epicases2``), which wrote that key's rows twice and, once each column
        had a noise model of its own, drew it with the wrong one (#986)."""
        keys = dict.fromkeys(suffix + column for suffixes in self._trajectory_suffixes().values()
                             for suffix in suffixes)
        return [k for k in keys if k in buffers]

    def _noise_draw_spec(self, column):
        """``(family, dispersion source, cumulative)`` for one ``output_noise_trajectory``
        column: the noise model that scores it, which must be a negative binomial with a fixed,
        fitted or parameter-formula dispersion. The draw used to be gated on the legacy
        ``objfunc``, which edition 2 does not have, so an edition-2 run wrote files of zeros
        (#986); a legacy ``neg_bin`` / ``neg_bin_dynamic`` resolves to the same spec. A column
        no data scores takes the whole-fit noise model.

        The draw is differenced as the scoring is (the column's ``cumulative`` declaration),
        with one exception kept for the legacy objfuncs' own default spec: a ``_Cum`` column
        has always been drawn differenced there, and still is, byte for byte, even under
        ``neg_bin``, whose scoring does not difference it."""
        from ...measurement.time_error import MarginalizedTimeObjective
        if isinstance(self.objective, MarginalizedTimeObjective):
            raise PybnfError(
                "output_noise_trajectory names %s, but the fit has a time_error" % column,
                "output_noise_trajectory draws %s's counts at the simulated times, and this fit "
                "scores each point with a timing error the draw would ignore. Remove "
                "output_noise_trajectory, or the time_error." % column)
        spec_for = getattr(self.objective, '_spec_for', None)
        family, sources = spec_for(column) if spec_for is not None else (None, {})
        if not isinstance(family, NegBinomial):
            what = type(family).__name__ if family is not None else type(self.objective).__name__
            raise PybnfError(
                "output_noise_trajectory names %s, whose noise model is %s, not neg_bin"
                % (column, what),
                "output_noise_trajectory draws negative binomial counts around %s's prediction "
                "from the noise model that scores it, and %s's is %s. Declare it with a neg_bin "
                "noise model, or list it under output_trajectory instead." % (column, column, what))
        source = sources.get('dispersion')
        if not isinstance(source, (ConstantSigma, FreeParameterSigma, FormulaSigma)):
            kind = type(source).__name__
            raise PybnfError(
                "output_noise_trajectory names %s, whose neg_bin dispersion is %s" % (column, kind),
                "output_noise_trajectory draws %s's counts with a dispersion fixed (fix_at), "
                "fitted (fit <parameter>) or a formula of parameters; %s's is a %s, which takes "
                "a value from the data and has none between or beyond the data rows."
                % (column, column, kind))
        legacy_default = (isinstance(self.objective, (NegBinLikelihood, NegBinLikelihood_Dynamic))
                          and column not in self.objective.overrides)
        cumulative = self.objective._is_cumulative(column) or (legacy_default and '_Cum' in column)
        return family, source, cumulative

    def generateBinomialNoise(self, timeseries, pset, rng, column):
        """Counts drawn around ``timeseries``, the predictions of ``column``, from its noise
        model (:meth:`_noise_draw_spec`) under the chain's own Generator ``rng``."""
        family, source, _ = _noise_spec(self, column)
        self.output = np.copy(timeseries)
        if isinstance(source, ConstantSigma):
            self.r = source.const
        elif isinstance(source, FreeParameterSigma):
            self.r = pset.get_param(source.name).value
        else:   # a formula of parameters, read from this chain's point
            self.r = source.value(SimpleNamespace(_pset_values={p.name: p.value for p in pset}),
                                  None, None, None, None, column)
        for i in range(len(timeseries)):
            # The distribution mean the noise model scores: a negative prediction is 0, and
            # a median-located prediction is inverted to its mean.
            mean = family._mean(float(timeseries[i]), float(self.r))
            self.output[i] = draw_count(rng, mean, float(self.r))

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
            for i in self._keys_of(l, self.output_run_current):
                if (i, idx) in self.output_run_recorded:
                    with open(runs_dir / f'traj_{i}_chain_{idx}.txt', 'a') as f:
                        np.savetxt(f, self.output_run_all[i][idx])
    def write_out_trajactorys_noise(self, idx):
        # Basically this IO on every iter is to expensice timewise
        runs_dir = Path(self.config.config['output_dir']) / 'Results' / 'A_MCMC' / 'Runs'
        for l in self.output_noise_columns:
            for i in self._keys_of(l, self.output_run_noise_current):
                if (i, idx) in self.output_run_noise_recorded:   # (#758, as above)
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
                for i in self._keys_of(l, buffers):
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
