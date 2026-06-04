# PyBNF

PyBNF fits the free parameters of rule-based and SBML models to experimental
data by minimizing an objective function with a chosen optimization or
Bayesian-sampling algorithm. This glossary fixes the vocabulary used across the
configuration file, the source, and the docs. When several words exist for one
concept, the preferred term is the heading and the rejected ones are listed
under _Avoid_.

## Fitting core

**Fit**:
One complete run of PyBNF: load a configuration, repeatedly simulate the model under candidate parameter values, and search for those that best match the experimental data.
_Avoid_: optimization run, job, session

**Fit Type** (`fit_type`):
The search algorithm a fit uses, selected by a short code in the configuration (`de`, `pso`, `am`, `dream`, …).
_Avoid_: method, mode, solver, optimizer

**Free Parameter**:
A model quantity PyBNF is allowed to vary during a fit, declared in the configuration with a `*_var` keyword and a prior range or distribution.
_Avoid_: variable, fitted parameter, bare "parameter"

**Prior**:
The probability distribution assigned to a free parameter — used both as the Bayesian prior by samplers and as the initial-sampling distribution by optimizers. Defined by an orthogonal distribution family × scale, and evaluated in the parameter's own scale.
_Avoid_: initial distribution, proposal distribution (that is the sampler's step kernel), parameter range

**Parameter Scale**:
The space a free parameter is sampled, proposed, and stored in — linear or base-10 logarithmic. Shared by the parameter's prior and its proposal arithmetic; the posterior target is defined directly in this scale, with no change-of-variables.
_Avoid_: log space (informal), transform, parameterization

**PSet** (Parameter Set):
One concrete assignment of values to every free parameter — a single point in parameter space that can be simulated and scored.
_Avoid_: parameter vector, individual, particle, sample, candidate (these are algorithm-specific views of a PSet)

**Objective Function** (`objfunc`):
The scalar measure of disagreement between a PSet's simulated output and the experimental data; PyBNF searches for the PSet that minimizes it (e.g. `chi_sq`, `sos`, `neg_bin`).
_Avoid_: cost function, loss, fitness, error function

**Objective Value**:
The number the objective function returns for a given PSet; lower is a better fit.
_Avoid_: score, cost, loss, error (the code uses "score"; prefer "objective value" in prose)

**Trajectory**:
The running record of the best PSet and its objective value as a fit progresses.
_Avoid_: history, log, progress curve

**Configuration** (`.conf`):
The keyword file that defines a fit — models, data, free parameters, fit type, and algorithm settings.
_Avoid_: config (informal), settings file, input file

## Models & simulation

**Model**:
The mechanistic model whose free parameters are being fit; supplied as BNGL, SBML, or Antimony.
_Avoid_: system, network (a network is one *product* of a model — see Network generation)

**BNGL**:
The BioNetGen Language — PyBNF's native rule-based model format (`.bngl`).
_Avoid_: BioNetGen file, rules file

**Rule-based model**:
A model defined by reaction rules that BioNetGen expands into an explicit reaction network, rather than by enumerating every reaction by hand.
_Avoid_: agent-based model, rule model

**Network generation**:
BioNetGen's step of expanding a rule-based model into its full set of species and reactions, performed once before network-based simulation.
_Avoid_: compilation, build

**Observable**:
A model output (e.g. a molecule count or concentration) recorded during simulation and matched by name to a column of experimental data.
_Avoid_: output, readout, variable

**Action**:
A simulation directive attached to a model telling PyBNF what to simulate; the two kinds are a Time Course and a Parameter Scan.
_Avoid_: command, task, run

**Time Course**:
An action that simulates the model over time, producing a time series to compare against data.
_Avoid_: simulation (too general), trajectory (that is the fit's best-fit record)

**Parameter Scan**:
An action that sweeps one model parameter across a range, producing output as a function of that parameter.
_Avoid_: sweep, bare "scan"

**Suffix**:
The label that pairs a model action's simulated output with the experimental data file it is compared to.
_Avoid_: tag, key

**Simulation Method** (`method`):
How an action is simulated: `ode` (deterministic, CVODE), `ssa` (stochastic Gillespie), `pla` (partitioned leaping), or `nf` (network-free, via NFsim).
_Avoid_: solver, integrator (the integrator is one detail of the `ode` method), engine

**Backend** (`bngl_backend`, `sbml_backend`):
The software PyBNF drives to actually run a simulation (e.g. a BioNetGen subprocess or bngsim for BNGL; libRoadRunner for SBML).
_Avoid_: engine, driver

## Data, objectives & uncertainty

**Experimental Data** (`.exp`):
The measured, whitespace-delimited time series (an independent-variable column plus observable columns, with optional `_SD` columns) that a fit is scored against.
_Avoid_: dataset, observations file, ground truth

**Constraint** (`.prop`):
A qualitative or quantitative condition on the simulation that contributes a penalty to the objective, rather than a point-by-point data comparison.
_Avoid_: rule, assertion, restraint

**Replicate**:
One repeat stochastic simulation of a single PSet; the `smoothing` setting is the number of replicates averaged to reduce noise.
_Avoid_: repeat, trial, sample

**Bootstrap**:
Refitting on resampled experimental data to estimate the uncertainty in the fitted parameters.
_Avoid_: resampling run, jackknife

**Noise Model**:
A probabilistic observation model mapping a deterministic prediction plus noise parameters to a distribution over the observed data; its negative log-likelihood is the objective value. Defined by three orthogonal axes — distribution family, the scale the noise is additive on, and the location interpretation. (Non-probabilistic objectives such as `sos` and `sod` are losses, not noise models.)
_Avoid_: error model, noise function, likelihood (reserve "likelihood" for the density itself)

**Location Interpretation**:
Which summary of a noise model's distribution the deterministic prediction is taken to be — conditional mean, median, or mode. PyBNF makes this an explicit, overridable choice (PEtab v2 hardcodes median); it only matters when the noise is asymmetric on the prediction's scale.
_Avoid_: central tendency, link convention

## Algorithms

PyBNF's fit types fall into three families — optimization algorithms, Bayesian
samplers, and checkers (the `checker` family, currently just `check`); the code,
configuration, and the registry `family` field treat them distinctly (`mh`,
`pt`, `am`, `dream`, `p_dream` form the Bayesian group).

**Optimization Algorithm**:
A fit type that searches for the single best-fitting PSet. Codes: `de` (Differential Evolution, the default), `ade` (Asynchronous DE), `pso` (Particle Swarm), `ss` (Scatter Search), `sim` (Nelder–Mead Simplex).
_Avoid_: optimizer, minimizer, solver

**Bayesian Sampler**:
A fit type that samples the posterior distribution of the free parameters instead of returning one best PSet. Codes: `am` (Adaptive MCMC), `dream` (DREAM(ZS)), `p_dream` (P-DREAM), `pt` (Parallel Tempering); `mh` (Metropolis–Hastings) is deprecated. `sa` (Simulated Annealing) is a deprecated *optimizer*, not a sampler (registry family `optimizer`); M2.2 extracted it to its own class in `optimizers/simulated_annealing.py`, where it minimizes the raw objective (ADR-0008).
_Avoid_: MCMC run, posterior fit

**DREAM(ZS)** (`dream`):
PyBNF's DiffeRential Evolution Adaptive Metropolis sampler (Vrugt 2016), drawing proposal donors from a growing ZS archive of past chain states.
_Avoid_: bare "DREAM", DE-MC

**P-DREAM** (`p_dream`):
Preconditioned DREAM — DREAM(ZS) with proposals computed in a covariance-whitened parameter space, for better sampling of correlated posteriors.
_Avoid_: parallel DREAM (the "P" is *preconditioned*, not parallel)

**Snooker Update**:
One of DREAM's two proposal mechanisms (ter Braak & Vrugt 2008), projecting archive points onto the line through the current chain state; `snooker_prob` sets how often it is used versus the parallel-direction proposal.
_Avoid_: snooker move, snooker step

**Iteration**:
One round of an algorithm's main loop and the unit in which a fit's budget is counted (`max_iterations`). Population-based algorithms also call a round a "generation".
_Avoid_: step, epoch

**Model Check** (`fit_type = check`):
A first-class checking method — statistical model checking: evaluates the objective value and constraint satisfaction for given parameters without searching parameter space. Registers in the `checker` family, a peer of optimization algorithms and Bayesian samplers (not a utility afterthought).
_Avoid_: dry run, validation, utility run

## Architecture

**Registry**:
The single source of truth mapping a `fit_type` or `objfunc` code to the class that implements it, together with its family, defaults, and deprecation status. Methods self-register via a decorator, replacing the hand-maintained `if/elif` dispatch.
_Avoid_: dispatcher, factory map, lookup table

**Sampler Toolkit** (prospective — not yet built):
A possible future library of optional, composable sampler building blocks (a Metropolis kernel, proposals, tempering, cooling). **It does not exist today.** On inspection the only shared candidates were ~15 lines of textbook stepping used by `mh`/`pt` and the deprecated `sa`, with no growth path — `am` and `dream` carry their own proposals and kernels — so the ≥2-user bar was not met (ADR-0009). Stepping logic lives inside each sampler; harvest a toolkit only when a future sampler genuinely wants shared stepping, to its real shape then. (Chain *diagnostics*, which **are** shared, live in `pybnf/diagnostics.py` — see Convergence Diagnostics — not here.)
_Avoid_: framework, base class, mixins

**Metropolis Kernel**:
The propose → accept/reject step at the heart of a Metropolis sampler. PyBNF does **not** factor this into one shared implementation: `mh`/`pt` use a fixed-magnitude Gaussian random walk with a β-tempered accept; `am` uses an adaptive multivariate-normal proposal; `dream` uses DE-archive donors with a snooker Hastings correction. Each sampler owns its kernel by design (ADR-0009). The deprecated optimizer `sa` uses the same fixed-magnitude Gaussian-step + Metropolis-accept shape over the *raw objective* (not a posterior), in its own `optimizers/simulated_annealing.py` (ADR-0008).
_Avoid_: MCMC step, sampler core

**Convergence Diagnostics**:
The R-hat (rank-normalized split potential scale reduction factor) and bulk/tail effective sample size (ESS) statistics that quantify MCMC convergence and sampling efficiency, in the Vehtari et al. (2021) / Stan / ArviZ conventions. The pure math lives in the top-level `pybnf/diagnostics.py` — a peer of `objective.py`, importable by the benchmark harness without reaching into `samplers/` — while the instance-coupled reporting/stopping glue (`report_convergence_diagnostics`, `check_convergence`, `_write_diagnostics`) and the PSet→array bridge (`_param_vec`) stay on `BayesianAlgorithm`, which delegates the math (ADR-0009, M2.2).
_Avoid_: convergence test, R-hat (as a synonym for the whole pair)
