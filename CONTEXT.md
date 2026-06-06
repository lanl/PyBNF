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
The space a free parameter is sampled, proposed, and stored in — linear or base-10 logarithmic. Shared by the parameter's prior and its proposal arithmetic; the posterior target is defined directly in this scale, with no change-of-variables. The two scales are **Linear** and **Log10**; the scale owns the `θ↔u` transform (`u = log10(θ)` for Log10) that maps a stored value to the space the family and proposals operate in.
_Avoid_: log space (informal), transform, parameterization

**Distribution Family**:
The shape of a prior independent of scale — Normal, Uniform, Laplace, … A free parameter's prior is one family combined with one scale, the family evaluated in that scale; adding a family yields its linear and log10 forms for free.
_Avoid_: distribution type, prior type, `*_var` keyword (a `*_var` keyword names one family×scale *pair*, not the family)

**Support**:
The region of nonzero prior density a parameter's initial sampling draws from — intrinsic to the distribution family (Uniform is finite, Normal and Laplace are unbounded), evaluated in the parameter's scale.
_Avoid_: range, domain, bounds (reserve "bounds" for the reflecting box)

**Reflecting Bounds**:
The box a proposal is folded back into during proposal arithmetic (the triangle-wave reflection). A property of the free parameter, **not** the prior: it exists only when the family has finite support *and* the parameter is declared bounded (the `b`/`u` flag), so an unbounded `uniform_var` has a finite support yet no reflecting bounds.
_Avoid_: box constraint, bounds (unqualified), limits

**No Prior** (`var`, `logvar`):
A free parameter given a single start value and no prior distribution — a start point for the start-point optimizers (Simplex, Powell, CMA-ES). It still carries a scale (`logvar` is Log10), contributes nothing to the log prior, and cannot be prior-sampled.
_Avoid_: null parameter, fixed parameter (it is varied during the fit, just not prior-sampled)

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

**Refine** (`refine`, `refine_method`):
An optional local-optimizer polish run after the main fit completes, locally improving its best-fit PSet; enabled by `refine = 1`. The optimizer is chosen by `refine_method` — one of the **Refiners** `sim` (Nelder–Mead Simplex, the default), `powell` (Powell), or `cmaes` (CMA-ES). It runs that optimizer on the *original* fit's configuration, so a refined fit of any fit_type needs the chosen refiner's full set of settings available — the one cross-fit_type configuration reach in PyBNF (a registry-keyed lookup off `refine_method`, ADR-0013/0015). Skipped when the fit_type already *is* the chosen refiner.
_Avoid_: polish, local search, post-optimization

**Refiner** / **Start-point optimizer** (registry `refiner=True`):
A derivative-free local optimizer that begins from a single start point and can serve as a `refine_method`: Simplex (`sim`), Powell (`powell`), CMA-ES (`cmaes`). These are exactly the fit types that take the no-prior `var`/`logvar` start point. All search in the parameter sampling space `u` (ADR-0015).
_Avoid_: local solver, polisher

**Box / global-start mode** (registry `start_from_box=True`):
A start-point optimizer's second start mode: instead of a single `var`/`logvar` point, it accepts a bounded-prior box (`uniform_var`/`loguniform_var`) and runs as a standalone *global* optimizer — starting at the box center (in `u`) and, for CMA-ES, seeding its covariance with the per-coordinate box widths so the first generation spans the whole box. Only CMA-ES (`cmaes`) does this today. The capability is a strict addition on top of being a **Refiner** — `start_from_box` is the flag ADR-0015 anticipated would split off `refiner` once a refiner learned to start from a box rather than a point (ADR-0017). In box mode `cmaes_sigma0` is read as a fraction of each box width; the bounded-prior-box-vs-point-start choice is made by the variable keywords, not a config switch.
_Avoid_: box mode (without "global-start"), bounded refine (it is not a refine), global refiner

**Noise Model**:
A probabilistic observation model mapping a deterministic prediction plus noise parameters to a distribution over the observed data; its negative log-likelihood is the objective value. PyBNF recognizes two **shapes**. A **Per-point Noise Model** has a log-likelihood that factors into a sum of independent per-observation terms (`chi_sq` = Gaussian, `neg_bin` = NegBinomial); it is defined by the three orthogonal axes distribution family × scale-the-noise-is-additive-on × location interpretation. A **Column-joint Noise Model** has per-observation contributions coupled across a whole data column, so the likelihood does not factor point-by-point (today only `kl`, the multinomial cross-entropy). Non-probabilistic objectives (`sos`, `sod`, `norm_sos`, `ave_norm_sos`) are losses, not noise models.
_Avoid_: error model, noise function, likelihood (reserve "likelihood" for the density itself)

**Column-joint Noise Model**:
A noise model whose likelihood does not factor into independent per-point terms because the points are coupled across a data column — e.g. by a compositional/closure constraint (`kl`'s multinomial; a future Dirichlet-multinomial) or by correlated residuals (a future correlated-error / Gaussian-process likelihood). Only `kl` exists today, kept as a plain `ColumnSummationObjective`; the column-joint abstraction is harvested when a second member justifies it (per ADR-0009's ≥2-user bar), not built speculatively.
_Avoid_: joint likelihood (too general), correlated noise (only one of its coupling mechanisms)

**Location Interpretation**:
Which summary of a noise model's distribution the deterministic prediction is taken to be — conditional mean, median, or mode. PyBNF makes this an explicit, overridable choice (PEtab v2 hardcodes median); it only matters when the noise is asymmetric on the prediction's scale.
_Avoid_: central tendency, link convention

**Noise Parameter**:
The dispersion or scale parameter of a noise model (a Gaussian's σ, a NegBinomial's r). It reaches the model from one of three sources: per observation from the data (`chi_sq`'s `_SD` column), as a free parameter estimated during the fit (the `_dynamic` objfuncs, via `sigma__FREE` / `r__FREE`), or as a fixed configuration constant (`neg_bin_r`). Whether the noise parameter is itself estimated — rather than fixed — is what decides if the likelihood normalizer is retained or dropped as a parameter-independent constant.
_Avoid_: error bar, sigma (when meaning the general concept), hyperparameter

**Additive-Noise Scale**:
The scale on which a noise model's noise is additive — linear (Gaussian: `obs ≈ pred + ε`) or logarithmic (lognormal: `log(obs) ≈ log(pred) + ε`). One of the three orthogonal axes defining a per-point noise model. Distinct from a free parameter's **Parameter Scale**: that names the space a parameter is *sampled* in (and owns a `θ↔u` transform for the prior and proposals); this names the space a *measurement's noise* lives on. The two are deliberately separate concepts and separate code, despite both being Linear/Log.
_Avoid_: noise scale (ambiguous with the Noise Parameter), error scale, link function

## Algorithms

PyBNF's fit types fall into three families — optimization algorithms, Bayesian
samplers, and checkers (the `checker` family, currently just `check`); the code,
configuration, and the registry `family` field treat them distinctly (`mh`,
`pt`, `am`, `dream`, `p_dream` form the Bayesian group).

**Optimization Algorithm**:
A fit type that searches for the single best-fitting PSet. Codes: `de` (Differential Evolution, the default), `ade` (Asynchronous DE), `pso` (Particle Swarm), `ss` (Scatter Search), `sim` (Nelder–Mead Simplex), `powell` (Powell), `cmaes` (CMA-ES). The last three are the start-point **Refiners** (also usable as `refine_method`).
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
