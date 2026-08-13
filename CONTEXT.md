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
The space a free parameter is sampled, proposed, and stored in — linear or base-10 logarithmic. Shared by the parameter's prior and its proposal arithmetic; the posterior target is defined directly in this scale, with no change-of-variables. The two scales are **Linear** and **Log10**; the scale owns the `θ↔u` transform (`u = log10(θ)` for Log10) that maps a stored value to the space the family and proposals operate in. `FreeParameter` exposes this transform publicly as `to_sampling_space(θ)` / `from_sampling_space(u)`, so the algorithm layer asks the parameter for it rather than inlining `log10`/`10**` (ADR-0029, #412); the *guarded* inverse for a user-supplied `logvar`/`lognormal` start value is the separate `exp10` helper, which raises a configuration hint on overflow.
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

**Global Function**:
A derived scalar expression declared in a BNGL `begin functions` block, named with a trailing `()` (`y()`, `kcat_eff()`) — any legal BNGL name followed by `()`. Evaluated at each simulation output and, with `print_functions=>1` (which PyBNF always sets), written as an output column alongside observables; so a fitted `.exp` column matches an Observable *or* a Global Function. Nothing *declares* a global function to be a **measurement model**: it *becomes* one implicitly when its name appears in both a simulation-output header (`.gdat`/scan) and an `.exp` header — the usual home of scaling, offset, ratios, sums of observables. References parameters, observables, and other global functions by name. The PySB-`Expression` analog, and the model entity a PEtab `observableFormula` names by its bare BNGL name.
_Avoid_: function (ambiguous with Objective Function), measurement function, derived variable, local function (an in-rate-law `f(x)` form, not a global function)

**Measurement Model**:
The transform from a simulation's output trajectory (plus the current parameter values) to the quantity compared against data — a PEtab `observableFormula`. PyBNF evaluates it as a first-class **post-simulation observation layer** (`pybnf/measurement`, `MeasurementModel`/`MeasurementLayer`, ADR-0036): each measurement model's `observableFormula` is compiled (via `petab.v2.math`) to a vectorized numpy callable over the output columns + the PSet, materialized into the simulated `Data` *before* the objective scores it — **never** by editing the model file. A bare model-output name (an Observable or Global Function the backend already prints) is the trivial *identity* measurement model and needs no layer; an arbitrary expression over species/observables/parameters is the general case (the home of scaling, offset, ratios, sums) — the role a Global Function plays in BNGL, now reified so SBML (which has no functions block) gets it too. Backend-agnostic (RoadRunner, bngsim, the BNG stack) and language-agnostic (BNGL, SBML); the M2 peer to **Prior** and **Noise Model**, PEtab-defaulted not PEtab-bound (ADR-0004/0036).
_Avoid_: observable (that is a single model output column), observation function, measurement function, observable transformation

**Action**:
A simulation directive attached to a model telling PyBNF what to simulate; the two kinds are a Time Course and a Parameter Scan.
_Avoid_: command, task, run

**Time Course**:
An action that simulates the model over time, producing a time series to compare against data.
_Avoid_: simulation (too general), trajectory (that is the fit's best-fit record)

**Parameter Scan**:
An action that sweeps one model parameter across a range, producing output as a function of that parameter.
_Avoid_: sweep, bare "scan"

**Steady State**:
The equilibrium a model relaxes to — the `t → ∞` limit of a Time Course, reached by integrating with an early stop on `‖dx/dt‖` (BNGL `steady_state=>1`) or by an algebraic solve, and bounded by a **max-time bound** (`t_end:`, default `1e6`) that is *not* a readout time. It appears in three places, all the same idea measured differently: an *unmeasured* equilibration phase before a measured one (**Pre-equilibration**, ADR-0052); the per-dose readout of a **Parameter Scan** (the ADR-0046 default); and — the plain case — a *measured* observation with no swept axis, written `time = inf` in the `.exp` and scored against the run's final row (ADR-0086). Deterministic (`ode`) only: a stochastic run has a stationary *distribution*, not a fixed point. PEtab spells all three `inf` (or `-inf` for a pre-equilibration period).
_Avoid_: equilibrium simulation, long-time limit, "t=inf experiment" (that is the data's spelling, not the concept), stationary distribution (a stochastic notion, deliberately out of scope)

**Suffix**:
The label that pairs a model action's simulated output with the experimental data file it is compared to.
_Avoid_: tag, key

**Mutant** (`mutant`):
A model variant declared in the config as a set of parameter overrides applied to a base model, with its own experimental data bound via the Suffix (data file `<base_suffix><name>.exp`). The internal objects are a `Mutation` (one `var op val` override, `op` ∈ `= * / + -`) and a named `MutationSet`. An absolute set (`=`) replaces the base value; the relative ops (`* / + -`) scale/shift it — so a Mutant of a *fit* parameter overrides the value the search is currently trying. Exports to a PEtab **Condition**.
_Avoid_: variant, perturbation, knockout (only one operator-and-target shape)

**Surrogate-base parameter** (`<p>__REF`):
The PEtab-export rename of a *fit* parameter `p` that some Mutant also overrides. PEtab forbids one id from being both an estimated parameter and a condition target, so the estimated quantity is moved to `p__REF` (the only thing in the parameter table) while the model name `p` becomes a pure condition target set in every experiment — `p = p__REF` at baseline, `p = p__REF * 2` where a Mutant scales it. The `__REF` marker mirrors PyBNF's own `__FREE` is-fit marker (a double-underscore suffix), so it cannot clash with a user-defined model name (ADR-0027).
_Avoid_: shadow parameter, alias, proxy

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
An optional local-optimizer polish run after the main fit completes, locally improving its best-fit PSet; enabled by `refine = 1`. The optimizer is chosen by `refine_method` — any of the **Refiners**: `sim` (Nelder–Mead Simplex, the default), `powell` (Powell), `cmaes` (CMA-ES), or the gradient optimizers `trf` / `lbfgs` / `gntr`. It runs that optimizer on the *original* fit's configuration, so a refined fit of any fit_type needs the chosen refiner's full set of settings available — the one cross-fit_type configuration reach in PyBNF (a registry-keyed lookup off `refine_method`, ADR-0013/0015). Skipped when the fit_type already *is* the chosen refiner. `refine = 1` is a request for a **method** (search, then polish), not a flag, which is why a **Wall-Time Budget** reserves the refine a share of itself rather than letting the search spend it all (ADR-0107).
_Avoid_: polish, local search, post-optimization

**Refiner** / **Start-point optimizer** (registry `refiner=True`):
A local optimizer that begins from a single start point and can serve as a `refine_method`: the derivative-free Simplex (`sim`), Powell (`powell`), CMA-ES (`cmaes`), and the gradient optimizers TRF (`trf`), L-BFGS-B (`lbfgs`), Gauss–Newton trust-region (`gntr`). These are exactly the fit types that take the no-prior `var`/`logvar` start point. All search in the parameter sampling space `u` (ADR-0015).
_Avoid_: local solver, polisher

**Box / global-start mode** (registry `start_from_box=True`):
A start-point optimizer's second start mode: instead of a single `var`/`logvar` point, it accepts a bounded-prior box (`uniform_var`/`loguniform_var`) and runs as a standalone *global* optimizer — starting at the box center (in `u`) and, for CMA-ES, seeding its covariance with the per-coordinate box widths so the first generation spans the whole box. Only CMA-ES (`cmaes`) does this today. The capability is a strict addition on top of being a **Refiner** — `start_from_box` is the flag ADR-0015 anticipated would split off `refiner` once a refiner learned to start from a box rather than a point (ADR-0017). In box mode `cmaes_sigma0` is read as a fraction of each box width; the bounded-prior-box-vs-point-start choice is made by the variable keywords, not a config switch.
_Avoid_: box mode (without "global-start"), bounded refine (it is not a refine), global refiner

**Noise Model**:
A probabilistic observation model mapping a deterministic prediction plus noise parameters to a distribution over the observed data; its negative log-likelihood is the objective value. PyBNF recognizes two **shapes**. A **Per-point Noise Model** has a log-likelihood that factors into a sum of independent per-observation terms (`chi_sq` = Gaussian, `laplace` = Laplace, `neg_bin` = NegBinomial); it is defined by the three orthogonal axes distribution family × scale-the-noise-is-additive-on × location interpretation, paired with a **Noise Parameter Source**. A **Column-joint Noise Model** has per-observation contributions coupled across a whole data column, so the likelihood does not factor point-by-point (today only `kl`, the multinomial cross-entropy). Non-probabilistic objectives (`sos`, `sod`, `norm_sos`, `ave_norm_sos`) are losses, not noise models. A per-point noise model is selected **per observable** (per data column) — the native `noise_model` config key, or a PEtab `observables.tsv` row — defaulting to the single global `objfunc` applied to every column (ADR-0021).
_Avoid_: error model, noise function, likelihood (reserve "likelihood" for the density itself)

**Column-joint Noise Model**:
A noise model whose likelihood does not factor into independent per-point terms because the points are coupled across a data column — e.g. by a compositional/closure constraint (`kl`'s multinomial; a future Dirichlet-multinomial) or by correlated residuals (a future correlated-error / Gaussian-process likelihood). Only `kl` exists today, kept as a plain `ColumnSummationObjective`; the column-joint abstraction is harvested when a second member justifies it (per ADR-0009's ≥2-user bar), not built speculatively.
_Avoid_: joint likelihood (too general), correlated noise (only one of its coupling mechanisms)

**Location Interpretation**:
Which summary of a noise model's distribution the deterministic prediction is taken to be — conditional mean, median, or mode. PyBNF makes this an explicit, overridable choice (PEtab v2 hardcodes median); it only matters when the noise is asymmetric on the prediction's scale.
_Avoid_: central tendency, link convention

**Observation Domain**:
The set of measured values a **Noise Model** can assign a probability to — the whole real line for the location-scale families, the non-negative counts for `neg_bin`. A measurement outside it is not a **scored point**: it is dropped from the pointwise likelihood exactly as a NaN observation is, so it is out of `n` for AIC/BIC and off the LOO/WAIC observation axis (ADR-0090). Distinct from the *cost* path, where such a point contributes nothing (a zero, not an exclusion) — the asymmetry is deliberate, because on a self-normalizing PMF a zero cost would otherwise read as a log-density of 0, i.e. probability 1. Distinct also from **Support**, which is the free-parameter/prior-side concept.
_Avoid_: valid range, admissible values, support (reserved for the prior side)

**Noise Parameter**:
The dispersion or scale parameter of a noise model (a Gaussian's σ, a Laplace's b, a NegBinomial's r). Whether the noise parameter is itself estimated — rather than fixed — is what decides if the likelihood normalizer is retained or dropped as a parameter-independent constant.
_Avoid_: error bar, sigma (when meaning the general concept), hyperparameter

**Cumulative Prediction Transform**:
A per-observable, family-independent **prediction** transform: a column declared `cumulative` has its simulated prediction differenced row-to-row (cumulative count → per-interval *incident* count) before scoring, with row 0 kept raw. It is *how the prediction is formed from the simulation*, orthogonal to the noise family — so it composes with any per-point objective (`chi_sq`, `laplace`, `sos`, `neg_bin`, …), not just NegBinomial. Declared as a flag on the per-observable `noise_model` line but stored and applied independently of the noise spec, on the shared `_prediction` seam (ADR-0051). The legacy `_Cum` column-name substring is a `neg_bin_dynamic`-only compatibility bridge for the same transform, deliberately *not* widened to other families (that would silently change their scores; ADR-0021). No PEtab v2 representation — export refuses it rather than dropping it.
_Avoid_: incidence model, differencing noise (it is not noise), `_Cum` (the legacy trigger, not the concept)

**Noise Parameter Source** (σ-source):
Where a noise model reads its **Noise Parameter** for one observation — one of three first-class kinds (`SigmaSource`, ADR-0021): per observation from a data column (`read_exp_file`, conventionally the `_SD` column), as a free parameter estimated during the fit (`fit`, e.g. the `_dynamic` objfuncs' `sigma__FREE` / `r__FREE` defaults, or a per-observable `__FREE` name), or as a fixed configuration constant (`fix_at`, e.g. `neg_bin_r`). The source's kind is load-bearing: a *fixed* source contributes only the family's data-fit term, an *estimated* source contributes the full negative log-likelihood including the normalizer — the one rule that makes each legacy objfunc its exact (family × σ-source) default.
_Avoid_: sigma source (lowercase ambiguity), error column, noise channel

**Additive-Noise Scale**:
The scale on which a noise model's noise is additive — `LINEAR` (Gaussian: `obs ≈ pred + ε`), `LOG10` (log10 lognormal: `log10(obs) ≈ log10(pred) + ε`), or `LN` (natural-log lognormal). One of the three orthogonal axes defining a per-point noise model. **One log base across PyBNF: a bare "log" always means log10** (matching `logvar`/`lognormal_var` and the proposal arithmetic); natural log is never implied, only the explicit `LN` (ADR-0022) — so native `lognormal` is log10, while the explicit `lnnormal` token is `Gaussian(LN)` (ADR-0084), and there is no ambiguous bare `LOG`. Distinct from a free parameter's **Parameter Scale**: that names the space a parameter is *sampled* in (and owns a `θ↔u` transform for the prior and proposals); this names the space a *measurement's noise* lives on. The two are deliberately separate concepts and separate code, but share the one-log10-base convention.
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

**Wall-Time Budget** (`wall_time_fit`):
The total wall-clock seconds a **Fit** may run — the run-level peer of the per-unit-of-work limits `wall_time_sim` (one simulation) and `wall_time_gen` (one network generation). Distinct from `max_iterations`, which counts **Iterations** and is not convertible to wall time without knowing per-iteration cost. When it expires the run loop stops launching work, abandons what is in flight, and **finalizes**: the *same* end-of-fit path a converged run takes, against the best point so far, so a budgeted result is scoreable exactly like a completed one. Only the stop reason differs, and it is written to `Results/stop_reason.txt` — beside the results, never inside them. One budget bounds the whole run — each **Bootstrap** replicate is new work and does not begin once it is spent — and its clock starts at process start, so configuration loading and network generation are inside it (ADR-0093, #529). It is *partitioned*, not spent first-come-first-served: `wall_time_refine_frac` (default 0.1) holds a tail back from the search as the **Refine Reserve**, because a wall-clock-budgeted search would otherwise leave the requested **Refine** nothing to run on (ADR-0107, #564). Implemented as a `FitBudget` object (`pybnf/budget.py`); unbounded is the *absence* of one, not an infinite limit.
_Avoid_: timeout (that is `wall_time_sim`'s per-simulation limit), deadline (informal), time limit, iteration budget

**Refine Reserve** (`wall_time_refine_frac`):
The share of the **Wall-Time Budget** the search phase may not spend, kept for the **Refine**. `refine = 1` requests a *method* — search globally, then polish locally — and a wall-clock-budgeted search has no reason to stop early, so without a reserve the polish essentially never ran (#564: 15 of 15 benchmark runs configured `cmaes` + `refine = gntr` executed plain `cmaes`). A fraction in [0, 1), default 0.1; 0 restores the pre-#564 split. A floor under the refine rather than a cap on it — a search that converges early hands its leftovers on — and taken only when a refine will actually be attempted. Held on the `FitBudget` as `reserve`, which makes `expired()`/`remaining()` the *current phase's* deadline; the `spend_reserve(budget)` context manager releases it for the refine and restores it afterward (so every bootstrap replicate gets the same split). ADR-0107.
_Avoid_: refine budget (it is not a separate budget), `wall_time_refine` (no such key — the reserve is taken *from* `wall_time_fit`, never added to it)

**Method Chain** (`Results/method_chain.json`):
The record of which methods a run actually executed, written by every run: `requested_methods` (what the conf asked for, e.g. `["cmaes", "gntr"]`), `executed_methods` (what ran), and one entry per **Phase** with its status (`completed` / `wall_time_expired` / `skipped`), stop reason, elapsed seconds, completed simulations, and best objective. A shorter `executed_methods` is the machine-readable form of a silent downgrade. Beside the results, never inside them — the same placement rule `stop_reason.txt` follows. Filled in by `pybnf.pybnf` (the orchestrator, the only place that sees all the phases), not by `Algorithm.run`. ADR-0107, #564.
_Avoid_: run metadata, provenance file, manifest

**Phase**:
One stage of a run's method chain: the **Fit**, the **Refine**, a **Bootstrap** replicate. Distinct from the *method* that fills it (`cmaes` is a fit in one run and a refine in another) and from an **Iteration** (a phase contains many). The **Wall-Time Budget** bounds the run across all of them.
_Avoid_: stage, step, pass

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
The R-hat (rank-normalized split potential scale reduction factor) and bulk/tail effective sample size (ESS) statistics that quantify MCMC convergence and sampling efficiency, in the Vehtari et al. (2021) / Stan / ArviZ conventions. The pure math lives in the top-level `pybnf/diagnostics.py` — a peer of `objective.py`, importable by the benchmark harness without reaching into `samplers/` — while the instance-coupled reporting/stopping glue (`report_convergence_diagnostics`, `check_convergence`, `_write_diagnostics`) stays on `BayesianAlgorithm`, which delegates the math (ADR-0009, M2.2). The PSet→array bridge (`_param_vec`) was later hoisted one level onto `Algorithm` — the single PSet→sampling-space-`u` transform shared by the samplers and the start-point optimizers (where it is also exposed as `_u_from_pset`), once those optimizers grew the identical code (the ≥2-user event). Its inverse peer, the u-vector→PSet bridge (`_pset_from_u`, with a `reflect` flag so DREAM rejects an out-of-box proposal while the optimizers fold it in), was later hoisted alongside it, and both now ask each `FreeParameter` for its `to_sampling_space`/`from_sampling_space` transform rather than inlining `log10`/`10**` (ADR-0029, #412).
_Avoid_: convergence test, R-hat (as a synonym for the whole pair)
