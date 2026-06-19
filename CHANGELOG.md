# Changelog

All notable changes to PyBNF are documented below. This project adheres to
[Keep a Changelog](https://keepachangelog.com) conventions.

## [Unreleased]

### Added
- **New-era `observable:` column-header override** (`edition >= 2`, ADR-0028) — `observable: <entity>, column: <header>` remaps a data-file column **header** to a model observable/function **name** when the two differ. By default a `.exp` column header *is* the model observable name and the objective matches experimental columns to simulation columns by that name, so this line is needed only when the measured column is named something else (common with real data) — without it a differently-named data column has no matching simulation column and the fit raises (`_check_columns`). The override renames the `<header>` column to `<entity>` (and its `<header>_SD` per-point noise companion, ADR-0021, to `<entity>_SD`) in every experimental data file, so the by-name match succeeds; the rename rewires both column maps in place and leaves the data array untouched. Scope is **global** (a top-level line, not per-experiment): a data file that doesn't contain `<header>` is left unchanged, while a `<header>` present in *no* data file is treated as a typo and raises (listing the columns actually present). The independent-variable column cannot be remapped, and a remap colliding with an existing column raises. Requires `edition >= 2`; legacy and same-named confs are byte-unchanged, and the `('observable', …)` structural key passes through config building so golden configs stay byte-identical. With this the full new-era problem surface (`job_type` + `model:` + `condition:` + `experiment:`/`data:` + `observable:` + parameters + objective) is complete. (#423, ADR-0028)
- **New-era `experiment:` / `data:` syntax** (`edition >= 2`, ADR-0028) — a named simulation bound to its measurement files, a PyBNF Experiment = a PEtab v2 Experiment: `experiment: <name>, data: <f1.exp>[, <f2.exp>…]` plus optional `condition:`, `model:`, `type:`, and `method:` fields in any order. The experiment **name** replaces the legacy filename→suffix convention as the simulation's identity, so the data↔simulation link is *stated*, not inferred from filenames, and a data file can be named anything. Two long-standing warts go away: **multiple `data:` files are replicates** (their rows stack into one experiment — not averaged, which is smoothing — so the objective sees every replicate measurement, impossible on the legacy surface without pre-averaging), and **the simulation outputs at exactly the data's points** (the data's independent-variable column supplies the output grid; PyBNF synthesizes the `simulate` action via BNGL `sample_times` / RoadRunner `simulate(times=…)`, so the BNGL `begin actions` block is no longer needed for fitting and the scoring grid always lines up with the measurements). `condition:` applies a named condition (omitted ⇒ wildtype); `type:` is inferred from the data (`time` column ⇒ time course) and stated only when inference can't decide. A different front-end, the same internal objects — a new-era `experiment:` produces the same `self.models` suffixes / `exp_data` / `mapping` a legacy `model = X : e.exp` + `time_course` does. Currently time-course experiments only; a parameter scan via this surface is deferred (its simulation endpoint time has no home in the grammar yet) and raises a clear error. Requires `edition >= 2`; legacy confs are byte-unchanged. (#423, ADR-0028)
- **Negative-binomial median centering** (`location = median` / `noise_location = median` on a `neg_bin` noise model; #419, ADR-0031) — completes ADR-0031's "every means every": the prediction can now be interpreted as the count distribution's median (its 0.5-quantile), not only its mean. Because the negative binomial is parameterized by its mean and its median has no closed form, the mean placing the continuous median at the prediction is recovered by a per-point bounded CDF inversion (`scipy.special.betainc` + `scipy.optimize.brentq`); the *continuous* 0.5-quantile is used (not the discrete `ppf` step) so the objective stays smooth for the optimizers. The legacy `neg_bin` / `neg_bin_dynamic` objfuncs remain frozen-mean and byte-identical. (#419, ADR-0031)
- **Modern objective surface** (`edition >= 2`, ADR-0031) — three keys replace the legacy `objfunc`, which is now an error under a modern edition: **`objective`** (the named per-point catch-all — `sos` / `chi_sq` / `laplace` / `neg_bin` / … plus the bare `score` passthrough), a **whole-fit `noise_model` line** (`noise_model = <family>, …`, no observable — the recommended per-point form), and **`profile_objective`** (column-joint shape objectives: `kl`, and a new working **`wasserstein`** 1-Wasserstein / earth-mover distance). Under a modern edition exactly one must be named — there is no implicit default. The legacy least-squares objfuncs fold into the per-point noise-model engine: each token desugars to the equivalent `noise_model` (`objective = sos` ≡ `gaussian, sigma = fix_at 1`; `sod` ≡ `laplace, sigma = fix_at 1`; …), restoring the statistically-proper ½ that legacy `sos` / `norm_sos` / `ave_norm_sos` drop (argmin-identical — the located fit is unchanged). Two new `noise_model` σ-sources enable the fold: **`relative [<cv>]`** (constant-CV, σ ∝ the measurement — the honest `norm_sos`) and **`column_mean`** (σ = the observable's column mean — the honest `ave_norm_sos`). The legacy edition and the `objfunc` key remain byte-identical to before. (#424, ADR-0031)
- **New-era `condition:` syntax** (`edition >= 2`, ADR-0028) — a named set of parameter perturbations on a base model, `condition: <name>, perturbations: <var op val>, …` (`=` sets absolutely; `* / + -` apply relative to the nominal value), with an optional `model:` ref (omittable when the job declares one model, required under several). A condition is a PyBNF Mutant = a PEtab v2 Condition; it is the *perturbation half* of the legacy `mutant` line, carrying **no** data binding (data is introduced separately, via an experiment). Internally it reuses the existing `Mutation`/`MutationSet`/`add_mutant` machinery — a `condition:` produces the identical MutationSet a legacy `mutant … : none` would, minus the suffix-matched data coupling. Duplicate condition names are rejected at parse time. (#423, ADR-0028)
- **New-era `model:` declaration syntax** (`edition >= 2`, ADR-0028) — under a modern edition a model is declared with the colon form `model: egfr.bngl` (or a comma list `model: egfr.bngl, erbb2.bngl`), which carries **no** data binding: data is introduced separately (through an experiment's measurements, landing in a later chunk), retiring the legacy coupling of data onto the model line. `model:` lines are repeatable and accumulate; the `modelId` is the filename stem, unique across all declarations. Internally each declared file folds to exactly what a legacy `model = file : none` line produces, so the model loader and everything downstream are untouched (a different front-end, the same internal objects). The legacy `model = … : …` form is unchanged and still works at every edition. (#423, ADR-0028)
- **`fit_type` → `job_type` rename** (`edition >= 2`, ADR-0028) — under a modern edition the run-selector key is named `job_type`, because `fit_type` was a misnomer: the key chooses across point-estimate *optimizers* (`de` / `ade` / `pso` / `ss` / `sim` / `powell` / `cmaes` / `sa`), Bayesian *samplers* (`am` / `dream` / `p_dream` / `pt` / `mh`), and the model *checker* (`check`) — not just fitting. The value still names the specific procedure; the key now honestly names the kind of job. This is a surface-only rename: the config layer normalizes the chosen key into the internal slot, so the registry and every downstream read are untouched. Under a modern edition the legacy `fit_type` key is rejected and, like the modern objective surface, there is no implicit default — the run must be named. The legacy edition keeps `fit_type` (defaulting to `de`) byte-identical to before. (#423, ADR-0028)
- **`edition` config key** — an optional integer that opts a `.conf` into a frozen set of modernized PyBNF conventions. Editions are *select-and-freeze* (in the Rust-edition sense): a config written for `edition = 2` is interpreted under edition-2 conventions forever, even as later releases change other defaults under higher editions, so upgrading PyBNF never silently reinterprets an existing config. Omitting the key selects legacy behavior (the implicit edition 1), byte-identical to PyBNF's historical defaults; the newest syntax must opt in with an explicit `edition`. An unsupported (future) edition is rejected with the PyBNF version it requires. The first behavior it gates: under a modern edition (`edition >= 2`) the universal default prediction centering is the **median** (consistent with PEtab v2) — byte-identical for the location-scale noise models (`chi_sq` / `lognormal` / `laplace`, already median), and now also realized for `neg_bin` (legacy default mean), whose median has no closed form and is solved per-point (#419); a `neg_bin` fit with no explicit location resolves to the median and **warns** (set `noise_location = mean` to keep the legacy mean, or `= median` to silence). (#424, ADR-0031)
- **Truncated priors** — the unbounded-support prior families (`normal_var`, `laplace_var`, and their `log*` forms) now accept finite reflecting bounds, turning them into truncated priors. The prior is renormalized over the box and sampled inside it (truncated inverse-CDF), and the reflection fold that keeps MCMC proposals in-box — previously available only to the `uniform` families — now applies. Internally this is a family-agnostic `TruncatedPrior` decorator in `pybnf/priors/`, with the `FreeParameter` owning the box. This unblocks faithful import of the common PEtab v2 pattern of a `normal` prior with finite `lowerBound`/`upperBound`: the PEtab importer now maps such a two-sided truncation to a bounded `FreeParameter` instead of raising. One-sided truncation (a single infinite bound) is still unsupported and raises with a clear message. (#411, ADR-0020)
- Official support for Python 3.13 and 3.14. CI now tests every supported version (3.11–3.14).
- **Parameter-recovery test tier** (`pytest -m recovery`, opt-in) — synthetic-data parameter recovery for tiny ODE models (exponential decay, logistic, Lotka–Volterra, SIR) fit through the **real bngsim backend**: each model is simulated at known-true parameters to generate a zero-noise data file, then a real fit (DE→Simplex refine, plus the `am` sampler) must recover them. This exercises the simulate→score→propose loop end to end with a genuine engine, complementing the analytical integration tiers (which use no simulation backend). Needs bngsim and BNG2.pl; auto-skips otherwise, so it never runs in hosted CI. See `tests/README_integration.md`.

### Changed
- Raised the minimum Python to 3.11 (was 3.10), aligning with the scientific-Python ecosystem — the latest numpy and scipy require Python 3.11.
- Dropped the upper version caps on `paramiko`, `msgpack`, `libroadrunner`, `numpy`, and `scipy` so installs track their latest releases (`pydantic`, `pyparsing`, and `bngsim` stay capped).

### Removed
- Python 3.10 support, and the `tomli` test dependency it required (the standard-library `tomllib` is available on 3.11+).

## [v1.4.0] - 2026-06-06

### Added
- **Powell and CMA-ES optimizers** — two native, derivative-free black-box optimizers, usable both standalone (`fit_type = powell` / `cmaes`) and as the post-fit refinement step. PyBNF now offers three refiners; choose one with the new `refine_method` config key (`sim` (default, Nelder–Mead Simplex), `powell`, or `cmaes`) when `refine = 1`. Powell uses conjugate-direction parabolic line searches; CMA-ES is a population-based covariance-adapting evolution strategy, robust on ill-conditioned objectives. No new dependency (#403)
- **CMA-ES box / global-start mode** — `fit_type = cmaes` now also accepts bounded `uniform_var` / `loguniform_var` priors instead of a `var` / `logvar` start point. Given a box, CMA-ES runs as a standalone *global* optimizer: it starts at the box center, seeds its covariance with the per-coordinate box widths (so the first generation spans the whole box), and repairs candidates into the box — no start point required, the population-optimizer ergonomics of `de` / `pso` with covariance adaptation. In box mode `cmaes_sigma0` is read as a fraction of each box width (default 0.3); in point-start / refine mode it remains the absolute initial step. Refinement (`refine_method = cmaes`) is unchanged. No new config key (#404, ADR-0017)
- `rotated_gaussian` analytical target (full covariance matrix `Sigma`, NLL `0.5 (x-mu)^T Sigma^{-1} (x-mu)`) for the in-process integration harness, plus `powell`/`cmaes` recovery tests on a rotated, ill-conditioned bowl. The correlated (non-separable) objective exercises Powell's conjugate-direction update and CMA-ES's covariance adaptation — paths the axis-aligned (separable) Gaussian target leaves untested (#405)
- `rotated_quartic` analytical target (`k1 r1^4 + k2 r2^2`, a smooth, non-separable, non-quadratic, trap-free curved valley) for the integration harness, used to test Powell's bracketing+Brent line search where the old fixed-step parabola stalled. The rotated *Gaussian* (quadratic) cannot discriminate, since a parabola fits a quadratic exactly (#406)
- **BNGsim in-process simulation bridge** — optional bngsim backend for BNGL models, avoiding subprocess BNG2.pl on every fitting iteration. Supports ODE, SSA, PSA, and NFsim methods with codegen ODE RHS compilation
- BNGsim SBML backend via bngsim's SBML loader
- BNGsim Antimony backend via bngsim's Antimony loader
- BNGsim NFsim backend for network-free BNGL simulation
- Hybrid BNGL path: models combining `generate_network()` with `simulate({method=>"nf"})` now run BNG2.pl once for network generation, then use in-process NFsim for all fitting iterations
- BNGsim `parameter_scan` support for time-course, steady-state, and protocol scan modes
- BNGsim `parameter_scan` steady-state solver with automatic fallback to long time-course when the solver does not converge
- Threaded steady-state and batch time-course parallelization in `parameter_scan`
- BNGsim protocol execution: `begin protocol`/`end protocol` blocks with multi-step simulate, `setConcentration`, `saveConcentrations`, `resetConcentrations`, `setParameter`, `saveParameters`, `resetParameters`, method switching, and `stop_if` support
- `method=>"protocol"` support in `parameter_scan` for executing protocol blocks at each scan point
- `continue=>1` flag for multi-phase simulations with model time tracking
- `stop_if` conditional early stopping in simulate actions
- `atol`/`rtol`/`seed` passthrough to bngsim simulator
- `print_functions` control for observable selection
- Bifurcate action support (`reset_conc=0`)
- Safe expression evaluation in action arguments
- `addConcentration` support in network-backed NFsim path
- Species IC re-evaluation from `.net` expressions
- Table function support (file-based and inline) for network-backed models
- Constraint satisfaction reporting for Bayesian samplers: after MCMC runs, a summary file reports the percentage of posterior samples satisfying each constraint (#324)
- Formal EBNF grammar for BPSL in the documentation (#271)
- `max_failed_simulations` config key to control early abort threshold (#146)
- `random_seed` config key to seed and log PyBNF-side random number generation (#31)
- Command-line options reference in documentation
- BNGsim package dependency (`bngsim>=0.5.0`) and optional Antimony install extra (#372)
- `bngl_backend` config key for BNGL backend control: `auto`, `bionetgen`, or required `bngsim` (#371)
- `stochastic_seed` config key for BNGsim stochastic simulations (`auto`, `auto_honorbngl`, `random`, `random_honorbngl`); under the default `auto`, PyBNF derives deterministic per-action seeds from the evaluation context so re-evaluations of the same parameter point reproduce, smoothing replicates yield distinct trajectories, and explicit BNGL `seed=>N` arguments are overridden with a warning. Covers SSA, PSA, NFsim, RuleMonkey, and SBML/Antimony stochastic backends (#373)

### Changed
- **Powell's line search robustified to bracketing + Brent** (ADR-0016): each 1-D line minimization now brackets the minimum (geometric expansion from `±powell_step`) and refines it with Brent's method to the new `powell_line_tol` (default `1e-4`), instead of the fixed-step parabola. It follows long, curved, non-quadratic valleys where the old fixed step stalled, and is confined to the parameter box so refining a bounded fit finds a minimum that lies past a bound *on* the boundary. The search is now fully serial (one objective evaluation per step; Powell no longer evaluates the two `±` probes concurrently — CMA-ES is the parallel derivative-free optimizer), and a convergence-stop guard ensures Powell never quits before its conjugate-direction update has run. `powell_step` keeps its key but now means the *initial bracketing step*. Powell now also solves the Rosenbrock/banana valley, a #403 non-goal. State stays picklable so backup/resume are unchanged (#406)
- **Simulated annealing (`fit_type = sa`) rewritten as a true optimizer** (ADR-0008): it now **minimizes the raw objective function** instead of the posterior (prior + likelihood). For `uniform_var`/`loguniform_var` priors this is a no-op — the prior was a constant inside the box that cancelled in the acceptance ratio, with proposal reflection enforcing the bounds — but for `normal_var`/`lognormal_var` priors **results change**: `sa` no longer acts as a silent MAP estimator, aligning it with every other PyBNF optimizer (`de`/`pso`/`ss`/`sim`), which use the prior only for the initial random draw. `sa` is extracted to its own class (`optimizers/simulated_annealing.py`) with a standalone config; the sampler-only `starting_params`/`continue_run` start paths no longer apply to it. `sa` remains deprecated.
- Supported BNGL network and NFsim simulations now auto-select BNGsim by default when available; `PYBNF_NO_BNGSIM=1` keeps the legacy BioNetGen path (#371)
- BNGL `method=>"nf"` routes to BNGsim's vendored NFsim and `method=>"rm"` to vendored RuleMonkey; PyBNF now delegates network-free method normalization and capability detection to bngsim's public `normalize_method()` / `HAS_NFSIM` / `HAS_RULEMONKEY` surface instead of maintaining its own alias tables. Missing vendored backends surface bngsim's "recognized but not present in this install" message. PyBNF no longer carries `nf_exact`/`nf_fixed`/`dynstoc`/`ds` as first-class aliases; bngsim's normalization governs their acceptance. Requires `bngsim>=0.5.0` (#377)
- Modernized packaging metadata in `pyproject.toml`, added dependency upper bounds, documented `uv` installation, and refreshed the Dockerfile with Python 3.12 and BioNetGen 2.9.3 (#360)
- Minimum supported Python version is now 3.10 (#372)
- Replaced nose test dependency with pytest
- Demoted high-frequency per-iteration log messages from INFO to DEBUG to reduce log file size (#173)
- Bayesian parameter priors and initial sampling now use shared `scipy.stats` distribution objects; normal and lognormal prior log-probabilities include SciPy's normalization constant while MCMC acceptance ratios are unchanged (#5)
- BNGsim stochastic simulations (SSA/PSA/NFsim/RuleMonkey, plus SBML/Antimony SSA) now use deterministic context-derived seeds by default; trajectories from re-runs of the same parameter point reproduce bit-for-bit. Set `stochastic_seed = random` to restore the previous wall-clock-style randomization (#373)
- **PyBNF-side random number generation migrated from NumPy's legacy global RNG (MT19937) to a per-algorithm `numpy.random.Generator` (PCG64, `default_rng`)** (#31). Each algorithm builds its own Generator from `random_seed`; the parallel samplers (`pt`, `am`, `dream`, `p_dream`) additionally give every chain its own `SeedSequence.spawn` sub-stream, so a seeded run now reproduces **regardless of the order parallel results come back** — the old shared global stream interleaved chain draws by completion order and so did not reproduce under real parallelism. Prior sampling, latin-hypercube initialization, and bootstrap-weight resampling now draw from the seeded Generator as well. Resuming a run restores the exact Generator state. **Reproducibility note:** because PCG64 is a different stream than MT19937, `random_seed = N` produces *different* (but still fully reproducible) results than prior releases — re-running with the same seed still reproduces saved data, including stochastic simulations under `stochastic_seed = auto` (whose content-hashed sim seeds are unchanged). Saved reference data generated by older releases must be regenerated once.

### Removed
- Removed the experimental S-CREAM (`s_cream`) sampler and its user-facing configuration/docs.

### Fixed
- Simplex collinearity with `simplex_reflection=1` and `population_size>1` in low dimensions (#207)
- `smoothing` and `parallelize_models` can now be used together for multi-model stochastic jobs (#49)
- Test failures in test_job_groups and test_seed_determinism caused by incorrect fixture paths (#361)
- `wall_time_sim` for SBML models now works when PyBNF is installed via PyPI (#249)
- Dependency warning spam (numpy, YAML, etc.) no longer clutters the terminal; routed to log file (#274)
- Invalid escape sequence `SyntaxWarning`s in the test suite (regex literals such as `'\s+'` not marked raw); Python is escalating these toward `SyntaxError`

## [v1.3.0] - 2026-03-29 (changes relative to v1.2.2, which was untagged)

### Added
- New Bayesian sampler: DREAM(ZS) (`dream` fit type) with ZS archive, snooker updates, adaptive gamma, CR adaptation, R-hat convergence diagnostics, and outlier detection
- New Bayesian sampler: Preconditioned DREAM (`p_dream` fit type) with covariance-preconditioned DE proposals
- Effective Sample Size (ESS) computation and R-hat convergence diagnostics in BayesianAlgorithm base class
- Configurable convergence stopping criterion (`converge_criterion`) and configurable delta for R-hat
- RoadRunner `saveState`/`loadState` optimization to avoid re-parsing XML on every `execute()` (closes #288)
- Validation of `continue_run` files before loading in Adaptive MCMC (#355)
- `burn_in` vs `max_iterations` validation and guard against empty samples (#356)
- Adaptive MCMC (`am`) hardened for production use: input validation, robustness fixes
- Warning when BNGL model has no observables defined (#298)
- Sampler benchmarking suite with analytical test targets

### Changed
- Minimum libroadrunner version bumped from 1.5.2 to 1.6.0
- DREAM donor chain selection now uses all chains instead of subset
- DREAM acceptance ratio uses natural log instead of log10 (#353)
- Simplex refinement reuses generated networks (#112)
- Subprocess timeout now kills entire process group (#83)

### Fixed
- DREAM out-of-bounds proposals not recording chain state, causing silently empty results
- Normalization edge case affecting `.prop` constraint columns with shared suffix (closes #276)
- Unbounded parameters having implicit lower bound of 0 (#208)
- Crash on non-ASCII characters in model files (#189)
- Crash with floating-point step size in `time_course` for XML models (#314)
- Exception handler crash during network generation (#294)
- `KLLikelihood`: negated return value and fixed broken data access (#352)
- `_load_t_length`: compute step count instead of storing step size (#354)
- Skip variable correspondence check during model checking (#281)
- `np.Inf` replaced with `np.inf` for NumPy 2.0 compatibility (#349)

## v1.2.2 (untagged)

Versions 1.2.0–1.2.2 were untagged development versions (community-contributed examples, minor patches).

## [v1.1.9] - 2021-09-20

### Added
- Initial Adaptive MCMC algorithm implementation
- Negative binomial objective function

## [v1.1.2] - 2020-12-31

### Fixed
- Pinned `msgpack==0.6.2` to fix compatibility issue

## [v1.1.1] - 2019-08-22

### Added
- `once between` constraint type for event-based constraints

## [v1.1.0] - 2019-08-22

### Added
- `pmin` and `pmax` keywords for setting parameter bounds in likelihood-based fitting
- Logit likelihood as an alternative to static penalty for constraint evaluation
- `SplitAtConstraint` class for splitting data at constraint boundaries

### Changed
- Data file duplicate column names now raise an error
- Unused data in `.exp` files now raises an error instead of a warning

### Fixed
- Simulated annealing crash when looking for `samples.txt`
- Errors from setting `population_size` too low

## [v1.0.1] - 2019-07-05

### Added
- Windows installation instructions and compatibility improvements
- First few failed simulation logs are now saved even without debug flag

### Changed
- More descriptive error messages for BioNetGen configuration errors

### Fixed
- Absolute Windows paths starting with drive letter now recognized
- `os.rename` replaced with `os.replace` for Windows compatibility
- Crash when `dask-worker-space` removal fails

## [v1.0.0] - 2019-03-15

### Added
- Command line argument `--log-level` for controlling logging verbosity
- Cluster class consolidating all cluster setup code

### Changed
- Renamed `bmc` algorithm to `mh` (Metropolis-Hastings); `bmc` still accepted for backwards compatibility

### Fixed
- Missing final histogram output in MH/PT algorithms

## [v0.3.3] - 2019-03-05

### Changed
- Renamed `.con` constraint file extension to `.prop` (property)
- Updated license to Triad National Security, LLC

## [v0.3.2] - 2019-01-08

### Added
- Model checking mode for validating model configuration without running a fit
- `parallelize_models` key for model-level parallelism
- `simulation_dir` option to control simulation working directory
- Sum of differences objective function
- Itemized constraint evaluation output

### Changed
- Config file validation now checks for unrecognized keys

### Fixed
- Relative path bug for `failed_logs_dir` on clusters
- `beta_range` corrected and changed to geometric space
- Badly-timed interrupt could lose algorithm backup

## [v0.3.1] - 2018-11-21

### Added
- `scheduler-file` argument for connecting to an existing Dask scheduler

### Changed
- Dask client is now reused across multiple `Algorithm.run()` calls

### Fixed
- Compatibility with newest Dask version

## [v0.3.0] - 2018-11-14

### Added
- Dockerfile for containerized execution
- Specific package version requirements in `setup.py`

### Changed
- Increased failed simulation tolerance to 100 before aborting

## [v0.2.3] - 2018-11-01

### Added
- `save_best_data` option to rerun the best-fit simulation and save output data files

### Changed
- RoadRunner output changed from "particles" to "concentration" mode to avoid unexpected results from SBML compartment volumes

### Fixed
- Bootstrapping with sum of squares objective
- Postprocessing `FailedSimulation` results no longer crashes

## [v0.2.2] - 2018-08-31

### Added
- LANL open-source license
- `parallel_count` support for `dask-ssh` cluster launches

### Fixed
- Random number overflow catch

## [v0.2.1] - 2018-08-16

### Changed
- MCMC now prints accept rate to log
- PSO accepts first parameter set even if score is Inf

### Fixed
- Failed folder deletions no longer terminate the fitting run

## [v0.2.0] - 2018-08-07

### Added
- Custom postprocessing support via user-defined Python functions
- RoadRunner integrator configuration (Euler with subdivisions)
- `simulation_actions` support for SBML models

### Changed
- Parameters with `lower_bound == upper_bound` are now allowed (fixed parameters)

## [v0.1.2] - 2018-08-01

### Added
- Bootstrap method for uncertainty quantification

### Fixed
- Thread leak: error catch and message when running out of threads
- Chi-squared formula corrected in documentation

## [v0.1.1] - 2018-07-16

### Changed
- Default PSO inertia weight changed from 1.0 to 0.7

### Fixed
- Particle Swarm reflections corrected for log-space variables
- Recovery from "too many reflections" error instead of crash

## [v0.1] - 2018-07-09

Initial release of PyBNF. Core fitting engine with support for BioNetGen (BNGL)
and SBML models via libRoadRunner. Includes Particle Swarm Optimization, Differential
Evolution, Scatter Search, Metropolis-Hastings, Simulated Annealing, Parallel Tempering,
and Simplex algorithms. Distributed computing via Dask. Constraint evaluation,
data normalization, multi-model fitting, and `.conf` configuration file format.
