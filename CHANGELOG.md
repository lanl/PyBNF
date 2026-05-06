# Changelog

All notable changes to PyBNF are documented below. This project adheres to
[Keep a Changelog](https://keepachangelog.com) conventions.

## [Unreleased]

### Added
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
- BNGsim package dependency (`bngsim>=0.3.0`) and optional Antimony install extra (#372)
- `bngl_backend` config key for BNGL backend control: `auto`, `bionetgen`, or required `bngsim` (#371)

### Changed
- Supported BNGL network and NFsim simulations now auto-select BNGsim by default when available; `PYBNF_NO_BNGSIM=1` keeps the legacy BioNetGen path (#371)
- Modernized packaging metadata in `pyproject.toml`, added dependency upper bounds, documented `uv` installation, and refreshed the Dockerfile with Python 3.12 and BioNetGen 2.9.3 (#360)
- Minimum supported Python version is now 3.10 (#372)
- Replaced nose test dependency with pytest
- Demoted high-frequency per-iteration log messages from INFO to DEBUG to reduce log file size (#173)
- Bayesian parameter priors and initial sampling now use shared `scipy.stats` distribution objects; normal and lognormal prior log-probabilities include SciPy's normalization constant while MCMC acceptance ratios are unchanged (#5)

### Removed
- Removed the experimental S-CREAM (`s_cream`) sampler and its user-facing configuration/docs.

### Fixed
- Simplex collinearity with `simplex_reflection=1` and `population_size>1` in low dimensions (#207)
- `smoothing` and `parallelize_models` can now be used together for multi-model stochastic jobs (#49)
- Test failures in test_job_groups and test_seed_determinism caused by incorrect fixture paths (#361)
- `wall_time_sim` for SBML models now works when PyBNF is installed via PyPI (#249)
- Dependency warning spam (numpy, YAML, etc.) no longer clutters the terminal; routed to log file (#274)

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
