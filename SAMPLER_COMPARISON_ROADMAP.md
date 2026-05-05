# MCMC Sampler Comparison Roadmap

Compare the Bayesian samplers in PyBNF: Adaptive MCMC (`am`), DREAM(ZS) (`dream`), and P-DREAM (`p_dream`).

## Phase 1: Infrastructure

- [x] Add ESS (Effective Sample Size) computation to `BayesianAlgorithm`
  - Compute from chain autocorrelation (bulk ESS and tail ESS per Vehtari et al. 2021)
  - Report alongside rank-normalized split-R-hat every 10 iterations
  - Write ESS to output files for post-hoc analysis
- [x] Refactor rank-normalized split-R-hat out of `DreamAlgorithm` into `BayesianAlgorithm`
  - All samplers should use the same convergence diagnostic
  - Add automatic convergence stop (`rhat_threshold`) to `BayesianAlgorithm` base class
- [x] Add ESS/evaluation metric (ESS divided by total model evaluations)

## Phase 2: Analytical Test Targets

Build lightweight test models that compute score directly from parameters (no BioNetGen needed).

- [x] **Multivariate Gaussian** (d configurable): known mean, covariance, and marginals
  - Simplest correctness check: compare sampled mean/variance to truth
  - Scale d = 5, 10, 20, 50 for dimensionality study
- [x] **Banana (Rosenbrock)**: correlated curved posterior
  - Tests sampling along non-linear ridges
- [x] **Multimodal mixture**: 2-3 separated Gaussian modes with known weights
  - Tests mode-jumping and snooker updates
- [x] Each target needs: trivial `.bngl` or direct-score model, `.conf` files for all samplers, ground truth file with analytical moments

## Phase 3: Benchmark Harness

- [x] Script that runs each sampler N times (N >= 5 for variance) on the same target with the same evaluation budget
- [x] Collect per-run: ESS per parameter, ESS/evaluation, R-hat trajectory, acceptance rate, wall-clock time
- [x] For analytical targets: distance metric D = normalized error in mean and std (Laloy & Vrugt 2012, Eq. 10)
- [x] Output: comparison tables and convergence trajectory plots

## Phase 4: Experiments

All experiment configs and the master runner are ready in `benchmarks/`.
Samplers: `am`, `dream` (ZS), `p_dream` (preconditioned ZS).
Run: `python benchmarks/run_all_experiments.py --replicates 5 --skip-egfr`
Or specific experiments: `python benchmarks/run_all_experiments.py -e 1 2`

- [ ] **Correctness**: all samplers on multivariate Gaussian, verify sampled moments match truth
  - Config: `benchmarks/gaussian_d5/`, run with `run_benchmark.py gaussian_d5`
- [ ] **Efficiency (low-d)**: Gaussian d=5, compare ESS/evaluation across samplers
  - Same run as Correctness — both metrics extracted from `gaussian_d5`
- [ ] **Efficiency (scaling)**: Gaussian at d=5, 10, 20, 50 — how does ESS/evaluation scale?
  - Configs: `benchmarks/gaussian_d{5,10,20,50}/`
  - Scaling table printed by `run_all_experiments.py --summary-only`
- [ ] **Correlated target**: Banana distribution, compare ESS and mixing
  - Config: `benchmarks/banana/`
- [ ] **Multimodal target**: Mixture of Gaussians, compare mode coverage and ESS
  - Config: `benchmarks/multimodal/`
- [ ] **Real-world**: EGFR benchmark (d=37), compare convergence speed and posterior agreement
  - Config: `benchmarks/egfr/` (requires BioNetGen; use `--skip-egfr` to exclude)

## Phase 5: Analysis and Write-up

- [ ] Summarize results: which sampler wins where, and why
- [ ] Identify failure modes by target geometry and dimensionality
- [ ] Update documentation with recommendations
