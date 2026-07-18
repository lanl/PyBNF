# PyBNF edition-2 tutorial

A hands-on tour of PyBNF's modern (**edition-2**) features, built on small ODE
models with known closed-form solutions. Every model is fit to synthetic data
generated *from that same model at known-true parameters*, so a correct fit
recovers the truth — which makes each lesson both a teaching example and an
automated regression test (see `tests/test_tutorial_examples.py`).

Each lesson is a self-contained folder: a commented model (`*.bngl`), its data
(`*.exp`), one or more heavily-commented fits (`*.conf`), and a short README.

## Getting started

You need [BioNetGen](https://bionetgen.org) (set `BNGPATH`) and PyBNF's bngsim
backend. Run any lesson from its own folder:

```bash
cd examples/tutorial/01_logistic_growth
pybnf -c logistic_growth_trf.conf
```

Results land in `output/` inside the lesson folder.

## The lessons

| # | Folder | You learn… | Feature(s) |
| --- | --- | --- | --- |
| 1 | [`01_logistic_growth`](01_logistic_growth) | your first fit; **plus** two gradient methods, fitting qualitative data + model checking | gradient least-squares (`trf`) + quasi-Newton (`lbfgs`), **BPSL constraints**, `check` |
| 2 | [`02_bateman_chain`](02_bateman_chain) | fitting several observables at once; **is each rate identifiable?** | differential evolution (`de`), multi-observable, **profile likelihood** |
| 3 | [`03_gompertz_growth`](03_gompertz_growth) | global search then local polish | particle swarm (`pso`) + `refine` |
| 5 | [`05_noisy_decay`](05_noisy_decay) | uncertainty from resampling; noise-weighted fitting | `bootstrap`, `chi_sq` |
| 7 | [`07_algorithm_bakeoff`](07_algorithm_bakeoff) | **six optimizers on one oscillatory fit** — how each global search behaves | `de`/`ade`/`pso`/`cmaes`/`sa`/`ss` |
| 8 | [`08_robust_objectives`](08_robust_objectives) | when outliers wreck a fit — and the **noise models** that shrug them off | `noise_model` (Gaussian vs Laplace vs Student-t) |
| 9 | [`09_experiment_design`](09_experiment_design) | richer designs — **dose-response at steady state** and a two-phase washout | `condition:` / `preequilibrate:`, parameter scans |
| 10 | [`10_per_observable_noise`](10_per_observable_noise) | give each reporter **its own noise model** — robust only where you need it | per-observable `noise_model <obs> = …` |
| 11 | [`11_interop`](11_interop) | fit the **same model as BNGL, SBML, and Antimony** — one backend, one answer | SBML/Antimony on bngsim (`sbml_backend`) |
| 6 | [`06_step_input`](06_step_input) | when a gradient fit is *refused* — and how to fix it by smoothing the step | gradient-refusal + smooth (sigmoid) approximation |
| 12 | [`12_petab_roundtrip`](12_petab_roundtrip) | export/import/validate a PEtab v2 problem | PEtab v2 interop + the BNGL linter |
| 13 | [`13_petab_lint_clinic`](13_petab_lint_clinic) | a gallery of broken problems — **watch the linter catch each mistake** (errors, warnings, and load-time rejections across petab's whole default lint set) | PEtab v2 lint tasks (`petab.v2.lint`) through the BNGL loader |
| 14 | [`14_observable_layer`](14_observable_layer) | fit what the **instrument** reports, not the raw species — a scale, a ratio, a log | measurement models / `observableFormula` (ADR-0036) |
| 15 | [`15_petab_priors`](15_petab_priors) | **declare what you believe** — a PEtab prior gallery and how each imports | PEtab `priorDistribution` → `FreeParameter` priors |
| 16 | [`16_joint_fit`](16_joint_fit) | fit **two experiments at once** with one shared rate set — a two-route PK study | multi-experiment joint fit, shared parameters (multi-model) |
| 17 | [`17_bayesian_uncertainty`](17_bayesian_uncertainty) | a **posterior, not just a best fit** — credible intervals from MCMC | DREAM sampler (`dream`), credible intervals *(slow tier)* |
| 18 | [`18_count_likelihood`](18_count_likelihood) | fit **integer molecule counts** with the likelihood built for them | negative-binomial noise model (`neg_bin`, `dispersion`, `location`) |
| 19 | [`19_shape_objectives`](19_shape_objectives) | fit the **shape** of a signal when its amplitude is arbitrary | column-joint `profile_objective` (`kl` / `wasserstein`) |
| 20 | [`20_petab_observable_parameters`](20_petab_observable_parameters) | import PEtab **per-observable gains and noise** (the Boehm `sd_*` pattern) | PEtab `observableParameters` / `noiseParameters` import |
| 21 | [`21_numerical_hazards`](21_numerical_hazards) | keep a fit alive when some simulations **blow up or hang** | `wall_time_sim`, `max_failed_simulations` |
| 22 | [`22_normalization`](22_normalization) | fit data reported **relative to a reference** (init / peak / …) | `normalization <obs> = init\|peak\|zero\|unit` |
| 23 | [`23_resume`](23_resume) | **stop and resume** a fit, or extend it with more iterations | `--resume` / backups |
| 24 | [`24_moment_equations`](24_moment_equations) | fit a model whose states are the **mean and variance** | moment-equation observables |
| 25 | [`25_island_de`](25_island_de) | a **multi-island** differential evolution with migration | `islands`, `migrate_every`, `num_to_migrate` |
| 26 | [`26_mcmc_samplers`](26_mcmc_samplers) | two more posterior samplers — **Metropolis-Hastings and parallel tempering** | `mh`, `pt` *(slow tier)* |
| 27 | [`27_priors`](27_priors) | an **informative prior** vs a flat one on a weakly-identified rate | `gamma_var` in a sampler *(slow tier)* |
| 28 | [`28_cumulative_counts`](28_cumulative_counts) | fit **incident counts** from a cumulative prediction | per-observable `cumulative`, `neg_bin` |
| 29 | [`29_petab_protocols`](29_petab_protocols) | round-trip **dose-response and pre-equilibration** through PEtab | PEtab `conditions` / `experiments` |
| 30 | [`30_data_fusion`](30_data_fusion) | one fit to **time-course + steady-state + qualitative** data at once | multi-experiment, `.prop` constraints |
| 31 | [`31_bngl_sbml_fit`](31_bngl_sbml_fit) | one fit **mixing a BNGL model and an SBML model** | multi-model BNGL + SBML on bngsim |
| 32 | [`32_prior_gallery`](32_prior_gallery) | the **whole catalog of prior families** — how each is spelled and shaped | `normal_var`/`gamma_var`/`beta_var`/… + `student_t` record *(slow tier)* |
| 33 | [`33_sbml_petab`](33_sbml_petab) | import a standard **SBML** PEtab problem and fit it through bngsim | SBML PEtab v2 import, `sbml_backend = bngsim` |
| 34 | [`34_petab_observable_formula`](34_petab_observable_formula) | an **arithmetic** `observableFormula` (ratio/log/scale) in a PEtab table, and its round-trip | PEtab `observableFormula` expressions (ADR-0036) |
| 35 | [`35_scale_free_objectives`](35_scale_free_objectives) | when data spans **orders of magnitude** — relative vs absolute error | `norm_sos` / `ave_norm_sos` / `sod` objectives |
| 36 | [`36_estimate_noise`](36_estimate_noise) | fit noisy data with **no error bars** — let the fit estimate the noise | `noise_model = normal, sigma = fit <name>` |
| 37 | [`37_hmc_benchmark_geometry`](37_hmc_benchmark_geometry) | **Hamiltonian Monte Carlo / NUTS** on the built-in benchmark geometries (the banana) | `job_type = hmc` (jax extra) *(slow tier)* |
| 38 | [`38_hmc_analytical_ode`](38_hmc_analytical_ode) | an ODE's **closed-form solution as an HMC likelihood** — no simulator | `objective = expression` + `job_type = hmc` *(slow tier)* |
| 39 | [`39_adaptive_mcmc`](39_adaptive_mcmc) | **Adaptive Metropolis** on a correlated posterior + formal **R-hat/ESS via ArviZ** | `job_type = am` + ArviZ diagnostics *(slow tier)* |
| 40 | [`40_preconditioned_dream`](40_preconditioned_dream) | **Preconditioned DREAM** — covariance-whitened proposals for a strongly correlated posterior | `job_type = p_dream` + `precondition_adapt` *(slow tier)* |
| 41 | [`41_estimate_dispersion`](41_estimate_dispersion) | **estimating count over-dispersion** jointly with the dynamics (from replicate counts) | `noise_model = neg_bin, dispersion = fit …` *(recovery tier)* |
| 42 | [`42_lognormal_error`](42_lognormal_error) | **multiplicative (lognormal) measurement error** over orders of magnitude | `noise_model = lognormal, …, location = mean` *(recovery tier)* |
| 43 | [`43_custom_objective`](43_custom_objective) | **bring your own objective** — a custom Python callable (a robust mixture likelihood) | `objective = callable` + `callable = mod:func` |
| 44 | [`44_initialization`](44_initialization) | **where the search starts** — seeding the initial population from an informative prior | `initialization` / `initialization_distribution` *(recovery tier)* |
| 45 | [`45_model_selection`](45_model_selection) | **which growth law?** — fit competing models, rank by **AIC** | multi-model comparison (`de` + AIC) *(recovery tier)* |
| 46 | [`46_model_checking`](46_model_checking) | **does the model satisfy the spec?** — check a model against qualitative properties, no fitting | `job_type = check`, BPSL `.prop` (`at`/`once`/`always`/`between`) *(recovery tier)* |
| 47 | [`47_condition_perturbations`](47_condition_perturbations) | **one model, many conditions** — fit a wildtype AND a knockout with one model file | `condition: … perturbations:` + per-experiment `condition:` *(recovery tier)* |

## The edition-2 config surface, in one place

Every lesson uses these keys (see any `.conf` for the full commentary):

| Key | Meaning |
| --- | --- |
| `edition = 2` | opt into the modern config language |
| `model: file.bngl` | declare the model (no data bound here; no `begin actions` block) |
| `bngl_backend = bngsim` | simulate in-process with bngsim (required for gradient fits) |
| `job_type = …` | the algorithm: `trf`/`lbfgs` (gradient), `de`/`pso`/`ss`/… (metaheuristic), `am`/`dream`/… (Bayesian) |
| `objective = sos` \| `chi_sq` | the fit metric (`chi_sq` when the data has `_SD` columns) |
| `noise_model = family, …` | assemble the objective from a noise family + parameter sources (Lesson 8) |
| `experiment: name, data: file.exp` | bind a named simulation to its data; the data's time column is the output grid |
| `observable: id, formula: expr` | a measurement model — score a post-simulation formula over the outputs (Lesson 14) |
| `condition: name, perturbations: …` | a named parameter setting; `preequilibrate:`/`condition:` on an experiment build multi-phase protocols (Lesson 9) |
| `uniform_var = p lo hi` | a free parameter, bound by name to model parameter `p`, searched in `[lo, hi]` |
| `refine = 1` | polish the best fit with a local optimizer |

## Not covered here (and why)

These models are **deterministic ODEs with closed-form solutions**, chosen so
every fit has a known right answer. That deliberately leaves three PyBNF
capabilities out of scope, because this palette can't exercise them honestly:

- **Stochastic simulation (SSA) and network-free (NFsim)** — these are tiny,
  fully-enumerable ODE networks; SSA would be non-deterministic and network-free
  simulation buys nothing.
- **Distributed-cluster execution** — orthogonal to the modelling; it's a
  deployment concern, not a per-model feature.

## Regenerating the data

The `.exp` files are the models' own output at the true parameters. To rebuild
them (needs bngsim + BNG2.pl):

```bash
python examples/tutorial/regenerate_data.py            # all lessons
python examples/tutorial/regenerate_data.py 02_bateman_chain   # one lesson
```
