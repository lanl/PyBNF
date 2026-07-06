.. _tutorial:

Tutorial
========

PyBNF ships a hands-on, self-contained tutorial that tours its modern
(**edition-2**) features on small ODE models. It lives under
``examples/tutorial/`` in the source tree, and every lesson links below to its
folder on GitHub.

Each lesson is a self-contained folder: a commented model (``*.bngl``), its data
(``*.exp``), one or more heavily-commented fits (``*.conf``), and a short
walkthrough (``README.md``). Run any lesson from its own folder, for example::

  cd examples/tutorial/01_logistic_growth
  pybnf -c logistic_growth_trf.conf

Results land in an ``output/`` directory inside the lesson folder. You need
`BioNetGen <https://bionetgen.org>`__ (with ``BNGPATH`` set) and PyBNF's bngsim
backend.

How the lessons work
--------------------

Every model is a deterministic ODE with a **known closed-form solution**, and its
data is generated *from that same model at known-true parameters*. A correct fit
therefore recovers the truth — which makes each lesson both a teaching example
and an **automated regression test**: the suite in
``tests/test_tutorial_examples.py`` re-runs the lessons and checks that the
recovered parameters land within tolerance of the truth (truths and tolerances
live in ``examples/tutorial/_manifest.py``). Some lessons drive samplers or
real-simulator parameter recovery and run slower; the test suite marks those as
opt-in *slow* and *recovery* tiers.

The lessons
-----------

Getting started
^^^^^^^^^^^^^^^

- `1. Logistic growth <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/01_logistic_growth>`__
  — your first fit; then two gradient optimizers, fitting qualitative data, and
  model checking (``trf``, ``lbfgs``, BPSL, ``check``).
- `3. Gompertz growth <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/03_gompertz_growth>`__
  — a global search followed by a local polish (``pso`` + ``refine``).

Optimizers and identifiability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `2. Bateman chain <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/02_bateman_chain>`__
  — fit several observables at once, and ask whether each rate is identifiable
  (``de``, profile likelihood).
- `6. Step input <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/06_step_input>`__
  — when a gradient fit is refused, and how a smooth (sigmoid) step fixes it.
- `7. Algorithm bakeoff <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/07_algorithm_bakeoff>`__
  — six optimizers on one oscillatory fit (``de``/``ade``/``pso``/``cmaes``/``sa``/``ss``).
- `25. Island DE <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/25_island_de>`__
  — a multi-island differential evolution with migration (``islands``,
  ``migrate_every``, ``num_to_migrate``).
- `44. Initialization <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/44_initialization>`__
  — where the search starts: seeding the initial population from an informative
  prior (``initialization``).

Objectives and noise models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `8. Robust objectives <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/08_robust_objectives>`__
  — when outliers wreck a fit, and the noise models that shrug them off
  (``noise_model``: Gaussian vs Laplace vs Student-t).
- `10. Per-observable noise <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/10_per_observable_noise>`__
  — give each reporter its own noise model (per-observable ``noise_model``).
- `18. Count likelihood <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/18_count_likelihood>`__
  — fit integer molecule counts with the likelihood built for them (``neg_bin``).
- `19. Shape objectives <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/19_shape_objectives>`__
  — fit the shape of a signal when its amplitude is arbitrary (column-joint
  ``kl`` / ``wasserstein``).
- `28. Cumulative counts <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/28_cumulative_counts>`__
  — fit incident counts from a cumulative prediction (per-observable
  ``cumulative``, ``neg_bin``).
- `35. Scale-free objectives <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/35_scale_free_objectives>`__
  — when data spans orders of magnitude: relative vs absolute error
  (``norm_sos`` / ``ave_norm_sos`` / ``sod``).
- `36. Estimate noise <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/36_estimate_noise>`__
  — fit noisy data with no error bars: let the fit estimate the noise
  (``noise_model = normal, sigma = fit``).
- `41. Estimate dispersion <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/41_estimate_dispersion>`__
  — estimate count over-dispersion jointly with the dynamics
  (``noise_model = neg_bin, dispersion = fit``).
- `42. Lognormal error <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/42_lognormal_error>`__
  — multiplicative (lognormal) measurement error over orders of magnitude
  (``noise_model = lognormal, location = mean``).
- `43. Custom objective <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/43_custom_objective>`__
  — bring your own objective: a custom Python callable (``objective = callable``).

Data, experiments and protocols
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `5. Noisy decay <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/05_noisy_decay>`__
  — uncertainty from resampling, and noise-weighted fitting (``bootstrap``,
  ``chi_sq``).
- `9. Experiment design <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/09_experiment_design>`__
  — richer designs: dose-response at steady state and a two-phase washout
  (``condition`` / ``preequilibrate``, parameter scans).
- `16. Joint fit <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/16_joint_fit>`__
  — fit two experiments at once with one shared rate set (multi-experiment,
  shared parameters).
- `21. Numerical hazards <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/21_numerical_hazards>`__
  — keep a fit alive when some simulations blow up or hang (``wall_time_sim``,
  ``max_failed_simulations``).
- `22. Normalization <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/22_normalization>`__
  — fit data reported relative to a reference (``normalization`` init / peak /
  zero / unit).
- `23. Resume <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/23_resume>`__
  — stop and resume a fit, or extend it with more iterations (``--resume``,
  backups).
- `24. Moment equations <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/24_moment_equations>`__
  — fit a model whose states are the mean and variance (moment-equation
  observables).
- `30. Data fusion <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/30_data_fusion>`__
  — one fit to time-course, steady-state, and qualitative data at once
  (multi-experiment + ``.prop``).

Bayesian inference and uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `17. Bayesian uncertainty <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/17_bayesian_uncertainty>`__
  — a posterior, not just a best fit: credible intervals from MCMC (``dream``).
- `26. MCMC samplers <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/26_mcmc_samplers>`__
  — two more posterior samplers: Metropolis-Hastings and parallel tempering
  (``mh``, ``pt``).
- `27. Priors <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/27_priors>`__
  — an informative prior vs a flat one on a weakly-identified rate
  (``gamma_var`` in a sampler).
- `32. Prior gallery <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/32_prior_gallery>`__
  — the whole catalog of prior families: how each is spelled and shaped
  (``normal_var`` / ``gamma_var`` / ``beta_var`` / … + ``student_t``).
- `37. HMC benchmark geometry <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/37_hmc_benchmark_geometry>`__
  — Hamiltonian Monte Carlo / NUTS on the built-in benchmark geometries
  (``job_type = hmc``).
- `38. HMC analytical ODE <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/38_hmc_analytical_ode>`__
  — an ODE's closed-form solution as an HMC likelihood, no simulator
  (``objective = expression`` + ``hmc``).
- `39. Adaptive MCMC <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/39_adaptive_mcmc>`__
  — Adaptive Metropolis on a correlated posterior, with R-hat / ESS via ArviZ
  (``job_type = am``).
- `40. Preconditioned DREAM <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/40_preconditioned_dream>`__
  — covariance-whitened proposals for a strongly correlated posterior
  (``job_type = p_dream``).
- `45. Model selection <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/45_model_selection>`__
  — which growth law? fit competing models and rank by AIC (multi-model).

PEtab interoperability
^^^^^^^^^^^^^^^^^^^^^^

- `12. PEtab round-trip <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/12_petab_roundtrip>`__
  — export, import, and validate a PEtab v2 problem (PEtab v2 + the BNGL linter).
- `13. PEtab lint clinic <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/13_petab_lint_clinic>`__
  — a gallery of broken problems: watch the linter catch each mistake
  (``petab.v2.lint``).
- `14. Observable layer <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/14_observable_layer>`__
  — fit what the instrument reports, not the raw species: a scale, a ratio, a log
  (measurement models / ``observableFormula``).
- `15. PEtab priors <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/15_petab_priors>`__
  — declare what you believe: a PEtab prior gallery and how each imports
  (``priorDistribution`` → priors).
- `20. PEtab observable parameters <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/20_petab_observable_parameters>`__
  — import per-observable gains and noise, the Boehm ``sd_*`` pattern
  (``observableParameters`` / ``noiseParameters``).
- `29. PEtab protocols <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/29_petab_protocols>`__
  — round-trip dose-response and pre-equilibration through PEtab
  (``conditions`` / ``experiments``).
- `33. SBML PEtab <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/33_sbml_petab>`__
  — import a standard SBML PEtab problem and fit it through bngsim
  (``sbml_backend = bngsim``).
- `34. PEtab observableFormula <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/34_petab_observable_formula>`__
  — an arithmetic ``observableFormula`` (ratio / log / scale) in a PEtab table,
  and its round-trip.

Model interoperability and checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `11. Interop <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/11_interop>`__
  — fit the same model as BNGL, SBML, and Antimony, one backend, one answer
  (``sbml_backend`` on bngsim).
- `31. BNGL + SBML fit <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/31_bngl_sbml_fit>`__
  — one fit mixing a BNGL model and an SBML model (multi-model).
- `46. Model checking <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/46_model_checking>`__
  — does the model satisfy the spec? check a model against qualitative properties,
  no fitting (``job_type = check``, BPSL ``at`` / ``once`` / ``always`` /
  ``between``).

For the authoritative lesson map — including the edition-2 config keys each lesson
uses — see the
`tutorial README on GitHub <https://github.com/lanl/PyBNF/tree/main/examples/tutorial>`__.

Interactive notebooks
---------------------

A companion set of Jupyter notebooks shows the same edition-2 workflows driven
**interactively** from a running kernel (PyBNF + bngsim), with results plotted
inline. They live under
`examples/notebooks <https://github.com/lanl/PyBNF/tree/main/examples/notebooks>`__
and are committed pre-executed, so you can read them without running anything.

- `01. Quickstart <https://github.com/lanl/PyBNF/blob/main/examples/notebooks/01_quickstart.ipynb>`__
  — write a model, make data, run a differential-evolution fit, plot fit vs data.
- `02. bngsim simulation <https://github.com/lanl/PyBNF/blob/main/examples/notebooks/02_bngsim_simulation.ipynb>`__
  — simulate a model forward with bngsim (no fitting): time courses, a parameter
  sweep, steady state.
- `03. Posterior exploration <https://github.com/lanl/PyBNF/blob/main/examples/notebooks/03_posterior_exploration.ipynb>`__
  — sample a posterior with HMC/NUTS, export an ArviZ ``InferenceData``, and read
  trace / pair / R-hat / ESS.
- `04. PEtab in a notebook <https://github.com/lanl/PyBNF/blob/main/examples/notebooks/04_petab_in_a_notebook.ipynb>`__
  — import a PEtab problem with ``pybnf.petab.import_job`` and fit it
  interactively.
- `05. Gradient fitting and profiles <https://github.com/lanl/PyBNF/blob/main/examples/notebooks/05_gradient_fitting_profiles.ipynb>`__
  — a multi-start trust-region least-squares fit using bngsim forward
  sensitivities, then profile-likelihood identifiability.
