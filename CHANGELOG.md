# Changelog

All notable changes to PyBNF are documented below. This project adheres to
[Keep a Changelog](https://keepachangelog.com) conventions.

## [Unreleased]

### Fixed
- **A pre-equilibrated dose-response experiment can now be fit by a gradient method (#532,
  ADR-0098).** The preincubate → wash → dose-scan protocol (`preequilibrate:` + `condition:` +
  `type: parameter_scan`) refused every scored gradient evaluation, and the refusal landed at
  *scoring* — so a `trf` fit of Erickson 2019's `igf1r` job "finished" with `inf` at all ten starts
  and said only `Unknown error during job bestfit_infocrit`. Two things were wrong. The guard
  itself was **stale**: bngsim 0.12.0 (lanl/bngsim#81, #111) carries the state each dose restores
  *together with* its `dx/dθ`, which is exactly the capability the guard said did not exist; it is
  now a capability gate (`bngsim >= 0.12.0`, the new `pyproject` floor), and each dose's tensor
  stacks down the dose axis like any other scan's. Underneath it, the protocol's **wash** was
  silently discarding the equilibration's derivative — `Model.set_concentration` drops the pending
  `dx/dθ` rather than guess an externally supplied amount's, so *no* pre-equilibrated experiment
  with a species intervention could be gradient-fit, a measured time course failing outright with
  `carry_sensitivities=True, but no matching forward-sensitivity seed from a prior phase is
  available`. PyBNF now supplies the row it knows: the intervention's own `∂x_k(0)/∂θ` — `0` for a
  literal amount, the exact derivative for one written over model parameters (differentiated
  through the `.net`'s derived ids), the carried row for an `addConcentration` — with the rest of
  the matrix preserved, and an honest refusal naming the assignment when it lies outside the
  arithmetic grammar. A `resetConcentrations()` that follows a `saveConcentrations()` is likewise
  recognised as returning to a *carried* state, which is what made a model's **second**
  pre-equilibration experiment refuse. All seven `igf1r` rate constants now agree with central
  differences to ≤ 2.3e-04 on all three experiments, and the fit reaches a finite objective.
- **A refusal raised while simulating now stops the fit and states its reason, instead of
  returning `inf` at every start (#532).** `Job.run_simulation` swallowed a user-targeted
  `PybnfError` into its generic "unknown error" arm, so a property of the *setup* — a model
  construct this `job_type` cannot handle, a missing backend capability — was reported once per
  evaluation as a failed simulation and the run continued to a meaningless finish. The documented
  fail-fast policy (re-raise; it would fail every job) existed one layer up and was never reached.
  Scoring failures are unchanged: a per-point objective failure still penalizes that point (#388).
- **A gradient start that reaches a point it cannot differentiate no longer takes the whole
  multi-start fit down with it (#528, ADR-0092).** A stiff parameter point can score finitely while
  its forward sensitivities diverge, leaving a finite objective with a non-finite gradient. That
  model went straight into the trust-region factorization, where LAPACK refuses it
  (`Sorry, an unknown error occurred: numpy.linalg.LinAlgError: SVD did not converge`) and the
  exception unwound out of the run loop — 19 healthy starts of a 20-start `gntr` fit discarded
  because of one, which inverts the reason multi-start exists. (`lbfgs` aborted the same fit by a
  different route: its NaN direction proposed a NaN point, rejected as
  `OutOfBoundsException: Free parameter k cannot be assigned the value nan`.) An unusable local
  model is now treated exactly as a failed simulation already was: **mid-search the trial is
  rejected** — the trust region shrinks, or the line search backtracks — and that start carries on
  from its current iterate; **at the start point that one start stops**, saying which model was
  unusable (`the Fisher model (gradient + EFIM Hessian) at the start point is not finite (the point
  scored, but its derivatives did not)`), while every other start keeps running and the global best
  is taken across the survivors. The two LAPACK calls in the step math (`svd`, `eigh`) are wrapped
  as well, so a factorization that fails on finite-but-pathological input routes the same way.
  `profile_likelihood`, which drives the same runners, was a third casualty of the same missing
  guard: a slice whose derivatives diverge now ends that one direction at a wall it names, with the
  un-optimizable grid point contributing no profile value — rather than entering an un-minimized
  upper bound as if it were the profile, which would inflate that point's Δχ² and could close the
  confidence interval too narrowly. A fit whose models are all finite is unchanged.
- **A negative count is no longer scored as a perfect fit, nor counted in `n` (#523, ADR-0090).**
  The `neg_bin` family has a negative observation contribute nothing to the objective — right for
  the fit, since a negative count has no negative-binomial probability, and real surveillance data
  contains them (a downward revision of a cumulative total makes a negative daily increment). But a
  negative-binomial PMF is self-normalizing, so that zero cost became `log p = 0` — probability
  **one** — in the pointwise log-density, a better per-point density than the family assigns any
  real count, including one the model predicts exactly. Those points were also counted as scored
  points, entering `n` for AIC/BIC and the LOO/WAIC observation axis. An observation outside its
  noise family's **observation domain** is now excluded exactly as a NaN observation already was:
  off the observation axis, out of `n`, and reported once per observable with a count
  (`excluded 4 measurement(s) of 'cases' in ...: this observable's noise model scores only a
  non-negative count`). Scoring data containing negative counts now matches scoring the same data
  with those rows deleted. The **cost** path is deliberately unchanged — such a point still
  contributes nothing to the objective and to the gradient — and every family whose support is the
  whole real line (`normal`, `lognormal`, `lnnormal`, `laplace`, `student_t`) is byte-identical.
- **Steady-state (`time = inf`) measurements now load and fit (#521, ADR-0086).** A PEtab problem
  measured only at equilibrium imported fine but crashed at configuration load
  (`OverflowError: cannot convert float infinity to integer`): the experiment was materialized as
  an ordinary time course, which derives its step count from a (here infinite) endpoint. PyBNF's
  only steady-state route was the dose-response `parameter_scan` of ADR-0046, which needs a swept
  axis a plain equilibrium observation does not have. An `.exp` whose `time` column is all `inf`
  is now recognized as a **steady-state experiment**: it emits
  `simulate({...,steady_state=>1,n_steps=>1})` — the relaxation-with-early-stop primitive
  pre-equilibration already used — and the objective scores the datum against the run's final
  (equilibrium) row. `t_end:` bounds the relaxation (default `1e6`) instead of timing a readout,
  and `type: steady_state` may state explicitly what the data implies. Supported on BNGL
  (BNG2.pl/bngsim), bngsim SBML/Antimony, and RoadRunner (which uses its own steady-state solver,
  falling back to the bounded integration); forward sensitivities are carried at the equilibrium,
  so `trf`/`lbfgs`/`gntr` fit these problems. NFsim (`method: nf`) has no steady-state solve and is
  refused, as is an experiment mixing `inf` with finite times. This unblocks
  `Blasi_CellSystems2016`, the last unimported subset-I problem of the Grein et al. 2026 benchmark
  collection.
- **PEtab conditions measured only at `t = 0` now load and evaluate as initial-state
  observations (#510).** A data-derived ``TimeCourse`` previously required at least one positive
  output time, so one legitimate initial-state condition rejected the entire imported problem
  (including every ordinary time course); this blocked ``Schwen_PONE2014``. SBML/RoadRunner and
  SBML/bngsim now return the initialized model as a one-row ``t = 0`` trajectory without invoking
  an integrator. The bngsim gradient path also supplies the initial-condition identity derivative
  and differentiates parameter-driven SBML ``initialAssignment`` expressions, so ``trf`` /
  ``lbfgs`` / ``gntr`` retain correct forward sensitivities for an initial-only experiment.
- **PEtab natural-log Gaussian observables now import exactly (#509, ADR-0084).** PEtab v1
  ``observableTransformation = log`` and v2 ``noiseDistribution = log-normal`` previously reached
  PyBNF's internal ``Gaussian(LN)`` kernel but could not be serialized into the generated ``.conf``;
  ``import_job`` raised ``NotImplementedError``. The new explicit ``lnnormal`` noise family is
  ``Gaussian(additive_on=LN, location=MEDIAN)`` and is kept distinct from PyBNF's existing
  ``lognormal`` (log10) family. Imports now preserve the natural-log residual and sigma units, and
  normalized pointwise log-likelihoods use the natural-log Jacobian ``-log(y)`` (so
  ``information_criteria.txt`` is on the correct absolute scale). This unblocks
  ``Blasi_CellSystems2016`` and ``Laske_PLOSComputBiol2019``. The exact reverse mapping also exports
  ``lnnormal`` as PEtab v2 ``log-normal``; log-scale Laplace remains unsupported by the native
  configuration surface.
- **PEtab import now preserves replicate-specific `observableParameters` / `noiseParameters`
  bindings (#508, ADR-0083).** The per-measurement sidecar was keyed only by column, time, and
  placeholder, so repeated PEtab cells from different replicates collided and the last
  replicate's token silently replaced the others. This blocked `Fiedler_BMCSystBiol2016` by
  orphaning the first gel's scale parameters and could silently fit other problems with the wrong
  per-row scaling/noise binding. Replicate-aware sidecars now add a 1-based `replicate` column,
  using the same row-dealing partition that creates each `_repN.exp`; configuration loading and
  PEtab re-export select tokens by `(replicate, time)`. Legacy four-column sidecars remain valid
  and retain their shared-across-replicates meaning.
- **PEtab import: `observableParameters`/`noiseParameters` placeholders with a fixed noise
  parameter, multiple noise tokens, or an affine/prediction-scaling noiseFormula now import (#495,
  ADR-0075).** Three related gaps in the placeholder→parameter mapping left benchmark-collection
  problems unimportable. (a) `Oliveira_NatCommun2021` — a `noiseParameters` id (`sd_cumulative_*`)
  that is **fixed** (`estimate=0`) was emitted as a `fit` free sigma the `.conf` never declared, so
  the job failed to load; it now inlines as a constant sigma. (b) `Fiedler_BMCSystBiol2016` — a
  **multi-token** `noiseParameters` cell (`s_gel;sigma`, a `noiseParameter1 * noiseParameter2`
  formula) was never split (only `observableParameters` was), so the whole cell was mis-read as one
  id; it now splits and binds each `noiseParameter${n}` per data point when row-varying. (c)
  `Raia_CancerResearch2011` — an affine `noiseParameter1 + noiseParameter2 * (species…)` produced a
  `noise_model` line that referenced the simulated trajectory a `formula` sigma cannot see, failing
  to parse; it now imports as the new `prediction_formula` source (see Added). All three land on
  crafted simulator-free fixtures scored against a hand-derived NLL.
- **PEtab SBML import: an `observableFormula` referencing an SBML `assignmentRule` variable now
  imports instead of being rejected as "not a model entity" (#493).** An `assignmentRule`-defined
  variable (a `<parameter>`/`<species>` with `constant="false"` whose value is set by an
  `<assignmentRule>`) is a *derived* model output — the backend recomputes it at every step — so it
  is exactly the kind of quantity an observable is built from (the SBML analogue of a BNGL global
  function, which PyBNF already accepts). `import_job`'s measurement-model importer recognized only
  species / parameters / observables / functions, so six PEtab benchmark-collection problems
  (`Giordano_Nature2020`, `Laske_PLOSComputBiol2019`, `Rahman_MBS2016`, `SalazarCavazos_MBoC2020`,
  `Smith_BMCSystBiol2013`, `Zhao_QuantBiol2020`) could not be imported at all — a bare-name formula
  raised *"has a bare observableFormula … which is not a model entity"* and an expression formula
  raised *"references … which is not a known model entity"*. The importer now **inlines** any
  referenced assignment-rule variable down to the species/parameters its rule is defined over
  (recursively), exactly the resolution the config-load measurement layer already performs (#465,
  ADR-0036); the model file is carried byte-verbatim and a formula naming no rule variable is
  returned unchanged (the bare-name common case stays dependency-free).
- **PEtab import: `observableTransformation = log10` is no longer dropped in the v1→v2 conversion
  (#499).** A PEtab **v1** observable with `observableTransformation = log10` (Perelson_Science1996,
  Borghans_BiophysChem1997, Elowitz_Nature2000, and other multi-decade-signal benchmark problems)
  imported as a **linear** `gaussian` noise model, so the fit optimized the *wrong* objective — a
  linear residual with no change-of-variables Jacobian instead of the `log10` residual the problem
  (and the paper) specify. The `log10` transformation was silently dropped by `petab.v2.petab1to2`
  (PEtab v2 removed the `observableTransformation` column and has **no** `log10` `noiseDistribution`;
  it downgrades `log10-normal` to a blank distribution), and `import_job` read only
  `noiseDistribution`, so the observable resolved to `Gaussian(LINEAR)`. Directly parallel to the
  `parameterScale` drop `petab1to2_preserve_scale` (#491) already fixes:
  - **`pybnf.petab.petab1to2_preserve_scale` now also re-injects `observableTransformation`** as a
    preserved extra column on the converted v2 observables table (v2 lint-clean; other tools ignore
    it). Since v2 has no faithful `log10` `noiseDistribution`, this extra column is the only channel
    for a `log10` residual — the observable twin of the `log-uniform` parameter-scale re-injection.
  - **The importer selects the noise family's *additive scale* from `observableTransformation`, not
    just the family from `noiseDistribution`.** `log10 + normal` → the native `lognormal` family
    (`Gaussian(LOG10, MEDIAN)`, which already carries the correct log10-space residual **and** the
    `Σ log(y·ln10)` Jacobian), emitted as `objective = lognormal` (or a `noise_model = lognormal, …`
    line); `log + normal` maps to the natural-log `Gaussian(LN)` (v2's `log-normal`). The
    `pybnf.petab.observables` adapter reads it the same way (`log10` → LOG10, `log` → LN, `lin` →
    unchanged), with a guard against a transformation that contradicts a log `noiseDistribution`.
    Natural-log Gaussian now routes to ``lnnormal`` (#509, ADR-0084); log ``laplace`` families still
    raise ``NotImplementedError`` — the boundary stays in code, never a silent mis-recovery. A
    linear problem is byte-for-byte unchanged.

### Added
- **A floored or analytically scaled observable is now a gradient target (#533, ADR-0099).** The
  two ADR-0066 normalization primitives shipped with deliberately deferred gradients, so a fit
  whose experiment declared either — `normalization <obs> = floor 0.03, scale`, the chain
  arbitrary-unit fluorescence / blot data is fit with — was unavailable to **every** gradient
  `job_type`, refusing with *"Analytic per-series scaling ('scale', #479) on column '…' is not
  differentiable on the gradient path"*. Both are now threaded. The **floor** (`x' = x + ρ·max x`)
  is additive, so every row picks up the same `ρ·s_argmax` term. The **analytic scale** is the one
  transform that is not per-point — its `c*` is profiled out of the whole matched series — so the
  scored value `c*(θ)·ŷ_i(θ)` differentiates by the product rule, with `∂c*/∂θ` the closed-form
  derivative of the profiling condition itself (the geometric-mean ratio for a log family, the
  least-squares optimum for a linear one), computed once per experiment and shared by every point of
  the column. That term does **not** drop out by the envelope theorem: the profiling is
  σ-unweighted, so `c*` is not in general the objective's own minimizer over the scale. A scaled
  fixed-σ Gaussian fit stays an exact least-squares model, so `trf`, `lbfgs`, and `gntr` all consume
  it; the profiled scale is resolved per experiment, so a column scaled in one experiment stays an
  ordinary column in another. One boundary is now stated rather than silently mis-differentiated: a
  chain of two or more *data-level* transforms on one column (`floor 0.03, peak`) keeps only the
  last one's facts, so the gradient path refuses it by name — `floor 0.03, scale` is unaffected,
  since `scale` is applied at scoring time rather than to the column.
- **A total wall-clock budget for a fit, `wall_time_fit`, that finalizes on expiry (#529,
  ADR-0093).** PyBNF's time limits were all per unit of work — `wall_time_sim` bounds one
  simulation, `wall_time_gen` one network generation — and nothing bounded a *run*: the only native
  budget was `max_iterations` × `population_size`, which is not convertible to wall time without
  knowing per-iteration cost in advance. The new global key sets the seconds a whole fit may run
  (`wall_time_fit = 10800`; `0`, the default, is unbounded). On expiry the run stops launching work,
  abandons what is in flight, and runs the **normal** end-of-fit path against the best point found
  so far — `sorted_params_final.txt`, the best-fit simulations, `information_criteria.txt`, the
  ArviZ sidecar, the backup rename — so a budgeted result is scoreable exactly like a converged one.
  Only the stop *reason* differs, and it is logged, printed, and written to
  `Results/stop_reason.txt` (whose presence is the signal; no existing file's format changes). The
  clock starts when PyBNF starts, so configuration loading and network generation count against it,
  and one budget bounds the whole run: no `refine` and no further bootstrap replicate begins once it
  is spent. This makes PyBNF runnable under wall-time-budgeted optimizer benchmarks (Grein et al.
  2026), where the previous alternative — killing the process — lost the artifacts scoring needs.
  Two overruns are deliberate and documented: one in-flight simulation may run up to `wall_time_sim`
  past the deadline before it is abandoned, and finalizing re-simulates the best fit once. Refused
  (rather than silently ignored) for `job_type = hmc`, which runs its own in-process sampling loop.
- **Published-source organization and two curated real-world jobs.** The real-world gallery now
  uses `Author-Year/job_slug` paths aligned with the BNGL-Models job corpus. The former flat
  Kozer EGFR, Monine TLBR, Gupta FcεRI, and Mitra receptor jobs retain their tested edition-2
  configurations under source-oriented collections; curated provenance, validation notes, and
  reproduction assets accompany the Kozer and Monine jobs. The reduced F5B-only IGF1R teaching
  fit is replaced here by `Erickson-2019/igf1r`, the authors' published seven-rate,
  three-dataset preincubate→wash→dose-scan fit (the reduced job remains under `examples/igf1r/`).
  Two additional workstation examples broaden the executable corpus:
  `Salazar-Cavazos-2019/egfr_simpull` adds an authors' multisite-EGFR ODE fit, and
  `Kirsch-2020/phosphoswitch_bpsl` adds a four-model, constraint-only BPSL fit. The default
  corpus test now distinguishes quantitative `.exp` jobs from qualitative `.prop` jobs.
- **Workstation-scale exact-SSA real-world examples (#472).** The new
  `examples/real-world/Rijal-2025/` collection fits lacUV5/lacUD5 and 5DL1 promoter-noise data
  from Jones et al. (2014) with the two-state model studied by Rijal and Mehta (2025). Each
  edition-2 SSA job uses `method: ssa`, 200-trajectory smoothing, and measurement formulas for
  ensemble mean and standard deviation; paired exact moment-ODE jobs provide deterministic
  references, with source tables, regeneration scripts, validation notes, and reproduction
  figures preserved alongside them. All four configurations receive backend-free corpus checks,
  and the bounded SSA fits enter the opt-in bngsim recovery tier, closing the gap left by the
  cluster-scale FcERI SSA reference.
- **CMA-ES gains an optional bounded per-run generation budget
  (`cmaes_run_maxgen`; #507, ADR-0085).** The global `max_iterations` budget previously
  left the initial run and every IPOP / BIPOP large run unbounded, so one run making
  slow progress in an ill-conditioned local basin could consume nearly the entire fit
  before later restarts launched. Setting the new positive-integer cap applies it to
  every run and turns reaching it into the existing per-run restart trigger; the final
  run then stops at the same cap. BIPOP small runs use the smaller of this user cap and
  their existing automatic evaluation-balancing cap. The default is unset, preserving
  the prior schedule and results.
- **Prediction-dependent noise: a `noise_model … = <family>, sigma = prediction_formula <expr>`
  source whose σ scales with the simulated output (#495, ADR-0075).** The honest combined
  additive+proportional error model `σ = σ_abs + σ_rel · y` — where `y` is the observable's
  *predicted* value — previously had no native form: `formula` (ADR-0044) evaluates only over free
  parameters, and `relative` / `column_mean` (ADR-0031) read the *data*, not the simulation. The new
  `prediction_formula` verb builds a `PredictionFormulaSigma` whose symbols resolve either from the
  PSet (the estimated coefficients) or from the current simulation column of that name (a model
  species / observable / function), evaluated per scored point. Any new-era job may author it; PEtab
  import is one way to reach it. (Gradient-free score path only — a prediction-dependent σ raises
  `GradientNotSupported` on the #385 gradient/EFIM path, a later sub-layer.)
- **Scale-preserving PEtab v1→v2 conversion: `pybnf.petab.petab1to2_preserve_scale`.** The
  official `petab.v2.petab1to2` **drops** the v1 `parameterScale` column (PEtab v2 removed it) and
  only *warns* — so a `parameterScale = log10` estimated parameter carrying no objective prior (the
  common case for a multi-decade kinetic parameter) converts to a *linear* `uniform_var` over the raw
  bounds: the same argmin, but a far harder, worse-conditioned optimization than the log10 search the
  modeler specified. This wrapper runs the standard converter and re-injects the dropped estimation
  scale in the **v2-native** form — `priorDistribution = log-uniform` over each such parameter's
  bounds — which PyBNF imports as a `loguniform_var` on the Log10 scale. Because the optimizer
  objective excludes the prior, this sets only the search scale and initial sampling, not the
  objective, so the fit stays the pure-MLE problem v1 specified; parameters petab1to2 already folded
  into a prior (`parameterScale*Normal` → `log-normal`, …) are left untouched, as are linear ones.
  `import_job` stays a pure v2 importer — the conversion is an explicit, named step, not a reach-back
  to v1 in the read path. Intended as the opt-in migration `petab1to2` itself should offer.
- **General-objective trust-region optimizer: `fit_type = gntr` (ADR-0068, #481).** Fills the last
  empty cell of the gradient-fitting (objective × curvature-model) matrix. `trf` gives a
  trust-region step with a `JᵀJ` (Gauss-Newton / empirical-Fisher) Hessian but only for an exact
  least-squares objective; the moment the objective stops being a pure sum of squares — an estimated
  noise scale, a Laplace / count likelihood, or an active constraint — the gradient path dropped to
  `lbfgs` (limited-memory quasi-Newton). `gntr` extends `trf`'s well-conditioned trust-region step
  to those general-NLL objectives: its Hessian is the **expected-Fisher / Gauss-Newton information**
  `H = Σ κᵢ sᵢsᵢᵀ` (+ estimated-noise and constraint blocks), built from the same #385 forward
  sensitivities `sᵢ = ∂predᵢ/∂θ` plus small analytic per-family curvature factors (new
  `NoiseModel.location_fisher` / `noise_param_fisher` and `Constraint.penalty_curvature` seams) — no
  second-order sensitivities. It consumes the *same scalar gradient* as `lbfgs`; only the curvature
  differs. Internally it reuses `trf`'s Coleman–Li reflective machinery unchanged by feeding
  `(g, H)` through a ridge-regularised pseudo-Jacobian, so on a Gaussian least-squares fit it reduces
  to `trf`'s step exactly. It runs natively in the distributed propose/score loop (picklable, no
  `run()` override, concurrent `N`-start multi-start, registered as a box-start refiner) like every
  other `fit_type`. New config keys `gntr_grad_tol` (1e-8), `gntr_step_tol` (1e-8), `gntr_ridge`
  (1e-10), and the runtime-guarded `gntr_max_iterations`. This cut supports an estimated-σ Gaussian
  (`chi_sq_dynamic`), a fixed-scale Laplace, a fixed-dispersion mean-centered negative-binomial, and
  a Gaussian fit with static-hinge constraints; the coupled corners it cannot yet build the Fisher
  Hessian for (a mean-on-log estimated scale, a free-dispersion / median count family, an estimated
  Student-t df, or an estimated constraint scale) refuse with a pointer to `lbfgs`, which fits them.
  `trf` / `lbfgs` are byte-identical (they never form the Hessian).
- **Kalman-inspired DREAM proposal: `proposal = kalman` (ADR-0067, Stage 3; DREAM(KZS), #358).**
  The third proposal operator on the unified DREAM engine (Zhang, Vrugt et al. 2020). During a
  burn-in window each proposal is steered toward the data by a Kalman gain `K = C_ZY (C_YY + R)⁻¹`
  built from the archive's parameter↔model-output cross-covariance, with the innovation `d - f(xᵢ) +
  ε` taken at the chain's current state (`ε ~ N(0, R)`), which accelerates burn-in on informative,
  mildly non-linear problems; after the window the chain reverts to `de` for a reversible sampling
  phase (the Kalman jump breaks detailed balance by design, so its samples are burn-in and
  discarded). The gain reads each archive entry's *model output vector* `f(Z)` — surfaced by the new
  `LikelihoodObjective.aligned_prediction_data` seam and carried in an output-augmented archive that
  turns on only for this proposal (the "implied axis 2b"; dormant and byte-identical for `de` /
  `whitened`). `kalman` requires a linear-scale Gaussian likelihood (`chi_sq` / `chi_sq_dynamic`,
  the source of `R = diag(σ²)`) and `n_try = 1`, and refuses any other objective or `n_try > 1`
  *before the run starts*. The internal ensemble size is fixed (`M = 20`, clamped to the available
  archive, falling back to `de` before enough outputs accrue — no new user key); one new
  proposal-scoped key `kalman_burnin_frac` (default `0.3`) sets the window as a fraction of
  `burn_in`. Validated end-to-end against a closed-form linear-Gaussian posterior (`f(x) = A x`
  scored by real `chi_sq`), plus pinned gain-math and burn-in-switch unit tests.
- **Multi-Try DREAM: the `n_try` count (ADR-0067, Stage 2; MT-DREAM(ZS), #357).** A new integer
  `n_try` config key turns each chain-generation into a multiple-try step (Liu, Liang & Wong 2000;
  Laloy & Vrugt 2012): with `n_try = k > 1` a chain draws `k` candidate proposals, selects one in
  proportion to its posterior importance weight, and accepts it over the current state with a
  multiple-try Metropolis ratio evaluated against a `k - 1`-point reference set drawn from the
  winner plus the current state (`2k - 1` evaluations per chain per generation). Multiple tries per
  generation raise the per-generation acceptance rate and help parameter-rich / strongly correlated
  posteriors mix. It is the second orthogonal axis of ADR-0067 and **composes with every `proposal`
  value** (`de`, `whitened`) and with the snooker update — MT-DREAM(ZS) is literally multi-try
  parallel-DE. `n_try = 1` (the default) is the classic single-try engine and is **byte-identical**
  to before (verified against the DREAM/P-DREAM oracle suites and the effective-config goldens; the
  only change is the additive `n_try` key). The snooker proposal is non-symmetric, so under
  multi-try its candidate and reference weights carry the ter Braak & Vrugt (2008) Jacobian
  `||p - z||^(d-1)`; the current-state reference slot uses the current state's distance to the
  **selected candidate's** anchor — the unique choice that reduces to the published single-try
  snooker ratio at `k = 1` (derived from first principles and confirmed by a stationary-distribution
  test; both the DREAM-Suite and PyDREAM reference implementations differ on this term). The
  multiple-try acceptance is validated to preserve a known Gaussian target with the snooker update
  active.
- **DREAM `proposal` operator key; P-DREAM folded into one DREAM engine (ADR-0067, Stage 1).**
  DREAM(ZS) and Preconditioned DREAM are now one `DreamAlgorithm` engine selected by a new
  `proposal` config key: `proposal = de` (default) is the classic parallel-direction proposal, and
  `proposal = whitened` is the covariance-preconditioned proposal that used to be a separate
  algorithm. The `p_dream` job type is unchanged for users — it is simply `dream` with
  `proposal = whitened` pinned — and `whitened` can now also be requested explicitly on a `dream`
  run. This is a pure refactor: `dream` at defaults and `p_dream` are **byte-identical** to before
  (verified against the existing DREAM/P-DREAM oracle suites and the effective-config goldens). It
  is the first step of ADR-0067's unification of the DREAM family into two orthogonal axes
  (`proposal` × `n_try`), which will absorb the requested MT-DREAM (#357) and DREAM-KZS (#358)
  without new sampler subclasses.
- **Composable floor normalization + analytic per-series scaling for relative / arbitrary-unit
  data (#479).** Two composable, per-series normalization primitives so a log/relative objective
  on arbitrary-unit data (fluorescence, blots) can be spelled with standard tokens instead of a
  bespoke objective class. (1) **`floor <rho>`** — an additive measurement-noise floor
  `x' = x + rho*max(x)` (default `rho = 0.03`) applied **identically to the simulated and the
  experimental** column, so a log objective stays finite where a series legitimately touches zero.
  (2) **`scale`** — analytic per-series **optimal** multiplicative scaling profiled out at scoring
  time (hierarchical / profiled scaling; Weber et al. 2011, Loos et al. 2018), family-appropriate:
  the geometric-mean ratio for a log family (`lognormal`) and the least-squares optimum
  `c* = Σ w s d / Σ w s²` for a linear one — so an overall model-vs-data scale difference is not
  penalized (no per-series `scale` parameter needed). They compose as an ordered chain
  (`normalization <obs> = floor 0.03, scale`, per-observable / `<exp>.<obs>` / whole-fit), and
  together with `objective = lognormal` spell the exact sum-of-squared-log-differences-of-
  geometric-mean-normalized-trajectories objective of Jaruszewicz-Błońska et al.
  (*PLoS ONE* 2023; 18(6):e0286416). Legacy `normalization = peak` / `normalization x = peak`
  round-trip byte-identically; `peak`/`init`/`zero`/`unit` stay sim-only. The `peak`, `unit`, and
  `floor` column reductions are **NaN-aware** (`np.nanmax`/`nanargmax`, etc.), so a sparse
  multi-observable target — NaN in the rows where a given observable is unmeasured — is reduced
  over its measured points only rather than collapsing the whole column to NaN (which had
  silently zeroed the objective); a dense column is byte-identical. Both new primitives have
  a **deferred gradient** (they raise `GradientNotSupported`, so a gradient fit falls back to a
  gradient-free step; the motivating fits are evolutionary), and both are **refused on PEtab
  export** (a whole-trajectory reduction has no pointwise PEtab v2 operator; `scale`'s
  `observableParameters` mapping is a future direction). See ADR-0066.
- **Gradient-based fitting extends to `parameter_scan` (dose-response) objectives (#476).**
  A gradient fit (`fit_type = trf`/`lbfgs`) can now target a dose-response objective, not
  just a time course. The default dose-response path already computed the per-dose forward
  sensitivities `∂obs(dose)/∂θ` — one sensitivity-configured ODE `run()` per swept dose —
  and then discarded them at row assembly; PyBNF now stacks those per-point final-row
  sensitivities down the dose axis into the scan `Data`, so the existing gradient assembly
  produces `d(objective)/dθ` for dose-response fits. The swept dose is the data's independent
  variable (not a fitted parameter), so the per-dose sensitivity is well-posed and consumed
  exactly as a time-course row is. Supported for the **reset-to-seed** strategies — the
  parity / integrate-to-steady-state default and the independent fixed-time scan — on both
  the native BNGL and SBML/Antimony backends. Newton/KINSOL (`ss_method=>"newton"`, now
  supported — see #478 below), continuation/bifurcate (`reset_conc=>0`),
  `method=>"protocol"`, and carried-state (pre-equilibration, ADR-0062) scans refuse cleanly
  on the gradient path with an actionable message (an *incidental*, unscored scan of the same
  shape still runs sensitivity-free, #475). The scalar (metaheuristic) path is byte-identical.
  See ADR-0064.
- **Scored Newton/KINSOL (`ss_method=>"newton"`) steady-state dose-response scans are now
  differentiable (#478).** The KINSOL accelerator solves each dose point's steady state as an
  algebraic `f(x)=0` (no forward-sensitivity *integration*), so #476 (ADR-0064) kept a
  *scored* Newton scan gradient-free and pointed at the parity default. It is now a real
  speed win under a gradient fit: the KINSOL solve returns `dY_ss/dp` **exactly** (the
  implicit-function-theorem derivative on the analytical Jacobian, not a finite difference),
  and bngsim ≥ 0.11.35 (lanl/bngsim#12) maps it through the observable/function Jacobian
  `∂g/∂x` and exposes it as `SteadyStateResult.output_sensitivities`, mirroring the CVODE
  `Result`. PyBNF stacks those per-dose slices down the dose axis exactly as the parity path
  does — no gradient-assembly change. On the gradient path the scan runs sequentially (the
  KINSOL sensitivity solve is kept off the thread pool) and the KINSOL→CVODE non-convergence
  fallback is itself differentiable and consistent with the converged path. **Requires
  bngsim ≥ 0.11.35**; a build lacking the accessor refuses a scored Newton scan cleanly with
  an upgrade hint (a scalar Newton scan is unaffected). Continuation/bifurcate,
  `method=>"protocol"`, and carried-state scans still refuse on the gradient path. See
  ADR-0065.
- **Edition-2 preincubate → wash → dose-response scan protocol (#474).** The new-era
  `experiment:`/`condition:` surface now expresses the full **equilibrate → intervene →
  measure a dose-response** protocol, so a published fit that needs it (the Erickson-2019
  IGF1R competition/dissociation fit — 7 rate constants to 3 datasets, two of them a
  2 h-preincubate → wash → cold-competition scan) runs in `edition = 2` with **no in-model
  actions block**. Two capabilities: (A) a `parameter_scan` may be the measured phase of a
  `preequilibrate:` experiment — the synthesizer emits `saveConcentrations()` +
  `parameter_scan(… reset_conc=>1)` so each dose resets to the carried post-intervention
  state; (B) a `condition:`'s `perturbations:` accepts a **quoted BNGL species pattern**
  target with a number *or a parameter-expression* value — a species `setConcentration`
  (a wash `"IGF1(ds,hs,label~hot)" = 0`, or a dose-tracking bolus
  `"IGF1(ds,hs,label~cold)" = IGF1_cold_conc*(NA*Vecf)`), vs. a parameter `setParameter`.
  The bngsim backend routes such a carried-state scan to its native
  reset-conc-to-snapshot `parameter_scan`/`bifurcate` (**requires bngsim ≥ 0.11.34**,
  lanl/bngsim#11), reproducing BNG2.pl exactly; the fresh-from-seed dose-response paths
  (ADR-0046) are unchanged. See ADR-0062.
- **PEtab v2 export/import of the preincubate → wash → dose-scan protocol (#477).** The two
  shapes ADR-0062 added to the edition-2 fitter now export to PEtab v2, import back, and
  round-trip byte-for-byte (validated by petab's full `default_validation_tasks`): (1) a
  **species `setConcentration`** condition target — a BNGL species pattern is not a valid
  PEtab id, so it is aliased through the **mapping table** (`petabEntityId` → the pattern) and
  the condition targets the synthesized `species_<…>` id with a number or a parameter-expression
  value; (2) a **pre-equilibrated dose-response** — each dose becomes a two-period Experiment
  (a `time = -inf` pre-equilibration period + a measurement period applying both the shared wash
  condition and a per-dose swept-parameter condition), the combination of ADR-0052 and ADR-0046.
  The exporter's previous "deferred" refusals are lifted. The surrogate split × a pre-equilibrated
  scan (an empty surrogate set M is required) and a whole-fit `normalization` transform (the real
  Erickson-2019 IGF1R job) stay out of scope, raised in code. See ADR-0063.
- **`examples/real-world/` — the 2019 PyBNF-paper case studies on the edition-2 surface.**
  The biological models from Mitra et al. (iScience 2019) — Kozer's EGFR (ODE and
  network-free), the ligand/receptor model (ODE and NFsim), IGF1R competition binding,
  the FcεRI γ-chain SSA network, and the trivalent-ligand aggregation model — re-expressed
  on the new-era `experiment:`/`condition:`/`data:` config surface, spanning the three
  simulator paths (deterministic ODE, Gillespie SSA, network-free NFsim). These validate
  PyBNF's bngsim-backed default path on representative, paper-scale models (issue #380),
  with `tests/test_real_world_examples.py` running a backend-free well-formedness tier in
  default CI and a real-bngsim end-to-end tier under `-m recovery`.
- **Network-free (NFsim) options on the edition-2 experiment surface.** A `method: nf`
  experiment now accepts `gml:` (global molecule limit) and `complex:` (track molecular
  complexes) — the network-free counterparts of `atol`/`rtol` — carried into the synthesized
  NFsim `simulate`/`parameter_scan` so a large aggregating model (e.g. the EGFR clustering
  fit) can raise its molecule limit and track complexes as its classic hand-written action did.
- **Fixed-time NF pre-equilibration (`equil_t_end:`).** Edition-2 pre-equilibration (ADR-0052)
  equilibrates *to steady state*, but NFsim has no steady-state solve. A `method: nf`
  pre-equilibration now takes `equil_t_end: <time>` and runs its (unmeasured) equilibration
  phase for that fixed duration instead; omitting it on the NF path is a clear config-time
  error rather than an unbounded run. ODE/SSA pre-equilibration is unchanged (still
  steady-state); any method may opt into a fixed-time equilibration with the field.

### Changed
- **A local fit now runs one single-threaded worker per core, whether or not `parallel_count` is
  set (#526, ADR-0089).** PyBNF built its local Dask client two different ways: with
  `parallel_count` set it pinned one thread per worker process, and without it took Dask's bare
  `Client()`, whose workers are multi-threaded on any machine with more than four cores — so
  whether two jobs could run concurrently *inside one process* depended on an unrelated key. The
  simulation backends hold process-wide state that is not thread-safe (both `dask-ssh` paths
  already pass `--nthreads 1`, and a scored Newton scan is already routed sequentially for the
  same reason), and issue #525 caught the consequence: with no `parallel_count`, concurrent
  worker threads raced on bngsim's cached sympy→C printer and a `trf` fit aborted before its
  first start, while `parallel_count = 4` — nothing else changed — completed every iteration.
  A/B on that job (8 concurrent `trf` starts, no `parallel_count`, three runs per side): the old
  default failed 3/3 times, each time losing the forward-sensitivity column of a different scored
  observable; the new default completed 3/3 with no dropped column.
  Both local branches are now one branch that always pins `threads_per_worker=1`;
  `parallel_count` chooses the number of worker *processes* and nothing else. Total concurrency
  is unchanged (a 6-core machine goes from 3 workers × 2 threads to 6 workers × 1 thread), but a
  default run now uses more processes, so it holds more model copies — lower `parallel_count` to
  reduce memory. Runs that already set `parallel_count`, and all cluster runs, are unaffected.
- **`gntr` now assembles each scored forward-sensitivity row once per objective evaluation
  (#488).** Its scalar-gradient and expected-Fisher calculations previously repeated the same
  experiment/row/column walk and rebuilt `d(prediction)/d(theta)` independently. A shared scored-
  point iterator now feeds both accumulators in one pass; standalone gradient-only (`trf` /
  `lbfgs`) and Fisher-Hessian assembly APIs retain their existing results.
- **Edition-2 one-model + `condition:` jobs now simulate only the scored `(experiment,
  condition)` diagonal, not the full `{action} × {condition}` cross-product (#484, ADR-0069).**
  Under edition-2 Mechanism A the single model ran every synthesized action under every
  `condition:` mutant, but only each experiment under *its own* condition is ever scored — so for
  **N** experiments and **M** conditions each objective evaluation ran **N×(M+1)** simulations to
  obtain **N** scored series and discarded the rest (e.g. the Miller et al. 2026 MEK-isoform job:
  25 simulations for 5 series). PyBNF now records a per-model **emit-set** — the full output
  suffixes any consumer reads (the scored `exp_data` diagonal ∪ constraint homes/references ∪
  postprocessing targets) — and each bngsim backend's `execute` skips every `(action, condition)`
  pair not in it, so the cost is **N** simulations. Results are unchanged (only unscored pairs are
  removed; the scored objective is provably invariant), and pruning is gated on edition ≥ 2 and
  action *separability* (no hand-written `begin actions` block mixed in), so legacy `mutant:`,
  non-edition-2, and mixed-action jobs — and the BNG2.pl / `.net` subprocess paths — are
  byte-identical. A consumer that references a pair no experiment produces is now a load-time
  `PybnfError` rather than a silent drop. This makes the #483 `am` `output_trajectory` write-guard
  defensive (off-diagonal suffixes are no longer produced). New tutorial lesson
  `47_condition_perturbations` is the reference Mechanism-A example.

### Fixed
- **PEtab import now reads a `problem.yaml` whose table-file lists are unindented (a column-0
  `- item`), the shape the official `petab.v2.petab1to2` converter emits (#407).** An externally
  authored v2 problem — e.g. any `Benchmark-Models-PEtab` problem converted from v1 — imported as
  `problem.yaml ... has no parameter_files`, because `read_problem_yaml`'s dependency-free
  hand-rolled scan treated a column-0 list item as a new top-level key and dropped it; only the
  two-space-indented list shape our own exporter writes was read. The section reset is now guarded
  on `not stripped.startswith('-')`, so a column-0 `- item` appends to the current section — making
  the reader a strict superset of both list shapes (our indented output and petab's unindented
  output parse identically). Two regression tests cover the `petab1to2` shape and the two-shape
  equivalence.
- **Adaptive MCMC (`am`) with `output_trajectory` no longer crashes on edition-2 one-model +
  `condition:` jobs (`KeyError` on `<action><mutant>` suffixes) (#483).** An `am` job that saves
  posterior-predictive trajectories (`output_trajectory`) over an edition-2 model with `condition:`
  perturbations bound to two or more experiments aborted on the first accepted sample with e.g.
  `KeyError: 'WTn78gMEK_pRDS'`, leaving `samples.txt` with only its header and
  `constraint_samples.txt` empty. The sampler allocates one trajectory buffer per *scored* data-key
  (`time_length`: the diagonal `WT`, `KOko`, …), but `got_result` wrote every raw simulation suffix
  it saw. Under edition-2 Mechanism A (one model + `condition:` mutants) the single model runs every
  action suffix under every condition-mutant, so `res.out[model]` yields the full `{action} ×
  {mutant}` cross-product — the first off-diagonal suffix (`WTn78g`) hit an unallocated buffer. Both
  the `output_trajectory` and `output_noise_trajectory` write blocks now skip any suffix that was not
  allocated (i.e. is not a scored data-key), writing only the scored diagonal. The `de` path on the
  same model/condition/experiment setup was unaffected. Surfaced on the Miller et al. 2026 MEK-isoform
  qualitative-constraint + Bayesian-UQ job ported to edition-2 as one model + four `condition:` cell
  lines; sibling to the edition-2/aMCMC cross-model fix in #480.
- **Adaptive MCMC (`am`) no longer crashes with `burn_in = 1` (`ValueError: no field of name
  <parameter>`).** The adaptive-covariance seed file `params_<chain>.txt` was given its
  column-name header only on the `iteration == burn_in - 1` write, which is unreachable when
  `burn_in = 1` — the iteration counter is already incremented to ≥ 1 by the time the seed rows
  are written, so `burn_in - 1 == 0` never matched and the file was left headerless. The
  `np.genfromtxt(..., names=True)` seed read at `iteration == burn_in + adaptive` then consumed
  the first data row as the header and failed on the first parameter name. The header is now
  emitted when the seed file is first created, independent of `burn_in`; output for
  `burn_in ≥ 2` is unchanged. Surfaced while verifying the #480 fix on the MEK aMCMC example.
- **Bayesian samplers no longer crash on the first accepted move when `.prop` constraints
  are attached (#480).** An adaptive-MCMC / MCMC / DREAM job (`fit_type = am`/`mh`/`dream`)
  that carries constraints aborted on the first accepted sample with
  `TypeError: 'NoneType' object is not iterable`, most visibly with cross-model dotted
  references (`WT.obs at time=t < KO.obs at time=t`). The per-sample constraint-satisfaction
  bookkeeping read the accepted `Result.simdata`, but the default worker-scoring path
  (`local_objective_eval=0` with `parallelize_models=1`) nulls `res.simdata` after scoring
  and moves the full multi-model dict to `res.out` — so `Constraint.penalty()` received
  `None` and cross-model suffix resolution iterated it. The samplers now resolve the
  simulation data through a `_result_simdata` helper that reads `res.out` on the
  worker-scoring path and `res.simdata` on the master-scoring path (`parallelize_models>1`),
  and `evaluate_constraints` guards a `None` dict into a graceful skip rather than a hard
  crash. The same fix restores the pointwise-log-likelihood (LOO/WAIC) sidecar, which was
  silently empty on the worker-scoring path for the same reason. Regression vs v1.1.9;
  verified end-to-end on the 5-model, 90-cross-model-constraint
  `examples/Miller2025_MEK_Isoforms/MEK_isoform_aMCMC` job — the sampler now draws posterior
  samples and writes per-sample constraint-satisfaction rows (`samples.txt`,
  `constraint_samples.txt`, `constraint_satisfaction_*.txt`), where it previously aborted
  before drawing a single sample.
- **Gradient fits no longer abort on an incidental non-differentiable action (#475).** A
  gradient-based fit (`fit_type = trf`/`lbfgs`) enables a forward-sensitivity request on the
  whole model, and any action that cannot carry sensitivities forward — a stochastic (`ssa`/
  `nfsim`) diagnostic `simulate`, or a carried-state pre-equilibration `parameter_scan` (#474)
  — used to abort the **entire** fit, even when that action's output is never scored against
  data. The two guards (`_sensitivity_request_kwargs`, `_scan_carried_state`) now gate on
  whether the action's output is a *scored* gradient target: a scored non-ODE / carried-state
  action still refuses cleanly (its gradient genuinely cannot be supplied), while an
  incidental/unscored one runs on the ordinary sensitivity-free path. The gradient optimizer
  declares each model's scored suffixes (from `exp_data`) in `_setup_gradient_path`, keyed
  per-instance by the mutant/condition suffix; ODE actions stay always-bearing so the
  persistent-simulator sensitivity continuity across carried states (#457) is untouched.
- **Edition-2 network-free (NFsim) experiments now run through the bngsim bridge.** A
  `method: nf` experiment synthesized an action set that (a) began with
  `resetConcentrations()` — which the bngsim NF bridge rejects (NFsim re-seeds each run, so
  it is a no-op) — and (b) forced `generates_network=True`, sending a network-*free* model
  (whose reaction network is unbounded) down the network-generation path. Both are now
  suppressed on the NF path, so an edition-2 NF experiment classifies as the NF bridge and
  routes to `writeXML` → `BngsimNfModel`, matching a hand-written NF actions block. Surfaced
  by the new `examples/real-world/` NFsim examples (#380); ODE/SSA synthesis is unchanged.

## [v1.6.0] - 2026-07-05

### Added
- **`qualitative_loss` selector and logit penalty model** (ADR-0060) — a new logit
  (softplus) qualitative-constraint penalty completes the hinge/probit/logit family,
  and a global `qualitative_loss = {auto|hinge|probit|logit}` config key re-runs a
  `.prop` set under any one family, coercing every constraint to it through a shared
  scale currency (a family authored in its own model round-trips to identity). The
  logit gradient rides the existing constraint-gradient path (no assembly change).
- **Estimable qualitative-constraint scale** (ADR-0061) — `qualitative_scale = fit <param>`
  promotes the logit scale (`s`) / probit tolerance (`σ`) from a fixed authored value
  to a fittable free parameter estimated jointly with the model parameters, globally
  tied across all qualitative constraints (one nuisance parameter, the identifiable
  case). Includes its closed-form `d(penalty)/d(scale)` contribution on the scalar
  gradient path, mirroring the estimated-noise pattern.
- **Online documentation on GitHub Pages** — <https://lanl.github.io/PyBNF/>, built
  from the Sphinx sources and deployed via GitHub Actions (interim host while Read the
  Docs access is provisioned). New pronghorn logo/favicon and a populated
  `pybnf.algorithms` API reference.

### Fixed
- **Packaging: the built wheel is now PyPI-uploadable.** The `tests` extra pinned petab
  to a `git+` URL, which setuptools wrote into the wheel's `Requires-Dist`; PyPI rejects
  any upload whose metadata carries a direct (`git+`) reference. Reverted the extra to
  stock `petab>=0.8,<1` — the sdist and wheel now pass `twine check` with zero direct
  references. The CI native-BNGL oracle still gets the fork via the `setup-pybnf`
  action's input, so there is no test-coverage impact.
- Corrected typos in the fatal-error (`CancelledError`) message, and migrated the
  in-code documentation links from readthedocs.io to the GitHub Pages URL.

### Changed
- The Sphinx documentation build is warning-clean and now enforced with `-W`
  (warnings-as-errors) in CI, so a malformed docstring or RST fails the docs job.

## [v1.5.0] - 2026-07-05

### Added
- **HMC / NUTS reference sampler** (`job_type = hmc`, ADR-0059, closing #425's sampler item) — a gradient-based No-U-Turn sampler (blackjax NUTS) for differentiable targets, the reference against which PyBNF's simulator-path samplers are judged. It runs only on the analytical/bring-your-own surface (the named analytical menu, `objective = expression`, and the full 16-family prior set), which it lowers to JAX: an `expression` NLL goes sympy→JAX, every prior family gains a `logpdf_jax`, and a log-scaled or bounded parameter is mapped to an unconstrained space through an unconstraining bijection so NUTS samples on ℝⁿ and transforms back. `jax`/`blackjax` are an optional extra, lazily imported (core stays dependency-free), and a *simulator* model is rejected with a pointed error — HMC is deliberately analytical-only, not a general fitter. New banana / multimodal / rotated-quartic stress geometries exercise the sampler. (#425, ADR-0059)
- **Bring-your-own & analytical objectives** (ADR-0050, closing #425's objective item) — three fileless ways to name the objective directly in the `.conf`, with no `.bngl` model and no `.exp` data required. **`objective = expression`** + `expression = <PEtab-math>` compiles an inline negative-log-likelihood over the declared free parameters (bind-by-name; PEtab arithmetic, so `^` not `**`) down to a numpy callable that is also fully HMC-differentiable via the sympy→JAX path. **`objective = callable`** + `callable = module:func` (or `file.py:func`) hands scoring to an arbitrary Python function (gradient-free). Both bind experimental data with **`data = f.exp, …`**: a callable receives a `{name: Data}` map, while a data-bound `expression` is evaluated per observation over the `.exp` columns and summed (still differentiable end to end). A **named analytical menu** — `objective = banana, a=1, b=100` and its siblings (rosenbrock, rotated quartic, …) — supplies closed-form test targets inline with their coordinates bound by name and no `.target` sidecar file. Documented in the new `docs/analytical_objectives.rst`. (#425, ADR-0050)
- **Gradient-based optimizers** (`job_type = trf` / `lbfgs`, #386) — two deterministic local optimizers that consume the new analytic objective gradient (#385): **`trf`**, a Trust-Region-Reflective / Levenberg–Marquardt least-squares solver over the residual Jacobian with proper bound handling (#460), and **`lbfgs`**, a full L-BFGS-B (generalized Cauchy point + subspace minimization) over the scalar objective. Both support box-sampled concurrent multi-start (keep-best), a discrete-events pre-flight gate that refuses a model whose gradient would be wrong (#461), and settable-from-`.conf` tunables (the `trf` / `lbfgs` / `powell_line_tol` knobs are now registered). (#386, #385)
- **Analytic objective-gradient engine** (`pybnf/gradient/`, #385) — the sensitivity-and-gradient infrastructure the gradient optimizers (#386) and standalone profile likelihood (#446) stand on. Forward output-sensitivity tensors are preserved through net execution (#447) with free parameters routed to bngsim's `sensitivity_params` / `sensitivity_ic` (#448) and assembled into the residual Jacobian and scalar objective gradient (#449). Coverage spans the whole objective surface: estimated-σ noise-scale columns (#451), log/lognormal scale (#452), trajectory-transform + normalization (#453), asymmetric/non-Gaussian families (Laplace, Student-t, mean-vs-median centering; #454, plus the MEAN-on-log-scale offset/noise coupling), constraint and qualitative/comparison penalties (#456), the SBML/Antimony measurement-model seam (#455), pre-equilibration/steady-state sensitivity continuity (#457), and the negative-binomial noise gradient via median CDF-inversion implicit differentiation (#458) — plus a Student-t exact sqrt-loss residual for LM/TRF (#459). (#385)
- **Standalone `profile_likelihood` job type** (`job_type = profile_likelihood`, #446) — likelihood-profile identifiability analysis as a first-class run: for each parameter it profiles the objective along a fixed grid, re-optimizing the others at each point through an exact inner path *and* an L-BFGS-B inner path (so it also profiles non-exact objectives). It emits profile plots and resumable per-point state, and parallelizes across parameters. (#446)
- **SBML/Antimony assignment-rule observables** (#463–#465) — the measurement-model layer now resolves an observable defined by an SBML assignment rule by inlining the rule's right-hand side (#465), routes Antimony (`.ant`) models through the SBML formula namespace (#463), and excludes assignment-rule variables from that namespace so they don't shadow species (#464). (#463, #464, #465)
- **AIC / BIC / AICc for likelihood fits** — a fit scored with a proper (normalized) likelihood objective now reports the information criteria for the best-fit parameter set: `AIC = 2k − 2·lnL`, `BIC = k·ln n − 2·lnL`, and `AICc = AIC + 2k(k+1)/(n−k−1)` (the last reported `n/a` when `n ≤ k+1`). Because the reported log-likelihood is the full normalized density (the same gate LOO/WAIC use, ADR-0056), the AIC is an **absolute** value comparable across noise families and data sets — the first-class form of the model-selection arithmetic the tutorial computes by hand.
- **Native BNGL PEtab-loader hardening** (#437, #420) — the dependency-free BNGL reader behind PyBNF's PEtab lint/import path now joins line continuations, strips BNGL line labels (indexed and named), and drops the `molecules`/`rules` block aliases so it matches BNG2.pl exactly, pinned by a corpus regression gate that differentials the reader against BNG2.pl; a CI leg now runs the native loader. (#437, #420)
- **Worked-example tutorial series + interactive notebooks** — a 46-lesson tutorial catalog spanning the toolbox (optimizer bake-offs and local-optimizer contrasts, Bayesian uncertainty and the full sampler family, robust/count/relative/lognormal noise models, per-observable and column-joint profile objectives, PEtab v2 round-trips and the lint clinic, gradient fitting and profile-likelihood identifiability, checkpoint/resume, model selection, and BPSL model checking) plus an interactive Jupyter notebook collection for PyBNF + bngsim.
- **Student-t (robust-regression) noise family** (ADR-0058, closing the second half of #438 item 1 — item 1 is now fully done) — `noise_model = student_t, sigma = <source>[, df = <source>]` gives the heavy-tailed, outlier-robust observation likelihood: a `normal` with a tail-heaviness knob `df` (degrees of freedom), where small `df` produces fat tails that downweight outliers and `df → ∞` recovers the Gaussian (the noise analogue of the robust `student_t` *prior* from ADR-0057, and Stan's/PyMC's `student_t(ν, μ, σ)`). It is the **first two-parameter noise family**: `sigma` (scale) and `df` (shape) are each **independently sourced** (`fix_at` a constant or `fit` a free parameter), so a fit may estimate 0, 1, or 2 noise parameters — a fixed-`df` robust fit (just `sigma` free), both free, or anything between. `df` is the one parameter that may be **omitted**, defaulting to a fixed `4` (the standard Stan/Gelman robust default), so `noise_model = student_t, sigma = fit s__FREE` is a complete robust fit; estimating `df` is weakly identified, so pair `df = fit nu__FREE` with a positive prior on it (the `gamma`/`half_*` families ADR-0057 added compose exactly here). The per-point NLL is `scipy.stats.t.logpdf`-exact: `data_fit = (ν+1)/2·log(1 + z²/ν)`, the `log σ` normalizer summed iff `sigma` is estimated, and the `df`-block (`−logΓ((ν+1)/2) + logΓ(ν/2) + ½log(νπ)`) summed iff `df` is estimated — when `df` is fixed that block is a constant the sampler drops, when `df` is free it is the term that keeps the fit honest, and either way it rides into `log_density` so LOO/WAIC (ADR-0056) see the complete normalized density. **The noise engine generalized to source a *mapping* of noise parameters** (was exactly one): a spec is now `(family, {param: source})`, each family declares its own `noise_params` (retiring the engine's parallel name table) plus a per-parameter `param_normalizers` keyed by name, and the objective gates each parameter's normalizer on its own source's estimated-ness — a backward-compatible extension (the single-parameter families gain only a declared name and an ignored trailing argument; their scores are byte-identical) mirroring ADR-0057's trailing-`p3` prior-carrier extension. Student-t is exposed only through the `noise_model` surface (no `objective = student_t` token), on the linear scale (it is symmetric there, so `location = mean` and `median` coincide); on a log scale it has *no finite mean* (its tails are too heavy for any `df`), so `location = mean` there raises and only `median` is safe. PEtab v2 has no Student-t `noiseDistribution`, so the family is PyBNF-native and not part of the PEtab round-trip. (#438, ADR-0058)
- **Eight new prior families + a three-parameter prior carrier** (ADR-0057, closing the prior-family half of #438 item 1) — the batch univariate priors Bayesian modelers reach for, each a `scipy.stats`-backed leaf in `pybnf/priors/` that self-registers and gets its `{base}_var` / `log{base}_var` / `ln{base}_var` keywords for free (ADR-0010): **`half_normal`** / **`half_cauchy`** (the standard weakly-informative *scale* priors, one parameter — the underlying scale), **`beta`** (bounded `[0,1]`, for fractions/probabilities), **`inv_gamma`** (the conjugate variance prior), **`weibull`** (lifetime/time-to-event), **`gumbel`** (extreme-value/max), **`logistic`** (a heavier-tailed Normal sibling), and **`student_t`** (the heavy-tailed *robust* prior — a drop-in for a Normal prior that tolerates outliers). Seven are pure two-or-fewer-parameter leaves usable on both the legacy positional `*_var` line and the new-era `parameter:` record (e.g. `half_normal_var = k__FREE 2` or `parameter: k, prior: beta, alpha: 2, beta: 5`); each is oracled against its scipy distribution and inherits the log/ln scale forms and one-/two-sided truncation (ADR-0022/0047) automatically. **`student_t` is the exception and the design content:** a useful Student-t prior wants *three* numbers (`df`, `location`, `scale` — the same knobs as Stan's/PyMC's `student_t`), but PyBNF's prior carrier was hardwired to two (`p1`/`p2`). The carrier now extends to a trailing `p3` (`FreeParameter`, `build_prior`, every `Prior.build`, the record builder), threaded through `set_value`/`__eq__`, with no behavior change for any existing family (the third slot defaults to `None`). Because a three-token positional `*_var` value already means a bounded box plus its `b`/`u` flag, **`student_t` (any `n_params >= 3` family) is authored only through the new-era `parameter:` record** — `parameter: x, prior: student_t, df: 4, location: 0, scale: 2.5` — and is omitted from the positional grammar (`var_keyword_grammar`), staying in the keyword map so the record path resolves it. A family's own `scale` field is distinct from the record's `parameter_scale` transform, so the two never collide. The student_t *noise* family (the other bullet of #438 item 1) and the architectural non-goals (multivariate/joint priors, constrained parameter types) remain out of scope. (#438, ADR-0057)
- **LOO/WAIC via a `log_likelihood` group** (ADR-0056, closing #438 item 4) — a Bayesian fit done with `output_inference_data` and a per-point likelihood objfunc (`chi_sq` / `chi_sq_dynamic` / `lognormal` / `laplace` / `neg_bin` / `neg_bin_dynamic`, or the `objective` / `noise_model` surface) now gets a `log_likelihood` group in its `InferenceData`, so `az.loo` (PSIS-LOO-CV) / `az.waic` / `az.compare` work directly — model comparison from pointwise log-likelihoods, the Stan `loo` ecosystem, for the cost of not discarding a decomposition PyBNF already computes. The pointwise log-likelihoods are **recorded during the run, not re-simulated**: at each accepted draw the per-observation densities are a cheap re-walk of the simulation already in hand (zero extra simulations; mirrors how constraint satisfaction is cached at accept and written at each sample), streamed to a new `Results/log_likelihood.txt` sidecar **row-aligned** with `samples.txt` (one row per saved sample), which the bridge reads in lockstep to build the group (variable `y` over a labelled `obs_id` axis, dims `chain × draw × obs`). The recorded values are the noise family's **complete, normalized, unweighted** per-point log-density (`scipy.stats`-exact: `norm`/`lognorm`/`laplace.logpdf`, `nbinom.logpmf`) — a new `NoiseModel.log_density` restores the parameter-independent constants the objective legitimately drops for sampling (Gaussian's `½log(2π)`, and a log-scale family's change-of-variables Jacobian via the scale's `log_abs_dforward`), so absolute WAIC/LOO and *cross-family* comparison are correct, not only same-family deltas; weights are a fitting device, not part of the generative likelihood, so they are excluded. The values therefore do **not** sum back to `-score` — they are the honest densities `az.loo` needs. Gating reuses the **one** key `output_inference_data` (no new key); with a non-likelihood objfunc (least-squares, `kl`, `direct_pass`, …) there is no normalized density, so the recorder is a no-op, the group is omitted, and a one-time note explains LOO/WAIC needs a likelihood objfunc. Note arviz 1.x dropped top-level `az.waic` (PSIS-LOO supersedes it) — the group powers `az.loo`/`az.compare` on both arviz lines and `az.waic` wherever a user's arviz still ships it. A run finished without the sidecar (key off, or pre-existing) simply yields no group; LOO is offered exactly where the data for it exists. `prior` / `observed_data` remain deferred. (#438, ADR-0056)
- **ArviZ `InferenceData` bridge** (ADR-0055, closing #438 item 3) — a documented `pybnf.inference_data.from_pybnf(source)` maps a finished MCMC run's saved samples onto an [ArviZ](https://python.arviz.org) `InferenceData`, so PyBNF's posterior output (`am`/`dream`/`p_dream`/`pt`/`mh`) becomes first-class in the ArviZ / bayesplot / loo ecosystem — trace/rank/forest/pair plots, `az.summary`, `az.compare` — for nearly free (PyBNF already runs the samplers, writes `Results/samples.txt`, and computes rank-normalized split-R-hat + bulk/tail ESS; this is a *format bridge*, not new statistics). `source` is a `Results/` directory, an output directory, or a `samples.txt` file: the chain×draw shape is recovered from the `iter<draw>run<chain>` sample names (chains = the distinct `run<c>` count = `population_size`), the `posterior` group carries one variable per parameter and `sample_stats` carries `lp` (the log-posterior). A **log-scaled parameter is emitted in its sampling space** (`log10`/`ln`, named e.g. `log10_k`) — the space PyBNF samples and diagnoses in — so ArviZ's recomputed diagnostics share PyBNF's parameterization and Vehtari method; scale is taken from the live parameters (auto-emit) or recovered from the `.conf` copied into `Results/` (standalone), falling back to natural-space-with-a-warning on an archived run whose config can't be reconstructed. The `posterior` is the **saved sample** (`samples.txt`: thinned by `sample_every`, post-burn-in — the same draws `credible*.txt`/histograms use), so ArviZ recomputes diagnostics on fewer draws than the dense `diagnostics.txt`: R-hat is comparable, `az.ess` reads lower **by design** (PyBNF's own final R-hat/ESS ride along in the object's attributes; lower `sample_every` for denser ArviZ diagnostics). The new MCMC-only config key **`output_inference_data = 1`** also auto-writes `Results/inference_data.nc` at run-end (the same builder, via the live parameters), mirroring the `_emit_best_fit_bngl` artifact hook. `arviz` is an **optional extra** (`pip install pybnf[arviz]`, uncapped), lazily imported with a clear install hint so core stays dependency-free; a missing extra is a logged no-op, never fatal. The bridge supports **both arviz major lines** — the classic 0.x `InferenceData` and the 1.x xarray-`DataTree` rewrite (they differ only in the one `from_dict` construction call, which the builder branches on), so installing it never downgrades a user's arviz and CI exercises whichever resolves. `log_likelihood` (→ `az.loo`/`az.waic`), `prior`, and `observed_data` are deferred to the #438 item-4 follow-on that rides on this bridge. (#438, ADR-0055)
- **Self-contained best-fit BNGL: smooth curve + inline data** (`edition >= 2`, ADR-0054, closing the smooth-curve item of #444) — the end-of-run `Results/<model>_bestfit.bngl` artifact (ADR-0048) can now plot as a smooth curve and self-contain its data in one file. In the new era a fitting job's output times come from the data, so the artifact reproduces a *ragged* trajectory (only the measured instants); the new tool key **`smooth_plot_points = N`** re-renders each data-derived time-course `simulate(...)` onto a uniform N-step grid over `[t_start, t_max]` instead of the data's `sample_times`, so running the artifact yields a smooth plot curve. This is **byte-identical** to the uniform form PyBNF already emits when output points aren't data-derived (`method`/`t_start`/`suffix`/condition `setParameter`s preserved), touches only the post-fit artifact copy (the fit already scored on the data grid — the objective is unaffected), and is **cross-engine** (BNG2.pl + bngsim); parameter-scan actions (a swept axis, not time) and the steady-state pre-equilibration phase are left untouched. Separately, **`embed_best_fit_data`** now embeds each time-indexed observable's experimental data **inline** as a `tfun([t...],[y...], time)` reference function (was a sidecar `.tfun` file under ADR-0048), so the artifact is a single self-contained file with no `_bestfit_tfun/` directory. ADR-0048 chose the sidecar form when the inline form was unexercised; bngsim 0.9.55 now supports inline `tfun` natively (verified end-to-end through the bngsim bridge), and since BNG2.pl 2.9.3 parses *no* `tfun` form (inline or file), the switch is engine-neutral and strictly better for self-containment — the embedded-data overlay is read through a bngsim path either way. (#444, ADR-0054)
- **New-era per-observable normalization** (`edition >= 2`, ADR-0053, closing the preprocessing-keys item of #444) — `normalization` now keys by **observable** on the new-era surface, never by filename, fixing a hard crash. ADR-0028 keys data by experiment name, so the legacy filename-keyed per-file form (`normalization = peak: alpha.exp`) raised `KeyError: None` under `edition >= 2` (the filename stem is no longer the data key). Normalization is a per-observable *prediction* transform — a sibling of the per-observable `noise_model` / `cumulative` surface (ADR-0021/0051) — so it now takes the same shape: a whole-fit default `normalization = <type>`, a per-observable `normalization <observable> = <type>` (every experiment), and a per-`(experiment, observable)` override `normalization <experiment>.<observable> = <type>`. The three layer into a single most-specific-wins rule (a strict total order: `<exp>.<obs>` > `<obs>` > default), so there is no tiebreak; a column matched by no rule is left un-normalized, and a target naming an unknown observable/experiment raises (a typo guard, like the `observable:` override). The two new forms ride a `('normalization', target)` structural key (ADR-0014) and compile down to the existing `{data_key: [(type, [columns])]}` representation, so nothing below the config layer changes. The legacy filename form is refused under `edition >= 2` (redirecting to the per-observable form) and kept byte-identical under the legacy edition. Normalization has no PEtab v2 representation (a whole-trajectory reduction, not a pointwise observable formula), so the exporter now **fails loud** on any normalization (`_reject_normalization`, beside `_reject_cumulative`) rather than silently scoring the raw columns. The other three #444 preprocessing keys — `smoothing`, `ind_var_rounding`, `constraint_scale` — are global scalars, not filename-coupled, and ride the new-era surface unchanged (now covered by a test). (#444, ADR-0053)
- **PEtab v2 importer read path** (BNGL-native; ADR-0032, closing the two-adapter proof at the read level) — `pybnf.petab.import_job` is the inverse of `export_job`: it turns a BNGL-native PEtab v2 problem (`problem.yaml` + its TSV tables + the BNGL model) back into a runnable new-era (`edition = 2`) `.conf` plus its `.exp` data files and a fit-instrumented model copy. The reverse asset mappers run backwards onto the shared neutral rows — measurements pivot long→wide (one `Data` per `experimentId`, the `_SD` column rebuilt from `noiseParameters`), conditions undo the surrogate-base `<p>__REF` rename (a base pin dropped, a surrogate relative op recovered, a fixed param's precomputed op recovered as an absolute set, the synthesized `cond_wildtype` mapped back to a wildtype experiment), the objective token recovered from the observables' noise columns (`chi_sq`/`sos`/`sod`/`ave_norm_sos`), and the model's `__FREE` markers re-added — so the **problem** (parameters/priors, observables/noise, measurements, conditions/experiments) is recovered exactly and round-trips byte-for-byte through a re-export. **PEtab is a problem spec; PyBNF is a job spec:** the run-recipe is *supplied, not recovered* and is excluded from that identity — `import_job(problem, out_dir, job_type='de', method='ode', method_overrides=None, settings=None)`, with `method` emitted per-experiment (never a global knob; round-trip-lossy, so a stochastic model does not survive a PEtab hop) and recipe defaults coming from the schema/registry. `job_type='all'` emits one runnable `imported_<jt>.conf` per registered optimizer + sampler (the ADR-0012 benchmark-harness pattern; the `check` checker excluded), so the importer covers the whole toolbox and a new method is importable with zero importer changes. Dependency-free + simulator-free (stdlib `csv` + `pybnf.data.Data`; `problem.yaml` hand-parsed; `petab` stays a test-only oracle), so it runs in the bngsim-less CI tier. The documented PEtab/PyBNF boundaries mirror the export side and raise (SBML model, the five unsupported prior families, a `neg_bin`/`log-normal`/`log-laplace` or expression noise model, replicate rows, multi-model, parameter-scan); **fitting** an imported job is gated on the ADR-0028 config loader (#423). (#407, ADR-0032)
- **PEtab v2 export reads the new-era surface** (ADR-0028, completing the new-era contract) — the PEtab v2 exporter (`pybnf.petab.export.export_job`) now reads a job's data, conditions, and observables directly from the new-era surface (`model:` / `experiment:` / `data:` / `condition:` / `observable:`), so export is a **transcription**: an `experiment:` becomes a PEtab Experiment (its name the `experimentId`; its `data:` files become measurement rows — multiple files are replicates, repeated rows under the one experiment), a `condition:` becomes a PEtab Condition (a fit-and-perturbed parameter handled by the surrogate-base `<p>__REF` rename of ADR-0027, generalized so a condition shared by several experiments emits its rows once), and an `observable:` renames a data column before classification. The exporter is now **new-era only** ("refuse legacy everything"): under a modern edition it refuses a legacy data linkage (`model = X : Y.exp` / `mutant` / `param_scan`), directing the user to the new surface — the gate is on the exporter alone; the fitter still runs legacy confs unchanged. Dose-response (parameter-scan) export stays deferred together with its authoring surface, whose simulation endpoint time has no home in the `experiment:` grammar yet (#426); a parameter-scan experiment raises that deferral. Every exported fixture passes petab's full `default_validation_tasks` via `Problem.from_yaml` + the native `BnglModel` loader (ADR-0026), and `examples/demo/demo_bng_v2.conf` is now a fully new-era, exportable twin of the demo (with `examples/demo/parabola_v2.bngl`, the demo model without a `begin actions` block, since the action is synthesized from the experiment). With this **ADR-0028 is Accepted**. (#407, #423, ADR-0028)
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

### Fixed
- **ArviZ bridge for `am`** — `pybnf.inference_data.from_pybnf` now reads Adaptive_MCMC's per-chain draws, so the `InferenceData` bridge (ADR-0055) works for an `am` run and not only the `dream`/`p_dream`/`pt`/`mh` samplers.
- **Dask future scattering** — scattered `Future`s are passed to `client.submit` as keyword arguments rather than as `Job` attributes, fixing a distributed-scheduler failure on scattered model lists.
- **Bootstrap replicate refinement** — a bootstrap replicate is now polished by the refine step and gated on the *refined* objective, and a retry resamples afresh rather than reusing the failed draw.
- **Initial-condition-only free parameters** — a free parameter that appears only in a species initial condition now actually moves the simulation: the bngsim bridge re-derives species ICs in `execute` (#450).
- **Orphan `_SD` column** — the error for a per-point noise column with no matching observable now names the estimated-noise-scale cause instead of failing opaquely.
- **Loud CLI failure** — running the package as `python -m pybnf.pybnf` now fails with a clear message instead of misbehaving on the relative import.
- **Multi-model `condition:` round-trip** (ADR-0041 addendum, closing #444 item 4) — a multi-model PEtab job whose experiment applies a named `condition:` now round-trips. The exporter and fitter already handled `condition: <name>, model: <file>` (the fitter attaches the condition's `MutationSet` to that model and **requires** the `model:` ref under more than one model), but the importer emitted the reconstructed `condition:` line with **no `model:` field**, so the re-imported multi-model conf failed to load with `Condition '<name>' does not name a model, but the job declares N models`. PEtab conditions are model-agnostic (no `modelId` column — the model↔data link lives on the measurements, ADR-0041), whereas a PyBNF condition belongs to one model, so the importer now **recovers** each condition's owning model from the experiment that applies it (via `condition:` or `preequilibrate:`) and emits the `model:` field under multiple models; single-model jobs stay byte-identical. A PEtab condition referenced by experiments on *different* models has no PyBNF representation (a condition can't span models) and is refused with a clear `NotImplementedError`. The byte-equal export→import→re-export identity is preserved (a condition's `model:` field doesn't alter the model-agnostic PEtab `conditions.tsv`). (#444, ADR-0041)

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
