# Changelog

All notable changes to PyBNF are documented below. This project adheres to
[Keep a Changelog](https://keepachangelog.com) conventions.

## [Unreleased]

### Added
- **A fit's start point is a supported, validated, per-parameter fact that every optimizer
  reads, and the resolved start is recorded beside the results (#583, #559, ADR-0117).**
  There was no supported way to say "start this fit at exactly this point, inside the
  declared box." Every route failed, and every one failed **silently** — the fit ran,
  converged, and reported a plausible number from a point that was not the one that was
  asked for. The new **`start_point = <parameter> <value>`** key says it directly: one line
  per parameter, at every edition, alongside the legacy `*_var` declarations, in the
  parameter's own units whatever its scale. The edition-2 `parameter:` record's
  `initial_value:` field now means the same thing and resolves through the same carrier.
  This closes a defect, not just a gap. `initial_value` has existed since ADR-0043 and is
  honored by twelve fit_types — but `StartPointOptimizer._resolve_start_pset` read
  `FreeParameter.value` on none of its branches, so `cmaes` / `powell` / `sim` / `gntr` /
  `lbfgs` / `trf` / `ms` — exactly the optimizers both issues are about — discarded it and
  started at the box centre. Declared at `k = 0.3, S0 = 100`, a fit started at
  `k = 1.505, S0 = 89.44`. ADR-0043's own text claims `initial_value` is "respected in
  every algorithm family"; that held only for the no-prior form, and was false the instant
  a bound or prior was present.
  A start point is **partial by design** — name only what you want pinned — and for a
  multi-start fit it pins start 0 while the rest stay independent draws, so the scatter
  survives. An out-of-box value is **refused, not moved**: PyBNF never clamped such a value
  to the nearest bound, it *reflected* it back inside with a periodic triangle-wave fold, so
  `250` into `[1, 100]` became `50` and the fit proceeded from an arbitrary interior point
  with a `DEBUG` line. Every refusal is a proper configuration error rather than the bare
  `OutOfBoundsException`, which reached the user as "an unknown error … please report this
  bug" — including on PEtab import, where an out-of-box `nominalValue` crashed that way on
  a file the user never wrote.
  **`Results/start_point.txt`** records where the run actually began — value, source, and
  the declared box, per parameter — before anything is scored. That artifact alone would
  have caught all four of the reported failures, and it is the only way to recover a CMA-ES
  start at all, since that start seeds the distribution mean and is never itself evaluated.
  The PEtab importer now emits `start_point` from `nominalValue`, a mapping ADR-0043's
  field table advertised and neither direction implemented.
  The two spellings are synonyms **except on `profile_likelihood`**, where a complete
  `initial_value:` specification keeps its established meaning — those values are θ\*, so the
  polish is skipped. `start_point` never carries that meaning; it names where the polish
  starts. The distinction is load-bearing: a `nominalValue` is not a claim of optimality, and
  unifying the two would have made every PEtab problem with a full `nominalValue` column
  profile around the nominal point rather than the optimum, putting every confidence bound in
  the wrong place without a word.
- **A `model:` declaration carries that model's own CVODE tolerances, so a per-species
  absolute tolerance can finally be written by hand (#586, ADR-0116).** `sbml_atol` and
  `sbml_rtol` are one key each over *every* SBML/Antimony model in a fit, and that — not the
  grammar — is why neither could ever take a vector: a positional one is ordered against a
  species list a conf author cannot see, and a species-keyed one has no reading across models
  that do not share species names. Give the statement one model to be about and both
  objections disappear. Under `edition >= 2` a single-model `model:` line now takes three
  optional labeled fields, in any order:

  - **`atol:`** takes exactly what `sbml_atol` takes (a number, `auto`, or
    `tracking [decades]`), for that model alone.
  - **`rtol:`** is the per-model `sbml_rtol`. A scalar, and only ever a scalar — CVODE takes
    one relative tolerance and there is no per-species one to route.
  - **`species_atol:`** is the hand-written vector, written as `<species> <number>` pairs:
    `model: weber.xml, atol: auto, species_atol: PKD 1e-3, CERT 1e-2, rtol: 1e-9`.

  **The map is a set of exceptions, not a replacement.** Every species it names takes the
  number stated, verbatim — neither ADR-0103's `1e-8` ceiling nor ADR-0105's model-scalar
  floor binds a number a person wrote, for the same reason a pinned `sbml_atol` number is not
  clamped either. Every species it does *not* name keeps whatever it would have had: the
  derived vector, a pinned scalar broadcast, or the backend default. The steady-state cutoff
  stays the model-wide number, and under `atol: tracking` the map becomes the ceiling the
  trajectory-following weights sit beneath.

  **The species names are bngsim's, and an unknown one is an error at config load** that
  lists the names the model does have. That matters twice: bngsim renames a species id
  colliding with an Antimony reserved word (`NULL` integrates as `_ant_NULL`, and the error
  says so), and a parameter driven by a rate rule becomes a state the `<listOfSpecies>` never
  mentions — nameable here and reachable by nothing else. On
  `Smith_BMCSystBiol2013`, the one subset-I slug where the two name lists disagree, the
  derived vector declines outright and a hand-written map is the only route to
  `CVodeSVtolerances` there.

  **Nothing changes without one of those fields.** `sbml_atol`/`sbml_rtol` keep every meaning
  and remain the fit-wide default; a job that states nothing gets the same `Simulator.run`
  call it got before, argument for argument. A `model:` line carrying a tolerance field must
  declare exactly one model, and that model must be an `.xml`/`.ant` one on
  `sbml_backend = bngsim` integrated by CVODE. A BNGL model states `atol`/`rtol` in its own
  `begin actions` block, the RoadRunner backend has its own integrator settings, and
  `sbml_integrator = gillespie` runs every action stochastically — all three are refused
  rather than accepted and ignored, as is a `species_atol` on a bngsim without
  lanl/bngsim#196.

  Measured on `Weber_BMC2015`: writing bngsim's own `rtol * y_i` rule by hand — the rule
  ADR-0105 measured and declined to apply automatically — costs 192 integrator steps against
  `auto`'s 184 and the clamped default's 210, agreeing with all of them to ~3e-08 at the
  final state. The derivation is still the better default; what it could not do was be
  overridden for one species of one model.
- **The information criteria are checkpointed alongside the parameter sets, so a run is
  scoreable before it ends (#560).** `sorted_params_backup.txt` has been written throughout a
  run since forever; `information_criteria.txt` was written **only** on the terminal path. Every
  downstream consumer of a fit's result needs both, so a run was un-scoreable at every moment
  except its last — even though the best parameter vector had been on disk the whole time and the
  number was fully determined long before the process exited. That gap is not cosmetic:
  `log_likelihood` in `information_criteria.txt` is the only place PyBNF reports the **full
  normalized** log-likelihood (every dropped per-point constant restored), while the minimized
  `Obj` column of the parameter table is the *reduced* objective, so an absolute AIC/BIC — or any
  benchmark score built on one — could not be computed from the checkpoint alone.
  Each checkpoint now also writes `Results/information_criteria_backup.txt` (and
  `information_criteria_refine_backup.txt` during a refine, mirroring the parameter file beside
  it). Same format as the final artifact, differing only in `#` comments that mark it a snapshot
  and name the parameter set it describes, so one parser reads either file. The final
  `information_criteria.txt` is unchanged, in name, content, and meaning.
  Cost is one extra simulation per checkpoint, and only when the checkpoint has something new to
  say. Nothing is spent while the best fit is unchanged — the file on disk already describes it,
  which is exactly the long converged tail that motivated the issue: a `gntr` fit of
  `Brannmark_JBC2010` (100 starts × 1000 iterations) reached its final objective with ~40 minutes
  left, all of it spent waiting for a file rather than for an answer. Nothing is spent at all
  unless the objective is a proper likelihood, since no information criterion is defined for
  `sos` / `sod` / `norm_sos` / `kl` / `wasserstein` / `direct_pass`. Otherwise it is one
  simulation per `backup_every * population_size * smoothing` returns — 1 in 1000 at the common
  `backup_every = 10, population_size = 100`. The new key **`backup_information_criteria`**
  (default 1) turns it off for a model where even that is too expensive.
  A killed or crashed long run is worth what it should be, too: the parameters survived it
  already, and now the score does.
- **`sbml_atol` takes `auto` and `tracking`, so a model whose own scale asks for a *looser*
  absolute tolerance can have one (#557, ADR-0114).** ADR-0103's derivation is allowed only to
  tighten. That is a no-regression rule and it is why the derivation could be applied to every model
  without a flag — but a model whose species all sit far above one has a real, computable tolerance
  need, and the derivation computes it and then discards it. `Weber_BMC2015`'s seven species live at
  `1.24e+02 .. 4.21e+07`, so it asks for `4.665e-03` and is handed `1e-08`, 5.7 decades tighter.
  ADR-0105's per-species vector cannot rescue that either, and the way it fails is the point: each
  entry clamps into `[scalar_atol, default_atol]`, the scalar has itself been clamped to
  `default_atol`, that interval collapses to a point, and the vector correctly declines — so the
  per-species mechanism silently declines exactly the models that span the most decades. Measured
  over the subset-I corpus, the clamp binds on **10 of the 22 slugs with a readable nominal state**.

  - **`sbml_atol = auto`** lifts the ceiling on both derivations and nothing else. ADR-0105's
    *floor* — no species resolved below the model's own scalar — stays exactly where its measurement
    put it (releasing it killed 91 of 100 `Brannmark_JBC2010` box points against 39), and the
    `1e-16` floor stays too. What is left, `rtol * max(y_i, median)`, is bngsim's own `derive_atol`
    with the model's median as its `floor`, and is asserted against the library rather than against
    a second copy of the arithmetic.
  - **`sbml_atol = tracking [decades]`** wires lanl/bngsim#213's `CVodeWFtolerances` — an absolute
    tolerance re-evaluated against the state being integrated, so a species that starts at order one
    and decays to nothing keeps a tolerance that means something for it. This is the half ADR-0105
    named as out of reach. The ceiling is `auto`'s vector, held from the **nominal** state, so
    `tracking 0` is `auto` exactly and a fit that moves initial conditions does not move its own
    tolerance; bngsim's bare `"auto"` ceiling, which re-derives from the live state at every run,
    is deliberately not used. An unstated depth is left to bngsim rather than copied.

  **Nothing changes without one of those two words.** Unset is byte-identical — same clamps, same
  vector, same steady-state pairing — and a number remains the documented off-switch that pins
  `CVodeSStolerances` ulp for ulp. `tracking` on a bngsim without the capability is *refused*, at
  config load and again in the model constructor, rather than silently integrating at something
  else.

  **One thing #557 claims does not reproduce, and it is recorded rather than repeated.** The issue's
  headline is that Weber integrates 6 of 30 sensitivity-applied box points at the clamped tolerance,
  against 22 at `1e-04`. On the current stack all six arms — unset, `1e-08`, `1e-04`, `auto`,
  `tracking`, `tracking 6` — integrate **30 of 30**, within 7.2 s of each other. bngsim moved from
  0.12.2 to 0.13.0 in between, and the documented Weber-specific change is lanl/bngsim#305, whose
  own entry measures that slug's `t = 24` crossing and reports the step count roughly halving. So
  this ships as a capability with a measured cost rather than as a rescue, and the instrument that
  discriminates is **integrator steps** rather than pass/fail. Over 20 box points per slug with the
  gradient sensitivity request applied, `auto` costs 0.38x–0.67x the CVODE steps on the six slugs the
  clamp binds hardest (`Perelson` 8 440 → 3 216; `Weber` 78 641 → 36 154; `Laske` 236 629 → 158 543),
  is bit-identical on `Giordano_Nature2020` — the control, whose derivation tightens and so cannot
  see the ceiling — and moves `J_paper` at the PEtab nominal point in its sixth decimal. It is not
  free in the other direction either: error-test failures rise as steps fall, and `Laske` lost one
  box point of twenty at one seed (both arms lose one at a second seed). ADR-0114 has every arm and
  says what it has not bisected.

### Fixed
- **A multiple-shooting segment that comes back short of its end knot is refused instead of
  being read at the wrong time (#584).** `job_type = ms` builds every continuity row from the
  **last row** of a segment's trajectory, which is the end knot only if the integration
  reached it. An integrator that stops early and returns a partial result rather than raising
  breaks that quietly: the trajectory is finite, in the right columns, and its final state
  belongs to an earlier instant, so the defect `Phi_j(z_j, θ) − z_{j+1}` becomes a difference
  of states taken at two different *times* — a different constraint, satisfied by a different
  trajectory. Nothing downstream could see it, because the symptom is a nonzero defect, which
  is what an honest stage shows after θ has moved. Measured on the offline fixture: a stage
  seeded to be feasible at iteration zero, whose defect norm is exactly `0`, reported `0.0399`
  and would have optimized against it. The segment seam now checks that a span reached the
  knot it was asked for, and treats one that did not the way it treats any point that did not
  integrate — the local model goes non-finite and the search backs off, rather than the run
  dying on a point-specific failure.
- **A segment simulation that carries no forward sensitivities now names the segment seam
  (#584).** It stopped on the gradient assembly's own message — *enable the gradient path
  (apply_routing) on every scored model* — which tells a `job_type = ms` user to enable a path
  they did enable. A sensitivity request is applied **per action** (#475/#482), so a segment
  run under a suffix the request never reached comes back with a perfectly good trajectory and
  no tensor at all; that is an internal wiring error and every point in the fit hits it, so it
  is now refused where it happens, saying which segment of which experiment returned what.
- **The shooting suite now tests behaviour at pathological parameter points, not only the
  arithmetic at well-behaved ones (#584).** The two defects above were found by these tests.
  The two before them (#578, #581) were not: both reached `main` through a suite of 51 passing
  offline tests and were caught by pointing `job_type = ms` at a real model. The offline
  fixtures are a closed-form flow and an exponential decay, deliberately well-behaved so that
  every derivative is checkable exactly — which is the right design for what they verify, and
  leaves nothing asking what the method does when a *point* misbehaves. The offline
  backend's failures are now switchable — a region that does not integrate, a region whose
  trajectory is finite while its forward sensitivities overflow (#581's exact shape), a span
  longer than the "model" can carry, a one-off refusal on the n-th call, a missing sensitivity
  tensor, and a span that stops short of its end knot — and the new tests assert the three
  properties ADR-0110 states as design and nothing checked: an unusable local model **backs
  the search off** rather than ending the run; a run that stops early still **reports what it
  has already earned**; and an unusable point **never becomes a reported fit**, including the
  corner the method's own advantage creates, where every segment integrates, the whole horizon
  does not, and the run therefore reports no fit rather than a segmented score no ordinary run
  could reproduce. Each runs in milliseconds with no simulator.
- **A gradient fit says, at job start, when bngsim declined the analytic `∂f/∂θ` for one of its
  models and CVODES' difference quotient is carrying every sensitivity column instead (#606,
  ADR-0121).** `CVodeSensInit1` takes one sensitivity-RHS callback for every column, so a single
  rate law bngsim cannot differentiate — an `abs()`/`floor()`/`erf()` term, a comparison it
  cannot solve, or a derivation that ran out of its build-time budget — declines the analytic
  path for the **whole model**. The substitution is correct and costs an extra right-hand-side
  evaluation per column per step, so an N-parameter fit pays roughly N× the sensitivity cost.
  On a fit measured in hours that is not a slower answer but no answer: on
  `Smith_BMCSystBiol2013` all 25 columns fell back, every start timed out to `inf`, and thirteen
  hours produced nothing. PyBNF surfaced none of it — no console line, no refusal, nothing
  distinguishing a gradient fit on the analytic path from one on the fallback. The decline did
  reach `<prefix>.log`, because bngsim's logger propagates to root, but as one line per model
  written mid-run from N worker processes into a shared, noisy file: discoverable by someone who
  already suspects the problem, which is the wrong order.
  PyBNF now checks **once per model at gradient-path setup**, on the head node, before the fit
  has evaluated anything, and names the model, its column count, the expected cost multiplier and
  bngsim's own reason. The check is not the log line: the verdict is read off the compiled codegen
  artifact — whether it exports the analytic sensitivity RHS symbol, the exact symbol bngsim's
  C++ resolves to choose between the two paths. That matters because bngsim reports a decline
  while *generating* codegen source, and since lanl/bngsim#174 a warm structural cache skips
  source generation entirely: measured on 0.14.0 and 0.13.0 alike, the same declining model
  reports its decline on the first construction and says nothing on the second, while running on
  the same fallback both times. Since the cache persists on disk, the run that hears nothing is typically the second run
  of a fit — the one made after the first came back empty. bngsim's reason is still captured and
  reported when it is heard, but only ever as prose; nothing keys off its absence.
  The new `sensitivity_fallback` key chooses what happens next: `warn` (the default — today's
  behaviour plus the sentence), `error` (refuse the fit, for a long unattended run), or `ignore`
  (skip the check, including the one simulator construction per model it costs). The policy keys off the verdict rather than the reason, so a
  fit refuses or does not refuse reproducibly whatever state the codegen cache is in. A model
  PyBNF cannot read an answer for — a `codegen=False` run has no artifact — reports **no
  opinion**: logged, never warned about, and never refused, because guessing is wrong in both
  directions.
  One decline is worse than slow: a model that also branches at a crossing whose time moves gets
  a difference quotient that integrates straight through it, so every column is wrong at and
  after the crossing. bngsim ≥ 0.14.0 refuses such a run rather than return a gradient it has
  flagged as wrong; on 0.13.0, which PyBNF's floor still admits, the same model only warns and
  returns it, and PyBNF now says so at verbosity 0 whenever bngsim's reason reaches it.
- **The discrete-event gradient gate reads a bngsim capability instead of a version floor, so a
  from-source build that merely *declares* a new enough version no longer passes it (#558,
  ADR-0119).** `BNGSIM_HAS_EVENT_SENS` gates forward sensitivities that survive a discrete
  event, and it does not gate a missing feature — it gates **silent wrongness**: a build below
  the line does not refuse an event it cannot differentiate, it returns a finite tensor with the
  event's contribution missing. It decided by comparing `bngsim.__version__` against exactly
  `0.12.2`. bngsim bumps its version at the *start* of a release cycle, so every from-source
  build made between that bump and the fixes that set the line (lanl/bngsim#144, #146) declares
  the same string as the release that carries them, clears the floor, and is reported as
  carrying them. The two failure directions are not symmetric — a false *absent* is a refusal
  and a metaheuristic fit; a false *present* is a gradient fit that runs to completion and
  reports a converged wrong number — and a version compare could only ever be wrong the second
  way. The neighbouring `BNGSIM_HAS_PER_SPECIES_ATOL` already carried the argument, for the same
  version string, in a comment on the wrong flag.
  The gate now resolves through published capabilities: `features['event_sensitivities']` if
  bngsim ever publishes a dedicated key (both directions, so the flag starts reading the real
  answer on the first build that grows one, with no PyBNF release), otherwise
  `features['effective_ic_sensitivity']` — a **witness**, usable because lanl/bngsim#155 added it
  three commits after #146 inside the same release window, so a build that publishes it
  necessarily carries the fixes. The version survives only as a veto: it can no longer report the
  capability present on its own, because the witness shipped *in* 0.12.2 and a build claiming
  0.12.2-or-newer without it is exactly the pre-release build at issue. **No install that works
  today is refused** — every released bngsim at or above the floor publishes the witness — and a
  refusal now names the route that decided (`event_sens_probe()`) rather than telling a reader
  who already has 0.12.2 to upgrade to 0.12.2.
- **A fit whose bngsim loaded a compiled core older than its own C++ says so at job start, not in
  import noise (#558, ADR-0119).** An editable bngsim serves live Python from the source tree
  while loading `_bngsim_core*.so` from a separately built artifact with auto-rebuild off, so the
  two halves drift — one install reporting `0.12.2` was found with a core binary three days older
  than the `.cpp` beside it. Every version, metadata and feature-key check passes there, because
  nothing in the Python layer moved. bngsim detects it by mtime and warns, but it warns at
  *import*, which for PyBNF is while the `pybnf` package loads: before `init_logging`, before the
  config is read, and before the user has committed to anything. PyBNF now repeats it at job
  start — the core's identity line (path, build commit, mtime) to the log unconditionally, the
  staleness report promoted to a console warning at verbosity 0 — where a reader can still stop a
  run that would otherwise spend hours producing statements about code that is no longer in the
  tree. `bngsim_build_id()` exposes the commit the core was built from, which is the only thing
  on hand that tells two installs declaring one version apart. Every read is guarded and
  memoized; an install that cannot answer reports no opinion rather than taking the fit down.
- **A `parameter:` record is now held to the same declaration rules as the equivalent `*_var`
  line (#603, ADR-0118).** `_check_variable_keyword_combination` refuses an incoherent pairing
  of free-parameter declarations and `job_type` — an unbounded prior handed to a box-mode
  optimizer, `var`/`logvar` handed to a method that draws a population, a mix of point starts
  and boxes. It decided what it was looking at by pattern-matching **config key names** with
  `re.search('var$', k[0])`, which a `('parameter', <id>)` key never matches, so the whole rule
  was silently bypassed by the edition-2 syntax:

      legacy  normal_var = p1 0 1          -> refused: Box-mode optimizer requires a bounded prior
      record  parameter: p1, prior: normal -> ACCEPTED

  That matters more than an ordinary validation gap, because the record syntax is the *only*
  one that can express `initial_value` — so the surface most likely to be used for careful,
  seeded work was the one with no coherence checking at all.
  The rule now keys on the **built `FreeParameter`** rather than on the key that declared it,
  and runs after the variables exist rather than before. Both syntaxes produce the same
  `FreeParameter`, so both now get the same answer. The obvious repair — re-deriving the
  keyword set from `{v.type}` — was tried and rejected: a *truncated* prior carries a real
  finite box while its family does not, so it would have falsely refused
  `prior: normal, ..., lower: X, upper: Y` on `job_type = gntr`, which is an entire benchmark
  corpus. Verified equivalent to the old rule for every untruncated declaration (family-level
  and parameter-level bounded-support agree across all 48 registered prior keywords), and run
  against **1049 real `.conf` files** with zero refusals.
  The dead branch for ADR-0015's third fit_type category is deleted: every registered refiner
  now also carries `start_from_box`, so the "point-only start optimizer" category is empty and
  its code was unreachable. Error messages now name the offending **parameters** rather than a
  keyword the record user never wrote, and point at `start_point` for the case they usually
  mean — a bounded box searched from a chosen point.
- **`starting_params` is now a configuration error on a `job_type` that has never read it
  (#559, ADR-0117).** It has exactly one read site — the Bayesian sampler base — so on the
  other fourteen `job_type`s it was accepted, validated against nothing, and then discarded
  without a word: a `gntr` job seeded with it produced **bit-identical** output to the same
  job with the line deleted. The error names `start_point` as the replacement, which every
  `job_type` reads and which is matched by **name** rather than by position (`starting_params`
  is positional against declaration order, while every result file PyBNF writes is
  alphabetical, so round-tripping a result row back into it silently permutes the values).
  Unchanged for the six samplers that do read it, which is every shipped conf that sets it.
- **A mixed bounded/unbounded parameter set no longer starts every parameter at the wrong
  place (#583, ADR-0117).** Start resolution was all-or-nothing, so one unbounded parameter
  sent *every* parameter down the point-start branch — where a bounded parameter's `p1` is
  its **lower bound**, read as if it were a sampling-space start value. A `loguniform_var`
  over `[1e-3, 1e3]` started at `10**1e-3 = 1.0023`, its lower corner, with nothing logged at
  any level. Resolution is now per parameter.
- **CMA-ES no longer freezes a coordinate whose prior is truncated (#583, ADR-0117).** The
  per-coordinate box width was `p2 - p1`, which is the box only for a `uniform`/`loguniform`
  declaration; for a **truncated** prior those are the family's location and scale, so the
  width came out as the scale and, for the entirely ordinary `sd == mean`, as exactly `0.0`.
  CMA-ES squares these into its initial covariance diagonal, so that coordinate got a
  singular covariance and could never move for the whole run. Widths now come from the
  prior's own support — bit-identical for every `uniform`/`loguniform` box.
- **`FreeParameter` no longer skips its bounds check for a value of exactly `0`, and no
  longer mutates the shared template on a fold (#583, ADR-0117).** The check was guarded by
  truthiness rather than `is not None`, so `initial_value: 0` — a legitimate value for a
  linear parameter — was stored unvalidated. Separately, folding an out-of-box value wrote
  `self.value = self.lower_bound` onto the template `FreeParameter` living in
  `Configuration.variables`, which every Algorithm aliases and which rides the algorithm's
  pickle, so the contamination survived a checkpoint and a `--resume`. Nothing read it.
- **The concurrent multi-start scatter honors `initialization`, and says so when it cannot
  scatter (#583, ADR-0117).** It called the Latin-hypercube sampler unconditionally, so
  `initialization = rand` was a silent no-op for the whole gradient/CMA-ES multi-start
  family; and a `population_size > 1` on a fit with no box to scatter across was silently
  reduced to a single start — the same "accepted, does nothing, says nothing" shape as
  `starting_params`.
- **`job_type = profile_likelihood` no longer reports that no start point was supplied when
  some were (#583, ADR-0117).** A partial specification cannot be θ\*, so it correctly falls
  through to the polish — but it said "No initial_value supplied" while doing so. It now
  names the parameters that were left undeclared. A complete specification keeps its
  established meaning there: those values are the optimum, and the polish is skipped.
- **`sbml_rtol` is checked for finiteness, not just for sign, so `sbml_rtol = inf` no longer
  reaches CVODE (#586).** The conf grammar's number token also matches `inf` (ADR-0047's open
  truncation side), so `sbml_rtol = inf` parsed to a float that cleared the config check's
  bare `tol <= 0.` and was installed as the relative tolerance — which turns relative error
  control off rather than erroring. `sbml_atol` never had the hole, because
  `parse_atol_setting` has always demanded finiteness. Found while adding the per-model
  `rtol:` field of #586, which would otherwise have inherited the same check.
- **`job_type = de` and `job_type = ade` no longer stop after generation 0 on a negative
  objective (#561, ADR-0115).** The Differential Evolution family tested convergence with a
  *ratio* of objectives — `max(fit) / min(fit) < 1 + stop_tolerance` — which reads as
  convergence only on a positive objective bounded below by 0 (a χ², an SSE). On a likelihood
  objective (a negative log-likelihood, unbounded below) it fired at generation 0: an
  all-negative population lands the ratio in `(0, 1]`, and a single `inf`-scored failed
  simulation makes it `-inf`, below *every* threshold — so no value of `stop_tolerance`
  disabled it, and both members of the family were unrunnable on any estimated-σ likelihood
  fit (the whole Grein et al. 2026 benchmark subset-I corpus, 23/23 slugs). On
  `Borghans_BiophysChem1997` (`islands = 4`, `population_size = 400`, `max_iterations = 600`)
  the run terminated inside the first generation, spending a 240,000-evaluation budget on 0
  generations of search — and `stop_tolerance = -1e9` still fired, because the failed-sim
  `-inf` is below that too.
  The convergence test is now an **absolute range in objective units**,
  `max - min <= de_tolfun`, assessed over the **finite** fitnesses only — sign-agnostic, and
  with failed simulations (`inf`) ignored so one dead candidate can neither trigger nor block
  the stop. It gets its own key, `de_tolfun` (a range in objective units, where
  `stop_tolerance` was a dimensionless ratio), which falls back to `stop_tolerance` when unset
  so an existing config keeps its threshold magnitude — mirroring how `cmaes_tolfun` splits
  off `cmaes_stop_tol` (ADR-0106, the CMA-ES sibling of exactly this defect). The shared check
  lives in one `DifferentialEvolutionBase` helper, so `de` and `ade` cannot drift apart again
  (the missing `!= 0` guard that let `ade` divide `0/0` on an all-zero population was such a
  drift; the range form has no division). In an island run, convergence is assessed only once
  every island has completed an iteration, so ignoring `inf` cannot let one finished island
  stop the whole search before the others have run.
  Regression tests pin each failure mode at its decision point (an all-negative spread and an
  `inf`-defeated threshold are *not* converged; a collapsed finite population *is*; `de_tolfun`
  is its own knob; the all-zero `ade` population no longer divides) plus an end-to-end guard
  that both optimizers advance past generation 0 on an objective that is negative everywhere
  the population lands.
- **`job_type = ms` no longer dies on a fit with a measurement-model formula observable (#578).**
  Multiple shooting was unusable on essentially the whole PEtab-imported corpus — including its
  own motivating problem, `Borghans_BiophysChem1997` — failing on the first outer iteration with
  `Measurement model 'Ca' would shadow an existing simulation-output column`. The setup was all
  correct (noise profiling, the `4-2-1` ladder, knot placement); it died the moment the loop
  asked for a second evaluation.
  The measurement layer materializes each `observable: <id>, formula: ...` column *into the
  trajectory in place* (ADR-0036) and deliberately refuses a column that already exists. Every
  ordinary fit satisfies that for free, because the propose/score loop scores a freshly
  simulated `Data` every time. Multiple shooting caches its segment trajectories per point — so
  that one augmented-model evaluation costs one pass of segment simulations rather than two —
  and the outer loop then re-evaluates at the point the inner solver finished at, which is a
  cache hit on those very objects.
  Fixed at the cause: the assembled objective is now memoized on the point, so each simulated
  trajectory is scored exactly once. That also removes a redundant gradient/Fisher assembly per
  outer iteration, the larger of the two costs. Sound because the objective at a point does not
  depend on the multipliers — only the augmented model combines them.
  The shooting suite structurally could not see this: its fixtures score native columns (a
  species, an observable), so the measurement layer never ran. The regression tests use a
  formula-observable fixture, and both fail with the exact production error when the memo is
  removed.
- **`job_type = check` runs again (#569).** #564's method-chain record was built from
  `alg.res_dir` on a line that ran before the `job_type != 'check'` branch nine lines below it, so
  every check run died in setup with `AttributeError: 'ModelCheck' object has no attribute
  'res_dir'` — no objective value at all, not merely a noisy tail. `ModelCheck` deliberately does
  not subclass `Algorithm`, so it has no `res_dir`, and none of `stop_reason`,
  `completed_simulations` or `trajectory` either; the fit-phase recording that reads all three was
  above the branch too. Both now sit inside it, with the boundary stated in a comment so the next
  addition to `main()` lands on the correct side of it. A check run still writes no
  `method_chain.json`: it is one evaluation of the parameters as given, not a chain of search
  phases. The regression test that was missing — a `job_type = check` job driven end to end
  through `main()` — now guards the path; every existing check test called `run_check()` directly,
  which is why two commits landed on top of the break.
- **`wall_time_fit` no longer silently downgrades `refine = 1` to no refine at all (#564,
  ADR-0107).** `refine = 1` requests a *method* — search globally, then polish the result with a
  local optimizer — but a wall-clock-budgeted search runs until the clock stops, so it has no
  reason to leave anything behind, and the polish (new work, forbidden once the budget is spent,
  ADR-0093) never started. Not occasionally: **15 of 15 runs** in a benchmark campaign
  (`Borghans_BiophysChem1997`) configured as `cmaes` + `refine = 1, refine_method = gntr` actually
  ran plain `cmaes`. And the downgrade was invisible — ADR-0093's whole promise is that a budgeted
  run "writes exactly what a converged one writes", so `sorted_params_final.txt` and
  `information_criteria.txt` looked identical either way and the only trace was one line on stdout.
  A harness that scores a directory could not tell which method it had measured.
  A new global key **`wall_time_refine_frac`** (default `0.1`) holds that share of `wall_time_fit`
  back from the search, so the refine runs on a slice the search was never allowed to spend. The
  run's total is unchanged — one deadline still bounds the whole run, it is just partitioned rather
  than first-come-first-served — and the split is stated on the console before the search starts.
  The reserve is a floor, not a cap: a search that converges early hands everything it did not
  spend to the polish. No reserve is taken when there is no refine to protect (no `refine`, no
  budget, or a `refine_method` naming the algorithm the fit itself ran), so a run that asks for no
  polish is byte-identical to before. `wall_time_refine_frac = 0` restores the old split, and the
  resulting skip is now a `print0` warning that names the method that did *not* run, the method
  that ran alone, and the key that would have made room.
- **A refined run's `sorted_params_final.txt` describes the refined point (#564).** The refiner
  wrote its result only to `sorted_params_refine_final.txt` while rewriting
  `information_criteria.txt` from the same end-of-run tail, so two files in one `Results/`
  disagreed about which parameter set they described — and the *conventional* name carried the
  point the requested method chain did not end on. A refine's end-of-run output is the run's
  end-of-run output, and is now written under both names; the `refine_`-prefixed file is unchanged.
- **A bootstrap replicate's refine no longer writes into the main run's `Results/` (#564).**
  `_refine_best_fit` redirected a replicate's `sim_dir` and `failed_logs_dir` to the
  `Results-boot{N}` peers but not its `res_dir`, so every replicate's polish overwrote the *main*
  fit's `sorted_params_refine_final.txt` and `stop_reason.txt`. The refiner now writes where the
  fit it is polishing wrote.
- **A refine's wall-time stop reason is appended to `Results/stop_reason.txt`, not written over the
  fit's (#564).** Both phases share one Results directory; a run where the search hit the deadline
  *and* the polish did has two facts to report, not one that replaces the other.

### Added
- **Measurement-time uncertainty via posterior marginalization, phase 1 (#587, ADR-0112).** A
  new `time_error` clause on the `noise_model` line treats the latent sampling time as a random
  variable and *integrates it out* of the likelihood, instead of assuming each datum was collected
  at exactly its reported time — an assumption that biases estimates and makes posteriors
  overconfident when sampling times actually drift (handling delays, imperfect synchronization,
  reporting error). Written whole-fit as `noise_model = <family>, <scale> = <source>, time_error =
  truncated_normal, sigma_t = fit st__FREE` (or `uniform`; `sigma_t = fix_at <w>`), it replaces the
  per-point likelihood with a `MarginalizedTimeObjective` whose per-observation contribution is
  `−log ∫ p(ȳ_k | y(τ)) p(τ | t_k) dτ` — the `n_t`-dimensional marginal factorizes into
  one-dimensional integrals (the method of Vanhoefer, Nakonecnij, Binder & Hasenauer, bioRxiv
  2026.05.09.724053; the temporal analogue of Raimúndez et al. 2023 nuisance marginalization). The
  search stays `n_θ` (+ one `σ_t`), not `n_θ + n_t`. Phase 1 evaluates each integral by log-space
  quadrature over the stored trajectory, reusing every noise family's normalized `log_density`
  (ADR-0056) as the integrand and the gradient-free optimizers/samplers (`de`/`pso`/`ss`/`mh`/
  `dream`/…) unchanged — nothing is added to the model file. Edition-2 only. The `σ_t → 0` limit
  is the standard likelihood (a `fix_at 0` clause short-circuits to it). LOO/WAIC and
  `information_criteria.txt` work out of the box: the marginal per-observation `log z_k` **is** a
  normalized per-observation log-likelihood, so the objective reports it through the same
  `evaluate_pointwise` hook the per-point families use (`Σ_k log z_k = −score`), and an estimated
  `σ_t` is already counted in `k`. A marginalized time course is simulated on a **dense uniform
  grid** over the support (`t_end:` required, `t_start:`/`n_steps:` optional on the experiment
  line — decoupled from the sparse reported times, which only centre each timing prior), and
  `sigma_t = fit …` estimates the timing scale jointly (recognized as a declared nuisance).
  Worked end to end in **tutorial lesson 49** (`examples/tutorial/49_measurement_time_uncertainty/`):
  ignoring the timing spread biases the decay rate to `k ≈ 1.36` (truth 1), marginalizing recovers
  `k ≈ 1.06`, and estimating `σ_t` recovers `k` while re-discovering a non-zero timing error.
  Deferred and refused at
  build with a reason: a per-observable time prior, a prediction-dependent `σ`, the count family,
  and every gradient `job_type` (`trf`/`lbfgs`/`gntr`/`hmc`/`ms` — phase 2's augmented-ODE
  sensitivities are what those need); `noise_profiling` (which *maximizes* a scale out) is refused
  as ill-defined alongside marginalization (which *integrates* the time out).
- **Multiple shooting, `job_type = ms` (#563, ADR-0110).** The consumer of the
  constrained-transcription layer below, and the thing #563 was actually asking for. Each scored
  experiment's time course is cut at knots; segment *j* is integrated from its own start state —
  segment 0's is the model's own initial conditions, and each interior knot carries an auxiliary
  state that is searched, bounded and differentiated but is **never** a reported fit result — and
  continuity `Phi_j(z_j, theta) - z_{j+1} = 0` is enforced by an augmented Lagrangian whose
  subproblem is solved by `gntr`'s own Gauss-Newton trust-region step machine. Every reported
  score comes from discarding the auxiliary states, re-simulating theta with ordinary single
  shooting, and scoring *that*, so a run that leaves continuity unconverged scores as what it
  actually is — and every certified iterate lands in the ordinary trajectory at that score, so
  `sorted_params`, the best-fit simulations, the information criteria and the inference-data
  sidecar are produced by the same code every other `job_type` uses.
  Why it is worth the machinery: on `Borghans_BiophysChem1997` a correctly-shaped oscillator whose
  period is wrong by more than about 3 % scores *worse than fitting no dynamics at all*, so under
  single shooting the flat line is the ceiling on essentially the whole box and fifteen
  independent global searches terminate at it. Over one short segment a period error cannot
  accumulate: the information moves out of a residual term that saturates and into continuity
  defects, which carry a direction.
  The prototype's structural finding is what keeps the implementation small — a segment-start
  state is an `IC` route with chain-rule factor 1, so the existing gradient/Fisher assembly builds
  its column with no new residual math, and each segment is presented to it as an ordinary
  *experiment*. The continuity block is the only new assembly surface. Knots are named by their
  exact fraction of the horizon, so a coarser rung recognises a finer one's knots and the `4-2-1`
  ladder *continues* rather than reseeds.
  New keys `ms_segments`, `ms_coarsening`, `ms_penalty`, `ms_penalty_growth`, `ms_max_penalty`,
  `ms_feasibility_tol`, `ms_optimality_tol`, `ms_inner_iterations`, `ms_aux_decades`,
  `ms_max_iterations`, defaulted from ADR-0109's measurements rather than from taste. Requires the
  bngsim backend — a knot carries the model's *state*, so both a generated network (`.net`) and
  an SBML/Antimony model are supported, through two backends that differ only in what a
  simulation returns: on the SBML path the columns an experiment scores and the columns a
  continuity row differences are the same columns, and on the `.net` path they are not, so that
  backend asks for the observable and species selector families together and one integration
  still serves both (#577). A network-free (NFsim) model enumerates no state and is refused. It
  also refuses, by name, a fit whose scored quantity is a
  function of a whole series — an analytic per-series scale, a data normalization, a
  cumulative-to-incident difference — since cutting the series would change it. An analytically
  profiled noise scale (ADR-0108) is deliberately fine: it is profiled over pooled residuals, so
  the segments pool the same ones, and the constraint terms never enter the likelihood.
  What is *not* claimed: that multiple shooting improves the typical fit (48 paired starts:
  24–24, medians tied at every radius, at 2–7x the simulations), or that it solves Borghans from
  an uninformed start (0/24). The measured case is the tail and the robustness. Segment
  simulations run serially on the master in this cut; parallelising them, and the acceptance
  benchmark, are the follow-on work.
- **A constrained-transcription layer, `pybnf.transcription` (#563, ADR-0109).** Infrastructure for
  restating a fit as a larger, better-conditioned problem with internal auxiliary variables and
  equality constraints that tie them back together — the reusable half of multiple shooting, which
  will be its first consumer. Four pieces: an **augmented variable layout** that carries the fit's
  reported free parameters and the transcription's internal blocks in one vector while keeping
  them rigorously apart (an auxiliary state is searched, bounded, and differentiated; it is never
  a reported fit result); an **equality residual/Jacobian interface** whose Jacobian is
  block-sparse with a condensing seam left open, and whose defects are scaled so one penalty means
  one thing across states of different magnitude; the augmented Lagrangian offered in all three
  forms PyBNF's optimizers consume (scalar for `lbfgs`, an *exact* stacked least-squares residual
  for `trf`, Gauss-Newton for `gntr`); and an **optimizer-agnostic augmented-Lagrangian outer
  loop** with a transcription homotopy and best-iterate certification through the ordinary
  single-shoot path.
  No behaviour change to any existing fit: nothing imports it yet, it defines no configuration key
  and no `job_type`, and it makes no simulator call — which is what lets the whole layer be
  verified offline (93 tests, ~1.4 s) against an equality-constrained quadratic whose multiplier is
  known analytically and a closed-form linear-ODE shooting problem measured against an
  independently computed single-shoot optimum.
  Three measurements from the #563 prototype are baked into the defaults rather than left as
  tuning advice, each contradicting the plan that preceded it: the penalty schedule starts **tight**
  (`rho0 = 10`, `gamma = 5` beat `0.1`/`3` on quality *and* halved the cost), the segment ladder is
  the **mechanism** rather than a later refinement and starts in the middle (`4-2-1`, not
  `8-4-2-1` — many short segments certified worse than their own start under partial
  observability), and a run reports its **best certified** iterate rather than its last (on one
  start the final stage held `-147.0` while an earlier iterate certified at `-196.3`).
- **`noise_profiling = 1`: profile an estimated noise scale out of the search analytically (#562,
  ADR-0108).** ADR-0066 already profiles a declared column's optimal multiplicative **scale** out
  of the fit; this is the other half of the same classical trick. Every noise parameter declared
  `= fit <parameter>` is removed from the search and replaced, at each evaluation, by its
  closed-form maximum-likelihood value over the scored points that share it — the weighted residual
  RMS `sqrt(sum w r**2 / sum w)` for the Gaussian families (`normal` / `lognormal` / `lnnormal`),
  the weighted mean absolute residual `sum w |r| / sum w` for `laplace`. Opt-in; `0` (the default)
  is an exact no-op.
  Why it matters beyond the dimension count: at a random point in the box the sampled scale is
  nowhere near its optimum, so the `log sigma` term dominates and a global sampler ranks candidates
  mostly by *how wrong their sigma happens to be* rather than by how well their dynamics fit. On
  `Borghans_BiophysChem1997` every optimizer that can run it converges to the same attractor, and
  that attractor is exactly the **no-dynamics solution** (a flat line at the best constant with
  sigma at the residual RMS, `-51.204092` analytically and `-51.204092` reported). Across the
  Grein et al. 2026 subset-I corpus a plain free-parameter sigma accounts for **32 parameters in 13
  of 23 slugs** — 4 % to 33 % of the search. A profiled scale also has no box to run into, so a fit
  can no longer optimize its sigma into an upper bound and absorb model misfit as "measurement
  noise" (`Schwen_PONE2015`'s `IR_obs_std`).
  The switch is all-or-nothing within a fit and is refused *before the run starts*, naming the
  reason, for anything without a closed form: a `formula` / `prediction_formula` / per-measurement
  sigma, a `student_t` `df`, the `neg_bin` dispersion, a `location = mean` prediction on a log
  scale, one free parameter serving as the scale of two different families, or a fit with nothing
  to profile. A **fixed** scale (a data column, `fix_at`, `relative`) is not searched, so it is
  simply left alone. Refused for the Bayesian samplers too: profiling *maximizes* the nuisance out
  where a posterior *integrates* it out, so the draws would not be posterior draws.
  Profiled parameters stay declared (the same `.conf` runs with and without the key; their bounds
  and prior become inert) and stay **estimated**, so they keep counting in `k` in
  `information_criteria.txt` — otherwise every AIC/BIC would shift between the two runs. Their
  fitted values are written to the new **`Results/profiled_noise.txt`** and echoed on the console,
  since a value the fit solves for rather than proposes is not a coordinate of the best parameter
  set and appears in no `sorted_params_*.txt` row.
  Gradient support comes free by the envelope theorem — `job_type = lbfgs` and `gntr` consume the
  exact scalar gradient with the sigma columns dropped, and no new forward sensitivity is needed.
  `job_type = trf` refuses a profiled fit (as it already refused a searched free scale): under
  profiling the least-squares residual norm is identically constant, so a trust-region residual
  model carries no information about the parameters.
- **`Results/method_chain.json`: which methods a run actually executed (#564, ADR-0107).** Written
  by every run — budget or no budget — it carries the chain the conf requested
  (`requested_methods`), the chain that ran (`executed_methods`), and one entry per phase (the fit,
  the refine, each bootstrap replicate) with its status (`completed` / `wall_time_expired` /
  `skipped`), its stop reason, its elapsed seconds, its completed simulations, and the best
  objective it reached. `requested_methods` longer than `executed_methods` is the machine-readable
  form of a downgrade, so a scoring harness can assert on the method it measured in one line
  instead of parsing stdout. A `bootstrap` phase records `replicates_requested` /
  `replicates_completed`, because `bootstrap = 30` in a conf is worth nothing if the budget stopped
  the run at 11. The file is rewritten after every phase (so a run whose refine raises still leaves
  the record of the fit that happened), is strictly valid JSON (a non-finite objective is recorded
  as `null`, never `Infinity`), and — like `stop_reason.txt` and `information_criteria.txt` — a
  failure to write it is logged and swallowed rather than taking a finished run down.

## [v1.7.0] - 2026-08-12

### Changed
- **An SBML model on `sbml_backend = bngsim` no longer charges every species for the tolerance its
  smallest one needs (#549, ADR-0105, supersedes ADR-0103).** ADR-0103 derived a single `atol` from
  the median species value because bngsim's `Simulator.run` took only a scalar, and it wrote down
  what that cost: `Brannmark_JBC2010` reads `3.3e-10`, which holds its `IR`/`IRS`/`X` species at ~10
  to `3.3e-11` *relative* — three decades tighter than the `rtol` that governs them — to buy a
  resolution for a `1.76e-9` transient that the same ADR had already decided not to chase. Now that
  lanl/bngsim#196 routes a vector to `CVodeSVtolerances`, that over-tightening is given back per
  species: `atol_i = clamp(sbml_rtol * y_i, the model's scalar, 1e-8)`. Measured on 100 points
  sampled from Brannmark's own fit box with the fit's sensitivity request applied, 39 dead
  simulations become **33**, in 428 s rather than 576 s.
  **The lower clamp is the change, and it is there because the obvious rule loses.** "Resolve each
  species to `rtol` of its own magnitude", full stop — which is what #549 proposes — puts that
  transient at `1.76e-17`, and on the same 100 points killed **91 of 100**: ADR-0103's withdrawn
  *minimum* rule reappearing one species at a time. The `1e-16` floor #549 asks about rescued
  nothing (91 either way), because the damage is done well above it. A tolerance below `rtol*|y|` is
  inert until a species has decayed far below its nominal value, and what it then demands is that a
  species which has decayed into nothing be resolved as if it had not; telling that apart from a
  genuinely tiny species needs the trajectory, not the initial values. So every entry now lies in
  `[the model's scalar, 1e-8]`: no species is integrated more tightly than PyBNF integrates it
  today, and no model that runs today can start failing. A control confirms the mechanism is the
  values and not the plumbing — the same scalar sent as a uniform vector reproduces the baseline
  exactly, 39 and 39.
  Over the 23-slug subset-I corpus 19 models take the same scalar call as today and the 4 that #546
  tightened take vectors, which is the shape of a refund: only a model that was charged can receive
  one. A species declared at zero has no magnitude of its own and falls out of the same expression
  at the model's scalar, leaving it where ADR-0103 put it.
  The derivation stays a property of the **model file** — read off the SBML document at load, held
  for the whole fit, never re-derived from the fit point, which is why bngsim's state-reading `AUTO`
  token is not used: a tolerance that moved with a fitted initial condition would put a step in the
  objective wherever the derivation crossed a rounding boundary, and it would be invisible, since
  the objective still looks correct and only the search behaves oddly. ADR-0103's median-derived
  scalar does not retire either — it becomes the steady-state convergence cutoff, passed explicitly
  whenever the vector is in force, because bngsim's own fallback for that cutoff is the Simulator's
  `1e-8` rather than anything derived from the vector, and taking it would silently return every
  `time = inf` measurement and every pre-equilibration phase to "equilibrium at t = 0" on a
  small-scale model. `sbml_atol` remains a single number and remains the off-switch: stating it
  integrates every species at that value and pins the pre-#196 code path bit-for-bit. A bngsim
  without the capability keeps ADR-0103's scalar unchanged, detected by name (`bngsim.AUTO`) rather
  than by version, because the build that first carried #196 declares the same version string as the
  wheel that predates it.

### Fixed
- **The CMA-ES restart battery's TolFun trigger no longer gets more eager as the fit gets better
  (#550, ADR-0106, amends ADR-0082).** ADR-0082 made TolFun's stagnation threshold *relative* to the
  current objective, `frange <= cmaes_stop_tol * max(1, |f|)`. PyBNF minimizes a negative
  log-likelihood, which is unbounded below, so `|f|` **grows** as the fit improves and that threshold
  rises as CMA-ES approaches the optimum — while Hansen's window `10 + ceil(30N/lambda)` shrinks as
  IPOP grows the population (30 generations at `lambda = 32`, 11 at `lambda ≈ 1900`). The two move in
  opposite directions and compound, so a late restart must improve by *more* within *fewer*
  generations than an early one, and IPOP's large-population restarts — the ones grown to do the
  heavy lifting — were the ones cut off mid-descent. On `Elowitz_Nature2000` (Grein subset-I, k=21)
  restart 3 was descending `OG` 53.0 → 26.4 → 5.105 and was killed at the bottom of that descent
  because `0.001/generation × 11 generations = 0.0105` fell just under `1e-4 × 121.06 = 0.0121`;
  across two fits **all 14 restarts** fired on TolFun and not one run ever converged. TolFun now
  compares an **absolute** range in objective units, as Hansen's `tolfun` does (pycma's relative
  variant `tolfunrel` normalizes by the run's *initial* median, a scale that does not drift with fit
  quality either; neither reference form uses the current `|f|`). On that fit the threshold becomes
  the configured `1e-4` and the descent clears it by a factor of 100.
  **TolFun also gains its own key, `cmaes_tolfun`.** `cmaes_stop_tol` is a step length in the
  parameter sampling space and TolFun is a range in objective units; no single value is right for
  both, and reaching a TolFun that fired at all on the fit above meant declaring the search
  distribution converged at `1e-4` in `u`, seven orders looser than the default. Unset, `cmaes_tolfun`
  follows `cmaes_stop_tol`, so an existing config keeps the threshold magnitude it had — the only
  change is dropping the `|f|` factor, which can only make TolFun fire less, and a fit whose
  objective satisfies `|f| <= 1` is unchanged outright. The restart reason now also reports the
  tolerance it used (`range 0.0105 over the last 11 generations, tolerance 0.0001`), so a restart's
  arithmetic is checkable from the log. `cmaes_restarts = 0` (the default) is untouched: the battery
  is still restart-gated (ADR-0070).
- **A PEtab v1 problem whose parameter table merely *has* a prior column no longer loses its log
  estimation scale (#548).** `petab1to2_preserve_scale` re-injects the `parameterScale` that
  `petab.v2.petab1to2` drops, skipping any row that already carries a prior so a scale petab1to2
  already folded into one is not clobbered. That guard is right in intent and impossible to
  implement in v2 alone: petab1to2 **materializes** v2's implicit default — `priorDistribution =
  uniform` over the bounds — into the converted table whenever the v1 table has a prior column at
  all, even an entirely empty one, and after conversion a materialized default and a declared
  `uniform` are the same cell. So the decider was whether the upstream TSV happened to carry a
  prior column, a cosmetic property of the file: `Zhao_QuantBiol2020` (four prior columns, 100%
  empty) lost **all 28** of its log10 parameters and `Schwen_PONE2014` (six real
  `parameterScaleNormal` priors, the rest blank) lost **24 of 25**, while `Giordano_Nature2020`,
  whose v1 table has no prior column, converted correctly. The conversion now reads which rows
  carried a prior from **v1**, where a blank is still a blank, and skips only those;
  `inject_log_uniform_priors` gains an optional `declared_prior_ids` and keeps the conservative
  v2-only reading when it is omitted. This was silent by construction: the re-injected prior sets
  only the search scale and initial sampling, and PyBNF's optimiser objective excludes the prior,
  so the objective, the `simulatedData` oracle check and the finite-difference gradient check all
  still passed — `Zhao`'s nominal `J_paper` is unchanged to 13 significant digits. Only the search
  was wrong, and a multi-decade parameter sampled linear-uniform presents as a fit needing more
  starts: `Zhao`'s `gamma_*` sit on `[1e-08, 1]` with an optimum near 0.05–0.39 and its `sd_*` on
  `[0.001, 1e5]` with MLEs of 186–5013, so across 28 parameters effectively no box-sampled start
  lands near the basin. A 100 × 1000 multi-start stalled at ~718 and was decelerating; on the
  corrected scale it beat that in under 90 seconds. `Schwen` is the discriminating case — its six
  declared priors survive as `log-normal`, its five genuinely `lin` parameters stay `uniform`.
- **An SBML model whose species are far below 1 no longer integrates — and differentiates —
  at a tolerance larger than its own state (#546, ADR-0103).** `Giordano_Nature2020`'s
  assembled gradient disagreed with central differences on 41 of its 50 fitted parameters, by
  up to 26%, identically at every finite-difference step size, with no refusal and no warning.
  The model is piecewise-in-time — 110 `piecewise` expressions across 14 assignment rules, all
  gated on the COVID NPI stage boundaries — and the error partitioned along whether a parameter
  sat behind a time gate, so it read as unhandled switching. It is not: bngsim's SBML loader
  already registers every `time` inequality as a CVODE root (13 for this model), and the
  crossings are landed on exactly. The defect is the **absolute** tolerance. CVODE weights each
  state by `rtol*|y| + atol`, so a constant `atol` declares values beneath it to be noise —
  a statement about the model's units, and bngsim's `1e-8` is BNG2.pl's, right for a model in
  molecule counts. Giordano is a population-*fraction* model whose species sit at `1.7e-8..1`,
  median `3.7e-7`: its early trajectory carries no significant digits, and the
  forward-sensitivity solve carries fewer still, since CVODES scales the state tolerances by the
  parameter magnitude for the sensitivity vectors. The gate correlation is real but incidental —
  a gated parameter acts only inside its own stage window, and the earliest windows are where
  the states are smallest. Tightening `rtol` by four decades changes nothing; tightening `atol`
  fixes it. The bngsim SBML/Antimony path now derives `atol` as `rtol` times the model's median
  strictly-positive species initial, clamped to at most the backend default and at least
  `1e-16`, so it can only ever tighten: across the 23-model subset-I corpus 19 are untouched and
  4 tighten. Giordano's worst column goes **7.7e-02 → 4.5e-04** for ~14% more wall clock;
  Brannmark 5.0e-05 → 3.6e-05 and Bertozzi 2.7e-05 → 2.4e-05 at no measurable cost. The median
  rather than the minimum, because `Brannmark_JBC2010` seeds one transient intermediate at
  `1.8e-9` against principal species at `0.1..10`, and resolving *that* asks for `1e-17`, which
  makes the model fail outright on `mxstep` at interior fit points. Unchanged: every BNGL/net
  model (its tolerances come from the actions block, and BNG2.pl parity is what that backend is
  measured against), every stochastic run, the RoadRunner backend, and every SBML model of
  order-one scale.

- **A scalar fit no longer re-derives the analytical Jacobian on every action (#544).** #543
  warmed the cached engine template so clones inherit the compiled sensitivity RHS, but it
  warmed only when a sensitivity request was active — and the same never-warmed-parent shape
  costs the **scalar** path too, one artifact over. bngsim's `Simulator.__init__` calls
  `model.prepare_analytical_jacobian()`; PyBNF builds that `Simulator` on the per-action
  *clone*, and the clone is discarded. `clone()` carries the `_jac_attempted` sentinel parent
  → child precisely so a derived parent yields cheap clones, but nothing ever derived it on
  the parent, so every scalar action re-ran the SymPy derivation from scratch. PyBNF now warms
  unconditionally and lets the *shape* of the warm depend on the request rather than gating
  the warm itself on there being one. Measured through PyBNF's own action path on the
  44-species `yeast_cell_cycle` model: **0.1542 s → 0.0057 s per action** (27x), derivations
  10 of 10 → 0 of 10. As reported on #544, `Smith_BMCSystBiol2013` goes 0.0401 s → 0.0224 s,
  20 of 20 → 0 of 20, taking that job's shipped 64,000-evaluation `cmaes` budget from ~36
  core-hours to ~14. Trajectories are bit-for-bit unchanged. Two guards this needed: a scalar
  warm must **not** satisfy a later gradient warm (bngsim clears a plain-RHS artifact and
  regenerates it at the first sensitivity request, so a scalar-warmed template would be
  correct and save the gradient path nothing) — the warm-state predicate and the per-shape
  attempt memo both answer "not yet" for the sensitivity shape; and a model with **no ODE
  action** warms nothing, since bngsim derives the Jacobian under ODE dispatch and nowhere
  else, having deliberately moved it off the load path so a stochastic run never pays SymPy.
  Applies to every SBML/Antimony fit through
  `sbml_backend = bngsim`, on every `job_type`; the larger the model the bigger
  the effect, since the derivation scales with the network and the solve does not. The `.net`
  backend clones from a held `_engine_model` in the same never-warmed shape and does re-attempt
  the derivation per evaluation (measured 4 of 4), but it costs it essentially nothing: a BNGL
  network is all-Elementary, so bngsim takes its closed-form C++ Jacobian rather than SymPy —
  0.1511 s → 0.1472 s per evaluation on `egfr_ground.net` (356 species), within noise. Left
  alone rather than warmed on a measurement that does not justify it.
- **A gradient fit no longer rebuilds the analytical sensitivity RHS on every action (#543).**
  `_get_engine_template` caches one loaded bngsim model per SBML text per worker process and
  #415 clones it per action, precisely so the parse and the derived Jacobian are paid once.
  bngsim's `clone()` cooperates, carrying the compiled sensitivity artifact parent → child —
  but the `Simulator` was built on the *clone*, so the clone was what discovered and recorded
  that artifact, and the clone is discarded at the end of the action. Discovery flowed
  child-ward only: the template's `_codegen_so_path` stayed empty forever, and every action
  regenerated the C source, and every symbolic derivative behind it, because bngsim keys its
  compiled `.so` on a hash of that source. PyBNF now warms the template once per process, with
  the sensitivity request the actions will use (a scalar-shaped warm would be correct and save
  nothing — bngsim regenerates it at the first sensitivity request), so `clone()` propagates it
  from then on. Measured on `Smith_BMCSystBiol2013` (133 species, 16 sensitivity columns)
  through PyBNF's own action path: **2.015 s → 0.537 s per action**, source generated 4 of 4
  times before and 0 of 4 after; the tensor is bit-for-bit unchanged. Inside a real `gntr` run
  of that job — dask worker, condition applied, the experiment's own sample times — its first
  action goes from 3.731 s to 0.327 s. Invisible in a profile
  that looks at simulation — the integration is untouched and all of the difference is
  `Simulator(...)` construction. A **scalar** (metaheuristic) fit never generates the source at
  all and was left untouched here (0.194 s per action either way), which bounded who *this*
  entry helps; #544 above warms it for a different artifact. The `.net` backend clones from a
  held `_engine_model` in the same never-warmed shape but is **not** affected here: its `.net`
  codegen memo is keyed on the file path rather than on generated source, and it regenerates
  nothing (measured 0 of 6).
- **A pre-equilibration condition that doses a species from a fitted parameter no longer reports a
  zero gradient column for it (#538, ADR-0101).** `preequilibrate:` applies its condition inline,
  so a species target becomes a `setConcentration` written *before* the first phase — and
  `Model.set_concentration` reads an assigned amount as a literal initial condition
  (`∂x_k(0)/∂θ = 0`), retiring whatever seeding the species' `.net` expression carried
  (lanl/bngsim#113). ADR-0098 supplies that row for a write between two phases; with nothing
  pending it had nothing to rebuild and left the write to the backend, so an amount like
  `"A()" = 2*k_deg` contributed **exactly zero** to `k_deg`'s derivative. Nothing failed and no
  refusal fired: the fit simply walked a wrong steepest direction to a plausible answer. PyBNF now
  *declares* the assignment's own `∂x_k(0)/∂θ` (`Model.declare_ic_sensitivity`, the API bngsim
  documents for a hand-assigned θ-dependent initial condition), so bngsim's own seeding starts from
  it — narrowly, only when a fitted parameter reaches the amount, so every protocol whose gradient
  was already right reaches the backend through the same calls as before. An `addConcentration`
  re-declares the row its constant shift left alone. Visible only with a **fixed-duration**
  equilibration (`equil_t_end:`, what the preincubate → wash → dose-scan protocols use); a
  steady-state equilibration relaxes the dose away, so the derivative is genuinely zero there.
  Also new: an intervention amount that reads a fitted parameter **no** requested
  forward-sensitivity column carries is now refused by name, on both this path and the
  mid-protocol one, rather than silently contributing a zero row.
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

### Fixed
- **The router now reads what the backend seeded instead of inferring it, so an IC-seeding
  parameter is routed exactly once (#537, ADR-0100).** bngsim 0.12.2 (lanl/bngsim#157, answering
  our lanl/bngsim#155) exposes `Model.effective_ic_sensitivity()`, the `{species: {param:
  ∂x(0)/∂θ}}` the solver will actually be seeded with, from model structure alone and with a
  present-but-zero entry distinguished from an absent one. The rule is now stated rather than
  guarded around: **route a bound id's own parameter axis, and add an initial-condition term only
  for a `(species, param)` pair the backend reports absent.** That subsumes both of the defects
  either side of it — #535 was an axis dropped that was needed, #537 an axis kept that
  duplicated, and both came from inferring what the parameter axis contained. `ode_rhs_symbols`
  is demoted to an optimization (dropping a provably identically-zero axis), so a model that
  cannot answer keeps its axis instead of facing two silent wrongs; the refusal added for that
  case is gone, as is the `lowered_ic_species` build discriminator that existed only to survive
  the interval. `Fiedler_BMCSystBiol2016` returns to 3.68e-06 on 0.12.2 — its value before
  lanl/bngsim#147 widened the seeded class — and seven slugs whose parameters seed an initial
  condition switch from the ic axis to the parameter axis, unchanged to the digit. **The minimum
  bngsim is now 0.12.2.** One limit is deliberate: the two axes are only interchangeable in a
  common unit convention, and the ic axis is rescaled by each species' PyBNF-value-to-
  concentration factor while the parameter axis is not, so a species whose factor is not 1 keeps
  the ic route (substituting overstates its column by `1/factor`, measured across three
  compartment sizes). Exactly one benchmark model has non-unit factors and it seeds nothing.
- **A gradient fit is refused, rather than silently doubled, on a bngsim build that seeds a
  lowered `initialAssignment` into the parameter axis (#537, ADR-0100).** The backend authors
  confirmed (lanl/bngsim#155) that `output_sensitivities(axis='parameter')` is the **total**
  derivative — `d_param[θ] = (RHS path) + Σ_j (∂x_j(0)/∂θ)·d_ic[x_j]` — so a route holding both a
  parameter's own axis and an IC term is correct only while the backend seeds nothing for that
  species. lanl/bngsim#147 changes that for compound parameter-only assignments, which it lowers
  to a synthetic `_ic_<species>` derived parameter. `route_for_model` now detects exactly that
  build-and-model combination and refuses by name, so `Fiedler_BMCSystBiol2016` — the one slug in
  23 whose free parameters legitimately route both axes, correct on every build through 0.12.1 at
  ≤3.7e-06 — fails loudly on the first build carrying #147 instead of reporting seven columns at
  the wrong value. The assembly's numeric guard was also generalized: it now compares the
  parameter slice against the **weighted sum** of the IC terms rather than a single slice, which
  catches a non-unit seed (`X(0) = a*X0` gives `d_param[X0] = 3.0·d_ic[X]`, agreeing to roundoff
  rather than bit-for-bit — the counterexample the first cut missed). The net backend is pinned to
  the same contract by test, having been verified bit-identical on `e2e_ode_decay.net`.
- **A parameter's own sensitivity axis is not "identically zero" when it seeds an initial
  condition — it is the whole derivative, and adding the IC axis to it doubles the column
  (#537, ADR-0100).** ADR-0097 drops the `sensitivity_params` axis of a parameter that only seeds species
  initial values, documented as sparing a wasted vector on an axis that "would be identically
  zero". Measured on `Raia_CancerResearch2011`, that axis is not zero: bngsim seeds `∂x(0)/∂p`
  into it as well (lanl/bngsim#43, widened to compound `initialAssignment` expressions by
  lanl/bngsim#147), so `d_param[init_Rec_i]` and `d_ic[Rec_i]` come back **bit-for-bit
  identical** across every output. The drop is therefore load-bearing for correctness, and
  forcing both axes into the route reproduces #537's signature exactly — `init_Rec_i` at
  2.00000× its central difference, every other column untouched. Three changes. The claim is
  corrected wherever it appears. The **unanswerable-model fallback is now a refusal**: ADR-0097
  kept the axis when `ode_rhs_symbols()` could not say, on the strength of that false premise,
  which means one error deletes half a derivative (#535) and the other doubles it — with no safe
  default left, the router names the parameter and points at a gradient-free `job_type`. Both
  shipped backends always answer, so no fit changes. And the assembly now **checks numerically**,
  per experiment: a route holding both its own parameter axis and an IC axis whose tensor slices
  are bit-identical is refused, since two independent derivatives of a live model do not coincide
  to the last bit. `Fiedler_BMCSystBiol2016`, which legitimately routes both axes for seven
  parameters (their columns genuinely differ — RHS path versus seeding), is unchanged at
  ≤3.7e-06.

### Changed
- **Tutorial Lesson 6 no longer teaches a refusal that stopped being one (#536).** Its whole
  premise was that a hard `if(t < tau, …)` step in a rate law is not differentiable, so `trf`
  refuses it and the fix is to smooth the step into a sigmoid. bngsim 0.12.2 differentiates it,
  and the lesson's three claims all failed when measured: the fit is not refused; its gradient is
  *correct* (agrees with a central difference to 3e-06 at a well-conditioned step — the apparent
  3e-03 disagreement at `h = 1e-6` grows as `h` shrinks, which is roundoff, not a defect); and
  `tau` — which the lesson said "the hard-`if()` model could never expose to a gradient" — is
  recovered to 1.2e-08, because the solver contributes the crossing term where the switch fires.
  `step_input_trf_refused.conf` is renamed `step_input_trf.conf`, now fits `tau` alongside the
  rest, and is checked as a *recovery* rather than a refusal.

  The refusal half of the lesson is kept rather than deleted, because Lesson 3 sends readers here
  for it and because "what does a gradient fit refuse" is still worth teaching — it moves to a new
  `step_input_ssa_refused.conf`, a scored `method: ssa` experiment. That reason is durable in a way
  the old one was not: an SSA trajectory is a random walk, so there is no derivative with respect
  to the rate parameters to carry, and forward sensitivities exist only for the ODE backend. The
  smooth-sigmoid variant stays, re-pointed from "the differentiable one" to a **conditioning**
  choice — both fit the same parameters to the same accuracy and differ in what they cost the
  integrator. Without this the nightly `recovery` job would have gone red on its own the moment
  bngsim 0.12.2 reached PyPI, since CI installs bngsim unpinned.

### Added
- **`sbml_rtol` and `sbml_atol` config keys (#546, ADR-0103).** State the CVODE tolerances for
  every deterministic run of every SBML/Antimony model in a fit, on the `sbml_backend = bngsim`
  path — which previously had no way to ask for either, since `atol`/`rtol` are read only out of
  a BNGL actions block and an SBML model has none. Unset (the default) leaves `rtol` at the
  backend default and derives `atol` from the model's own species magnitude, as described under
  Fixed above. Setting either under another `sbml_backend` is a config error rather than a
  silent no-op, and a model whose derived tolerance hits the `1e-16` floor is told once, by
  name, that `sbml_atol` is where to say so.
- **A chain of two or more data-level normalizations on one column is now differentiable (#539,
  ADR-0102).** `normalization` is an ordered chain (ADR-0066), and five of its transforms —
  `peak`, `init`, `zero`, `unit`, `floor` — rewrite the column in place. Stacking two of them
  (`normalization pStat = floor 0.03, peak`) was refused on every gradient `job_type`, because the
  sidecar recording *how* a column was normalized kept one record per column: the second transform
  overwrote the first one's facts, and its intermediate values were gone with them. The sidecar is
  now the **list** of a column's transforms in chain order, and the gradient folds it — each
  stage's chain rule is the same closed form it always was, read in that stage's own inputs (the
  previous stage's per-row sensitivities) rather than in the raw ones, so the fold wraps the
  sensitivity accessor once per stage. The values a stage produced, which its own rule reads, ride
  along on its record when the next transform overwrites them; a column normalized once — every
  fit in the wild — retains nothing and costs what it did before, and its gradient is the same
  arithmetic in the same order. Every normalized *value* is unchanged: this is about what is
  recorded alongside them, not what is computed. `floor 0.03, scale` is unaffected either way, and
  a chain of any length still composes with the analytic `scale` (ADR-0099), the cumulative and
  per-measurement transforms, and the EFIM path.
- **A gradient routing that would read one sensitivity column twice is now refused, not
  assembled (#537).** A route's derivative is the sum over its contributions, so two of them
  naming the same native `(axis, key)` column add that column twice — and the result is a clean
  integer multiple of the true derivative, which nothing downstream can detect: the objective
  stays finite, smooth and plausible, and the fit simply walks a scaled surface to a wrong
  answer. That is the shape a `Raia_CancerResearch2011` column came back in, once, during the
  #535 finite-difference sweep — exactly 2× its central difference on `init_Rec_i`, the fit's
  only initial-condition-axis parameter — and it has not reproduced in six runs at the same
  point, so there is no confirmed defect to fix. What there is now is a standing check on the
  narrow invariant it violated. Two halves: `route_experiment` **folds** terms that meet on one
  column into a single contribution carrying their summed derivative tree (the same sum, one
  tensor read, and `at_point` still refreshes a point-dependent path), which makes
  one-contribution-per-column structural rather than incidental; and the gradient, EFIM and
  constraint assemblers each `check_column_multiplicity()` on every routing they consume, once
  per experiment, raising a `PybnfError` that names the free parameter, the axis and key of the
  repeated column, and each duplicate's factor. The fold would otherwise blind that check —
  two same-column terms become one term of doubled factor, indistinguishable after the fact
  from a legitimate single term — so each contribution records the chain-rule path(s) it came
  from (`origins`: `bind`, or `ref:<target>` per condition parameter-reference) and the check
  reads those. One path reaching one column twice is the defect; two paths meeting on it is
  arithmetic, and only the labels separate them once the factors are summed. Provenance is
  metadata: excluded from equality, hash and repr, so a routing still compares by what it
  computes. Note the scope: this covers the routing and
  assembly half of #537's hypothesis. A doubled column arriving from the backend's own
  sensitivity tensor would still pass, and remains open on the issue.
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
  ordinary column in another. One boundary was stated here rather than silently mis-differentiated
  — a chain of two or more *data-level* transforms on one column (`floor 0.03, peak`), which kept
  only the last one's facts — and is closed by #539 above.
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
- **A model with discrete events fits on the gradient path (#536).** An SBML `event` — and so an
  Antimony `at (…): …`, the usual way a dosing or stimulation schedule is written — used to refuse
  `trf` / `lbfgs` / `gntr` at construction. That refusal (#461) was right when it was written: a
  forward sensitivity carried across a state jump is correct only if the solver applies the event's
  own jump `s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p` at each fire, and bngsim did not, so it
  refused sensitivities on any event-bearing model and PyBNF hoisted that refusal to a clean
  pre-flight gate. bngsim applies the jump now, across a fixed trigger time, a trigger whose
  threshold is a fitted constant (lanl/bngsim#49), and a state-dependent trigger whose crossing it
  differentiates in flight (lanl/bngsim#144). The gate is therefore a
  **capability check** rather than a blanket structural refusal: an event-bearing model is allowed
  through, and the subclasses the build genuinely cannot cross (an execution delay; a trigger that
  is not a single relational comparison) keep a per-simulation refusal naming the reason.

  **The floor is set by silent wrongness, not by a missing feature.** A build that refuses is
  safe; a build that answers an event it cannot actually differentiate, without saying so, is the
  thing this gate exists to prevent. Three such answers had to go first, which puts the floor
  *newer than bngsim 0.12.1*: a trigger reading the state came back as a finite tensor with the
  event's contribution missing rather than being refused (lanl/bngsim#52, through 0.11.x); an
  assignment reading the state — `A := A + dose`, the repeat-dosing idiom — dropped the carried
  `∂h/∂x·s⁻` and restarted the assigned row from zero, measured on 0.12.1 at `-10.96` against the
  model's own central difference of `-311.20` while the identical model built through
  `ModelBuilder.add_event` was right to `2e-6` (fixed after 0.12.1 by lanl/bngsim#144's
  jump-handler rework); and a solver root that fires nothing rewound the state but not the
  sensitivity history (lanl/bngsim#146, also after 0.12.1). On 0.12.1 or older the refusal stays,
  and stays blanket, with the message naming the upgrade. The floor lives in
  `pybnf._bngsim_caps` (`BNGSIM_HAS_EVENT_SENS`) and is not a dependency bump: the install floor
  stays `bngsim>=0.11.35`, and every scalar (metaheuristic) fit is unaffected either way.

  The SBML backend also gained a narrowed form of the net backend's #525 wrapper, which the lifted
  gate makes reachable: bngsim declining to differentiate a model's events is a *structural*
  verdict, identical at every parameter set, so it now surfaces as an actionable `PybnfError`
  instead of a `FailedSimulationError` the optimizer scores `inf` and steps around — which would
  have reported an unsupported event as a failed search. Every other backend failure keeps that
  back-off, which is the right answer for a candidate point the integrator cannot get through
  (#492).

  New coverage is a finite-difference oracle on a two-species event fixture at both levels, the
  backend tensor and the assembled objective gradient (`tests/test_gradient_events.py`): every
  sensitivity column is scored against a central difference of PyBNF's own trajectory / own loss,
  including the columns that cross the jump, which is the only instrument that catches a jump term
  that is missing rather than merely inaccurate. The backend-level oracles run on every build,
  since what bngsim computes is worth asserting whether or not PyBNF admits the model; the fits,
  and the bolus-assignment case that set the floor, are gated on `BNGSIM_HAS_EVENT_SENS`.
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
