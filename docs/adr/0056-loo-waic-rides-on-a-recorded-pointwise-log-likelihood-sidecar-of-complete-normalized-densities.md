# LOO/WAIC rides on a recorded pointwise-log-likelihood sidecar of complete, normalized densities (issue #438 item 4)

**Status: Accepted (implemented 2026-06-26).** Closes **item 4** of #438 (the low-cost
Stan-parity shortlist): model comparison from pointwise log-likelihoods. Builds directly on
ADR-0055 (the ArviZ `InferenceData` bridge — item 3, which set this up as "a purely additive
follow-on") and on ADR-0011/0021/0022 (the per-point `(family × σ-source)` noise engine that
already computes the per-observation likelihood terms LOO/WAIC need).

## Why

`az.loo` (PSIS-LOO-CV) and `az.waic` estimate a model's out-of-sample predictive accuracy — and
`az.compare` ranks models — from the **pointwise** log-likelihood: one genuine log-density
`log p(y_i | θ)` per observation `i` per posterior draw `θ`, shaped `chain × draw × obs`. ADR-0055
gave the bridge `posterior` + `sample_stats` (the total `lp`); it deliberately deferred
`log_likelihood` because `samples.txt` stores only parameters + the *summed* `Ln_probability`,
not the per-point decomposition. This ADR adds that decomposition. It is feasible because
PyBNF's noise engine is *already* a per-point negative-log-likelihood kernel (ADR-0011): every
likelihood evaluation walks the data point by point. The work is to **not throw the decomposition
away**, then to make each term a *complete* density (§3).

The design had four forks (mirroring ADR-0055's structure); each was resolved with the issue
author and is recorded below.

## The four decisions

### 1. Source — record the pointwise vector **during the run**, not re-simulate post-hoc

The pointwise log-likelihoods could come from either (A) recording them as a byproduct of the
scoring that already happens during sampling, written to a sidecar the bridge reads, or (B)
reconstructing the config + simulator in `from_pybnf` and re-running one simulation per saved
draw. We chose **(A)**:

- **It is the cheaper option overall, by far.** (A) re-walks the simulation data already in hand
  at each accepted draw — `O(num_points)` arithmetic, microseconds, **zero** extra simulations.
  (B) re-simulates every saved draw (thousands of full ODE/SSA solves) and pulls the entire
  `Job`/`core`/dask stack into what ADR-0055 deliberately kept a *pure disk parse*.
- **It is idiomatic.** PyBNF records-during-run for *everything* — trajectories
  (`output_run_all`), constraint satisfaction (`constraint_samples.txt`), the samples themselves
  (`samples.txt`). There is no post-hoc re-simulation machinery, and this does not add one.
- **It does not touch sampling efficiency.** The samplers' proposals/accepts/RNG are untouched,
  so the draws are byte-identical; the only run-time cost is the per-accept re-walk and one extra
  file append **at the existing `sample_every` cadence** (`sample_pset` already appends to
  `samples.txt`/`constraint_samples.txt` there) — not a new per-iteration stream.

The cost is that **only runs that recorded the sidecar get LOO** — an archived run finished before
this (or with the key off) yields an InferenceData without the group. That is honest: the bridge
offers `az.loo` exactly where the data for it exists, and omits it (no error) elsewhere.

**Mechanics.** Exactly parallel to constraint satisfaction: the accepted pset's pointwise vector
is cached per chain at accept time — `record_pointwise_loglik(res, index)` beside
`evaluate_constraints(res.simdata, index)` in each sampler's accept branch (`am`/`dream`/`basic_mcmc`;
`p_dream` inherits `dream`'s) — and the *current* chain's cached vector is written by the shared
`sample_pset` at each sample iteration. So `log_likelihood.txt` gets **one row per saved sample,
in the same order as `samples.txt`** → the two are row-aligned, and the bridge reads them in
lockstep (sidecar row `i` ↔ samples row `i`, the position counter advancing on every data line so
a corrupt samples row cannot shear the alignment). The sidecar's `#` header lists the observation
ids (`model/suffix/observable@indvar=value`); the bridge reshapes by the *same* chain grouping the
posterior uses, names the group's single variable `y` over an `obs_id` axis carrying those labels,
and a single bad/missing row drops the whole group (with a warning) rather than build a misaligned
array — the posterior is never left ragged.

### 2. Gating — the **one** existing key `output_inference_data`, not a new key

`output_inference_data` (ADR-0055) already opts a run into the InferenceData artifact. Setting it
now *also* turns on the pointwise recording and includes the `log_likelihood` group — one switch,
matching ADR-0055's "users won't use it if they must do something special." The cost (accepted) is
that `output_inference_data` gains a small *during-run* component (the accept-branch re-walk) on
top of its post-hoc `.nc` emission; the standalone `from_pybnf` still picks up the sidecar whenever
it is present, with or without the key.

### 3. Values — the **complete, normalized, unweighted** per-point density, not `−score`

The recorded value at each point is the noise family's genuine log-density
`log p(y_i | prediction, noise)` — **not** `−eval_point`. Two corrections matter, both because a
*predictive density* needs what a *sampler* never did:

- **The constants the objective drops.** `eval_point` is built for the sampler, which needs only
  likelihood *ratios*, so it omits every parameter-independent constant: Gaussian's `½log(2π)`, and
  — for a family additive on a log scale (lognormal) — the change-of-variables Jacobian. A density
  must keep them, or absolute WAIC/LOO are off and a **cross-family** comparison (Gaussian vs
  Laplace, the canonical robustness check) is *biased*. So the noise engine grew a
  `NoiseModel.log_density(prediction, observation, noise)` that restores them: a `_density_constant`
  hook (`0`; Gaussian's `½log(2π)`) and the scale's `log_abs_dforward(x)` Jacobian (`0` linear,
  `−log x − log ln10` for log10, `−log x` for ln). It is verified point-for-point against
  `scipy.stats.norm/lognorm/laplace.logpdf` and `nbinom.logpmf` — the oracle each family documents.
  (Laplace-linear and NegBinomial's `nll` were already complete, so the base `−nll` covers them.)
- **Weights are a fitting device, not the model.** PyBNF multiplies each point's term by
  `exp_data.weights`; the recorded density is **unweighted**, because a weight is not part of the
  generative likelihood `az.loo` reasons about (with default unit weights this is moot).

Consequence: these per-point values do **not** sum back to `−score`/`Ln_probability`. That is
intended — they are the honest densities LOO/WAIC consume, not a reproduction of the objective.

### 4. Scope — emit the **group**, not native statistics; no-op + warn off the likelihood family

We add the `log_likelihood` group and stop there: users call `az.loo`/`az.waic`/`az.compare`
themselves. This matches #438's framing ("not new statistics — a format bridge") and is robust to
arviz's own churn — note arviz **1.x dropped top-level `az.waic`** (PSIS-LOO supersedes WAIC, which
is *why* it was removed); the group powers `az.loo`/`az.compare` on both 0.x and 1.x and `az.waic`
wherever a user's arviz still ships it. PyBNF ships no `pybnf.loo()` wrapper.

Pointwise log-likelihoods are only meaningful for a genuine per-point likelihood. The gate is
`ObjectiveFunction.supports_pointwise_log_likelihood` — `True` only on `LikelihoodObjective` (the
ADR-0011 noise families: `chi_sq`, `chi_sq_dynamic`, `lognormal`, `laplace`, `neg_bin`,
`neg_bin_dynamic`, and the modern `objective`/`noise_model` surface). A least-squares
(`sos`/`norm_sos`/…), distance (`kl`/`wasserstein`), or pass-through (`direct_pass`) objective has
no normalized density to leave one out of, so `evaluate_pointwise` returns `None`: the run warns
once, records no sidecar, and the InferenceData simply omits the group. LOO is offered exactly
where it is valid.

## What this is not

- **Not a retrofit for old runs.** No sidecar → no group (decision 1). A user wanting LOO on a
  finished fit re-runs it with `output_inference_data` set.
- **Not `prior` / `observed_data` / `posterior_predictive` groups.** Still deferred; only
  `log_likelihood` is added here.
- **Not a change to any score.** `evaluate`/`evaluate_multiple` are untouched (the shared
  `_sim_row_for` extraction is behavior-preserving); the densities live in a new
  `evaluate_pointwise` path read only when the sidecar is recorded.

## Consequences

- One sidecar file (`Results/log_likelihood.txt`), one `NoiseModel.log_density` (+ a scale
  Jacobian and a Gaussian constant), one `ObjectiveFunction.evaluate_pointwise`, a per-chain cache
  + write in the sampler base, three one-line accept-branch calls, and a lockstep parse + group in
  the bridge. No new config key, so the config oracles are untouched.
- `az.loo`/`az.waic`/`az.compare` work directly on a PyBNF run done with `output_inference_data` +
  a likelihood objfunc — the Stan `loo` ecosystem, for the cost of not discarding a decomposition
  PyBNF already computed.
