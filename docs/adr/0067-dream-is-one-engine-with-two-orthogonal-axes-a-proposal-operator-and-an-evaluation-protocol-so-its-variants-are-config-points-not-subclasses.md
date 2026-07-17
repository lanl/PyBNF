# DREAM is one engine with two orthogonal extension axes — a proposal operator and an evaluation protocol — so its variants are config points, not subclasses (issues #357, #358)

**Status: Accepted (2026-07-16); Stages 1–2 implemented (2026-07-17).** Design accepted;
Stage 1 (proposal Strategy / P-DREAM fold-in) and Stage 2 (`n_try`, Multi-Try DREAM) are
shipped, Stage 3 (`kalman`) remains the build plan below. The scheduler-contract risk that
gated acceptance was pressure-tested and resolved — see *Principal risk*. Reframes the DREAM
family so that DREAM(ZS),
Preconditioned DREAM, the requested Multi-Try DREAM (#357), and the requested
Kalman-inspired DREAM (#358) stop being (or becoming) four sibling `Algorithm`
subclasses and instead become points in a small two-axis configuration space over
**one** `DreamAlgorithm` engine:

1. **Axis 1 — the proposal operator** (`proposal = de | whitened | kalman`): how a
   move is proposed given the ZS archive and the current state. A pure Strategy
   swap. Preconditioned DREAM already proves this seam works.
2. **Axis 2 — the evaluation protocol** (`n_try`, and the archive's *contents*): how
   many evaluations a chain spends per generation, and what data the archive carries
   back. This is the axis that touches the engine's scheduler contract, so it is
   generalized in the shared base with the classic behavior as the default.

The variants are then products of the two axes — `dream` ≡ `(de, n_try=1)`,
`p_dream` ≡ `(whitened, n_try=1)`, MT-DREAM(ZS) ≡ `(de, n_try=k)`, DREAM(KZS) ≡
`(kalman, n_try=1)` — and combinations the literature already names but a subclass
tree cannot express (MT-DREAM(ZS) is *literally* multi-try × parallel-DE; Laloy &
Vrugt 2012) fall out for free.

This ADR fixes the **model and the invariant**; it is a design decision, not an
implementation. The load-bearing invariant every later stage is measured against:
**`job_type = dream` at defaults (`proposal = de`, `n_try = 1`) is byte-identical to
today's `DreamAlgorithm`, and `job_type = p_dream` at defaults (`proposal =
whitened`) is byte-identical to today's `PDreamAlgorithm`.**

## The gap this closes

The engine that makes DREAM DREAM — a population of chains, a generation barrier
(`wait_for_sync`), proposals drawn from a growing ZS archive, the Metropolis–Hastings
accept, adaptive crossover (CR) and jump-scale (γ) selection, snooker updates
(ter Braak & Vrugt 2008), outlier detection/reset, and the rank-normalized
split-R̂/ESS diagnostics (Vehtari et al. 2021) — already lives once, in
`DreamAlgorithm.got_result`. `PDreamAlgorithm` (Preconditioned DREAM, Haario-style
covariance whitening of the DE difference; Haario et al. 2001) inherits all of it and
overrides *only* the proposal (`calculate_new_pset`) plus a one-line covariance-update
hook. So the family is already ~80% unified in practice; what is missing is a *named
model* that says so, and a place to hang the two open feature requests without
sprouting subclasses.

The status quo trajectory is a combinatorial subclass tree. Adding #357 and #358 as
`MTDreamAlgorithm` and `KZSDreamAlgorithm` does not just add two classes — it forbids
the cross-products. Multi-Try is defined in the literature *as a wrapper around*
DREAM(ZS) (Laloy & Vrugt 2012), and a multi-try Kalman or multi-try preconditioned run
is a coherent, nameable configuration; a subclass tree can express `MT` **or** `KZS`
**or** `whitened` but not `MT × whitened` without a new leaf per combination. The two
axes are orthogonal by construction, so they must be modelled as axes, not as a flat
enum of pre-combined names.

## The two axes sit at different seams

The reason this is a clean split — and not one uniform "pick a variant" flag — is that
the two axes touch the engine at architecturally different depths.

- **A proposal is a leaf operator.** `proposal ∈ {de, whitened, kalman}` swaps only
  *how a candidate vector is built* from the archive and current state; the barrier,
  the accept criterion, the CR/γ machinery, snooker, outliers, and diagnostics are all
  untouched. `PDreamAlgorithm.calculate_new_pset` is the existence proof: whitened-DE is
  a drop-in for parallel-DE with no change anywhere else, and it even *falls back* to
  `de` before its covariance estimate warms up. **Snooker is not a proposal value** — it
  is a second operator already blended in stochastically at rate `snooker_prob` inside
  the barrier and composes with any base proposal, so it stays a mix-in, not an entry in
  the `proposal` enum. Axis 1 is therefore a textbook Strategy: a `proposal` key
  selecting the primary (non-snooker) operator, default `de`.

- **The evaluation protocol is the engine contract.** Axis 2 changes the barrier's own
  assumptions, so it cannot be a leaf swap:

  - **`n_try` (Multi-Try, #357)** generalizes "one evaluation per chain per generation,
    accept inline the moment it lands" to "*k* evaluations per chain, buffered, then one
    selection ∝ posterior density and a reference-set acceptance" (Laloy & Vrugt 2012).
    That relocates the accept decision from the per-`Result` path
    (`dream.py:277`) to a per-chain-complete path, turns `wait_for_sync[i]` from a bool
    into an all-*k*-in counter, and needs a try-indexed job name so `k` psets can be
    tagged to one chain in a single `next_gen` batch. At `n_try = 1` every one of those
    generalizations saturates immediately and the path must be byte-identical to today
    (the invariant above). **This is the sole change of the four that can perturb the
    classic path, and is the design's principal risk** — see *Principal risk*.

  - **The archive's contents** must grow for `proposal = kalman` (#358): the
    Kalman-inspired proposal steers candidates using the cross-covariance
    `C(Z, f(Z))` and output covariance `C(f(Z), f(Z))` between archived *parameter*
    vectors and their *model outputs* `f(Z)` (Zhang, Vrugt et al. 2020). Today the
    archive stores only PSets and the accept path sees only a scalar score. The archive
    must therefore optionally carry each entry's output vector. **This is far cheaper
    than #358's own issue text assumes**: `_result_simdata` (`base.py:312`, added for
    #480) already extracts the full simulation output off a `Result` regardless of
    scoring path, and the "cache the accepted state per chain at accept time, consume it
    at the barrier" pattern already exists for constraint satisfaction
    (`current_constraint_satisfied`) and pointwise log-likelihood
    (`current_pointwise_loglik`, `base.py:213`). An archive-of-outputs is that same
    established pattern applied to the simulation vector — not new data plumbing.

## Decision

- **Keep `dream` and `p_dream` as public `job_type` codes over one class.** The
  internal `fit_type` registry already binds several public codes to one class with
  differing `kwargs` (`mh`/`pt`/`sa` → `BasicBayesMCMCAlgorithm`;
  `register_fit_type`, `registry.py:61`). `PDreamAlgorithm` collapses into
  `DreamAlgorithm`, and `p_dream` re-registers as the same class with its proposal
  pinned (`kwargs={'proposal': 'whitened'}`) plus its one extra key
  `precondition_adapt`. No user who writes `job_type = p_dream` is affected; the
  documented `DREAM(ZS)P → P-DREAM` naming history is preserved. This is a
  behaviour-preserving refactor, gated on the byte-identical invariant.

- **Axis 1 is a `proposal` key**, default `de`, valid values `de | whitened | kalman`,
  selecting the primary proposal operator. `whitened` requires `precondition_adapt`;
  `kalman` requires `kalman_burnin_frac` (default `0.3`, the fraction of burn-in over
  which the Kalman proposal is active before the run reverts to `de` and renormalizes
  the parallel-direction/snooker split, per Zhang et al. 2020). Snooker remains the
  `snooker_prob` stochastic mix-in, orthogonal to `proposal`.

- **Axis 2a is an `n_try` key**, integer, default `1`. `n_try = 1` is the classic
  single-try engine; `n_try > 1` selects the Multi-Try barrier. It composes with every
  `proposal` value.

- **Axis 2b is *implied*, not a user knob.** The output-augmented archive turns on
  because the *proposal declares it needs outputs* (`kalman`), not via a separate flag —
  storing per-entry output vectors for a proposal that never reads them would spend
  memory/bandwidth for nothing. The proposal operator exposes a "requires archived
  outputs" property the engine reads at `start_run`.

- **The config surface stays minimal and orthogonal.** Two new user keys (`proposal`,
  `n_try`) plus two proposal-scoped keys (`precondition_adapt` already exists;
  `kalman_burnin_frac` is new). No mega-enum of pre-combined variant names.

- **Sequencing is three staged, independently-shippable steps** — this ADR does not
  authorize implementing all of them at once:
  1. **Refactor only. — DONE (2026-07-16).** Extract the proposal Strategy seam; fold
     `PDreamAlgorithm` into `DreamAlgorithm` as `proposal = whitened`. No new behaviour.
     Acceptance = the byte-identical invariant, checked against the existing DREAM/P-DREAM
     oracle suites and the effective-config goldens (all green). Implementation note:
     `p_dream` pins `proposal = 'whitened'` via a `PDreamConfig` schema default rather than
     the registry `kwargs` sketched above — `kwargs` is constructor injection
     (`entry.cls(config, **kwargs)`), whereas a schema default co-locates the pin with the
     method (ADR-0006) and needs no constructor parameter. `PDreamAlgorithm` survives as a
     thin subclass (no logic of its own) so the public class name and `job_type = p_dream`
     resolve unchanged; `precondition_adapt` stays a `PDreamConfig`-owned key.
  2. **`n_try` (#357). — DONE (2026-07-17).** Generalized the barrier into a two-phase
     per-chain state machine (TRIALS → select ∝ importance weight → REFERENCE → multiple-try
     accept), `2k − 1` evaluations per chain per generation, confined to `DreamAlgorithm`'s
     `got_result` (split into a shared `_run_barrier` / `_advance_generation` plus a
     `_got_result_multitry` path), the proposal methods' optional base override, the try/ref
     job-name suffix, and the additive `n_try` key — as the *Principal risk* predicted. `n_try = 1`
     is byte-identical (the DREAM/P-DREAM oracle suites + config goldens pass unchanged). Multi-Try
     composes with the snooker mix-in: because the snooker proposal is non-symmetric its multi-try
     candidate/reference weights carry the ter Braak & Vrugt (2008) Jacobian `‖p − z‖^(d−1)`, and
     the current-state reference slot uses `‖x − z_Y‖^(d−1)` at the **selected candidate's** anchor
     `z_Y`. That current-slot term is the one subtlety of the method: it is the unique choice that
     reduces to the published single-try snooker ratio at `k = 1`, and it was pinned by
     first-principles derivation (the Liu-Liang-Wong importance weight `w = π/q` with the snooker
     transition density `q(x→x_p) ∝ ‖x_p − z‖^{−(d−1)}`) and confirmed by a stationary-distribution
     test — notably, both the DREAM-Suite (which zeroes it) and PyDREAM (which adds the winner's own
     term) reference implementations depart from it, in ways whose bias shrinks with `k` and so
     hides at the usual `k = 5`. Correctness validated by moment recovery on a Gaussian target with
     the snooker update active (`tests/test_multitry_dream.py`).
  3. **`kalman` (#358).** Add the Kalman proposal + the output-augmented archive
     (implied) + the burn-in switch, on the clean base.

### Principal risk — pressure-tested, resolved (2026-07-16)

Stage 2 (`n_try`) is the only change that can touch the byte-identical invariant,
because it relocates the accept decision from per-`Result` to per-chain-complete and
generalizes `wait_for_sync` and the job-naming scheme. A trace of the actual scheduler
contract (`Algorithm.run`, `base.py:1156`) settles that Multi-Try folds into the shared
barrier rather than needing a quarantined branch:

- **The scheduler is event-driven and population-size-agnostic.** The run loop pulls one
  completed future at a time (`f, res = next(pool)`, `base.py:1161`), calls `got_result`
  with that single `Result`, and submits back exactly the psets `got_result` returns —
  zero, one, or many (`base.py:1182`, `pool.update`). Nothing assumes the in-flight job
  count equals `population_size`. Submitting `k` proposals for one chain in a generation
  is a non-event for the scheduler: they become `k` independent futures in `pending`,
  each routed back tagged with its own candidate pset (`result_from_completed`,
  `base.py:1168`). Assumption (d) holds with **no contract change**.

- **Many-jobs-per-decision is already a supported shape.** Smoothing and
  `parallelize_models` already submit multiple jobs per logical evaluation and defer the
  decision until the set completes, via `JobGroup` accumulation (`_fold_group_result`,
  `base.py:1172`). Multi-Try is the same shape (buffer `k`, then decide); it buffers in
  `got_result` rather than at the job-group layer (the `k` tries are distinct candidates,
  not replicates to average), so it breaks no new ground.

- **The name parser already tolerates a try suffix.** `_chain_index_from_name` is
  `(?<=run)\d+` (`base.py:292`): `iter5run3try2` → `3`, because `\d+` stops at `t`.
  `make_job` keys the job on `params.name` (`base.py:873`), so distinct try-names give
  distinct job ids that never collide in `pending`. Assumption (a) holds — and the suffix
  is appended only when `n_try > 1`, so the `n_try = 1` name stays exactly `iter%irun%i`.

- **Everything that changes is inside `got_result`.** The all-`k`-in counter that
  replaces the `wait_for_sync[i]` bool (b) and the accept relocated from the per-`Result`
  path to the per-chain-complete path (c) are both local to `DreamAlgorithm.got_result`;
  the scheduler never sees them. At `n_try = 1` the buffer-of-one and counter-to-one
  reduce to today's inline accept, and returning `[]` until a chain's turn completes is
  exactly today's `wait_for_sync` behaviour (`base.py:1182` already handles an empty
  `decision` as a no-op `pool.update`).

**Verdict:** (a), (b), (d) are provably inert at `n_try = 1` and require no code outside
`got_result` and the naming string; only (c) — collapsing the buffer-of-one selection
back to today's exact append / CR-distance / snooker-correction sequence — is a genuine
local refactor, and its target is precisely the byte-identical DREAM/P-DREAM test suite.
Multi-Try is therefore **folded into the shared barrier; no quarantine branch is
needed**. This settles the *scheduler-contract* question only — the correctness of the
Multi-Try acceptance math itself (the reference-set Metropolis ratio and the detailed
balance of the ∝-posterior selection, which reduces to standard MH at `k = 1`; Laloy &
Vrugt 2012) remains Stage 2's own validation against a stationary-distribution oracle.

## Considered Options

- **Let the subclass tree grow (add `MTDreamAlgorithm`, `KZSDreamAlgorithm`).**
  Rejected: it cannot express the cross-products the literature names (multi-try
  DREAM(ZS) is the canonical form; multi-try preconditioned/Kalman runs are coherent),
  and every new axis multiplies the leaf count. The variants are orthogonal, so the tree
  is the wrong shape.

- **One class, one `job_type` mega-enum** (`dream | p_dream | mt_dream | kzs_dream |
  mt_p_dream | ...`). Rejected: it merely relocates the combinatorial explosion into the
  `job_type` namespace and hides that the axes are independent. `n_try = 3, proposal =
  whitened` is self-describing; `mt_p_dream` is not, and the enum must enumerate every
  legal pairing up front.

- **Make Multi-Try just another `proposal` value.** Rejected as a category error:
  multi-try is not a proposal, it is an evaluation protocol that *wraps* a proposal
  (Laloy & Vrugt 2012 define MT over DREAM(ZS)'s proposal). Folding it into the
  `proposal` enum would structurally forbid multi-try-whitened and multi-try-Kalman, the
  exact cross-products this ADR exists to keep expressible.

- **Drop `p_dream` and expose only `job_type = dream` + `proposal`.** Rejected:
  `p_dream` is a public, documented code with a recorded rename history; removing it
  churns users for zero architectural benefit. The `mh`/`pt`/`sa`-style `kwargs` alias
  keeps it a one-line registration over the unified class.

- **Store model outputs in the archive unconditionally.** Rejected: only the Kalman
  proposal reads `f(Z)`; carrying output vectors for `de`/`whitened` runs spends memory
  and per-generation copy cost for data nothing consumes. Gate it on the proposal's
  declared data requirement instead.

- **Implement all three stages under one change.** Rejected: DREAM's detailed balance
  is subtle (the snooker Hastings correction, `dream.py:182`; the reject-don't-reflect
  bounds convention) and the engine has not had the thorough benchmarking a Bayesian
  core warrants. The staged sequence keeps each step measurable against the
  byte-identical invariant (Stage 1) or a stationary-distribution oracle (Stages 2–3),
  and lets the refactor land and bake before either feature rides on it.

Relevant ADRs: **0006** (co-located per-method Pydantic config models and the stacked
`register_fit_type` multi-code registration this reuses), **0009** (the diagnostics /
proposal-math seam split the proposal Strategy extends), **0013** (the intra-family
`step_size` ↔ `adaptive_step_size` coupling that `DreamConfig.postprocess` already
owns, the template for proposal-scoped key coupling). The output-augmented archive
reuses the `_result_simdata` extraction and per-chain accept-time caching introduced for
**#480**. Literature: Vrugt (2016, *Environmental Modelling & Software* — DREAM(ZS));
ter Braak & Vrugt (2008, *Statistics and Computing* — snooker); Haario et al. (2001,
*Bernoulli* — adaptive covariance, the preconditioning lineage); Laloy & Vrugt (2012,
*Water Resources Research* 48, W01526 — MT-DREAM(ZS)); Zhang, Vrugt et al. (2020,
*Water Resources Research* 56, e2019WR025474 — DREAM(KZS)); Vehtari et al. (2021,
*Bayesian Analysis* — rank-normalized split-R̂ / ESS). Addresses issues **#357** and
**#358**.
