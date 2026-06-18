# Median is the universal prediction-centering default; the objective surface is `edition`-gated (issue #424)

ADR-0011 made the **location interpretation** (mean / median / mode — which
summary of the noise distribution the deterministic prediction is taken to be) a
first-class axis of a `NoiseModel`; ADR-0024 exposed it on the native surface and
declared median "the default … consistent with PEtab v2." But the **code
contradicted the doc**: `Gaussian.__init__` defaulted `location=MEAN` while
`Laplace.__init__` defaulted `MEDIAN`, and `location.py`'s own docstring called
median the default. The inconsistency was *latent* (invisible) only because every
live path was independently pinned — linear Gaussian/Laplace are symmetric
(`_Linear.mean_offset == 0.0`), `lognormal` passes `MEDIAN` explicitly, and
`neg_bin` is mean-by-parameterization — so the two opposite class defaults never
produced a different number. It would have surfaced the moment a mean-centered
log model or `neg_bin` divergence was exercised. Issue #424 is the policy that
settles this deliberately rather than per-PR.

Two facts decided the policy. **First, PEtab v2 mandates the median universally.**
The accepted spec (Pathirana et al. 2026, §9.2; also petab.readthedocs.io) states
it once for *all* noise distributions: *"y := observableFormula the simulated
value (the median of the noise distribution)."* Triangulated against the
reference implementation — `libpetab-python`'s `calculate.py` computes the
log-normal residual as `log(m) − log(sim)` with **no `σ²/2` moment correction**,
which is median-centering proven in code. PEtab v2 carries exactly four noise
families (normal / log-normal / laplace / log-laplace — **no `neg_bin`**).
**Second, the discussion established that nearly every PyBNF objfunc *is* a
likelihood**: we minimize a negative log-likelihood with parameter-independent
constants dropped, which is exactly why a fixed-σ model *looks* like a bare
metric. `sos` = Gaussian with σ≡1 (the `½` and normalizer dropped); `chi_sq` =
Gaussian × the `_SD` data column (already literally implemented as this); `sod` =
Laplace with `b≡1`; `norm_sos`/`ave_norm_sos` = Gaussian with a data-derived σ;
`kl` = multinomial cross-entropy. The *only* genuinely non-statistical objective
is `direct_pass`. So the "likelihood vs non-likelihood" split is false; the honest
cut is **per-point noise model vs not**, and the centering convention applies to
*every* noise model. We settled this shape:

- **Median is the universal default, defined family-agnostically: *the prediction
  is the 0.5-quantile (median) of the noise distribution.*** This generalizes
  ADR-0011's location axis instead of bolting on per-family special cases. For the
  location-scale families (Gaussian/Laplace) the 0.5-quantile *is* the location
  parameter, so the offset is `0` and the default is **byte-identical today**. For
  `neg_bin` it is the standard discrete median (`ppf(0.5)`), realized by solving
  for the mean μ that places the prediction at the continuous NB 0.5-quantile (the
  regularized-incomplete-beta CDF — a per-point root-find, the #419 capability).
  This **supersedes ADR-0024's** stance that `neg_bin` "rejects median as
  unimplemented": *every* means every (no carve-out); the uniform convention is
  worth a bounded numeric inversion that, in practice, is rarely exercised because
  users set `location = mean` for `neg_bin`. **`mean` is the explicit, native-only
  opt-in** (PEtab cannot express it; the importer/exporter always emit median).

- **Reconcile code to doc: flip `Gaussian.__init__`'s default `MEAN → MEDIAN`.**
  Provably byte-identical (linear `mean_offset == 0`; `lognormal` passes `MEDIAN`
  explicitly), so no existing fit changes — but "median is the default" becomes
  *true in code*, matching `location.py`, `Laplace`, ADR-0024, and PEtab. This is
  the one concrete commit that can land independently of the rest.

- **Backward compatibility rides a select-and-freeze `edition` marker, not a
  centering-specific switch.** A conf declares `edition = <integer>`; **absence ⇒
  legacy (implicit edition 1)**; using any new-era syntax *without* an `edition`
  line is an **immediate error** (naming the key and the fix). `edition` is
  *select-and-freeze* in the Rust sense: `edition = 2` means "interpret under
  edition-2 conventions, **forever**" — a future PyBNF that changes another default
  still reads an `edition = 2` conf with edition-2 semantics; only `edition = 3`
  gets the newer defaults. This is why the keyword is **`edition`, not
  `min_version`**: a `min_version` floor (`≥ X`) would let the *next* default
  change silently re-interpret every conf that declared an older floor — exactly
  the drift this mechanism exists to prevent. The value is a plain integer
  (decoupled from PyBNF release numbers and years, because editions change only
  when a convention changes, not every release); the tool *derives* the
  minimum-supporting PyBNF version and reports it.

- **The one divergent default (`neg_bin`) warns when it resolves to median
  implicitly.** Under a modern edition, a `neg_bin` whose `location` is
  *unspecified* resolves to median (its legacy default was mean). Because nobody
  genuinely wants median `neg_bin`, this almost always means a forgotten `location
  = mean` or an assumption that legacy semantics carried over — so it emits a
  **targeted warning** ("`neg_bin` is defaulting to median centering under
  `edition N`; legacy was mean — set `location` explicitly to silence"),
  regardless of spelling. An *explicit* `location = mean | median` is silent. This
  is the only place a number changes between eras, so it is the only place the
  warning fires. A legacy conf (`objfunc = neg_bin`, no `edition`) stays
  frozen-mean.

- **The modern objective surface is three keys on two orthogonal axes.** The
  *nature* axis (a doc label) is **statistical objective** (likelihood-anchored) vs
  **heuristic objective** (no probability model); the *shape* axis (the config key)
  is per-point / column-joint / named:
  - **`noise_model`** (recommended) — per-point families `gaussian` / `laplace` /
    `lognormal` / `neg_bin` × σ-source × `location` (median default). A line with
    **no observable name is the whole-fit default**; per-observable lines override.
  - **`profile_objective`** (new) — column-joint, shape-comparison objectives,
    seeded with **`kl` + `wasserstein`** (two members clear ADR-0011's
    "abstract on the 2nd member" bar; `wasserstein` is geometric, `kl` is the
    multinomial likelihood, so they span the family). It is `*_objective`, **not
    `*_model`**, precisely because the family is *mixed* statistical/geometric — the
    name must not promise a noise model. Its value grammar (e.g. `wasserstein`'s
    support/spacing) is **deferred to implementation**.
  - **`objective`** (allowed; "comfort food") — the named catch-all: legacy tokens
    that **desugar** to a `noise_model`, plus the bare `score` passthrough.

- **The least-squares family folds into the noise-model engine; legacy tokens
  survive as desugaring synonyms.** `sos` ≡ `gaussian, σ = fix_at 1` (which also
  restores the statistically-proper `½` that legacy `sos` amputates —
  argmin-identical); `chi_sq` ≡ `gaussian, σ = read_exp_file _SD`; `sod` ≡
  `laplace, σ = fix_at 1`. **Two new σ-source verbs are added** so the normalized
  variants fold in as honest heteroscedastic models rather than orphans:
  `relative` (constant-CV, σ ∝ the measurement) for `norm_sos`, and `column_mean`
  (σ = the observable's column mean) for `ave_norm_sos`. Docs lead with the
  general `gaussian` / `laplace` forms and present the legacy tokens as "≡ …"
  synonyms, recommending the general form without forbidding the familiar names.
  In the legacy edition every legacy token is byte-identical to today.

- **`direct_pass` and the analytical-objective surface are deferred to #425.** It
  is the sole current heuristic objective (reads a `score` cell, ignores the data);
  its name is never user-facing and it is kept as-is for developer/test use to
  avoid yak-shaving the rename and golden churn. Designing a first-class surface
  for analytical / user-defined objectives — which broadens PyBNF well beyond
  biological-model calibration — is its own targeted discussion in #425.

## Considered Options

- **Freeze the current per-family defaults and only document them** (issue #424
  option B). Rejected: it permanently enshrines the `Gaussian = MEAN` /
  `Laplace = MEDIAN` contradiction and the doc/code disagreement this ADR exists
  to remove.
- **Default to `mean`** (the principled CME mean-field interpretation). Rejected:
  it would silently shift every log/count model away from PEtab and PyBNF's own
  history (the units-trap failure mode ADR-0022 fought). Median-default + explicit
  `mean` opt-in gets the capability with zero silent change.
- **Carve `neg_bin` out** (keep its default mean, leave median "unimplemented").
  Rejected by the "every means every" decision: a uniform, teachable convention
  ("median unless you write `location = mean`") beats a per-family asterisk, and
  the cost is a bounded, rarely-hit CDF inversion (#419), not a behavior change to
  any existing fit (legacy `neg_bin` stays frozen-mean).
- **`min_version` as a `≥` floor.** Rejected: a floor pins tooling, not semantics,
  so the next default change leaks into every conf that declared an older floor.
  Select-and-freeze `edition` (each named edition's conventions frozen, all
  supported simultaneously) is the only mechanism that actually prevents drift.
- **A centering-specific compatibility switch** (issue #424 option C, scoped to
  noise). Rejected: it would guard a byte-identical no-op (the centering flip
  changes nothing today except the unimplemented `neg_bin` median). The era marker
  is the conf-wide modernization vehicle; centering rides it for free.
- **`profile_objective` as `profile_model`** (symmetry with `noise_model`).
  Rejected: the family is mixed — `kl` is a likelihood but `wasserstein` is a
  geometric distance — so `*_model` would overclaim. The `*_objective` suffix is
  the honest one: *model ⇒ always a likelihood; objective ⇒ maybe not.*
- **A new family token per centering** (`lognormal_mean`). Already rejected by
  ADR-0024 (it encodes the location axis as a code and needs a `_mean` twin per
  asymmetric family); the `location` field is the honest representation.

Relevant ADRs: **0011** (the location axis and per-point NLL kernel this
generalizes — median as "the prediction is the 0.5-quantile" makes the axis
uniform across families), **0021** (the family × σ-source engine and native
`noise_model` surface this folds the least-squares objfuncs into; the new σ-source
verbs extend its verb set), **0022** (the log-base convention the moment
correction uses), **0024** (the native location surface and "median default"
intent this *completes* in code — and whose `neg_bin`-rejects-median stance this
supersedes), **0004** (PEtab-defaulted not PEtab-bound — why native users get the
`mean` opt-in PEtab cannot express). Related issues: **#424** (this policy),
**#419** (the per-family mean/median capability, incl. the `neg_bin` 0.5-quantile
inversion), **#425** (analytical / user-defined objectives, the `direct_pass`
successor), **#418** (the `_Cum` generalization in the same objfunc space).
