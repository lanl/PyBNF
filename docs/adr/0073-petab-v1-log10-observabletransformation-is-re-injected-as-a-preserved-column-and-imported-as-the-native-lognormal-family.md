# A dropped v1 `observableTransformation = log10` is re-injected as a preserved column and imported as the native `lognormal` family (issue #499)

**Status: Accepted (implemented, 2026-07-18).** `pybnf.petab.petab1to2_preserve_scale` now
re-injects the v1 `observableTransformation` column (alongside the `parameterScale`
re-injection it already did, #491), and the importer + the `observables.py` asset adapter
read it to select the noise family's **additive scale**. A v1 `log10` observable converts and
imports as the native `lognormal` family (`Gaussian(LOG10, MEDIAN)`) — `objective = lognormal`
or a `noise_model = lognormal, …` line — instead of a linear `gaussian`. Verified end to end
(v1 → convert → import → the built objective's noise is `Gaussian(LOG10)`) plus the boundary
raises; a linear problem is byte-for-byte unchanged.

## Context — a real objective silently lost in the v1 → v2 hop

A PEtab **v1** observable with `observableTransformation = log10` fits its residual on the
log10 scale, **with** the change-of-variables Jacobian `Σ log(y·ln10)` — a *different
objective* from the linear residual, not merely a different search scale. Several
multi-decade-signal benchmark problems use it (e.g. Perelson_Science1996,
Borghans_BiophysChem1997, Elowitz_Nature2000): their signals span decades, so the linear
Gaussian cannot beat a huge `σ̂` and the fit collapses to the wrong optimum (Perelson: linear
global min `J ≈ 232.3` vs the reference `J* = 222.28`, which only the log10 residual **plus**
the data Jacobian reproduces).

But PEtab **v2** removed the `observableTransformation` column and folded transformation into
`noiseDistribution` as the **natural-log** `log-normal` / `log-laplace` prefixes — with **no
`log10` form** (ADR-0022/0023). `petab.v2.petab1to2` therefore drops a v1 `log10`
transformation entirely: it downgrades `log10-normal` to a *blank* `noiseDistribution`, and
the importer (reading only `noiseDistribution`, ADR-0032) resolves the observable to
`Gaussian(LINEAR)` — the wrong objective, scored silently. This is the observable-axis twin of
the `parameterScale` drop that ADR (`petab1to2_preserve_scale`, #491) fixes for the
*estimation* scale.

There is a genuine v2-spec gap: PyBNF's **native `lognormal` token is `log10`** (to match
log10 priors, ADR-0022), whereas PEtab v2's `log-normal` is natural log. So even emitting
`log-normal` would give the wrong base. **v1 `log10` has no faithful PEtab-v2 `noiseDistribution`
representation.** ADR-0023 knew v2 had removed the column and, on that basis, **rejected**
"keep the v1-style separate `observableTransformation` column … encoding a `log10` scale PEtab
v2 does not have." That rejection was right *for a native v2 problem*: a hand-authored v2
observables table has no such column and no log10. It did not cover the **v1 → v2 migration**
case, where refusing to carry the scale doesn't make the problem cleaner — it silently changes
the objective.

## Decision

**Re-inject `observableTransformation` in the scale-preserving converter as a preserved extra
column, and have the importer + the observables adapter select the noise family's additive
scale from it — the observable-axis twin of the `parameterScale → log-uniform` re-injection
(#491).** Because v2 has no faithful `log10` `noiseDistribution`, the column is carried
verbatim rather than folded: it is a **PyBNF-specific channel** (v2-lint-clean; other tools
ignore the unknown column), not a claim that `observableTransformation` is standard v2. The
scale is chosen from the transformation, not just the family from `noiseDistribution`:

| v1 `observableTransformation` + `noiseDistribution` | PyBNF `NoiseModel` | native token |
|---|---|---|
| `lin` + `normal` (or column absent) | `Gaussian(LINEAR, MEDIAN)` | `gaussian` / `chi_sq` / `sos` / … |
| **`log10` + `normal`** | **`Gaussian(LOG10, MEDIAN)`** | **`lognormal`** |
| `log` + `normal` | `Gaussian(LN, MEDIAN)` | *(none — see boundary)* |
| `lin` + `laplace` | `Laplace(LINEAR, MEDIAN)` | `laplace` / `sod` |
| `log10` / `log` + `laplace` | `Laplace(LOG10/LN, MEDIAN)` | *(none — see boundary)* |

- **The converter (`convert.py`).** After the standard `petab1to2`, read each v1 problem's
  `observableTransformation` column and write a `{observableId: log|log10}` map onto the v2
  observables table (`inject_observable_transformations`, mirroring
  `inject_log_uniform_priors`). Linear/absent observables are omitted (blank cell); the
  converted problem stays byte-identical on the observables when nothing is log-scaled.
- **The importer conf path (`import_.py`).** `_native_noise_family(row)` combines
  `noiseDistribution` (the Gaussian/Laplace family and its own `log-` scale) with
  `observableTransformation` (the overriding scale) into the **native conf family token**
  (`gaussian` / `lognormal` / `laplace`), threaded through the objective-directive recovery.
  A `log10` per-point `_SD` observable emits `objective = lognormal`; a fixed/free/formula
  sigma emits a `noise_model = lognormal, sigma = …` line.
- **The observables adapter (`observables.py`).** `noise_model_from_row` reads the same column
  to *override* the additive scale (`log10 → LOG10`, `log → LN`, `lin → unchanged`),
  constructing the family directly — so it faithfully represents even the natural-log and
  log-Laplace combinations the conf path cannot name.

The native `lognormal` kernel already carries the log10-space squared residual **and** the
Jacobian (ADR-0011/0022), so once the family is right the score is exact — matching the
paper's `J*` to `OG ≈ 5e-7` (verified in the issue). This **un-disavows** ADR-0023's
`log10 → LOG10` mapping, but scopes it to the converter re-injection channel: a native v2
problem (no column) is byte-for-byte unchanged, and `log-normal` / `log-laplace` still import
as the natural-log families.

## Boundaries (in code, never a silent mis-recovery)

- **No native token for the natural-log families or a log Laplace.** `Gaussian(LN)` (from
  `log-normal`, or `observableTransformation = log`) and any `Laplace(LOG10/LN)` have no
  native `.conf` token (ADR-0023: the native grammar has no natural-log family, and `lognormal`
  is log10 only), so the *conf-emitting importer* raises `NotImplementedError`. The
  `observables.py` adapter still builds them (it constructs the kernel directly); only the conf
  round-trip lacks a spelling.
  **Superseded for Gaussian by ADR-0084 / issue #509:** `Gaussian(LN)` now has the explicit
  `lnnormal` token and imports exactly. Log Laplace remains outside the native surface.
- **A transformation that contradicts a log `noiseDistribution`** (e.g. `log10` over
  `log-normal`'s LN) is an ambiguous double-spelling of the scale → `PybnfError`. An unknown
  transformation spelling → `PybnfError`. A distribution v2 removed (`neg_bin`) → the existing
  `NotImplementedError`.
- **Export is unchanged.** PyBNF's `lognormal` still has no PEtab-v2 export home (log10 vs
  natural log; ADR-0025's documented boundary), so a `log10`-imported job does not round-trip
  back out through a re-export. #499 is an import + conversion fix; symmetric export is a
  separate, deferred sigma-scale-conversion.

## Considered options

- **Rewrite `log` to v2-native `log-normal` and carry a column only for `log10`.** Rejected:
  two mechanisms for one axis. `log10` needs the column regardless (no v2 home), so carrying
  the transformation uniformly is simpler, faithful to what v1 said, and keeps the importer's
  scale-selection one rule.
- **Leave the importer alone; document that log10 problems must be hand-corrected.** Rejected:
  running the problem *as specified* is the whole point of a benchmark (the same argument that
  justified #491). Silently scoring the wrong objective is the bug, not a UX gap.
- **Add a natural-log native token so `log` also imports through the conf path.** Deferred here,
  then resolved as `lnnormal` by ADR-0084 / issue #509.

Relevant ADRs: **0023** (the observables noise-half mapping this scopes an exception to —
`noiseDistribution × noiseFormula`, and the earlier "no observableTransformation" rejection
this amends for the converter channel), **0022** (`LINEAR`/`LOG10`/`LN`; "PyBNF `lognormal` is
log10, PEtab `log-normal` is natural log"), **0032** (the importer read path that read only
`noiseDistribution`), **0011** (the per-point kernel and the median location). Sibling:
**#491** (`petab1to2_preserve_scale` re-injecting `parameterScale` as a `log-uniform` prior —
the parameter-axis twin). Related: the benchmark build siblings (#492–#496).

## Addendum (2026-09-10): the converter also resets `noiseDistribution` to the v1 base family (issue #679)

**Accepted and implemented 2026-09-10.** The re-injection above was silently leaning on a
petab bug. `petab1to2` is *designed* to fold the v1 `observableTransformation` into the v2
`noiseDistribution`; a missing `return` in that merge left the column blank in every
petab < 0.9.0, the importer defaulted the blank to `normal`, and `log10` over `normal`
imported as `lognormal` exactly as this ADR describes. petab 0.9.0 (2026-09-07,
PEtab-dev/libpetab-python#502) fixed the merge. Because PEtab v2 has no `log10-normal`, the
converter now substitutes the natural-log family — `log10` + `normal` → `log-normal`, with a
warning — which is a silent `ln 10` rescaling of sigma. Stacked under our re-injected `log10`
column, the importer's "give the residual scale in one place" rule refused every converted
log10 observable as a contradiction, and `test_converted_log10_problem_imports_as_lognormal`
went red on every Python ≥ 3.12 CI leg (petab 0.9.0 requires 3.12, so 3.11 stayed on 0.8.2
and green).

The fix keeps the ADR's design and closes the gap it left: `petab1to2_preserve_scale` now
reads each log observable's **v1** `noiseDistribution` (blank → `normal`) alongside its
transformation, and `inject_observable_transformations` takes an optional
`{observableId: 'normal' | 'laplace'}` map and resets the row's `noiseDistribution` to that
linear base while writing the transformation. The scale is then stated once, in the preserved
column, regardless of which petab produced the v2 table — a blank cell (0.8.2) and a folded
`log-normal` / `log-laplace` (0.9.0) both come back to the v1 family. Rows without a
transformation keep whatever petab1to2 wrote, so a linear problem is still byte-identical.
petab's own substitution warning is inside the `catch_warnings` block the converter already
uses to silence the `parameterScale` warning, for the same reason: re-adding the scale is
this function's job.

This is not an upstream bug. The substitution is documented and warned, and the underlying
gap — no log10-normal in PEtab v2 — is a specification decision (ADR-0022's "PyBNF
`lognormal` is log10, PEtab `log-normal` is natural log"). The preserved column is the
workaround the specification leaves us.

## Addendum (2026-09-25): declared v1 priors are rewritten from v1, and the warning filter is narrowed (issue #893)

**Accepted and implemented 2026-09-25.** The converter's module docstring said petab1to2
"already preserves scale where it is attached to an objective prior", so rows with a declared
v1 prior were left exactly as petab1to2 wrote them. That is false for two cases. petab1to2
renames a `parameterScale*` prior to the matching v2 family but keeps its numbers. For a
`log10` `parameterScaleNormal` it warns ("Prior distribution `log10-normal' ... Using
`log-normal` instead"), and v2 then reads the unchanged mean and sd as natural-log numbers,
so the imported prior on log10(theta) has both divided by ln 10. The blanket
`simplefilter('ignore')` described in the previous addendum hid that warning. For a
natural-log `parameterScaleUniform(a;b)` it writes `log-uniform(a;b)` with no warning, and v2
reads those numbers as bounds on theta rather than on ln(theta). `Schwen_PONE2014`,
`Isensee_JCB2018`, `Raimundez_PCB2020` and `Bachmann_MSB2011` hit the first case unmodified.
Across those four problems and `Lang_PLOSComputBiol2024`, 44 of the 72 declared objective
priors on estimated parameters converted to a different distribution, as libpetab's own v1
and v2 prior densities show.

**Decision.** The converter no longer keeps petab1to2's translation of any prior the v1 author
declared. `v2_prior_from_v1` maps each declared row from the v1 table to the v2 prior with the
same distribution over theta, and `write_v2_priors` writes it. A row declares a prior if its
`objectivePriorType` or its `objectivePriorParameters` cell is filled; a blank type under
filled parameters is v1's default type, `parameterScaleUniform`. The v2 table is rewritten
only when a cell changes, so a table petab1to2 already converted exactly keeps petab1to2's
bytes. Rows with no declared prior go through the #491/#548 log-uniform re-injection as
before.

Every v1 objective prior, as petab 0.9.0's `petab1to2` treats it (theta is the parameter;
"kept" means petab1to2's output was already exact and is unchanged):

| v1 scale | v1 prior | v1 meaning | petab1to2 writes | converter now |
|---|---|---|---|---|
| any | `uniform` / `normal` / `laplace` (a;b) | on theta | same name, same numbers | kept |
| any | `logNormal` / `logLaplace` (mu;s) | on ln theta | raises (pydantic `ValidationError`: not a v2 name) | still refused upstream; exact v2 is `log-normal` / `log-laplace` (mu;s) |
| lin | `parameterScale{Uniform,Normal,Laplace}` | on theta | `uniform` / `normal` / `laplace`, same numbers | kept |
| log | `parameterScaleNormal` / `Laplace` (mu;s) | on ln theta | `log-normal` / `log-laplace` (mu;s) | kept |
| log | `parameterScaleUniform` (a;b) | ln theta in [a, b] | `log-uniform` (a;b), no warning; a v2 lint error if a <= 0 | **corrected** to `log-uniform` (e^a;e^b); a <= 0 still refused by petab1to2's lint |
| log10 | `parameterScaleNormal` (mu;s) | log10 theta ~ N(mu, s) | `log-normal` (mu;s), warns | **corrected** to `log-normal` (mu ln10; s ln10) |
| log10 | `parameterScaleUniform` (a;b) | log10 theta in [a, b] | raises `NotImplementedError` (`log10-uniform`) | still refused upstream; exact v2 is `log-uniform` (10^a;10^b) |
| log10 | `parameterScaleLaplace` (mu;b) | log10 theta ~ Laplace | raises `NotImplementedError` (`log10-laplace`) | still refused upstream; exact v2 is `log-laplace` (mu ln10; b ln10) |
| lin | blank type, (a;b) | = `parameterScaleUniform` | `uniform` (a;b) | kept |
| log / log10 | blank type, (a;b) | ln / log10 theta in [a, b] | `uniform` (a;b), linear; the old converter then overwrote it with `log-uniform` over the bounds | **corrected** to `log-uniform` (e^a;e^b) / (10^a;10^b) |
| lin / log | `parameterScaleUniform`, blank parameters | v1 default: uniform over the bounds on the scale | `uniform` / `log-uniform` over the bounds | kept |
| log10 | `parameterScaleUniform`, blank parameters | same | raises `NotImplementedError` | still refused upstream (a blank prior cell says the same thing and converts) |

Every case is exactly representable in v2: v2 has linear and natural-log forms of uniform,
normal and laplace, a log10 location-scale prior is the natural-log one with both numbers
times ln 10, and a log uniform prior is `log-uniform` over the exponentiated bounds. So the
converter adds no refusal of its own. The four petab1to2 refusals are loud and are left in
place: converting them would mean running petab1to2 on a rewritten copy of the v1 problem.
The mapping already covers them (and its unit tests pin their values), so a petab release
that accepts them converts them exactly.

**Initialization priors.** petab1to2 drops `initializationPriorType` /
`initializationPriorParameters` for every row and warns once. PEtab v2 has no initialization
prior and PyBNF has no channel for one: the importer starts from `nominalValue` and draws
initial points from the bounds on the parameter's search scale. v1's default initialization
prior is that same box, so a blank cell, a blank `parameterScaleUniform` (the shape
`Armistead_CellDeathDis2024` uses), or one that states the bounds is dropped without comment.
Any other initialization prior is dropped with a warning that names the parameter, its prior
and its scale. It is a warning rather than a refusal because the objective, the prior and the
posterior do not depend on it.

**Warnings.** `simplefilter('ignore')` is replaced by four `filterwarnings('ignore', ...)`
patterns, one for each petab1to2 warning whose subject the converter repairs: the dropped
`parameterScale`, the observable `log10-normal` substitution (#679), the prior `log10-normal`
substitution (above), and the blanket initialization-prior warning (replaced by the named
one). Every other warning petab1to2 raises now reaches the caller. An unrecognized warning is
not made an error. With petab 0.9.0, no substitution survives uncorrected: every declared
prior is rewritten from v1 whatever petab1to2 wrote, and every log observable's
`noiseDistribution` is reset from v1. A future warning could be anything, including a
library deprecation, and making all of them fatal would stop working conversions.

**Left out.** A declared prior on theta itself (`uniform` / `normal` / `laplace`) on a `log` or
`log10` parameter keeps its exact distribution, but v2 has no column for the search scale, so
PyBNF searches that parameter linearly. The objective is unchanged. This is the
declared-prior counterpart of the bare-scale loss #548 fixed, and it is not addressed here;
the "Parameter scales are not supported" warning that would have covered it is still
silenced. None of the benchmark problems above has such a row.
