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
- **Add a natural-log native token so `log` also imports through the conf path.** Deferred: a
  native-UX question (ADR-0023 already deferred it), not required by #499's log10 problems.

Relevant ADRs: **0023** (the observables noise-half mapping this scopes an exception to —
`noiseDistribution × noiseFormula`, and the earlier "no observableTransformation" rejection
this amends for the converter channel), **0022** (`LINEAR`/`LOG10`/`LN`; "PyBNF `lognormal` is
log10, PEtab `log-normal` is natural log"), **0032** (the importer read path that read only
`noiseDistribution`), **0011** (the per-point kernel and the median location). Sibling:
**#491** (`petab1to2_preserve_scale` re-injecting `parameterScale` as a `log-uniform` prior —
the parameter-axis twin). Related: the benchmark build siblings (#492–#496).
