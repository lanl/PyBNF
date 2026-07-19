# Two PEtab observables on one model column are each materialized to their own observableId column via an identity measurement model, so their per-observable noise keys by observableId, not the shared column (issue #503)

**Status: Accepted and implemented (2026-07-19).** `Bertozzi_PNAS2020` — the same problem
whose condition side ADR-0076 fixed (#496) — now imports *and* loads. Its observables table
measures one model output (`I_`) in two experiments, each experiment its own estimated sigma:

```
observableId   observableFormula   noiseFormula
y_I_NY         I_                  sd_I_NY        # New York
y_I_CA         I_                  sd_I_CA        # California
```

Two distinct observables, one model entity. `sd_I_NY` / `sd_I_CA` are estimated parameters —
each experiment fits its own noise level, the common PEtab idiom for "the same readout measured
on different instruments/cohorts." This is orthogonal to #496: the condition side is done; this
is the noise side.

## Why it collided

A **bare-name** `observableFormula` (the common case, ADR-0025/0036) maps its `observableId`
straight to that model column — PyBNF matches the `.exp` column to the model
observable/function/species by name and the backend already produces it, so no measurement-model
translator runs and the path stays dependency-free. The importer's per-observable noise
(ADR-0021/0037, the Boehm shape) then emits one `noise_model <column> = …` override per
observable, **keyed by that column**:

```
noise_model I_ = gaussian, sigma = fit sd_I_NY
noise_model I_ = gaussian, sigma = fit sd_I_CA
```

Two overrides on the one column `I_`. `parse.ploop` rejects a repeated `noise_model <col>`
("`noise_model for observable 'I_' is specified multiple times`"), so the imported conf did not
even parse. The data pivot did *not* collide — it groups by `(experimentId, modelId)`, so the two
observables already land in separate `.exp` groups — the collision was purely in the noise
directive: two observableIds, one column key.

## The fix: materialize a shared model entity to per-observableId columns

An **expression** `observableFormula` already gets its own column named after the `observableId`
(a *measurement model* `(id, formula)` evaluated post-simulation, ADR-0036) — that is exactly
why two expression observables never collide. This ADR extends the same materialization to the
**degenerate identity case**: when >1 bare-name observable names the *same* model entity, each is
routed to its own `observableId` column via an **identity measurement model** `(observableId,
entity)` — `observable: y_I_NY, formula: I_` and `observable: y_I_CA, formula: I_` — instead of
the shared entity column. The data columns *and* the noise overrides then key by the distinct
`observableId`:

```
noise_model y_I_NY = gaussian, sigma = fit sd_I_NY
noise_model y_I_CA = gaussian, sigma = fit sd_I_CA
observable: y_I_NY, formula: I_
observable: y_I_CA, formula: I_
```

No collision; each dataset keeps its own sigma. This reuses the existing measurement-model
(ADR-0036) and per-observable-noise (ADR-0021) machinery unchanged — the noise engine is not
touched. `y_I_NY` / `y_I_CA` materialize `I_` post-simulation exactly as `scaling*x` would, only
the "formula" is the bare entity itself.

## Collision detection is bare-name-only and byte-conservative

`import_.py::_shared_bare_entities` counts, before the mapping loop, how many observables reduce
to a **bare model entity** (no observableParameters placeholder, no row-varying scale, the
formula an identifier in the model namespace — the assignment-rule inlining mirrors the main pass
so a rule-variable formula, #493, an expression and never bare, is correctly excluded). A model
entity named by **>1** such observable is "shared."

`_observable_id_to_column` then takes the bare-name early-return **only** for a uniquely-targeted
entity (byte-identical to before); a shared entity **falls through** to the measurement-model
branch, which validates the bare name against the namespace (trivially — it is a known entity),
runs the existing shadow check (the materialized `observableId` column must not itself name a
model entity — it never does here), and appends `(observableId, entity)`. Expression observables,
which already own their `observableId` column, are never candidates and are unaffected.

The **single-observable-per-column case — the overwhelming common case — stays byte-identical.**
Only a detected same-entity collision among ≥2 bare-name observables triggers materialization.

## Scope

**In:** the shared-bare-entity detection + per-observableId identity-measurement-model
materialization in the importer (BNGL and SBML alike — the driver `Bertozzi_PNAS2020` is SBML).
New-era only (PEtab interop is new-era, ADR-0034). The exporter needs **no** change: it already
emits a measurement model `(id, formula)` as an `observableFormula = formula` row, so the two
materialized observables round-trip as two `observables.tsv` rows on the one entity, and a
re-import re-detects the shared entity and re-materializes — closing the loop.

Oracled by: a simulator-free import-and-parse test (two observables on `z`, each an estimated
sigma, the imported conf parses with two distinct `noise_model <obsId>` lines, never a colliding
`noise_model z`); an `export → import → re-export` **byte round trip** of the materialized form;
and the real `Bertozzi_PNAS2020` problem importing *and* loading via `Configuration` (the way
#496's Bruno was verified). Full local gate (with bngsim) green.

**Out (deliberately not done):**

- **Per-experiment noise on one shared column** — the heavier alternative: keep both observables
  on the single `I_` column and let the noise engine vary sigma *per experiment* rather than per
  observableId. That touches the per-observable noise engine (ADR-0021) — the override key would
  become `(column, experiment)` — not just the importer, and buys nothing here: the two datasets
  are already distinct observables, so a distinct column per observableId is the honest and
  minimal representation. Rejected as overkill.
- A **shadowing** materialized column — an `observableId` that itself names a model entity is
  refused by the existing measurement-model shadow check (ADR-0036), unchanged.

## Boundaries (in code, each pointing here)

- `pybnf/petab/import_.py` — `_shared_bare_entities` (the pre-scan: model entities named by >1
  bare-name observable, assignment-rule inlining mirrored, placeholder/row-varying excluded);
  `_observable_id_to_column` (the bare-name early-return is gated on *not* shared; a shared entity
  falls through to the identity-measurement-model branch that names the column by `observableId`).

## Consequences

- Two (or more) observables measuring one model output, each with its own noise, are a **native
  importable shape**, not a rejected one — the systems-biology "same readout, different
  experiment, own error model" the benchmark collection routinely encodes.
- The measurement-model layer (ADR-0036) is now the single mechanism for "an observableId owns a
  distinct data column," whether the formula is an expression (`scaling*x`), a placeholder-bearing
  per-measurement model (ADR-0045), or — now — a bare model entity shared by siblings. No new
  runtime concept; the identity case is just the formula being the entity itself.
- See ADR-0021 (the per-observable noise engine whose key was the collision point), ADR-0036 (the
  measurement-model observation layer this reuses), ADR-0025 (the bare-name common path preserved
  byte-for-byte), and ADR-0076 (the #496 sibling that fixed `Bertozzi_PNAS2020`'s condition side).
  Closes #503.
