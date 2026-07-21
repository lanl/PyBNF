# PEtab export round-trip completes for the two new noise shapes: a multi-token noiseParameters cell re-emits every noiseParameter${n} token per row, and a prediction-dependent sigma exports verbatim as a noiseFormula that re-imports as prediction_formula because it names a model entity (issue #502)

**Status: Accepted and implemented (2026-07-19).** ADR-0075 (issue #495) completed the
*import* of a fixed noise-parameter id, a multi-token `noiseParameters` cell, and a
prediction-dependent `noiseFormula`, and deliberately deferred the *export* round trip for the
last two — the `measurement_rows_from_data` sidecar path emitted only **one** `noiseParameter`
series per column, and a `prediction_formula` sigma had **no** exporter arm. This ADR closes
that round trip: `import → export → import` is now fit-preserving for all three ADR-0075
fixtures (`fixedsigma_v2` / `multisigma_v2` / `predsigma_v2`), mirroring how ADR-0044/0045 landed
`FormulaSigma` / row-varying export in a later milestone than their import.

## The two gaps and where each closes

The importer recovers the new shapes; the exporter could not re-emit two of them. Each gap is a
one-spot completion of an engine that already handled the single-token / free-parameter case.

1. **A multi-token `noiseParameters` cell (Fiedler).** `noiseFormula = noiseParameter1_X *
   noiseParameter2_X`, `noiseParameters = s_lo;sig` (row-varying: the per-gel scale differs by
   row). On export the per-row tokens ride the `measurement_params:` sidecar into
   `measurement_rows_from_data`, whose `_column_placeholder_series` split one sidecar column's
   `{placeholder: {time: token}}` into the observable placeholders (**index-keyed, collected**)
   but the noise placeholder as a **scalar** — each `noiseParameter${n}` overwrote the last, so
   only one token survived. **Fix:** the noise side is now collected by 1-based index exactly like
   the observable side (`[noise[i] for i in sorted(noise)]`), and `measurement_rows_from_data`
   builds a `noise_param_tokens` tuple that the writer semicolon-joins into the `noiseParameters`
   cell — the noise-side peer of the existing `observableParameters` join. The single-token cases
   keep their dedicated `noise_parameter_id` field (a length-1 series → the byte-identical scalar
   write), so the pre-#502 output is unchanged.

2. **A prediction-dependent `sigma = prediction_formula <expr>` (Raia).** `sigma =
   prediction_formula sd_abs + sd_rel*y`, where `y` is a model entity. **Fix:** it exports
   **verbatim** as a plain `noiseFormula` — `sd_abs + sd_rel*y` — the *direct mirror of a
   `FormulaSigma`*, carrying **no separate prediction arm**. This works because the classification
   is the importer's, not the exporter's: `_resolve_noise` reclassifies the same substituted
   expression as `prediction_formula` iff it references a model entity (`y ∈ namespace`), else
   `formula` (ADR-0075). So the exporter emits the expression, admits its free-parameter
   coefficients (`sd_abs` / `sd_rel`) as observation-layer nuisances (the model-entity symbol `y`
   is *not* a declared free parameter, so it is harmlessly ignored), and the round trip lands back
   on `PredictionFormulaSigma`. No `noiseParameters` placeholder abstraction is required — the
   coefficients are inlined into the noiseFormula, exactly as a `formula` sigma inlines its own.

The **fixed-id** shape (`fixedsigma_v2`, Oliveira) already round-tripped: the importer inlines the
fixed id to `fix_at 2`, and the pre-existing `fix_at → ('constant', value)` arm emits an inline
constant `noiseFormula = 2`. It is kept in the oracle as a regression guard that the completion did
not disturb the constant path.

## Why verbatim, not placeholder re-abstraction

ADR-0075's "Out" list framed the prediction-σ export as mapping back to a PEtab `noiseFormula` +
`noiseParameters` (the placeholder form). The oracle it named, though, is **fit-preservation**, and
the verbatim form is both simpler and strictly the mirror of the already-shipped `formula` arm: a
`PredictionFormulaSigma` and a `FormulaSigma` differ only in whether a symbol resolves from the
simulation or the PSet — a namespace-dependent split the *importer* makes with the model in hand,
never a content the exporter must reconstruct. Re-abstracting `sd_abs + sd_rel*y` into
`noiseParameter1_X + noiseParameter2_X * y` with a per-row `noiseParameters = sd_abs;sd_rel` cell
would be a fragile inverse of a substitution the round trip does not need. A source carries **no**
PEtab-token metadata anyway (ADR-0075: the coefficients replaced the placeholders at import), so
the verbatim expression is the only faithful trace to re-emit. The multi-token case (gap 1) *does*
need the placeholder + per-row `noiseParameters` form, because its tokens are **row-varying** and
cannot be substituted into one expression — which is exactly why that gap is the sidecar
completion, not an expression re-emit.

## Scope

**In:** `_column_placeholder_series` collects noise placeholders by index (a list, not a scalar);
`measurement_rows_from_data` fills `noise_param_tokens`; `_noise_cell` joins a multi-token cell with
`;`; a `prediction_formula` sigma verb maps to the verbatim `('formula', expr)` export arm;
`_referenced_nuisance_symbols` gathers a `prediction_formula` expression's free-parameter
coefficients. The `multisigma_v2` / `predsigma_v2` fixtures gained a `noisePlaceholders` column so
they pass petab's own `CheckOverridesMatchPlaceholders` validator (PyBNF's importer detects
placeholders by pattern, so the column does not change the import path — the multi-token / affine
formulas classify by substitution regardless). New-era only (PEtab interop is new-era, ADR-0034).

**Out (unchanged boundaries, each pointing here or at ADR-0075):**

- The **gradient/EFIM** column for a `PredictionFormulaSigma` (issue #502 item 2) — the *scalar
  gradient* half has since landed (ADR-0079, the σ-formula chain rule on the L-BFGS path), the
  *EFIM Fisher* half remaining deferred. Export is orthogonal to the gradient path either way.
- A **`relative`** sigma export (a noiseFormula expression over the measurement) — the unchanged
  ADR-0023/0031 boundary.
- A **`lognormal`** (log10) family, a `mean`-centered location, `neg_bin` — unchanged
  `_reduce_noise_spec` boundaries. Natural-log Gaussian was subsequently given the exact
  `lnnormal` -> PEtab `log-normal` mapping by ADR-0084 / issue #509.

## Boundaries (in code, each pointing here)

- `pybnf/petab/measurements.py` — `_column_placeholder_series` returns `([noise_series, ...],
  [obs_series, ...])` (both index-ordered); `measurement_rows_from_data` fills `noise_param_tokens`
  for a multi-token cell (single-token keeps `noise_parameter_id`, byte-identical);
  `_noise_cell` joins the tuple with `;`.
- `pybnf/petab/export.py` — `_noise_source_for_column`'s `prediction_formula` arm →
  `('formula', expr)` (verbatim, the `FormulaSigma` mirror); `_referenced_nuisance_symbols` gathers
  a `prediction_formula` expression's coefficients (over-matches the model entity, harmlessly).
- `pybnf/petab/observables.py` — the `('formula', expr)` kind emits the noiseFormula verbatim with
  no placeholder (the arm both `formula` and `prediction_formula` reach).
- `tests/petab_fixtures/{multisigma,predsigma}_v2/observables.tsv` — the added `noisePlaceholders`
  column (petab-validity, import-inert).
- `tests/test_petab_export.py::TestExportNewNoiseShapesRoundTrip` — the fit-preserving round-trip
  oracle: each fixture re-imports and scores the same fixed trajectory against its hand-derived
  NLL (σ differs by row for `multisigma` / `predsigma`, so a dropped token or a σ-broadcast bug
  scores differently), plus petab-validator checks on the source and re-exported problems.

## Consequences

- The PEtab noise round trip is **closed on the export side** for every shape the import side
  recovers except a `relative` sigma: fixed, per-observable estimated, expression (`FormulaSigma`),
  row-varying single- and **multi-token** (`PerMeasurementFormulaSigma`), and
  **prediction-dependent** (`PredictionFormulaSigma`) noise all export and re-import to the same
  fit. `import → export → import` is a supported workflow for the new noise shapes.
- The multi-token noise export reuses the observable-placeholder machinery verbatim (both sides are
  now index-keyed collections joined per row), so a future n-token noiseFormula needs no further
  export change.
- The remaining #502 follow-up is the gradient/EFIM column for a prediction-dependent σ (item 2)
  and the state-dependent-noise tutorial (item 3); neither touches this export path.
- See ADR-0075 (the import completions this round-trips), 0044/0045 (`FormulaSigma` /
  row-varying export, the milestone-later export deferral this mirrors), 0021 (the σ-source engine),
  0034 (bind-by-id nuisance). Closes issue #502 item 1.
