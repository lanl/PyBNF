# PEtab observableParameters/noiseParameters placeholder import completes: a fixed noise-parameter id inlines as a constant sigma, a multi-token noiseParameters binds per index (row-varying → PerMeasurementFormulaSigma), and a noiseFormula that scales with the simulated prediction becomes a new prediction-dependent sigma source (issue #495)

**Status: Accepted and implemented (2026-07-19).** The three related gaps issue #495 named
in the PEtab benchmark collection — `Oliveira_NatCommun2021`, `Fiedler_BMCSystBiol2016`,
`Raia_CancerResearch2011` — now import and fit. Each is a case the constant-placeholder path
(ADR-0037/0044) and the row-varying binding table (ADR-0045) *almost* covered but stopped short
of: a `noiseParameters` id that is **fixed** rather than estimated; a **multi-token**
`noiseParameters` cell (a multi-parameter noiseFormula, which PyBNF only ever split on the
`observableParameters` side); and a `noiseFormula` whose σ scales with the **simulated
prediction**, not merely the free parameters. The first two reuse the existing engines by
generalizing the token split and the constant-vs-fixed check; the third is a genuinely new
`SigmaSource` and the one architectural change here.

ADR-0021 built the per-observable `(NoiseModel, σ-source)` engine; ADR-0037 reduced a
constant-per-observable `noiseParameters` id to a per-observable fit sigma; ADR-0044 added
`FormulaSigma` (a σ that is an expression over free parameters); ADR-0045 added
`PerMeasurementFormulaSigma` (a row-varying placeholder σ bound per data point from a sidecar).
This ADR completes that frontier for the noise side.

## The three gaps and where each lands

A PEtab `noiseParameters` cell binds the `noiseParameter${n}_${observableId}` placeholders of
the `noiseFormula` — exactly as `observableParameters` binds the `observableParameter${n}` ones,
the n-th semicolon token to the n-th placeholder. PyBNF's import had three blind spots in that
mapping:

1. **A fixed noise-parameter id (Oliveira).** `noiseFormula = noiseParameter1_X`,
   `noiseParameters = sd_X`, but `sd_X` has `estimate=0` (a fixed σ, value 1). The importer
   assumed every `noiseParameters` id was an estimated per-observable sigma and emitted
   `sigma = fit sd_X`, which the config then rejected (`sd_X` is not a declared free parameter).
   **Fix:** an id that resolves to a **fixed** PEtab parameter inlines as its numeric value —
   `('constant', value)` → `ConstantSigma` — not a `fit` free sigma. A one-line check against the
   `fixed_params` map the importer already builds, applied on both the bare-placeholder fast path
   and the substitute-and-classify path.

2. **A multi-token noiseParameters (Fiedler).** `noiseFormula = noiseParameter1_X *
   noiseParameter2_X`, `noiseParameters = s_gel;sigma` — two tokens, the first row-varying (a
   per-gel scale), the second constant. PyBNF **never split** a multi-token `noiseParameters`
   cell: `_noise_parameters` did `float(whole_cell)` and, on failure, kept the entire
   `"s_gel;sigma"` string as one id. **Fix:** the reader now splits `noiseParameters` into a
   `noise_param_tokens` tuple (the noise-side mirror of `observable_parameters`), the two
   single-token shapes keeping their dedicated scalar fields byte-identically. A constant
   multi-token tuple substitutes into the noiseFormula by index (Raia); a row-varying one keeps
   its placeholders and binds **every** `noiseParameter${n}` per data point from the sidecar —
   the ADR-0045 `PerMeasurementFormulaSigma`, its per-point binding loop generalized from one
   placeholder to n.

3. **A prediction-dependent noiseFormula (Raia).** `noiseFormula = noiseParameter1_X +
   noiseParameter2_X * (species…)` — after substitution `sd_abs + sd_rel * y`, the classic
   combined additive+proportional error model where the proportional term is the observable's
   **simulated value** `y`. This is neither a data column nor a PSet-only expression: it must be
   evaluated against the *current simulation*. It is the one genuinely-new capability — a new
   `SigmaSource` and a widened per-point noise seam.

## `PredictionFormulaSigma`: the one new object, and the seam it widens

PyBNF's `SigmaSource.value` seam read only `(owner, exp_data, exp_row, col_name)` — the objective,
the experimental data, and the point. No source saw the *simulation*, because none needed to: a
sigma was a data column, a constant, a free parameter, an expression over free parameters, or a
per-row token. A prediction-dependent σ breaks that: `σ_i = σ_abs + σ_rel · y_sim(t_i)` needs the
model output at the scored point.

`PredictionFormulaSigma` (`noise/source.py`) closes the gap. It holds a PEtab-math expression
whose free symbols resolve **either** from `owner._pset_values` (a free-parameter coefficient —
`σ_abs` / `σ_rel`) **or** from the simulated `Data` column of that name (a model entity — the
trajectory the σ scales with), read at `(sim_data, sim_row)`. It is the noise-side peer of the
measurement-model observation layer (ADR-0036): a lazily-`lambdify`-compiled callable, not pickled
(rebuilt worker-side), *estimated* (it references the estimated coefficients, so the family keeps
its likelihood normalizer, ADR-0011).

**The seam change is uniform and mechanical.** `SigmaSource.value` gains `(sim_data, sim_row)`
after `owner`; every existing source ignores them (a data column, constant, free parameter,
relative, column-mean, formula, per-measurement token — none read the simulation). `_noise_values`
and its twelve per-point call sites (`eval_point`, the Kalman/pointwise loops, the gradient/Fisher
seams) already had `sim_data`/`sim_row` in scope, so they thread through unchanged. The default
path is byte-identical — only `PredictionFormulaSigma` reads the new arguments.

**Which symbols are coefficients vs simulated columns is model-namespace dependent**, so it is
settled at config load, not at parse (`config._load_prediction_noise`, a sibling of
`_load_measurement_models`): each `prediction_formula` expression is validated against the model
namespace ∪ the declared free parameters, its free-parameter symbols recorded on the source
(`set_param_names`) so they surface through `required_free_noise_params` to the orphan check as
legitimate nuisances, and the rest are the simulated columns. A `prediction_formula` that
references **no** model entity is rejected (it should be a plain `formula`).

**Surface — a new native source verb.** `noise_model <obs> = <family>, <param> =
prediction_formula <expr>` (`parse.py` grammar + `objective._build_sigma_source`), a distinct verb
from ADR-0044's `formula` because the routing (formula ⇒ PSet-only; prediction_formula ⇒ reads the
simulation) is a namespace-dependent classification the importer makes with the model in hand, not
a content sniff the config builder could make alone. Both use the same rest-of-field grammar. As
always the verb is a first-class native capability, PEtab merely one way to reach it (ADR-0004): a
new-era job may author `sigma = prediction_formula <expr>` directly.

**Gradient boundary (the one deferral).** A prediction-dependent σ makes the per-point loss depend
on the prediction through the scale as well as the residual, which the #385 residual/Fisher
assembly does not model. It falls under the existing gradient gate (`estimated and not
FreeParameterSigma → GradientNotSupported`), so a gradient/EFIM fit raises cleanly and a
gradient-free optimizer/sampler is unaffected — the score path is complete. The full chain-rule
column is a later sub-layer, exactly as `FormulaSigma` / `PerMeasurementFormulaSigma`'s are.

## Scope

**In:** a fixed `noiseParameters` id → constant sigma; a multi-token `noiseParameters` cell split
and substituted by index (constant → the classified source below; row-varying → a multi-placeholder
`PerMeasurementFormulaSigma`); a substituted arithmetic noiseFormula classified as `('formula',
expr)` when it references only free parameters and `('prediction_formula', expr)` when it also
references a model entity; the `PredictionFormulaSigma` source + the `(sim_data, sim_row)` seam +
the `prediction_formula` verb + `config._load_prediction_noise`. New-era only (PEtab interop is
new-era, ADR-0034). Oracled against three crafted BNGL fixtures (`fixedsigma_v2`, `multisigma_v2`,
`predsigma_v2`), each import-only, simulator-free, scored against a hand-derived NLL — the
prediction fixture's σ differs by row *because the prediction does*, so a σ-at-the-measurement or
σ-broadcast bug is caught.

**Out (boundary raised in code / deferred, each pointing here):**

- The **gradient/EFIM** column for a `PredictionFormulaSigma` (`GradientNotSupported`, the existing
  composite-estimated-source gate) — a later #385 sub-layer.
- **Export** of the new shapes back to a PEtab `noiseParameters` / `noiseFormula` — the export
  round trip for a multi-token / prediction-dependent noise (the `measurement_rows_from_data`
  sidecar path emits one noise placeholder per column; a prediction σ has no exporter arm yet). The
  importer + fit + crafted fixtures are the gate (#495 "done when"), mirroring ADR-0044's initial
  export deferral; the round trip is a follow-up.
- A row-varying placeholder mixed **with** a simulated-trajectory column (a σ that is *both* bound
  per data point *and* a function of the sim output) — neither a pure binding-table lookup nor a
  pure prediction expression; unchanged ADR-0045 boundary.
- A `log-normal` / `log-laplace` distribution, a per-point laplace placeholder — unchanged
  ADR-0023/0037 boundaries.

## Boundaries (in code, each pointing here)

- `pybnf/noise/source.py` — `PredictionFormulaSigma` (lazy-compiled-not-pickled, estimated,
  PSet-coefficient-or-sim-column symbol resolution); the `value(owner, sim_data, sim_row,
  exp_data, exp_row, col_name)` signature on `SigmaSource` and every source.
- `pybnf/objective.py` — `_noise_values` threads `(sim_data, sim_row)`; `_build_sigma_source`'s
  `prediction_formula` verb → `PredictionFormulaSigma`; the gradient gate already covers it.
- `pybnf/parse.py` — the `noise_model … = prediction_formula <expr>` grammar (`nm_prediction_formula_field`).
- `pybnf/config.py` — `_load_prediction_noise` classifies each source's symbols against the model
  namespace, sets its free-parameter subset, and validates it (a no-op when none declared).
- `pybnf/petab/measurements.py` — `noise_param_tokens` on `PetabMeasurementRow`; `_noise_parameters`
  splits multi-token cells; `noise_parameters_by_observable` / `row_varying_noise_param_ids` (the
  multi-token siblings of the observable-side classifiers); `measurement_param_bindings` binds every
  `noiseParameter${n}`.
- `pybnf/petab/import_.py` — `_placeholder_subs` substitutes multi-token noise placeholders + inlines
  a fixed token; `_resolve_noise` gains the fixed-id → constant path, the tightened bare-placeholder
  test, and the free-parameter-expression vs prediction-expression classification; the directive
  builders emit `prediction_formula`.

## Consequences

- The per-measurement placeholder frontier (ADR-0033/0035/0036/0037/0044/0045) is **fully cleared
  on the noise side**: fixed, estimated, multi-parameter (affine / product), row-varying, and
  prediction-dependent noise all import and fit; the only remaining deferral is the *export* round
  trip for the two new shapes and the gradient column for a prediction σ.
- `PredictionFormulaSigma` is a **permanent native capability**, not a PEtab-import artifact: any
  new-era job can declare a combined additive+proportional error model with `noise_model <obs> =
  gaussian, sigma = prediction_formula <σ_abs> + <σ_rel> * <observable>` — the honest
  heteroscedastic error model systems-biology fits routinely need, previously reachable only through
  the fixed `relative`/`column_mean` sources (ADR-0031).
- The `SigmaSource.value` seam now carries the simulation, so a future noise model that needs the
  prediction (a Poisson/count σ = √μ, a variance-stabilizing scale) has a home without another seam
  change.
- See ADR-0011 (the `NoiseModel` kernel), 0021 (the σ-source engine), 0031 (`relative` /
  `column_mean`), 0034 (bind-by-id nuisance), 0036 (the measurement-model observation layer — this
  is its noise-side peer), 0037/0044/0045 (the placeholder reductions this completes). Closes #495.
