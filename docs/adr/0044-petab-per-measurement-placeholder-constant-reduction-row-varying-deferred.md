# PEtab per-measurement placeholders import by constant reduction: a placeholder constant across an observable's rows is substituted into the formula; a row-varying placeholder stays deferred (issue #428)

**Status: Accepted and implemented (decision + implementation 2026-06-21).** Phase 1 of #428,
the per-measurement placeholder frontier ADR-0036/0037 pinned ("constant-per-observable = fit
sigma; row-varying = deferred"). Three PEtab v2 capabilities that previously raised a clean
`NotImplementedError` now import when their per-measurement value is **constant across an
observable's measurement rows**: an `observableParameters` placeholder in the
`observableFormula` (a per-observable scale/offset), an `observableParameter*`/`noiseParameter*`
placeholder inside an **expression** `noiseFormula`, and an expression `noiseFormula` over free
parameters + constants. The genuinely **row-varying** case (a placeholder that differs
timepoint-to-timepoint, or per-condition) stays deferred behind the same cross-row guards
ADR-0037 introduced — its home is the deeper per-data-row binding seam, tracked as the #428
follow-up (Phase 2).

ADR-0021 built the per-observable `(NoiseModel, σ-source)` engine; ADR-0036 built the
measurement-model observation layer (a PEtab-math `observableFormula` compiled to a numpy
callable over the trajectory + the PSet); ADR-0037 reduced a **constant-per-observable**
`noiseParameters` *id* to a per-observable fit sigma. This ADR generalizes that one reduction
to the observable side and to expression noise formulae, reusing those engines unchanged.

## The one principle (the load-bearing distinction, generalized)

A PEtab placeholder (`observableParameter${n}_${observableId}` /
`noiseParameter${n}_${observableId}`) is substituted **per measurement row** from the
measurements table's `observableParameters` / `noiseParameters` column. The *kind of variation*
decides whether PyBNF can represent it, exactly as in ADR-0037:

- **Constant across all of an observable's rows** (the same scale/offset/sigma at every
  timepoint) — the placeholder is **not per-measurement at all**: it is a single scalar (a free
  parameter id, or a number) for that observable. **Substituting it into the formula reduces it
  to the existing per-observable machinery**: an id stays a free symbol that resolves from the
  PSet (the measurement layer, ADR-0036; a nuisance free parameter, ADR-0034); a number inlines
  as a constant (ADR-0037's `inline_constants`, generalized). This is Phase 1.
- **Row-varying** (a different value per timepoint, or per condition) — genuinely
  per-measurement. PyBNF's noise and observation models are per-**observable** (one σ-source,
  one materialized column per observable), so there is no analogue. It **raises
  `NotImplementedError`** — the boundary is in code, not a silent mis-import. This is Phase 2
  (the deeper seam: the measurement layer evaluates one column over the *simulation* trajectory,
  but per-row parameters are keyed to *data* rows, matched to sim rows only later in
  `objective.eval_point`; binding them needs a new per-data-row evaluation contract and a way to
  carry the per-measurement parameter table through the `(sim_data, pset)`-only layer seam).

The constant-vs-row-varying line is checked once, cross-row, in two sibling helpers in
`measurements.py` — `observable_parameters_by_observable` (new) alongside ADR-0037's
`noise_parameter_ids_by_observable`. The per-row reader only splits the cell into tokens.

## The reduction is substitution, reusing three existing engines

1. **Read** the measurements' `observableParameters` (new) and `noiseParameters` (ADR-0037)
   columns; require each observable's cell constant across its rows (else raise). The n-th
   semicolon-delimited token binds `observableParameter${n}_${id}` (resp.
   `noiseParameter${n}_${id}`).
2. **Substitute** the placeholders into the `observableFormula` and `noiseFormula` via
   `formula.substitute_placeholders` — the sibling of ADR-0037's `inline_constants`: it parses
   with `sympify_petab` (never a string tokenizer, ADR-0033), substitutes each placeholder
   symbol with its token (a number → `sp.Float`; an id → `sp.Symbol`, kept as a free symbol),
   re-serializes through the precedence-safe PEtab printer, and is guarded by the same numeric
   round-trip self-check. A bare-name / no-placeholder formula is returned verbatim (it never
   reaches `sympy`).
3. **Route** the substituted formula to the engine it now fits:
   - the substituted **`observableFormula`** is an ordinary measurement model — the ADR-0036
     observation layer, unchanged. A nuisance scale/offset id (one that is not a model
     parameter) resolves from the PSet at eval time (ADR-0036 §4); the validator's allowed set
     is widened from "model entities" to "model entities ∪ declared free parameters" so it
     passes (a free parameter was always intended to resolve from the PSet — ADR-0036's comment
     already assumed every free parameter was a model id; this lifts that assumption for an
     observation-layer nuisance).
   - the substituted **`noiseFormula`** is classified: a number → `ConstantSigma`, a bare id →
     `FreeParameterSigma` (both ADR-0021), an **arithmetic expression** → a new `FormulaSigma`.

## `FormulaSigma`: the one genuinely-new object

PyBNF's `SigmaSource` vocabulary (ADR-0021) had no "the sigma is a formula" member: an
expression `noiseFormula` like `0.1 + 0.05*scaling` (after placeholder substitution) is a
per-observable σ that depends on the *current* PSet, so it is neither a data column nor a single
free parameter. `FormulaSigma` (`noise/source.py`) closes that gap: it holds a PEtab-math
expression over free-parameter ids (+ constants), lazily compiles it to a numpy callable (the
same compile-once-per-worker, not-pickled pattern as `MeasurementModel`, ADR-0036 §5), and
evaluates it against `owner._pset_values` at each point. It is **estimated** (it references
estimated parameters), so the family's likelihood normalizer is retained (ADR-0011); its free
symbols are validated as declared nuisances via the generalized `required_free_noise_params`
(now a *set* of names per source, since a formula references several).

Its conf home is a new native source verb: `noise_model <obs> = <family>, sigma = formula
<expr>` (`parse.py` grammar + `objective._build_sigma_source`). The expression arg is the rest
of the field (no comma), so the comma-delimited field grammar is untouched. This is the line the
importer emits and the config reads — `FormulaSigma` is a first-class native source, PEtab
merely one way to reach it (ADR-0004, PEtab-defaulted not -bound).

## Scope

**In:** constant-per-observable `observableParameters` substituted into the `observableFormula`
(scale/offset); constant-per-observable `observableParameter*`/`noiseParameter*` placeholders
substituted into an expression `noiseFormula`; an arithmetic `noiseFormula` over free parameters
+ constants → `FormulaSigma`. New-era only (PEtab interop is new-era, ADR-0034). Oracled against
a **crafted** BNGL fixture (`tests/petab_fixtures/scaling_v2/`) using both an `observableFormula`
scale and an expression `noiseFormula`.

**Out (boundary raised in code, each pointing here / at #428 Phase 2):**

- A **row-varying** `observableParameters` / `noiseParameters` cell — a placeholder differing
  timepoint-to-timepoint (`observable_parameters_by_observable` /
  `noise_parameter_ids_by_observable`). **Per-condition** variation (constant within an
  experiment but differing across experiments) is row-varying under the cross-*all*-rows check
  and so is deferred too: binding a per-condition scale needs the per-data-row seam, not a
  single substituted symbol.
- **Export** of a `FormulaSigma` back to a PEtab `noiseFormula` — the importer + fit + crafted
  fixture is the gate (#428 "Done when"); the exporter-first round trip (ADR-0025) for the
  expression-noise case is Phase 2.
- A `log-normal` / `log-laplace` distribution, a per-point laplace placeholder, `param_scan` /
  dose-response — unchanged ADR-0023/0032/0037 boundaries.

## Boundaries (in code, each pointing here)

- `pybnf/petab/measurements.py` — `PetabMeasurementRow.observable_parameters` (the
  semicolon-split tokens) + `observable_parameters_by_observable` (constant-per-observable, else
  `NotImplementedError`), the sibling of `noise_parameter_ids_by_observable`.
- `pybnf/petab/formula.py` — `substitute_placeholders` (the sibling of `inline_constants`).
- `pybnf/noise/source.py` — `FormulaSigma` (lazy-compiled, estimated, multi-name nuisance).
- `pybnf/objective.py` — the `formula` source verb; `SigmaSource.required_free_params` (set)
  and the generalized `required_free_noise_params`.
- `pybnf/parse.py` — the `noise_model ... = formula <expr>` grammar.
- `pybnf/config.py` — the measurement-model allowed set widened to include declared free
  parameters; the measurement layer loaded before the free-parameter orphan check so an
  observation-layer nuisance is not mis-flagged as a typo.
- `pybnf/petab/import_.py` — substitute constant-per-observable placeholders into the
  `observableFormula` / `noiseFormula` before validation; emit a `formula` noise source for an
  arithmetic noiseFormula. Row-varying still raises.

## Consequences

- The per-measurement placeholder frontier (ADR-0033/0035/0036/0037) is **half cleared**: the
  common case (a scale/offset/sigma constant per observable — the overwhelming majority of real
  `observableParameters` usage) imports and fits; the genuinely row-varying remainder is a
  well-scoped Phase 2 with the design (the per-data-row seam) named, not open.
- `FormulaSigma` is a permanent native capability, not a PEtab-import artifact: any new-era job
  can now declare `noise_model <obs> = gaussian, sigma = formula <expr>`.
- The reduction adds **no new evaluation machinery** — it is substitution feeding the ADR-0036
  layer and the ADR-0021 σ-source engine; the only new runtime object is `FormulaSigma`, which
  is the same lazy-compiled-callable shape as `MeasurementModel`.
- See ADR-0011 (`NoiseModel` kernel), 0021 (the σ-source engine), 0033 (the deferral), 0034
  (bind-by-id, the nuisance path), 0036 (the measurement-model layer + `inline_constants`'
  printer/parser), 0037 (the constant-per-observable reduction this generalizes). Advances #428
  (the row-varying remainder is the Phase 2 follow-up).
