# PEtab per-observable noise imports as a per-observable fit sigma when the placeholder is constant per observable; a row-varying placeholder is deferred (issue #407)

**Status: Accepted and implemented (decision + implementation 2026-06-20).** The final #407
import chunk: the externally-authored **Boehm** v2 problem now imports end to end. The
measurement reader records a parameter-id `noiseParameters` placeholder
(`measurements.py::PetabMeasurementRow.noise_parameter_id` +
`noise_parameter_ids_by_observable`); the importer emits per-observable
`noise_model <obs> = <family>, <param> = fit <id>` lines (`import_.py::_objective_directives`
now returns a *list* — a single whole-fit line, or a base objective plus per-observable
overrides); a fixed PEtab parameter the model file lacks is inlined into the
`observableFormula` (`formula.py::inline_constants`). Oracles green:
`tests/test_petab_import.py::TestRealWorldBoehmV2` (full import + a `Configuration` load),
`TestPerObservableNoiseImport` (a dependency-free crafted BNGL problem), the
`noise_parameter_ids_by_observable` guard unit test, and the new
`tests/test_petab_sbml_layer.py::TestBoehmRecovery` (`-m recovery`: RoadRunner reproduces the
published measurement table; RoadRunner and bngsim agree on the materialized columns).

ADR-0021 built the engine (per-observable `(NoiseModel, σ-source)` overrides) and ADR-0036 the
measurement-model observation layer; ADR-0032 imported the *whole-fit* objective. What remained
for "PyBNF reads a real, externally-authored v2 problem end to end" was the noise shape real v2
problems actually use — a **placeholder** noise mechanism — and the resolution of an
observableFormula symbol that lives only in the parameters table. Both are import-side (emit)
work over engines that already exist, not new abstractions.

## The load-bearing distinction: constant-per-observable vs row-varying

A PEtab observable declares its noise scale as a **placeholder** in `noiseFormula` (a
`noiseParameter*` token, or a bare id listed in `noisePlaceholders`); the measurements'
`noiseParameters` column supplies the substitution value **per measurement row**. The kind of
value decides the PyBNF mapping:

- **A number** (the per-point `_SD` cell). Already imported: `objective = chi_sq` (a per-point
  data-column σ-source). Unchanged.
- **A parameter id constant across all of an observable's rows** (Boehm's `sd_pSTAT5A_rel`).
  This *is* PyBNF's native **per-observable estimated sigma** (ADR-0021): the placeholder is the
  same free parameter at every timepoint, so it imports as
  `noise_model <obs> = <family>, <param> = fit <id>`. The id is emitted as an ordinary free
  parameter and bound as a **nuisance** (it matches no model parameter id — ADR-0034's allowed
  nuisance path), exactly as `chi_sq_dynamic`'s free sigma is.
- **A parameter id that *varies* across an observable's rows** (a different scale per timepoint),
  or a row that **mixes** a parameter id with numeric values. This is genuinely
  per-measurement: PyBNF noise is per-**observable** (one σ-source per column), so there is no
  analogue. `noise_parameter_ids_by_observable` **raises `NotImplementedError`** — the boundary
  is in code, not a silent mis-import.

This constant-per-observable vs row-varying line is the decision this ADR pins. It is checked
once, cross-row, in `noise_parameter_ids_by_observable`; the per-row reader only classifies the
token (numeric → `noise_parameters`, else → `noise_parameter_id`).

## The objective directive is a list, not a line

`import_.py::_objective_directive` collapsed the observables' noise to **one** whole-fit line and
**raised** when sigma sources differed per observable. It now returns a **list**
(`_objective_directives`):

- **Uniform** (one family + one σ-source across all observables) → a single line, the tidy
  byte-for-byte-round-tripping case preserved exactly: an `objective = chi_sq/sos/sod/ave_norm_sos`
  sugar token, or a whole-fit `noise_model = <family>, ...` line (`_try_uniform_directive`, which
  returns `None` to signal "not uniform" instead of raising).
- **Per-observable** (the Boehm shape: each observable its own σ-source) → a structural base
  `objective = chi_sq` plus one `noise_model <obs> = ...` override per observable
  (`_per_observable_directives`). Under edition ≥ 2 a base objective is required and the
  per-observable overrides "accompany" it (`config.py`); since every observable is overridden the
  base is a never-exercised structural default (Gaussian, no free parameter, no data column), so
  `chi_sq` is the neutral choice. Each override names the **column** the objective compares (the
  measurement-model column = `observableId` for an expression observable, else the model entity).

## A fixed parameter the model file lacks is inlined into the formula

Boehm's `observableFormula` references `specC17` (a species-activity constant, `0.107`) that lives
only in the PEtab **parameters table** (`estimate=false`), not in the SBML. It resolves against
neither the model namespace (so the importer's symbol validation would reject it) nor the
simulation trajectory (so the measurement layer could not evaluate it). Because it is **fixed**,
substituting its numeric value is **exact**: `formula.py::inline_constants` parses the formula
(`sympify_petab`, never a string tokenizer — ADR-0033), substitutes the fixed-parameter symbols
that are **not** model entities (a fixed parameter that *is* a model entity stays a symbol — it
resolves as a model constant), and re-serializes through the precedence-safe PEtab printer with
the same numeric round-trip self-check the exporter uses. The model file stays **unedited**
(ADR-0036), and the bare-name / model-only common case never reaches `sympy` (nothing to inline →
the formula is returned verbatim, so the demo round trip is byte-stable).

Inlining (vs carrying the constant to the config) is the contained choice: the config builds the
measurement layer's namespace + constants from model files only, so carrying `specC17` would need
a new conf surface; inlining needs none and is exact for a fixed value.

## The Boehm recovery oracle

Boehm has no closed form (the crafted decay model in `test_petab_sbml_layer.py` keeps the analytic
oracle). Its recovery oracle is the **published optimum**: the vendored SBML embeds the fitted
parameter values, so simulating the imported problem at those values and applying the imported
measurement layer reproduces the published measurement table to its fitted noise (correlation
> 0.9, RMSE < 0.15 × data range across all three observables), and RoadRunner and bngsim agree on
the materialized columns to 1e-3 — proving the import is runnable + correct on a real stiff model
with assignment/initial-assignment rules, on both backends.

## Boundaries (each still raises, in code)

- A **row-varying** `noiseParameters` parameter id, or a mix with numeric values
  (`noise_parameter_ids_by_observable`) — the genuinely per-measurement frontier.
- `observableParameters` / per-measurement `observableParameter*` / `noiseParameter*` scale-offset
  placeholders inside a formula — no PyBNF analogue (`formula._check_symbols`, unchanged).
- A `log-normal` / `log-laplace` noise distribution, an expression `noiseFormula`, a per-point
  laplace placeholder — unchanged ADR-0023/0032 boundaries.

See ADR-0021 (the per-observable `(family × σ-source)` engine), 0034 (bind-by-id, the nuisance
path), 0036 (the measurement-model observation layer + `inline_constants`' sibling printer/parser),
0032 (the whole-fit import this extends). Closes the `noiseFormula` / per-observable noise checkbox
on #407; the catalog-parity checkbox (5 prior families + one-sided truncation, #417) is the
remaining import work.
