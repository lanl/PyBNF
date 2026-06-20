# The PEtab observableFormula expression layer is built; the exporter inlines a function body so the synthesizer has a round-trip oracle (issue #407)

**Status: Implemented (decision 2026-06-20; implemented 2026-06-20).** The reversible
translator (`pybnf/petab/formula.py`), the exporter's `inline_functions` mode, and the
importer's function synthesis are in place, with both oracle layers green (the syntactic
round trip in `tests/test_petab_formula.py` and the bngsim semantic round trip under
`-m recovery`). Supersedes the *deferral* in ADR-0033 (the bare-name analysis and the
per-measurement placeholder deferral in ADR-0033 still stand; only its "defer the
expression layer for want of an oracle" conclusion is overturned). The
expression `observableFormula` is built as a **reversible translator pair** on the existing
asset seam, and the exporter **generates its own oracle** by inlining a BNGL function body
into `observableFormula` — so the synthesizer is graded by the same byte-equal
`export → import → re-export` round trip every other importer chunk relied on (ADR-0032),
with no upstream problem and no SBML required.

## Context

ADR-0033 drew a clean boundary — a **bare-name** `observableFormula` passes through with no
translator (`import_.py::_observable_id_to_column`), an **expression** raises
`NotImplementedError` — and deferred the expression layer for two stated reasons: (1) "no
oracle," because the exporter only ever emits a bare name (it keeps the function in the model
and references it by name, ADR-0025), so there was no round trip to grade a synthesizer
against; and (2) the genuinely hard `observableParameters`/`noiseParameters` per-measurement
placeholders, which have no PyBNF analogue.

Two things have changed the calculus since:

1. **The bare-name-only stance is a PyBNF *authoring convention*, not a PEtab rule.** A BNGL
   function does not *have* to live in the model; PEtab v2 lets **any** problem — BNGL
   included — define a measurement model as an arbitrary `observableFormula` expression over
   model entities. A perfectly valid, externally-authored BNGL v2 problem can hand PyBNF
   `observableFormula = (100*pApB + 200*pApA*specC17) / (...)`. Refusing it is refusing
   conformant PEtab, not refusing an SBML-only feature.

2. **It is the prerequisite for *every* real problem, BNGL or SBML.** The BNGL-native
   importer now fits end to end (ADR-0028 Chunks 0–6, ADR-0034). The one remaining
   read-path `NotImplementedError` that bites real, externally-authored problems is the
   expression formula. SBML problems define their observables **entirely** as expressions
   (the Boehm tutorial proves it — SBML carries no observables of its own), so SBML import
   imports *nothing* without this layer. The translation is model-language-agnostic; built
   once, it serves the BNGL importer today and the SBML adapter later.

The "no oracle" objection is therefore **self-imposed**: the exporter *chose* to emit only
bare names. Lift that choice and the oracle exists.

## Decision

**Build the expression `observableFormula` layer as a reversible translator pair, and have
the exporter generate the oracle that grades it.** Concretely:

- **Exporter (oracle side).** Teach `_bngl.parse_model` to capture each global function's
  **body** (today it captures only `function_names`), and give the exporter an
  *opt-in inlining mode* in which a fitted **function** column emits its body as the
  `observableFormula` (`<body>`) instead of the bare name. The default export stays
  bare-name (model-faithful, lossless, dependency-free) — inlining is the path the
  round-trip oracle drives and that a user wanting a model-portable problem can request. A
  fitted **observable** column always stays bare (an observable is a model species/group,
  not an algebraic expression).
- **Importer (synthesizer side).** When `observableFormula` is an expression,
  `_observable_id_to_column` no longer raises: it (1) validates every free symbol against
  the BNGL entity namespace `_bngl.parse_model` already exposes (parameters ∪ observables ∪
  functions; an unknown symbol is an **error**, never a silent free parameter), (2)
  translates the PEtab math expression into a BNGL function body, and (3) **synthesizes**
  `begin functions`/`<observableId>() = <body>` into the model text before it is written,
  mapping the `.exp` column to `<observableId>`. The model is no longer carried byte-verbatim
  on this path — it gains a targeted `begin functions` edit (the same *kind* of edit ADR-0032
  once made for `__FREE`, before ADR-0034 removed that one).
- **The translator is one reversible pair on the shared seam**, mirroring every other asset
  mapper in `pybnf/petab/`: BNGL-function-body → PEtab-math (export) and PEtab-math →
  BNGL-function-body (import). The hard semantic part — operator precedence, `^`/`**`,
  `log`/`ln`/`exp`/`sqrt` spellings — is written once and run both directions.
- **`petab`/`sympy` is the optional runtime extra `pybnf[petab]`**, pulled in **only** on the
  expression path. PEtab math is a specified, `sympy`-backed grammar (`petab.math`); we
  translate via its parsed tree rather than a fragile string pass-through (ADR-0033 §"Why
  defer" warned exactly against the string approach). The bare-name path stays
  dependency-free and simulator-free; an expression import with `petab` absent raises a clear
  "install `pybnf[petab]`" error, not an `ImportError`.

## The oracle (what makes this gradable)

Two layers, weakest-to-strongest:

1. **Syntactic round trip (fast tier, no simulator).** Start from a BNGL model whose
   measurement model is a function `f() = <body>`. Export **with inlining** →
   `observableFormula = <expr>`. Import → synthesize `g() = <body′>`. Re-export with inlining
   → `<expr′>`. Grade `<body> ≡ <body′>` and `<expr> ≡ <expr′>` (sympy-normalized equality,
   not bytes — the translators are mutually inverse *up to* normalization). This grades the
   translator **pair** against itself — the exporter-first discipline of ADR-0025/0032,
   now available to this chunk because the exporter emits the expression.
2. **Semantic round trip (recovery tier, bngsim).** Simulate the original model and the
   import-synthesized model and assert the observable column is identical. This catches a
   self-consistent-but-wrong translator pair (the failure mode a purely syntactic oracle
   misses). Runs in the opt-in `-m recovery` tier.

## MVP scope (what ships in this chunk)

Inherited from ADR-0033 §"MVP scope," now built rather than deferred — **arithmetic over
existing model entities, no placeholders**:

- Free-symbol validation against the `_bngl` namespace.
- PEtab-math ↔ BNGL-function-body translation via `petab.math`/`sympy`.
- `begin functions` synthesis + column mapping on import; opt-in body inlining on export.
- Both oracle layers above, with at least one crafted multi-operator fixture (a quotient of
  sums like Boehm's, exercised in the *BNGL* scope so it needs no SBML).

## The deferral that remains (unchanged from ADR-0033)

PEtab's `observableParameters`/`noiseParameters` **per-measurement placeholders**
(`observableParameter1_*` / `noiseParameter1_*` substituted per measurement row for
scale/offset or a per-point noise value) have **no direct PyBNF analogue** (PyBNF noise is
per-observable, and there is no per-measurement observable scale/offset). The MVP **excludes**
them; the existing boundary raises stand:

- `import_.py::_observable_id_to_column` — an expression with an observable-parameter
  placeholder still raises (a placeholder is not a model entity, so symbol validation rejects
  it with a message pointing here).
- `measurements.py::read_measurement_table` — a non-numeric `noiseParameters` (a parameter
  id / placeholder, as in Boehm) still raises `NotImplementedError`.

Lifting placeholders is a later ADR. Importing Boehm specifically therefore needs **both**
this layer *and* the placeholder layer *and* the SBML adapter; this chunk delivers the first
and the largest shared piece.

## Why this, not SBML, is next (the sequencing call)

The fitter **already simulates SBML** (`config.py` routes `.xml` to `SbmlModel` /
`BngsimSbmlModel` via RoadRunner/bngsim, and those bind free parameters *by id* — exactly the
contract ADR-0034 adopted for new-era BNGL). So the SBML *model* adapter is mostly lifting
`import_.py::_require_bngl_model` and copying the `.xml` verbatim — cheap, and best done
**after** this layer, because (a) SBML observables are 100% expressions, so SBML import is
inert until the formula layer exists, and (b) deferring SBML keeps the `python-libsbml`
dependency weight off the critical path until it is actually exercised.

## Boundaries (in code, each pointing here)

- `_bngl.py` — `parse_model` gains a `function_bodies` (name → body) mapping alongside
  `function_names`; the bare-name path ignores it.
- `pybnf/petab/observables.py::petab_observable_row` — gains the ability to emit a function
  body as `observable_formula` under the exporter's inlining mode.
- `pybnf/petab/import_.py::_observable_id_to_column` — routes an expression to the translator
  + synthesizer instead of raising; an unknown symbol or a placeholder still raises here.
- A new translator module (e.g. `pybnf/petab/formula.py`) — the reversible
  PEtab-math ↔ BNGL-function-body pair, `petab`/`sympy` imported lazily.

### Implementation notes (confirmed 2026-06-20)

- **Math grammar import path:** `petab.v2.math` (the v2-specific grammar) — both
  `sympify_petab` (string → sympy tree) and `petab_math_str` (sympy tree → PEtab math
  string) live there. `petab.math` is a v1 alias of the same symbols; we use the v2 path.
- **Both directions parse via `sympify_petab(..., evaluate=False)`** (so the written
  structure is preserved). The forward (export-inline) direction serializes with petab's
  own `petab_math_str` so the emitted formula is exactly what petab's validator accepts;
  the reverse (import-synthesis) direction serializes with a small `sympy.StrPrinter`
  subclass we own, because BNGL math differs from sympy's default on the `^` power
  operator, the `ln`/`log10`/`log2`/`sqrt` spellings, and BNGL's zero-arg `func()`
  reference convention.
- **Why we own the BNGL printer (not `petab_math_str` reversed):** petab 0.8.2's
  `petab_math_str` serializes a one-half power as the precedence-unsafe `z ^ 1/2` (which
  re-parses as `z/2`). The MVP arithmetic surface never hits it, but owning the BNGL
  printer keeps the reverse direction precedence-safe regardless — exactly the
  silent-wrongness ADR-0033 warned about, caught and contained here rather than trusted.
- **BNGL function references** (`f()` for a global function) are bridged by a bounded,
  anchored rename of the *known* function names (export side strips `f()`→`f`; the BNGL
  printer re-appends `()` for a function-kind symbol) — not a general math tokenizer, which
  stays the job of `sympify_petab`.

## Consequences

- The importer reads conformant **BNGL** v2 problems with expression observables — not just
  PyBNF's own bare-name exports — with oracle backing, advancing "reads the ecosystem"
  honestly rather than by self-round-trip alone.
- The exporter can emit a model-portable problem (the function inlined into the table) for
  consumers that cannot read a BNGL function in the model.
- `petab`/`sympy` enters as an *optional runtime extra*; core and the bare-name path stay
  dependency-free (ADR-0019).
- The SBML model adapter becomes a small, well-scoped follow-on rather than a parallel build.
- See ADR-0025 (exporter-first; a function is the measurement model), ADR-0026 (the BNGL
  model adapter + entity namespaces), ADR-0032 (importer read path + the round-trip
  discipline), ADR-0033 (the boundary this supersedes; placeholder deferral stands), and
  ADR-0034 (bind-by-id; the verbatim-carry this chunk edits on the expression path).
