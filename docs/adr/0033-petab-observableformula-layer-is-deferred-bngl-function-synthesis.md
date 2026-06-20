# The PEtab observableFormula layer: a bare name passes through; an expression is deferred BNGL function synthesis with petab/sympy as an optional extra (issue #407)

**Status: Superseded in part by ADR-0035 (2026-06-20).** The deferral conclusion below is
overturned — the expression `observableFormula` layer is now *built* (ADR-0035), because the
"no oracle" objection was self-imposed: the exporter can inline a function body to generate
its own round-trip oracle. **What still stands:** the bare-name analysis (a bare name needs
no translator and is dependency-free), the MVP scope, and the deferral of the
`observableParameters`/`noiseParameters` **per-measurement placeholders** (no PyBNF
analogue). Read ADR-0035 for the current decision; this ADR remains the record of the
bare-name boundary and the placeholder frontier.

*(Originally — Accepted, boundary drawn; synthesis deferred, 2026-06-19.)* The BNGL-native
importer (ADR-0032) reads the **bare-name** observableFormula common case with no formula
translator and dependency-free. An **expression** observableFormula raises a clear
`NotImplementedError` at two seams (`import_.py::_observable_id_to_column` for the formula,
`measurements.py::read_measurement_table` for a parameter-id `noiseParameters`). This ADR
records *why* the bare-name path needs no translator, *what* an expression import would
take, and *why* it is deferred rather than shipped speculatively.

## Context

A PEtab v2 observables row gives an `observableFormula` — the model-output expression a
measurement is compared against. Two shapes occur in the wild:

- **A bare model-entity name** — `observableFormula = y`, where `y` is a BNGL observable or
  (usually) a `function` carried verbatim in the model file. PyBNF's objective matches an
  `.exp` column to a model observable/function **by name** and evaluates that entity *in the
  model* during simulation; it never parses `observableFormula`. So the bare-name case is a
  pure *rename* (column header ↔ `observableId`), which the exporter writes (ADR-0025,
  "a function is the measurement model") and the importer reads — no translator, no `sympy`,
  runs in the bngsim-less CI tier.
- **An arithmetic expression over model entities** — e.g. Boehm's `observableFormula =
  (100 * pApB + 200 * pApA * specC17) / (pApB + STAT5A * specC17 + 2 * pApA * specC17)`,
  often with PEtab `observableParameters`/`noiseParameters` placeholders for per-measurement
  scale/offset. **No PyBNF model entity has that name**, so there is nothing for the column
  to match by name.

Every real, externally-authored v2 problem we have (the Boehm tutorial, the only v2 example
upstream) uses the expression form — and is SBML besides — so it hits this boundary together
with the SBML boundary. No upstream **BNGL** v2 problem exists (PyBNF is pioneering
BNGL-as-PEtab, #420), so this layer has *no published problem to read and no oracle*.

## Decision

**The bare-name observableFormula passes through unchanged (the live, dependency-free path).
An expression observableFormula is a deferred, bounded BNGL *function-synthesis* problem:
import it by emitting a `begin functions` entry `<observableId>() = <translated expr>` and
naming the `.exp` column `<observableId>`. When built, only that formula layer may adopt
`petab`/`sympy` as an *optional extra*; it must never be pulled onto the bare-name path. It
is deferred now behind a clear boundary raise, not shipped speculatively.**

Synthesizing an algebraic BNGL function (not a reaction network) is exactly the bounded BNGL
*generation* ADR-0025 set aside when it chose exporter-first: reading correct BNGL is easy,
writing it is the work. Adding a `begin functions` entry is a targeted edit to the model
text — the same *kind* of edit the importer once made to re-add `__FREE` markers (since
removed: ADR-0034 made new-era BNGL bind free parameters by id, so the model is now carried
verbatim except for exactly this synthesized-function case, ADR-0035).

### The MVP scope (when this is built)

The tractable first slice — **arithmetic over existing model entities, no placeholders**:

1. Validate every free symbol in `observableFormula` is a known model entity
   (`_bngl.parse_model` already exposes the parameter/observable/function namespaces,
   ADR-0026) — a symbol that is neither is an error, not a silent free parameter.
2. Translate the expression to a BNGL function body. PEtab math and BNGL function bodies are
   both standard infix arithmetic, but the surface is **not** identical (power operator,
   function-call spellings, division/precedence edge cases), which is precisely why the
   translation wants `sympy`'s parsed tree (via `petab`'s math module) rather than a fragile
   string pass-through.
3. Emit `<observableId>() = <body>` into `begin functions` and map the column to
   `<observableId>` (the same neutral seam the bare-name path uses downstream).

### The genuinely hard part (a separate, deeper deferral)

PEtab's `observableParameters`/`noiseParameters` **placeholders** — `observableParameter1_*`
/ `noiseParameter1_*` substituted **per measurement** (a row in the measurements table) for
scale/offset or a per-point noise value — have **no direct PyBNF analogue**: PyBNF's noise
model is per-observable, not per-measurement, and it has no per-measurement observable
scale/offset. Boehm exercises both (an expression formula *and* a `noiseParameters` column
carrying a parameter id). The arithmetic MVP above explicitly excludes placeholders; mapping
them is the real frontier and a later ADR.

## Why defer rather than ship a translator now

- **No oracle.** The exporter never emits an expression observableFormula (it keeps the
  function in the model and references it by name, ADR-0025), so there is no byte-equal
  `export → import → re-export` round trip to grade a synthesizer against — the discipline
  every other importer chunk relied on (ADR-0032). The only real fixtures are SBML.
- **Silent-wrongness risk.** An oracle-less string translator can be subtly wrong (operator
  precedence, `^` vs `**`, `pow`/`log` spellings) in ways a weak "asserts its own output"
  test would not catch. A wrong measurement model is worse than a refused one.
- **MVP-first, boundary-in-code.** The bare-name path + the broadened noise recovery
  (ADR-0032 follow-up: a `noise_model =` line for fixed/free sigma) already advance "reads
  the ecosystem" with oracle backing. The expression layer is the next step, gated on a
  *crafted* BNGL oracle and the optional `petab`/`sympy` extra — built deliberately, like the
  read path deferred its own hard recipe cases.

## Boundaries (in code, each pointing here)

- `import_.py::_observable_id_to_column` — a non-bare `observableFormula` raises
  `NotImplementedError` naming the synthesis target (`<id>() = <translated expr>`), the
  optional petab/sympy extra, and the placeholder deferral.
- `measurements.py::read_measurement_table` — a **non-numeric** `noiseParameters` (a
  parameter id / placeholder override, as in Boehm) raises `NotImplementedError` rather than
  a raw `float()` `ValueError`: this read path reconstructs a numeric per-point `_SD`, and
  the per-measurement placeholder override is the deferred semantics above.

## Consequences

- The importer reads the bare-name observable common case and the whole PEtab v2 noise
  surface (objective tokens + `noise_model` lines) with oracle backing, and stays
  dependency-free + simulator-free on that path.
- An expression observableFormula is a *documented* boundary, not a crash or a silent
  mis-import; the path to lifting it (arithmetic translation via the optional extra, then the
  placeholder semantics) is scoped here.
- See ADR-0025 (a function is the measurement model; exporter-first), ADR-0026 (the BNGL
  model adapter + entity namespaces), ADR-0032 (the importer read path), and the chunk-2
  noise broadening it documents.
