# BNGL is a first-class PEtab model language: a runtime-registered `BnglModel` adapter gives the exporter the full `petablint` oracle (issue #420)

The PEtab v2 exporter (ADR-0025, chunk 1) emits a `language: bngl` problem, but its
external oracle is **partial**: `petab` 0.8.2 (= `PEtab-dev/libpetab-python`) ships only
`sbml` and `pysb` model loaders, so `petablint <problem.yaml>` hard-fails on
`language: bngl` with `ValueError: Unknown model format: bngl` *before any table check*.
ADR-0025's salvage was a **model-less** `petab.v2.Problem(models=[], …)` that runs the
~13 table-level validation tasks and **excludes the ~5 model-cross tasks** (`CheckModel`,
`CheckObservablesDoNotShadowModelEntities`, `CheckAllParametersPresentInParameterTable`,
`CheckValidConditionTargets`, `CheckInitialChangeSymbols`), self-asserting their intent in
our own tests. That leaves the exporter's most important guarantees — *every
`observableFormula` is a real model entity*, *every `parameterId` is a model parameter*,
*observable ids don't shadow model entities* — graded only by hand-written assertions, not
by petab.

The deeper framing (memory, #420; verified upstream **PEtab-dev/PEtab#436** "Add support
for non-SBML models", OPEN, milestone PEtab 2.0.0, by petab lead `dweindl`, explicitly names
`bngl`): the real PyBNF↔PEtab integration is **BNGL as a first-class PEtab model language**,
not a pile of file adapters. PEtab v2 was deliberately made model-agnostic (PR #538); the
`petab.v1.models.model.Model` ABC (~14 tiny introspection methods; `PySBModel` = 264 lines)
exists for exactly this — and `bngl` is *named-but-unimplemented*, maintainers want
contributors, no competing effort. Their rule-based-model convention (reference *named*
observables/expressions, not species formulas) **is already our ADR-0025 rule**
(`observableFormula` = bare model name).

**Decision: implement `BnglModel(petab Model)` backed by PyBNF's own stdlib BNGL block
reader, and register it into `petab` at runtime by rebinding `petab.v2.core.model_factory`.
Then the exporter's test upgrades to the FULL validation task set (drop the `_MODEL_TASKS`
exclusion) and `Problem.from_yaml` loads + lints the BNGL problem at model level.** This is
**Step A** (a local reference implementation); **Step B** — upstreaming the same code as a
`libpetab-python` PR mirroring `PySBModel` — is sequenced after Step A proves the adapter,
and is out of scope here.

This was **de-risked by a throwaway probe before writing this ADR**: a minimal `BnglModel`
+ `register_bngl()` made `Problem.from_yaml(demo/problem.yaml)` load the model and pass
**all 18** default validation tasks with zero errors — including every previously-excluded
model-cross task *and* `CheckValidParameterInConditionOrParameterTable` (which silently
no-ops on `model is None` and only becomes a real check once a model is present). The
sections below are therefore verified design, not speculation, and the entity-enumeration
decisions were each **checked against BNG2.pl** (the source of truth) during a design review,
not inferred from the `PySBModel` analogy.

## Terminology (added to `CONTEXT.md` during the design review)

A fitted `.exp` column matches an **Observable** *or* a **Global Function** — the precise
BNGL term for a `begin functions` entry named `name()` (any legal BNGL name + `()`). A
global function is *implicitly* the measurement model when its name appears in both a
simulation-output header and an `.exp` header; it is never declared as one. "Model entity"
is **petab's own** vocabulary (`has_entity_with_id`, `CheckObservablesDoNotShadowModelEntities`)
for any named thing the model declares; in BNGL that is the union { parameter, observable,
global function, molecule type, compartment, seed species }. BNGL has **no first-class
"species"** in a model file — only the concrete species written in `begin seed species`;
the full species set is a *network-generation* product that lives in a `.net`, which
validation never produces.

## What `petab` actually asks of a `Model` (the 14 ABC methods, and which checks read them)

The `Model` ABC (`petab/v1/models/model.py`) is pure introspection. Validation needs only
*parsing* — never network generation — for every method **except `is_valid`**, where we *do*
shell out to BNG2.pl when it is available (see "is_valid", below). Grouped by what backs each
in BNGL:

| `Model` method | BNGL source | demo (`parabola.bngl`) | read by |
|---|---|---|---|
| `get_parameter_ids()` | `parameters` names | `v1, v2, v3` | — |
| `get_parameter_value(id)` | numeric RHS → `float`; **expression RHS → `NotImplementedError`**; unknown id → `ValueError` | `v1=5` | — |
| `get_free_parameter_ids_with_values()` | `parameters` with numeric RHS | `(v1,5),(v2,5),(v3,5)` | — |

> **Superseded by issue #666 (PR #673):** both rows now *evaluate* an expression RHS.
> A parameters block is arithmetic over other parameters, so resolving it needs no
> network generation — see `pybnf.petab._bngl_expr`. `get_parameter_value` returns the
> number (or a `ValueError` naming why it could not be computed), and
> `get_free_parameter_ids_with_values()` returns every parameter it can resolve,
> warning about the rest instead of dropping them in silence.
| `has_entity_with_id(id)` | **params ∪ observables ∪ global functions ∪ molecule types ∪ compartments ∪ seed species** | `True` for `v1`/`x`/`y`/`counter`; `False` for `obs_x` | `CheckObservablesDoNotShadowModelEntities`, `CheckValidParameterInConditionOrParameterTable` |
| `get_valid_parameters_for_parameter_table()` | `parameters` names | `v1, v2, v3` | `CheckAllParametersPresentInParameterTable` (`allowed`) |
| `get_valid_ids_for_condition_table()` | parameters ∪ compartments | `v1, v2, v3` | `CheckValidConditionTargets`, `CheckValidParameterInConditionOrParameterTable` |
| `symbol_allowed_in_observable_formula(id)` | **params ∪ observables ∪ global functions** (the BNG `ParamList`) | `True` for `x`,`y`,`v1`; `False` for `obs_x` | `Problem.get_output_parameters()` → feeds `CheckAllParametersPresentInParameterTable` |
| `is_state_variable(id)` | **`seed species` only → `True`, else `False`** | `True` for the seed species, `False` for `v1` | `CheckValidParameterInConditionOrParameterTable` |
| `is_valid()` | **`BNG2.pl --check` when locatable, else `True`** | `True` | `CheckModel` |
| `type_id` | class attribute `'bngl'` | `'bngl'` | yaml round-trip |
| `model_id` | yaml model key / file stem | `parabola` | `__repr__`, errors |
| `from_file(fp, model_id, base_path)` | read text, parse | — | `model_factory` |
| `to_file(filename)` | write the BNGL text | — | (symmetry; unused by lint) |
| `__init__` | concrete (holds parsed text + entity sets) | — | — |

The single method that makes the exporter's bare-name `observableFormula` strategy *lint-clean*
is **`symbol_allowed_in_observable_formula`**: `get_output_parameters()` walks each formula's
free symbols and, for any symbol the model vouches for, treats it as a model entity rather than
an output parameter that must appear in `parameters.tsv`. Because `x` (observable) and `y`
(global function) are vouched for, they are *not* demanded as parameters — exactly the
function-as-measurement-model design of ADR-0025, now enforced by petab instead of by us.

Two ABC subtleties (mirrored from `PySBModel`): `type_id` is declared abstract as a
`classmethod`+`property` but is satisfied by a **plain concrete class attribute**
(`type_id = 'bngl'`); `from_file` is a `@staticmethod` returning a fresh instance. PyBNF's
`v1/v2` valid-identifier rule accepts `parabola`/`v1`/`x`/`y` unchanged (no mapping table —
ADR-0025).

## The two name-set methods are verified against BNG2.pl, not the PySB analogy

`symbol_allowed_in_observable_formula` and `has_entity_with_id` are membership tests over BNGL
names; their *contents* were settled by reading BNG2.pl's `Perl2/` modules (the source of
truth), which corrected a wrong first guess:

- **`symbol_allowed_in_observable_formula` = parameters ∪ observables ∪ global functions.**
  BNG resolves every bareword in an expression through the `ParamList` (`Expression.pm`,
  `"Can't reference undefined parameter $name"`), whose only member types are
  `Constant`/`ConstantExpression` (parameters), `Observable`, `Function`, and `Local`
  (`Param.pm`). **Compartments never enter the `ParamList`** (grep of all of `Perl2/` finds no
  `ParamList->set` for a compartment) — so a compartment name is *not* a legal formula symbol,
  contrary to the initial PySB-analogy guess that almost added it. `Local` is a transient
  function argument, never a top-level symbol.
- **`has_entity_with_id` = the full declared-identifier namespace** (the 6-way union above).
  petab uses it to tell "a model thing" from "an output parameter", where **over-inclusion is
  the safe direction**: a missing entity is a missed error, an extra one is no false positive
  (the `obs_`/`func_` prefixes keep PEtab ids clear). It folds in molecule types and seed
  species beyond the `ParamList`.
- **Nothing else contributes a name** — verified, not omitted: `EnergyPattern.pm`'s struct has
  **no `Name` field** (an energy pattern is a `{Pattern, Gf, Weights}` triple with at most a
  line label), so energy models add no entity; **actions** (`simulate`, `generate_network`) are
  control directives, never `ParamList` members, and are stripped on export; **built-in/special
  functions** (`sin`, `mratio`) are reserved math that lives inside global-function bodies
  carried verbatim in the model, and sympy parses them as functions, not free symbols, so the
  two name-set methods are never consulted about them. **Local functions** `f(x)` need no
  special handling: the *name* `f` is an ordinary `Function` param (caught by reading the
  functions block); the arg `x` is local.

## Entity-enumeration source: extend the stdlib block reader; one reader, two consumers

Enumerate BNGL entities by **extending the exporter's own dependency-free block reader**, not by
reaching into `pset.BNGLModel` / `bngsim_model`. The latter drags in `bngsim` — a mac-only,
private wheel with no Linux build (the CI blocker; `lanl/PyBNF#414`) — which would push
`BnglModel` *out of the bngsim-less CI tier* the whole `pybnf/petab/` package deliberately lives
in (ADR-0019/0023), and reaches toward simulation concerns validation never needs. The stdlib
scanner keeps `BnglModel` **stdlib-only, simulator-free, CI-runnable**.

To avoid **two drifting BNGL readers** (the recurring "verify, don't trust an earlier note"
lesson), factor the canonical scanner into a new **`pybnf/petab/_bngl.py`**: a
`parse_model(text) -> BnglEntities` frozen dataclass carrying `parameters: dict[str,str]`
(name → raw RHS), `observable_names`, `function_names`, `molecule_type_names`,
`seed_species`, `compartment_names`, and `free_to_param`. `export.py`'s `_read_bngl`/
`_BnglModel` is refactored to *consume* `parse_model` (behavior-preserving — its 22 tests stay
green, only the parse moves), and `BnglModel` (in **`pybnf/petab/bngl_model.py`**, mirroring
petab's `pysb_model.py`) wraps the same `BnglEntities` into the ABC. One reader, two consumers —
the neutral-seam discipline ADR-0025 used for `PetabParameterRow`.

Grammar points the enrichment must honor (verified against the EBNF + BNG2.pl):
- **Parameters** are `Name (WS | "=") MathExpression` — both `v1 v1__FREE` *and* `NA = 6.02e23`
  parse (split on the first `=` *or* whitespace). `get_parameter_value` returns a `float` for a
  bare-number RHS, raises `NotImplementedError` for an **expression RHS** (`k_on 2*base_rate` —
  evaluating an expression tree is the simulation-grade work scoped out), and `ValueError` for
  an unknown id (the ABC's documented exception).
  **Superseded by issue #666 (PR #673):** the expression RHS is evaluated now; of this
  sentence only the unknown-id `ValueError` survives.
- **Observables** are `("Molecules"|"Species"|"Counter") WS Name WS Pattern` — name is the
  *second* token, guarded by a leading observable keyword.
- **Global functions** are `Name "(" [args] ")" (WS|"=") MathExpression` — `y()=…`, `f(x)=…`.
- **Seed species** are `Pattern value`; the (often composite) pattern string is matched verbatim,
  no canonicalization (a `.net` would be needed to canonicalize, and validation produces none).
- **Compartments** are `Name WS (2|3) WS size [outside]` — name is the first token; optional block.

## Runtime registration: rebind `petab.v2.core.model_factory`, idempotently, no teardown

`petab`'s `model_factory` (`petab/v1/models/model.py`) is a **hardcoded `if/elif` on language
with no plugin hook** — unknown language → `ValueError`. `petab.v2.core` does `from
..v1.models.model import … model_factory` at import (line 51) and *calls the name resolved in
its own module globals* (line 1273, inside `Problem.from_yaml`). So `register_bngl()` must:

1. **Rebind `petab.v2.core.model_factory`** to a wrapper that routes `model_language == 'bngl'`
   → `BnglModel.from_file(location, model_id=…, base_path=…)` and **delegates everything else to
   the captured original** (so `sbml`/`pysb` are untouched). Rebinding the *v2.core* binding is
   what matters — that is the name v2 actually calls.
2. **Add `'bngl'` to `petab.v1.models.known_model_types`** (define `MODEL_TYPE_BNGL = 'bngl'`),
   so `bngl` is an honestly *known* type rather than reported as *unknown* on any unrebound path.
3. Be **idempotent**: stash the original under a sentinel on first call and reuse it, so
   re-registration neither double-wraps nor loses the original.

**Guarded permanent rebind, no teardown** (chosen over a reversible context manager): the
mutation is purely additive (sbml/pysb unchanged), idempotent, and confined to the test tier,
so the blast radius is nil; a context manager is more machinery for a stand-in whose whole
purpose is to be deleted once `libpetab-python` ships a `bngl` loader (then `register_bngl()`
collapses to a no-op).

`register_bngl()` is **public API on `pybnf.petab.bngl_model`**, called by the **test fixture**
(and any future loader of a BNGL PEtab problem). It is deliberately **not** called from
`export_job`: the exporter only *writes* files and must stay petab-free so core keeps its
dependency-free guarantee (ADR-0025 — `petab` is a *test-time oracle*). For the same reason
`bngl_model.py` (which `import`s `petab`) is **not** imported by `pybnf/petab/__init__.py`.

## `is_valid`: real BNG2.pl validation when available, graceful `True` fallback

`CheckModel` calls only `is_valid()`. Rather than the PySB precedent (`return True`, a structural
no-op), `BnglModel.is_valid()` runs **`BNG2.pl --check <cleaned-model>`** when a BNG2.pl is
locatable (via `BNGPATH` / PyBNF's existing `config_schema._default_bng_command`), and falls back
to `True` when it is not. Justification, established during the review against the actual
environments and de-risked on the acceptance artifact:

- **BNG2.pl is already a hard PyBNF prerequisite**, not a new dependency (the `setup-pybnf` CI
  action: *"BNG2.pl is still required: Configuration validation execs `BNG2.pl -v`"*), and it is
  present **in the CI leg that runs this test** (`tests.yml` fetches BioNetGen 2.9.3, sets
  `BNGPATH`); it is also present on dev machines and the pre-push hook. So real validation runs
  wherever the oracle test actually executes.
- **`BNG2.pl --check` is parse + semantic validation without network generation** — it reads and
  checks every block (undefined parameters, malformed patterns, duplicate names) and exits. On
  the cleaned, actions-stripped `parabola.bngl` it returns **exit 0 in ~0.3 s, no network gen**
  (de-risked directly). So the "no simulation" scope holds.
- **Graceful `True` fallback** means `is_valid` never produces a *false* failure where BNG2.pl is
  absent, and keeps the adapter portable for Step B's upstream (a petab user from the SBML world
  may lack a BNG backend → "best validation available", exactly as SBML always-validates because
  libsbml is always present).

## Test contract (the acceptance)

- **Upgrade `tests/test_petab_export.py::test_table_level_petab_validation_is_clean`** to the
  FULL task set: drop the `_MODEL_TASKS` exclusion, `register_bngl()`, build the problem via
  `Problem.from_yaml(exported/'problem.yaml')` (the real `petablint` path — exercises
  `model_factory` → `from_file` → `BNG2.pl --check`), and assert **zero ERROR-level issues across
  all `default_validation_tasks`**. (Probe-confirmed green on the demo.)
- **Unit-test the `BnglModel` ABC directly** against the parsed demo: `get_parameter_ids` =
  `{v1,v2,v3}` with values; `has_entity_with_id` True for `v1`/`x`/`y`/`counter` and False for
  `obs_x`/unknowns; `symbol_allowed_in_observable_formula` True for `x`,`y`,`v1` and False for
  `obs_x`/`func_y`/compartments/unknowns; `is_state_variable` True for the seed species, False
  for `v1`; `get_parameter_value` returns the nominal, raises `ValueError` on unknown and
  `NotImplementedError` on an expression-valued parameter
  (**superseded by issue #666 (PR #673):** the expression is evaluated instead).
- **Unit-test `register_bngl()`** idempotency (two calls; `sbml`/`pysb` still route to the
  originals) and that the `_bngl.parse_model` refactor leaves the exporter's 22 tests green.
- **Make the oracle run in CI**: add `petab` to the `setup-pybnf` composite action so
  `Problem.from_yaml` + the full task set actually execute in hosted CI (today the test
  `importorskip`s petab and skips in CI; BNG2.pl is already present there). Local + pre-push
  already exercise it.
- **Stay dependency-light**: `BnglModel` *parsing* is stdlib (no bngsim); BNG2.pl is invoked only
  inside `is_valid` and only when present; `petab` stays a `pybnf[tests]` extra. `ruff` clean.

## Scope & boundaries (chunk A)

**In:** validation-grade (lint) for PyBNF-shaped BNGL — numeric parameters, observables, global
functions, seed species, optional compartments; the demo (`parabola.bngl`) and its kin.

**Out (raised or documented, never silently mis-handled):**
- **Simulation** — network generation / ODE export for pyPESTO/AMICI (a separate, larger effort
  via BNGL→SBML or AMICI-BNGL). `BnglModel` is introspection-only; `is_valid` runs only the
  no-network `--check`.
- **Expression-valued parameter *values*** (`k_on 2*base_rate`): the id is enumerated (it is a
  model entity), but `get_parameter_value` raises `NotImplementedError` rather than evaluating an
  expression tree. Confined, not silent.
  **Superseded by issue #666 (PR #673).** The confinement was the wrong call, and it was not
  really confined: `get_free_parameter_ids_with_values()` *dropped* the parameter in silence,
  and 20.8% of parameter declarations across our model corpora (1934 of 9323) are
  expression-valued. A parameters block needs no network generation to resolve, so this is
  now in scope; the BNGL semantics are pinned against a real BNG2.pl in
  `tests/test_petab_bngl_expr.py`.
- **The full generated-species list** (network gen). `is_state_variable` answers at the
  seed-species grain validation needs; it does not enumerate the reaction network. For the demo
  (no conditions) it is never consequential. Revisited when the conditions/experiments export
  chunk gives the method real requirements.
- **The upstream `libpetab-python` PR** — Step B, sequenced after this lands.

## Considered options

- **Stay with the partial table-level oracle (ADR-0025 status quo).** Rejected: leaves the
  exporter's headline guarantees graded only by hand assertions, and forgoes the first-class-BNGL
  direction #420/#436 calls for.
- **Emit an SBML twin and lint under `language: sbml`** (ADR-0025 option B). Rejected for chunk A:
  needs BNG `writeSBML`, and BNGL functions become SBML assignment rules with *different* entity
  names — a lossy cross-check, not a faithful BNGL model oracle.
- **Back the adapter with `pset.BNGLModel` / `bngsim`.** Rejected: drags the mac-only `bngsim`
  wheel into the validation path and breaks the bngsim-less CI tier.
- **`is_valid` = `True` (PySB-mirror).** Rejected: BNG2.pl is already present where the test runs,
  so a real `--check` makes `CheckModel` a genuine check at negligible cost; the trivial version
  was the initial recommendation, reversed once CI was verified to carry BNG2.pl.
- **Fork `petab` to add a plugin hook / vendored loader.** Rejected: heavier than a two-binding
  idempotent rebind, and Step B (a clean `libpetab-python` PR) is the right place to add real
  extensibility upstream.

Relevant ADRs: **0025** (exporter-first; the partial-oracle limitation this lifts; naming /
bare-formula conventions), **0019/0023** (the dependency-free / bngsim-less `pybnf/petab/` tier
and the neutral-seam discipline reused for `_bngl.parse_model`). Issues: **#420** (this work),
**#407** (umbrella), upstream **PEtab-dev/PEtab#436** (Step B target / spec). Follow-ups: Step B
(`libpetab-python` PR mirroring `PySBModel`); conditions/experiments + dose-response export (which
will exercise `get_valid_ids_for_condition_table` / `is_state_variable` for real).
