# A parameter-valued PEtab condition targetValue is a per-condition estimated initial condition: it imports as a condition perturbation whose value references a free parameter, a new condition-perturbation value kind resolved from the PSet at apply time (issue #496)

**Status: Accepted and implemented (2026-07-19).** The two PEtab benchmark-collection
problems issue #496 named — `Bertozzi_PNAS2020`, `Bruno_JExpBot2016` — now import and fit.
Each sets a condition target to a **parameter**, not a number:

```
conditionId   targetId   targetValue
u_CA          I0_        I0_CA          # I0_CA is an estimated parameter
u_NY          I0_        I0_NY
```

`I0_` is a model entity (a species / a parameter); `I0_CA` / `I0_NY` are estimated parameters
that exist only in the parameter table. Per condition, the model's `I0_` is set to a
*different* estimated parameter — a **per-condition estimated initial condition**, the common
PEtab idiom for "each experiment fits its own starting amount." The importer's condition reader
(ADR-0027/0028) accepted a number, a `<p>__REF` base pin, or a surrogate relative op, and
raised `NotImplementedError` on a parameter-valued `targetValue`.

## Why the surrogate machinery does not already cover it

ADR-0027's surrogate split handles PyBNF's *own* export of a fit-and-perturbed parameter: a fit
parameter `v1` that a condition also perturbs is renamed to `v1__REF` in the parameter table
(PEtab forbids one id in both the parameter table and a condition target), and the condition
sets `v1 = v1__REF`. On import the base pin is dropped and `v1` is recovered as an ordinary
free parameter that binds the model id `v1` — because the surrogate and the model entity share
one name (modulo the `__REF` suffix).

The `I0_ = I0_CA` shape breaks that identity in two ways: the free parameter (`I0_CA`) and the
model entity (`I0_`) have **different** ids, and **several** free parameters map to the same
model entity across conditions (`I0_CA` for `u_CA`, `I0_NY` for `u_NY`). There is no rename that
collapses this to a model-bound free parameter; it is a genuine per-condition *wiring* of one
model entity to a condition-specific free parameter. So it needs a runtime capability PyBNF did
not have: **a condition perturbation whose value is a free parameter, resolved at apply time.**

## The one new capability: a parameter-reference condition perturbation value

A PyBNF `condition:` is a `MutationSet` of `var <op> val` perturbations (ADR-0028); `val` was a
float (a parameter set/scale) or, for a species `setConcentration`, an unevaluated string
(ADR-0062). This ADR adds a third value kind to `pybnf.pset.Mutation`: **a free-parameter
reference** (`is_param_ref`), where `value` is a free-parameter *id* whose current fit value
supplies the amount.

`Mutation.amount(param_values)` is the resolution seam: a numeric perturbation returns its
`value`; a reference resolves `value` against `param_values`, a `{param_id: value}` snapshot of
the model's current PSet, and raises a clear `PybnfError` when the id is not a free parameter.
`Mutation.mutate(base, param_values)` composes the resolved amount with the operator exactly as
before (`=` returns it, `*`/`/`/`+`/`-` combine it with the base) — so a relative op on a
reference works too, though the affected problems use only `=`. **The reference is to the fit
vector, not a model quantity, so the same lookup serves every backend.**

**Every apply site threads the PSet snapshot, mechanically.** The net (BNGL-emit)
`_get_mutant_model` snapshots the PSet *before* the mutation loop (a reference resolves against
the fit vector, not an intermediate mutated value) and passes it to `mutate`; the RoadRunner and
bngsim SBML/Antimony apply paths (`_apply_mutant`, `_apply_mutant_engine`) build the same
`{id: value}` map (`_param_set_values`) and pass it to `amount`. A free parameter that binds no
model entity (`I0_CA`) is simply skipped by the model's own `_apply_param_set` (it is not a model
id) and read only here — exactly the shape the value-reference needs.

## The importer maps it, the exporter emits it back

**Import.** `conditions.conditions_from_rows` / `_perturbation_from_row` gain the estimated set
(`free_names`) and the fixed-parameter map (`fixed_params`) the importer already builds. A
non-numeric `targetValue` that names exactly one parameter binds the target to it: an **estimated**
id becomes a parameter reference (`val` stays a *string* naming the free parameter), a **fixed**
one inlines its nominal value (an absolute set). A **multi-symbol** expression (`a*b + c`) is
still the deferred sympy-layer boundary — the narrowed `NotImplementedError`. The imported conf
carries the reference verbatim (`condition: cA, perturbations: I0_ = I0_CA`), so the round trip
is source-of-truth-in-the-conf.

**Config load.** The grammar's condition value now accepts a bare identifier as well as a number
(identifier-first, so an alpha-leading value is a reference and a digit/sign/`inf`-leading value
is a number — the token shapes never overlap). `config._build_condition_mutation` routes a
non-numeric parameter value to a reference `Mutation`; the referenced free parameters are recorded
(`_condition_free_params`) and admitted as **nuisances** by the bind-by-id typo check
(`_check_variable_correspondence_modern`) — like an observableParameters scale or a free sigma,
a condition-referenced parameter binds no model entity of its own and must not read as a typo.

**Export (symmetric, PEtab-legal).** `conditions.mutation_target_value` emits a parameter-reference
value as the bare id verbatim; `export._read_conditions` passes a non-numeric parameter value
through as a string; `_referenced_nuisance_symbols` admits the referenced id as a model-unbound
estimated parameter. This is PEtab-legal **without** the surrogate split: the referenced id is a
parameter-table entry and the fixed target is not, so no id is in both tables. A PyBNF job authored
as `condition: cA, perturbations: s = s_A` exports and re-imports byte-for-byte.

## Scope

**In:** a bare parameter-reference condition value (a per-condition estimated initial condition)
— the `Mutation` `is_param_ref` value kind + the PSet-snapshot resolution seam on all four apply
paths (net, RoadRunner, bngsim SBML, bngsim Antimony); the grammar's identifier value; the config
build + nuisance registration; the importer's free/fixed resolution; the symmetric exporter.
New-era only (PEtab interop is new-era, ADR-0034). Oracled by an SBML runtime test (`K3 = k3_ref`
with `k3_ref` bound to no model entity reproduces the `K3 *= 4` literal-mutant numeric expectation
exactly), a config-load test, unit tests on `Mutation.amount`/`mutate` and `conditions_from_rows`,
and an export → import → re-export byte round trip.

**Out (boundary raised in code, each pointing here):**

- A **multi-symbol** condition `targetValue` expression (`a*b + c`) — the deferred condition-formula
  sympy layer (`conditions._perturbation_from_row`, unchanged #407 boundary).
- A parameter reference **inline in a pre-equilibration phase** — `config._preequilibration_perturbations`
  raises; the inline `setParameter` emission has no PSet in hand at that seam. Use the reference in an
  ordinary measurement `condition:`.
- The **gradient / EFIM** column for a parameter-reference condition — `gradient.routing.route_experiment`
  raises `GradientNotSupported` rather than emit a silent zero column: the referenced free parameter
  reaches the trajectory through a condition target of a *different* id, a coupling bind-by-id routing
  does not model. A gradient-free optimizer / sampler (the importer's `de` default) is unaffected; the
  chain-rule column is a later #385 sub-layer, exactly as the estimated-sigma sources' are.
- A **relative op** on a parameter reference **on export** — the runtime evaluates it, but it is a
  multi-symbol PEtab expression with no plain `targetValue`, so `mutation_target_value` defers it.

## Boundaries (in code, each pointing here)

- `pybnf/pset.py` — `Mutation(is_param_ref=…)`, `Mutation.amount(param_values)` (the resolution +
  clear missing-reference error), `Mutation.mutate(base, param_values)`; the PSet snapshot in
  `_get_mutant_model` and the RoadRunner `_apply_mutant`.
- `pybnf/bngsim_sbml_model.py` — `_param_set_values`; `_apply_mutant` / `_apply_mutant_engine` resolve
  via `amount` (inherited by the Antimony backend).
- `pybnf/parse.py` — the condition value alternative (`cond_param_val`: identifier | number).
- `pybnf/config.py` — `_build_condition_mutation` routes a non-numeric value to a reference;
  `_load_conditions` collects `_condition_free_params`; `_check_variable_correspondence_modern` admits
  them; `_preequilibration_perturbations` refuses a reference.
- `pybnf/gradient/routing.py` — `route_experiment` gates a parameter-reference condition
  (`GradientNotSupported`).
- `pybnf/petab/conditions.py` — `conditions_from_rows` / `_perturbation_from_row` (free/fixed
  resolution); `mutation_target_value` (emit the bare id).
- `pybnf/petab/import_.py` — thread `free_names` / `fixed_params`; `_render_perturbation` emits a
  string value verbatim.
- `pybnf/petab/export.py` — `_read_conditions` passes a reference string through;
  `_referenced_nuisance_symbols` admits condition-referenced parameters.

## Consequences

- A parameter-reference condition value is a **permanent native capability**, not a PEtab-import
  artifact (ADR-0004): any new-era job may write `condition: c, perturbations: init_x = x0_c` to fit a
  per-condition initial condition or per-condition parameter value, the honest "each dataset has its
  own baseline" the systems-biology fits in the benchmark collection routinely need.
- The `Mutation` value is now one of {number, species string, parameter reference}; the resolution
  seam (`amount(param_values)`) gives a future value kind (e.g. an expression over the fit vector) a
  home without another apply-site change.
- See ADR-0027 (the surrogate-base split this is the decoupled-name sibling of), 0028 (the new-era
  `condition:` surface), 0034 (bind-by-id + the nuisance typo check), 0052/0062 (the pre-equilibration
  / species-perturbation value kinds), and #385 (the gradient gate). Closes #496.
