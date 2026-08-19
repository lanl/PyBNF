# A per-species tolerance has no reading under a global key, so the new-era `model:` declaration carries `atol` / `rtol` / `species_atol` and checks a species name against the one model it can mean (issue #586)

**Status: Accepted and implemented (2026-08-19). Extends ADR-0028, ADR-0103, ADR-0105 and
ADR-0114; supersedes none.** The default path is byte-identical after this change: a job
that states nothing new gets the same `Simulator.run` call it got before, and `sbml_atol` /
`sbml_rtol` keep every meaning ADR-0114 gave them. What this adds is a place to say those
things about *one* model, and one thing that could not be said at all.

## The key, and why the grammar was never the problem

ADR-0114 shipped `sbml_atol = auto` and `sbml_atol = tracking`, and closed by naming what
it had not shipped:

> **`sbml_atol` still does not take a hand-written vector**, which is #557's second ask.
> […] The first clause is left open as **#586**: `sbml_atol` is one global key applying to
> every SBML/Antimony model in a fit, so a positional vector has no ordering a conf author
> can see and a species-keyed one has no unambiguous reading across models that do not
> share species. That needs a per-model tolerance record and its own design.

That is the whole of it, and it is worth restating because the obvious reading is wrong.
The blocker was never that `sbml_atol = 1e-3 1e-6 1e-4` is hard to parse — it is trivial to
parse. It is that neither spelling **means** anything definite once the key is global, and
PyBNF fits run several models routinely (ADR-0041 multi-model interop; a PEtab problem with
more than one `modelId`):

- **A positional vector** is ordered against `bngsim.Model.species_names` — the *engine*
  model's ordering, which a conf author cannot see and which is not the document's
  `<listOfSpecies>` order in general. lanl/bngsim#212's `normalize_atol_vector` exists
  precisely because a mis-ordered vector assigns one species' tolerance to another and
  nothing downstream can tell: every entry is a plausible tolerance. Under two models with
  different species counts, one line cannot even be length-checked against both.
- **A species-keyed map** is safe to order and still has no reading across models. Is a
  name absent from model B an error, or a per-model override B does not use? Do two models
  sharing a species name share its tolerance? A name that collides across models with
  different units silently means two different things.

Both objections are about the **key**. Give the statement one model to be about and both
disappear — the name has exactly one species list to be checked against, the ordering is
that model's, and two models cannot collide because neither can see the other's line.

ADR-0028 already built the place for it. A new-era `model:` declaration names one model
file, is edition-gated, and is the sibling of `condition:` / `experiment:` / `parameter:`,
every one of which already carries labeled sub-fields. #586 proposed exactly this, and this
ADR is that proposal built.

## The decision

**The new-era `model:` declaration carries this model's own CVODE tolerances**, in any
order after the file, and every field is optional:

```
model: weber.xml, atol: auto, species_atol: PKD 1e-3, CERT 1e-2, rtol: 1e-9
```

- **`atol:`** takes exactly what `sbml_atol` takes — a number, `auto`, or
  `tracking [decades]` — and means exactly what that key means, for this model only.
- **`rtol:`** is the per-model `sbml_rtol`. A scalar, and only ever a scalar: CVODE takes
  one relative tolerance (`double rtol`), and bngsim has no `rtol_vec` to route a
  per-species one to. #586 names a per-model `rtol` as something the record "also gives a
  home for"; this is that home, and it is the whole of what that half can be.
- **`species_atol:`** is the hand-written vector, written as `<species> <number>` pairs.

`sbml_atol` and `sbml_rtol` remain the fit-wide defaults and are unchanged in every
respect. A field here overrides the matching global key **for the one model it names**, and
says nothing about the others. That is the property the global key could not have, and it
is what makes a species name mean something.

### 1. The record is refused where it could only be decoration

A `model:` line carrying any of these fields must declare **exactly one** model, because
`model: a.xml, b.xml, atol: 1e-4` cannot say which model the `1e-4` is about — and stopping
one number from meaning several things at once is the entire point. Refused in `ploop`
rather than in the grammar, so the message can say to give each model its own line.

The model must also be one with a CVODE tolerance to state: an `.xml`/`.ant` model on
`sbml_backend = bngsim`, integrated by CVODE. Three cases are pointed errors at config load
because in each the record could only be decoration — a BNGL model takes `atol`/`rtol` from
its own `begin actions` block, the RoadRunner backend has its own integrator settings, and
`sbml_integrator = gillespie` makes `_resolve_method` return `'ssa'` for every action of
every model, where a run carries no tolerances at all.

This is stricter than the global key, which is guarded only on the backend and is a silent
no-op today on a BNGL-only job with `sbml_backend = bngsim`, or on any job under
`gillespie`. That asymmetry is deliberate: a key applying to "every SBML model, of which
there are none" is a vacuous statement, while a record naming `egfr.bngl` is a mistake about
that file.

### 2. `species_atol` is an override layer, and a stated number is used verbatim

The map does not replace the derivation. Every species it names takes the number stated;
every species it does not keeps whatever it would otherwise have had — the ADR-0105 vector
when there is one, and the model-wide scalar broadcast when there is not (a model whose
species share one scale, an ordering the derivation cannot take, or a pinned `sbml_atol`).
So `atol: 1e-6, species_atol: PKD 1e-3` reads as it looks — everything at `1e-6` except
`PKD` — rather than as a contradiction one of the two has to lose.

**Neither clamp binds a stated entry.** Not ADR-0103's `1e-8` ceiling, which ADR-0114
established is a no-regression rule about the *derivation* rather than a property of the
model; and not ADR-0105's model-scalar floor, which bounds how far a rule reading *initial
values* may reach on its own. A number a person wrote is neither of those. It is the same
kind of statement a pinned `sbml_atol` number is, and it wins for the same reason: the
derivation's job is to answer when nobody has, and somebody has.

The one bound kept is CVODE's own — finite and **strictly positive**. bngsim accepts `0` in
a plain per-species vector and refuses it outright as a `TrackingAtol` ceiling, so accepting
it here would make `species_atol: X 0` parse, integrate, and then fail the moment the same
model added `atol: tracking`. One rule that holds under every setting is worth more than the
one value it declines.

Nor does the **"elementwise the scalar, so send the scalar"** off-ramp apply to a stated
map. That off-ramp is ADR-0105's, and it is correct precisely because nobody asked for a
vector: a model with no over-tightening to give back keeps its `CVodeSStolerances` call byte
for byte (19 of 23 subset-I slugs) rather than paying lanl/bngsim#196's ~1 ulp
`cvEwtSetSS`/`cvEwtSetSV` difference for a constant vector. Here somebody did ask, so the
vector travels and pays that ulp for having been stated.

### 3. The names are bngsim's, and that is checked rather than assumed

A `species_atol` is written in the names bngsim **integrates the model under**
(`Model.species_names`), not the SBML document's `<listOfSpecies>` ids. The two lists agree
on almost every model and diverge in two ways that matter:

- **bngsim renames an id that collides with an Antimony reserved word.** `NULL` loads as
  `_ant_NULL`. This is not hypothetical: it is why `Smith_BMCSystBiol2013`'s derived vector
  declines today, warning and falling back to the scalar (`_per_species_atol`).
- **A parameter or compartment driven by a rate rule or an event assignment becomes a
  state**, appended after every declared species. It has no id in `<listOfSpecies>` and no
  declared initial value, so nothing read off the document can give it a tolerance at all.

Engine names are the only list that can be both checked and applied, because they are the
ordering a per-species vector *is*. Validating against document ids would accept a name that
then silently did nothing — the failure this whole line of work exists to stop making. So
an unknown name is a `ModelError` naming the species the model does have, and naming the
`_ant_` rename outright when it recognizes the case, since that is the one a conf author
cannot guess from their own file.

That check runs **at config load**, which is what #586 asks for, and it costs nothing: the
model constructor already loads a bngsim engine model to validate the file and throws it
away. It now keeps that load's species names.

### 4. Refused, not degraded

A `species_atol` on a bngsim without lanl/bngsim#196 is a `ModelError` at construction and a
`PybnfError` at config load, not a silent fallback to the scalar. The *derived* vector may
fall back silently — nobody asked for it, and the scalar is a correct tolerance for that
model, merely a less discriminating one. A hand-written map is somebody asking, and a stated
tolerance that did not apply is a plausible trajectory with a wrong conclusion attached.
This is exactly the line #557 drew for `sbml_atol = tracking`, applied to the one other
setting that is a request rather than a permission.

## What is deliberately not changed

- **`sbml_atol` and `sbml_rtol` keep every meaning and stay the fit-wide default.** They
  are not deprecated under edition 2 and gain no new spelling. A one-model job that wants
  one number still writes one number in the place it always did.
- **The scalar derivation, the vector derivation, and both clamps are untouched.** A model
  that states nothing reaches the same `Simulator.run` call, argument for argument.
- **The steady-state cutoff stays ADR-0105's pairing.** bngsim resolves a vector `atol` to
  its own `1e-8` for every scalar-shaped consumer, and `steady_state_tol` is one, so a
  stated vector travels with the model's derived (or pinned) scalar exactly as a derived one
  does. Forgetting this would silently revert ADR-0103's steady-state fix and make every
  `time = inf` measurement and every pre-equilibration phase return the initial state and
  call it equilibrium.
- **No sparse vector reaches bngsim.** There is no partial form to send: PyBNF densifies the
  map against the engine species list and `normalize_atol_vector` takes the length contract
  once, at setup.
- **The vector remains a constant of the model, never of the fit point.** A number read out
  of a conf is trivially that, and bngsim's live-state `AUTO` remains ruled out for the
  reason ADR-0105 gives.

## Verification

### What the surface reaches that nothing else does, measured on the corpus

Over the 23 subset-I slugs with a loadable `.xml`, the engine and document species-name sets
disagree on **one**: `Smith_BMCSystBiol2013`, whose `NULL`/`null` load as
`_ant_NULL`/`_ant_null`. On that model, today and after this change:

| `Smith_BMCSystBiol2013` | what CVODE is called with |
|---|---|
| unset (the derivation) | **scalar** — the vector declines, unorderable past the rename |
| `sbml_atol = auto` | **scalar** — same decline; `auto` lifts a clamp, not an ordering |
| `species_atol: NULL 1e-3` | refused, naming `_ant_NULL` |
| `species_atol: _ant_NULL 1e-3, _ant_null 1e-3` | **vector[133]**, entries in their own slots, cutoff stated |

So on the one corpus model where the derivation cannot order a vector at all, a hand-written
map is the only route to `CVodeSVtolerances` that exists. **Zero of the 23 have a promoted
rate-rule state**, so the second capability §3 describes is real and the corpus has no case
for it — recorded rather than claimed.

### What a hand-written vector costs on the slug #557 was filed from

`Weber_BMC2015`, seven species at `1.24e+02 .. 4.21e+07`, median `4.665e+05`; 100 time
units, 101 output points, `rtol = 1e-8`, one fresh engine model per arm (a second `run()` on
one Simulator continues from where the first stopped, ADR-0104):

| `atol` | integrator steps | vs unset | max rel. state difference at `t_end` |
|---|---:|---:|---:|
| unset (the clamped derivation, `1e-08`) | 210 | 1.00x | — |
| `auto` (ADR-0114's vector) | 184 | 0.88x | 3.3e-08 |
| `species_atol` = `rtol * y_i`, by hand | 192 | 0.91x | 2.9e-08 |

The third arm is the interesting one. `rtol * y_i` with **no model-median floor** is
bngsim's own `derive_atol` default, and it is exactly the rule ADR-0105 measured and
declined to apply on its own — over 100 points of `Brannmark_JBC2010`'s fit box it killed 91
simulations out of 100 against the scalar's 39. This surface makes it reachable, per model,
by a person who has decided to reach for it. And on Weber it is **worse than `auto`** — 192
steps against 184, because `PKDDAGa` at 124 drops from the median's `4.67e-03` to
`1.24e-06`, four decades tighter, for a species that is not the problem.

That is the honest shape of this change, and it is worth stating plainly: **the surface is
not a faster tolerance, it is a place to put a decision.** The derivation remains the better
default on every slug measured; what it cannot do is be overridden for one species of one
model, and now it can be.

Neither arm is an accuracy trade: both agree with the clamped run's final state to ~3e-08,
i.e. to the relative tolerance that governs it.

### One hole closed on the way past

`sbml_rtol` was checked for sign and not for finiteness (`if tol <= 0.`), and the conf
grammar's number token also matches `inf` (ADR-0047's open truncation side). So
`sbml_rtol = inf` parsed, cleared the check, and was installed as the relative tolerance —
which turns relative error control off rather than erroring. `sbml_atol` never had it,
because `parse_atol_setting` has always demanded finiteness. Fixed here rather than filed,
because the per-model `rtol:` field would otherwise have inherited exactly the same check;
`test_config_refuses_an_rtol_that_is_not_a_relative_tolerance` is the assertion.

### The default path is untouched

The fast tier is green at **4170 passed, 10 skipped** with the change in, up from 4086
before it (the difference is this change's own tests). `test_a_stated_map_travels_even_when
_it_agrees_with_the_scalar` asserts the unstated half of that directly: a model that states
nothing still gets `{'rtol': 1e-08, 'atol': 1e-08}` and no `steady_state_tol`, which is the
scalar call ADR-0105 promised 19 of 23 slugs would keep byte for byte.

### The mechanism, in fixtures

Parse and record: `test_the_model_line_states_one_models_own_tolerances`,
`test_the_tolerance_fields_are_order_independent`,
`test_a_declaration_that_states_no_tolerance_is_unchanged` (which is also the assertion that
the legacy `model = f : d.exp` line still backtracks past the new fields),
`test_a_multi_model_declaration_cannot_state_a_tolerance`,
`test_a_model_is_given_its_tolerances_once`, `test_a_species_is_given_its_tolerance_once`,
`test_the_model_format_hint_names_the_tolerance_fields`.

Shape: `test_parse_species_atol_setting_reads_a_map`,
`test_parse_species_atol_setting_refuses_what_is_not_a_tolerance` (which is where the
strictly-positive decision is recorded).

Composition: `test_the_species_a_map_does_not_name_keep_what_they_would_have_had`,
`test_a_stated_entry_is_bound_by_neither_clamp`,
`test_a_pinned_scalar_becomes_the_base_the_map_overrides`,
`test_a_stated_map_becomes_the_tracking_ceiling`,
`test_a_stated_map_keeps_the_steady_state_cutoff_the_models_own_scalar`,
`test_a_stated_map_can_say_exactly_what_the_derivation_says` — the last being the property
that says this is one mechanism rather than two: what the derivation computes is inside what
a person can write.

Ordering and reach: `test_a_stated_species_tolerance_reaches_the_run_call_by_name`,
`test_a_stated_map_reaches_the_bngsim_run_call`,
`test_a_stated_map_survives_the_rename_that_makes_the_derivation_decline`,
`test_a_stated_map_is_a_constant_of_the_model_not_of_the_fit_point`,
`test_the_model_still_integrates_to_its_closed_form_under_a_stated_vector` (an oracle, at a
tolerance two decades above what any derivation produces).

Names and refusals: `test_the_engine_species_names_are_captured_at_construction`,
`test_an_unknown_species_is_refused_and_the_message_names_the_ones_it_has`,
`test_a_renamed_species_is_named_in_the_refusal`,
`test_a_stated_map_is_refused_rather_than_dropped_without_the_backend`, plus the Antimony
trio in `tests/test_bngsim_antimony_bridge.py` —
`test_an_antimony_model_takes_its_own_per_species_tolerance`,
`test_an_antimony_reserved_word_species_is_named_by_the_name_bngsim_gives_it`, and
`test_a_promoted_state_is_nameable_even_though_the_document_never_declares_it`.

Config surface: `test_a_per_model_record_is_resolved_field_by_field`,
`test_a_tolerance_for_an_undeclared_model_is_refused`,
`test_a_model_with_no_cvode_tolerance_to_state_is_refused`,
`test_a_per_model_tolerance_needs_the_bngsim_backend`,
`test_config_refuses_a_per_model_setting_that_is_not_a_tolerance`, the two capability
refusals, `test_the_per_model_tolerances_require_edition_2`, and the end-to-end
`test_a_conf_states_one_models_tolerance_and_leaves_the_other_alone` — two models, one
statement, no collision, which is #586 reduced to one assertion.

## Alternatives considered

**`atol:` carrying the pairs itself**, which is #586's own sketch
(`model: weber.xml, atol: PKD 1e-3, CERT 1e-2`). The three forms are lexically disjoint — a
lone number, a keyword, `<name> <number>` pairs — so it parses. It was split in two anyway,
because `atol:` also has to be the per-model form of `sbml_atol` to be worth having in a
multi-model fit, and one key cannot then carry both a mode and a map: `atol: auto,
species_atol: PKD 1e-3` is a thing people will want to write, and under the merged spelling
there is nowhere to write it. Two fields that each answer one question compose; one field
answering two does not.

**A `species_atol` keyed on document ids**, so a conf author can write what they read in
their own model file. Rejected in §3: it accepts a name the engine does not carry, and the
result is a stated tolerance that silently does nothing. The refusal message carries the
translation instead, which costs one error and no ambiguity.

**Filling unnamed species from the scalar rather than from the derivation.** Simpler to
describe, and wrong in the direction that matters: it would make `species_atol: PKD 1e-3`
silently *undo* ADR-0105's vector for every other species of that model. An override layer
means the unnamed species are exactly where they were.

**Leaving `sbml_atol` as the only surface and adding a positional vector to it.** This is
what ADR-0105 declined ("a per-species vector in a `.conf` would need a new species-keyed
grammar, and the case for it is hypothetical") and what #586 re-declined. Nothing about the
key changed; only the place did.

### Scope

**In:** `pybnf/parse.py` (the `model:` sub-field grammar, the record helpers, the format
hint), `pybnf/config.py` (`_resolve_model_tolerances`, the edition gate, the two
constructor call sites), `pybnf/bngsim_sbml_model.py` (`parse_species_atol_setting`,
`_config_species_atol`, `_engine_species_names`, `_check_stated_species_atol`,
`_apply_stated_per_species_atol`, `_per_species_atol`),
`pybnf/bngsim_antimony_model.py` (the kwarg), `docs/config_keys.rst`,
`tests/test_sbml_solver_tolerances.py`, `tests/test_bngsim_antimony_bridge.py`.

One grammar detail is worth recording because it is not obvious and was caught by review
rather than by design. `model_file`'s regex is unanchored and lazy, so on a line where a
comma can now be followed by something that is not a file it will run straight through the
field — and through a trailing comment — to the next extension it can find:
`model: decay.xml, atol: 1e-10  # tighter for decay.xml` parsed as *two* models, the second
named `atol: 1e-10  # tighter for decay.xml`. The declaration's file token therefore sits
behind a negative lookahead for a field label followed by its colon, which ends the file
list exactly where the fields begin. Nothing else about the token is narrowed — `#` stays
legal inside a filename, because the legacy `model = … : …` form still accepts it and two
spellings of one declaration disagreeing about which files exist would be worse than the
corner it would close. `test_a_tolerance_field_is_never_read_as_a_model_filename` and
`test_a_model_file_written_after_a_tolerance_field_is_a_parse_error` hold both halves.

**Out (unchanged):** `_derive_atol`, `_derive_atol_vector`, `_derive_per_species_atol`,
`_effective_tolerances`, `_run_tolerance_kwargs`'s pairing rules, `pybnf/_bngsim_caps.py`,
`pybnf/config_schema.py` (the record is a structural tuple key, so no schema field and no
golden-config movement), the BNGL/`.net` tolerance route, and bngsim itself — nothing in
lanl/bngsim needed changing for this.

Relevant ADRs: **0028** (the `model:` declaration this rides), **0103** (the scalar
derivation), **0105** (the per-species vector and both clamps), **0114** (`auto` /
`tracking`, and the paragraph that opened this issue). Relevant issues: lanl/bngsim#196
(`CVodeSVtolerances`), lanl/bngsim#212 (`normalize_atol_vector`), lanl/bngsim#213
(`CVodeWFtolerances`). Closes issue **#586**.
