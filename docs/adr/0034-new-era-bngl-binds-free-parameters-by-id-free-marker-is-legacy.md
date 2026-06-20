# New-era BNGL binds free parameters by id (like SBML); `__FREE` is a legacy-edition marker (issue #423)

**Status: Accepted and implemented (decision 2026-06-19; implemented 2026-06-20 as #423
Chunk 6).** The model-namespace exposure, the edition-gated bind-by-id binding (both
backends), and the typo check have landed; see the Implementation note below. The #407
exporter/importer simplification this unlocks is a tracked follow-on, not yet done. In a
new-era (`edition >= 2`) job, a BNGL model's
fit/sampled parameters bind to the config's free parameters **by id**, exactly as the SBML /
roadrunner / bngsim backends already do — no `__FREE` markers in the model file. Legacy
edition keeps the `__FREE` mechanism unchanged for backward compatibility. PEtab v2 interop
(#407) is a new-era-only feature, so legacy is deliberately never made PEtab-compatible.

## Context

Legacy BNGL fitting identifies the fit knobs with a `<param> <param>__FREE` marker that does
two jobs (`pybnf/pset.py`): **(a) discovery** — a text scan (`re.findall(r'…__FREE', line)`,
`pset.py:524`) collects the `*__FREE` tokens as `param_names`, and hard-errors if a `.bngl`
has none (`pset.py:610`); **(b) value injection** — for the file+subprocess `BNGLModel`,
`model_text` (`pset.py:714`) prepends `k1__FREE <value>` into the parameters block of the
`.bngl` it writes for BNG2.pl, and for the in-process bngsim backends, `set_param('k1__FREE',
v)` sets the marker-named parameter (`net_model.py:204`) / seeds it into the in-process BNGL
expression evaluator (`bngsim_model/expressions.py:187`).

Both jobs are artifacts of the **file + subprocess** BNG era. The injection half is a
symptom of writing a `.bngl` for BNG2.pl; the in-process engine replaced the file write with
`set_param`, but kept `__FREE` as the *name it sets*. Meanwhile, every other model backend
binds free parameters **by id**: a config free parameter named `k1` sets the model entity
`k1` via a live `set_param('k1', v)` (`bngsim_sbml_model.py:416`; `config.py:1467` unions each
model's `param_names` and matches by name), with the model file untouched. **PEtab binds by
id too** — a `parameterId` is the model entity's name. So the BNGL marker is the lone
free-parameter idiom in PyBNF that is *not* bind-by-id, and it is the thing the #407
exporter/importer must translate around (`clean_model_for_petab` strips the marker;
`_reinstrument_free_parameters` re-adds it; `free_to_param` maps `k1__FREE`→`k1`).

## Decision

**Under `edition >= 2`, a BNGL free parameter binds to the model by id: a config free
parameter whose name matches a model parameter sets that parameter (`set_param`), the same
contract the SBML backend uses. `__FREE` markers are not required, and the model file is
carried verbatim. Under legacy edition, `__FREE` (discovery + injection) is unchanged.** This
is edition-gated exactly like the objective surface (ADR-0031) and the data surface
(ADR-0028): new behavior only in the new era; legacy confs and the existing `__FREE` model
corpus keep working byte-for-byte.

`__FREE` is **removed**, not merely made optional, in the new era — the one case that looked
like it needed a marker does not:

- **A fit symbol embedded in a parameter expression** (`kaf = kaf__FREE/(NA*Vo)` — fit a
  *factor* of `kaf`, not `kaf`) is served *better* by naming the knob as a **real parameter**
  and binding it by id:
  ```
  begin parameters
    kaf_scale  1.0
    kaf        kaf_scale/(NA*Vo)
  end parameters
  ```
  Conf: `uniform_var = kaf_scale 0 10`; the loader does `set_param('kaf_scale', v)`; BNG
  re-evaluates `kaf`. Identical result, no conjured symbol, and it is exactly how SBML/PEtab
  expresses the same thing (a named parameter used in a rule). The only thing the marker
  bought was *not having to write the `kaf_scale 1.0` line* — an authoring shortcut, traded
  for the explicitness the new era is built on.
- **A nuisance fit parameter the model never sees** (a free σ — `chi_sq_dynamic`'s
  `sigma__FREE`, or the #407 importer's free-sigma `noise_model = … sigma = fit <id>`) is a
  *config* free parameter referenced by the objective/noise surface and bound to **no** model
  id. That already works — an unmatched free parameter is a warning, not an error
  (`net_model.py:208`), the path the importer's free-sigma case relies on.

So bind-by-id covers every case; the marker adds a token, not a capability.

## The safety-net trade

The model-file marker gave a wiring check bind-by-id loses: it caught "you wrote `v4__FREE`
in the model but forgot it in the conf." The new-era loader must replace that direction with
a typo check on the config free parameters:

- a free parameter that **matches a model parameter id** → bound (the common case);
- a free parameter **referenced by the objective / noise_model / constraint surface** but
  matching no model id → an intended nuisance parameter (e.g. a free σ) → fine;
- a free parameter that matches **no** model id **and** is referenced by **no** such surface
  → almost certainly a typo → **error** (with the model's parameter names listed).

This is a loader obligation, not a reason to keep the marker.

## Consequences

- **The BNGL model must expose its full parameter namespace.** Today a `BNGLModel`'s
  `param_names` is *only* the `__FREE` tokens (`pset.py:524,617`); bind-by-id and the typo
  check need the complete `begin parameters` name set, mirroring the SBML model's
  species ∪ globals (`bngsim_sbml_model.py:_extract_sbml_structure`). The in-process bngsim
  BNGL backend already parses parameter lines (`bngsim_model/expressions.py`), so the
  namespace is mostly in hand.
- **The #407 interop ceremony evaporates for new-era.** `_reinstrument_free_parameters`
  (import) and `clean_model_for_petab`'s marker-strip + `free_to_param` lookup (export) exist
  *only* to bridge PEtab's bind-by-id to the BNGL marker. Once BNGL *is* bind-by-id, the
  importer carries the model verbatim and emits `uniform_var = k1 0 10`; the `k1`↔`k1__FREE`
  round-trip disappears. (The current importer still emits `__FREE` + re-instruments because
  it targets the legacy binding mechanism; it simplifies once this lands.)
- **The #423 new-era config loader now exists.** When this ADR was written the fitter had
  no new-era front-end; by the time Chunk 6 was implemented, #423 Chunks 0–4 had landed
  (`job_type`, `model:`, `condition:`, `experiment:`/`data:`, `observable:` all parse and
  build through `config.py`), so a new-era BNGL job loads and binds end to end. The
  remaining gate is only the `param_scan`/dose-response inference parked in #426.
- **Legacy is never made PEtab-compatible.** PEtab v2 interop is a new-era feature; legacy
  edition keeps `__FREE` and the filename→suffix linkage for backward compat, and the
  exporter already refuses legacy (`_require_modern_edition`).

## Implementation note (2026-06-20, #423 Chunk 6)

- **Namespace.** `BNGLModel.model_param_names` (`pset.py`) is the full `begin parameters`
  id list, parsed by `_parse_param_block_names` from the same `model_lines` the in-process
  bngsim NF backend feeds to `_parse_bngl_param_block` (a test pins the two parsers
  together). It is *distinct from* `param_names` (still the `__FREE` tokens, untouched).
- **Binding, both backends.** The in-process backends already bound by id (the net
  backend's `set_param` over the PSet keys, the NF backend's `_evaluate_bngl_params`
  name override) — they needed nothing. The file+subprocess backend's `model_text` did
  *not*: it injected only `__FREE` tokens, so a new-era pset's bare-id values were silently
  dropped (the model ran at nominal — a confidently-wrong fit). `model_text` now overrides
  each bare parameter id *in place* in the parameters block (`_override_param_block_values`).
  Because legacy psets hold only `__FREE` tokens, which are disjoint from the parameter ids,
  this branch never fires for a legacy job and its output is byte-identical (golden +
  `test_model_class` pin it).
- **Edition gates.** `config.py` passes `suppress_free_param_error=True` for `edition >= 2`
  (the `pset.py` "no `__FREE` → error" guard becomes legacy/`check`-only, in both
  `_load_models` and `_load_t_length`), and `_check_variable_correspondence` delegates to a
  modern branch (`_check_variable_correspondence_modern`): a config free parameter matching a
  parameter id (unioned across models via `_bindable_param_ids`) binds; one referenced by
  the objective/noise surface (`self.obj.required_free_noise_params()`) is an intended
  nuisance; anything else is a typo error listing the parameter ids. There is no
  model → config direction in the new era. The legacy branch is byte-unchanged.
- **Deferred (the #407 payoff).** The importer's `_reinstrument_free_parameters` and the
  exporter's `clean_model_for_petab` marker-strip + `free_to_param` lookup still target the
  legacy mechanism; simplifying them to carry the model verbatim and emit `uniform_var = k1`
  is a tracked follow-on, deliberately out of this chunk.

## References

ADR-0028 (new-era PEtab-aligned config; this is its free-parameter chunk), ADR-0031 (edition
select-and-freeze + the edition-gating template to mirror), ADR-0025/0032 (the PEtab interop
and the `__FREE` round-trip this removes), the SBML backend (`bngsim_sbml_model.py`,
`config.py:_load_models`) as the bind-by-id precedent.
