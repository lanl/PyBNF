# M2.1 Stage (b) preserves the full-union effective config; β-ladder is a co-located hook

In M2.1 Stage (b) each fit type's configuration knowledge (defaults, types,
validation, preprocessing) moves into a Pydantic model **co-located with its
algorithm class** and reached through the registry (`FitTypeEntry.schema`;
ADR-0002, ADR-0005). But the effective `Configuration.config` dict stays the
**full union of every method's defaults** for every fit type, exactly as before:
`_build_config` reassembles that union from a default baseline (collected from
all registered schemas' field defaults) overlaid by the run-level `GlobalConfig`
and the selected method's validated, preprocessed values. We deliberately do
**not** narrow each fit's dict to only its own keys yet.

Why preserve the union: it keeps the golden-config net (`test_config_golden.py`)
byte-identical through the whole migration — one method per commit, golden silent
— and it avoids breaking legitimate cross-method reaches. `_refine_best_fit`
runs `SimplexAlgorithm` on a *non*-simplex config and reads `simplex_*`, so a
`de` fit genuinely needs simplex keys present. Narrowing the dict per method
(and giving cross-method dependencies like refine→simplex an explicit
declaration) is real design work that is far safer after Stage (c)'s typed
access; it is tracked as a follow-up issue and intended, not forgotten.

## Considered Options

- **Shrink each fit's effective config to its method's keys now.** Rejected for
  Stage (b): it is an intended golden regen across every entry *and* forces an
  audit of every cross-method dict reach (refine→simplex proves at least one
  exists) — exactly the risky reach audit the plan defers to Stage (c). Cleaner
  end-state, wrong step. This is the "B" we will come back to.
- **β-ladder (`postprocess_mcmc_keys`) as a Pydantic `@model_validator`.**
  Rejected: its outputs (`beta_list`, `reps_per_beta`) must exist only for MCMC
  fits, and it rewrites `population_size`. Modeling those as validated fields
  leaks them into non-MCMC fits' defaults and pulls `population_size` into the
  model. See Consequences for the chosen form.

## Consequences

- A future reader will see `_build_config` rebuild a ~95-key dict though the
  per-method models "own" their keys, and may be tempted to strip it. **Don't** —
  it breaks refine and lights up the golden net. Narrow only when the follow-up
  issue's groundwork (Stage c typed access + declared cross-method deps) is in
  place.
- The β-ladder becomes a co-located `postprocess(conf, fit_type)` classmethod on
  the shared MCMC-family model, called uniformly by `_build_config` (every method
  schema inherits a no-op `postprocess` from a common base). It keeps the riskiest
  preprocessing byte-identical to today (a plain dict mutation) while still
  co-locating it with the family — the deepening goal without the pydantic
  field-dump friction.
- Required keys (`population_size` / `max_iterations`) and a few cross-cutting
  checks stay in `config.py` for Stage (b): the user-facing required-key error,
  the `step_size`-set → `adaptive_step_size=False` coupling (it spans two models —
  really a B/Stage-c item), the objfunc cross-config requirements (`neg_bin` →
  `neg_bin_r`, `neg_bin_dynamic` → `r__FREE`, `chi_sq_dynamic` → `sigma__FREE`),
  and the Simplex `var`/`logvar`-only free-parameter rule. None is single-method
  knowledge.
- `config.py` gains a side-effect `from . import algorithms` (twin of the existing
  `from . import objective`) so `FIT_TYPE_REGISTRY` is populated before any config
  is built. Verified acyclic: nothing in `algorithms`/`objective` imports `config`.
- CFG-CHECK-1 is fixed in this stage (guard the model-checking key scan with
  `isinstance(k, str)` so `fit_type = check` + a `*_var` key no longer crashes);
  this is the one *intended* golden change (`matrix/check_with_var` + a regression
  test), regenerated knowingly.
