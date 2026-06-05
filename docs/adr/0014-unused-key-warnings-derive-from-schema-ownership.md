# Unused-key warnings derive from schema ownership; the broad "non-structural extra" policy

ADR-0013 (#399) narrowed each fit_type's effective config to its own schema, leaving
**three** places that still hand-maintained the per-fit_type key-ownership knowledge
the schemas now own (`schema.owned_keys()`):

1. `check_unused_keys` — a hand-coded `alg_specific` dict, the bayesian-collapse
   (`pt/sa/dream/p_dream/am → mh`), and the refine→simplex exemption.
2. `check_unused_keys_model_checking` — a `used` whitelist for `fit_type = check`.
3. `MCMCFamilyConfig.postprocess` — warn-only branches for `exchange_every`/
   `reps_per_beta` (non-pt), `cooling`/`beta_max` (sa-only), and
   `crossover_number`/`zeta`/`lambda`/`gamma_prob` (mh/pt/am).

These drift against `owned_keys()`: the warnings are print/log side-effects, **not**
in the golden-config snapshot (`test_config_golden.py`), so #399's oracle never
protected them. This is the deliberately-deferred **issue #401** (ADR-0013 §"three
follow-ups"). We settled the shape (Bill chose the broad policy, 2026-06-05):

- **One schema-derived derivation replaces all three encodings.** A key is *unused*
  precisely when it is a **non-structural extra** — owned by neither `GlobalConfig`,
  the selected fit_type's own method schema, nor the refine→simplex group, and not a
  structural model-path / free-parameter-tuple / `models` / `exp_data` / required
  key. This is the *same* ownership notion `_build_config`'s narrowing partitions a
  raw dict by, so "what narrowing keeps" and "what the warning accepts" can no longer
  drift. `Configuration._valid_config_keys(d)` is that single source:
  `SCHEMA_KEYS ∪ schema.valid_keys() ∪ (refine→simplex) ∪ STRUCTURAL_PASSTHROUGH`.

- **The broad policy: warn on unknown/typo keys too, not just known-foreign keys.**
  The old non-check path only warned about keys it recognized as belonging to another
  algorithm; a typo (`maxx_iterations`) was silently tolerated. The broad policy warns
  any non-structural extra, so it also **unifies check and non-check** into one
  derivation (acceptance criterion 2: `check` has no method schema, so its valid set
  is just global + structural). The risk — a legitimate non-schema key warning
  spuriously — is bounded by `STRUCTURAL_PASSTHROUGH` (the schema-free always-valid
  keys: `fit_type`, `models`/`exp_data`/`mutant`, `population_size`/`max_iterations`,
  `verbosity`, `postprocess`) plus the positional model-path/tuple recognition, and
  is **fenced by a corpus test** asserting no real `.conf` fixture warns about a key
  its algorithm consumes (`test_config_unused_keys.py::test_real_configs_no_spurious_warnings`).
  The conservative "warnable-universe" alternative (warn only keys owned by *some*
  other method) was rejected: it leaves `check` on a different rule and never catches
  typos.

- **Runtime-defaulted keys ride on the schema via `RUNTIME_KEYS`, not a residual
  dict.** Some keys an algorithm reads are deliberately **not** schema fields because
  they default at runtime from other state, not a literal (scatter search's
  `init_size → 10·len(variables)` and `reserve_size → max_iterations`; PSO's
  `particle_weight_final → particle_weight`; Simplex's `simplex_max_iterations →
  max_iterations` and `simplex_log_step`; the MCMC β-ladder's `beta_range`/
  `reps_per_beta`). They are valid keys, so a pure `owned_keys()` check would
  false-positive on a fit_type's *own* runtime key. Each schema declares them in a
  `ClassVar` `RUNTIME_KEYS` frozenset (merged across the MRO by `runtime_keys()`), so
  `valid_keys() = owned_keys() ∪ runtime_keys()` is the method's full key surface —
  co-located with the algorithm class, matching ADR-0006's "config knowledge travels
  with the method." This keeps the consolidation a *relocation onto the schema*, not a
  second hand-maintained map in `config.py`.

- **`postprocess` keeps the transformation, sheds the warnings.** The β-ladder still
  pins `exchange_every = inf` / `reps_per_beta = 1` for the non-pt methods (a genuine
  config transformation), but its warn-only branches are gone — `check_unused_keys`
  now emits those warnings, *more precisely* than the hand-listed branches did.

- **Intended behavior changes (oracle = warning-assertion tests).** The bayesian-
  collapse is removed, so each MCMC fit now warns about exactly the family keys it
  does **not** own (an `am` fit warns about `crossover_number`/`archive_size`; the old
  collapse silently accepted every MCMC key on every MCMC fit). `check` stops warning
  about global-schema keys (e.g. `initialization`) — the old `used` whitelist omitted
  them, wrongly flagging them. `mh` stops warning about `exchange_every`/`reps_per_beta`
  (it shares `BasicMCMCConfig`, and so the valid-key surface, with `pt`). Two latent
  bugs in the old `used` whitelist surfaced and are fixed by deriving from the schema:
  it mis-spelled `postprocess` as `postprocessing` (so a real `postprocess` key would
  have warned) and listed `model`/`time_length`, neither a live key.

- **Scope unchanged elsewhere.** `_strip_uncheckable_keys` keeps the crash-prevention
  deletion of `refine`/`bootstrap` for `check` (a transformation, not an ownership
  encoding). The refine→simplex fact stays at the single `_REFINER_SCHEMA` /
  `_refine_pulls_in` seam (ADR-0013), now consumed by `_valid_config_keys` too. The
  CFG-CHECK-1 tuple-key safety is preserved by the `isinstance(k, str)` guard in
  `_is_unused_key`.

## Considered Options

- **Conservative "warnable-universe" (warn only keys owned by some other method).**
  Rejected: does not unify `check` (which must warn on unknowns to be useful), never
  catches typos, and still needs `RUNTIME_KEYS` — so it pays the same cost for a
  narrower result.
- **A residual per-fit_type runtime-key map in `config.py`.** Rejected: it re-creates
  a second hand-maintained ownership encoding next to the schemas — exactly the drift
  this issue removes. `RUNTIME_KEYS` co-locates the knowledge with the method.
- **Add the runtime keys as `Optional = None` schema fields.** Rejected: they would
  then appear in every effective config (`init_size: None`), changing the golden
  snapshot — a #399-protected surface this issue must not touch. They default at
  runtime *by design*; modeling them as fields misrepresents that.
- **Keep the warnings in `postprocess`.** Rejected: that is one of the three encodings
  to consolidate; the per-method MCMC distinctions it drew are exactly what the
  schema's `owned_keys()` now expresses, more precisely.

## Consequences

- The unused-key warning is now a pure function of the registry's schemas; adding a
  config key to a method (field or `RUNTIME_KEYS`) automatically makes it valid for
  that fit_type and unused for others — no second edit in `config.py`.
- `test_config_golden.py` is **unchanged and still green**: the refactor touches only
  side-effects, and the `check` config still strips `refine`/`bootstrap` before
  `_build_config`, so every snapshot is byte-identical.
- The new oracle is `tests/test_config_unused_keys.py`: per-fit_type ownership cases,
  MCMC per-method precision, the refine→simplex exemption, the `check` cases, and the
  real-fixture no-spurious-warnings guard the broad policy requires.
- A user with a typo or a leftover foreign key in a config now sees a warning where
  one was previously silent — a UX improvement, and the reason the corpus guard exists.
- Issue #403 (a second refiner) still generalizes the refine→simplex constant to a
  lookup at the one seam, now feeding both `_build_config` and `_valid_config_keys`.
