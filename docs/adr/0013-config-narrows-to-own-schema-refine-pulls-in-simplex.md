# Per-fit_type config narrows to its own schema; refine pulls in the whole Simplex schema (M2.1 Stage c, the deferred "B")

ADR-0006 preserved the **full-union** effective `Configuration.config` — every
method's defaults present for every fit_type — as the low-risk "A now, B later"
choice, deferring the per-method **narrowing** ("B") behind M2.3/M2.4 so it would
not narrow to a shape priors/noise then reshaped. M2.3 (ADR-0010) and M2.4
(ADR-0011) have landed, so we now narrow. The contract is the
`tests/test_config_golden.py` net, which flips from a byte-identical guard to the
**one intended regeneration** ADR-0006 anticipated. We settled this shape (grilled
2026-06-04):

- **Narrowing drops foreign *defaults* only.** Each fit's effective config becomes
  `global defaults + its own method schema (defaults + overrides) + extras`. The
  `extras` bucket (raw string keys owned by neither global nor the selected method
  — including a user-set *foreign* key like `cognitive` on a `de` fit) **still
  passes through unchanged**, reported by `check_unused_keys` exactly as today. So
  the only thing narrowing removes is the *unset defaults* of every other method;
  no user-set value silently disappears.

- **The refine→simplex reach is the whole Simplex schema as a coherent group,
  conditionally included — not a loose grab-bag of keys.** `_refine_best_fit`
  (`pybnf.py`) runs `SimplexAlgorithm` on a *non*-simplex fit's config, reading all
  six `SimplexConfig` fields. Naive narrowing would leave an incoherent partial
  state (a lone user-set `simplex_step` with its five siblings dropped → `KeyError`
  at refine). Instead the Simplex keys appear as an **all-or-nothing group, present
  exactly when Simplex is reachable**: via the own schema when `fit_type = sim`, and
  via a **conditional full-schema overlay when `refine = 1` and `fit_type ≠ sim`**.
  The overlay is built at **build time** in `_build_config` as
  `build_effective_method(SimplexConfig, simplex_user_input)` — the six defaults
  overlaid by any simplex keys the user set — so the group is coherent, validated,
  and visible in the golden snapshot.

- **The conditional is an explicit one-off in `_build_config`, not a general
  cross-dependency mechanism.** There is exactly one cross-fit_type reach
  (refine→simplex); `bootstrap` re-runs the *same* fit_type and adds none. Per
  ADR-0009's ≥2-user bar, a registry-declared dependency framework would be
  speculative generality for a population of one. The coupling is cross-config
  knowledge, so it lives in `config.py` (ADR-0006 #5). It is routed through a
  **single seam** — one `_REFINER_SCHEMA`/`_refine_pulls_in(d)` source shared by the
  `_build_config` conditional *and* `check_unused_keys`'s existing
  `not(alg == 'sim' and refine == 1)` exemption — so narrowing does not *duplicate*
  the refine fact, and a future second refiner (Powell, issue #403) generalizes the
  overlay from a constant to a lookup without a rewrite.

- **The `step_size → adaptive_step_size = False` coupling co-locates into
  `DreamConfig.postprocess`; the `config.py` global write is deleted.**
  `adaptive_step_size` is read only by `dream`/`p_dream` and owned only by their
  schemas — which **also own `step_size`** — so the coupling is intra-dream-family,
  not cross-model; it only *looked* global because `config.py:149-150` applied it to
  every fit. The Stage-b global write left an orphan `adaptive_step_size = False` in
  any `mh`/`am`/`sa` fit that set `step_size`. `DreamConfig.postprocess` (override
  calling `super().postprocess()` then `if 'step_size' in conf_dict:
  conf_dict['adaptive_step_size'] = False`) runs only for the family that owns both
  keys. This is the exact Stage-c migration ADR-0006 #5 earmarked ("spans two models
  — really a B/Stage-c item").

- **Scope is contained; three follow-ups are tracked as issues.** The three
  hand-maintained per-fit_type ownership encodings (`check_unused_keys`,
  `check_unused_keys_model_checking`, the warn-only branches of
  `MCMCFamilyConfig.postprocess`) are **not** derived from `owned_keys()` here —
  their oracle is warning-assertion tests, not the golden, so consolidating them is
  **issue #401** (depends on this). Migrating `parse.py`'s coercion token lists into
  the schema is **issue #402** (orthogonal; its oracle is `test_parse_class.py`).
  Adding Powell as a second refiner is **issue #403** (the event that may finally
  justify the general cross-dependency mechanism).

- **Verification — four parts.** (1) The intended golden regeneration
  (`PYBNF_REGEN_GOLDEN=1`), diff reviewed: foreign defaults removed per fit_type,
  `check` narrowed to global + extras. (2) A **new** simulator-free golden entry
  `matrix/de_refine` (`de` + `refine = 1`) snapshots the coherent six-key Simplex
  group the conditional produces — otherwise the new path is unsnapshotted. (3) A
  **new** slow analytical end-to-end test (`test_optimizer_integration.py`) runs a
  non-sim fit with `refine = 1` over an `AnalyticalModel` target and asserts it
  *completes* — the run-time net the build-only golden cannot provide. (4) The full
  `pytest -m slow` tier executes every fit_type end-to-end, `KeyError`-ing on any
  missed reach; gate on fast + slow green, run sequentially.

## Considered Options

- **Keep the full union forever (ADR-0006 "A").** Rejected now that its deferral
  reason is gone (M2.3/M2.4 have reshaped the surface): the union keeps the footgun
  ADR-0006 named — a future reader strips it and breaks refine — and each fit's
  config misrepresents the keys the method actually reads.
- **Naive narrowing (drop every non-own key, including the simplex group).**
  Rejected: it produces the incoherent partial state (a lone user-set `simplex_step`)
  and breaks `refine` with a `KeyError`. Coherence demands the all-or-nothing group.
- **Also drop user-set *foreign* keys (aggressive narrowing).** Rejected: silently
  discarding a key the user wrote is worse than the existing unused-key warning, and
  it is a larger behavior change with its own UX question — out of scope.
- **A general registry cross-dependency mechanism (ADR-0006's "B" as framework).**
  Rejected for one reach (ADR-0009 ≥2-user bar). The `_REFINER_SCHEMA` seam leaves
  the door open: a second refiner (#403) is when it would be built, to its real shape.
- **Guard the `adaptive_step_size` write in `config.py` (membership check).**
  Behaviorally identical to co-locating, but rejected as a half-migration: the
  coupling is single-family knowledge, and ADR-0006 #5 promised to move it in Stage c.

## Consequences

- `test_config_golden.py` flips from a byte-identical guard to "every snapshot
  changes — review the intended diff," then re-freezes on the narrowed configs. The
  net is briefly at its weakest exactly on this broad change, so the diff review and
  the two new fixtures (build-time `matrix/de_refine`, run-time refine test) carry
  the proof the byte-identical guard usually would.
- A `de`/`pso`/`ss`/… fit's effective config no longer carries `simplex_*`,
  `cooling`, or the MCMC defaults; it reflects what the method reads. The ADR-0006
  "don't strip the union — refine needs it" footgun is **dissolved**, not relocated:
  refine self-supplies the Simplex group at the one site that needs it.
- `check` (the only fit_type with no co-located schema) narrows to global + extras.
- The one remaining cross-fit_type reach is now explicit at a single seam. A future
  second refiner (#403) turns the constant overlay into a refiner-keyed lookup, at
  which point the registry cross-dependency mechanism (option above) may be built to
  its real shape — the ≥2-user bar finally met.
