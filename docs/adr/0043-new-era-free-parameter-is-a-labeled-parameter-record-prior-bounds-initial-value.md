# The new-era free parameter is a labeled `parameter:` record — every part is named (issue #417)

**Status: Accepted (decision 2026-06-20); authoring surface implemented 2026-06-20 (Phase 1);
`initial_value` population-seeding implemented 2026-06-21 (Phase 2).** The grammar, the per-family
field names (`Prior.field_names`), the edition-gated loader (`config._free_parameter_from_record`),
and the `FreeParameter` wiring are built and tested (`tests/test_parse_class.py::TestParameterRecord`,
`tests/test_config_class.py::TestParameterRecordConfig`). Phase 2 — exactly one initial-population
member seeded at the `initial_value` point (see "initial_value vs prior") — is built and tested
(`tests/test_optimizer_integration.py`, the `initial_value`/reserve/unseeded cases). Under
`edition >= 2` a free parameter is declared as a labeled, comma-separated record:

```
parameter: <id>[, parameter_scale: linear|log10|ln][, prior: <family>][, <family field>: <v> ...][, lower: <x>, upper: <y>][, initial_value: <v>]
```

so a position-free, self-annotating line replaces the legacy positional `*_var = <id> <p1> <p2>`.
Issue #417's truncation capability (ADR-0020) is delivered as the `lower`/`upper` fields of this
record — not as appended unnamed numbers on the legacy line (the approach drafted and rejected
below). Legacy `uniform_var = k 0 10` is untouched (edition-gated, exactly like ADR-0028/0031/0034).

## Why a record, not a patched positional line

The legacy line `normal_var = k 0 1` is positional: the reader must remember that slots are
mean, sd — and a truncation box would be slots 4–5, a *box* spliced into a run of distribution
parameters. That is precisely the "remember which position means what" wart the new era was built
to kill. ADR-0028 already replaced the filename→suffix convention with labeled records
(`experiment:`, `condition:`, `observable:`) and pinned the rule: links and fields are **stated,
never inferred from position**. ADR-0034 dropped the `__FREE` marker so a free parameter binds to
the model **by bare id**. The free-parameter line was the one authoring surface ADR-0028 left in
the legacy positional shape ("free parameters are unchanged"); this closes that gap. `parameter:`
joins `model:` / `condition:` / `experiment:` / `observable:` as the fifth new-era colon-keyword
record, and the whole problem block reads as one consistent structured language.

## The fields

Universal fields (any family):

| Field | Meaning | PEtab v2 column |
|---|---|---|
| `parameter:` | the head: the free parameter's **id** (binds to the model entity by id, ADR-0034) | `parameterId` |
| `prior:` | the distribution **family** (`normal`/`laplace`/`cauchy`/`gamma`/`exponential`/`chisquare`/`rayleigh`/`uniform`) | `priorDistribution` |
| `parameter_scale:` | `linear` (default), `log10`, or `ln` — the sampling-space transform (the base prefixes the family keyword: `log{f}_var`/`ln{f}_var`) | (PEtab v2 has **no** scale column; it bakes scale into the prior name) |
| `lower:` / `upper:` | the parameter **bounds**; they *truncate* the prior (ADR-0020) and are the box for box-optimizers | `lowerBound` / `upperBound` |
| `initial_value:` | optional **start point** of the walk (see "initial_value vs prior") | `nominalValue` |

Per-family distribution parameters are **named by the family**, so no slot is positional:

| `prior:` | fields | n |
|---|---|---|
| `normal` | `mean`, `sd` | 2 |
| `laplace` | `location`, `scale` | 2 |
| `cauchy` | `location`, `scale` | 2 |
| `gamma` | `shape`, `scale` | 2 |
| `exponential` | `scale` | 1 |
| `rayleigh` | `scale` | 1 |
| `chisquare` | `dof` | 1 |
| `uniform` | (none — `lower`/`upper` **are** the prior) | 0 |

Examples:

```
parameter: k, prior: normal,  mean: 4, sd: 1, lower: 2, upper: 6
parameter: k, parameter_scale: ln, prior: normal, mean: 1, sd: 0.5, lower: 0.1, upper: 100
parameter: k, prior: uniform, lower: 0, upper: 10        # uniform: lower/upper are support AND bounds
parameter: k, prior: gamma,   shape: 2, scale: 3
parameter: k, initial_value: 5                           # no prior -> a local-optimizer start point
```

Field **order is free** — the loader reads fields by name (any order parses identically). The
canonical order above leads with `parameter_scale:` so it primes how `mean`/`sd` are read (they are
in the parameter's sampling space). `lower`/`upper` are in θ (the parameter's own value) on every
scale; on a log scale they must be strictly positive (`log(≤0)` is `-inf`), as must a log start
point's `initial_value` — both caught with a clear error.

## Three design questions this settles

**`sd`, not `variance` (naming forces the truth into the open).** PyBNF parameterizes a normal by
mean + **standard deviation** (`Normal(loc, sigma)`), as does PEtab. The positional form hid which
of the two numbers was the width *and* whether it was sd or variance; the named field must match the
math (`variance = sd²`) or the fit is silently the wrong width. The field is `sd`. (This is the
general argument for the record: the label is a contract.)

**`lower`/`upper` are two named fields, not a `bound: lo hi` pair.** A single `bound: 2 6` still
hides two numbers behind one label, re-importing the positional sin in miniature — and forces a bad
choice on ordering: *first = lower* (a position) or *min/max* (magic that silently accepts a flipped
typo). Naming both **dissolves** the ordering question: order is irrelevant, nothing is inferred,
and it is PEtab's two columns verbatim. Both are required together for a two-sided truncation;
one alone is the still-deferred one-sided case (ADR-0020) and raises. Omit both → unbounded prior.

**`initial_value` belongs to the parameter; the step/kernel belongs to the method.** The legacy
`var = k 5 0.5` put a per-parameter *step size* on the line. A step is how a method **moves** —
Simplex's reflection step, MH's proposal width, Adaptive-MCMC's *learned* covariance (ADR: `am`) —
so it is a **tool** setting (problem-vs-tool split, ADR-0028 §4), parameterized once per run (or
learned), never per parameter. It leaves the line entirely. What stays is `initial_value`: *where
the walk starts*, which is a legitimate per-parameter (PEtab `nominalValue`) fact.

### `initial_value` vs `prior` are orthogonal (you can have both)

A parameter may carry **both** a `prior:` and an `initial_value:`: the prior is the distribution,
the `initial_value` is where the chain/search starts (e.g. start a sampler near a known mode). This
is not a confusion of optimization and sampling — *where you start* (a point) and *how you move* (the
kernel) and *the target* (the prior) are three separate things; the record carries the first and the
third, the method owns the second. Default start (no `initial_value`): drawn from the prior or the
bounds per the initialization distribution (ADR-0030). `initial_value` is **respected in every
algorithm family**: a population sampler/optimizer (DREAM/AM/PT/de/pso/ss/…) sets **exactly one
member of its initial population to the `initial_value` point** (the point defined across whichever
parameters carry one; any parameter without an `initial_value` is drawn as usual *for that member*),
and **every other member is the normal draw from the prior/bounds, unchanged** — there is no
clustering around the point, so the population keeps the full diversity global search needs. A
single-start local optimizer (Simplex/Powell) takes it as its one start point. So a named
`initial_value` always seeds exactly one real start at that point, and nothing else changes.

## Behaviors

- **`prior:` omitted but `lower`/`upper` given → uniform over the bounds.** PEtab requires bounds on
  every estimated parameter and defaults a prior-less estimated parameter to uniform-over-bounds;
  the importer already does exactly this. So `parameter: k, lower: 0, upper: 10` ≡
  `parameter: k, prior: uniform, lower: 0, upper: 10`. A bare `parameter: k, initial_value: 5` with
  no prior and no bounds is the local-optimizer start point (today's `var`); `parameter_scale: log10` on it is
  today's `logvar`, and `parameter_scale: ln` a natural-log start point (`lnvar`).
- **Export is transcription.** Each field is a PEtab column (table above), so the new-era exporter
  emits the parameter row directly, and a unbounded native prior can now carry bounds → it exports
  *valid* PEtab (closing the "blank bounds" limitation ADR-0038 recorded).
- **Edition-gated.** New surface only under `edition >= 2`; legacy `*_var = …` parses and builds
  byte-for-byte unchanged. A job uses one style or the other.

## Implementation

- **Grammar (`parse.py`).** `parameter:` is a colon-keyword record like `experiment:`: parsed
  permissively into ordered `[name, value]` field groups (the `noise_model`/`observable` pattern), a
  value being a number or a bare word. `ploop` stores `('parameter', id) -> {field: str}`; numeric
  strings stay strings until the loader floats them. *(Done, Phase 1.)*
- **Per-family field names (`priors/`).** Each family declares an ordered `field_names` tuple
  (`Normal.field_names = ('mean', 'sd')`, …; `Uniform = ('lower', 'upper')`; the one-parameter
  families a singleton) — declared on the class like `has_bounded_support`/`n_params`, length matching
  `n_params`. The loader maps named fields → the family's `build(p1, p2, scale)`. *(Done, Phase 1.)*
- **Loader (`config._free_parameter_from_record`).** Resolves `parameter_scale:` + `prior:` to the
  existing `{prefix}{family}_var` keyword (`prefix ∈ '' | log | ln`), pulls the family's fields by name
  (missing/unknown/non-numeric → a clear error), passes `lower`/`upper` to `FreeParameter(lb=, ub=)` for
  an unbounded family (ADR-0020) — or as the support for `uniform` / a prior-less bounded default — and
  `initial_value` to the `value=` slot (prior params) or the first slot (no-prior `var`/`lnvar`, which
  Simplex reads via `p1`). Edition-gated `>= 2` in `_load_variables`. The whole
  TruncatedPrior/reflection/sampling capability is unchanged — this is only the **authoring surface**
  over it. *(Done, Phase 1.)*
- **The `ln` scale (`priors/scale.py`, `priors/__init__.py`).** A first-class `Ln` scale
  (`forward=np.log`, `inverse=np.exp`, `name='ln'`) joins `LINEAR`/`LOG10`; the registry gains the
  `ln{family}_var` + `lnvar` keywords (record-only — `var_keyword_grammar` still emits just linear+log10,
  so the legacy positional grammar is unchanged). Each `Scale` now carries an explicit `name` so output
  labels its base. The handful of base-10 hardcodes that bypassed `_scale` are routed through it —
  `FreeParameter.diff` (now `_scale.forward(a/b)`), the Simplex/`local_base` start (`from_sampling_space(p1)`
  instead of `exp10`), the histogram header (labels `log10_`/`ln_`) — **bit-identical for log10/linear,
  correct for ln**. This amends ADR-0022's log10-only simplification while keeping its rule (every log
  scale names its base; bare `log` is still rejected as ambiguous). *(Done, Phase 1.)*
- **`initial_value` population seeding (Phase 2, done 2026-06-21).** Exactly one member of the initial
  population sits at the `initial_value` point. A shared helper `Algorithm._seed_initial_value_pset`
  (`algorithms/base.py`) takes a member already drawn from the prior/bounds and overwrites each
  parameter that declares an `initial_value` (carried on `FreeParameter.value` by the loader) with that
  value, leaving the rest at their draw — so a partially-specified seed is still a complete pset, and a
  parameter without an `initial_value` is drawn as usual *for that member*. It returns the input
  unchanged when no parameter declares an `initial_value`, so non-record runs are byte-for-byte
  unaffected. The helper is **not** routed through `random_pset` (which returns one pset and would seed
  *every* member); instead each population algorithm's `start_run` calls it on exactly one member of its
  main initial population — `de` on `proposed_individuals[0][0]` (and `ade` on `individuals[0]`), `pso`
  on the first particle, `ss` on the first init pset (**not** the latin-hypercube reserve), `sa` on the
  first replicate, and the Bayesian samplers (`am`/`dream`/`p_dream`/`pt`/`mh`) on `first_psets[0]` in
  the shared `BayesianAlgorithm.start_run`, placed *before* the `continue_run`/`starting_params`
  overrides so those still take precedence. The single-start local optimizers (Simplex/Powell) already
  take `initial_value` as their lone start via `p1`, untouched. (Distinct from `starting_params`, which
  overrides *all* members uniformly — samplers only.) Edition-agnostic at the algorithm layer:
  `FreeParameter.value` is set the same way however the parameter was declared.
- Tests use **bare ids** (no `__FREE`) per ADR-0034; the golden-config corpus can gain additive
  new-era cases (ADR-0013, additions only).

## Considered / rejected

- **Append unnamed `lb ub` to the legacy line** (`normal_var = k 0 1 -5 5`). Rejected: positional
  slots 4–5, the exact wart the new era kills; and it kept the dropped `__FREE` examples. (Drafted,
  then reverted, on the way to this ADR — superseding the would-be ADR-0042.)
- **A single `bound: lo hi` (or `bounds:`) field.** Rejected: two unnamed numbers behind one label;
  forces first=lower (positional) or min/max (masks a flipped typo). `lower`/`upper` instead.
- **Keep `step` on the parameter line (as `init_step:`).** Rejected: a step/kernel is a method
  attribute, not a problem fact (ADR-0028 §4); it lives with the method (global `simplex_step`, or
  the learned AM covariance), not the parameter.
- **`variance:` for the normal width.** Rejected: the families parameterize by `sd`; a mismatched
  label silently fits the wrong width.
- **`scale:` for the sampling-space transform.** Rejected on a name collision: `scale` is already a
  *distribution* field on five families (laplace/cauchy `location, scale`; gamma `shape, scale`;
  exponential/rayleigh `scale`), so `scale:` would mean two things on one line, disambiguated only by
  value type — exactly the inference this record banishes. The transform field is **`parameter_scale:`**.
- **Baking the scale into the prior name (the PEtab v2 way: `normal` / `logNormal`).** Rejected for
  us: PEtab v2 removed the `parameterScale` column and carries scale in the distribution name, but it
  has only **two** scales (linear + natural-`log`; no log10). PyBNF has **three** (`linear`/`log10`/`ln`),
  so baking in gives `normal`/`log10normal`/`lnnormal` × every family ≈ 24 user-facing prior names — a
  zoo. One `parameter_scale:` field with three values is the clean form; we diverge from PEtab's naming
  here and translate at the import/export seam (which already does, e.g. natural-log priors ↔ log10).
- **Reject `ln` / sample log10-only (ADR-0022).** Rejected: forcing a user to convert ln↔log10 in
  their head is user-hostile, and ln is a legitimate ask. ADR-0022's real rule is *every log scale
  names its base — no ambiguous bare `log`*; an explicit `parameter_scale: ln` honors that (a bare
  `log` is still rejected as ambiguous). So this **amends ADR-0022's log10-only simplification**: PyBNF
  gains a first-class `Ln` scale, and the few base-10 hardcodes (Simplex's `exp10` start, `FreeParameter.diff`)
  now route through the parameter's own `Scale` -- bit-identical for log10/linear, correct for ln.
- **Keep the positional `*_var` line as the only surface.** Rejected for new-era: it is the gap
  ADR-0028 left; legacy keeps it, the new era gets the record.

Relevant: ADR-0028 (new-era labeled records + problem-vs-tool split this extends), 0034 (bind by id,
`__FREE` is legacy), 0020 (the two-sided truncated-prior capability `lower`/`upper` reach), 0010 (the
prior-family registry that declares the per-family fields), 0030 (initialization distribution vs
prior — the default start when `initial_value` is absent), 0003 (prior in the parameter's own scale),
0031 (edition select-and-freeze), 0038 (which filed the native truncation grammar on #417, and the
blank-bounds export limitation this lifts). Issue: **#417** (reframed from "truncation grammar" to
"the new-era `parameter:` record"; truncation is its `lower`/`upper` fields).
