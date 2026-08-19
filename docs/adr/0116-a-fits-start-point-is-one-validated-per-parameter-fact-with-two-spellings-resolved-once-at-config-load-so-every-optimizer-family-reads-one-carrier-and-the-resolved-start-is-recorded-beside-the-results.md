# A fit's start point is one validated, per-parameter fact with two spellings, resolved once at config load, so every optimizer family reads one carrier — and the resolved start is recorded beside the results (issues #583, #559)

**Status: Accepted and implemented (2026-08-19).** Completes ADR-0043's `initial_value`,
whose own text claims a coverage it never had.

There was no supported way to say **"start this fit at exactly this point, inside the
declared box."** Every route failed, and every one of them failed **silently**: the fit
ran, converged, and reported a plausible number from a point that was not the one that
was asked for. This ADR makes the start point a first-class, validated, per-parameter
fact; makes the one key that pretended to be it say so when it is not; and writes the
resolved start into `Results/` so a displaced fit is visible after the fact.

## The defect

The capability was almost entirely present. ADR-0043 gave the new-era `parameter:` record
an `initial_value:` field, and `Algorithm._seed_initial_value_pset` honors it for twelve
fit_types. But `StartPointOptimizer._resolve_start_pset` reads `FreeParameter.value` on
**none** of its three branches:

```python
if self.START_POINT_KEY in self.config.config:
    return self.config.config[self.START_POINT_KEY]
if self._is_box_start():
    return PSet([v.value_from_quantile(0.5) for v in self.variables])
start_vars = [v.set_value(v.from_sampling_space(v.p1)) for v in self.variables]
```

So the seven start-point optimizers — `cmaes`, `powell`, `sim`, `gntr`, `lbfgs`, `trf`,
`ms`, which is precisely the set both issues are about — silently discarded a declared
`initial_value` and started at the box centre. Measured, on a two-parameter box with
`initial_value` declared at `k = 0.3`, `S0 = 100`:

```
declared initial_values: {'k': 0.3, 'S0': 100.0}
resolved start         : {'k': 1.505, 'S0': 89.44}     <- the box centre
```

ADR-0043 states in writing that `initial_value` is "respected in every algorithm family"
and that "a single-start local optimizer (Simplex/Powell) takes it as its one start
point." That is true only of the **no-prior** form, whose start the loader folds into
`p1` in sampling space (`config.py`); it is false the instant a bound or a prior is
present, which is every configuration either issue is about. `docs/gradient_fitting.rst`
documents the working case (for `profile_likelihood`) and the silently-broken case (for
`trf`/`lbfgs`/`gntr`) on the same page.

### The four silent failures, and what they actually are

**1. A bounded prior pins the prior's MEDIAN, not its mean.** The documented trick for
pinning a box-mode optimizer's start was a narrow prior at the desired value, because the
box branch returns `value_from_quantile(0.5)`. For a normal truncated to `[lower, upper]`
the median is not the mean whenever the truncation is asymmetric. Reproduced exactly on
the Borghans `init_Z_state` (mean `0.0879205`, `sd 0.2`, bounds `[0, 1]`):

```
init_Z_state   start=0.17318    intended=0.0879205    ratio 1.9697
```

This is not an edge case of the box branch. `TruncatedPrior.has_bounded_support` is
`True`, so the box branch is what fires for **every** bounded prior, normal included —
contradicting `local_base.py`'s own docstring, which described it as the
`uniform_var`/`loguniform_var` case.

**2. An out-of-box value is not clamped — it is reflected.** Both issues describe it as
"silently clamped to the bound". `FreeParameter.set_value` applies a periodic
triangle-wave fold, so on a box `[1e-5, 1e3]` both `1e9` and `1e-99` land on `0.001` — an
arbitrary interior point, not a wall — reported at `logger.debug`. That is worse than
filed: a clamp is at least predictable.

**3. `var`/`logvar` carry no bounds at all**, and nothing downstream checks a final
result against any bound in that mode. Unchanged by this ADR, and now stated out loud
(see below), because no resolver change can invent a bound that was never declared.

**4. `START_POINT_KEY` is unreachable from a conf** — the grammar rejects the key and its
value is a `PSet` object, not a scalar. Confirmed.

### Two more, found while fixing these

**`starting_params` is a silent no-op for fourteen fit_types.** It has exactly one read
site, `BayesianAlgorithm.start_run`. It is a `GlobalConfig` field typed `Any`, so it is
accepted for every `job_type` and warned about for none. It also assigns `p.value`
directly, bypassing `set_value`, so an out-of-box value is stored verbatim — neither
clamped, reflected, nor refused — and it is matched **positionally against declaration
order** while every file PyBNF writes is **alphabetical**.

**A mixed declaration starts bounded parameters at their lower bound.** `_is_box_start`
was all-or-nothing, so a single unbounded parameter sent every parameter down the `p1`
branch — and `p1` for a bounded parameter is its **lower bound**, read as if it were a
sampling-space start value. A `loguniform_var` over `[1e-3, 1e3]` started at
`10**1e-3 = 1.0023`, its lower corner, at no log level.

## The decision

### One fact, two spellings, one carrier

`start_point = <parameter> <value>` is new: edition-agnostic, one line per parameter,
independent of how the parameter was declared. `initial_value:` on a `parameter:` record
keeps its spelling and now means the same thing. Both resolve into
`Configuration.start_point`, a validated theta-space dict.

Two spellings rather than one is a deliberate cost. `initial_value` alone cannot serve
#559: the legacy grammar has no truncation tokens, so a legacy conf cannot express a
truncated prior at all, and reaching `initial_value` means migrating to edition 2 — which
rejects `fit_type` and requires `job_type`, rejects `objfunc` and requires one of
`objective`/`noise_model`/`profile_objective`, rebinds parameters by bare id, and flips
the centering default. That is a whole-surface migration, not advice a burned user can
act on. Conversely `initial_value` is shipped, documented, PEtab-mapped and honored by
twelve fit_types, so deleting it is churn for no one's benefit. Declaring both for one
parameter is accepted when they agree and **refused** when they disagree; silently
preferring one would reintroduce exactly the class of failure this work removes.

The carrier is the config's dict rather than `FreeParameter.value` because **no single
attribute on `FreeParameter` holds every declaration style's start in theta**: the loader
puts a prior/box record's start on `.value` and a no-prior record's into `p1` in sampling
space, and a legacy `*_var` conf can reach neither. A dict is also immune to the shared
template mutation described below.

### Resolution is per parameter, not per fit

`_resolve_start_pset` now resolves each parameter independently: a declared start, else
the box centre for a bounded-support prior, else the `var`/`logvar` point. This is what
makes a **partial** start point work — naming one parameter must not force a user to
restate every other — and it fixes the mixed-declaration failure for free, since a
bounded parameter now gets its box centre regardless of what its siblings declare.

The injected refiner start still wins outright. A refine begins from what the search
**found**; that a declared start point does not hijack the polish phase is the whole
content of a method chain, and it is a pinned contract (`tests/test_gradient_optimizer.py`,
`tests/test_shooting_sbml.py` both assert an injected start collapses multi-start to one).

`_is_box_start` is deliberately **not** touched, so a declared start pins start 0 and
leaves starts `1..N-1` as independent draws. Routing a user start through
`START_POINT_KEY` — #559's own suggested resolution (1) — would have collapsed the
scatter to a single start, silently making `population_size` a no-op: the same defect
class the issue was filed about. Pinning start 0 only also matches the contract the
population algorithms have had since ADR-0043, where exactly one member of the initial
population is seeded and the rest keep full diversity.

### Refuse, never fold

Every value is validated at config load, where the declared box is still in hand, against
`FreeParameter.prior_support()` — the declared box, correct for a uniform box and for a
truncated family alike, and deliberately independent of the `b`/`u` flag, since
`uniform_var = k 1 10 u` still declares `[1, 10]` on its own conf line. Refused: an
unknown or profiled-out name, a non-finite value, an out-of-box value, a non-positive
value on a log-scaled parameter (the symptom of writing the log, which the legacy
`logvar` convention invites), a spelling disagreement, `start_point` alongside
`starting_params`, and a start point on `job_type = check`.

Every one is a `PybnfError`, never the bare `OutOfBoundsException`, which subclasses
`Exception` and so reaches the user as *"an unknown error occurred … please report this
bug"* — including on the PEtab import path, where an out-of-box `nominalValue` crashed
that way on a file the user never wrote.

Warned, not refused: a start point on a parameter with no finite support. There is no box
to check it against and no resolver change can invent one, so the honest answer to #583
item 3 is to say so once.

### `starting_params` becomes loud

A hard `PybnfError` on any `job_type` whose registry family is not `sampler`, pointing at
`start_point`. #559 asks for exactly this — *"silently discarding a start point is the
worst of the three options"* — and a warning about a key that has never done anything is
indistinguishable from the silence that caused the report. Blast radius in this tree is
**zero**: all 338 shipped confs that set it are `am` (337) or `mh` (1), and no test
asserts its behaviour. Wiring it to the optimizers instead was rejected: it is positional
against declaration order, has no length check, and `noise_profiling` prunes
`self.variables` after load, so it would introduce a fresh silent index-shift class.

### The resolved start is recorded

`Results/start_point.txt` — one row per parameter in **declaration** order, with the
start value, its source, and the declared box. This is #583's second ask, and it is worth
doing independently of the first: *every* failure above is silent precisely because no
artifact recorded where a run began, so a displaced fit is indistinguishable from a
correct one. It is also the only way to recover a CMA-ES start at all, since that start
seeds the distribution mean and is never itself evaluated — there is no first row of
`sorted_params_*.txt` to read it from.

It is written from `Algorithm.run`, after `start_run()` and before any job is submitted:
the one point every family passes through with its start in hand. A refine writes
`start_point_refine.txt`, the `_refine` suffix every second-phase artifact already uses,
since `pybnf.py` points the refiner at the fit's own `res_dir`.

## Consequences

* A fit seeded at a published point now starts there, for every `job_type`, with the box
  intact. That is the routine benchmarking operation both issues were filed about.
* A conf that set `starting_params` on a non-sampler `job_type` now fails to load. It was
  doing nothing before; this is the point.
* `pset.py`'s bounds check moves from `if value:` to `if value is not None:`, closing a
  hole where exactly `initial_value: 0` — a legitimate value for a linear parameter —
  skipped validation. Verified to affect only `0.0`.
* `set_value` no longer writes `self.value = self.lower_bound` when folding. That
  mutated the shared template `FreeParameter` in `Configuration.variables`, which every
  Algorithm aliases and which rides the algorithm's pickle, so the contamination survived
  a checkpoint and a `--resume`. Nothing read the value it wrote.
* `_box_widths_u` now derives from `prior_support()` rather than `p2 - p1`. For a
  `uniform`/`loguniform` box these are the same number bit for bit; for a **truncated**
  prior `p1`/`p2` are location and scale, so the width came out as the scale and, for the
  entirely ordinary `sd == mean`, as exactly `0.0` — which CMA-ES squares into a singular
  covariance diagonal, freezing that coordinate for the whole run.
* The concurrent multi-start scatter now honors `initialization`, which was a silent
  no-op there, and says so when `population_size > 1` cannot be scattered for want of a
  box.
* The PEtab importer emits `start_point` from `nominalValue`. ADR-0043's field table has
  always advertised that mapping; neither direction implemented it.

## Deferred

Recorded because they were found in this work and are real, not because they are out of
reach:

* **`_check_variable_keyword_combination` is blind to `parameter:` records.** It matches
  free-parameter keys with `re.search('var$', k[0])`, so the edition-2 surface bypasses
  the whole var/logvar-vs-prior gate. Fixing it needs a call-order restructure — the
  method runs from inside `_load_variables`, before the `FreeParameter`s exist — and the
  obvious rewrite (re-deriving from `{v.type}`) is *verified* to break working configs: a
  truncated normal has type `normal_var`, whose family reports
  `has_bounded_support == False`, and would be falsely refused.
* **Ungating `parameter:` records from edition 2**, which would make a cross-style
  duplicate check mandatory: `uniform_var = k 0 10` plus `parameter: k, ...` silently
  yields two `FreeParameter`s named `k` today.
* **CMA-ES start asymmetries**: `reset()` does not re-resolve the start, and each restart
  re-draws a fresh random box point, so with `cmaes_restarts > 0` a declared start
  governs run 0 only.
* **The `b`/`u` flag itself**: `uniform_var = k 1 10 u` gives infinite reflecting bounds
  while `has_bounded_support` stays `True`, so the box governs the start, the scatter and
  now start-point validation, but nothing enforces it during the search.
