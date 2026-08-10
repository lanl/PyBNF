# A scalar absolute tolerance charges every species for the smallest one, so the bngsim SBML path gives that tightening back **per species** — while refusing to tighten any species past what it integrates today (issue #549)

**Status: Accepted and implemented (2026-08-10). Supersedes ADR-0103.** ADR-0103 was correct for
the capability available when it was written: bngsim's `Simulator.run` took a scalar `atol`, so the
only question open to it was *which* scalar. lanl/bngsim#196 removed that constraint. What #549
proposed doing with the new freedom turned out to be wrong, and the measurement that says so is the
substance of this ADR.

## What ADR-0103 could not say

ADR-0103 derived `atol` from the model and let it only ever tighten, with the scale read as the
**median** strictly-positive nominal species value. It documented the cost of the median in its own
text:

> The median rather than the minimum, because `Brannmark_JBC2010` seeds one transient intermediate
> at `1.8e-9` against principal species at `0.1..10`, and resolving *that* asks for `1e-17`, which
> makes the model fail outright on `mxstep` at interior fit points.

That is a compromise, not a rule. CVODE's error test weights state *i* by `rtol*|y_i| + atol`, so a
scalar `atol` is a single statement about a state that may span decades. `Brannmark_JBC2010` reads
`3.3e-10`, which is **too tight for the top**: `IR`, `IRS` and `X` sit at ~10, so `3.3e-10` puts
them under an absolute floor a hundredth of what `rtol` alone would ask — `3.3e-11` *relative* —
and that is the end of the model that never needed anything. ADR-0103 listed the fix under
**Deliberately out**, naming the blocker (`Simulator.run` takes a scalar). lanl/bngsim#196 is that
backend change: `run(atol=...)` now accepts a sequence and routes it to `CVodeSVtolerances`, on a
code path that leaves a float taking `CVodeSStolerances` bit-for-bit unchanged.

## The decision, and the measurement that shaped it

**Derive one absolute tolerance per species from the model's nominal state, clamped into
`[the model's own scalar, the backend default]`.**

```python
atol_i = clamp(rtol * y_i, scalar_atol, default_atol)
```

Both clamps carry an argument.

### The upper clamp is ADR-0103's, applied per species

Nothing is ever loosened past the backend default, so a species of order one keeps `1e-8` and a
model whose species are all of order one integrates exactly as before. This cannot be delegated:
bngsim's own `derive_atol` has **no** upper clamp and returns `1.0e-07` for `Brannmark`'s `X`,
looser than the default. `test_only_ever_tighten_is_pybnfs_to_apply_not_the_backends` pins that
against the library rather than against a copy of our own arithmetic.

### The lower clamp is the model's own scalar, because the alternative was measured and lost

This is where #549's plan and this ADR part company. Read literally, "resolve each species to
`rtol` of its own magnitude" says `Brannmark`'s `IRp` (nominal `1.76e-9`) should be integrated at
`1.76e-17`. That rule was implemented and measured, on 100 points sampled from Brannmark's own fit
box the way a multi-start start is sampled, with the fit's union sensitivity request applied — the
isolate #549 asks for under "claim A", no optimizer in the loop:

| rule | simulations dead / 100 | CVODES `mxstep` lines | wall clock |
|---|---:|---:|---:|
| ADR-0103's scalar (`3.3e-10` for every species) | **39** | 3542 | 576 s |
| the same scalar, sent as a uniform *vector* | **39** | 3957 | 587 s |
| per-species, floored at `1e-16` (#549 as written) | **91** | 6220 | 985 s |
| per-species, no floor at all | **91** | — | — |
| **per-species, floored at the model's own scalar** | **33** | 2930 | 428 s |

Every failure in every row is a solver failure; none is a `wall_time_sim` kill. The first round of
this measurement ran the regimes concurrently and had to be discarded and re-run serially, because
a 10 s per-simulation budget on a contended machine kills points the tolerance had nothing to do
with.

Three things follow.

- **#549's rule as written more than doubles the deaths on the slug it was filed from.** It is
  ADR-0103's withdrawn *minimum* rule reappearing one species at a time, and the reasoning that
  withdrew it did not stop being true when the tolerance became a vector.

- **#549's "one genuinely open design question" — whether to keep `_DERIVED_ATOL_FLOOR = 1e-16` —
  is moot.** The issue frames it as a 5.7x decision on one species of one model, settleable by
  measurement. The measurement says 91 either way: the damage is done well above `1e-16`, by the
  species landing at `1.1e-13` and `1.6e-12`. Under the clamp adopted here nothing reaches `1e-16`
  at all, since the scalar is itself clamped at or above it.

- **The uniform-vector control lands exactly on the baseline, 39 = 39.** So the vector plumbing and
  the `CVodeSVtolerances`-vs-`SStolerances` switch are a non-event, and the differences above are
  attributable to the tolerance *values* alone. That control is what makes the rest of the table
  worth anything.

**Why the tightening hurts.** A tolerance below `rtol*|y_i|` is inert until `|y_i|` falls far below
its *nominal* value, and what it then demands is that a species which has decayed into nothing be
resolved as if it had not. Initial values cannot tell that case apart from a species that genuinely
lives down there. Distinguishing them needs the trajectory, which is lanl/bngsim#213's
`CVodeWFtolerances`, not this.

**So what this change does is give back ADR-0103's over-tightening to the species that never needed
it, and nothing else.** Every entry lands in `[scalar_atol, default_atol]`, so no species is
integrated more tightly than PyBNF integrates it today and **no model that runs today can start
failing** — a property, not a hope, and asserted as one
(`test_no_species_is_ever_integrated_more_tightly_than_it_is_today`).

### The rest of the decision

- **A species with no magnitude falls out of the same expression.** `rtol * 0` clamps up to
  `scalar_atol`, so a species seeded at zero is measured against the model's own scale and needs no
  rule of its own. That is also the no-regression choice: bngsim's `derive_atol` substitutes the
  smallest strictly positive entry instead, which on `Giordano_Nature2020` — this change's control
  — would put its four zero-seeded species at `1.667e-08` against a model scale of `3.67e-07`,
  tightening them 22x for no measured reason.

- **The vector is a constant of the model, not of the fit point.** Read off the document at load and
  memoized, exactly as ADR-0103's scalar was, and for the same two reasons: a tolerance that moved
  with a fitted initial condition would put a step in the objective wherever the derivation crossed
  a rounding boundary, and a gradient fit's scalar (line-search) evaluations must be integrated to
  the same accuracy as the sensitivities they are compared against or the two disagree about what
  the objective *is*. bngsim ships an `AUTO` token that derives from the model's **live** state —
  the one the next `run()` would start from, which on a fit is the fit point — so it is the wrong
  tool here. PyBNF fits initial conditions on several problems; bare `AUTO` would be invisible in
  the usual way, since the objective still looks correct, the finite-difference gradient check still
  passes, and only the search behaves oddly.

- **The vector is ordered to the *engine* model's `species_names`,** the names bngsim indexes its
  state by. `normalize_atol_vector` (lanl/bngsim#212) takes the length/position contract once at
  setup rather than at the first `run()`, which on a fit is a long way from where the vector was
  built.

- **ADR-0103's scalar does not retire; it changes job.** bngsim falls its steady-state cutoff back
  to the scalar `atol` when unset — "also when a per-species atol is in force (issue #196): the
  criterion is a single norm over every species and has no per-species reading to take" — and
  `_resolve_atol` returns the *Simulator's own* `1e-8` in that case, not anything derived from the
  vector. Left alone that silently reverts ADR-0103's steady-state fix: on a model whose states are
  ~1e-8, `||dx/dt|| < 1e-8` is satisfied *at t = 0*, so every `time = inf` measurement (ADR-0086)
  and every pre-equilibration phase (ADR-0052/0104) would return the initial state and call it
  equilibrium. So the median-derived scalar is passed explicitly as `steady_state_tol` whenever the
  vector is in force. One statement about the model's magnitude still governs both, and today's
  steady-state behaviour is reproduced exactly.

- **`sbml_atol` stays scalar-only, and stating it switches the derivation off.** PyBNF's config
  grammar is scalar (`numkeys_float`); a per-species vector in a `.conf` would need a new
  species-keyed grammar, and the case for it is hypothetical. Keeping it scalar also keeps a clean
  off-switch: stating it pins the pre-#196 `CVodeSStolerances` path bit-for-bit, ulp included.

- **A vector that says nothing a scalar does not is not sent.** True of 19 of the 23 subset-I slugs
  — which is what "ADR-0103 had nothing to give back here" looks like, since a model the scalar
  derivation left at the backend default has no over-tightening to undo.

## What actually moves

| slug | ADR-0103 scalar | ADR-0105 vector `[min, max]` | released |
|---|---:|---|---:|
| `Armistead_CellDeathDis2024` | 4.94e-09 | `[4.944e-09, 1.000e-08]` | 2 of 4 |
| `Bertozzi_PNAS2020` | 5.00e-09 | `[5.000e-09, 9.000e-09]` | 1 of 3 |
| `Brannmark_JBC2010` | 3.30e-10 | `[3.302e-10, 1.000e-08]` | 4 of 9 |
| `Giordano_Nature2020` | 3.67e-15 | `[3.667e-15, 1.000e-08]` | 4 of 13 |

The other 19 take the scalar call unchanged. These are exactly the four slugs #546 tightened, which
is the shape of the change: it is a *refund*, so only a model that was charged can receive one.

Two consequences worth stating rather than discovering:

- **`Raia_CancerResearch2011` does not move**, although an earlier reading of #549 predicted it
  would. Its smallest species is `0.34`, so an unclamped derivation would tighten it from the
  backend default to `3.4e-09`. Under this ADR the vector never tightens, so Raia keeps the scalar
  path. #549's verification bullet — "the 19 that ADR-0103 left untouched should be unchanged" — is
  therefore satisfied literally, which the plan it was written against would not have been.

- **`Smith_BMCSystBiol2013` reaches the ordering fallback** and keeps its scalar: bngsim renames a
  species whose SBML id collides with an Antimony reserved word, so its `NULL`/`null` load as
  `_ant_NULL`/`_ant_null` and the vector cannot be ordered against the document without guessing. A
  mis-ordered vector would assign one species' tolerance to another and nothing downstream would say
  so. It costs nothing here — Smith's vector would have been elementwise the default anyway — and
  the fallback is taken *after* the "says nothing a scalar does not" test, so a model in Smith's
  position whose vector was a no-op stays silent rather than warning about a difference that would
  not have existed.

## Scope

**In:** `_bngsim_caps.py` (`BNGSIM_HAS_PER_SPECIES_ATOL`); `bngsim_sbml_model.py`
(`_derive_atol_vector`, `_compute_nominal_species_values`, `_per_species_atol`,
`_derive_per_species_atol`, `_run_tolerance_kwargs`, and the tolerance kwargs on `_run_simulation`'s
`sim.run` call); `docs/config_keys.rst`.

**Out (unchanged):** every BNGL/net model, whose tolerances come from BNG2.pl's actions block. Every
stochastic (`ssa`) run, which has no CVODE tolerances to set. The RoadRunner SBML backend. Every
model the scalar derivation left at the backend default. Any install without lanl/bngsim#196, which
keeps ADR-0103's scalar bit-for-bit and runs every fit it runs today.

**The capability probe is a name, not a version floor.** The build that first carried #196 declares
`0.12.2`, which is also the version of the released wheel 25 commits behind it, so a version floor
would report *present* on an install without the capability — the expensive direction, since the
vector would then be handed to a `run` that takes only a scalar. `AUTO` and `normalize_atol_vector`
are exported from the package namespace by lanl/bngsim#212 and listed in `__all__`; probing the two
names PyBNF actually calls keeps the flag honest if either ever moves. (This was worth getting
right: when #549 was filed both lived in the private `bngsim._atol`, so `hasattr(bngsim, 'AUTO')`
returned `False` on a build that *had* the capability — silently routing a capable install down the
scalar fallback, with a result that still looked correct.)

**Deliberately out:** resolving a species *below* the model's own scale, and a tolerance that
follows the trajectory — the same thing from two directions, and the reason the lower clamp exists.
A vector built from initial values cannot see a species that starts at order one and decays to
something tiny, nor tell a genuinely-tiny species from one that is merely transient. That is
lanl/bngsim#213's `CVodeWFtolerances`, which takes a #196 vector as its *ceiling*, and it is also
what `Weber_BMC2015` needs: Weber's seven species all start **above** one, so every entry of its
nominal-state vector is the backend default and the vector is elementwise the scalar it replaces.
#549 originally claimed Weber as a second slug this change would unblock; it is not reachable from
here, and it and the `decades` config surface belong to the follow-up.

## Verification

Two claims are braided together in #549 and they cost very different amounts. **A**, that the
`mxstep` tax is lifted, is a property of simulating at a parameter point. **B**, that
`Brannmark_JBC2010` now solves — final `OG` under 1.92 rather than 3.21 — needs a real fit, is a
claim about the benchmark corpus rather than about PyBNF's tolerance handling, and belongs to
wshlavacek/BNGL-Models#38 where the 16-of-23 count is tracked. Gating a code review on an hours-long
stochastic fit whose single-run outcome near the threshold would not settle much is the wrong trade.
This ADR verifies **A**, and the box-sample table above is that verification: 39 dead simulations to
33, and 576 s to 428 s, on 100 points of Brannmark's own fit box.

- **The corpus derivation**, over all 23 subset-I slugs: 19 unchanged, and the 4 that #546 tightened
  moving to vectors. Table above.
- **Nothing moves that should not**, checked as an objective rather than as a tolerance.
  `tools/nominal_check.py` over all 23 slugs, run twice in the same process against the same bngsim
  build with `BNGSIM_HAS_PER_SPECIES_ATOL` forced off and on — a cleaner control than the committed
  `nominal_check.json`, whose values predate this wheel and would have conflated this change with
  everything since:

  | | slugs | `J_paper` |
  |---|---:|---|
  | scalar path | 19 | **bit-identical**, every one |
  | vector, and unmoved anyway | 1 (`Giordano_Nature2020`) | **bit-identical** |
  | vector, moved | 3 | 8.2e-15 (`Bertozzi`), 1.8e-09 (`Armistead`), 1.9e-08 (`Brannmark`) |

  Giordano is the interesting row: it takes a vector, and its objective does not move at all, because
  the four species released are the ones already governed by the relative term. The three that do
  move, move in the eighth significant digit or beyond — a trajectory integrated at a different
  tolerance, not a different model.
- **`Giordano_Nature2020` does not regress** — the slug ADR-0103 was written for, and the one whose
  wrong gradient started this line. Its four released species are the ones sitting at or above the
  model's scale; its small ones keep `3.67e-15`. Its assembled gradient against central differences,
  at an interior bounds-clear point, worst relative error over all 50 fitted columns:

  | | `h = 1e-3` | `h = 3e-3` | `h = 1e-2` |
  |---|---:|---:|---:|
  | ADR-0103's scalar | 5.89e-04 | 2.45e-03 | 1.50e-03 |
  | ADR-0105's vector | 8.57e-04 | 2.56e-04 | 1.11e-03 |

  Read that by the corpus's own rule (`pybnf-jobs/.../tools/README.md`): a real defect does not move
  with `h`, FD noise does. Both rows move with `h` by more than they differ from each other, and the
  ordering between them flips — the vector is worse at `1e-3`, ten times better at `3e-3`. So there
  is no defect in either and no difference between them that this instrument can resolve; both sit
  two to three decades under the 7.7e-02 #546 opened on. A single `h` would have supported a
  confident wrong statement in either direction, which is exactly what that README warns about.
- **The mechanism, in a fixture** (`tests/test_sbml_solver_tolerances.py`): a species at 10 in a
  model whose median is 1e-2, so ADR-0103 hands it `1e-10` and the vector gives it the `1e-8` it was
  always entitled to. 288 CVODE steps against 377, with the released species still matching its
  closed form to 2.0e-05 wherever it is clear of its own tolerance.
- **The boundary, also in a fixture.** `test_the_vector_does_not_resolve_what_adr_0103_declined_to`
  asserts that a species *below* the model's scale keeps the compromise ADR-0103 struck for it and
  is still under-resolved. It is tempting to read "one tolerance per species" as "every species is
  finally resolved against its own magnitude", and the measurement above is why that reading is not
  shipped; a test is a better place to say so than a comment.
- **The safety property, asserted as a property**: every entry lies in
  `[scalar_atol, default_atol]`.
- **The scalar tests did not move.** They now also cover the fallback path — an older bngsim, an
  explicit `sbml_atol`, a model with no over-tightening to undo.
- **The full default suite, against the same suite on the base commit** — run by stashing the
  working tree rather than in a second worktree, because PyBNF is installed editable and pinned to
  this checkout, so a worktree would have run the base commit's *tests* against the branch's *code*.
  263 failed / 35 errors on both, and the 298 `FAILED`/`ERROR` lines are byte-identical between the
  two; the branch passes 21 more tests, which is exactly the number added here. (The 263 are this
  machine's BNG2.pl-dependent tests, which need a BNG2.pl it does not have.)

Relevant ADRs: **0103** (superseded — the scalar, and why the median was the least-bad single
answer), **0086** and **0052**/**0104** (the steady-state and pre-equilibration paths the explicit
`steady_state_tol` protects), **0093** (the wall-clock budget a tolerance change spends against).
Relevant issues: lanl/bngsim **#196** (the vector), **#212** (its public export), **#213** (the
trajectory-following successor, and `Weber_BMC2015`), **#546** (the scalar this supersedes).
Closes issue **#549**.
