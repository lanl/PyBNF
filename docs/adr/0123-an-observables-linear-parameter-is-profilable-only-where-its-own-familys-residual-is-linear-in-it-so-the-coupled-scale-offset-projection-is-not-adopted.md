# An observable's linear parameter is profilable only where its own family's residual is linear in it, so the coupled `(scale, offset)` projection is not adopted (issue #572)

**Status: Evaluated 2026-08-21; the proposed method is NOT adopted, and #572's own kill
criterion is met.** ADR-0066/0099 profile a declared column's optimal multiplicative **scale**
out analytically, and ADR-0108 (#562) profiles out an estimated **noise scale**. #572 proposed
the third member of that family — an additive **offset**, and the coupled `(scale, offset)`
pair — by variable projection, and asked for an evaluation *before* anything was built,
with a stated kill criterion. The evaluation was run. It kills the proposal, for a reason
neither the issue's case-for nor its case-against anticipated: **variable projection is a
linear-least-squares identity, and every coupled pair and every additive offset in the
corpus is scored on a log residual scale, where that identity does not hold.** Where an
exact profile is taken anyway — numerically, so the missing closed form is not the
obstacle — it collapses essentially the entire search box onto the no-dynamics score.

No code changes. This ADR exists so the measurement is not re-litigated, and so the small,
genuinely sound remainder is scoped rather than lost.

## What was asked

> Fewer dimensions is not on its own a reason to ship a change to what the objective means.
> — #572

#572 asked four things: (1) compare the reduced and searched landscapes offline; (2) on
`Borghans`, does the `-165.98` no-dynamics attractor survive when scale, offset and σ are all
profiled; (3) head-to-head fits at matched budget; (4) confirm the `Fiedler` double-binding
refusal and that no other slug hides the same pattern.

## How it was measured

Two tools, both in the corpus (`Grein-2026-benchmark-subset-I/tools/`):

* **`linear_scope.py`** classifies every declared free parameter that reaches an observable:
  is the formula affine in it, is it a pure multiplicative factor, what residual space does
  its observable's noise family score in (`family.additive_on.ln_base`), how many series is
  it tied across, and is it *also* bound as a noise parameter. Static; no simulation.
* **`linear_profile.py`** simulates once per point and then minimizes **PyBNF's own
  `evaluate_multiple`** over the linear parameters with θ held fixed. The profile is taken
  *numerically*, on purpose: it is the exact conditional optimum whatever the family is, so
  the landscape question is answered independently of whether a closed form exists. (ADR-0108
  pinned its own σ closed form against a numeric minimization of the reported objective for
  the same reason: a plausible-but-wrong formula survives re-derived algebra and does not
  survive this.)

Neither tool runs a fit. A `Borghans` point costs one simulation plus a few hundred
re-scorings of it.

Three self-checks carry the result, and all three passed:

* The independently computed **flat-line reference** — every non-intercept coefficient pinned
  to exactly `0`, the intercept profiled — reproduces `-165.982113` at **every** point, spread
  `0`, which is the no-dynamics attractor already on record for this slug.
* The **reference optimum** re-scores at `-248.0524` against the recorded `-248.0692`, and the
  PEtab nominal point at `-198.207` against the recorded `-198.1017`; both differences are the
  σ-profiling and the bngsim build, not the harness.
* The profiled score is **never worse than the searched score** at any of the 250-odd points
  measured, which it cannot be, since the searched value is in the set being minimized over.

## Finding 1 — the scope in #572 is wrong in both directions

Across all 23 subset-I slugs there are **47** affine observable-layer parameters in **9**
slugs, not 22 in 6. Split by whether a closed-form profile exists at all:

| slug | k | affine obs params | linear-lsq | log-geomean | none | why not |
|---|---:|---:|---:|---:|---:|---|
| `Schwen_PONE2015` | 30 | 10 | 0 | 1 | 9 | log family, not multiplicative |
| `Smith_BMCSystBiol2013` | 25 | 9 | 9 | 0 | 0 | — |
| `Fiedler_BMCSystBiol2016` | 22 | 8 | 0 | 0 | 8 | double-bound as a noise parameter |
| `Raia_CancerResearch2011` | 39 | 5 | 0 | 0 | 5 | double-bound as a noise parameter |
| `Weber_BMC2015` | 36 | 5 | 5 | 0 | 0 | — |
| `Brannmark_JBC2010` | 22 | 4 | 4 | 0 | 0 | — |
| `Borghans_BiophysChem1997` | 23 | 2 | 0 | 0 | 2 | log family, not multiplicative |
| `Elowitz_Nature2000` | 21 | 2 | 0 | 0 | 2 | log family, not multiplicative |
| `Laske_PLOSComputBiol2019` | 13 | 2 | 1 | 0 | 1 | log family, not multiplicative |

Two slugs the issue does not list at all:

* **`Smith_BMCSystBiol2013`** — nine tied per-observable scales, 36 % of `k = 25`, every one of
  them closed-form profilable. It is the **largest** opportunity in the corpus and it is
  missing from the issue's table.
* **`Raia_CancerResearch2011`** — see finding 5.

And the issue undercounts `Brannmark` (4, not 2: `k_IRP_1Step` and `k_IRSiP_DosR` are scales
too) and miscounts `Schwen`'s share as 42 % of the search when 9 of its 10 are not profilable
by any closed form.

## Finding 2 — variable projection does not apply where #572 wants it

`Phi = [s, 1]`, `c* = (Phi^T W Phi)^-1 Phi^T W d` is an identity about a residual `d - Phi c`.
A PyBNF noise family declares the space its residual lives in. For `lognormal`
(`ln_base = ln 10`) and `lnnormal` (`ln_base = 1`) the residual is `log d - log(a*s + b)`,
which is affine in `log a` when `b == 0` — that is exactly why ADR-0066's geometric-mean ratio
is a closed form for a *pure* scale — and is affine in `b` for no `b` at all.

Sorted by role, the corpus splits cleanly and unhelpfully:

* **All three coupled `(scale, offset)` pairs** — `Borghans` (`Z_state*scale + offset`),
  `Elowitz` (`GFP*scale + background`), `Schwen` (`scale*(IR1 + IR1in + offset)`) — are
  `lognormal`. Zero have a closed form.
* **Of the 8 offset-role parameters in the corpus, exactly one has a closed form**:
  `Laske`'s `Int_nuc_off`, a Gaussian offset shared by nine observables — which #572 does not
  mention. `Laske`'s `vRNA_offset`, the one offset the issue *does* name, sits on an
  `lnnormal` observable and has none.
* The two slugs where `Phi = [s, 1]` is the right object, `Weber` and `Brannmark`, carry **no
  offset at all**. They are pure scales — "one more scalar", the case the issue explicitly
  sets aside as not the interesting one.

So the interesting half of the proposal has no instances, and the instances have no
interesting half. `Schwen`'s `scale` is the single log-family parameter with a closed form,
and it is the geometric-mean ratio ADR-0066 already implements — though note it is a
*different object*: ADR-0066 profiles an **undeclared, per-series** scale, while `Schwen`'s is
one **declared coefficient tied across 7 series**, so it is a stacked solve, not ADR-0066's.

## Finding 3 — the exact profile collapses the box onto the no-dynamics score

This is the kill. The closed form is not the obstacle: the profile below is the exact
conditional optimum, taken numerically, with σ additionally profiled by ADR-0108 — #572's
item 2 verbatim, "scale, offset and σ all profiled".

`Borghans_BiophysChem1997`, 76 box draws that integrate, `noise_profiling = 1`:

| | searched | profiled |
|---|---:|---:|
| range over the box | `[-160.88, +339.69]` | **`[-167.15, -165.98]`** |
| interquartile range | **129.5** | **0.081** |
| median − flat line | +374.5 | −2.8e−14 |
| draws scoring better than the flat line | 0 / 76 | 40 / 76 |
| draws scoring **exactly** the flat line | 0 / 76 | **46 / 76** |
| draws within 1.0 of the flat line | 0 / 76 | **73 / 76** |
| draws whose profiled `scale < 1e-6` | — | **39 / 76** |
| Spearman rank correlation, searched ~ profiled | — | **0.035** |

The reference optimum is untouched (`-248.0524 -> -248.0525`, gain `1.1e-4`) and so is the
nominal point (`-198.207 -> -198.262`, gain `0.055`) — the envelope theorem doing exactly what
#572 says it does. What moves is everything else: **the entire box collapses into a 1.16-unit
band sitting on the no-dynamics score**, while the reference optimum stays 82 units below it.

The mechanism is worse than "the flat line becomes available". At 39 of 76 draws the profile
drives the scale below `1e-6` — it *discards the dynamics entirely*, because on a log residual
a wrong-shaped trajectory makes the fit worse than no trajectory. The no-dynamics answer is
not a floor the reduced surface can reach; over half the box it **is** the reduced surface.

`Elowitz_Nature2000` reproduces it on the same mechanism, less extremely: 57 draws, IQR
`70.21 -> 7.77`, 27/57 exactly at the flat line `-53.656041`, 19/57 discarding the dynamics,
rank correlation **−0.43**.

### The section is the picture

Thirteen points interpolated in sampling space from the reference optimum (`t00`) out to a box
draw (`t12`), nine of which integrate:

| | t00 | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| searched | −248.05 | −152.29 | −66.49 | +112.97 | +150.42 | +166.09 | +178.64 | +189.77 | +199.87 |
| profiled | −248.05 | −166.58 | −167.01 | **−165.9821** | **−165.9821** | **−165.9821** | **−165.9821** | **−165.9821** | **−165.9821** |

The searched objective rises monotonically away from the optimum — a continuous descent
direction all the way in from the box. The profiled objective is **bit-identical at six
consecutive section points**. Profiling does not sharpen this landscape; it deletes it.

### Answering (1) and (2) directly

1. **Worse, decisively.** The reduced surface does not separate the reference optimum from the
   flat-line floor better — it removes the separation between the flat-line floor and
   *everything else*, which is the only structure a global sampler had.
2. **The `-165.98` attractor does not merely remain a converged endpoint. It becomes the value
   of the search space.** Half the box scores it to the last bit, and 96 % of the box scores
   within 1.0 of it.

The conclusion does not depend on the parametrization. Repeating `Borghans` with the profile
taken on the real line rather than in each parameter's sampling space (`--linear-space`,
i.e. what an unconstrained variable projection would do) gives the same collapse, and
additionally trips ADR-0108's degenerate-scale refusal, because an unconstrained offset drives
predictions out of the log family's domain. An unconstrained projection is not merely
unavailable on these slugs; it is infeasible.

## Finding 4 — where it does apply, `noise_profiling` has already collected most of it

`Weber_BMC2015`, five pure Gaussian scales, 20 box draws, measured twice:

| | searched IQR | profiled IQR | rank corr |
|---|---:|---:|---:|
| σ **searched** (as the conf ships) | 1.96e18 | 3.38e5 | 0.43 |
| σ **profiled** (`noise_profiling = 1`) | 1.67e3 | **1.72e3** | **0.87** |

With ADR-0108 switched on, profiling all five observable scales changes the spread of the
landscape by **nothing** — it is marginally *larger* — and leaves the ranking almost intact.
The thirteen-orders-of-magnitude improvement in the first row is real, and it is the σ
profile's, not the scale profile's: a wrong scale inflates residuals, and a profiled σ absorbs
that inflation. #572's case-for is "better conditioning, every draw linear-optimal". On the
one corpus slug where both switches are available, the shipped one already delivers it.

`Brannmark_JBC2010` is the counter-case and it is instructive: `noise_profiling = 1` is
**refused** there (`IRS1_P` estimates σ from a `PerMeasurementFormulaSigma`), so the scale
profile has the field to itself and does compress the landscape (IQR `1.01e17 -> 3.94e12`,
rank corr 0.62). Which also disposes of #572's proposed ordering rule: "linear-first, then σ
in closed form" describes a situation that does not arise on the slug it would matter for.

`Laske`'s `Int_nuc_off` — the corpus's one genuine closed-form additive offset — behaves
exactly as a good nuisance should: it profiles to ≈ `57.13` at essentially every draw
regardless of θ, compresses the IQR from `1.01e9` to `5.01e7`, and moves the nominal point by
`0.02`. It is a real, small, well-behaved win. It is one parameter of 13 in one slug.

## Finding 5 — the double-binding refusal is needed, and `Fiedler` is not the only slug

#572 asks for `Fiedler_BMCSystBiol2016` to be refused by name and asks whether anything else
hides the pattern. Both halves check out, and the answer to the second is **yes**:

* **`Fiedler`** — confirmed exactly as described. All eight `s_pErk_*` / `s_pMek_*` tokens are
  bound to `observableParameter1_*` *and* `noiseParameter1_*` in the same per-measurement
  tables.
* **`Raia_CancerResearch2011`** — the same defect by a different route, and **not mentioned in
  the issue**. Five `scaling_*` parameters are the observable scale *and* appear inside a
  `prediction_formula` σ (`CD274mRNA*scaling_CD274mRNA*sd_CD274mRNA_rel + sd_CD274mRNA_abs`),
  so moving the scale moves that observable's σ too. Thirteen of the corpus's 47 affine
  observable parameters are double-bound this way — more than a quarter.

A "detect the `noiseParameter` placeholder" rule catches `Fiedler` and misses `Raia`. Any
implementation must test the resolved parameter *names* a noise source reads, formula sources
included.

## The decision

**The coupled `(scale, offset)` variable projection of #572 is not implemented**, and #572 is
closed against its own kill criterion. Restating that criterion:

> If (1) and (2) show the reduced landscape is no better — or worse — for global search, and
> (3) shows no reliable improvement, then close this as a tidiness item and do **not**
> implement.

(1) and (2) are worse by the largest margin the measurement could have produced: the reduced
objective is constant to 0.08 units over the interquartile range of the box, and bit-identical
across six consecutive points of a section. Fewer dimensions was never the argument, and it is
all that is left.

### What was deliberately not measured, and why

**Item 3, head-to-head fits at matched budget, was not run.** For the offset half that is a
consequence of the result rather than a gap in it: a fit comparison on a landscape that is
constant over half the box measures the sampler's tie-breaking, not the objective. Reporting
"profiled did not converge better" from such a run would attribute to the search what the
measurement already attributes to the surface.

For the pure-scale remainder, item 3 *is* the right test — and it cannot be run without first
building the feature, which is the thing #572 exists to gate. It is therefore named below as
the acceptance test for the follow-up rather than performed here.

### What survives, and what it would have to be

A **pure multiplicative scale, on a linear-scale family, tied across the series that share
it** is sound: the closed form is real, the corpus has 19 such parameters in 4 slugs, and
`Laske`'s `Int_nuc_off` shows the additive case behaving well where the family permits it.
That is a materially different, smaller feature than #572 proposes, and it inherits three
constraints this evaluation established:

* It must **refuse a parameter that any noise source also reads**, by resolved name, formula
  sources included — 13 of 47 corpus parameters, across `Fiedler` **and** `Raia`.
* It must refuse any parameter whose observable scores on a **log** family unless the formula
  is homogeneous in it, and it must apply the geometric-mean form rather than least squares
  when it is.
* Its acceptance test is **item 3 on `Smith` and `Brannmark`** — the two slugs where it is not
  redundant with `noise_profiling` — not on `Weber`, where finding 4 shows there is nothing
  left for it to collect.

Whether that is worth building is a separate question from this one, and this ADR does not
answer it. It is not opened here.

## Consequences

* No behaviour change. No new config key. `noise_profiling` (ADR-0108) and the ADR-0066
  `normalization = ... , scale` chain are unaffected.
* `docs/adr/0108` is unchanged; its "pyPESTO/AMICI profile scale, offset and σ analytically by
  default" prior-art line stands, with the qualification recorded here that those defaults are
  a linear-Gaussian construction and PyBNF's log families are outside it.
* Two reusable tools land in the corpus, with their gotchas written down:
  `Grein-2026-benchmark-subset-I/tools/linear_scope.py` and `.../linear_profile.py`.

## Prior art

Variable projection (Golub & Pereyra 1973); hierarchical optimization for ODE models (Loos et
al. 2018); pyPESTO/AMICI. All three are stated over an additive-Gaussian residual. That
assumption is load-bearing and is the one this corpus violates.
